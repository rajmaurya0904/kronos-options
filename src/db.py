"""
SQLite schema and helpers.
All tables created on first run via init_db().
"""
from __future__ import annotations
import sqlite3
import json
import hashlib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_DB = "data/kronos_options.db"


@contextmanager
def get_conn(db_path: str = DEFAULT_DB):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB) -> None:
    """Create all tables if they don't exist."""
    with get_conn(db_path) as conn:
        conn.executescript("""
        -- ── Forecast cache ──────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS forecasts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT NOT NULL,
            created_at    TEXT NOT NULL,           -- ISO8601 IST
            input_hash    TEXT NOT NULL,           -- SHA256 of (symbol, last_ts, lookback, pred_len, samples)
            last_bar_ts   TEXT NOT NULL,           -- timestamp of last input bar
            pred_len      INTEGER NOT NULL,
            samples       INTEGER NOT NULL,
            median_path   TEXT NOT NULL,           -- JSON: list of {ts, open, high, low, close}
            sample_paths  TEXT NOT NULL,           -- JSON: list of lists
            prob_up       REAL NOT NULL,
            expected_move_pct REAL NOT NULL,
            path_dispersion   REAL NOT NULL,
            expected_close    REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_forecast_symbol_hash ON forecasts(symbol, input_hash);

        -- ── Signals ─────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT NOT NULL,
            signal_time     TEXT NOT NULL,         -- IST timestamp
            signal          TEXT NOT NULL,         -- STRONG_BULLISH | BULLISH | etc.
            confidence      REAL NOT NULL,
            expected_move_pct REAL NOT NULL,
            prob_up         REAL NOT NULL,
            expected_range_low  REAL,
            expected_range_high REAL,
            current_close   REAL NOT NULL,
            forecast_id     INTEGER REFERENCES forecasts(id)
        );

        -- ── Recommended option trades ────────────────────────────────────
        CREATE TABLE IF NOT EXISTS trade_recommendations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id       INTEGER REFERENCES signals(id),
            symbol          TEXT NOT NULL,
            strategy        TEXT NOT NULL,         -- buy_atm_ce | bull_put_spread | etc.
            created_at      TEXT NOT NULL,
            expiry          TEXT NOT NULL,
            legs            TEXT NOT NULL,         -- JSON: list of {strike, option_type, action, lots}
            lot_size        INTEGER NOT NULL,
            max_profit_rs   REAL,
            max_loss_rs     REAL,
            breakeven_upper REAL,
            breakeven_lower REAL
        );

        -- ── Paper trades ─────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS paper_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            recommendation_id INTEGER REFERENCES trade_recommendations(id),
            symbol          TEXT NOT NULL,
            strategy        TEXT NOT NULL,
            entry_time      TEXT NOT NULL,
            exit_time       TEXT,
            status          TEXT NOT NULL DEFAULT 'OPEN',  -- OPEN | CLOSED | ABANDONED (process restarted)
            legs            TEXT NOT NULL,         -- JSON: entry prices per leg
            entry_total_premium REAL NOT NULL,
            exit_total_premium  REAL,
            raw_pnl_rs      REAL,
            charges_rs      REAL,
            net_pnl_rs      REAL,
            exit_reason     TEXT                   -- SIGNAL_EXIT | SQUARE_OFF | SL_HIT | TGT_HIT
        );

        -- ── Backtest results ─────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at          TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            start_date      TEXT NOT NULL,
            end_date        TEXT NOT NULL,
            total_trades    INTEGER,
            wins            INTEGER,
            losses          INTEGER,
            total_pnl_rs    REAL,
            sharpe          REAL,
            max_drawdown_rs REAL,
            win_rate_pct    REAL,
            profit_factor   REAL,
            params          TEXT                   -- JSON: config snapshot
        );

        CREATE TABLE IF NOT EXISTS backtest_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER REFERENCES backtest_runs(id),
            trade_date      TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            signal          TEXT NOT NULL,
            strategy        TEXT NOT NULL,
            strike          INTEGER,
            option_type     TEXT,
            expiry          TEXT,
            entry_price     REAL,
            exit_price      REAL,
            lots            INTEGER,
            lot_size        INTEGER,
            raw_pnl_rs      REAL,
            charges_rs      REAL,
            net_pnl_rs      REAL,
            sl_tgt_tag      TEXT,
            data_source     TEXT,                  -- "real_option_data" | "bs_approximation"
            legs            TEXT,                  -- e.g. "S23750PE B23700PE"
            premium_type    TEXT,                  -- CREDIT | DEBIT
            entry_time      TEXT,                  -- HH:MM IST
            exit_time       TEXT
        );
        """)
        # Databases created before these columns existed: add them in place.
        have = {r["name"] for r in conn.execute("PRAGMA table_info(backtest_trades)")}
        for col in ("legs", "premium_type", "entry_time", "exit_time"):
            if col not in have:
                conn.execute(f"ALTER TABLE backtest_trades ADD COLUMN {col} TEXT")


# ── Forecast helpers ──────────────────────────────────────────────────

def make_forecast_hash(symbol: str, last_ts: str, lookback: int, pred_len: int, samples: int) -> str:
    key = f"{symbol}|{last_ts}|{lookback}|{pred_len}|{samples}"
    return hashlib.sha256(key.encode()).hexdigest()


def get_cached_forecast(
    input_hash: str,
    ttl_minutes: int = 5,
    db_path: str = DEFAULT_DB,
) -> Optional[dict]:
    """Return cached forecast if it exists and is within TTL."""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM forecasts WHERE input_hash = ? ORDER BY id DESC LIMIT 1",
            (input_hash,),
        ).fetchone()
        if not row:
            return None
        created = datetime.fromisoformat(row["created_at"])
        age_minutes = (datetime.now() - created).total_seconds() / 60
        if age_minutes > ttl_minutes:
            return None
        return dict(row)


def save_forecast(data: dict, db_path: str = DEFAULT_DB) -> int:
    with get_conn(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO forecasts
               (symbol, created_at, input_hash, last_bar_ts, pred_len, samples,
                median_path, sample_paths, prob_up, expected_move_pct, path_dispersion, expected_close)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                data["symbol"], data["created_at"], data["input_hash"],
                data["last_bar_ts"], data["pred_len"], data["samples"],
                json.dumps(data["median_path"]), json.dumps(data["sample_paths"]),
                data["prob_up"], data["expected_move_pct"],
                data["path_dispersion"], data["expected_close"],
            ),
        )
        return cur.lastrowid


def save_signal(data: dict, db_path: str = DEFAULT_DB) -> int:
    with get_conn(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO signals
               (symbol, signal_time, signal, confidence, expected_move_pct,
                prob_up, expected_range_low, expected_range_high, current_close, forecast_id)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                data["symbol"], data["signal_time"], data["signal"],
                data["confidence"], data["expected_move_pct"], data["prob_up"],
                data.get("expected_range_low"), data.get("expected_range_high"),
                data["current_close"], data.get("forecast_id"),
            ),
        )
        return cur.lastrowid


def get_signals_today(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    # signal_time is stored in IST, so "today" must be the IST date too.
    from src.utils import today_ist
    today = today_ist().strftime("%Y-%m-%d")
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM signals WHERE signal_time LIKE ? ORDER BY signal_time DESC",
            (f"{today}%",),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def get_paper_trades(status: str = "OPEN", db_path: str = DEFAULT_DB) -> pd.DataFrame:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE status = ? ORDER BY entry_time DESC",
            (status,),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])
