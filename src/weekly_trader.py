"""
weekly_trader.py — Dedicated loop for weekly options-selling strategies.

Runs alongside paper_trader.py (or live_trader.py). Manages:
  1. DefaultShortStrangle  — symmetric ATM, no forecast.
  2. KronosSkewedStrangle  — asymmetric, Kronos p10/p90-driven.

Exact lifecycle per week:
  Monday 09:30–10:00  →  enter() both strategies (whichever is active in config).
  Mon–Wed every 5 min  →  monitor() both; exit if SL/target hit.
  Wednesday 15:15     →  force-exit (day before NIFTY Thursday expiry).
  Thursday+           →  no open position; ready for next Monday.

Config (config.yaml):
  strategies:
    weekly:
      active: ["default", "kronos"]   # remove a name to disable
      symbol: NIFTY
      entry_window:
        start: "09:30"
        end:   "10:00"

Run standalone:
    python -m src.weekly_trader
    python -m src.weekly_trader --dry-run          # paper mode (default)
    python -m src.weekly_trader --no-kronos        # skip Kronos strategy
    python -m src.weekly_trader --no-default       # skip Default strategy

Or import WeeklyTrader and call tick() from inside PaperTrader._tick().
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, date, time as dtime
from typing import Optional

import pandas as pd

from src.broker.base import BrokerInterface
from src.data_fetcher import DataFetcher
from src.data_cleaner import clean
from src.forecaster import KronosForecaster
from src.risk_manager import RiskManager
from src.strategies.default_short_strangle import (
    DefaultShortStrangle, is_entry_day, next_weekly_expiry,
)
from src.strategies.kronos_skewed_strangle import KronosSkewedStrangle
from src.utils import get_broker, load_config, now_ist, is_market_open, IST
from src.db import init_db, get_conn

logger = logging.getLogger(__name__)

# ── Brokerage charge constants (Zerodha) ────────────────────────────
_CHARGE_PER_ORDER_RS = 20.0   # Zerodha flat brokerage per order
_LEGS_PER_STRANGLE   = 2      # CE + PE


class WeeklyTrader:
    """
    Orchestrates one DefaultShortStrangle and one KronosSkewedStrangle
    on the same symbol, sharing a single RiskManager (₹3L cap).
    """

    def __init__(
        self,
        broker: BrokerInterface,
        symbol: str = "NIFTY",
        config_path: str = "config.yaml",
        db_path: str = "data/kronos_options.db",
        enable_default: bool = True,
        enable_kronos: bool = True,
        dry_run: bool = True,
    ):
        self.broker     = broker
        self.symbol     = symbol
        self.cfg        = load_config(config_path)
        self.db_path    = db_path
        self.dry_run    = dry_run

        init_db(db_path)

        live_cfg = self.cfg.get("live_trading", {})
        # Single shared RiskManager so both strategies count against the same ₹3L cap
        self.risk = RiskManager(
            broker=broker,
            max_total_capital    = live_cfg.get("max_capital_total",    300_000),
            max_capital_per_trade= live_cfg.get("max_capital_per_trade",  50_000),
            max_daily_loss       = live_cfg.get("max_daily_loss_rs",       5_000),
        )

        self.default_strat: Optional[DefaultShortStrangle] = (
            DefaultShortStrangle(broker=broker, config_path=config_path,
                                 risk_manager=self.risk)
            if enable_default else None
        )

        self.kronos_strat: Optional[KronosSkewedStrangle] = None
        if enable_kronos:
            try:
                forecaster = KronosForecaster(config_path=config_path, db_path=db_path)
                self.kronos_strat = KronosSkewedStrangle(
                    broker=broker, forecaster=forecaster,
                    config_path=config_path, risk_manager=self.risk,
                )
            except Exception as e:
                logger.warning("KronosForecaster init failed (%s) — Kronos strategy disabled.", e)

        self.fetcher = DataFetcher(broker, config_path)

        # Entry window from config
        wk_cfg = self.cfg.get("strategies", {}).get("weekly", {})
        entry_win = wk_cfg.get("entry_window", {})
        self._entry_start = dtime(*[int(x) for x in entry_win.get("start", "09:30").split(":")])
        self._entry_end   = dtime(*[int(x) for x in entry_win.get("end",   "10:00").split(":")])

        sq = self.cfg.get("trading", {}).get("square_off_time", "15:15").split(":")
        self._sq_off = dtime(int(sq[0]), int(sq[1]))

        self._entered_today = False   # prevent double-entry on same day
        self._last_entry_date: Optional[date] = None

    # ── Public API: call from PaperTrader._tick() or standalone loop ─────

    def tick(self, now: Optional[datetime] = None) -> None:
        """
        One 5-min tick. Call this every 5 minutes during market hours.
        Handles entry (Monday window) and monitor (all days).
        """
        now = now or now_ist()
        t   = now.time()
        d   = now.date()

        # Reset daily flag on new day
        if self._last_entry_date != d:
            self._entered_today = False
            self._last_entry_date = d
            self.risk.reset_daily()
            logger.info("WeeklyTrader: new day %s — daily counters reset.", d)

        # Square-off all positions at end of day
        if t >= self._sq_off:
            self._force_exit_all("INTRADAY_SQUAREOFF", now)
            return

        # Monitor open positions (every tick)
        self._monitor_all(now)

        # Entry: only on Monday within entry window, and not already entered today
        if is_entry_day(self.symbol, d) and self._entry_start <= t <= self._entry_end:
            if not self._entered_today:
                self._enter_all(now)
                self._entered_today = True

    # ── Entry ───────────────────────────────────────────────────

    def _enter_all(self, now: datetime) -> None:
        """Attempt entry for all active strategies. Fetches shared data once."""
        logger.info("WeeklyTrader: Monday entry window — attempting entries for %s", self.symbol)

        # Fetch OHLCV once; share between both strategies
        lookback = self.cfg.get("kronos", {}).get("lookback", 400)
        df_ohlcv = None
        try:
            df_raw   = self.fetcher.fetch_latest_bars(self.symbol, n_bars=lookback + 50)
            df_ohlcv = clean(df_raw, symbol=self.symbol)
        except Exception as e:
            logger.warning("OHLCV fetch failed: %s — Kronos entry may fail.", e)

        # DefaultShortStrangle entry
        if self.default_strat is not None:
            try:
                result = self.default_strat.enter(symbol=self.symbol, dry_run=self.dry_run)
                self._log_entry("DefaultShortStrangle", result, now)
            except Exception as e:
                logger.error("DefaultShortStrangle entry error: %s", e, exc_info=True)

        # KronosSkewedStrangle entry
        if self.kronos_strat is not None:
            if df_ohlcv is not None and not df_ohlcv.empty:
                try:
                    result = self.kronos_strat.enter(
                        symbol=self.symbol, df_ohlcv=df_ohlcv, dry_run=self.dry_run
                    )
                    self._log_entry("KronosSkewedStrangle", result, now)
                except Exception as e:
                    logger.error("KronosSkewedStrangle entry error: %s", e, exc_info=True)
            else:
                logger.warning("KronosSkewedStrangle skipped: no OHLCV data available.")

    # ── Monitor ──────────────────────────────────────────────────

    def _monitor_all(self, now: datetime) -> None:
        """Check SL/target for all open strangle positions."""
        for strat, name in [
            (self.default_strat, "DefaultShortStrangle"),
            (self.kronos_strat,  "KronosSkewedStrangle"),
        ]:
            if strat is None or strat.position is None:
                continue
            try:
                result = strat.monitor(dry_run=self.dry_run)
                if result:
                    self._log_exit(name, result, now)
            except Exception as e:
                logger.error("%s monitor error: %s", name, e, exc_info=True)

    # ── Force exit ──────────────────────────────────────────────

    def _force_exit_all(self, reason: str, now: datetime) -> None:
        """Close any still-open strangle positions immediately."""
        for strat, name in [
            (self.default_strat, "DefaultShortStrangle"),
            (self.kronos_strat,  "KronosSkewedStrangle"),
        ]:
            if strat is None or strat.position is None:
                continue
            try:
                result = strat._exit(reason=reason, dry_run=self.dry_run)
                self._log_exit(name, result, now)
            except Exception as e:
                logger.error("%s force exit error: %s", name, e, exc_info=True)

    # ── DB logging ───────────────────────────────────────────────

    def _log_entry(self, strategy_name: str, result: dict, now: datetime) -> None:
        status = result.get("status", "")
        if status == "ENTERED":
            logger.info(
                "✅ %s ENTERED: %s | CE=%d@%.2f PE=%d@%.2f | "
                "combined=%.2f SL=%.2f Target=%.2f | MaxProfit=₹%.0f",
                strategy_name,
                result.get("symbol"), result.get("ce_strike"), result.get("ce_entry"),
                result.get("pe_strike"), result.get("pe_entry"),
                result.get("combined_premium", 0),
                result.get("sl_premium", 0), result.get("target_premium", 0),
                result.get("max_profit_rs", 0),
            )
            # Persist to weekly_trades table
            try:
                with get_conn(self.db_path) as conn:
                    conn.execute(
                        """INSERT OR IGNORE INTO weekly_trades
                           (strategy, symbol, expiry, ce_strike, pe_strike,
                            ce_entry, pe_entry, combined_premium,
                            sl_premium, target_premium, max_profit_rs,
                            entry_ts, status, dry_run, extra_json)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            strategy_name,
                            result.get("symbol"), result.get("expiry"),
                            result.get("ce_strike"), result.get("pe_strike"),
                            result.get("ce_entry"), result.get("pe_entry"),
                            result.get("combined_premium"),
                            result.get("sl_premium"), result.get("target_premium"),
                            result.get("max_profit_rs"),
                            result.get("entry_ts"), "OPEN",
                            1 if self.dry_run else 0,
                            json.dumps(result.get("skew_metrics") or result.get("forecast") or {}),
                        ),
                    )
            except Exception as e:
                logger.warning("DB entry log failed: %s", e)

        elif status in ("BLOCKED", "SKIPPED", "ERROR"):
            logger.warning("%s %s: %s", strategy_name, status, result.get("reason", ""))

    def _log_exit(self, strategy_name: str, result: dict, now: datetime) -> None:
        pnl = result.get("realised_pnl_rs", 0)
        charges = _CHARGE_PER_ORDER_RS * _LEGS_PER_STRANGLE * 2  # entry + exit × 2 legs
        net_pnl = pnl - charges
        emoji = "✅" if net_pnl >= 0 else "❌"
        logger.info(
            "%s %s EXITED (%s): CE@%.2f→%.2f PE@%.2f→%.2f | "
            "P&L=₹%.0f charges=₹%.0f NET=₹%.0f",
            emoji, strategy_name, result.get("reason"),
            result.get("ce_entry", 0), result.get("ce_exit", 0),
            result.get("pe_entry", 0), result.get("pe_exit", 0),
            pnl, charges, net_pnl,
        )
        try:
            with get_conn(self.db_path) as conn:
                conn.execute(
                    """UPDATE weekly_trades
                       SET exit_ts=?, status='CLOSED', exit_reason=?,
                           ce_exit=?, pe_exit=?,
                           realised_pnl_rs=?, charges_rs=?, net_pnl_rs=?
                       WHERE strategy=? AND symbol=? AND status='OPEN'
                       ORDER BY rowid DESC LIMIT 1""",
                    (
                        result.get("exit_ts"), result.get("reason"),
                        result.get("ce_exit"), result.get("pe_exit"),
                        pnl, charges, net_pnl,
                        strategy_name, result.get("symbol"),
                    ),
                )
        except Exception as e:
            logger.warning("DB exit log failed: %s", e)

    # ── Status ─────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "symbol":    self.symbol,
            "dry_run":   self.dry_run,
            "risk":      self.risk.status(),
            "default":   self.default_strat.status() if self.default_strat else None,
            "kronos":    self.kronos_strat.status()  if self.kronos_strat  else None,
        }


# ── DB schema migration: add weekly_trades table ─────────────────────────

def ensure_weekly_trades_table(db_path: str = "data/kronos_options.db") -> None:
    """Idempotent: create weekly_trades table if it doesn't exist."""
    with get_conn(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weekly_trades (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy         TEXT NOT NULL,
                symbol           TEXT NOT NULL,
                expiry           TEXT,
                ce_strike        REAL,
                pe_strike        REAL,
                ce_entry         REAL,
                pe_entry         REAL,
                combined_premium REAL,
                sl_premium       REAL,
                target_premium   REAL,
                max_profit_rs    REAL,
                entry_ts         TEXT,
                exit_ts          TEXT,
                status           TEXT DEFAULT 'OPEN',
                exit_reason      TEXT,
                ce_exit          REAL,
                pe_exit          REAL,
                realised_pnl_rs  REAL,
                charges_rs       REAL,
                net_pnl_rs       REAL,
                dry_run          INTEGER DEFAULT 1,
                extra_json       TEXT
            )
        """)


# ── Standalone run ─────────────────────────────────────────────────

def main():
    from src.utils import setup_logging
    setup_logging("weekly_trader")

    parser = argparse.ArgumentParser(description="Kronos Options Weekly Trader")
    parser.add_argument("--symbol",     default="NIFTY")
    parser.add_argument("--live",       action="store_true", help="Live orders (default: paper)")
    parser.add_argument("--no-default", action="store_true", help="Disable DefaultShortStrangle")
    parser.add_argument("--no-kronos",  action="store_true", help="Disable KronosSkewedStrangle")
    args = parser.parse_args()

    dry_run = not args.live
    if not dry_run:
        confirm = input('Type "I CONFIRM LIVE TRADING" to proceed: ').strip()
        if confirm != "I CONFIRM LIVE TRADING":
            print("Aborted.")
            return

    broker = get_broker()
    ensure_weekly_trades_table()

    trader = WeeklyTrader(
        broker         = broker,
        symbol         = args.symbol,
        enable_default = not args.no_default,
        enable_kronos  = not args.no_kronos,
        dry_run        = dry_run,
    )

    logger.info("WeeklyTrader started | symbol=%s dry_run=%s", args.symbol, dry_run)

    while True:
        now = now_ist()
        if not is_market_open(now):
            time.sleep(60)
            continue
        try:
            trader.tick(now)
        except Exception as e:
            logger.error("WeeklyTrader tick error: %s", e, exc_info=True)

        seconds_past = now.minute % 5 * 60 + now.second
        time.sleep(max(5, 300 - seconds_past))


if __name__ == "__main__":
    main()
