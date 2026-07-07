"""
chart_exporter.py — Export DB trade data to JSON for the HTML profit chart.

Usage:
    python -m src.chart_exporter                        # writes dashboard/chart_data.json
    python -m src.chart_exporter --db data/kronos_options.db
    python -m src.chart_exporter --source paper         # paper_trades only (default)
    python -m src.chart_exporter --source backtest      # backtest_trades only
    python -m src.chart_exporter --source both          # merged
    python -m src.chart_exporter --open                 # also launch browser
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import webbrowser
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any

DEFAULT_DB   = "data/kronos_options.db"
OUTPUT_DIR   = Path("dashboard")
OUTPUT_JSON  = OUTPUT_DIR / "chart_data.json"
OUTPUT_HTML  = OUTPUT_DIR / "profit-chart.html"
LOOKBACK_YEARS = 2


# ── helpers ──────────────────────────────────────────────────────────────

def _since_date() -> str:
    d = date.today() - timedelta(days=LOOKBACK_YEARS * 365)
    return d.isoformat()


def _fetch_paper_trades(conn: sqlite3.Connection, since: str) -> List[Dict]:
    rows = conn.execute(
        """
        SELECT
            id, symbol, strategy, exit_time, entry_time,
            net_pnl_rs, raw_pnl_rs, charges_rs,
            entry_total_premium, exit_total_premium,
            status, exit_reason
        FROM paper_trades
        WHERE status IN ('CLOSED', 'SQUARED_OFF')
          AND exit_time >= ?
        ORDER BY exit_time ASC
        """,
        (since,),
    ).fetchall()
    return [
        {
            "id":            r["id"],
            "source":        "paper",
            "symbol":        r["symbol"],
            "strategy":      r["strategy"],
            "date":          (r["exit_time"] or r["entry_time"] or "")[:10],
            "net_pnl":       round(r["net_pnl_rs"] or 0, 2),
            "raw_pnl":       round(r["raw_pnl_rs"] or 0, 2),
            "charges":       round(r["charges_rs"] or 0, 2),
            "entry_premium": round(r["entry_total_premium"] or 0, 2),
            "exit_premium":  round(r["exit_total_premium"] or 0, 2),
            "exit_reason":   r["exit_reason"] or "",
        }
        for r in rows
        if (r["net_pnl_rs"] is not None)
    ]


def _fetch_backtest_trades(conn: sqlite3.Connection, since: str) -> List[Dict]:
    rows = conn.execute(
        """
        SELECT
            bt.id, bt.trade_date, bt.symbol, bt.strategy,
            bt.signal, bt.net_pnl_rs, bt.raw_pnl_rs, bt.charges_rs,
            bt.entry_price, bt.exit_price, bt.lots, bt.lot_size,
            bt.sl_tgt_tag, bt.data_source
        FROM backtest_trades bt
        WHERE bt.trade_date >= ?
        ORDER BY bt.trade_date ASC
        """,
        (since,),
    ).fetchall()
    return [
        {
            "id":            r["id"],
            "source":        "backtest",
            "symbol":        r["symbol"],
            "strategy":      r["strategy"] or "",
            "date":          (r["trade_date"] or "")[:10],
            "net_pnl":       round(r["net_pnl_rs"] or 0, 2),
            "raw_pnl":       round(r["raw_pnl_rs"] or 0, 2),
            "charges":       round(r["charges_rs"] or 0, 2),
            "entry_premium": round(r["entry_price"] or 0, 2),
            "exit_premium":  round(r["exit_price"] or 0, 2),
            "exit_reason":   r["sl_tgt_tag"] or "",
        }
        for r in rows
        if (r["net_pnl_rs"] is not None)
    ]


def _build_series(trades: List[Dict]) -> Dict[str, Any]:
    """Build cumulative P&L series, daily P&L, and stats from raw trade list."""
    if not trades:
        return {
            "dates":        [],
            "cumulative":   [],
            "daily_pnl":    [],
            "drawdown":     [],
            "by_strategy":  {},
            "by_symbol":    {},
            "weekly_pnl":   [],
            "monthly_pnl":  [],
            "stats":        _empty_stats(),
        }

    # Group by date
    from collections import defaultdict
    daily: Dict[str, float] = defaultdict(float)
    daily_raw: Dict[str, float] = defaultdict(float)
    strategy_pnl: Dict[str, float] = defaultdict(float)
    symbol_pnl:   Dict[str, float] = defaultdict(float)
    wins = losses = 0

    for t in trades:
        d = t["date"]
        daily[d]        += t["net_pnl"]
        daily_raw[d]    += t["raw_pnl"]
        strategy_pnl[t["strategy"]] += t["net_pnl"]
        symbol_pnl[t["symbol"]]     += t["net_pnl"]
        if t["net_pnl"] > 0:
            wins += 1
        elif t["net_pnl"] < 0:
            losses += 1

    dates_sorted = sorted(daily.keys())
    cumulative   = []
    drawdown     = []
    running      = 0.0
    peak         = 0.0
    max_dd       = 0.0

    for d in dates_sorted:
        running += daily[d]
        cumulative.append(round(running, 2))
        if running > peak:
            peak = running
        dd = running - peak
        drawdown.append(round(dd, 2))
        if dd < max_dd:
            max_dd = dd

    total_pnl = running
    total_trades = wins + losses
    win_rate     = round(wins / total_trades * 100, 1) if total_trades else 0.0

    daily_vals   = [round(daily[d], 2) for d in dates_sorted]

    # Weekly P&L
    from collections import OrderedDict
    weekly: Dict[str, float] = defaultdict(float)
    monthly: Dict[str, float] = defaultdict(float)
    for d in dates_sorted:
        dt = datetime.strptime(d, "%Y-%m-%d")
        wk = dt.strftime("%Y-W%V")
        mo = dt.strftime("%Y-%m")
        weekly[wk]   += daily[d]
        monthly[mo]  += daily[d]

    # Consecutive win/loss
    streak_win = streak_loss = cur_streak = 0
    last_sign = None
    pnl_list = [daily[d] for d in dates_sorted]
    for pnl in pnl_list:
        s = 1 if pnl >= 0 else -1
        if s == last_sign:
            cur_streak += 1
        else:
            cur_streak = 1
            last_sign = s
        if s > 0 and cur_streak > streak_win:
            streak_win = cur_streak
        if s < 0 and cur_streak > streak_loss:
            streak_loss = cur_streak

    avg_win  = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0) / wins if wins else 0
    avg_loss = sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0) / losses if losses else 0
    profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses and avg_loss else 0

    return {
        "dates":       dates_sorted,
        "cumulative":  cumulative,
        "daily_pnl":   daily_vals,
        "drawdown":    drawdown,
        "weekly_pnl":  [{"week": k, "pnl": round(v, 2)} for k, v in sorted(weekly.items())],
        "monthly_pnl": [{"month": k, "pnl": round(v, 2)} for k, v in sorted(monthly.items())],
        "by_strategy": {k: round(v, 2) for k, v in strategy_pnl.items()},
        "by_symbol":   {k: round(v, 2) for k, v in symbol_pnl.items()},
        "stats": {
            "total_pnl":        round(total_pnl, 2),
            "total_trades":     total_trades,
            "wins":             wins,
            "losses":           losses,
            "win_rate_pct":     win_rate,
            "avg_win_rs":       round(avg_win, 2),
            "avg_loss_rs":      round(avg_loss, 2),
            "profit_factor":    round(profit_factor, 2),
            "max_drawdown_rs":  round(max_dd, 2),
            "best_day_rs":      round(max(daily_vals), 2) if daily_vals else 0,
            "worst_day_rs":     round(min(daily_vals), 2) if daily_vals else 0,
            "max_consec_wins":  streak_win,
            "max_consec_losses":streak_loss,
        },
    }


def _empty_stats() -> dict:
    return {
        "total_pnl": 0, "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate_pct": 0, "avg_win_rs": 0, "avg_loss_rs": 0,
        "profit_factor": 0, "max_drawdown_rs": 0,
        "best_day_rs": 0, "worst_day_rs": 0,
        "max_consec_wins": 0, "max_consec_losses": 0,
    }


def export_chart_data(db_path: str = DEFAULT_DB, source: str = "paper") -> Dict[str, Any]:
    """Query DB and return the full JSON payload for the chart."""
    since = _since_date()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    trades: List[Dict] = []
    if source in ("paper", "both"):
        trades += _fetch_paper_trades(conn, since)
    if source in ("backtest", "both"):
        trades += _fetch_backtest_trades(conn, since)

    conn.close()

    trades_sorted = sorted(trades, key=lambda t: t["date"])

    # Separate series by source type if both
    series = {"all": _build_series(trades_sorted)}
    if source == "both":
        series["paper"]    = _build_series([t for t in trades_sorted if t["source"] == "paper"])
        series["backtest"] = _build_series([t for t in trades_sorted if t["source"] == "backtest"])

    return {
        "generated_at": datetime.now().isoformat(),
        "source":       source,
        "lookback_years": LOOKBACK_YEARS,
        "since_date":   since,
        "series":       series,
        "trades":       trades_sorted,
    }


def main():
    ap = argparse.ArgumentParser(description="Export P&L chart data from DB")
    ap.add_argument("--db",     default=DEFAULT_DB,  help="Path to SQLite DB")
    ap.add_argument("--source", default="paper",
                    choices=["paper", "backtest", "both"],
                    help="Data source")
    ap.add_argument("--open",   action="store_true", help="Open chart in browser after export")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Exporting from {args.db} (source={args.source}, last {LOOKBACK_YEARS} years)...")
    data = export_chart_data(db_path=args.db, source=args.source)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Chart data written to {OUTPUT_JSON}")
    print(f"Total trades exported: {len(data['trades'])}")
    print(f"Cumulative P&L: ₹{data['series']['all']['stats']['total_pnl']:,.0f}")
    print(f"Win rate: {data['series']['all']['stats']['win_rate_pct']}%")

    if args.open and OUTPUT_HTML.exists():
        webbrowser.open(OUTPUT_HTML.resolve().as_uri())
        print(f"Opening {OUTPUT_HTML} in browser.")
    elif args.open:
        print(f"HTML chart not found at {OUTPUT_HTML}. Run the setup step first.")


if __name__ == "__main__":
    main()
