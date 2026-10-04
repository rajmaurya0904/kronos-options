"""
Streamlit dashboard for Kronos Options Signal System.

Run:
    streamlit run dashboard/app.py

Pages:
  1. Live Forecasts   — predicted vs actual chart for all 3 indices
  2. Today's Signals  — signal log + recommended trades
  3. Paper Trading P&L — equity curve + open positions
  4. Backtest Results — run viewer and trade log
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow importing src modules from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import get_paper_trades, get_signals_today, get_conn
from src.utils import load_config, now_ist

st.set_page_config(
    page_title="Kronos Options Dashboard",
    page_icon="📈",
    layout="wide",
)

cfg = load_config()
DB = cfg["database"]["path"]

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.title("Kronos Options")
st.sidebar.caption(f"IST: {now_ist().strftime('%Y-%m-%d %H:%M:%S')}")
page = st.sidebar.radio(
    "Page",
    ["Live Forecasts", "Today's Signals", "Paper P&L", "Backtest Results"],
)

# ══════════════════════════════════════════════════════════════════════
#  PAGE 1 — Live Forecasts
# ══════════════════════════════════════════════════════════════════════

if page == "Live Forecasts":
    st.title("Live Forecasts")
    st.caption("Kronos-small probabilistic OHLCV forecasts. Refreshes when new forecast is logged.")

    with get_conn(DB) as conn:
        forecasts = conn.execute(
            "SELECT * FROM forecasts ORDER BY created_at DESC LIMIT 10"
        ).fetchall()

    if not forecasts:
        st.info("No forecasts yet. Run paper_trader.py to generate forecasts.")
        st.stop()

    symbols = list({r["symbol"] for r in forecasts})
    sel_symbol = st.selectbox("Symbol", sorted(symbols))

    latest = next((r for r in forecasts if r["symbol"] == sel_symbol), None)
    if not latest:
        st.warning(f"No forecast for {sel_symbol}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Move", f"{latest['expected_move_pct']:+.3f}%")
    col2.metric("Prob Up", f"{latest['prob_up']:.1%}")
    col3.metric("Path Dispersion", f"{latest['path_dispersion']:.3f}%")
    col4.metric("Expected Close", f"₹{latest['expected_close']:,.2f}")

    # Forecast fan chart
    sample_paths = json.loads(latest["sample_paths"])
    median_data  = json.loads(latest["median_path"])

    fig = go.Figure()

    # Draw sample paths (faint)
    for path in sample_paths[:10]:
        fig.add_trace(go.Scatter(
            y=path, mode="lines",
            line=dict(color="rgba(100, 149, 237, 0.2)", width=1),
            showlegend=False,
        ))

    # Draw median
    median_closes = [r.get("close", r.get("close")) for r in median_data]
    fig.add_trace(go.Scatter(
        y=median_closes, mode="lines",
        line=dict(color="royalblue", width=2),
        name="Median forecast",
    ))

    fig.update_layout(
        title=f"{sel_symbol} Forecast — {latest['pred_len']} bars ahead",
        xaxis_title="Bar ahead", yaxis_title="Price (₹)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Generated: {latest['created_at']} | Bars ahead: {latest['pred_len']}")

# ══════════════════════════════════════════════════════════════════════
#  PAGE 2 — Today's Signals
# ══════════════════════════════════════════════════════════════════════

elif page == "Today's Signals":
    st.title("Today's Signals & Trade Recommendations")

    signals = get_signals_today(db_path=DB)

    if signals.empty:
        st.info("No signals generated today.")
        st.stop()

    # Signal summary
    for _, row in signals.iterrows():
        sig = row["signal"]
        color_map = {
            "STRONG_BULLISH": "🟢", "BULLISH": "🟩",
            "NEUTRAL": "⬜", "BEARISH": "🟥", "STRONG_BEARISH": "🔴",
        }
        icon = color_map.get(sig, "")
        with st.expander(f"{icon} {row['signal_time'][:16]} | {row['symbol']} | {sig} | conf={row['confidence']:.2f}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Expected Move", f"{row['expected_move_pct']:+.3f}%")
            c2.metric("Prob Up", f"{row['prob_up']:.1%}")
            c3.metric("Range Low",  f"₹{row['expected_range_low'] or 0:,.0f}")
            c4.metric("Range High", f"₹{row['expected_range_high'] or 0:,.0f}")

    st.divider()
    st.subheader("Raw Signal Log")
    st.dataframe(signals, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  PAGE 3 — Paper P&L
# ══════════════════════════════════════════════════════════════════════

elif page == "Paper P&L":
    st.title("Paper Trading P&L")

    open_trades  = get_paper_trades("OPEN",   db_path=DB)
    closed_trades = get_paper_trades("CLOSED", db_path=DB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Open Positions ({len(open_trades)})")
        if open_trades.empty:
            st.info("No open positions.")
        else:
            st.dataframe(open_trades[["symbol", "strategy", "entry_time", "entry_total_premium"]])

    with col2:
        total_pnl = closed_trades["net_pnl_rs"].sum() if not closed_trades.empty else 0
        wins = (closed_trades["net_pnl_rs"] > 0).sum() if not closed_trades.empty else 0
        total = len(closed_trades)
        st.metric("Total Paper P&L", f"₹{total_pnl:,.0f}")
        st.metric("Win Rate", f"{wins}/{total} ({100*wins/total:.0f}%)" if total else "—")

    # Equity curve
    if not closed_trades.empty and "exit_time" in closed_trades.columns:
        closed_trades["exit_time"] = pd.to_datetime(closed_trades["exit_time"])
        closed_trades = closed_trades.sort_values("exit_time")
        closed_trades["cumulative_pnl"] = closed_trades["net_pnl_rs"].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=closed_trades["exit_time"],
            y=closed_trades["cumulative_pnl"],
            mode="lines+markers",
            name="Equity Curve",
            line=dict(color="green" if total_pnl >= 0 else "red"),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="Paper P&L Equity Curve", xaxis_title="Date", yaxis_title="₹")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Closed Trades")
    if closed_trades.empty:
        st.info("No closed trades yet.")
    else:
        st.dataframe(closed_trades, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  PAGE 4 — Backtest Results
# ══════════════════════════════════════════════════════════════════════

elif page == "Backtest Results":
    st.title("Backtest Results")

    with get_conn(DB) as conn:
        runs = pd.DataFrame([dict(r) for r in conn.execute(
            "SELECT * FROM backtest_runs ORDER BY run_at DESC LIMIT 20"
        ).fetchall()])

    if runs.empty:
        st.info("No backtest runs yet. Run backtester.py to generate results.")
        st.stop()

    st.subheader("Recent Runs")
    st.dataframe(
        runs[["run_at", "symbol", "start_date", "end_date",
              "total_trades", "win_rate_pct", "total_pnl_rs",
              "sharpe", "max_drawdown_rs", "profit_factor"]],
        use_container_width=True,
    )

    sel_run = st.selectbox("Select run to inspect", runs["id"].tolist())
    if sel_run:
        with get_conn(DB) as conn:
            trades = pd.DataFrame([dict(r) for r in conn.execute(
                "SELECT * FROM backtest_trades WHERE run_id = ? ORDER BY trade_date",
                (sel_run,),
            ).fetchall()])

        if not trades.empty:
            trades["cumulative_pnl"] = trades["net_pnl_rs"].cumsum()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades.index, y=trades["cumulative_pnl"],
                mode="lines+markers", name="Equity Curve",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(title=f"Backtest Equity Curve — Run {sel_run}",
                              xaxis_title="Trade #", yaxis_title="₹")
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                bs_pct = (trades["data_source"] == "bs_approximation").mean() * 100
                if bs_pct > 0:
                    st.warning(f"⚠ {bs_pct:.1f}% of trades used Black-Scholes approximation "
                               f"(real option data unavailable). Filter these for accurate stats.")
                else:
                    st.success("All trades priced from real option data.")
            # Readable trade log: what was traded, when, at what net premium.
            # Premiums are per unit; CREDIT = received on entry, DEBIT = paid.
            view = trades.rename(columns={
                "trade_date": "date", "premium_type": "type",
                "entry_price": "entry_prem", "exit_price": "exit_prem",
                "net_pnl_rs": "net_pnl_₹", "charges_rs": "costs_₹", "sl_tgt_tag": "exit",
            })
            cols = ["date", "entry_time", "exit_time", "signal", "strategy", "legs", "type",
                    "entry_prem", "exit_prem", "lots", "lot_size", "costs_₹", "net_pnl_₹",
                    "exit", "data_source"]
            view = view[[c for c in cols if c in view.columns]]
            st.caption("Premiums are net per unit across all legs. CREDIT = received on entry, "
                       "DEBIT = paid. Legs: B = buy, S = sell.")
            st.dataframe(view, use_container_width=True, hide_index=True)
