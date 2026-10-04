"""
Walk-forward backtester.

For each bar in the test period:
  1. Use only data up to that bar (no lookahead).
  2. Run KronosForecaster on the lookback window.
  3. Generate signal via SignalEngine.
  4. Map to option trade via OptionsMapper.
  5. Price EVERY leg at the real entry time and the real exit time using
     Upstox expired-instruments 1-min option candles.
  6. Apply per-leg brokerage, STT, exchange/SEBI fees, GST, stamp duty and
     slippage (src/costs.py).

A trade is priced either entirely from real option data or, if any leg is
missing and backtest.bs_fallback_when_missing is true, entirely from a
Black-Scholes estimate — never a mix. Fallback trades are tagged
"bs_approximation" in the trade log so you can filter them.

Positions are intraday: squared off at trading.square_off_time, or at the
last bar of the day if the session ends early.

Usage:
    backtester = Backtester(broker=get_broker())
    results = backtester.run("NIFTY", "2025-01-01", "2025-12-31")
    results["equity_curve"].plot()
"""
from __future__ import annotations
import json
import logging
import math
from datetime import datetime, time as dtime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.broker.base import BrokerInterface
from src.costs import trade_result
from src.data_fetcher import DataFetcher
from src.data_cleaner import clean, is_expiry_day
from src.forecaster import KronosForecaster
from src.signal_engine import SignalEngine
from src.options_mapper import OptionsMapper
from src.utils import load_config, IST, is_event_day
from src.db import init_db, get_conn

logger = logging.getLogger(__name__)

# Calendar days fetched before `start` (400 five-minute bars ≈ 5.3 sessions).
WARMUP_DAYS = 14


class Backtester:
    def __init__(
        self,
        broker: BrokerInterface,
        config_path: str = "config.yaml",
        db_path: str = "data/kronos_options.db",
    ):
        self.broker = broker
        self.cfg = load_config(config_path)
        self.db_path = db_path
        self.fetcher   = DataFetcher(broker, config_path)
        self.forecaster = KronosForecaster(config_path, db_path)
        self.signal_eng = SignalEngine(config_path, db_path)
        self.mapper     = OptionsMapper(broker, config_path, db_path)
        init_db(db_path)

    def run(
        self,
        symbol: str,
        start: str,
        end: str,
        bar_resolution: str = "5min",
        signal_bar_interval: int = 6,   # Generate signal every N bars (6 × 5min = 30 min)
        lots: int = 1,
    ) -> dict:
        """
        Walk-forward backtest.

        Args:
            symbol:               NIFTY | BANKNIFTY | SENSEX
            start / end:          Date range "YYYY-MM-DD"
            bar_resolution:       Bar size (currently only "5min" tested)
            signal_bar_interval:  How many bars between signal evaluations
            lots:                 Number of lots per trade

        Returns dict with:
            trade_log:      pd.DataFrame of all trades
            equity_curve:   pd.Series of running P&L
            stats:          dict of performance metrics
        """
        logger.info("Starting backtest: %s %s → %s", symbol, start, end)

        # Load and clean historical index data, plus a warm-up window before
        # `start` so the first test day already has a full lookback.
        warmup = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
        df_raw = self.fetcher.fetch_date_range(symbol, warmup, end, bar_resolution)
        df = clean(df_raw, symbol=symbol)
        start_date = datetime.strptime(start, "%Y-%m-%d").date()

        if df.empty:
            raise ValueError(f"No clean data for {symbol} {start}→{end}")

        inst_cfg  = self.cfg["instruments"][symbol]
        inst_key  = inst_cfg["upstox_key"]
        lot_size  = inst_cfg["lot_size"]
        exchange  = inst_cfg.get("exchange", "NSE")
        bt_cfg    = self.cfg.get("backtest", {})
        tc_cfg    = self.cfg.get("trading", {})
        skip_exp  = tc_cfg.get("skip_expiry_day", True)
        skip_evt  = tc_cfg.get("skip_event_days", True)
        use_real  = bt_cfg.get("use_real_option_data", True)
        bs_ok     = bt_cfg.get("bs_fallback_when_missing", True)
        lookback  = self.cfg["kronos"]["lookback"]

        no_open  = dtime(9, 15 + tc_cfg.get("no_trade_open_mins", 5))
        sq_off_t = dtime(*[int(x) for x in tc_cfg.get("square_off_time", "15:15").split(":")])

        # Real expiry dates — used both to pick the contract and to detect
        # expiry days (exchanges moved expiry weekdays during 2024-2025).
        expiries = sorted(self.broker.get_expired_expiries(inst_key) or [])
        if not expiries:
            logger.warning("%s: broker returned no expired expiries; option pricing will fail", symbol)

        trade_records: list[dict] = []
        equity = 0.0
        equity_series: dict = {}
        open_trade: Optional[dict] = None
        bar_count = 0

        dates = sorted(d for d in set(df.index.date) if d >= start_date)
        if not dates:
            raise ValueError(f"No sessions for {symbol} between {start} and {end}")

        for d in dates:
            date_str = d.strftime("%Y-%m-%d")

            if skip_exp and is_expiry_day(d, symbol, self.cfg, expiries):
                logger.info("Skipping expiry day: %s %s", symbol, date_str)
                continue
            if skip_evt and is_event_day(d, self.cfg):
                logger.info("Skipping event day: %s %s", symbol, date_str)
                continue

            day_bars = df[df.index.date == d]
            if len(day_bars) < 2:
                continue

            for ts, bar in day_bars.iterrows():
                bar_count += 1
                t = ts.time()

                # Square off open position at EOD
                if open_trade and t >= sq_off_t:
                    rec = self._close_trade(open_trade, ts, df, lot_size, lots, exchange, "SQUARE_OFF")
                    if rec:
                        equity += rec["net_pnl_rs"]
                        equity_series[ts] = equity
                        trade_records.append(rec)
                    open_trade = None

                # Only generate new signal every N bars, no open position, within trading hours
                if (
                    open_trade is None
                    and bar_count % signal_bar_interval == 0
                    and no_open <= t < sq_off_t
                ):
                    # Input: all bars up to (not including) the current one.
                    # Bars are labelled by their start, so the bar at `ts` is
                    # still forming at `ts` and must not be seen.
                    idx_pos = df.index.get_loc(ts)
                    hist = df.iloc[max(0, idx_pos - lookback): idx_pos]
                    if len(hist) < 20:
                        continue

                    try:
                        forecast = self.forecaster.forecast(symbol, hist, use_cache=False)
                        signal   = self.signal_eng.generate(forecast, save_to_db=False)
                    except Exception as e:
                        logger.warning("Forecast failed at %s: %s", ts, e)
                        continue

                    if signal["signal"] == "NEUTRAL":
                        continue

                    expiry = next((e for e in expiries if e >= date_str), None)
                    if not expiry:
                        continue

                    recommendation = self.mapper.map(
                        {**signal, "symbol": symbol},
                        expiry=expiry,
                        option_chain=pd.DataFrame(),  # no live chain for a past date
                    )
                    if recommendation.get("strategy") == "NO_TRADE" or not recommendation.get("legs"):
                        continue

                    entry = self._price_legs(
                        symbol, recommendation["legs"], ts, expiry, df, use_real, bs_ok, source=None,
                    )
                    if entry is None:
                        logger.debug("No entry prices for %s %s @ %s — skipped", symbol,
                                     recommendation["strategy"], ts)
                        continue
                    leg_prices, source = entry

                    open_trade = {
                        "recommendation": recommendation,
                        "entry_ts":       ts,
                        "signal":         signal,
                        "expiry":         expiry,
                        "entry_prices":   leg_prices,   # [(price, contract_lot_size)]
                        "data_source":    source,
                    }
                    logger.debug("Opened %s %s @ %s (%s)", symbol, recommendation["strategy"], ts, source)

            # Session ended before square-off time (half day / missing bars):
            # intraday positions never carry overnight.
            if open_trade:
                last_ts = day_bars.index[-1]
                rec = self._close_trade(open_trade, last_ts, df, lot_size, lots, exchange, "SESSION_END")
                if rec:
                    equity += rec["net_pnl_rs"]
                    equity_series[last_ts] = equity
                    trade_records.append(rec)
                open_trade = None

        trade_log = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()
        equity_curve = pd.Series(equity_series, name="equity_rs", dtype=float)

        stats = self._compute_stats(trade_log, dates)

        logger.info(
            "Backtest complete: %d trades | P&L=₹%.0f | Win=%.1f%% | Sharpe=%.2f",
            stats["total_trades"], stats["total_pnl_rs"],
            stats["win_rate_pct"], stats["sharpe"],
        )

        self._save_run(symbol, start, end, stats, trade_log)

        return {
            "trade_log":    trade_log,
            "equity_curve": equity_curve,
            "stats":        stats,
        }

    # ── Pricing ────────────────────────────────────────────────────────

    def _price_legs(
        self,
        symbol: str,
        legs: list[dict],
        ts: pd.Timestamp,
        expiry: str,
        df: pd.DataFrame,
        use_real: bool,
        bs_ok: bool,
        source: Optional[str],
    ) -> Optional[tuple[list[tuple[float, int]], str]]:
        """
        Price all legs at ts. Returns ([(price, lot_size)], source) or None.

        source=None: try real data first, then BS (entry).
        source given: price with that same source (exit), so a trade is never
        opened on real prices and closed on model prices or vice versa.
        """
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H:%M")

        if use_real and source in (None, "real_option_data"):
            prices = []
            for leg in legs:
                got = self.fetcher.get_option_price_at(
                    symbol, date_str, leg["strike"], leg["option_type"], time_str, expiry,
                )
                if got is None:
                    prices = None
                    break
                prices.append(got)
            if prices is not None:
                return prices, "real_option_data"
            if source == "real_option_data":
                # A real-priced trade with no candle at or before the exit
                # minute: drop it rather than close it on model prices.
                return None

        if not bs_ok or source not in (None, "bs_approximation"):
            return None
        spot = self._spot_at(df, ts)
        if spot is None:
            return None
        prices = []
        for leg in legs:
            p = self._bs_price(symbol, spot, leg["strike"], leg["option_type"], ts, expiry)
            prices.append((p, 0))
        return prices, "bs_approximation"

    @staticmethod
    def _spot_at(df: pd.DataFrame, ts: pd.Timestamp) -> Optional[float]:
        """Index level known at ts: close of the last completed bar before ts."""
        prior = df[df.index < ts]
        if prior.empty:
            return None
        return float(prior["close"].iloc[-1])

    def _bs_price(self, symbol, spot, strike, opt_type, ts: pd.Timestamp, expiry: str) -> float:
        """
        Black-Scholes estimate. CAUTION: a flat assumed IV ignores skew and
        intraday IV moves — filter 'bs_approximation' trades from any serious
        performance analysis.
        """
        from scipy.stats import norm

        iv = self.cfg["backtest"].get("iv_assumption_pct", 15.0) / 100
        r = 0.065  # approx. Indian risk-free rate
        exp_ts = IST.localize(datetime.strptime(expiry + " 15:30", "%Y-%m-%d %H:%M"))
        # Time to expiry in years, measured to the minute so intraday theta shows up.
        T = max((exp_ts - ts).total_seconds(), 60) / (365 * 24 * 3600)
        S, K = spot, float(strike)

        d1 = (math.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
        d2 = d1 - iv * math.sqrt(T)
        if opt_type == "CE":
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return round(max(price, 0.05), 2)

    # ── Trade close helper ─────────────────────────────────────────────

    def _close_trade(
        self,
        open_trade: dict,
        exit_ts: pd.Timestamp,
        df: pd.DataFrame,
        lot_size: int,
        lots: int,
        exchange: str,
        exit_reason: str,
    ) -> Optional[dict]:
        rec  = open_trade["recommendation"]
        legs = rec["legs"]
        exp  = open_trade["expiry"]
        bt_cfg = self.cfg.get("backtest", {})

        exit_ = self._price_legs(
            rec["symbol"], legs, exit_ts, exp, df,
            use_real=bt_cfg.get("use_real_option_data", True),
            bs_ok=bt_cfg.get("bs_fallback_when_missing", True),
            source=open_trade["data_source"],
        )
        if exit_ is None:
            logger.warning("No exit prices for %s %s @ %s — trade dropped",
                           rec["symbol"], rec["strategy"], exit_ts)
            return None
        exit_prices, _ = exit_

        priced_legs = []
        for leg, (entry_p, entry_lot), (exit_p, _) in zip(legs, open_trade["entry_prices"], exit_prices):
            priced_legs.append({
                "action":      leg["action"],
                "lots":        lots,
                "lot_size":    entry_lot or lot_size,   # contract's own lot size if known
                "entry_price": entry_p,
                "exit_price":  exit_p,
            })
        result = trade_result(priced_legs, lot_size, self.cfg, exchange)

        # Net premium per unit (debit positive), for the trade log.
        def net(side: str) -> float:
            return sum((p[f"{side}_price"] if p["action"] == "BUY" else -p[f"{side}_price"])
                       for p in priced_legs)

        leg0 = legs[0]
        entry_net, exit_net = net("entry"), net("exit")
        return {
            "trade_date":    exit_ts.strftime("%Y-%m-%d"),
            "entry_time":    open_trade["entry_ts"].strftime("%H:%M"),
            "exit_time":     exit_ts.strftime("%H:%M"),
            "symbol":        rec["symbol"],
            "signal":        open_trade["signal"]["signal"],
            "strategy":      rec["strategy"],
            "strike":        leg0["strike"],
            "option_type":   leg0["option_type"],
            "legs":          " ".join(f"{l['action'][0]}{l['strike']}{l['option_type']}" for l in legs),
            "expiry":        exp,
            # Net premium per unit, always positive; premium_type says whether
            # it was paid (DEBIT) or received (CREDIT) on entry.
            "premium_type":  "DEBIT" if entry_net >= 0 else "CREDIT",
            "entry_price":   round(abs(entry_net), 2),
            "exit_price":    round(abs(exit_net), 2),
            "lots":          lots,
            "lot_size":      priced_legs[0]["lot_size"],
            **result,
            "sl_tgt_tag":    exit_reason,
            "data_source":   open_trade["data_source"],
        }

    # ── Performance stats ──────────────────────────────────────────────

    @staticmethod
    def _compute_stats(trade_log: pd.DataFrame, dates: list) -> dict:
        if trade_log.empty:
            return {"total_trades": 0, "wins": 0, "losses": 0, "total_pnl_rs": 0,
                    "win_rate_pct": 0, "avg_win_rs": 0, "avg_loss_rs": 0,
                    "sharpe": 0, "max_drawdown_rs": 0, "profit_factor": 0}

        pnls = trade_log["net_pnl_rs"]
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        total_trades = len(pnls)
        win_rate = len(wins) / total_trades * 100 if total_trades else 0
        profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")

        # Daily P&L over EVERY session in the test (0 on days without a
        # trade) — dropping flat days would inflate the Sharpe ratio.
        all_days = pd.Index([d.strftime("%Y-%m-%d") for d in dates])
        daily = trade_log.groupby("trade_date")["net_pnl_rs"].sum().reindex(all_days, fill_value=0.0)
        std = daily.std()
        sharpe = float(daily.mean() / std * np.sqrt(252)) if std and not np.isnan(std) else 0.0

        # Max drawdown from a zero starting equity
        equity = pd.concat([pd.Series([0.0]), pnls.cumsum()], ignore_index=True)
        max_dd = float((equity - equity.cummax()).min())

        return {
            "total_trades":   total_trades,
            "wins":           len(wins),
            "losses":         len(losses),
            "total_pnl_rs":   round(float(pnls.sum()), 2),
            "win_rate_pct":   round(win_rate, 2),
            "avg_win_rs":     round(float(wins.mean()), 2) if len(wins) else 0,
            "avg_loss_rs":    round(float(losses.mean()), 2) if len(losses) else 0,
            "profit_factor":  round(profit_factor, 3),
            "sharpe":         round(sharpe, 3),
            "max_drawdown_rs": round(max_dd, 2),
        }

    def _save_run(self, symbol, start, end, stats, trade_log):
        with get_conn(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO backtest_runs
                   (run_at, symbol, start_date, end_date, total_trades, wins, losses,
                    total_pnl_rs, sharpe, max_drawdown_rs, win_rate_pct, profit_factor, params)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.now(IST).isoformat(), symbol, start, end,
                    stats["total_trades"], stats.get("wins", 0), stats.get("losses", 0),
                    stats["total_pnl_rs"], stats["sharpe"], stats["max_drawdown_rs"],
                    stats["win_rate_pct"], stats["profit_factor"],
                    json.dumps({"signal": self.cfg.get("signal", {}),
                                "kronos": self.cfg.get("kronos", {}),
                                "costs": self.cfg.get("costs", {})}),
                ),
            )
            run_id = cur.lastrowid
            if not trade_log.empty:
                for _, row in trade_log.iterrows():
                    conn.execute(
                        """INSERT INTO backtest_trades
                           (run_id, trade_date, symbol, signal, strategy, strike, option_type,
                            expiry, entry_price, exit_price, lots, lot_size,
                            raw_pnl_rs, charges_rs, net_pnl_rs, sl_tgt_tag, data_source,
                            legs, premium_type, entry_time, exit_time)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            run_id, row.get("trade_date"), symbol,
                            row.get("signal"), row.get("strategy"),
                            _py(row.get("strike")), row.get("option_type"), row.get("expiry"),
                            _py(row.get("entry_price")), _py(row.get("exit_price")),
                            _py(row.get("lots")), _py(row.get("lot_size")),
                            _py(row.get("raw_pnl_rs")), _py(row.get("charges_rs")), _py(row.get("net_pnl_rs")),
                            row.get("sl_tgt_tag"), row.get("data_source"),
                            row.get("legs"), row.get("premium_type"),
                            row.get("entry_time"), row.get("exit_time"),
                        ),
                    )


def _py(v):
    """numpy scalar -> Python scalar (sqlite3 rejects numpy.int64)."""
    return v.item() if hasattr(v, "item") else v
