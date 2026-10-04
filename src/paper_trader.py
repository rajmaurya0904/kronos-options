"""
Paper trader — live loop that runs during market hours.

Every 5 minutes:
  1. Fetch latest bars (history + today's intraday) for all enabled instruments.
  2. Run Kronos forecast → signal → option recommendation.
  3. Log simulated trade to SQLite (no real orders placed).
  4. Send desktop notification on new signal.
  5. Mark open positions to market.
  6. Square off all positions by 15:15 IST.

Run:
    python -m src.paper_trader
    python -m src.paper_trader --symbols NIFTY BANKNIFTY
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from datetime import datetime, time as dtime

import pandas as pd

from src.broker.base import BrokerInterface
from src.costs import trade_result
from src.data_fetcher import DataFetcher
from src.data_cleaner import clean, is_expiry_day
from src.forecaster import KronosForecaster
from src.signal_engine import SignalEngine
from src.options_mapper import OptionsMapper
from src.utils import get_broker, load_config, now_ist, is_market_open, is_event_day
from src.db import init_db, get_conn

logger = logging.getLogger(__name__)

# Seconds to wait past each 5-min boundary so the just-closed bar has been
# published by the intraday candle endpoint before we read it.
BAR_SETTLE_S = 10


def notify(title: str, message: str) -> None:
    """Send desktop notification. Silently skips if plyer unavailable."""
    try:
        from plyer import notification
        notification.notify(title=title, message=message, timeout=10)
    except Exception:
        pass


class PaperTrader:
    def __init__(
        self,
        broker: BrokerInterface,
        symbols: list[str],
        config_path: str = "config.yaml",
        db_path: str = "data/kronos_options.db",
    ):
        self.broker    = broker
        self.symbols   = symbols
        self.cfg       = load_config(config_path)
        self.db_path   = db_path
        self.fetcher   = DataFetcher(broker, config_path)
        self.forecaster = KronosForecaster(config_path, db_path)
        self.signal_eng = SignalEngine(config_path, db_path)
        self.mapper     = OptionsMapper(broker, config_path, db_path)

        init_db(db_path)

        pt_cfg = self.cfg.get("paper_trading", {})
        self.max_open  = pt_cfg.get("max_open_positions", 3)
        self.lots      = pt_cfg.get("lots", 1)
        self.notify_on = pt_cfg.get("notify_on_signal", True)

        tc = self.cfg.get("trading", {})
        sq_h, sq_m = [int(x) for x in tc.get("square_off_time", "15:15").split(":")]
        self.sq_off_time    = dtime(sq_h, sq_m)
        self.no_trade_until = dtime(9, 15 + tc.get("no_trade_open_mins", 5))

        self._open_positions: dict[str, dict] = {}  # symbol → trade rec
        self._expiry_cache: dict[str, tuple] = {}   # symbol → (date, [expiries])

        self._abandon_stale_rows()

    def _abandon_stale_rows(self) -> None:
        """
        Open positions live in memory only, so after a restart any row still
        marked OPEN can never be closed. Mark them so P&L pages don't show
        phantom positions forever.
        """
        with get_conn(self.db_path) as conn:
            n = conn.execute(
                "UPDATE paper_trades SET status='ABANDONED', exit_reason='RESTART' WHERE status='OPEN'"
            ).rowcount
        if n:
            logger.warning("Marked %d OPEN paper trade(s) from a previous run as ABANDONED.", n)

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info("Paper trader started for: %s", self.symbols)

        while True:
            now = now_ist()

            if not is_market_open(now):
                logger.debug("Market closed. Sleeping 60s.")
                time.sleep(60)
                continue

            try:
                self._tick(now)
            except Exception as e:
                logger.error("Tick error: %s", e, exc_info=True)

            # Sleep until just after the next 5-min bar boundary
            now = now_ist()
            seconds_past = now.minute % 5 * 60 + now.second
            sleep_for = 300 - seconds_past + BAR_SETTLE_S
            logger.debug("Sleeping %ds until next bar.", sleep_for)
            time.sleep(sleep_for)

    def _tick(self, now: datetime) -> None:
        """Single processing tick: fetch → forecast → signal → log."""
        t = now.time()

        # Square off all positions
        if t >= self.sq_off_time:
            self._square_off_all(now)
            return

        # Mark open positions to market
        self._mark_to_market(now)

        if t < self.no_trade_until:
            logger.debug("Waiting for market to settle (before %s).", self.no_trade_until)
            return

        if self.cfg["trading"].get("skip_event_days", True) and is_event_day(now.date(), self.cfg):
            logger.info("Event day in config — no new trades today.")
            return

        # Process each symbol
        for symbol in self.symbols:
            if len(self._open_positions) >= self.max_open:
                break
            if symbol in self._open_positions:
                continue
            expiries = self._expiries(symbol, now)
            if (self.cfg["trading"].get("skip_expiry_day", True)
                    and is_expiry_day(now.date(), symbol, self.cfg, expiries)):
                logger.info("Skipping %s — expiry day.", symbol)
                continue

            self._process_symbol(symbol, now)

    def _expiries(self, symbol: str, now: datetime) -> list[str]:
        """Current/future expiries, fetched once per day per symbol."""
        cached = self._expiry_cache.get(symbol)
        if cached and cached[0] == now.date():
            return cached[1]
        inst_key = self.cfg["instruments"][symbol]["upstox_key"]
        try:
            expiries = self.broker.get_expiries(inst_key)
        except Exception as e:
            logger.error("%s: could not fetch expiries: %s", symbol, e)
            expiries = []
        if expiries:  # don't cache a failure for the whole day
            self._expiry_cache[symbol] = (now.date(), expiries)
        return expiries

    def _process_symbol(self, symbol: str, now: datetime) -> None:
        """Fetch → forecast → signal → map → log for one symbol."""
        lookback = self.cfg["kronos"]["lookback"]

        try:
            df_raw = self.fetcher.fetch_latest_bars(symbol, n_bars=lookback + 50)
            df = clean(df_raw, symbol=symbol)
            if len(df) < 20:
                logger.warning("%s: not enough bars (%d).", symbol, len(df))
                return
            if df.index[-1].date() != now.date():
                logger.warning("%s: no bars from today yet — not forecasting off yesterday.", symbol)
                return

            forecast = self.forecaster.forecast(symbol, df)
            signal   = self.signal_eng.generate(forecast)

            if signal["signal"] == "NEUTRAL":
                logger.info("%s: NEUTRAL — no trade.", symbol)
                return

            # Nearest listed expiry (today's is excluded when we got here,
            # since expiry days are skipped above)
            today_str = now.strftime("%Y-%m-%d")
            expiry = next((e for e in self._expiries(symbol, now) if e >= today_str), None)
            if not expiry:
                logger.warning("%s: no valid expiry found.", symbol)
                return

            # Get live option chain
            inst_key = self.cfg["instruments"][symbol]["upstox_key"]
            try:
                chain = self.broker.get_option_chain(inst_key, expiry)
            except Exception:
                chain = pd.DataFrame()
            if chain.empty:
                logger.warning("%s: empty option chain for %s — no trade.", symbol, expiry)
                return

            signal["symbol"] = symbol
            rec = self.mapper.map(signal, expiry=expiry, option_chain=chain)

            if rec.get("strategy") == "NO_TRADE":
                return
            missing = [f"{l['strike']}{l['option_type']}" for l in rec["legs"] if not l.get("ltp")]
            if missing:
                # A zero entry price would turn into fake P&L at exit.
                logger.warning("%s: no LTP for %s — no trade.", symbol, ", ".join(missing))
                return

            self._open_paper_position(symbol, rec, signal, now)

        except Exception as e:
            logger.error("Error processing %s: %s", symbol, e, exc_info=True)

    # ── Position management ────────────────────────────────────────────

    @staticmethod
    def _net_premium(legs: list[dict], prices: list[float]) -> float:
        """Net premium per unit: debit positive, credit negative."""
        return sum(p if l["action"] == "BUY" else -p for l, p in zip(legs, prices))

    def _open_paper_position(self, symbol: str, rec: dict, signal: dict, now: datetime) -> None:
        legs = rec.get("legs", [])
        entry_premium = self._net_premium(legs, [l["ltp"] for l in legs])

        legs_json = json.dumps(legs)
        with get_conn(self.db_path) as conn:
            trade_id = conn.execute(
                """INSERT INTO paper_trades
                   (symbol, strategy, entry_time, status, legs, entry_total_premium)
                   VALUES (?,?,?,?,?,?)""",
                (symbol, rec["strategy"], now.isoformat(), "OPEN", legs_json, entry_premium),
            ).lastrowid

        self._open_positions[symbol] = {
            **rec,
            "trade_id": trade_id,
            "entry_time": now.isoformat(),
            "entry_premium": entry_premium,
            "signal": signal["signal"],
            "confidence": signal["confidence"],
        }

        msg = (f"{rec['strategy']} | confidence={signal['confidence']:.2f} | "
               f"max_loss=₹{rec.get('max_loss_rs', 0):.0f}")
        logger.info("PAPER TRADE OPEN: %s %s", symbol, msg)
        if self.notify_on:
            notify(f"New Signal: {symbol}", msg)

    def _square_off_all(self, now: datetime) -> None:
        """Close all open positions at current prices."""
        if not self._open_positions:
            return
        logger.info("Squaring off %d positions at %s.", len(self._open_positions), now.strftime("%H:%M"))
        for symbol in list(self._open_positions.keys()):
            self._close_position(symbol, now, reason="SQUARE_OFF")

    def _result(self, symbol: str, pos: dict, exit_prices: list[float]) -> dict:
        inst = self.cfg["instruments"][symbol]
        priced = [
            {"action": l["action"], "lots": l.get("lots", 1),
             "entry_price": l["ltp"], "exit_price": p}
            for l, p in zip(pos["legs"], exit_prices)
        ]
        return trade_result(priced, inst["lot_size"], self.cfg, inst.get("exchange", "NSE"))

    def _close_position(self, symbol: str, now: datetime, reason: str = "SIGNAL_EXIT") -> None:
        pos = self._open_positions.pop(symbol, None)
        if not pos:
            return

        exit_prices = self._fetch_leg_ltps(symbol, pos["legs"], pos["expiry"])
        exit_premium = self._net_premium(pos["legs"], exit_prices)
        res = self._result(symbol, pos, exit_prices)

        logger.info("PAPER TRADE CLOSE: %s %s | exit_prem=%.2f | net_pnl=₹%.0f | reason=%s",
                    symbol, pos["strategy"], exit_premium, res["net_pnl_rs"], reason)

        with get_conn(self.db_path) as conn:
            conn.execute(
                """UPDATE paper_trades SET
                   exit_time=?, status='CLOSED', exit_total_premium=?,
                   raw_pnl_rs=?, charges_rs=?, net_pnl_rs=?, exit_reason=?
                   WHERE id=?""",
                (now.isoformat(), exit_premium, res["raw_pnl_rs"], res["charges_rs"],
                 res["net_pnl_rs"], reason, pos["trade_id"]),
            )
        return res

    def _fetch_leg_ltps(self, symbol: str, legs: list[dict], expiry: str) -> list[float]:
        """
        Live LTP for each leg from the option chain, in leg order.
        A leg with no live price keeps its entry price (logged), so a data
        gap shows as zero P&L for that leg rather than a fake swing.
        """
        inst_key = self.cfg["instruments"][symbol]["upstox_key"]
        try:
            chain = self.broker.get_option_chain(inst_key, expiry)
        except Exception:
            chain = pd.DataFrame()

        prices = []
        for leg in legs:
            col = f"{leg['option_type']}_ltp"
            ltp = None
            if not chain.empty and col in chain.columns:
                row = chain[chain["strike"] == leg["strike"]]
                if not row.empty and pd.notna(row[col].iloc[0]) and float(row[col].iloc[0]) > 0:
                    ltp = float(row[col].iloc[0])
            if ltp is None:
                logger.warning("%s: no live price for %s%s — using entry price",
                               symbol, leg["strike"], leg["option_type"])
                ltp = leg["ltp"]
            prices.append(ltp)
        return prices

    def _mark_to_market(self, now: datetime) -> None:
        """Log current unrealised P&L for all open positions."""
        for symbol, pos in self._open_positions.items():
            try:
                prices = self._fetch_leg_ltps(symbol, pos["legs"], pos["expiry"])
                res = self._result(symbol, pos, prices)
                logger.info("MTM %s %s: unrealised P&L = ₹%.0f (after costs)",
                            symbol, pos["strategy"], res["net_pnl_rs"])
            except Exception as e:
                logger.debug("MTM failed for %s: %s", symbol, e)


# ── CLI entry point ────────────────────────────────────────────────────

def main():
    from src.utils import setup_logging
    setup_logging("paper_trader")

    parser = argparse.ArgumentParser(description="Kronos Options Paper Trader")
    parser.add_argument("--symbols", nargs="+", default=["NIFTY", "BANKNIFTY", "SENSEX"])
    args = parser.parse_args()

    broker = get_broker()
    enabled = [s for s in args.symbols
               if load_config()["instruments"].get(s, {}).get("enabled", False)]
    if not enabled:
        logger.error("No enabled symbols found in config.")
        return

    trader = PaperTrader(broker=broker, symbols=enabled)
    trader.run()


if __name__ == "__main__":
    main()
