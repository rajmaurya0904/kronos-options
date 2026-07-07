"""
Paper trader — live loop that runs during market hours.

Every 5 minutes:
  1. Fetch latest bars for all enabled instruments.
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
from datetime import datetime, date

import pandas as pd

from src.broker.base import BrokerInterface
from src.data_fetcher import DataFetcher
from src.data_cleaner import clean, is_expiry_day
from src.forecaster import KronosForecaster
from src.signal_engine import SignalEngine
from src.options_mapper import OptionsMapper
from src.utils import get_broker, get_instrument_key, load_config, now_ist, is_market_open, IST
from src.db import init_db, save_signal, get_conn, get_paper_trades

logger = logging.getLogger(__name__)


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
        self.config_path = config_path
        self.db_path   = db_path
        self.fetcher   = DataFetcher(broker, config_path)
        self.forecaster = KronosForecaster(config_path, db_path)
        self.signal_eng = SignalEngine(config_path, db_path)
        self.mapper     = OptionsMapper(broker, config_path, db_path)

        init_db(db_path)

        pt_cfg = self.cfg.get("paper_trading", {})
        self.max_open  = pt_cfg.get("max_open_positions", 3)
        self.lots      = pt_cfg.get("lots", 1)

        tc = self.cfg.get("trading", {})
        self.sq_off_h, self.sq_off_m = [int(x) for x in tc.get("square_off_time", "15:15").split(":")]
        self.no_trade_open = tc.get("no_trade_open_mins", 5)

        self._open_positions: dict[str, dict] = {}  # symbol → trade rec

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info("Paper trader started for: %s", self.symbols)

        while True:
            now = now_ist()

            if not is_market_open(now):
                next_check = 60
                logger.debug("Market closed. Sleeping %ds.", next_check)
                time.sleep(next_check)
                continue

            try:
                self._tick(now)
            except Exception as e:
                logger.error("Tick error: %s", e, exc_info=True)

            seconds_past = now.minute % 5 * 60 + now.second
            sleep_for = max(5, 300 - seconds_past)
            logger.debug("Sleeping %ds until next bar.", sleep_for)
            time.sleep(sleep_for)

    def _tick(self, now: datetime) -> None:
        """Single processing tick: fetch → forecast → signal → log."""
        t = now.time()
        from datetime import time as dtime
        no_trade_until = dtime(9, 15 + self.no_trade_open)
        sq_off_time    = dtime(self.sq_off_h, self.sq_off_m)

        if t >= sq_off_time:
            self._square_off_all(now)
            return

        self._mark_to_market(now)

        if t < no_trade_until:
            logger.debug("Waiting for market to settle (before %s).", no_trade_until)
            return

        open_count = len(self._open_positions)
        for symbol in self.symbols:
            if open_count >= self.max_open:
                break
            if symbol in self._open_positions:
                continue
            if is_expiry_day(now.date(), symbol, self.cfg) and self.cfg["trading"].get("skip_expiry_day", True):
                logger.info("Skipping %s — expiry day.", symbol)
                continue

            self._process_symbol(symbol, now)
            open_count = len(self._open_positions)

    def _process_symbol(self, symbol: str, now: datetime) -> None:
        """Fetch → forecast → signal → map → log for one symbol."""
        lookback = self.cfg["kronos"]["lookback"]

        try:
            df_raw = self.fetcher.fetch_latest_bars(symbol, n_bars=lookback + 50)
            df = clean(df_raw, symbol=symbol)
            if len(df) < 20:
                logger.warning("%s: not enough bars (%d).", symbol, len(df))
                return

            forecast = self.forecaster.forecast(symbol, df)
            signal   = self.signal_eng.generate(forecast)

            if signal["signal"] == "NEUTRAL":
                logger.info("%s: NEUTRAL — no trade.", symbol)
                return

            inst_key = get_instrument_key(symbol, self.config_path)
            expiries = self.broker.get_expired_expiries(inst_key)
            today_str = now.strftime("%Y-%m-%d")
            valid = [e for e in expiries if e >= today_str]
            if not valid:
                logger.warning("%s: no valid expiry found.", symbol)
                return
            expiry = sorted(valid)[0]

            try:
                chain = self.broker.get_option_chain(inst_key, expiry)
            except Exception:
                chain = pd.DataFrame()

            signal["symbol"] = symbol
            rec = self.mapper.map(signal, expiry=expiry, option_chain=chain)

            if rec.get("strategy") == "NO_TRADE":
                return

            self._open_paper_position(symbol, rec, signal, now)

        except Exception as e:
            logger.error("Error processing %s: %s", symbol, e, exc_info=True)

    # ── Position management ────────────────────────────────────────────

    def _open_paper_position(self, symbol: str, rec: dict, signal: dict, now: datetime) -> None:
        entry_premium = sum(
            leg.get("ltp", 0) for leg in rec.get("legs", [])
            if leg["action"] == "BUY"
        ) - sum(
            leg.get("ltp", 0) for leg in rec.get("legs", [])
            if leg["action"] == "SELL"
        )

        trade = {
            **rec,
            "entry_time": now.isoformat(),
            "entry_premium": entry_premium,
            "signal": signal["signal"],
            "confidence": signal["confidence"],
        }
        self._open_positions[symbol] = trade

        legs_json = json.dumps(rec.get("legs", []))
        with get_conn(self.db_path) as conn:
            conn.execute(
                """INSERT INTO paper_trades
                   (symbol, strategy, entry_time, status, legs, entry_total_premium)
                   VALUES (?,?,?,?,?,?)""",
                (symbol, rec["strategy"], now.isoformat(), "OPEN", legs_json, entry_premium),
            )

        msg = (f"{rec['strategy']} | confidence={signal['confidence']:.2f} | "
               f"max_loss=\u20b9{rec.get('max_loss_rs', 0):.0f}")
        logger.info("PAPER TRADE OPEN: %s %s", symbol, msg)
        notify(f"New Signal: {symbol}", msg)

    def _square_off_all(self, now: datetime) -> None:
        """Close all open positions at current prices."""
        if not self._open_positions:
            return
        logger.info("Squaring off %d positions at %s.", len(self._open_positions), now.strftime("%H:%M"))
        for symbol in list(self._open_positions.keys()):
            self._close_position(symbol, now, reason="SQUARE_OFF")

    def _close_position(self, symbol: str, now: datetime, reason: str = "SIGNAL_EXIT") -> None:
        pos = self._open_positions.pop(symbol, None)
        if not pos:
            return

        inst_key = get_instrument_key(symbol, self.config_path)
        lot_size = self.cfg["instruments"][symbol]["lot_size"]
        lots     = pos.get("lots", 1)
        legs     = pos.get("legs", [])
        n_legs   = len(legs)

        exit_premium = self._fetch_legs_premium(symbol, inst_key, legs, pos["expiry"])

        raw_pnl = (exit_premium - pos["entry_premium"]) * lot_size * lots * n_legs
        charges = 20.0 * 2 * n_legs
        net_pnl = raw_pnl - charges

        logger.info("PAPER TRADE CLOSE: %s %s | exit_prem=%.2f | net_pnl=\u20b9%.0f | reason=%s",
                    symbol, pos["strategy"], exit_premium, net_pnl, reason)

        with get_conn(self.db_path) as conn:
            conn.execute(
                """UPDATE paper_trades SET
                   exit_time=?, status='CLOSED', exit_total_premium=?,
                   raw_pnl_rs=?, charges_rs=?, net_pnl_rs=?, exit_reason=?
                   WHERE symbol=? AND status='OPEN'
                   ORDER BY rowid DESC LIMIT 1""",
                (now.isoformat(), exit_premium, raw_pnl, charges, net_pnl, reason, symbol),
            )

    def _fetch_legs_premium(
        self,
        symbol: str,
        inst_key: str,
        legs: list[dict],
        expiry: str,
    ) -> float:
        """
        Fetch live LTP for each option leg from the option chain.
        Returns net premium (BUY legs positive, SELL legs negative).
        Falls back to entry LTP if chain fetch fails.
        """
        try:
            chain = self.broker.get_option_chain(inst_key, expiry)
        except Exception:
            chain = pd.DataFrame()

        net = 0.0
        for leg in legs:
            col = f"{leg['option_type']}_ltp"
            ltp = leg.get("ltp", 0.0)
            if not chain.empty:
                row = chain[chain["strike"] == leg["strike"]]
                if not row.empty and col in row.columns:
                    fetched = float(row[col].iloc[0] or 0.0)
                    if fetched > 0:
                        ltp = fetched
            net += ltp if leg["action"] == "BUY" else -ltp
        return net

    def _mark_to_market(self, now: datetime) -> None:
        """Log current unrealised P&L for all open positions."""
        for symbol, pos in self._open_positions.items():
            try:
                inst_key = get_instrument_key(symbol, self.config_path)
                lot_size = self.cfg["instruments"][symbol]["lot_size"]
                lots     = pos.get("lots", 1)
                legs     = pos.get("legs", [])
                current_premium = self._fetch_legs_premium(
                    symbol, inst_key, legs, pos["expiry"]
                )
                unrealised = (current_premium - pos["entry_premium"]) * lot_size * lots * len(legs)
                logger.info("MTM %s %s: unrealised P&L = \u20b9%.0f", symbol, pos["strategy"], unrealised)
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
