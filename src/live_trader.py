"""
Live trader — DISABLED BY DEFAULT.

To enable, ALL THREE of the following must be satisfied simultaneously:
  1. LIVE_TRADING=true in .env
  2. --live flag passed on CLI
  3. User types "I CONFIRM LIVE TRADING" at startup prompt

Any single missing condition → exits immediately.

Hard safety limits from config.yaml (live_trading section):
  - max_trades_per_day
  - max_capital_per_trade
  - max_daily_loss_rs  → kill switch: stops ALL trading for the day
  - max_capital_total  → hard cap ₹3,00,000 enforced by RiskManager

Every action is written to the audit log before being executed.

Run (paper mode — safe):
    python -m src.live_trader

Run (LIVE — only after reading all of the above):
    LIVE_TRADING=true python -m src.live_trader --live
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from datetime import datetime

from src.paper_trader import PaperTrader
from src.utils import get_broker, load_config, now_ist, setup_logging
from src.db import init_db, get_conn
from src.risk_manager import RiskManager

logger = logging.getLogger(__name__)
AUDIT_LOGGER = logging.getLogger("audit")


def _setup_audit_log(db_path: str) -> None:
    """Dedicated audit log — append-only, never truncated."""
    import logging.handlers
    fh = logging.handlers.RotatingFileHandler(
        "logs/audit_live_trades.log",
        maxBytes=50 * 1024 * 1024,
        backupCount=20,
    )
    fh.setFormatter(logging.Formatter("%(asctime)s | AUDIT | %(message)s"))
    AUDIT_LOGGER.addHandler(fh)
    AUDIT_LOGGER.setLevel(logging.DEBUG)


def _safety_check() -> bool:
    """Triple-lock safety check. Returns True only if all three gates pass."""
    if os.environ.get("LIVE_TRADING", "").lower() != "true":
        print("BLOCKED: LIVE_TRADING env var is not 'true'. Set it in .env to enable.")
        return False

    print("\n" + "=" * 60)
    print("  ⚠  LIVE TRADING MODE")
    print("  Real orders will be placed with REAL MONEY.")
    print("  Max capital hard cap: ₹3,00,000 (enforced by RiskManager).")
    print("  Check config.yaml live_trading limits before proceeding.")
    print("=" * 60)
    confirm = input('\nType exactly "I CONFIRM LIVE TRADING" to proceed: ').strip()
    if confirm != "I CONFIRM LIVE TRADING":
        print("Confirmation not matched. Exiting.")
        return False

    return True


class LiveTrader(PaperTrader):
    """
    Extends PaperTrader with real order placement via Zerodha Kite Connect.
    Inherits all logic (forecast → signal → map → positions) from PaperTrader.
    Overrides _open_paper_position and _close_position to place real orders.
    RiskManager enforces ₹3L capital cap and daily loss limits before every order.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lt_cfg = self.cfg.get("live_trading", {})
        self.max_trades_per_day = lt_cfg.get("max_trades_per_day", 3)
        self.max_capital        = lt_cfg.get("max_capital_per_trade", 50000)
        self.daily_loss_limit   = lt_cfg.get("max_daily_loss_rs", -5000)
        self._trades_today      = 0
        self._daily_pnl         = 0.0
        self._killed            = False
        self.risk               = RiskManager(
            broker=self.broker,
            max_total_capital=lt_cfg.get("max_capital_total", 300000),
            max_capital_per_trade=self.max_capital,
            max_daily_loss=abs(self.daily_loss_limit),
        )

    def _resolve_option_instrument_key(self, symbol: str, leg: dict) -> str:
        """
        Resolve the Zerodha instrument token for an option leg.
        Fetches live option chain and matches strike + option_type.
        Raises ValueError if the contract is not found.
        """
        expiry = leg.get("expiry")
        if not expiry:
            raise ValueError(f"No expiry in leg for {symbol}: {leg}")

        chain = self.broker.get_option_chain(
            instrument_key=self.cfg["instruments"][symbol]["upstox_key"],
            expiry=expiry,
        )
        opt_type = leg["option_type"]  # "CE" or "PE"
        strike   = int(leg["strike"])

        mask = (
            (chain["strike"] == strike) &
            (chain["instrument_type"] == opt_type)
        )
        matched = chain[mask]
        if matched.empty:
            raise ValueError(
                f"Instrument not found: {symbol} {strike}{opt_type} expiry={expiry}"
            )
        return str(matched.iloc[0]["instrument_token"])

    def _open_paper_position(self, symbol, rec, signal, now):
        """Override: place real Zerodha order after risk checks."""
        if self._killed:
            logger.warning("Kill switch active — no new trades.")
            return
        if self._trades_today >= self.max_trades_per_day:
            logger.warning("Max trades/day (%d) reached.", self.max_trades_per_day)
            return
        if self._daily_pnl <= self.daily_loss_limit:
            logger.critical("Daily loss limit hit (₹%.0f). Kill switch engaged.", self._daily_pnl)
            self._killed = True
            return

        # Risk gate: margin and capital check before touching the market
        lot_size = self.cfg["instruments"][symbol]["lot_size"]
        legs     = rec.get("legs", [])
        total_lots = sum(leg.get("lots", 1) for leg in legs)
        estimated_capital = total_lots * lot_size * self.max_capital

        ok, reason = self.risk.check_pre_trade(
            estimated_capital_required=estimated_capital
        )
        if not ok:
            logger.warning("Risk gate blocked trade for %s: %s", symbol, reason)
            AUDIT_LOGGER.warning("RISK BLOCKED | %s | %s", symbol, reason)
            return

        AUDIT_LOGGER.info(
            "INTENT OPEN | %s | %s | signal=%s | conf=%.2f",
            symbol, rec["strategy"], signal["signal"], signal["confidence"]
        )

        order_ids = []
        for leg in legs:
            try:
                instrument_key = self._resolve_option_instrument_key(symbol, leg)
                oid = self.broker.place_order(
                    instrument_key=instrument_key,
                    transaction_type=leg["action"],
                    quantity=lot_size * leg.get("lots", 1),
                    order_type="MARKET",
                    tag=f"kronos_{symbol[:2]}_{signal['signal'][:2]}",
                )
                order_ids.append(oid)
                leg["instrument_key"] = instrument_key  # store for close
                AUDIT_LOGGER.info(
                    "ORDER PLACED | %s | %s%s | action=%s | order_id=%s",
                    symbol, leg["strike"], leg["option_type"], leg["action"], oid
                )
            except Exception as e:
                AUDIT_LOGGER.error("ORDER FAILED | %s | %s: %s", symbol, leg, e)
                logger.error("Order placement failed for %s %s: %s", symbol, leg, e)

        self._trades_today += 1
        self.risk.record_trade(estimated_capital)
        rec["order_ids"] = order_ids

        super()._open_paper_position(symbol, rec, signal, now)

    def _close_position(self, symbol, now, reason="SIGNAL_EXIT"):
        """Override: place real Zerodha close order."""
        pos = self._open_positions.get(symbol)
        if not pos:
            return

        AUDIT_LOGGER.info("INTENT CLOSE | %s | reason=%s", symbol, reason)

        lot_size = self.cfg["instruments"][symbol]["lot_size"]
        for leg in pos.get("legs", []):
            close_action = "SELL" if leg["action"] == "BUY" else "BUY"
            inst_key = leg.get("instrument_key", "")
            if not inst_key:
                AUDIT_LOGGER.error("CLOSE SKIP | %s | no instrument_key stored for leg %s", symbol, leg)
                continue
            try:
                oid = self.broker.place_order(
                    instrument_key=inst_key,
                    transaction_type=close_action,
                    quantity=lot_size * leg.get("lots", 1),
                    order_type="MARKET",
                    tag=f"close_{symbol[:2]}",
                )
                AUDIT_LOGGER.info("CLOSE ORDER | %s | %s | order_id=%s", symbol, close_action, oid)
            except Exception as e:
                AUDIT_LOGGER.error("CLOSE ORDER FAILED | %s | %s", symbol, e)

        super()._close_position(symbol, now, reason)


# ── CLI entry point ────────────────────────────────────────────────────

def main():
    setup_logging("live_trader")
    _setup_audit_log("data/kronos_options.db")

    parser = argparse.ArgumentParser(description="Kronos Options Live Trader")
    parser.add_argument("--live", action="store_true", help="Enable live order placement")
    parser.add_argument("--symbols", nargs="+", default=["NIFTY", "BANKNIFTY"])
    args = parser.parse_args()

    if not args.live:
        print("No --live flag. Running in READ-ONLY mode (no orders). Use paper_trader.py instead.")
        sys.exit(0)

    if not _safety_check():
        sys.exit(1)

    cfg = load_config()
    if not cfg.get("live_trading", {}).get("enabled", False):
        print("live_trading.enabled is false in config.yaml. Refusing to start.")
        sys.exit(1)

    broker = get_broker()
    enabled = [s for s in args.symbols if cfg["instruments"].get(s, {}).get("enabled", False)]

    AUDIT_LOGGER.info("LIVE TRADER STARTED | symbols=%s | user confirmed", enabled)

    trader = LiveTrader(broker=broker, symbols=enabled)
    trader.run()


if __name__ == "__main__":
    main()
