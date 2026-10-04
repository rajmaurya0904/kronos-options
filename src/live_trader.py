"""
Live trader — DISABLED BY DEFAULT.

To enable, ALL FOUR of the following must be satisfied simultaneously:
  1. LIVE_TRADING=true in .env
  2. --live flag passed on CLI
  3. User types "I CONFIRM LIVE TRADING" at startup prompt
  4. live_trading.enabled: true in config.yaml

Any single missing condition → exits immediately.

Hard safety limits from config.yaml (live_trading section):
  - max_trades_per_day
  - max_capital_per_trade   (debit paid, or max loss for credit structures)
  - max_daily_loss_rs       → kill switch: stops ALL trading for the day

Order safety:
  - Every leg is resolved to its tradable option instrument key before any
    order is sent; if one can't be resolved, nothing is sent.
  - Hedge (BUY) legs go first, short (SELL) legs after; on exit, shorts are
    bought back first. A naked short never exists, even for a moment.
  - Each order must reach status "complete". If any leg fails, the legs
    already filled are flattened and the kill switch is engaged.

Every action is written to the audit log before being executed.

Run (LIVE — only after reading all of the above):
    python -m src.live_trader --live
"""
from __future__ import annotations
import argparse
import logging
import logging.handlers
import os
import sys
import time

from src.paper_trader import PaperTrader
from src.utils import get_broker, load_config, setup_logging

logger = logging.getLogger(__name__)
AUDIT_LOGGER = logging.getLogger("audit")

ORDER_CONFIRM_TIMEOUT_S = 10


def _setup_audit_log() -> None:
    """Dedicated audit log — append-only, rotated but never truncated."""
    os.makedirs("logs", exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        "logs/audit_live_trades.log",
        maxBytes=50 * 1024 * 1024,  # 50 MB
        backupCount=20,
    )
    fh.setFormatter(logging.Formatter("%(asctime)s | AUDIT | %(message)s"))
    AUDIT_LOGGER.addHandler(fh)
    AUDIT_LOGGER.setLevel(logging.DEBUG)


def _safety_check() -> bool:
    """Env + typed-confirmation gates. Returns True only if both pass."""
    # Gate 1: Environment variable (.env is loaded in main() before this)
    if os.environ.get("LIVE_TRADING", "").lower() != "true":
        print("BLOCKED: LIVE_TRADING env var is not 'true'. Set it in .env to enable.")
        return False

    # Gate 2: CLI flag (checked in main())

    # Gate 3: Typed confirmation
    print("\n" + "=" * 60)
    print("  ⚠  LIVE TRADING MODE")
    print("  Real orders will be placed with REAL MONEY.")
    print("  Check config.yaml live_trading limits before proceeding.")
    print("=" * 60)
    confirm = input('\nType exactly "I CONFIRM LIVE TRADING" to proceed: ').strip()
    if confirm != "I CONFIRM LIVE TRADING":
        print("Confirmation not matched. Exiting.")
        return False

    return True


class LiveTrader(PaperTrader):
    """
    Extends PaperTrader with real order placement.
    Inherits all logic (forecast → signal → map → positions) from PaperTrader.
    Overrides _open_paper_position and _close_position to place real orders.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lt_cfg = self.cfg.get("live_trading", {})
        self.max_trades_per_day = lt_cfg.get("max_trades_per_day", 3)
        self.max_capital        = lt_cfg.get("max_capital_per_trade", 50000)
        self.daily_loss_limit   = lt_cfg.get("max_daily_loss_rs", -5000)
        self._day               = None
        self._trades_today      = 0
        self._daily_pnl         = 0.0
        self._killed            = False

    # ── Day bookkeeping ────────────────────────────────────────────────

    def _tick(self, now):
        if self._day != now.date():           # counters are per trading day
            self._day = now.date()
            self._trades_today = 0
            self._daily_pnl = 0.0
            self._killed = False
        super()._tick(now)

    def _kill(self, why: str) -> None:
        self._killed = True
        logger.critical("KILL SWITCH: %s — no new trades today.", why)
        AUDIT_LOGGER.critical("KILL SWITCH | %s", why)

    # ── Orders ─────────────────────────────────────────────────────────

    def _resolve_keys(self, symbol: str, rec: dict) -> bool:
        """Fill leg['instrument_key'] for every leg. False if any is unknown."""
        inst_key = self.cfg["instruments"][symbol]["upstox_key"]
        for leg in rec["legs"]:
            if leg.get("instrument_key"):
                continue
            c = self.broker.get_option_contract(inst_key, rec["expiry"], leg["strike"], leg["option_type"])
            leg["instrument_key"] = (c or {}).get("instrument_key")
            if not leg["instrument_key"]:
                AUDIT_LOGGER.error("UNRESOLVED | %s %s%s %s", symbol, leg["strike"],
                                   leg["option_type"], rec["expiry"])
                return False
        return True

    def _execute(self, leg: dict, action: str, qty: int, tag: str) -> str | None:
        """Place one MARKET order and wait for it to complete. Returns order_id or None."""
        AUDIT_LOGGER.info("ORDER SEND | %s | %s%s | %s x%d", leg["instrument_key"],
                          leg["strike"], leg["option_type"], action, qty)
        try:
            oid = self.broker.place_order(
                instrument_key=leg["instrument_key"],
                transaction_type=action,
                quantity=qty,
                order_type="MARKET",
                tag=tag,
            )
        except Exception as e:
            AUDIT_LOGGER.error("ORDER FAILED | %s | %s", leg["instrument_key"], e)
            return None

        deadline = time.time() + ORDER_CONFIRM_TIMEOUT_S
        status = ""
        while time.time() < deadline:
            status = str(self.broker.get_order_status(oid).get("status", "")).lower()
            if status in ("complete", "rejected", "cancelled"):
                break
            time.sleep(0.5)
        AUDIT_LOGGER.info("ORDER STATUS | %s | %s", oid, status or "unknown")
        return oid if status == "complete" else None

    def _qty(self, symbol: str, leg: dict) -> int:
        return self.cfg["instruments"][symbol]["lot_size"] * leg["lots"]

    def _open_paper_position(self, symbol, rec, signal, now):
        """Override: place real orders, then record the position."""
        if self._killed:
            logger.warning("Kill switch active — no new trades.")
            return
        if self._trades_today >= self.max_trades_per_day:
            logger.warning("Max trades/day (%d) reached.", self.max_trades_per_day)
            return
        if self._daily_pnl <= self.daily_loss_limit:
            self._kill(f"daily loss limit hit (₹{self._daily_pnl:.0f})")
            return

        lot_size = self.cfg["instruments"][symbol]["lot_size"]
        debit = sum(l["ltp"] * lot_size * l["lots"] for l in rec["legs"] if l["action"] == "BUY")
        max_loss = abs(rec.get("max_loss_rs") or 0)
        exposure = max(debit, max_loss if max_loss != float("inf") else debit)
        if exposure > self.max_capital:
            logger.warning("%s %s needs ~₹%.0f > max_capital_per_trade ₹%.0f — skipped.",
                           symbol, rec["strategy"], exposure, self.max_capital)
            return

        if not self._resolve_keys(symbol, rec):
            logger.error("%s: could not resolve every option contract — no orders sent.", symbol)
            return

        AUDIT_LOGGER.info("INTENT OPEN | %s | %s | signal=%s | conf=%.2f",
                          symbol, rec["strategy"], signal["signal"], signal["confidence"])

        # Hedges first, shorts last.
        ordered = sorted(rec["legs"], key=lambda l: l["action"] != "BUY")
        filled = []
        tag = f"kronos_{symbol[:2]}"
        for leg in ordered:
            oid = self._execute(leg, leg["action"], self._qty(symbol, leg), tag)
            if oid is None:
                self._kill(f"{symbol} entry leg {leg['strike']}{leg['option_type']} not filled")
                self._flatten(symbol, filled)
                return
            leg["order_id"] = oid
            filled.append(leg)

        self._trades_today += 1
        super()._open_paper_position(symbol, rec, signal, now)

    def _flatten(self, symbol: str, legs: list[dict]) -> bool:
        """Close the given filled legs, shorts first. True if all closed."""
        ok = True
        for leg in sorted(legs, key=lambda l: l["action"] == "BUY"):
            close_action = "SELL" if leg["action"] == "BUY" else "BUY"
            if self._execute(leg, close_action, self._qty(symbol, leg), f"close_{symbol[:2]}") is None:
                ok = False
                logger.critical("COULD NOT CLOSE %s %s%s — CLOSE IT MANUALLY IN THE BROKER APP",
                                symbol, leg["strike"], leg["option_type"])
        return ok

    def _close_position(self, symbol, now, reason="SIGNAL_EXIT"):
        """Override: place real close orders, then record the result."""
        pos = self._open_positions.get(symbol)
        if not pos:
            return None

        AUDIT_LOGGER.info("INTENT CLOSE | %s | reason=%s", symbol, reason)
        if not self._flatten(symbol, pos.get("legs", [])):
            self._kill(f"{symbol} exit not fully filled")

        res = super()._close_position(symbol, now, reason)
        if res:
            self._daily_pnl += res["net_pnl_rs"]
            AUDIT_LOGGER.info("DAY P&L | ₹%.0f", self._daily_pnl)
            if self._daily_pnl <= self.daily_loss_limit:
                self._kill(f"daily loss limit hit (₹{self._daily_pnl:.0f})")
        return res


# ── CLI entry point ────────────────────────────────────────────────────

def main():
    from dotenv import load_dotenv
    load_dotenv()  # gate 1 reads LIVE_TRADING, which lives in .env

    setup_logging("live_trader")
    _setup_audit_log()

    parser = argparse.ArgumentParser(description="Kronos Options Live Trader")
    parser.add_argument("--live", action="store_true", help="Enable live order placement")
    parser.add_argument("--symbols", nargs="+", default=["NIFTY", "BANKNIFTY"])
    args = parser.parse_args()

    if not args.live:
        print("No --live flag. Nothing to do — use `python -m src.paper_trader` for paper trading.")
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
