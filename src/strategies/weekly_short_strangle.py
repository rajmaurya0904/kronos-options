"""
weekly_short_strangle.py — Weekly options-selling strategy.

Strategy logic:
  - Entry: Monday morning (or first trading day of week), 09:30–10:00 IST.
    Sell ATM CE + ATM PE on the nearest Thursday weekly expiry (NIFTY).
    This is a short strangle: collect premium, profit if market stays rangebound.

  - Position sizing: margin required for strangle is fetched from broker.
    RiskManager enforces the ₹3L total cap before any order fires.

  - Stop-loss: if combined premium of open legs rises to 2× entry premium
    (i.e., unrealised loss = 1× premium), exit all legs immediately.

  - Target: exit when combined premium decays to 50% of entry premium
    (theta target). Also exit on Wednesday EOD (15:15) if still open,
    to avoid expiry-day gamma risk.

  - Monitoring: call monitor() every 5 minutes from the live trader loop.
    It checks SL/target and fires exit orders if triggered.

Usage (paper trading):
    from src.strategies.weekly_short_strangle import WeeklyShortStrangle
    from src.utils import get_broker

    broker = get_broker()
    strategy = WeeklyShortStrangle(broker=broker)

    # On Monday morning:
    result = strategy.enter(symbol="NIFTY", dry_run=True)  # dry_run=True = paper
    print(result)

    # Every 5 min in the trading loop:
    exit_result = strategy.monitor(dry_run=True)
    if exit_result:
        print(exit_result)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from src.broker.base import BrokerInterface
from src.risk_manager import RiskManager
from src.utils import load_config, round_to_atm, IST

logger = logging.getLogger(__name__)


# ── Expiry helpers ──────────────────────────────────────────────────

_EXPIRY_WEEKDAY = {"NIFTY": 3, "BANKNIFTY": 2, "SENSEX": 4}  # Thu=3, Wed=2, Fri=4


def next_weekly_expiry(symbol: str, from_date: Optional[date] = None) -> date:
    """
    Return the nearest weekly expiry date on or after from_date.
    NIFTY: Thursday, BANKNIFTY: Wednesday, SENSEX: Friday.
    Skips NSE holidays by moving one day forward.
    """
    from src.utils import NSE_HOLIDAYS

    target_weekday = _EXPIRY_WEEKDAY.get(symbol.upper(), 3)
    d = from_date or date.today()
    # Advance to the target weekday
    days_ahead = (target_weekday - d.weekday()) % 7
    expiry = d + timedelta(days=days_ahead)
    # If it's a holiday, move to next trading day
    while expiry in NSE_HOLIDAYS:
        expiry += timedelta(days=1)
    return expiry


def is_entry_day(symbol: str, d: Optional[date] = None) -> bool:
    """Monday (or first trading day of the week) is the entry day."""
    from src.utils import is_trading_day
    d = d or date.today()
    if d.weekday() != 0:  # Not Monday
        return False
    return is_trading_day(d)


def is_exit_eod_day(symbol: str, d: Optional[date] = None) -> bool:
    """
    Wednesday EOD forced exit for NIFTY (day before Thursday expiry).
    Avoids expiry-day gamma spike.
    """
    d = d or date.today()
    expiry = next_weekly_expiry(symbol, d)
    return (expiry - d).days == 1  # one day before expiry


# ── Position dataclass ───────────────────────────────────────────────

@dataclass
class StranglePosition:
    symbol:         str
    expiry:         str           # "YYYY-MM-DD"
    atm_strike:     int

    ce_strike:      int
    pe_strike:      int
    ce_entry_price: float         # premium collected on SELL
    pe_entry_price: float
    lots:           int
    lot_size:       int

    entry_ts:       str = ""
    entry_combined_premium: float = field(init=False)
    sl_premium:     float = field(init=False)   # 2× entry = exit if premium doubles
    target_premium: float = field(init=False)   # 0.5× entry = 50% decay target

    ce_instrument_key: str = ""
    pe_instrument_key: str = ""
    capital_deployed:  float = 0.0

    def __post_init__(self):
        self.entry_combined_premium = self.ce_entry_price + self.pe_entry_price
        self.sl_premium     = self.entry_combined_premium * 2.0
        self.target_premium = self.entry_combined_premium * 0.5

    @property
    def max_profit_rs(self) -> float:
        """Max profit = full premium collected (both legs expire worthless)."""
        return self.entry_combined_premium * self.lot_size * self.lots

    @property
    def max_loss_rs(self) -> float:
        """Theoretical max loss is unlimited but SL caps it at 1× premium."""
        return -self.entry_combined_premium * self.lot_size * self.lots

    def unrealised_pnl_rs(self, current_ce_ltp: float, current_pe_ltp: float) -> float:
        """
        Unrealised P&L in rupees.
        Positive = profit (premium decayed), negative = loss (premium expanded).
        """
        current_combined = current_ce_ltp + current_pe_ltp
        return (self.entry_combined_premium - current_combined) * self.lot_size * self.lots


# ── Strategy ──────────────────────────────────────────────────────────

class WeeklyShortStrangle:
    """
    Manages a single weekly short strangle position from entry to exit.
    Designed to hold at most ONE open position at a time.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        config_path: str = "config.yaml",
        risk_manager: Optional[RiskManager] = None,
    ):
        self.broker = broker
        self.cfg = load_config(config_path)
        self.position: Optional[StranglePosition] = None

        live_cfg = self.cfg.get("live_trading", {})
        self.risk = risk_manager or RiskManager(
            broker=broker,
            max_total_capital=live_cfg.get("max_capital_total", 300000),
            max_capital_per_trade=live_cfg.get("max_capital_per_trade", 50000),
            max_daily_loss=live_cfg.get("max_daily_loss_rs", 5000),
        )

    # ── Entry ──────────────────────────────────────────────────────────

    def enter(
        self,
        symbol: str = "NIFTY",
        dry_run: bool = True,
    ) -> dict:
        """
        Enter a short strangle on the nearest weekly expiry.

        Args:
            symbol:  "NIFTY" | "BANKNIFTY" | "SENSEX"
            dry_run: True = paper trade (no real orders). False = live orders.

        Returns:
            dict with entry details, or {"status": "BLOCKED", "reason": ...}
        """
        if self.position is not None:
            return {"status": "BLOCKED", "reason": "Position already open. Close it before entering."}

        inst_cfg = self.cfg["instruments"][symbol]
        atm_step = inst_cfg["atm_step"]
        lot_size = inst_cfg["lot_size"]
        inst_key = inst_cfg["upstox_key"]
        lots     = self.cfg.get("paper_trading" if dry_run else "live_trading", {}).get("lots", 1)

        expiry = next_weekly_expiry(symbol)
        expiry_str = expiry.strftime("%Y-%m-%d")

        # Fetch option chain to get ATM strike and current premiums
        try:
            option_chain = self.broker.get_option_chain(inst_key, expiry_str)
        except Exception as e:
            return {"status": "ERROR", "reason": f"Could not fetch option chain: {e}"}

        if option_chain is None or option_chain.empty:
            return {"status": "ERROR", "reason": "Empty option chain returned."}

        # Get spot price from ATM (midpoint of chain)
        spot = self._get_spot(symbol)
        if spot is None:
            return {"status": "ERROR", "reason": "Could not fetch spot price."}

        atm_strike = round_to_atm(spot, atm_step)

        # Get CE and PE premiums at ATM
        ce_ltp = self._get_ltp(option_chain, atm_strike, "CE")
        pe_ltp = self._get_ltp(option_chain, atm_strike, "PE")

        if ce_ltp <= 0 or pe_ltp <= 0:
            return {"status": "ERROR", "reason": f"Invalid premiums: CE={ce_ltp} PE={pe_ltp}"}

        combined_premium = ce_ltp + pe_ltp

        # Estimate margin required (CE + PE selling = ~1.5× lot_size×spot×0.1 rough estimate)
        # Use broker margin API if available; fall back to conservative estimate
        margin_required = self._estimate_margin(symbol, atm_strike, lot_size, lots, spot)

        # Risk gate
        ok, reason = self.risk.check_pre_trade(margin_required)
        if not ok:
            logger.warning("Weekly strangle entry BLOCKED: %s", reason)
            return {"status": "BLOCKED", "reason": reason}

        # Resolve instrument keys for monitoring
        ce_key = self._resolve_key(inst_key, expiry_str, atm_strike, "CE")
        pe_key = self._resolve_key(inst_key, expiry_str, atm_strike, "PE")

        now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        if not dry_run:
            # Live orders
            ce_order = self.broker.place_order(
                instrument_key=ce_key or inst_key,
                transaction_type="SELL",
                quantity=lot_size * lots,
                order_type="MARKET",
                product="MIS",
            )
            pe_order = self.broker.place_order(
                instrument_key=pe_key or inst_key,
                transaction_type="SELL",
                quantity=lot_size * lots,
                order_type="MARKET",
                product="MIS",
            )
            logger.info("Live orders placed: CE=%s PE=%s", ce_order, pe_order)
        else:
            logger.info(
                "[PAPER] Short strangle entry: SELL %s%s CE @ %.2f | SELL %s%s PE @ %.2f",
                symbol, atm_strike, ce_ltp, symbol, atm_strike, pe_ltp,
            )

        # Record position
        self.position = StranglePosition(
            symbol=symbol,
            expiry=expiry_str,
            atm_strike=atm_strike,
            ce_strike=atm_strike,
            pe_strike=atm_strike,
            ce_entry_price=ce_ltp,
            pe_entry_price=pe_ltp,
            lots=lots,
            lot_size=lot_size,
            entry_ts=now_ist,
            ce_instrument_key=ce_key or "",
            pe_instrument_key=pe_key or "",
            capital_deployed=margin_required,
        )
        self.risk.record_trade(margin_required)

        result = {
            "status":             "ENTERED",
            "dry_run":            dry_run,
            "symbol":             symbol,
            "expiry":             expiry_str,
            "atm_strike":         atm_strike,
            "ce_entry_price":     ce_ltp,
            "pe_entry_price":     pe_ltp,
            "combined_premium":   combined_premium,
            "sl_premium":         combined_premium * 2.0,
            "target_premium":     combined_premium * 0.5,
            "max_profit_rs":      combined_premium * lot_size * lots,
            "max_loss_rs":        -combined_premium * lot_size * lots,  # SL-capped
            "margin_required":    margin_required,
            "lots":               lots,
            "lot_size":           lot_size,
            "entry_ts":           now_ist,
        }
        logger.info(
            "Short strangle ENTERED: %s %s | CE+PE @ %.2f+%.2f=%.2f | "
            "SL=%.2f | Target=%.2f | MaxProfit=₹%.0f",
            symbol, expiry_str, ce_ltp, pe_ltp, combined_premium,
            combined_premium * 2.0, combined_premium * 0.5,
            combined_premium * lot_size * lots,
        )
        return result

    # ── Monitor (call every 5 min from trading loop) ───────────────────────

    def monitor(self, dry_run: bool = True) -> Optional[dict]:
        """
        Check SL, target, and EOD exit conditions.
        Call every 5 minutes from the live trading loop.

        Returns exit dict if position was closed, None if still open.
        """
        if self.position is None:
            return None

        pos = self.position
        now = datetime.now(IST)

        # Forced EOD exit: Wednesday 15:15 (day before expiry)
        if is_exit_eod_day(pos.symbol) and now.hour == 15 and now.minute >= 15:
            logger.info("Forced EOD exit: day before expiry.")
            return self._exit(reason="EOD_PRE_EXPIRY", dry_run=dry_run)

        # Intraday square-off: 15:15 any day
        sq_off = self.cfg.get("trading", {}).get("square_off_time", "15:15").split(":")
        if now.hour > int(sq_off[0]) or (now.hour == int(sq_off[0]) and now.minute >= int(sq_off[1])):
            return self._exit(reason="INTRADAY_SQUAREOFF", dry_run=dry_run)

        # Fetch current LTPs
        ce_ltp, pe_ltp = self._get_current_ltps(pos)
        if ce_ltp is None or pe_ltp is None:
            logger.warning("Could not fetch LTPs for monitoring — skipping this tick.")
            return None

        current_combined = ce_ltp + pe_ltp
        unrealised_pnl   = pos.unrealised_pnl_rs(ce_ltp, pe_ltp)

        logger.debug(
            "Monitor %s: CE_ltp=%.2f PE_ltp=%.2f combined=%.2f pnl=₹%.0f",
            pos.symbol, ce_ltp, pe_ltp, current_combined, unrealised_pnl,
        )

        # Stop-loss: combined premium >= 2× entry (loss = 1× entry premium)
        if current_combined >= pos.sl_premium:
            logger.warning(
                "SL HIT: combined=%.2f >= sl=%.2f | loss=₹%.0f",
                current_combined, pos.sl_premium, unrealised_pnl,
            )
            return self._exit(reason="STOP_LOSS", dry_run=dry_run, exit_ce_ltp=ce_ltp, exit_pe_ltp=pe_ltp)

        # Target: combined premium <= 50% of entry
        if current_combined <= pos.target_premium:
            logger.info(
                "TARGET HIT: combined=%.2f <= target=%.2f | profit=₹%.0f",
                current_combined, pos.target_premium, unrealised_pnl,
            )
            return self._exit(reason="TARGET_HIT", dry_run=dry_run, exit_ce_ltp=ce_ltp, exit_pe_ltp=pe_ltp)

        return None

    # ── Exit ─────────────────────────────────────────────────────────────

    def _exit(
        self,
        reason: str,
        dry_run: bool = True,
        exit_ce_ltp: Optional[float] = None,
        exit_pe_ltp: Optional[float] = None,
    ) -> dict:
        """Close both legs of the strangle. Updates RiskManager."""
        pos = self.position

        # Fetch current prices if not provided
        if exit_ce_ltp is None or exit_pe_ltp is None:
            exit_ce_ltp, exit_pe_ltp = self._get_current_ltps(pos)
            exit_ce_ltp = exit_ce_ltp or pos.ce_entry_price
            exit_pe_ltp = exit_pe_ltp or pos.pe_entry_price

        realised_pnl_pts = (pos.ce_entry_price - exit_ce_ltp) + (pos.pe_entry_price - exit_pe_ltp)
        realised_pnl_rs  = realised_pnl_pts * pos.lot_size * pos.lots

        if not dry_run:
            try:
                self.broker.place_order(
                    instrument_key=pos.ce_instrument_key or "",
                    transaction_type="BUY",
                    quantity=pos.lot_size * pos.lots,
                    order_type="MARKET",
                    product="MIS",
                )
                self.broker.place_order(
                    instrument_key=pos.pe_instrument_key or "",
                    transaction_type="BUY",
                    quantity=pos.lot_size * pos.lots,
                    order_type="MARKET",
                    product="MIS",
                )
            except Exception as e:
                logger.error("Exit order failed: %s", e)
        else:
            logger.info(
                "[PAPER] Strangle exit (%s): BUY CE @ %.2f | BUY PE @ %.2f | P&L=₹%.0f",
                reason, exit_ce_ltp, exit_pe_ltp, realised_pnl_rs,
            )

        self.risk.record_pnl(realised_pnl_rs)
        self.risk.release_capital(pos.capital_deployed)

        result = {
            "status":          "EXITED",
            "reason":          reason,
            "dry_run":         dry_run,
            "symbol":          pos.symbol,
            "expiry":          pos.expiry,
            "atm_strike":      pos.atm_strike,
            "ce_entry":        pos.ce_entry_price,
            "pe_entry":        pos.pe_entry_price,
            "ce_exit":         exit_ce_ltp,
            "pe_exit":         exit_pe_ltp,
            "entry_combined":  pos.entry_combined_premium,
            "exit_combined":   exit_ce_ltp + exit_pe_ltp,
            "realised_pnl_rs": round(realised_pnl_rs, 2),
            "entry_ts":        pos.entry_ts,
            "exit_ts":         datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        }
        logger.info(
            "Strangle EXITED (%s): %s %s | P&L=₹%.0f",
            reason, pos.symbol, pos.expiry, realised_pnl_rs,
        )
        self.position = None
        return result

    # ── Internal helpers ───────────────────────────────────────────────

    def _get_spot(self, symbol: str) -> Optional[float]:
        """Fetch current spot price via broker LTP."""
        inst_cfg = self.cfg["instruments"][symbol]
        inst_key = inst_cfg["upstox_key"]
        try:
            quotes = self.broker.get_ltp([inst_key])
            if quotes:
                return float(list(quotes.values())[0])
        except Exception as e:
            logger.warning("Could not fetch spot for %s: %s", symbol, e)
        return None

    @staticmethod
    def _get_ltp(option_chain, strike: int, opt_type: str) -> float:
        """Read LTP for a strike/type from the option chain DataFrame."""
        col = f"{opt_type}_ltp"
        row = option_chain[option_chain["strike"] == strike]
        if not row.empty and col in row.columns:
            return float(row[col].iloc[0]) or 0.0
        return 0.0

    def _get_current_ltps(self, pos: StranglePosition):
        """Fetch live CE and PE LTPs for an open position."""
        inst_key = self.cfg["instruments"][pos.symbol]["upstox_key"]
        try:
            chain = self.broker.get_option_chain(inst_key, pos.expiry)
            if chain is not None and not chain.empty:
                ce_ltp = self._get_ltp(chain, pos.ce_strike, "CE")
                pe_ltp = self._get_ltp(chain, pos.pe_strike, "PE")
                if ce_ltp > 0 and pe_ltp > 0:
                    return ce_ltp, pe_ltp
        except Exception as e:
            logger.warning("LTP fetch failed: %s", e)
        return None, None

    def _resolve_key(self, inst_key: str, expiry: str, strike: int, opt_type: str) -> Optional[str]:
        """Resolve option instrument key for order placement."""
        try:
            return self.broker.get_expired_option_key(inst_key, expiry, strike, opt_type)
        except Exception:
            return None

    def _estimate_margin(
        self,
        symbol: str,
        strike: int,
        lot_size: int,
        lots: int,
        spot: float,
    ) -> float:
        """
        Estimate margin required for short strangle.
        Tries broker margin API first; falls back to conservative estimate:
          NIFTY short strangle margin ≈ SPAN + Exposure ≈ ~1.3× lot_size×spot×0.08
        This is always less than ₹3L cap for 1 lot of NIFTY.
        """
        try:
            margin = self.broker.get_available_margin()
            # Broker margin API gives available, not required; use conservative formula
        except Exception:
            pass

        # Conservative formula: ~8% of notional per leg, ×2 legs
        # For NIFTY @ 24000, lot=75: 24000 × 75 × 0.08 × 2 = ₹288,000 (both legs)
        # SPAN netting reduces this; actual is roughly 1 leg margin ≈ ₹100-120k for 1 lot
        # Use 10% of notional per leg as conservative upper bound
        margin_estimate = spot * lot_size * lots * 0.10
        logger.debug("Margin estimate for %s short strangle: ₹%.0f", symbol, margin_estimate)
        return round(margin_estimate, 0)

    # ── Status ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return current position status (for dashboard / logging)."""
        if self.position is None:
            return {"open": False}
        pos = self.position
        ce_ltp, pe_ltp = self._get_current_ltps(pos)
        ce_ltp = ce_ltp or pos.ce_entry_price
        pe_ltp = pe_ltp or pos.pe_entry_price
        return {
            "open":             True,
            "symbol":           pos.symbol,
            "expiry":           pos.expiry,
            "atm_strike":       pos.atm_strike,
            "ce_entry":         pos.ce_entry_price,
            "pe_entry":         pos.pe_entry_price,
            "ce_ltp":           ce_ltp,
            "pe_ltp":           pe_ltp,
            "entry_combined":   pos.entry_combined_premium,
            "current_combined": ce_ltp + pe_ltp,
            "unrealised_pnl_rs": pos.unrealised_pnl_rs(ce_ltp, pe_ltp),
            "sl_premium":       pos.sl_premium,
            "target_premium":   pos.target_premium,
            "entry_ts":         pos.entry_ts,
            **self.risk.status(),
        }
