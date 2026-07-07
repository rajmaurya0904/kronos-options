"""
default_short_strangle.py — Weekly ATM short strangle (symmetric, no forecast).

Strategy logic:
  - Entry: Monday morning (or first trading day of week), 09:30–10:00 IST.
    Sell ATM CE + ATM PE on the nearest weekly expiry.
    Symmetric strangle: both strikes = ATM, no directional bias.

  - Stop-loss : combined premium rises to 2× entry → exit (loss capped at 1× premium).
  - Target    : combined premium decays to 50% of entry → exit.
  - Forced exit: day before expiry at 15:15, or any day 15:15 square-off.

  - RiskManager enforces ₹3L hard capital cap before every entry.

Usage:
    from src.strategies.default_short_strangle import DefaultShortStrangle

    strat = DefaultShortStrangle(broker=broker)

    # Monday entry
    result = strat.enter(symbol="NIFTY", dry_run=True)

    # Every 5 min in trading loop
    exit_result = strat.monitor(dry_run=True)
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


# ── Expiry helpers ────────────────────────────────────────────────────

_EXPIRY_WEEKDAY = {"NIFTY": 3, "BANKNIFTY": 2, "SENSEX": 4}  # Thu=3, Wed=2, Fri=4


def next_weekly_expiry(symbol: str, from_date: Optional[date] = None) -> date:
    """
    Nearest weekly expiry on or after from_date.
    NIFTY: Thursday, BANKNIFTY: Wednesday, SENSEX: Friday.
    Skips NSE holidays.
    """
    from src.utils import NSE_HOLIDAYS
    target_weekday = _EXPIRY_WEEKDAY.get(symbol.upper(), 3)
    d = from_date or date.today()
    days_ahead = (target_weekday - d.weekday()) % 7
    expiry = d + timedelta(days=days_ahead)
    while expiry in NSE_HOLIDAYS:
        expiry += timedelta(days=1)
    return expiry


def is_entry_day(symbol: str = "NIFTY", d: Optional[date] = None) -> bool:
    """Monday (or first trading day of the week) is the entry day."""
    from src.utils import is_trading_day
    d = d or date.today()
    return d.weekday() == 0 and is_trading_day(d)


def is_exit_eod_day(symbol: str, d: Optional[date] = None) -> bool:
    """Day before expiry = forced EOD exit to avoid expiry gamma."""
    d = d or date.today()
    expiry = next_weekly_expiry(symbol, d)
    return (expiry - d).days == 1


# ── Position dataclass ────────────────────────────────────────────────

@dataclass
class StranglePosition:
    symbol:         str
    expiry:         str
    atm_strike:     int
    ce_strike:      int
    pe_strike:      int
    ce_entry_price: float
    pe_entry_price: float
    lots:           int
    lot_size:       int
    entry_ts:       str = ""
    ce_instrument_key: str = ""
    pe_instrument_key: str = ""
    capital_deployed:  float = 0.0
    strategy_name:     str = "DefaultShortStrangle"

    entry_combined_premium: float = field(init=False)
    sl_premium:             float = field(init=False)
    target_premium:         float = field(init=False)

    def __post_init__(self):
        self.entry_combined_premium = self.ce_entry_price + self.pe_entry_price
        self.sl_premium             = self.entry_combined_premium * 2.0
        self.target_premium         = self.entry_combined_premium * 0.5

    @property
    def max_profit_rs(self) -> float:
        return self.entry_combined_premium * self.lot_size * self.lots

    @property
    def max_loss_rs(self) -> float:
        return -self.entry_combined_premium * self.lot_size * self.lots

    def unrealised_pnl_rs(self, current_ce_ltp: float, current_pe_ltp: float) -> float:
        current_combined = current_ce_ltp + current_pe_ltp
        return (self.entry_combined_premium - current_combined) * self.lot_size * self.lots


# ── Strategy ──────────────────────────────────────────────────────────

class DefaultShortStrangle:
    """
    Weekly ATM short strangle — symmetric, no directional forecast.
    Holds at most ONE open position at a time.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        config_path: str = "config.yaml",
        risk_manager: Optional[RiskManager] = None,
    ):
        self.broker = broker
        self.cfg    = load_config(config_path)
        self.position: Optional[StranglePosition] = None

        live_cfg = self.cfg.get("live_trading", {})
        self.risk = risk_manager or RiskManager(
            broker=broker,
            max_total_capital    = live_cfg.get("max_capital_total",   300000),
            max_capital_per_trade= live_cfg.get("max_capital_per_trade", 50000),
            max_daily_loss       = live_cfg.get("max_daily_loss_rs",     5000),
        )

    # ── Entry ─────────────────────────────────────────────────────────

    def enter(self, symbol: str = "NIFTY", dry_run: bool = True) -> dict:
        """
        Enter symmetric ATM short strangle on nearest weekly expiry.
        Returns entry details dict or {"status": "BLOCKED"/"ERROR", "reason": …}.
        """
        if self.position is not None:
            return {"status": "BLOCKED", "reason": "Position already open."}

        inst_cfg = self.cfg["instruments"][symbol]
        atm_step = inst_cfg["atm_step"]
        lot_size = inst_cfg["lot_size"]
        inst_key = inst_cfg["upstox_key"]
        lots     = self.cfg.get("paper_trading" if dry_run else "live_trading", {}).get("lots", 1)

        expiry     = next_weekly_expiry(symbol)
        expiry_str = expiry.strftime("%Y-%m-%d")

        spot = self._get_spot(symbol)
        if spot is None:
            return {"status": "ERROR", "reason": "Could not fetch spot price."}

        atm_strike = round_to_atm(spot, atm_step)

        option_chain = self._fetch_chain(inst_key, expiry_str)
        if option_chain is None:
            return {"status": "ERROR", "reason": "Empty option chain."}

        ce_ltp = self._get_chain_ltp(option_chain, atm_strike, "CE")
        pe_ltp = self._get_chain_ltp(option_chain, atm_strike, "PE")
        if ce_ltp <= 0 or pe_ltp <= 0:
            return {"status": "ERROR", "reason": f"Invalid premiums CE={ce_ltp} PE={pe_ltp}"}

        margin = self._estimate_margin(spot, lot_size, lots)
        ok, reason = self.risk.check_pre_trade(margin)
        if not ok:
            return {"status": "BLOCKED", "reason": reason}

        ce_key = self._resolve_key(inst_key, expiry_str, atm_strike, "CE")
        pe_key = self._resolve_key(inst_key, expiry_str, atm_strike, "PE")
        now    = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        if not dry_run:
            self._place_sell(ce_key, lot_size * lots)
            self._place_sell(pe_key, lot_size * lots)
        else:
            logger.info("[PAPER] DefaultShortStrangle ENTER: SELL %s%sCE@%.2f SELL %s%sPE@%.2f",
                        symbol, atm_strike, ce_ltp, symbol, atm_strike, pe_ltp)

        combined = ce_ltp + pe_ltp
        self.position = StranglePosition(
            symbol=symbol, expiry=expiry_str, atm_strike=atm_strike,
            ce_strike=atm_strike, pe_strike=atm_strike,
            ce_entry_price=ce_ltp, pe_entry_price=pe_ltp,
            lots=lots, lot_size=lot_size, entry_ts=now,
            ce_instrument_key=ce_key or "", pe_instrument_key=pe_key or "",
            capital_deployed=margin,
        )
        self.risk.record_trade(margin)

        return {
            "status": "ENTERED", "dry_run": dry_run,
            "strategy": "DefaultShortStrangle",
            "symbol": symbol, "expiry": expiry_str,
            "ce_strike": atm_strike, "pe_strike": atm_strike,
            "ce_entry": ce_ltp, "pe_entry": pe_ltp,
            "combined_premium": combined,
            "sl_premium": combined * 2.0, "target_premium": combined * 0.5,
            "max_profit_rs": combined * lot_size * lots,
            "max_loss_rs":   -combined * lot_size * lots,
            "margin_required": margin, "lots": lots, "lot_size": lot_size,
            "entry_ts": now,
        }

    # ── Monitor ───────────────────────────────────────────────────────

    def monitor(self, dry_run: bool = True) -> Optional[dict]:
        """Check SL/target/EOD. Call every 5 min. Returns exit dict or None."""
        if self.position is None:
            return None
        pos = self.position
        now = datetime.now(IST)

        if is_exit_eod_day(pos.symbol) and now.hour == 15 and now.minute >= 15:
            return self._exit("EOD_PRE_EXPIRY", dry_run=dry_run)

        sq = self.cfg.get("trading", {}).get("square_off_time", "15:15").split(":")
        if now.hour > int(sq[0]) or (now.hour == int(sq[0]) and now.minute >= int(sq[1])):
            return self._exit("INTRADAY_SQUAREOFF", dry_run=dry_run)

        ce_ltp, pe_ltp = self._live_ltps(pos)
        if ce_ltp is None:
            return None

        current = ce_ltp + pe_ltp
        if current >= pos.sl_premium:
            return self._exit("STOP_LOSS", dry_run, ce_ltp, pe_ltp)
        if current <= pos.target_premium:
            return self._exit("TARGET_HIT", dry_run, ce_ltp, pe_ltp)
        return None

    # ── Exit ──────────────────────────────────────────────────────────

    def _exit(
        self, reason: str, dry_run: bool = True,
        exit_ce_ltp: Optional[float] = None,
        exit_pe_ltp: Optional[float] = None,
    ) -> dict:
        pos = self.position
        if exit_ce_ltp is None or exit_pe_ltp is None:
            exit_ce_ltp, exit_pe_ltp = self._live_ltps(pos)
            exit_ce_ltp = exit_ce_ltp or pos.ce_entry_price
            exit_pe_ltp = exit_pe_ltp or pos.pe_entry_price

        pnl_pts = (pos.ce_entry_price - exit_ce_ltp) + (pos.pe_entry_price - exit_pe_ltp)
        pnl_rs  = pnl_pts * pos.lot_size * pos.lots

        if not dry_run:
            self._place_buy(pos.ce_instrument_key, pos.lot_size * pos.lots)
            self._place_buy(pos.pe_instrument_key, pos.lot_size * pos.lots)
        else:
            logger.info("[PAPER] DefaultShortStrangle EXIT (%s): CE@%.2f PE@%.2f P&L=₹%.0f",
                        reason, exit_ce_ltp, exit_pe_ltp, pnl_rs)

        self.risk.record_pnl(pnl_rs)
        self.risk.release_capital(pos.capital_deployed)

        result = {
            "status": "EXITED", "reason": reason, "dry_run": dry_run,
            "strategy": "DefaultShortStrangle",
            "symbol": pos.symbol, "expiry": pos.expiry,
            "ce_strike": pos.ce_strike, "pe_strike": pos.pe_strike,
            "ce_entry": pos.ce_entry_price, "pe_entry": pos.pe_entry_price,
            "ce_exit": exit_ce_ltp, "pe_exit": exit_pe_ltp,
            "entry_combined": pos.entry_combined_premium,
            "exit_combined": exit_ce_ltp + exit_pe_ltp,
            "realised_pnl_rs": round(pnl_rs, 2),
            "entry_ts": pos.entry_ts,
            "exit_ts": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        }
        logger.info("DefaultShortStrangle EXITED (%s): %s P&L=₹%.0f", reason, pos.symbol, pnl_rs)
        self.position = None
        return result

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_spot(self, symbol: str) -> Optional[float]:
        inst_key = self.cfg["instruments"][symbol]["upstox_key"]
        ltps = self.broker.get_ltp([inst_key])
        v = ltps.get(inst_key, 0.0)
        return float(v) if v and v > 0 else None

    def _fetch_chain(self, inst_key: str, expiry: str):
        try:
            chain = self.broker.get_option_chain(inst_key, expiry)
            return chain if chain is not None and not chain.empty else None
        except Exception as e:
            logger.warning("Option chain fetch failed: %s", e)
            return None

    @staticmethod
    def _get_chain_ltp(chain, strike: int, opt_type: str) -> float:
        col = f"{opt_type}_ltp"
        row = chain[chain["strike"] == strike]
        if not row.empty and col in row.columns:
            return float(row[col].iloc[0]) or 0.0
        # fallback: instrument_type column style (from ZerodhaBroker)
        row2 = chain[(chain["strike"] == strike) & (chain["instrument_type"] == opt_type)]
        if not row2.empty and "ltp" in row2.columns:
            return float(row2["ltp"].iloc[0]) or 0.0
        return 0.0

    def _live_ltps(self, pos):
        inst_key = self.cfg["instruments"][pos.symbol]["upstox_key"]
        try:
            chain = self.broker.get_option_chain(inst_key, pos.expiry)
            if chain is not None and not chain.empty:
                ce = self._get_chain_ltp(chain, pos.ce_strike, "CE")
                pe = self._get_chain_ltp(chain, pos.pe_strike, "PE")
                if ce > 0 and pe > 0:
                    return ce, pe
        except Exception as e:
            logger.warning("LTP fetch error: %s", e)
        return None, None

    def _resolve_key(self, inst_key, expiry, strike, opt_type):
        try:
            return self.broker.get_expired_option_key(inst_key, expiry, strike, opt_type)
        except Exception:
            return None

    def _place_sell(self, key, qty):
        if key:
            self.broker.place_order(key, "SELL", qty, "MARKET")

    def _place_buy(self, key, qty):
        if key:
            self.broker.place_order(key, "BUY", qty, "MARKET")

    def _estimate_margin(self, spot: float, lot_size: int, lots: int) -> float:
        return round(spot * lot_size * lots * 0.10, 0)

    def status(self) -> dict:
        if self.position is None:
            return {"open": False}
        pos = self.position
        ce_ltp, pe_ltp = self._live_ltps(pos)
        ce_ltp = ce_ltp or pos.ce_entry_price
        pe_ltp = pe_ltp or pos.pe_entry_price
        return {
            "open": True, "strategy": "DefaultShortStrangle",
            "symbol": pos.symbol, "expiry": pos.expiry,
            "ce_strike": pos.ce_strike, "pe_strike": pos.pe_strike,
            "ce_entry": pos.ce_entry_price, "pe_entry": pos.pe_entry_price,
            "ce_ltp": ce_ltp, "pe_ltp": pe_ltp,
            "entry_combined": pos.entry_combined_premium,
            "current_combined": ce_ltp + pe_ltp,
            "unrealised_pnl_rs": pos.unrealised_pnl_rs(ce_ltp, pe_ltp),
            "sl_premium": pos.sl_premium, "target_premium": pos.target_premium,
            "entry_ts": pos.entry_ts,
            **self.risk.status(),
        }
