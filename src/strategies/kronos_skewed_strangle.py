"""
kronos_skewed_strangle.py — Kronos-forecast-driven asymmetric short strangle.

Difference from DefaultShortStrangle:
  - Uses KronosForecaster to predict where spot will be at expiry.
  - Shifts strikes AWAY from the predicted close to maximise probability
    that both legs expire OTM (i.e. we keep full premium).
  - Widens/narrows spread dynamically based on path_dispersion (Kronos
    uncertainty): high dispersion → wider strikes for safety.
  - Skips entry if prob_up confidence is too low (unclear direction).

Strike selection algorithm:
    1. Run Kronos forecast → expected_close, expected_range (p10, p90),
       prob_up, path_dispersion.
    2. CE strike = max(ATM, round_to_atm(p90 * (1 + ce_buffer)))
       PE strike = min(ATM, round_to_atm(p10 * (1 - pe_buffer)))
       Buffers are configurable (default 0.5% of spot).
    3. If path_dispersion > high_disp_threshold → apply extra_width_steps
       additional strike steps on each side.
    4. Verify both legs have reasonable premium (> min_premium_pts).
    5. Verify risk/reward: max_loss < max_loss_limit_rs (from config).
    6. RiskManager gate (₹3L cap).

Risk management:
  - Same SL/target as DefaultShortStrangle (2× premium / 50% decay).
  - Asymmetric leg adjustments tracked separately (CE/PE may differ).
  - Forced exit day before expiry at 15:15.

Usage:
    from src.strategies.kronos_skewed_strangle import KronosSkewedStrangle
    from src.forecaster import KronosForecaster

    forecaster = KronosForecaster()
    strat = KronosSkewedStrangle(broker=broker, forecaster=forecaster)

    # Monday morning — pass today's 5-min OHLCV DataFrame
    result = strat.enter(symbol="NIFTY", df_ohlcv=df, dry_run=True)
    print(result)  # includes forecast details + skewed strikes

    # Every 5 min in trading loop
    exit_result = strat.monitor(dry_run=True)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

from src.broker.base import BrokerInterface
from src.forecaster import KronosForecaster
from src.risk_manager import RiskManager
from src.strategies.default_short_strangle import (
    StranglePosition, next_weekly_expiry, is_exit_eod_day,
)
from src.utils import load_config, round_to_atm, IST

logger = logging.getLogger(__name__)


# ── Config defaults (overridable in config.yaml under strategies.kronos_skewed) ──

_DEFAULTS = {
    "ce_buffer_pct":          0.005,   # shift CE strike 0.5% above p90
    "pe_buffer_pct":          0.005,   # shift PE strike 0.5% below p10
    "high_disp_threshold":    2.0,     # path_dispersion % above which extra width applies
    "extra_width_steps":      1,       # extra ATM steps each side when high dispersion
    "min_premium_pts":        10.0,    # skip leg if premium < this (too far OTM)
    "min_prob_up":            0.35,    # skip entry if prob_up < 0.35 or > 0.65 (unclear)
    "max_prob_up":            0.65,    # (i.e. too strong directional → don't sell)
    "sl_multiplier":          2.0,     # SL = entry_combined × sl_multiplier
    "target_multiplier":      0.50,    # Target = entry_combined × target_multiplier
    "max_loss_limit_rs":      8000.0,  # Hard max-loss filter before entry (₹)
}


class KronosSkewedStrangle:
    """
    Kronos-powered asymmetric short strangle.
    Holds at most ONE open position at a time.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        forecaster: KronosForecaster,
        config_path: str = "config.yaml",
        risk_manager: Optional[RiskManager] = None,
    ):
        self.broker     = broker
        self.forecaster = forecaster
        self.cfg        = load_config(config_path)
        self.position: Optional[StranglePosition] = None

        live_cfg = self.cfg.get("live_trading", {})
        self.risk = risk_manager or RiskManager(
            broker=broker,
            max_total_capital    = live_cfg.get("max_capital_total",    300000),
            max_capital_per_trade= live_cfg.get("max_capital_per_trade",  50000),
            max_daily_loss       = live_cfg.get("max_daily_loss_rs",       5000),
        )

        strat_cfg  = self.cfg.get("strategies", {}).get("kronos_skewed", {})
        self.scfg  = {**_DEFAULTS, **strat_cfg}   # config.yaml overrides defaults

    # ── Entry ─────────────────────────────────────────────────────────

    def enter(
        self,
        symbol:   str,
        df_ohlcv: pd.DataFrame,
        dry_run:  bool = True,
    ) -> dict:
        """
        Enter Kronos-skewed short strangle.

        Args:
            symbol:    "NIFTY" | "BANKNIFTY" | "SENSEX"
            df_ohlcv:  5-min OHLCV DataFrame with IST DatetimeIndex
                       (typically last 400 bars, fetched by the trading loop)
            dry_run:   True = paper trade.

        Returns entry dict with forecast details and chosen strikes,
        or {"status": "BLOCKED"/"SKIPPED"/"ERROR", "reason": …}.
        """
        if self.position is not None:
            return {"status": "BLOCKED", "reason": "Position already open."}

        inst_cfg = self.cfg["instruments"][symbol]
        atm_step = inst_cfg["atm_step"]
        lot_size = inst_cfg["lot_size"]
        inst_key = inst_cfg["upstox_key"]
        lots     = self.cfg.get(
            "paper_trading" if dry_run else "live_trading", {}
        ).get("lots", 1)

        expiry     = next_weekly_expiry(symbol)
        expiry_str = expiry.strftime("%Y-%m-%d")

        # ── Step 1: Run Kronos forecast ────────────────────────────────
        try:
            forecast = self.forecaster.forecast(
                symbol=symbol,
                df=df_ohlcv,
                pred_len=self._days_to_expiry(expiry) * 75,  # ~75 bars/trading day (5-min)
                samples=self.cfg.get("kronos", {}).get("samples", 20),
            )
        except Exception as e:
            logger.error("Kronos forecast failed: %s — falling back to DEFAULT entry", e)
            return {"status": "ERROR", "reason": f"Kronos forecast failed: {e}"}

        prob_up    = forecast["prob_up"]
        dispersion = forecast["path_dispersion"]
        p10, p90   = forecast["expected_range"]
        spot       = forecast["current_close"]
        expected_close = forecast["expected_close"]

        logger.info(
            "Kronos forecast [%s]: spot=%.2f expected=%.2f prob_up=%.2f "
            "dispersion=%.2f%% range=(%.2f, %.2f)",
            symbol, spot, expected_close, prob_up, dispersion, p10, p90,
        )

        # ── Step 2: Directional filter ─────────────────────────────────
        # If Kronos is very strongly directional (outside 35–65% range),
        # selling a strangle has elevated risk — skip.
        if prob_up < self.scfg["min_prob_up"] or prob_up > self.scfg["max_prob_up"]:
            reason = (
                f"Kronos prob_up={prob_up:.2f} is strongly directional "
                f"(threshold: [{self.scfg['min_prob_up']}, {self.scfg['max_prob_up']}]). "
                f"Short strangle not suitable. Consider directional strategy instead."
            )
            logger.info("KronosSkewedStrangle SKIPPED: %s", reason)
            return {"status": "SKIPPED", "reason": reason, "forecast": self._fmt_forecast(forecast)}

        # ── Step 3: Compute skewed strikes ─────────────────────────────
        atm_strike = round_to_atm(spot, atm_step)

        ce_raw = p90 * (1 + self.scfg["ce_buffer_pct"])
        pe_raw = p10 * (1 - self.scfg["pe_buffer_pct"])

        ce_strike = max(atm_strike, round_to_atm(ce_raw, atm_step))
        pe_strike = min(atm_strike, round_to_atm(pe_raw, atm_step))

        # High dispersion → push strikes further out
        if dispersion > self.scfg["high_disp_threshold"]:
            extra = int(self.scfg["extra_width_steps"]) * atm_step
            ce_strike += extra
            pe_strike -= extra
            logger.info(
                "High dispersion (%.2f%%) → extra_width=%d applied. "
                "New strikes: CE=%d PE=%d", dispersion, extra, ce_strike, pe_strike,
            )

        # ── Step 4: Fetch option chain and validate premiums ───────────
        try:
            option_chain = self.broker.get_option_chain(inst_key, expiry_str)
        except Exception as e:
            return {"status": "ERROR", "reason": f"Option chain fetch failed: {e}"}

        if option_chain is None or option_chain.empty:
            return {"status": "ERROR", "reason": "Empty option chain."}

        ce_ltp = self._get_ltp(option_chain, ce_strike, "CE")
        pe_ltp = self._get_ltp(option_chain, pe_strike, "PE")

        # If premium too thin (strikes too far OTM), pull back one step
        if ce_ltp < self.scfg["min_premium_pts"]:
            ce_strike -= atm_step
            ce_ltp    = self._get_ltp(option_chain, ce_strike, "CE")
            logger.info("CE premium too thin — pulled back to %d (ltp=%.2f)", ce_strike, ce_ltp)

        if pe_ltp < self.scfg["min_premium_pts"]:
            pe_strike += atm_step
            pe_ltp    = self._get_ltp(option_chain, pe_strike, "PE")
            logger.info("PE premium too thin — pulled back to %d (ltp=%.2f)", pe_strike, pe_ltp)

        if ce_ltp <= 0 or pe_ltp <= 0:
            return {"status": "ERROR", "reason": f"Still invalid premiums after adjustment: CE={ce_ltp} PE={pe_ltp}"}

        combined = ce_ltp + pe_ltp

        # ── Step 5: Risk/reward filter ─────────────────────────────────
        # Max loss (SL-capped) = 1× combined premium collected
        max_loss_rs = combined * lot_size * lots
        if max_loss_rs > self.scfg["max_loss_limit_rs"]:
            return {
                "status": "BLOCKED",
                "reason": f"max_loss_rs=₹{max_loss_rs:.0f} exceeds limit ₹{self.scfg['max_loss_limit_rs']:.0f}",
                "forecast": self._fmt_forecast(forecast),
            }

        # ── Step 6: RiskManager gate ───────────────────────────────────
        margin = round(spot * lot_size * lots * 0.10, 0)
        ok, reason = self.risk.check_pre_trade(margin)
        if not ok:
            return {"status": "BLOCKED", "reason": reason}

        # ── Step 7: Resolve instrument keys and place orders ───────────
        ce_key = self._resolve_key(inst_key, expiry_str, ce_strike, "CE")
        pe_key = self._resolve_key(inst_key, expiry_str, pe_strike, "PE")
        now    = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        if not dry_run:
            if ce_key:
                self.broker.place_order(ce_key, "SELL", lot_size * lots, "MARKET")
            if pe_key:
                self.broker.place_order(pe_key, "SELL", lot_size * lots, "MARKET")
        else:
            logger.info(
                "[PAPER] KronosSkewedStrangle ENTER: SELL %sCE=%d@%.2f | SELL %sPE=%d@%.2f "
                "(ATM=%d, expected_close=%.2f, dispersion=%.2f%%)",
                symbol, ce_strike, ce_ltp,
                symbol, pe_strike, pe_ltp,
                atm_strike, expected_close, dispersion,
            )

        self.position = StranglePosition(
            symbol=symbol, expiry=expiry_str, atm_strike=atm_strike,
            ce_strike=ce_strike, pe_strike=pe_strike,
            ce_entry_price=ce_ltp, pe_entry_price=pe_ltp,
            lots=lots, lot_size=lot_size, entry_ts=now,
            ce_instrument_key=ce_key or "",
            pe_instrument_key=pe_key or "",
            capital_deployed=margin,
            strategy_name="KronosSkewedStrangle",
        )
        self.risk.record_trade(margin)

        return {
            "status":          "ENTERED",
            "dry_run":         dry_run,
            "strategy":        "KronosSkewedStrangle",
            "symbol":          symbol,
            "expiry":          expiry_str,
            "atm_strike":      atm_strike,
            "ce_strike":       ce_strike,
            "pe_strike":       pe_strike,
            "ce_entry":        ce_ltp,
            "pe_entry":        pe_ltp,
            "combined_premium":combined,
            "sl_premium":      combined * self.scfg["sl_multiplier"],
            "target_premium":  combined * self.scfg["target_multiplier"],
            "max_profit_rs":   combined * lot_size * lots,
            "max_loss_rs":     -max_loss_rs,
            "margin_required": margin,
            "lots":            lots,
            "lot_size":        lot_size,
            "entry_ts":        now,
            "forecast":        self._fmt_forecast(forecast),
            # Skew metrics for logging / dashboard
            "skew_metrics": {
                "prob_up":        round(prob_up, 4),
                "dispersion_pct": round(dispersion, 4),
                "p10":            round(p10, 2),
                "p90":            round(p90, 2),
                "expected_close": round(expected_close, 2),
                "ce_width_pts":   ce_strike - atm_strike,
                "pe_width_pts":   atm_strike - pe_strike,
            },
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

        pnl_rs = (
            (pos.ce_entry_price - exit_ce_ltp) +
            (pos.pe_entry_price - exit_pe_ltp)
        ) * pos.lot_size * pos.lots

        if not dry_run:
            if pos.ce_instrument_key:
                self.broker.place_order(pos.ce_instrument_key, "BUY",
                                        pos.lot_size * pos.lots, "MARKET")
            if pos.pe_instrument_key:
                self.broker.place_order(pos.pe_instrument_key, "BUY",
                                        pos.lot_size * pos.lots, "MARKET")
        else:
            logger.info(
                "[PAPER] KronosSkewedStrangle EXIT (%s): CE@%.2f PE@%.2f P&L=₹%.0f",
                reason, exit_ce_ltp, exit_pe_ltp, pnl_rs,
            )

        self.risk.record_pnl(pnl_rs)
        self.risk.release_capital(pos.capital_deployed)

        result = {
            "status": "EXITED", "reason": reason, "dry_run": dry_run,
            "strategy": "KronosSkewedStrangle",
            "symbol": pos.symbol, "expiry": pos.expiry,
            "atm_strike": pos.atm_strike,
            "ce_strike": pos.ce_strike, "pe_strike": pos.pe_strike,
            "ce_entry": pos.ce_entry_price, "pe_entry": pos.pe_entry_price,
            "ce_exit": exit_ce_ltp, "pe_exit": exit_pe_ltp,
            "entry_combined": pos.entry_combined_premium,
            "exit_combined":  exit_ce_ltp + exit_pe_ltp,
            "realised_pnl_rs": round(pnl_rs, 2),
            "entry_ts": pos.entry_ts,
            "exit_ts":  datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        }
        logger.info("KronosSkewedStrangle EXITED (%s): %s P&L=₹%.0f", reason, pos.symbol, pnl_rs)
        self.position = None
        return result

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _get_ltp(chain: pd.DataFrame, strike: int, opt_type: str) -> float:
        col = f"{opt_type}_ltp"
        row = chain[chain["strike"] == strike]
        if not row.empty and col in row.columns:
            return float(row[col].iloc[0]) or 0.0
        row2 = chain[(chain["strike"] == strike) & (chain["instrument_type"] == opt_type)]
        if not row2.empty and "ltp" in row2.columns:
            return float(row2["ltp"].iloc[0]) or 0.0
        return 0.0

    def _live_ltps(self, pos: StranglePosition):
        inst_key = self.cfg["instruments"][pos.symbol]["upstox_key"]
        try:
            chain = self.broker.get_option_chain(inst_key, pos.expiry)
            if chain is not None and not chain.empty:
                ce = self._get_ltp(chain, pos.ce_strike, "CE")
                pe = self._get_ltp(chain, pos.pe_strike, "PE")
                if ce > 0 and pe > 0:
                    return ce, pe
        except Exception as e:
            logger.warning("LTP fetch: %s", e)
        return None, None

    def _resolve_key(self, inst_key, expiry, strike, opt_type):
        try:
            return self.broker.get_expired_option_key(inst_key, expiry, strike, opt_type)
        except Exception:
            return None

    @staticmethod
    def _days_to_expiry(expiry: date) -> int:
        d = max(1, (expiry - date.today()).days)
        return d

    @staticmethod
    def _fmt_forecast(f: dict) -> dict:
        """Compact forecast summary for inclusion in entry/skip result."""
        return {
            "prob_up":        round(f.get("prob_up", 0), 4),
            "expected_close": round(f.get("expected_close", 0), 2),
            "expected_move_pct": round(f.get("expected_move_pct", 0), 4),
            "dispersion_pct": round(f.get("path_dispersion", 0), 4),
            "p10":            round(f["expected_range"][0], 2),
            "p90":            round(f["expected_range"][1], 2),
            "current_close":  round(f.get("current_close", 0), 2),
        }

    def status(self) -> dict:
        if self.position is None:
            return {"open": False}
        pos = self.position
        ce_ltp, pe_ltp = self._live_ltps(pos)
        ce_ltp = ce_ltp or pos.ce_entry_price
        pe_ltp = pe_ltp or pos.pe_entry_price
        return {
            "open": True, "strategy": "KronosSkewedStrangle",
            "symbol": pos.symbol, "expiry": pos.expiry,
            "atm_strike": pos.atm_strike,
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
