"""
Options Mapper — translates a signal into a concrete option trade recommendation.

Given a signal (BULLISH, BEARISH, etc.) + dispersion/IV regimes, picks a strategy
and maps it to specific strikes using the live option chain.

Lot sizes are fetched from config (which should be updated when NSE changes them).
Strategy logic:
  STRONG_BULLISH  → Buy ATM CE  (or Bull Call Spread if configured)
  BULLISH         → Bull Put Spread (sell ATM PE, buy OTM PE)
  NEUTRAL/high-dispersion/low-IV  → Long Straddle
  NEUTRAL/low-dispersion/high-IV  → Iron Condor
  BEARISH         → Bear Call Spread (sell ATM CE, buy OTM CE)
  STRONG_BEARISH  → Buy ATM PE  (or Bear Put Spread if configured)
"""
from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

from src.broker.base import BrokerInterface
from src.utils import load_config, round_to_atm, get_instrument_key

logger = logging.getLogger(__name__)


class OptionsMapper:
    def __init__(
        self,
        broker: BrokerInterface,
        config_path: str = "config.yaml",
        db_path: str = "data/kronos_options.db",
    ):
        self.broker = broker
        self.cfg = load_config(config_path)
        self.config_path = config_path
        self.db_path = db_path

    def map(
        self,
        signal: dict,
        expiry: str,
        option_chain: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Convert a signal dict into a trade recommendation.

        Args:
            signal:       Output of SignalEngine.generate()
            expiry:       Target expiry date "YYYY-MM-DD"
            option_chain: Live option chain DataFrame (fetched if not provided)

        Returns recommendation dict with:
            strategy, legs (list of {strike, option_type, action, lots, ltp}),
            lot_size, max_profit_rs, max_loss_rs, breakeven_upper, breakeven_lower
        """
        symbol = signal["signal_name"] if "signal_name" in signal else signal.get("symbol", "")
        sig    = signal["signal"]
        dispersion_regime = signal.get("dispersion_regime", "medium")
        iv_regime         = signal.get("iv_regime", None)
        current_close     = signal["current_close"]

        inst_cfg  = self.cfg["instruments"][symbol]
        atm_step  = inst_cfg["atm_step"]
        lot_size  = inst_cfg["lot_size"]
        inst_key  = get_instrument_key(symbol, self.config_path)
        spread_steps = self.cfg.get("options", {}).get("spread_otm_strikes", 1)
        lots = self.cfg.get("paper_trading", {}).get("lots", 1)

        atm_strike = round_to_atm(current_close, atm_step)

        # Fetch option chain if not provided
        if option_chain is None or option_chain.empty:
            try:
                option_chain = self.broker.get_option_chain(inst_key, expiry)
            except Exception as e:
                logger.warning("Could not fetch option chain: %s — using ATM strike only", e)
                option_chain = pd.DataFrame()

        # Determine strategy
        strategy_map = self.cfg.get("options", {}).get("default_strategy", {})

        if sig == "NEUTRAL":
            if dispersion_regime == "high" and iv_regime in (None, "low"):
                sig_key = "NEUTRAL_HIGH_DISP_LOW_IV"
            elif dispersion_regime == "low" and iv_regime == "high":
                sig_key = "NEUTRAL_LOW_DISP_HIGH_IV"
            else:
                return self._no_trade(symbol, "NEUTRAL — no clear volatility edge")
        else:
            sig_key = sig

        strategy = strategy_map.get(sig_key, "buy_atm_ce")

        legs = self._build_legs(
            strategy=strategy,
            atm_strike=atm_strike,
            atm_step=atm_step,
            spread_steps=spread_steps,
            lots=lots,
            option_chain=option_chain,
        )

        if not legs:
            return self._no_trade(symbol, f"Could not build legs for {strategy}")

        metrics = self._compute_metrics(strategy, legs, lot_size)

        rec = {
            "symbol":           symbol,
            "strategy":         strategy,
            "signal":           signal["signal"],
            "confidence":       signal["confidence"],
            "expiry":           expiry,
            "atm_strike":       atm_strike,
            "legs":             legs,
            "lot_size":         lot_size,
            "lots":             lots,
            "max_profit_rs":    metrics["max_profit_rs"],
            "max_loss_rs":      metrics["max_loss_rs"],
            "breakeven_upper":  metrics.get("breakeven_upper"),
            "breakeven_lower":  metrics.get("breakeven_lower"),
            "net_premium_rs":   metrics["net_premium_rs"],
        }
        logger.info("%s → %s | strikes=%s | max_loss=₹%.0f | max_profit=₹%.0f",
                    symbol, strategy,
                    [f"{l['strike']}{l['option_type']}" for l in legs],
                    metrics.get("max_loss_rs", 0),
                    metrics.get("max_profit_rs", 0))
        return rec

    # ── Leg builders ───────────────────────────────────────────────────

    def _build_legs(
        self,
        strategy: str,
        atm_strike: int,
        atm_step: int,
        spread_steps: int,
        lots: int,
        option_chain: pd.DataFrame,
    ) -> list[dict]:
        """Return list of leg dicts: {strike, option_type, action, lots, ltp}."""
        otm = atm_step * spread_steps

        def get_ltp(strike: int, opt_type: str) -> float:
            """Lookup LTP from chain; return 0 if unavailable."""
            if option_chain.empty:
                return 0.0
            col = f"{opt_type}_ltp"
            row = option_chain[option_chain["strike"] == strike]
            if not row.empty and col in row.columns:
                return float(row[col].iloc[0]) or 0.0
            return 0.0

        builders = {
            "buy_atm_ce": lambda: [
                {"strike": atm_strike, "option_type": "CE", "action": "BUY", "lots": lots,
                 "ltp": get_ltp(atm_strike, "CE")},
            ],
            "buy_atm_pe": lambda: [
                {"strike": atm_strike, "option_type": "PE", "action": "BUY", "lots": lots,
                 "ltp": get_ltp(atm_strike, "PE")},
            ],
            "bull_call_spread": lambda: [
                {"strike": atm_strike,       "option_type": "CE", "action": "BUY",  "lots": lots,
                 "ltp": get_ltp(atm_strike, "CE")},
                {"strike": atm_strike + otm, "option_type": "CE", "action": "SELL", "lots": lots,
                 "ltp": get_ltp(atm_strike + otm, "CE")},
            ],
            "bull_put_spread": lambda: [
                {"strike": atm_strike,       "option_type": "PE", "action": "SELL", "lots": lots,
                 "ltp": get_ltp(atm_strike, "PE")},
                {"strike": atm_strike - otm, "option_type": "PE", "action": "BUY",  "lots": lots,
                 "ltp": get_ltp(atm_strike - otm, "PE")},
            ],
            "bear_call_spread": lambda: [
                {"strike": atm_strike,       "option_type": "CE", "action": "SELL", "lots": lots,
                 "ltp": get_ltp(atm_strike, "CE")},
                {"strike": atm_strike + otm, "option_type": "CE", "action": "BUY",  "lots": lots,
                 "ltp": get_ltp(atm_strike + otm, "CE")},
            ],
            "bear_put_spread": lambda: [
                {"strike": atm_strike,       "option_type": "PE", "action": "BUY",  "lots": lots,
                 "ltp": get_ltp(atm_strike, "PE")},
                {"strike": atm_strike - otm, "option_type": "PE", "action": "SELL", "lots": lots,
                 "ltp": get_ltp(atm_strike - otm, "PE")},
            ],
            "straddle": lambda: [
                {"strike": atm_strike, "option_type": "CE", "action": "BUY", "lots": lots,
                 "ltp": get_ltp(atm_strike, "CE")},
                {"strike": atm_strike, "option_type": "PE", "action": "BUY", "lots": lots,
                 "ltp": get_ltp(atm_strike, "PE")},
            ],
            "iron_condor": lambda: [
                {"strike": atm_strike - otm, "option_type": "PE", "action": "BUY",  "lots": lots,
                 "ltp": get_ltp(atm_strike - otm, "PE")},
                {"strike": atm_strike,       "option_type": "PE", "action": "SELL", "lots": lots,
                 "ltp": get_ltp(atm_strike, "PE")},
                {"strike": atm_strike,       "option_type": "CE", "action": "SELL", "lots": lots,
                 "ltp": get_ltp(atm_strike, "CE")},
                {"strike": atm_strike + otm, "option_type": "CE", "action": "BUY",  "lots": lots,
                 "ltp": get_ltp(atm_strike + otm, "CE")},
            ],
        }

        builder = builders.get(strategy)
        if builder is None:
            logger.error("Unknown strategy: %s", strategy)
            return []
        return builder()

    # ── Risk/reward calculation ─────────────────────────────────────────

    def _compute_metrics(self, strategy: str, legs: list[dict], lot_size: int) -> dict:
        """
        Compute max profit, max loss, and breakevens from leg premiums.
        These are approximate (based on LTP at time of signal) — actual fill prices will differ.
        """
        net_premium_pts = sum(
            (leg["ltp"] if leg["action"] == "SELL" else -leg["ltp"]) * leg["lots"]
            for leg in legs
        )
        net_premium_rs = net_premium_pts * lot_size

        if strategy in ("buy_atm_ce", "buy_atm_pe"):
            debit = abs(net_premium_rs)
            return {
                "max_loss_rs":   -debit,
                "max_profit_rs": debit * 5,
                "net_premium_rs": net_premium_rs,
            }
        elif strategy in ("bull_put_spread", "bear_call_spread"):
            width = abs(legs[0]["strike"] - legs[1]["strike"]) * lot_size * legs[0]["lots"]
            credit = net_premium_rs
            atm = legs[0]["strike"]
            return {
                "max_profit_rs":  credit,
                "max_loss_rs":    -(width - credit),
                "breakeven_upper": atm + (credit / lot_size / legs[0]["lots"])
                                   if strategy == "bear_call_spread" else None,
                "breakeven_lower": atm - (credit / lot_size / legs[0]["lots"])
                                   if strategy == "bull_put_spread" else None,
                "net_premium_rs": net_premium_rs,
            }
        elif strategy in ("bull_call_spread", "bear_put_spread"):
            debit = -net_premium_rs
            width = abs(legs[0]["strike"] - legs[1]["strike"]) * lot_size * legs[0]["lots"]
            return {
                "max_profit_rs": width - debit,
                "max_loss_rs":   -debit,
                "net_premium_rs": net_premium_rs,
            }
        elif strategy == "straddle":
            debit = -net_premium_rs
            atm = legs[0]["strike"]
            return {
                "max_profit_rs":  float("inf"),
                "max_loss_rs":    -debit,
                "breakeven_upper": atm + debit / lot_size / legs[0]["lots"],
                "breakeven_lower": atm - debit / lot_size / legs[0]["lots"],
                "net_premium_rs": net_premium_rs,
            }
        elif strategy == "iron_condor":
            credit = net_premium_rs
            width = abs(legs[1]["strike"] - legs[0]["strike"]) * lot_size * legs[0]["lots"]
            return {
                "max_profit_rs":  credit,
                "max_loss_rs":    -(width - credit),
                "breakeven_upper": legs[2]["strike"] + credit / lot_size / legs[0]["lots"],
                "breakeven_lower": legs[1]["strike"] - credit / lot_size / legs[0]["lots"],
                "net_premium_rs": net_premium_rs,
            }
        return {"max_profit_rs": 0, "max_loss_rs": 0, "net_premium_rs": net_premium_rs}

    def _no_trade(self, symbol: str, reason: str) -> dict:
        logger.info("%s: no trade — %s", symbol, reason)
        return {"symbol": symbol, "strategy": "NO_TRADE", "reason": reason, "legs": []}
