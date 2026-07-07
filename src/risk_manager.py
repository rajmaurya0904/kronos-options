"""
risk_manager.py — Hard capital and loss guards shared across all strategies.

Enforces the ₹3L total capital constraint from Space config.
All strategy enter() calls must pass check_pre_trade() before placing orders.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Stateful risk guard. One instance shared across all open strategies
    so the ₹3L cap is enforced in aggregate, not per-strategy.

    Args:
        broker:               Broker instance (used to query live margin).
        max_total_capital:    Hard cap on total capital deployed (₹). Default ₹3,00,000.
        max_capital_per_trade: Max margin for a single trade entry (₹). Default ₹50,000.
        max_daily_loss:       If daily realised P&L drops below this (negative), kill switch
                              blocks all new entries for the rest of the day. Default -₹5,000.
    """

    def __init__(
        self,
        broker,
        max_total_capital:     float = 300_000.0,
        max_capital_per_trade: float =  50_000.0,
        max_daily_loss:        float =  -5_000.0,
    ):
        self.broker               = broker
        self.max_total_capital    = max_total_capital
        self.max_capital_per_trade= max_capital_per_trade
        self.max_daily_loss       = max_daily_loss

        self._capital_deployed: float = 0.0   # running sum of open trade margins
        self._daily_pnl:        float = 0.0   # realised P&L today
        self._killed:           bool  = False  # daily kill switch
        self._trades_today:     int   = 0

    # ── Pre-trade check (call before every entry) ──────────────────────

    def check_pre_trade(self, margin_required: float) -> Tuple[bool, str]:
        """
        Returns (True, "") if the trade is allowed.
        Returns (False, reason_string) if blocked.
        Call this BEFORE placing any order.
        """
        if self._killed:
            return False, f"Kill switch active. Daily P&L=₹{self._daily_pnl:.0f} <= limit ₹{self.max_daily_loss:.0f}"

        if self._daily_pnl <= self.max_daily_loss:
            self._killed = True
            logger.critical(
                "KILL SWITCH: Daily loss ₹%.0f hit limit ₹%.0f. No new trades today.",
                self._daily_pnl, self.max_daily_loss,
            )
            return False, f"Daily loss limit ₹{self.max_daily_loss:.0f} breached."

        if margin_required > self.max_capital_per_trade:
            return False, (
                f"margin_required=₹{margin_required:.0f} exceeds "
                f"max_capital_per_trade=₹{self.max_capital_per_trade:.0f}"
            )

        projected = self._capital_deployed + margin_required
        if projected > self.max_total_capital:
            return False, (
                f"Total capital would reach ₹{projected:.0f}, "
                f"exceeding ₹3L cap (₹{self.max_total_capital:.0f}). "
                f"Currently deployed: ₹{self._capital_deployed:.0f}"
            )

        # Optional: check broker's live available margin
        try:
            available = self.broker.get_available_margin()
            if available < margin_required:
                return False, (
                    f"Broker margin insufficient: available=₹{available:.0f} "
                    f"vs required=₹{margin_required:.0f}"
                )
        except Exception as e:
            logger.warning("Could not fetch live margin: %s — skipping broker check.", e)

        return True, ""

    # ── State updates (call after entry/exit) ──────────────────────────

    def record_trade(self, margin_deployed: float) -> None:
        """Call after a successful entry. Adds margin to deployed capital."""
        self._capital_deployed += margin_deployed
        self._trades_today     += 1
        logger.debug(
            "RiskManager: trade recorded. deployed=₹%.0f total_today=%d",
            self._capital_deployed, self._trades_today,
        )

    def release_capital(self, margin_released: float) -> None:
        """Call after exit. Frees up deployed capital."""
        self._capital_deployed = max(0.0, self._capital_deployed - margin_released)
        logger.debug("RiskManager: capital released ₹%.0f. deployed=₹%.0f",
                     margin_released, self._capital_deployed)

    def record_pnl(self, pnl_rs: float) -> None:
        """Call after exit with realised P&L (positive=profit, negative=loss)."""
        self._daily_pnl += pnl_rs
        logger.info("RiskManager: P&L recorded ₹%.0f. daily_pnl=₹%.0f", pnl_rs, self._daily_pnl)
        if self._daily_pnl <= self.max_daily_loss:
            self._killed = True
            logger.critical(
                "KILL SWITCH ENGAGED: daily_pnl=₹%.0f <= limit=₹%.0f",
                self._daily_pnl, self.max_daily_loss,
            )

    def reset_daily(self) -> None:
        """Call at start of each trading day to reset daily counters."""
        logger.info(
            "RiskManager: daily reset. yesterday_pnl=₹%.0f trades=%d",
            self._daily_pnl, self._trades_today,
        )
        self._daily_pnl    = 0.0
        self._trades_today = 0
        self._killed       = False
        # NOTE: _capital_deployed is NOT reset — open overnight positions still count.

    # ── Status ───────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "capital_deployed_rs": round(self._capital_deployed, 2),
            "capital_free_rs":     round(max(0, self.max_total_capital - self._capital_deployed), 2),
            "daily_pnl_rs":        round(self._daily_pnl, 2),
            "trades_today":        self._trades_today,
            "kill_switch_active":  self._killed,
            "max_total_capital":   self.max_total_capital,
            "max_daily_loss":      self.max_daily_loss,
        }

    def __repr__(self) -> str:
        return (
            f"RiskManager(deployed=₹{self._capital_deployed:.0f}, "
            f"daily_pnl=₹{self._daily_pnl:.0f}, killed={self._killed})"
        )
