"""
RiskManager — enforces hard capital and loss limits before every live trade.

Constraints enforced:
  - max_total_capital: ₹3,00,000 (hard cap across all open positions)
  - max_capital_per_trade: per-trade capital limit (from config.yaml)
  - max_daily_loss: daily loss kill switch (from config.yaml)

Usage:
    risk = RiskManager(broker, max_total_capital=300000, ...)
    ok, reason = risk.check_pre_trade(estimated_capital_required=25000)
    if ok:
        broker.place_order(...)
        risk.record_trade(25000)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class RiskManager:
    broker: object                      # ZerodhaBroker instance
    max_total_capital: float = 300000   # ₹3,00,000 hard cap
    max_capital_per_trade: float = 50000
    max_daily_loss: float = 5000        # absolute value; loss beyond this triggers kill switch

    _capital_deployed: float = field(default=0.0, init=False, repr=False)
    _daily_pnl: float        = field(default=0.0, init=False, repr=False)
    _killed: bool            = field(default=False, init=False, repr=False)

    # ── Pre-trade gate ─────────────────────────────────────────────────

    def check_pre_trade(
        self,
        estimated_capital_required: float,
    ) -> Tuple[bool, str]:
        """
        Run all risk checks before placing a new order.
        Returns (True, "") if trade is allowed.
        Returns (False, reason) if trade must be blocked.
        """
        if self._killed:
            return False, "Kill switch is active — no new trades allowed today."

        # 1. Per-trade capital check
        if estimated_capital_required > self.max_capital_per_trade:
            return False, (
                f"Trade requires ₹{estimated_capital_required:,.0f} which exceeds "
                f"per-trade limit of ₹{self.max_capital_per_trade:,.0f}."
            )

        # 2. Total deployed capital check (₹3L hard cap)
        projected_total = self._capital_deployed + estimated_capital_required
        if projected_total > self.max_total_capital:
            return False, (
                f"Total capital would reach ₹{projected_total:,.0f}, "
                f"exceeding hard cap of ₹{self.max_total_capital:,.0f}."
            )

        # 3. Broker margin check — live available margin must cover the trade
        try:
            available_margin = self.broker.get_available_margin()
            if available_margin < estimated_capital_required:
                return False, (
                    f"Insufficient margin: available=₹{available_margin:,.0f}, "
                    f"required=₹{estimated_capital_required:,.0f}."
                )
        except Exception as e:
            logger.warning("Margin check failed (non-blocking): %s", e)
            # Non-fatal: allow trade but log the warning

        # 4. Daily loss check
        if abs(self._daily_pnl) >= self.max_daily_loss and self._daily_pnl < 0:
            self._killed = True
            return False, (
                f"Daily loss limit of ₹{self.max_daily_loss:,.0f} hit "
                f"(current P&L: ₹{self._daily_pnl:,.0f}). Kill switch engaged."
            )

        return True, ""

    # ── Post-trade recording ───────────────────────────────────────────

    def record_trade(self, capital_used: float) -> None:
        """Call after a successful order placement to track deployed capital."""
        self._capital_deployed += capital_used
        logger.info(
            "Capital recorded: +₹%.0f | total deployed: ₹%.0f / ₹%.0f",
            capital_used, self._capital_deployed, self.max_total_capital,
        )

    def record_pnl(self, pnl_delta: float) -> None:
        """
        Update daily P&L (positive = profit, negative = loss).
        Call on each position close with the realised P&L.
        """
        self._daily_pnl += pnl_delta
        logger.info("Daily P&L updated: ₹%.0f | limit: -₹%.0f", self._daily_pnl, self.max_daily_loss)
        if self._daily_pnl <= -self.max_daily_loss:
            self._killed = True
            logger.critical(
                "KILL SWITCH ENGAGED: daily loss ₹%.0f exceeds limit ₹%.0f",
                self._daily_pnl, self.max_daily_loss,
            )

    def release_capital(self, capital_freed: float) -> None:
        """Call when a position is closed to free up tracked capital."""
        self._capital_deployed = max(0.0, self._capital_deployed - capital_freed)
        logger.info(
            "Capital released: -₹%.0f | total deployed: ₹%.0f",
            capital_freed, self._capital_deployed,
        )

    # ── Status ─────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return current risk state as a dict (for dashboards/logging)."""
        return {
            "capital_deployed": self._capital_deployed,
            "capital_remaining": self.max_total_capital - self._capital_deployed,
            "daily_pnl":         self._daily_pnl,
            "kill_switch":       self._killed,
            "max_total_capital": self.max_total_capital,
            "max_daily_loss":    self.max_daily_loss,
        }

    def reset_daily(self) -> None:
        """Reset daily counters. Call at start of each trading day."""
        self._daily_pnl        = 0.0
        self._killed           = False
        self._capital_deployed = 0.0
        logger.info("RiskManager daily counters reset.")
