"""Portfolio-level position sizing and trading circuit breakers."""

from __future__ import annotations

import math

from src.utils.logger import logger


class PortfolioManager:
    """Size positions with conservative fractional Kelly and enforce portfolio limits."""

    def __init__(
        self,
        max_daily_drawdown_pct: float = 0.03,
        max_global_drawdown_pct: float = 0.15,
        max_open_exposure_pct: float = 0.20,
        target_atr_pct: float = 0.02,
        max_atr_pct: float = 0.10,
    ) -> None:
        limits = (
            max_daily_drawdown_pct,
            max_global_drawdown_pct,
            max_open_exposure_pct,
            target_atr_pct,
            max_atr_pct,
        )
        if any(not math.isfinite(value) or value <= 0 for value in limits):
            raise ValueError("Portfolio limits and ATR targets must be finite and positive.")
        if max_daily_drawdown_pct >= 1 or max_global_drawdown_pct >= 1 or max_open_exposure_pct > 1:
            raise ValueError("Drawdown limits must be below 100% and exposure cannot exceed 100%.")

        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_global_drawdown_pct = max_global_drawdown_pct
        self.max_open_exposure_pct = max_open_exposure_pct
        self.target_atr_pct = target_atr_pct
        self.max_atr_pct = max_atr_pct

    def calculate_position_size(
        self,
        signal_prob: float,
        win_rate: float,
        win_loss_ratio: float,
        current_balance: float,
        atr: float,
        price: float,
        max_kelly_fraction: float = 0.25,
        active_positions: dict[str, dict] | None = None,
    ) -> float:
        """Return units sized by fractional Kelly, ATR normalization, and remaining exposure.

        The model probability is conservatively capped at the observed win rate before
        applying Kelly, preventing an uncalibrated model score from overstating edge.
        """
        values = (signal_prob, win_rate, win_loss_ratio, current_balance, atr, price, max_kelly_fraction)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Position sizing inputs must be finite.")
        if not 0 <= signal_prob <= 1 or not 0 <= win_rate <= 1:
            raise ValueError("signal_prob and win_rate must be between 0 and 1.")
        if win_loss_ratio <= 0 or current_balance <= 0 or atr <= 0 or price <= 0:
            raise ValueError("win_loss_ratio, current_balance, ATR, and price must be positive.")
        if not 0 < max_kelly_fraction <= 1:
            raise ValueError("max_kelly_fraction must be in (0, 1].")

        atr_pct = atr / price
        if atr_pct >= self.max_atr_pct:
            logger.warning(
                f"Position sizing blocked by volatility circuit breaker: ATR/price={atr_pct:.2%} "
                f">= {self.max_atr_pct:.2%}."
            )
            return 0.0

        probability = min(signal_prob, win_rate)
        kelly_full = (probability * win_loss_ratio - (1.0 - probability)) / win_loss_ratio
        fractional_kelly = (kelly_full * max_kelly_fraction) if kelly_full > 1e-12 else 0.0
        atr_scalar = min(1.0, self.target_atr_pct / atr_pct)
        desired_notional = current_balance * fractional_kelly * atr_scalar

        open_notional = sum(
            float(position.get("amount", 0.0)) * float(position.get("entry_price", price))
            for position in (active_positions or {}).values()
        )
        remaining_exposure = max(
            0.0, current_balance * self.max_open_exposure_pct - open_notional
        )
        return max(0.0, min(desired_notional, remaining_exposure) / price)

    def check_risk_limits(
        self,
        current_equity: float,
        peak_equity: float,
        daily_pnl: float,
        total_open_exposure: float = 0.0,
    ) -> tuple[bool, str]:
        """Check daily/global drawdown and total exposure.

        `daily_pnl` is a fractional return (for example, -0.03 means -3%), while
        `total_open_exposure` is the current or proposed notional exposure in currency.
        """
        values = (current_equity, peak_equity, daily_pnl, total_open_exposure)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Risk-check inputs must be finite.")
        if current_equity < 0 or peak_equity < 0 or total_open_exposure < 0:
            raise ValueError("Equity and exposure values cannot be negative.")

        if daily_pnl <= -self.max_daily_drawdown_pct:
            reason = (
                f"Daily drawdown limit reached ({daily_pnl:.2%} <= "
                f"-{self.max_daily_drawdown_pct:.2%})."
            )
            logger.warning(reason)
            return False, reason
        if peak_equity > 0:
            global_drawdown = (peak_equity - current_equity) / peak_equity
            if global_drawdown >= self.max_global_drawdown_pct:
                reason = (
                    f"Global drawdown limit reached ({global_drawdown:.2%} >= "
                    f"{self.max_global_drawdown_pct:.2%})."
                )
                logger.warning(reason)
                return False, reason
        max_exposure = current_equity * self.max_open_exposure_pct
        if total_open_exposure > max_exposure:
            reason = (
                f"Open exposure limit exceeded ({total_open_exposure:.2f} > "
                f"{max_exposure:.2f})."
            )
            logger.warning(reason)
            return False, reason
        return True, "Risk limits satisfied."

    def check_volatility(self, atr: float, price: float) -> tuple[bool, str]:
        """Trip a volatility circuit breaker when ATR is too large relative to price."""
        if not math.isfinite(atr) or not math.isfinite(price) or atr < 0 or price <= 0:
            raise ValueError("ATR must be non-negative and price must be positive and finite.")
        atr_pct = atr / price
        if atr_pct >= self.max_atr_pct:
            reason = f"Volatility circuit breaker active (ATR/price={atr_pct:.2%} >= {self.max_atr_pct:.2%})."
            logger.warning(reason)
            return False, reason
        return True, "Volatility within configured limit."
