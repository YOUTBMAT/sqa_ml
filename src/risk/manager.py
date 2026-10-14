"""Risk management and capital protection module featuring fractional Kelly and circuit breakers."""

import numpy as np
from src.utils.logger import logger


class RiskManager:
    """Manager for position sizing, dynamic stop-loss/take-profit, and portfolio risk controls."""

    def __init__(
        self,
        max_portfolio_risk_pct: float = 0.02,
        max_daily_drawdown_pct: float = 0.05,
        max_global_drawdown_pct: float = 0.15,
        max_total_allocation_pct: float = 0.15,
        kelly_fraction: float = 0.25,
        max_leverage: float = 1.0,
    ) -> None:
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_global_drawdown_pct = max_global_drawdown_pct
        self.max_total_allocation_pct = max_total_allocation_pct
        self.kelly_fraction = kelly_fraction
        self.max_leverage = max_leverage
        self.circuit_breaker_notified = False

    def calculate_position_size(
        self,
        account_balance: float,
        current_price: float,
        win_probability: float,
        reward_risk_ratio: float = 1.5,
        volatility: float = 0.02,
        active_positions: dict[str, dict] | None = None,
    ) -> float:
        """Calculate optimal position size using Fractional Kelly Criterion adjusted for volatility,
        constrained by max portfolio risk per trade and max total combined allocation across open positions (15%).
        """
        logger.info("Calculating position size via Fractional Kelly criterion with multi-asset allocation check...")
        if reward_risk_ratio <= 0 or current_price <= 0 or account_balance <= 0:
            return 0.0

        p = max(0.0, min(1.0, win_probability))
        b = reward_risk_ratio

        # Kelly fraction calculation
        kelly_full = (p * b - (1.0 - p)) / b
        kelly_adjusted = self.kelly_fraction * max(0.0, kelly_full)

        # Volatility adjustment (inverse volatility weighting relative to baseline 2%)
        vol_scalar = 0.02 / max(0.001, volatility)
        target_allocation_pct = min(self.max_leverage, kelly_adjusted * vol_scalar)

        # Cap by max portfolio risk per trade
        allocation_pct = min(self.max_portfolio_risk_pct, target_allocation_pct)

        position_value = account_balance * allocation_pct

        # Multi-asset combined global allocation check (max 15% across all open positions)
        current_allocated_value = sum(
            pos.get("amount", 0.0) * pos.get("entry_price", current_price)
            for pos in (active_positions or {}).values()
        )
        max_allowed_total_value = account_balance * self.max_total_allocation_pct
        remaining_allowed_value = max(0.0, max_allowed_total_value - current_allocated_value)

        if position_value > remaining_allowed_value:
            logger.warning(
                f"Position value {position_value:.2f} restricted by multi-asset limit (Remaining allowed: {remaining_allowed_value:.2f}, Total allocated: {current_allocated_value:.2f})"
            )
            position_value = remaining_allowed_value

        position_size = position_value / current_price

        logger.success(f"Calculated position size: {position_size:.4f} units (allocation value: ${position_value:.2f})")
        return float(position_size)

    def calculate_dynamic_stops(
        self,
        current_price: float,
        position_side: str,
        atr: float,
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 3.0,
    ) -> tuple[float, float]:
        """Calculate dynamic Stop-Loss and Take-Profit price levels based on ATR."""
        logger.info(f"Calculating dynamic ATR stops for {position_side.upper()} position at price {current_price}...")
        if current_price <= 0 or atr <= 0:
            return current_price, current_price

        side = position_side.lower()
        if side == "long":
            stop_loss = current_price - (atr * sl_atr_mult)
            take_profit = current_price + (atr * tp_atr_mult)
        elif side == "short":
            stop_loss = current_price + (atr * sl_atr_mult)
            take_profit = current_price - (atr * tp_atr_mult)
        else:
            raise ValueError(f"Invalid position side: {position_side}. Expected 'long' or 'short'.")

        logger.success(f"Dynamic Stops -> SL: {stop_loss:.4f}, TP: {take_profit:.4f}")
        return float(stop_loss), float(take_profit)

    def check_circuit_breaker(
        self, current_equity: float, peak_equity: float, daily_pnl_pct: float
    ) -> bool:
        """Evaluate system Circuit Breaker (Kill Switch).

        Returns False if drawdown limits are breached (halting trading), True otherwise.
        """
        logger.info("Checking circuit breaker status...")
        
        # Calculate peak drawdown
        if peak_equity > 0:
            current_drawdown = (peak_equity - current_equity) / peak_equity
        else:
            current_drawdown = 0.0

        if current_drawdown >= self.max_global_drawdown_pct:
            logger.critical(f"CIRCUIT BREAKER TRIGGERED: Global drawdown {current_drawdown*100:.2f}% exceeds limit ({self.max_global_drawdown_pct*100}%).")
            return False

        if daily_pnl_pct <= -self.max_daily_drawdown_pct:
            logger.critical(f"CIRCUIT BREAKER TRIGGERED: Daily PnL {daily_pnl_pct*100:.2f}% exceeds limit.")
            return False

        self.circuit_breaker_notified = False
        logger.success("Circuit breaker status: SYSTEM OPERATIONAL.")
        return True
