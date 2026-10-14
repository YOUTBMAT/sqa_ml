"""Unit tests for RiskManager (position sizing, dynamic stops, circuit breaker)."""

import pytest
from src.risk.manager import RiskManager


def test_kelly_position_sizing() -> None:
    """Test RiskManager calculates valid position size using fractional Kelly criterion."""
    risk_manager = RiskManager(max_portfolio_risk_pct=0.02, kelly_fraction=0.25)

    account_balance = 10000.0
    current_price = 100.0
    win_probability = 0.60
    reward_risk_ratio = 1.5
    volatility = 0.02

    size = risk_manager.calculate_position_size(
        account_balance, current_price, win_probability, reward_risk_ratio, volatility
    )

    assert size > 0.0
    position_value = size * current_price
    assert position_value <= account_balance * 0.02


def test_circuit_breaker_trigger() -> None:
    """Test RiskManager circuit breaker triggers when drawdown limits are breached."""
    risk_manager = RiskManager(max_daily_drawdown_pct=0.05, max_global_drawdown_pct=0.15)

    # Normal state -> should return True
    assert risk_manager.check_circuit_breaker(current_equity=9800.0, peak_equity=10000.0, daily_pnl_pct=-0.02) is True

    # Daily drawdown breached (-6%) -> should return False
    assert risk_manager.check_circuit_breaker(current_equity=9400.0, peak_equity=10000.0, daily_pnl_pct=-0.06) is False

    # Global drawdown breached (-18% >= 15%) -> should return False
    assert risk_manager.check_circuit_breaker(current_equity=8200.0, peak_equity=10000.0, daily_pnl_pct=-0.01) is False


def test_dynamic_stops() -> None:
    """Test RiskManager calculates correct ATR-based stops for Long and Short positions."""
    risk_manager = RiskManager()

    current_price = 100.0
    atr = 2.0

    # Long position stops
    sl_long, tp_long = risk_manager.calculate_dynamic_stops(current_price, "long", atr, sl_atr_mult=2.0, tp_atr_mult=3.0)
    assert sl_long == 96.0
    assert tp_long == 106.0

    # Short position stops
    sl_short, tp_short = risk_manager.calculate_dynamic_stops(current_price, "short", atr, sl_atr_mult=2.0, tp_atr_mult=3.0)
    assert sl_short == 104.0
    assert tp_short == 94.0
