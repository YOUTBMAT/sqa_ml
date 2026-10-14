"""Unit tests for PaperTrader and ExecutionEngine."""

import pytest
import asyncio
from src.execution.paper import PaperTrader
from src.execution.engine import ExecutionEngine
from src.risk.manager import RiskManager


def test_paper_trading_execution() -> None:
    """Test PaperTrader order execution, fee deduction, slippage, and position tracking."""
    trader = PaperTrader(initial_balance=10000.0, fee_rate=0.001, slippage_rate=0.0005, load_saved_state=False)

    symbol = "BTC/USDT"
    current_price = 50000.0
    amount = 0.1

    order = trader.execute_paper_order(symbol, "buy", amount, current_price, stop_loss=48000.0, take_profit=55000.0)

    assert order["status"] == "filled"
    assert symbol in trader.positions
    assert trader.balance < 10000.0  # Balance reduced by cost + fee

    # Test position update & SL/TP trigger
    # Price drop below stop_loss (48000)
    events = trader.update_positions({symbol: 47500.0})
    assert len(events) == 1
    assert events[0]["exit_reason"] == "stop_loss"
    assert symbol not in trader.positions


@pytest.mark.asyncio
async def test_execution_engine_risk_integration() -> None:
    """Test ExecutionEngine integrates with RiskManager to block orders when circuit breaker triggers."""
    async with ExecutionEngine(mode="PAPER", load_saved_state=False) as engine:
        risk_manager = RiskManager(max_daily_drawdown_pct=0.05)

        # Normal condition -> order should pass
        order = await engine.place_order(
            symbol="BTC/USDT",
            side="buy",
            amount=0.1,
            current_price=50000.0,
            risk_manager=risk_manager,
            peak_equity=10000.0,
            daily_pnl_pct=-0.01,
        )
        assert order["status"] == "filled"

        # Breached condition (daily drawdown -6%) -> order should raise PermissionError
        with pytest.raises(PermissionError):
            await engine.place_order(
                symbol="BTC/USDT",
                side="buy",
                amount=0.1,
                current_price=50000.0,
                risk_manager=risk_manager,
                peak_equity=10000.0,
                daily_pnl_pct=-0.06,
            )
