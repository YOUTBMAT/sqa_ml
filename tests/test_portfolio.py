"""Tests for fractional-Kelly sizing and portfolio-level risk controls."""

import pytest

from src.execution.engine import ExecutionEngine
from src.execution.paper import PaperTrader
from src.risk.portfolio import PortfolioManager
from src.risk.manager import RiskManager


def test_fractional_kelly_position_size_tracks_signal_probability() -> None:
    manager = PortfolioManager()
    lower_probability = manager.calculate_position_size(
        signal_prob=0.50,
        win_rate=0.70,
        win_loss_ratio=1.5,
        current_balance=10_000.0,
        atr=2.0,
        price=100.0,
    )
    higher_probability = manager.calculate_position_size(
        signal_prob=0.70,
        win_rate=0.70,
        win_loss_ratio=1.5,
        current_balance=10_000.0,
        atr=2.0,
        price=100.0,
    )

    assert lower_probability == pytest.approx(4.1666666667)
    assert higher_probability == pytest.approx(12.5)


def test_position_sizing_uses_conservative_win_rate_cap_and_zero_edge() -> None:
    manager = PortfolioManager()
    capped_size = manager.calculate_position_size(
        signal_prob=0.8,
        win_rate=0.5,
        win_loss_ratio=1.0,
        current_balance=10_000.0,
        atr=2.0,
        price=100.0,
    )
    no_edge_size = manager.calculate_position_size(
        signal_prob=0.4,
        win_rate=0.4,
        win_loss_ratio=1.5,
        current_balance=10_000.0,
        atr=2.0,
        price=100.0,
    )

    assert capped_size == 0.0
    assert no_edge_size == 0.0


def test_high_atr_reduces_position_size_and_trips_volatility_circuit() -> None:
    manager = PortfolioManager()
    normal_volatility = manager.calculate_position_size(
        signal_prob=0.7,
        win_rate=0.7,
        win_loss_ratio=1.5,
        current_balance=10_000.0,
        atr=2.0,
        price=100.0,
    )
    high_volatility = manager.calculate_position_size(
        signal_prob=0.7,
        win_rate=0.7,
        win_loss_ratio=1.5,
        current_balance=10_000.0,
        atr=4.0,
        price=100.0,
    )

    assert high_volatility == pytest.approx(normal_volatility / 2)
    assert manager.calculate_position_size(
        signal_prob=0.9,
        win_rate=0.9,
        win_loss_ratio=1.5,
        current_balance=10_000.0,
        atr=10.0,
        price=100.0,
    ) == 0.0
    assert manager.check_volatility(10.0, 100.0)[0] is False


def test_position_size_respects_remaining_exposure_and_margin() -> None:
    manager = PortfolioManager()
    size = manager.calculate_position_size(
        signal_prob=0.9,
        win_rate=0.9,
        win_loss_ratio=1.5,
        current_balance=10_000.0,
        atr=2.0,
        price=100.0,
        active_positions={
            "BTC/USDT": {"amount": 15.0, "entry_price": 100.0},
        },
    )

    assert size == pytest.approx(5.0)
    assert (size + 15.0) * 100.0 <= 10_000.0 * manager.max_open_exposure_pct


def test_risk_limits_block_daily_drawdown_and_excess_exposure() -> None:
    manager = PortfolioManager()
    allowed, reason = manager.check_risk_limits(9_700.0, 10_000.0, -0.03)
    assert not allowed
    assert "Daily drawdown" in reason

    allowed, reason = manager.check_risk_limits(
        10_000.0, 10_000.0, 0.0, total_open_exposure=2_001.0
    )
    assert not allowed
    assert "exposure" in reason.lower()


def test_paper_trader_enforces_exposure_cap_for_direct_orders(tmp_path) -> None:
    trader = PaperTrader(
        initial_balance=10_000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        load_saved_state=False,
        state_dir=str(tmp_path),
        portfolio_manager=PortfolioManager(),
    )

    with pytest.raises(PermissionError, match="exposure"):
        trader.execute_paper_order("ETH/USDT", "buy", 1.1, 2_000.0)
    assert not trader.has_open_position("ETH/USDT")


@pytest.mark.asyncio
async def test_execution_engine_blocks_opening_at_daily_drawdown_and_volatility(tmp_path) -> None:
    manager = PortfolioManager()
    async with ExecutionEngine(
        mode="PAPER",
        load_saved_state=False,
        state_dir=str(tmp_path),
        portfolio_manager=manager,
    ) as engine:
        with pytest.raises(PermissionError, match="Daily drawdown"):
            await engine.place_order(
                symbol="ETH/USDT",
                side="buy",
                amount=0.01,
                current_price=2_000.0,
                risk_manager=RiskManager(),
                daily_pnl_pct=-0.03,
                current_equity=10_000.0,
                peak_equity=10_000.0,
                atr_value=20.0,
            )

        with pytest.raises(PermissionError, match="Volatility circuit breaker"):
            await engine.place_order(
                symbol="ETH/USDT",
                side="buy",
                amount=0.01,
                current_price=2_000.0,
                risk_manager=RiskManager(),
                daily_pnl_pct=0.0,
                current_equity=10_000.0,
                peak_equity=10_000.0,
                atr_value=250.0,
            )

        assert not engine.paper_trader.has_open_position("ETH/USDT")


@pytest.mark.asyncio
async def test_execution_engine_blocks_order_over_exposure_cap(tmp_path) -> None:
    manager = PortfolioManager()
    async with ExecutionEngine(
        mode="PAPER",
        load_saved_state=False,
        state_dir=str(tmp_path),
        portfolio_manager=manager,
    ) as engine:
        with pytest.raises(PermissionError, match="exposure"):
            await engine.place_order(
                symbol="ETH/USDT",
                side="buy",
                amount=1.1,
                current_price=2_000.0,
                risk_manager=RiskManager(),
                current_equity=10_000.0,
                peak_equity=10_000.0,
                daily_pnl_pct=0.0,
            )
        assert not engine.paper_trader.has_open_position("ETH/USDT")
