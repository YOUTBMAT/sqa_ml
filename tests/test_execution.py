"""Unit tests for PaperTrader and ExecutionEngine."""

import pytest
from datetime import datetime, timezone, timedelta

from src.execution.paper import PaperTrader
from src.execution.engine import ExecutionEngine
from src.risk.manager import RiskManager

T0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)


def _bar(hours: int, o: float, h: float, l: float, c: float) -> dict:
    return {"timestamp": T0 + timedelta(hours=hours), "open": o, "high": h, "low": l, "close": c}


def _long(tmp_path) -> PaperTrader:
    trader = PaperTrader(initial_balance=10000.0, fee_rate=0.001, slippage_rate=0.0005, load_saved_state=False, state_dir=str(tmp_path))
    trader.execute_paper_order(
        "BTC/USDT", "buy", 0.1, 50000.0, stop_loss=49000.0, take_profit=51500.0,
        entry_bar_ts=T0, expires_at=T0 + timedelta(hours=24),
    )
    return trader


def test_paper_trading_execution(tmp_path) -> None:
    """Order fill deducts cost and fee; the stop is detected on a later bar's low and fills at the stop."""
    trader = _long(tmp_path)
    assert trader.has_open_position("BTC/USDT")
    assert trader.balance < 10000.0

    # The entry bar itself is ignored even though its low is below the stop.
    assert trader.update_positions({"BTC/USDT": [_bar(0, 50000, 50000, 48000, 50000)]}) == []

    events = trader.update_positions({"BTC/USDT": [_bar(0, 50000, 50000, 48000, 50000), _bar(1, 50000, 50200, 48900, 49500)]})
    assert len(events) == 1 and events[0]["exit_reason"] == "stop_loss"
    assert events[0]["execution_price"] == pytest.approx(49000.0 * (1 - 0.0005))
    assert not trader.has_open_position("BTC/USDT")


def test_paper_stop_wins_ambiguous_bar_and_gap_fills_at_open(tmp_path) -> None:
    trader = _long(tmp_path)
    events = trader.update_positions({"BTC/USDT": [_bar(1, 50000, 52000, 48000, 50000)]})
    assert events[0]["exit_reason"] == "stop_loss"

    trader = _long(tmp_path)
    events = trader.update_positions({"BTC/USDT": [_bar(1, 52000, 52500, 51800, 52200)]})
    assert events[0]["exit_reason"] == "take_profit"
    assert events[0]["execution_price"] == pytest.approx(52000.0 * (1 - 0.0005))


def test_paper_time_barrier_closes_at_expiry_bar_close(tmp_path) -> None:
    trader = _long(tmp_path)
    bars = [_bar(h, 50000, 50100, 49900, 50000 + h) for h in range(1, 30)]
    events = trader.update_positions({"BTC/USDT": bars})
    assert events[0]["exit_reason"] == "TIME_BARRIER_24H"
    assert events[0]["execution_price"] == pytest.approx((50000 + 24) * (1 - 0.0005))


def test_paper_state_round_trip_keeps_barrier_times(tmp_path) -> None:
    _long(tmp_path)
    reloaded = PaperTrader(load_saved_state=True, state_dir=str(tmp_path))
    pos = reloaded.positions["BTC/USDT"]
    assert pos["entry_bar_ts"] == T0 and pos["expires_at"] == T0 + timedelta(hours=24)


def test_mark_to_market_equity_long_and_short(tmp_path) -> None:
    trader = PaperTrader(initial_balance=10000.0, fee_rate=0.0, slippage_rate=0.0, load_saved_state=False, state_dir=str(tmp_path))
    trader.execute_paper_order("ETH/USDT", "sell", 1.0, 2000.0)  # short proceeds are a liability, not profit
    assert trader.balance == pytest.approx(12000.0)
    assert trader.mark_equity({"ETH/USDT": 2000.0}) == pytest.approx(10000.0)
    assert trader.mark_equity({"ETH/USDT": 2100.0}) == pytest.approx(9900.0)
    assert trader.peak_equity == pytest.approx(10000.0)
    assert trader.mark_equity({"ETH/USDT": 1800.0}) == pytest.approx(10200.0)
    assert PaperTrader(load_saved_state=True, state_dir=str(tmp_path)).peak_equity == pytest.approx(10200.0)


@pytest.mark.asyncio
async def test_execution_engine_risk_integration(tmp_path) -> None:
    """Test ExecutionEngine integrates with RiskManager to block orders when circuit breaker triggers."""
    async with ExecutionEngine(mode="PAPER", load_saved_state=False, state_dir=str(tmp_path)) as engine:
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
