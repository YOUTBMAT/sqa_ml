"""Tests for order finite-state transitions and execution integration."""

from datetime import datetime, timezone, timedelta

import pytest

from src.execution.engine import ExecutionEngine
from src.execution.fsm import InvalidStateTransitionError, OrderFSM, OrderState
from src.execution.paper import PaperTrader
from src.risk.manager import RiskManager


def test_order_fsm_allows_valid_transitions_and_records_history() -> None:
    fsm = OrderFSM("order-1", "ETH/USDT", "buy", 1.0, 3000.0)

    fsm.transition_to(OrderState.OPEN, "accepted")
    fsm.transition_to(OrderState.PARTIALLY_FILLED, "partial fill")
    fsm.transition_to(OrderState.FILLED, "filled")

    assert fsm.state is OrderState.FILLED
    assert fsm.is_terminal()
    assert not fsm.is_active()
    assert [event["state"] for event in fsm.history] == [
        "PENDING_SUBMIT",
        "OPEN",
        "PARTIALLY_FILLED",
        "FILLED",
    ]
    assert fsm.history[-1]["reason"] == "filled"


@pytest.mark.parametrize(
    "terminal_state",
    [OrderState.FILLED, OrderState.REJECTED, OrderState.EXPIRED, OrderState.FAILED],
)
def test_order_fsm_terminal_states(terminal_state: OrderState) -> None:
    fsm = OrderFSM("order-2", "ETH/USDT", "buy", 1.0)
    fsm.transition_to(terminal_state)

    assert fsm.is_terminal()
    assert not fsm.is_active()


def test_order_fsm_rejects_invalid_transition() -> None:
    fsm = OrderFSM("order-3", "ETH/USDT", "buy", 1.0)
    fsm.transition_to(OrderState.REJECTED, "invalid order")

    with pytest.raises(InvalidStateTransitionError, match="REJECTED -> FILLED"):
        fsm.transition_to(OrderState.FILLED)

    with pytest.raises(InvalidStateTransitionError, match="REJECTED -> OPEN"):
        fsm.transition_to(OrderState.OPEN)


def test_order_fsm_serialization_round_trip() -> None:
    fsm = OrderFSM("order-4", "ETH/USDT", "sell", 2.0, 3000.0)
    fsm.transition_to(OrderState.OPEN)
    restored = OrderFSM.from_dict(fsm.to_dict())

    assert restored.order_id == fsm.order_id
    assert restored.state is OrderState.OPEN
    assert restored.history == fsm.history


def test_paper_trader_order_lifecycle_closes_on_stop_loss(tmp_path) -> None:
    entry_bar_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trader = PaperTrader(
        initial_balance=10000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        load_saved_state=False,
        state_dir=str(tmp_path),
    )
    entry_order = trader.execute_paper_order(
        "ETH/USDT",
        "buy",
        1.0,
        3000.0,
        stop_loss=2800.0,
        take_profit=3300.0,
        entry_bar_ts=entry_bar_ts,
    )
    entry_fsm = trader.order_fsms[entry_order["order_id"]]
    assert entry_fsm.state is OrderState.FILLED
    assert [item["state"] for item in entry_fsm.history] == ["PENDING_SUBMIT", "OPEN", "FILLED"]

    events = trader.update_positions(
        {
            "ETH/USDT": [
                {
                    "timestamp": entry_bar_ts + timedelta(hours=1),
                    "open": 3000.0,
                    "high": 3050.0,
                    "low": 2750.0,
                    "close": 2800.0,
                }
            ]
        }
    )

    assert len(events) == 1
    assert events[0]["exit_reason"] == "stop_loss"
    exit_fsm = trader.order_fsms[events[0]["order_id"]]
    assert exit_fsm.state is OrderState.FILLED
    assert exit_fsm.history[-1]["reason"] == "stop_loss"
    assert not trader.has_open_position("ETH/USDT")


def test_paper_trader_persists_rejected_order_fsm(tmp_path) -> None:
    trader = PaperTrader(
        initial_balance=100.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        load_saved_state=False,
        state_dir=str(tmp_path),
    )

    with pytest.raises(ValueError, match="Insufficient paper balance"):
        trader.execute_paper_order("ETH/USDT", "buy", 1.0, 3000.0)

    rejected = next(iter(trader.order_fsms.values()))
    assert rejected.state is OrderState.REJECTED
    assert trader.order_history[-1]["fsm"]["state"] == "REJECTED"
    assert not rejected.is_active()


class _FakeExchange:
    def __init__(self, response: dict) -> None:
        self.response = response

    async def create_order(self, **kwargs) -> dict:
        return self.response


@pytest.mark.asyncio
async def test_execution_engine_tracks_partial_exchange_order(tmp_path) -> None:
    engine = ExecutionEngine(mode="LIVE", load_saved_state=False, state_dir=str(tmp_path))
    engine.exchange = _FakeExchange(
        {"id": "exchange-1", "status": "open", "amount": 2.0, "filled": 0.5}
    )

    response = await engine.place_order(
        "ETH/USDT",
        "buy",
        2.0,
        3000.0,
        RiskManager(),
        current_equity=10000.0,
        peak_equity=10000.0,
    )

    assert response["fsm_state"] == OrderState.PARTIALLY_FILLED.value
    assert engine.order_fsms["exchange-1"].state is OrderState.PARTIALLY_FILLED
    assert engine.order_fsms["exchange-1"].is_active()


@pytest.mark.asyncio
async def test_execution_engine_tracks_orphaned_exchange_response(tmp_path) -> None:
    engine = ExecutionEngine(mode="LIVE", load_saved_state=False, state_dir=str(tmp_path))
    engine.exchange = _FakeExchange({"status": "open"})

    with pytest.raises(RuntimeError, match="order is orphaned"):
        await engine.place_order(
            "ETH/USDT",
            "buy",
            1.0,
            3000.0,
            RiskManager(),
            current_equity=10000.0,
            peak_equity=10000.0,
        )

    orphan = next(iter(engine.orphan_orders.values()))
    assert orphan.state is OrderState.FAILED
    assert not orphan.is_active()


@pytest.mark.asyncio
async def test_execution_engine_marks_unknown_exchange_status_orphaned(tmp_path) -> None:
    engine = ExecutionEngine(mode="LIVE", load_saved_state=False, state_dir=str(tmp_path))
    engine.exchange = _FakeExchange({"id": "exchange-unknown", "status": "unrecognized"})

    with pytest.raises(RuntimeError, match="needs reconciliation"):
        await engine.place_order(
            "ETH/USDT",
            "buy",
            1.0,
            3000.0,
            RiskManager(),
            current_equity=10000.0,
            peak_equity=10000.0,
        )

    assert engine.orphan_orders["exchange-unknown"].state is OrderState.FAILED
