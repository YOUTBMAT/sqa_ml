"""Finite state machine for order lifecycle tracking."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar


class OrderState(str, Enum):
    """States an order can occupy during submission and execution."""

    PENDING_SUBMIT = "PENDING_SUBMIT"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELING = "CANCELING"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class InvalidStateTransitionError(ValueError):
    """Raised when an order attempts a transition not allowed by its lifecycle."""


@dataclass
class OrderFSM:
    """Track one order's identity, state and transition history."""

    order_id: str
    symbol: str
    side: str
    amount: float
    price: float | None = None
    state: OrderState = OrderState.PENDING_SUBMIT
    history: list[dict[str, object]] = field(default_factory=list)

    TRANSITIONS: ClassVar[dict[OrderState, frozenset[OrderState]]] = {
        OrderState.PENDING_SUBMIT: frozenset(
            {
                OrderState.OPEN,
                OrderState.PARTIALLY_FILLED,
                OrderState.FILLED,
                OrderState.EXPIRED,
                OrderState.REJECTED,
                OrderState.FAILED,
            }
        ),
        OrderState.OPEN: frozenset(
            {
                OrderState.PARTIALLY_FILLED,
                OrderState.FILLED,
                OrderState.CANCELING,
                OrderState.EXPIRED,
                OrderState.FAILED,
            }
        ),
        OrderState.PARTIALLY_FILLED: frozenset(
            {
                OrderState.FILLED,
                OrderState.CANCELING,
                OrderState.EXPIRED,
                OrderState.FAILED,
            }
        ),
        OrderState.CANCELING: frozenset(
            {OrderState.FILLED, OrderState.EXPIRED, OrderState.FAILED}
        ),
        OrderState.FILLED: frozenset(),
        OrderState.EXPIRED: frozenset(),
        OrderState.REJECTED: frozenset(),
        OrderState.FAILED: frozenset(),
    }

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append(
                {
                    "state": self.state.value,
                    "timestamp": datetime.now(timezone.utc),
                    "reason": "Order created",
                }
            )

    def transition_to(self, new_state: OrderState, reason: str = "") -> None:
        """Move the order to a valid next state and append an audit record."""
        if new_state not in self.TRANSITIONS[self.state]:
            raise InvalidStateTransitionError(
                f"Invalid order transition {self.state.value} -> {new_state.value} "
                f"for order {self.order_id}."
            )
        self.state = new_state
        self.history.append(
            {
                "state": new_state.value,
                "timestamp": datetime.now(timezone.utc),
                "reason": reason,
            }
        )

    def is_terminal(self) -> bool:
        """Return whether this order can no longer transition."""
        return self.state in {
            OrderState.FILLED,
            OrderState.EXPIRED,
            OrderState.REJECTED,
            OrderState.FAILED,
        }

    def is_active(self) -> bool:
        """Return whether the order is awaiting completion or cancellation."""
        return self.state in {
            OrderState.PENDING_SUBMIT,
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
            OrderState.CANCELING,
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize the FSM for persistence alongside the order record."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "amount": self.amount,
            "price": self.price,
            "state": self.state.value,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "OrderFSM":
        """Restore an FSM serialized by :meth:`to_dict`."""
        history = data.get("history", [])
        if not isinstance(history, list):
            raise ValueError("Serialized order history must be a list.")
        parsed_history: list[dict[str, object]] = []
        for entry in history:
            if not isinstance(entry, dict):
                raise ValueError("Serialized order history entries must be objects.")
            item = dict(entry)
            timestamp = item.get("timestamp")
            if isinstance(timestamp, str):
                item["timestamp"] = datetime.fromisoformat(timestamp)
            parsed_history.append(item)
        return cls(
            order_id=str(data["order_id"]),
            symbol=str(data["symbol"]),
            side=str(data["side"]),
            amount=float(data["amount"]),
            price=float(data["price"]) if data.get("price") is not None else None,
            state=OrderState(str(data["state"])),
            history=parsed_history,
        )
