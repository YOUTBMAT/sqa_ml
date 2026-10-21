import pytest

from scripts.send_telegram_status import format_status_message


def test_status_message_reports_balance_lock_and_open_positions() -> None:
    message = format_status_message(
        {
            "balance": 10_500.0,
            "initial_balance": 10_000.0,
            "positions": {
                "ETH/USDT": {
                    "side": "short",
                    "amount": 0.5,
                    "entry_price": 3000.0,
                }
            },
        },
        "active",
        "ALERT — severe drift detected",
        {"active": True, "reason": "holdout failed"},
    )

    assert "Accumulated PnL: *+5.00%*" in message
    assert "Safety lock: ACTIVE" in message
    assert "ETH/USDT" in message
    assert "amount 0\\.5" in message


def test_status_message_rejects_zero_initial_balance() -> None:
    with pytest.raises(RuntimeError, match="initial_balance"):
        format_status_message(
            {"balance": 100.0, "initial_balance": 0.0, "positions": {}},
            "active",
            "No alert",
            None,
        )
