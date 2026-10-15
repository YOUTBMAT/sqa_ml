"""Integration tests for BotRunner execution flow."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import polars as pl
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.execution.runner import BotRunner


@pytest.mark.asyncio
@patch("src.labeling.triple_barrier.TripleBarrierLabeler.get_primary_signal")
@patch("src.execution.runner.CryptoCollector")
@patch("src.execution.runner.ModelStorage")
@patch("src.execution.runner.TelegramNotifier")
async def test_bot_runner_single_iteration(
    mock_notifier_cls: MagicMock,
    mock_storage_cls: MagicMock,
    mock_collector_cls: MagicMock,
    mock_primary_signal: MagicMock,
) -> None:
    """Test BotRunner processes a symbol iteration correctly with mocked collector and model."""
    # Mock Notifier
    mock_notifier = MagicMock()
    mock_notifier.send_message = AsyncMock(return_value=True)
    mock_notifier.notify_order_execution = AsyncMock(return_value=True)
    mock_notifier_cls.return_value = mock_notifier

    # Mock Primary Signal to return 1 (Long)
    mock_primary_signal.return_value = pd.Series([1] * 250)

    # Mock ModelStorage & Model (binary classifier returns [p_0, p_1])
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.1, 0.9]]  # High probability for class 1 (p_y1 = 0.9 >= 0.75)
    mock_storage = MagicMock()
    mock_storage.load_model.return_value = (mock_model, {"feature_cols": ["feat_1"]})
    mock_storage_cls.return_value = mock_storage

    # Mock Collector OHLCV candles
    n = 250
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=i) for i in range(n)]
    df_mock = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTC/USDT"] * n,
            "asset_class": ["CRYPTO"] * n,
            "timeframe": ["1h"] * n,
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": [95.0] * n,
            "close": [102.0] * n,
            "volume": [1000.0] * n,
            "trades_count": [10] * n,
        }
    )

    mock_collector = MagicMock()
    mock_collector.__aenter__.return_value.fetch_ohlcv_historical = AsyncMock(return_value=df_mock)
    mock_collector_cls.return_value = mock_collector

    runner = BotRunner(symbols=["BTC/USDT"], timeframe="1h", mode="PAPER", poll_interval_seconds=1, buy_threshold=0.75)

    # Test invoking process_symbol with a mock execution engine
    mock_engine = MagicMock()
    mock_engine.get_account_balance = AsyncMock(return_value={"free": 10000.0})
    mock_engine.place_order = AsyncMock(return_value={"status": "filled"})
    mock_engine.paper_trader.has_open_position.return_value = False

    await runner._process_symbol("BTC/USDT", mock_engine)
    mock_engine.place_order.assert_awaited()
    kwargs = mock_engine.place_order.await_args.kwargs
    # Same barriers as labels/backtest: ATR-14 (true range is 10 on every mock bar), TP 1.5x, SL 1.0x.
    assert kwargs["stop_loss"] == pytest.approx(102.0 - 10.0)
    assert kwargs["take_profit"] == pytest.approx(102.0 + 15.0)
    assert kwargs["entry_bar_ts"] == timestamps[-1]
    assert kwargs["expires_at"] == timestamps[-1] + timedelta(hours=24)
