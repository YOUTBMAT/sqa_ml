"""Unit tests for data validation, storage, and collectors."""

import pytest
from datetime import datetime, timezone
import polars as pl
from unittest.mock import patch, MagicMock

from src.data.validator import DataValidator
from src.data.storage import DataStorage


def test_data_validator_valid_candles() -> None:
    """Test DataValidator correctly cleans and validates OHLCV data."""
    data = {
        "timestamp": [
            datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc), # duplicate
        ],
        "symbol": ["BTC/USDT", "BTC/USDT", "BTC/USDT"],
        "asset_class": ["CRYPTO", "CRYPTO", "CRYPTO"],
        "timeframe": ["1h", "1h", "1h"],
        "open": [100.0, 105.0, 100.0],
        "high": [110.0, 112.0, 110.0],
        "low": [95.0, 101.0, 95.0],
        "close": [105.0, 108.0, 105.0],
        "volume": [1000.0, 1500.0, 1000.0],
        "trades_count": [10, 15, 10],
    }
    df = pl.DataFrame(data)
    validated_df = DataValidator.validate_ohlcv(df)

    # Should remove duplicate and keep 2 rows, sorted by timestamp
    assert len(validated_df) == 2
    assert validated_df["timestamp"][0] == datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def test_data_validator_sanity_checks() -> None:
    """Test DataValidator filters out invalid candles (e.g., high < low, negative prices)."""
    data = {
        "timestamp": [
            datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc),
        ],
        "symbol": ["BTC/USDT", "BTC/USDT", "BTC/USDT"],
        "asset_class": ["CRYPTO", "CRYPTO", "CRYPTO"],
        "timeframe": ["1h", "1h", "1h"],
        "open": [100.0, -10.0, 100.0],       # negative price invalid
        "high": [110.0, 112.0, 90.0],        # high < low invalid
        "low": [95.0, 101.0, 95.0],
        "close": [105.0, 108.0, 105.0],
        "volume": [1000.0, 1500.0, 1000.0],
        "trades_count": [10, 15, 10],
    }
    df = pl.DataFrame(data)
    validated_df = DataValidator.validate_ohlcv(df)

    # Only row 0 should pass
    assert len(validated_df) == 1
    assert validated_df["open"][0] == 100.0


@patch("src.data.storage.get_db_session")
def test_data_storage_get_latest_timestamp(mock_get_session: MagicMock) -> None:
    """Test retrieving latest timestamp from storage."""
    mock_session = MagicMock()
    mock_session.execute.return_value.scalar.return_value = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    mock_get_session.return_value.__enter__.return_value = mock_session

    latest = DataStorage.get_latest_timestamp("BTC/USDT", "1h")
    assert latest == datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
