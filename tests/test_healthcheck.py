"""Unit tests for healthcheck functionality."""

import pytest
from unittest.mock import patch, MagicMock
from src.utils.healthcheck import check_database, check_timescaledb_extension, check_ccxt_api


@patch("src.utils.healthcheck.engine")
def test_check_database_success(mock_engine: MagicMock) -> None:
    """Test successful database healthcheck."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.scalar.return_value = 1
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    assert check_database() is True


@patch("src.utils.healthcheck.engine")
def test_check_database_failure(mock_engine: MagicMock) -> None:
    """Test database healthcheck failure on exception."""
    mock_engine.connect.side_effect = Exception("Connection refused")

    assert check_database() is False


@patch("src.utils.healthcheck.ccxt")
def test_check_ccxt_api_success(mock_ccxt: MagicMock) -> None:
    """Test successful CCXT API check."""
    mock_exchange = MagicMock()
    mock_exchange.fetch_ticker.return_value = {"last": 65000.0}
    setattr(mock_ccxt, "binance", MagicMock(return_value=mock_exchange))

    assert check_ccxt_api("BTC/USDT") is True


@patch("src.utils.healthcheck.ccxt")
def test_check_ccxt_api_failure(mock_ccxt: MagicMock) -> None:
    """Test CCXT API check failure."""
    mock_exchange = MagicMock()
    mock_exchange.fetch_ticker.side_effect = Exception("Network timeout")
    setattr(mock_ccxt, "binance", MagicMock(return_value=mock_exchange))

    assert check_ccxt_api("BTC/USDT") is False
