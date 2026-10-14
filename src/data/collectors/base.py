"""Abstract base class for asynchronous market data collectors."""

from abc import ABC, abstractmethod
from datetime import datetime
import polars as pl


class BaseCollector(ABC):
    """Abstract interface for all market data collectors (Crypto, Stocks, etc.)."""

    @abstractmethod
    async def fetch_ohlcv_historical(
        self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime
    ) -> pl.DataFrame:
        """Fetch historical OHLCV candles between start_time and end_time.

        Returns a Polars DataFrame with columns:
        [timestamp, symbol, asset_class, timeframe, open, high, low, close, volume, trades_count]
        """
        pass

    @abstractmethod
    async def fetch_market_metrics(self, symbol: str) -> pl.DataFrame:
        """Fetch dynamic market metrics (e.g., funding rate, open interest).

        Returns a Polars DataFrame with columns:
        [timestamp, symbol, metric_name, metric_value]
        """
        pass
