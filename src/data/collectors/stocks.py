"""Stock market data collector using Yahoo Finance (yfinance)."""

import asyncio
from datetime import datetime, timezone
import yfinance as yf
import polars as pl

from src.data.collectors.base import BaseCollector
from src.utils.logger import logger


class StockCollector(BaseCollector):
    """Collector for stock markets utilizing Yahoo Finance."""

    def __init__(self) -> None:
        pass

    async def fetch_ohlcv_historical(
        self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime
    ) -> pl.DataFrame:
        """Fetch historical stock OHLCV data using yfinance asynchronously."""
        logger.info(f"[YFINANCE] Fetching OHLCV for {symbol} ({timeframe}) from {start_time} to {end_time}")

        # Map timeframe to yfinance interval
        tf_mapping = {
            "1m": "1m",
            "2m": "2m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "60m": "60m",
            "1h": "1h",
            "1d": "1d",
            "1wk": "1wk",
            "1mo": "1mo",
        }
        interval = tf_mapping.get(timeframe, "1d")

        def _download() -> pl.DataFrame:
            # yfinance download
            df_pd = yf.download(
                symbol,
                start=start_time.strftime("%Y-%m-%d"),
                end=end_time.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if df_pd.empty:
                return pl.DataFrame()

            # Reset index to make Date/Datetime a column
            df_pd = df_pd.reset_index()
            # Handle multi-index columns if returned by yfinance
            if isinstance(df_pd.columns, pl.MultiIndex) or hasattr(df_pd.columns, "levels"):
                df_pd.columns = [col[0] for col in df_pd.columns]

            # Rename columns to standard
            column_mapping = {}
            for col in df_pd.columns:
                col_lower = str(col).lower()
                if "date" in col_lower or "datetime" in col_lower or "index" in col_lower:
                    column_mapping[col] = "timestamp"
                elif "open" in col_lower:
                    column_mapping[col] = "open"
                elif "high" in col_lower:
                    column_mapping[col] = "high"
                elif "low" in col_lower:
                    column_mapping[col] = "low"
                elif "close" in col_lower:
                    column_mapping[col] = "close"
                elif "volume" in col_lower:
                    column_mapping[col] = "volume"

            df_pd = df_pd.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required = ["timestamp", "open", "high", "low", "close", "volume"]
            for r in required:
                if r not in df_pd.columns:
                    raise ValueError(f"Missing column {r} in yfinance response for {symbol}")

            df_pl = pl.from_pandas(df_pd[required])
            return df_pl

        try:
            # Run blocking yfinance call in executor
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(None, _download)
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol} via yfinance: {e}")
            raise

        if df.is_empty():
            logger.warning(f"No stock data found for {symbol} in given range.")
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
                    "symbol": pl.String,
                    "asset_class": pl.String,
                    "timeframe": pl.String,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                    "trades_count": pl.Int64,
                }
            )

        # Standardize schema & types
        df = df.with_columns(
            [
                pl.col("timestamp").dt.replace_time_zone("UTC").alias("timestamp"),
                pl.lit(symbol).alias("symbol"),
                pl.lit("STOCK").alias("asset_class"),
                pl.lit(timeframe).alias("timeframe"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
                pl.lit(None, dtype=pl.Int64).alias("trades_count"),
            ]
        ).select(
            [
                "timestamp",
                "symbol",
                "asset_class",
                "timeframe",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trades_count",
            ]
        )

        logger.success(f"Fetched {len(df)} stock candles for {symbol}")
        return df

    async def fetch_market_metrics(self, symbol: str) -> pl.DataFrame:
        """Fetch stock market metrics (not applicable for basic yfinance, returns empty)."""
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
                "symbol": pl.String,
                "metric_name": pl.String,
                "metric_value": pl.Float64,
            }
        )
