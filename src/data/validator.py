"""Data validation, cleaning, and sanity check module using Polars."""

import polars as pl
from src.utils.logger import logger


class DataValidator:
    """Validator and cleaner for market data DataFrames."""

    @staticmethod
    def validate_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
        """Validate and clean OHLCV DataFrame according to strict quantitative rules.

        Rules:
        1. Drop rows with null values in essential columns.
        2. Remove duplicates based on (timestamp, symbol).
        3. Ensure prices > 0, volume >= 0, high >= low.
        4. Sort chronologically by timestamp.
        5. Ensure UTC timezone.
        """
        if df.is_empty():
            return df

        initial_count = len(df)
        logger.info(f"Validating OHLCV DataFrame with {initial_count} rows...")

        # 1. Drop nulls in core columns
        df = df.drop_nulls(subset=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

        # 2. Sanity checks (prices > 0, volume >= 0, high >= low, open >= low, open <= high, close >= low, close <= high)
        df = df.filter(
            (pl.col("open") > 0)
            & (pl.col("high") > 0)
            & (pl.col("low") > 0)
            & (pl.col("close") > 0)
            & (pl.col("volume") >= 0)
            & (pl.col("high") >= pl.col("low"))
            & (pl.col("high") >= pl.col("open"))
            & (pl.col("high") >= pl.col("close"))
            & (pl.col("low") <= pl.col("open"))
            & (pl.col("low") <= pl.col("close"))
        )

        # 3. Deduplicate by (timestamp, symbol) keeping last or first
        df = df.unique(subset=["timestamp", "symbol"], keep="last")

        # 4. Ensure UTC timezone on timestamp
        df = df.with_columns(
            pl.col("timestamp").dt.replace_time_zone("UTC").alias("timestamp")
        )

        # 5. Sort by timestamp ascending
        df = df.sort("timestamp")

        cleaned_count = len(df)
        dropped_count = initial_count - cleaned_count
        if dropped_count > 0:
            logger.warning(f"Data validation dropped {dropped_count} invalid/duplicate rows ({cleaned_count} retained).")
        else:
            logger.success(f"Data validation passed successfully ({cleaned_count} rows).")

        return df
