"""Data validation, cleaning, and sanity check module using Polars."""

import polars as pl
from datetime import timedelta
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

        # 3. Normalize naive timestamps as UTC and convert aware timestamps to UTC.
        timestamp_dtype = df.schema["timestamp"]
        df = df.with_columns(
            (
                pl.col("timestamp").dt.replace_time_zone("UTC")
                if getattr(timestamp_dtype, "time_zone", None) is None
                else pl.col("timestamp").dt.convert_time_zone("UTC")
            ).alias("timestamp")
        )

        # 4. Deduplicate after normalization so equivalent instants collapse.
        df = df.unique(subset=["timestamp", "symbol"], keep="last")

        # 5. Sort by timestamp ascending
        df = df.sort("timestamp")

        # 6. Report temporal gaps without fabricating candles.
        if "timeframe" in df.columns:
            for symbol, group in df.partition_by("symbol", as_dict=True).items():
                timeframes = group["timeframe"].drop_nulls().unique().to_list()
                if not timeframes:
                    continue
                expected = _timeframe_duration(str(timeframes[0]))
                if expected is None:
                    continue
                timestamps = group["timestamp"].to_list()
                gaps = [
                    timestamps[i] - timestamps[i - 1]
                    for i in range(1, len(timestamps))
                    if timestamps[i] - timestamps[i - 1] > expected
                ]
                if gaps:
                    logger.warning(
                        f"Detected {len(gaps)} temporal gaps for {symbol} "
                        f"(expected interval {expected})."
                    )

        cleaned_count = len(df)
        dropped_count = initial_count - cleaned_count
        if dropped_count > 0:
            logger.warning(f"Data validation dropped {dropped_count} invalid/duplicate rows ({cleaned_count} retained).")
        else:
            logger.success(f"Data validation passed successfully ({cleaned_count} rows).")

        return df


def _timeframe_duration(timeframe: str) -> timedelta | None:
    units = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    if not timeframe or timeframe[-1].lower() not in units:
        return None
    try:
        return timedelta(seconds=int(timeframe[:-1]) * units[timeframe[-1].lower()])
    except ValueError:
        return None
