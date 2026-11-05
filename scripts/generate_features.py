"""CLI script for generating and persisting advanced quantitative features from database OHLCV data."""

import argparse
import sys
import time
from datetime import datetime, timezone, timedelta
import polars as pl
from sqlalchemy import text

from src.data.database import init_db, get_db_session
from src.features.pipeline import FeaturePipeline
from src.features.storage import FeatureStorage
from src.utils.logger import logger


def fetch_candles_from_db(
    symbol: str,
    timeframe: str,
    limit_days: int | None = None,
    chunk_size: int = 10_000,
):
    """Fetch OHLCV candles from TimescaleDB for a given symbol and timeframe."""
    logger.info(f"Fetching candles for {symbol} ({timeframe}) from database...")
    
    query_str = """
        SELECT timestamp, symbol, asset_class, timeframe, open, high, low, close, volume, trades_count
        FROM ohlcv_candles
        WHERE symbol = :symbol AND timeframe = :timeframe
    """
    params = {"symbol": symbol, "timeframe": timeframe}

    if limit_days:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=limit_days)
        query_str += " AND timestamp >= :cutoff_date"
        params["cutoff_date"] = cutoff_date

    query_str += " ORDER BY timestamp ASC;"

    try:
        with get_db_session() as session:
            result = session.execute(text(query_str), params)
            total_rows = 0
            yielded = False
            while True:
                rows = result.fetchmany(chunk_size)
                if not rows:
                    break
                yielded = True
                data = [dict(row._mapping) for row in rows]
                chunk = pl.DataFrame(data)
                total_rows += len(chunk)
                yield chunk
            if not yielded:
                logger.warning(f"No records found in database for {symbol} ({timeframe}).")
            else:
                logger.success(f"Fetched {total_rows} candles from database for {symbol}.")
    except Exception as e:
        logger.error(f"Failed to fetch candles from database for {symbol}: {e}")
        raise


def main() -> None:
    """Parse CLI arguments and run feature generation pipeline."""
    parser = argparse.ArgumentParser(description="SQA-ML Feature Generation CLI")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT"],
        help="List of symbols to process (e.g., BTC/USDT ETH/USDT AAPL)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candle timeframe (e.g., 1m, 1h, 1d)",
    )
    parser.add_argument(
        "--d-value",
        type=float,
        default=0.4,
        help="Fractional differentiation order 'd' (default: 0.4)",
    )
    parser.add_argument(
        "--limit-days",
        type=int,
        default=None,
        help="Optional limit on past days of data to process",
    )

    args = parser.parse_args()

    logger.info("=== INITIALIZING SQA-ML FEATURE GENERATION CLI ===")

    try:
        init_db()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)

    pipeline = FeaturePipeline(d_fractional=args.d_value)
    storage = FeatureStorage()
    start_wall_time = time.time()
    total_processed = 0

    for symbol in args.symbols:
        logger.info(f"Processing features for symbol: {symbol} ({args.timeframe})")
        chunks = fetch_candles_from_db(symbol, args.timeframe, limit_days=args.limit_days)
        processed_chunks = []
        history = pl.DataFrame()
        last_timestamp = None
        for df_candles in chunks:
            if df_candles.is_empty():
                continue
            combined = (
                pl.concat([history, df_candles], how="vertical_relaxed")
                if not history.is_empty()
                else df_candles
            )
            processed = pipeline.run(combined)
            if last_timestamp is not None:
                processed = processed.filter(pl.col("timestamp") > last_timestamp)
            if not processed.is_empty():
                processed_chunks.append(processed)
                last_timestamp = processed["timestamp"].max()
            history = combined.tail(250)

        if not processed_chunks:
            logger.warning(f"Skipping {symbol}: No candle data available.")
            continue

        df_features = pl.concat(processed_chunks, how="vertical_relaxed")

        # Save to Parquet
        storage.save_features(df_features, symbol, args.timeframe)
        total_processed += len(df_features)

    elapsed = time.time() - start_wall_time
    logger.success(
        f"=== FEATURE GENERATION COMPLETE ===\n"
        f"Total Feature Rows Generated: {total_processed}\n"
        f"Elapsed Time: {elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    main()
