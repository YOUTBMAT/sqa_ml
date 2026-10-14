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


def fetch_candles_from_db(symbol: str, timeframe: str, limit_days: int | None = None) -> pl.DataFrame:
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
            rows = result.fetchall()
            if not rows:
                logger.warning(f"No records found in database for {symbol} ({timeframe}).")
                return pl.DataFrame()

            # Convert to Polars DataFrame
            data = [dict(row._mapping) for row in rows]
            df = pl.DataFrame(data)
            logger.success(f"Fetched {len(df)} candles from database for {symbol}.")
            return df
    except Exception as e:
        logger.error(f"Failed to fetch candles from database for {symbol}: {e}")
        return pl.DataFrame()


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
        df_candles = fetch_candles_from_db(symbol, args.timeframe, limit_days=args.limit_days)

        if df_candles.is_empty():
            logger.warning(f"Skipping {symbol}: No candle data available.")
            continue

        # Run feature pipeline
        df_features = pipeline.run(df_candles)

        if df_features.is_empty():
            logger.warning(f"Skipping {symbol}: Feature pipeline returned empty DataFrame.")
            continue

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
