"""CLI script to trigger market data collection and ETL pipeline execution."""

import asyncio
import argparse
import sys
import time
from datetime import datetime, timezone, timedelta

from src.data.database import init_db
from src.data.collectors.crypto import CryptoCollector
from src.data.collectors.stocks import StockCollector
from src.data.pipeline import ETLPipeline
from src.utils.logger import logger


async def main() -> None:
    """Parse CLI arguments and run ETL pipeline for specified symbols."""
    parser = argparse.ArgumentParser(description="SQA-ML Market Data Collector CLI")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT"],
        help="List of symbols to collect (e.g., BTC/USDT ETH/USDT AAPL TSLA)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candle timeframe (e.g., 1m, 1h, 1d)",
    )
    parser.add_argument(
        "--asset-class",
        type=str,
        choices=["CRYPTO", "STOCK"],
        default="CRYPTO",
        help="Asset class type: CRYPTO or STOCK",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Exchange ID for crypto (e.g., binance, coinbase)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of historical days to fetch if no prior data exists",
    )
    parser.add_argument(
        "--market-type",
        type=str,
        choices=["spot", "future"],
        default="spot",
        help="Market type: spot or future",
    )

    args = parser.parse_args()

    logger.info("=== INITIALIZING SQA-ML ETL COLLECTOR CLI ===")
    
    # Initialize DB & TimescaleDB hypertables
    try:
        init_db()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)

    start_time_default = datetime.now(timezone.utc) - timedelta(days=args.days)
    start_wall_time = time.time()

    total_inserted = 0

    if args.asset_class == "CRYPTO":
        async with CryptoCollector(exchange_id=args.exchange, default_type=args.market_type) as collector:
            pipeline = ETLPipeline(collector, asset_class="CRYPTO")
            for symbol in args.symbols:
                try:
                    count = await pipeline.run(symbol, args.timeframe, default_start_date=start_time_default)
                    total_inserted += count
                except Exception as e:
                    logger.error(f"Failed ETL pipeline for crypto symbol {symbol}: {e}")
    else:
        collector = StockCollector()
        pipeline = ETLPipeline(collector, asset_class="STOCK")
        for symbol in args.symbols:
            try:
                count = await pipeline.run(symbol, args.timeframe, default_start_date=start_time_default)
                total_inserted += count
            except Exception as e:
                logger.error(f"Failed ETL pipeline for stock symbol {symbol}: {e}")

    elapsed_time = time.time() - start_wall_time
    logger.success(
        f"=== ETL EXECUTION COMPLETE ===\n"
        f"Total Records Processed: {total_inserted}\n"
        f"Elapsed Time: {elapsed_time:.2f} seconds"
    )


if __name__ == "__main__":
    asyncio.run(main())
