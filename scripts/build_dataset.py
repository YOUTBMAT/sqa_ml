"""CLI script for building supervised machine learning datasets via DatasetBuilder."""

import argparse
import sys
from pathlib import Path

from src.data.dataset import DatasetBuilder
from src.utils.logger import logger


def main() -> None:
    """Parse CLI arguments and build labeled ML dataset."""
    parser = argparse.ArgumentParser(description="SQA-ML Dataset Building CLI")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Asset symbol (e.g., BTC/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candle timeframe (e.g., 1h, 1d)",
    )
    parser.add_argument(
        "--pt-sl",
        nargs=2,
        type=float,
        default=[2.0, 1.0],
        help="Profit Taking and Stop Loss multipliers (e.g., 2.0 1.0)",
    )
    parser.add_argument(
        "--num-bars",
        type=int,
        default=24,
        help="Number of bars for vertical barrier expiration",
    )

    args = parser.parse_args()

    logger.info("=== INITIALIZING SQA-ML DATASET BUILDER CLI ===")

    builder = DatasetBuilder()
    
    logger.info(f"Building dataset for {args.symbol} ({args.timeframe}) with PT/SL={args.pt_sl}, num_bars={args.num_bars}...")
    df_dataset = builder.build_dataset(
        symbol=args.symbol,
        timeframe=args.timeframe,
        pt_sl=args.pt_sl,
        num_bars=args.num_bars,
    )

    if df_dataset.is_empty():
        logger.error(f"Failed to build dataset for {args.symbol} ({args.timeframe}). Check if feature parquet exists.")
        sys.exit(1)

    filepath = builder.save_dataset(df_dataset, args.symbol, args.timeframe)
    logger.success(f"=== DATASET BUILD COMPLETE === Saved {len(df_dataset)} samples to {filepath}")


if __name__ == "__main__":
    main()
