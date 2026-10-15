"""CLI script for launching the SQA-ML automated live/paper trading bot runner."""

import argparse
import asyncio
import sys

from src.execution.runner import BotRunner
from src.utils.logger import logger


def main() -> None:
    """Parse CLI arguments and start BotRunner."""
    parser = argparse.ArgumentParser(description="SQA-ML Automated Live/Paper Trading Bot CLI")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT"],
        help="List of symbols to trade (e.g., BTC/USDT ETH/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candle timeframe (e.g., 1h, 1d)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["PAPER", "LIVE"],
        default="PAPER",
        help="Execution mode: PAPER or LIVE",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Polling interval in seconds between iterations",
    )
    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=0.65,
        help="Probability threshold for Long/Short entries",
    )

    args = parser.parse_args()

    if args.mode.upper() == "LIVE":
        raise RuntimeError("O modo LIVE esta temporariamente desativado ate a conclusao total da auditoria e re-treino do sistema.")

    logger.info("=== INITIALIZING SQA-ML LIVE/PAPER BOT CLI ===")

    runner = BotRunner(
        symbols=args.symbols,
        timeframe=args.timeframe,
        mode=args.mode,
        poll_interval_seconds=args.poll_interval,
        buy_threshold=args.buy_threshold,
        sell_threshold=1.0 - args.buy_threshold,
    )

    try:
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        logger.info("Bot execution terminated by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error in BotRunner: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
