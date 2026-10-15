"""CLI script for running continuous hourly backtests of the 2-stage meta-labeling strategy."""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from config.settings import settings
from src.data.dataset import DatasetBuilder
from src.features.storage import FeatureStorage
from src.models.trainer import QuantModelTrainer
from src.models.storage import ModelStorage
from src.backtest.engine import VectorizedBacktester
from src.backtest.metrics import PerformanceMetrics
from src.utils.logger import logger


def main() -> None:
    """Parse CLI arguments and execute the continuous hourly backtest."""
    parser = argparse.ArgumentParser(description="SQA-ML Continuous Hourly Backtest CLI")
    parser.add_argument("--symbol", type=str, default=settings.SYMBOLS[0], help="Asset symbol (e.g., ETH/USDT)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe (e.g., 1h)")
    parser.add_argument(
        "--mode", type=str, choices=["standard", "walk-forward"], default="walk-forward",
        help="walk-forward: out-of-sample probabilities only. standard: saved model on its own training data (in-sample)",
    )
    parser.add_argument("--threshold", type=float, default=settings.PROB_THRESHOLD, help="Minimum P(y=1) to confirm a signal")
    parser.add_argument("--fee-rate", type=float, default=0.0005, help="Transaction fee rate per side")
    parser.add_argument("--slippage", type=float, default=0.0002, help="Slippage rate per side")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Initial capital in USD")
    parser.add_argument("--export", action="store_true", help="Export the trade list to Parquet")
    args = parser.parse_args()

    logger.info("=== INITIALIZING SQA-ML CONTINUOUS HOURLY BACKTEST CLI ===")

    bars = FeatureStorage().load_features(args.symbol, args.timeframe)
    df_dataset = DatasetBuilder().load_or_build(args.symbol, args.timeframe)
    if bars.is_empty() or df_dataset.is_empty():
        logger.error(f"Features or dataset for {args.symbol} ({args.timeframe}) not found. Run generate_features.py first.")
        sys.exit(1)

    if args.mode == "walk-forward":
        feature_cols = [col for col in settings.MODEL_FEATURES if col in df_dataset.columns]
        p_y1 = QuantModelTrainer().cross_val_predict_walk_forward(df_dataset, feature_cols=feature_cols)[:, 1]
        first_scored = pd.to_datetime(df_dataset["timestamp"].to_pandas())[~np.isnan(p_y1)].min()
    else:
        logger.warning("STANDARD mode scores the saved model on data it was trained on: results are in-sample and optimistic.")
        model, meta = ModelStorage().load_model(args.symbol, args.timeframe)
        p_y1 = model.predict_proba(df_dataset, feature_cols=meta["feature_cols"])[:, 1]

    sides = VectorizedBacktester.sides_from_events(bars, df_dataset, p_y1, args.threshold)
    trades = VectorizedBacktester(
        initial_capital=args.initial_capital, fee_rate=args.fee_rate, slippage=args.slippage
    ).run(bars, sides)
    bar_ts = pd.to_datetime(bars["timestamp"].to_pandas())
    if args.mode == "walk-forward":
        bar_ts = bar_ts[bar_ts >= first_scored]  # unscored rows never trade, so the period starts at the first scored event
    metrics = PerformanceMetrics.calculate_metrics(trades, args.initial_capital, PerformanceMetrics.span_years(bar_ts))

    print("\n" + "=" * 55)
    print(f"  SQA-ML CONTINUOUS HOURLY BACKTEST ({args.mode.upper()})")
    print(f"  Asset: {args.symbol} | Timeframe: {args.timeframe} | Threshold: {args.threshold}")
    print("=" * 55)
    for k, v in metrics.items():
        if "pct" in k:
            print(f" {k.replace('_', ' ').title():<25} : {v:>10.2f}%")
        elif "ratio" in k or "factor" in k:
            print(f" {k.replace('_', ' ').title():<25} : {v:>10.4f}")
        else:
            print(f" {k.replace('_', ' ').title():<25} : {v:>10.2f}")
    print("=" * 55 + "\n")

    if args.export:
        results_dir = Path("data/backtest_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        sanitized_symbol = args.symbol.replace("/", "_").replace("-", "_").lower()
        export_path = results_dir / f"{sanitized_symbol}_{args.timeframe}_{args.mode}_trades.parquet"
        trades.to_parquet(export_path)
        logger.success(f"Trade list exported to {export_path}")


if __name__ == "__main__":
    main()
