"""CLI script for running complete out-of-sample backtest simulations and performance reporting (2-Stage Meta-Labeling)."""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

from src.data.dataset import DatasetBuilder
from src.models.trainer import QuantModelTrainer
from src.labeling.triple_barrier import TripleBarrierLabeler
from src.backtest.engine import VectorizedBacktester
from src.backtest.metrics import PerformanceMetrics
from src.utils.logger import logger


def main() -> None:
    """Parse CLI arguments and execute 2-stage meta-labeling backtest pipeline."""
    parser = argparse.ArgumentParser(description="SQA-ML 2-Stage Meta-Labeling Backtest CLI")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Asset symbol (e.g., BTC/USDT)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe (e.g., 1h)")
    parser.add_argument("--mode", type=str, choices=["standard", "walk-forward"], default="walk-forward", help="Backtest prediction mode")
    parser.add_argument("--buy-threshold", type=float, default=0.75, help="Probability threshold for Long entries")
    parser.add_argument("--sell-threshold", type=float, default=0.25, help="Probability threshold for Short entries")
    parser.add_argument("--fee-rate", type=float, default=0.0005, help="Transaction fee rate per trade")
    parser.add_argument("--slippage", type=float, default=0.0002, help="Slippage rate assumption")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Initial capital in USD")
    parser.add_argument("--export", action="store_true", help="Export backtest results to Parquet")

    args = parser.parse_args()

    logger.info("=== INITIALIZING SQA-ML 2-STAGE BACKTEST CLI ===")

    builder = DatasetBuilder()
    df_dataset = builder.load_dataset(args.symbol, args.timeframe)

    if df_dataset.is_empty():
        logger.info(f"Dataset for {args.symbol} ({args.timeframe}) not found. Automatically building dataset...")
        df_dataset = builder.build_dataset(args.symbol, args.timeframe)
        if not df_dataset.is_empty():
            builder.save_dataset(df_dataset, args.symbol, args.timeframe)

    if df_dataset.is_empty():
        logger.error(f"Dataset for {args.symbol} ({args.timeframe}) could not be built or loaded.")
        sys.exit(1)

    exclude_cols = {
        "timestamp", "symbol", "asset_class", "timeframe", "open", "high", "low", "close",
        "volume", "trades_count", "target_label", "target_return", "barrier_touched", "sample_weight", "primary_signal", "ofi", "t1", "touch_time",
    }
    feature_cols = [col for col in df_dataset.columns if col not in exclude_cols]

    # 1. Compute Primary Signal (EMA200 & ADX14)
    pdf_dataset = df_dataset.to_pandas()
    primary_signals = TripleBarrierLabeler.get_primary_signal(pdf_dataset)

    # 2. Generate Secondary Meta-Label probabilities P(y=1)
    trainer = QuantModelTrainer()
    if args.mode == "walk-forward":
        logger.info("Generating Out-Of-Sample secondary probabilities via Purged Walk-Forward Cross-Validation...")
        oos_probs = trainer.cross_val_predict_walk_forward(df_dataset, feature_cols=feature_cols, n_splits=5, pct_embargo=0.01)
        p_y1 = oos_probs[:, 1]
    else:
        logger.info("Training full model for secondary probabilities...")
        trainer.train(df_dataset, feature_cols=feature_cols)
        full_probs = trainer.predict_proba(df_dataset, feature_cols=feature_cols)
        p_y1 = full_probs[:, 1]

    # 3. Two-Stage Filtering: Primary Signal + Secondary Model Confidence P(y=1) >= args.buy_threshold
    proba_matrix = np.zeros((len(df_dataset), 3))
    proba_matrix[:, 1] = 1.0

    for i in range(len(df_dataset)):
        sig = primary_signals.iloc[i]
        p = p_y1[i]
        if sig == 1 and p >= args.buy_threshold:
            proba_matrix[i, 2] = 0.80  # Long confirmed
            proba_matrix[i, 1] = 0.20
        elif sig == -1 and p >= args.buy_threshold:
            proba_matrix[i, 0] = 0.80  # Short confirmed
            proba_matrix[i, 1] = 0.20

    # 4. Vectorized Backtest Engine
    backtester = VectorizedBacktester(
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )
    results_df = backtester.run(df_dataset, proba_matrix)

    # 5. Performance Metrics Calculation
    metrics = PerformanceMetrics.calculate_metrics(results_df, initial_capital=args.initial_capital)

    # 6. Display Results in Terminal
    print("\n" + "=" * 50)
    print(f"       SQA-ML 2-STAGE META-LABELING REPORT     ")
    print(f"       Asset: {args.symbol} | Timeframe: {args.timeframe} | Mode: {args.mode}       ")
    print("=" * 50)
    for k, v in metrics.items():
        if "pct" in k or "return" in k or "drawdown" in k:
            print(f" {k.replace('_', ' ').title():<25} : {v:>10.2f}%")
        elif "ratio" in k or "factor" in k:
            print(f" {k.replace('_', ' ').title():<25} : {v:>10.4f}")
        else:
            print(f" {k.replace('_', ' ').title():<25} : {v:>10.2f}")
    print("=" * 50 + "\n")

    # 7. Export results if requested
    if args.export:
        results_dir = Path("data/backtest_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        sanitized_symbol = args.symbol.replace("/", "_").replace("-", "_").lower()
        export_path = results_dir / f"{sanitized_symbol}_{args.timeframe}_metalabel_{args.mode}.parquet"
        results_df.reset_index().to_parquet(export_path)
        logger.success(f"2-Stage backtest results exported successfully to {export_path}")

    logger.info("=== 2-STAGE BACKTEST PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
