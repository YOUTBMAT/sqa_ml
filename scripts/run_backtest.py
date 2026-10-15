"""CLI script for running complete continuous hourly backtest simulations and performance reporting (2-Stage Meta-Labeling)."""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

from src.features.storage import FeatureStorage
from src.models.trainer import QuantModelTrainer
from src.models.storage import ModelStorage
from src.labeling.triple_barrier import TripleBarrierLabeler
from src.backtest.engine import VectorizedBacktester
from src.backtest.metrics import PerformanceMetrics
from src.utils.logger import logger


def main() -> None:
    """Parse CLI arguments and execute 2-stage continuous hourly backtest pipeline."""
    parser = argparse.ArgumentParser(description="SQA-ML Continuous Hourly Backtest CLI")
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

    logger.info("=== INITIALIZING SQA-ML CONTINUOUS HOURLY BACKTEST CLI ==PS===")

    feature_storage = FeatureStorage()
    df_features = feature_storage.load_features(args.symbol, args.timeframe)

    if df_features.is_empty():
        logger.error(f"Features for {args.symbol} ({args.timeframe}) not found. Please run generate_features.py first.")
        sys.exit(1)

    exclude_cols = {
        "timestamp", "symbol", "asset_class", "timeframe", "open", "high", "low", "close",
        "volume", "trades_count", "ofi", "t1", "touch_time",
    }
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    top10_features = [
        "vol_parkinson_pct",
        "ema_200_dist",
        "frac_diff_close",
        "hurst_100",
        "vol_garman_klass",
        "vol_ratio_6_24",
        "vol_parkinson",
        "rsi_adaptive",
        "vwap_lower_2_dist",
        "adx_14",
    ]
    feature_cols = [col for col in feature_cols if col in top10_features]

    # 1. Compute Primary Signal for all continuous hourly bars
    pdf_features = df_features.to_pandas()
    primary_signals = TripleBarrierLabeler.get_primary_signal(pdf_features)

    # 2. Load or train model / generate probabilities for all bars
    storage = ModelStorage()
    try:
        model, meta = storage.load_model(args.symbol, args.timeframe)
        logger.info("Loaded trained model for continuous inference.")
        # Predict on all feature rows
        probs = model.predict_proba(df_features, feature_cols=feature_cols)
        p_y1 = probs[:, 1]
    except Exception:
        logger.info("No saved model found. Training fresh model...")
        trainer = QuantModelTrainer()
        # Build dataset just for training weights/labels
        from src.data.dataset import DatasetBuilder
        builder = DatasetBuilder()
        df_dataset = builder.load_dataset(args.symbol, args.timeframe)
        if df_dataset.is_empty():
            df_dataset = builder.build_dataset(args.symbol, args.timeframe)
        trainer.train(df_dataset, feature_cols=feature_cols)
        probs = trainer.predict_proba(df_features, feature_cols=feature_cols)
        p_y1 = probs[:, 1]

    # 3. Two-Stage Filtering: Primary Signal + Secondary Model Confidence P(y=1) >= args.buy_threshold
    proba_matrix = np.zeros((len(df_features), 3))
    proba_matrix[:, 1] = 1.0

    for i in range(len(df_features)):
        sig = primary_signals.iloc[i]
        p = p_y1[i]
        if sig == 1 and p >= args.buy_threshold:
            proba_matrix[i, 2] = 0.80  # Long confirmed
            proba_matrix[i, 1] = 0.20
        elif sig == -1 and p >= args.buy_threshold:
            proba_matrix[i, 0] = 0.80  # Short confirmed
            proba_matrix[i, 1] = 0.20

    # 4. Continuous Hourly Backtest Engine
    backtester = VectorizedBacktester(
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )
    results_df = backtester.run(df_features, proba_matrix)

    # 5. Performance Metrics Calculation
    metrics = PerformanceMetrics.calculate_metrics(results_df, initial_capital=args.initial_capital)

    # 6. Display Results in Terminal
    print("\n" + "=" * 55)
    print(f"       SQA-ML CONTINUOUS HOURLY BACKTEST REPORT     ")
    print(f"       Asset: {args.symbol} | Timeframe: {args.timeframe}       ")
    print("=" * 55)
    for k, v in metrics.items():
        if "pct" in k or "return" in k or "drawdown" in k:
            print(f" {k.replace('_', ' ').title():<25} : {v:>10.2f}%")
        elif "ratio" in k or "factor" in k:
            print(f" {k.replace('_', ' ').title():<25} : {v:>10.4f}")
        else:
            print(f" {k.replace('_', ' ').title():<25} : {v:>10.2f}")
    print("=" * 55 + "\n")

    # 7. Export results if requested
    if args.export:
        results_dir = Path("data/backtest_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        sanitized_symbol = args.symbol.replace("/", "_").replace("-", "_").lower()
        export_path = results_dir / f"{sanitized_symbol}_{args.timeframe}_continuous_backtest.parquet"
        results_df.reset_index().to_parquet(export_path)
        logger.success(f"Continuous backtest results exported successfully to {export_path}")

    logger.info("=== CONTINUOUS HOURLY BACKTEST PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
