"""CLI script for calibrating 2-stage meta-labeling probability thresholds via grid search."""

import argparse
import sys
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
    """Parse CLI arguments and run threshold calibration grid search."""
    parser = argparse.ArgumentParser(description="SQA-ML Threshold Calibration Grid Search")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Asset symbol (e.g., BTC/USDT)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe (e.g., 1h)")

    args = parser.parse_args()

    logger.info("=== INITIALIZING SQA-ML THRESHOLD CALIBRATION GRID SEARCH ===")

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
        "volume", "trades_count", "target_label", "target_return", "barrier_touched", "sample_weight", "primary_signal",
    }
    feature_cols = [col for col in df_dataset.columns if col not in exclude_cols]

    # 1. Compute Primary Signal
    pdf_dataset = df_dataset.to_pandas()
    primary_signals = TripleBarrierLabeler.get_primary_signal(pdf_dataset)

    # 2. Generate Out-Of-Sample secondary probabilities via Purged Walk-Forward CV
    trainer = QuantModelTrainer()
    logger.info("Generating Out-Of-Sample secondary probabilities via Purged Walk-Forward Cross-Validation...")
    oos_probs = trainer.cross_val_predict_walk_forward(df_dataset, feature_cols=feature_cols, n_splits=5, pct_embargo=0.01)
    p_y1 = oos_probs[:, 1]

    # 3. Grid Search over candidate thresholds
    thresholds = [0.60, 0.65, 0.70, 0.72, 0.75]
    results = []

    print("\n" + "=" * 90)
    print(f"       SQA-ML THRESHOLD CALIBRATION GRID SEARCH ({args.symbol} | {args.timeframe})       ")
    print("=" * 90)
    header = f"{'Threshold':<12} | {'Return (%)':<12} | {'Sharpe':<10} | {'Profit Factor':<15} | {'Win Rate (%)':<12} | {'Max DD (%)':<12} | {'Fees ($)':<10}"
    print(header)
    print("-" * 90)

    best_threshold = 0.65
    best_score = -999.0

    for th in thresholds:
        proba_matrix = np.zeros((len(df_dataset), 3))
        proba_matrix[:, 1] = 1.0

        for i in range(len(df_dataset)):
            sig = primary_signals.iloc[i]
            p = p_y1[i]
            if sig == 1 and p >= th:
                proba_matrix[i, 2] = 0.80
                proba_matrix[i, 1] = 0.20
            elif sig == -1 and p >= th:
                proba_matrix[i, 0] = 0.80
                proba_matrix[i, 1] = 0.20

        backtester = VectorizedBacktester(
            initial_capital=10000.0,
            fee_rate=0.0005,
            slippage=0.0002,
            buy_threshold=th,
            sell_threshold=1.0 - th,
        )
        results_df = backtester.run(df_dataset, proba_matrix)
        metrics = PerformanceMetrics.calculate_metrics(results_df, initial_capital=10000.0)

        ret = metrics["total_return_pct"]
        sharpe = metrics["sharpe_ratio"]
        pf = metrics["profit_factor"]
        wr = metrics["win_rate_pct"]
        mdd = metrics["max_drawdown_pct"]
        fees = metrics["total_fees_paid"]

        print(f"{th:<12.2f} | {ret:>11.2f}% | {sharpe:>9.4f} | {pf:>14.4f} | {wr:>11.2f}% | {mdd:>11.2f}% | {fees:>9.2f}")

        results.append({"threshold": th, "metrics": metrics})

        # Scoring function: weighted combination of Sharpe and Profit Factor targeting Phase 2 criteria
        score = sharpe * 0.5 + min(pf, 3.0) * 0.5
        if score > best_score and pf >= 1.0:
            best_score = score
            best_threshold = th

    print("=" * 90)
    logger.success(f"Optimal threshold selected: {best_threshold} (Score: {best_score:.4f})")
    print(f"\n👉 Recommended threshold to fulfill Phase 2 Acceptance Criteria: {best_threshold}\n")


if __name__ == "__main__":
    main()
