"""CLI script for calibrating the meta-label probability threshold on continuous hourly bars.

Thresholds are chosen on the first part of the out-of-sample period and then evaluated once on the
remaining holdout, so the reported result is not the best of a grid search.
"""

import argparse
import sys
import numpy as np
import pandas as pd

from config.settings import settings
from src.data.dataset import DatasetBuilder
from src.features.storage import FeatureStorage
from src.models.trainer import QuantModelTrainer
from src.backtest.engine import VectorizedBacktester
from src.backtest.metrics import PerformanceMetrics
from src.utils.logger import logger

THRESHOLDS = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]


def evaluate(bars, sides: np.ndarray) -> dict:
    """Run the continuous backtest on a slice of bars and return its metrics."""
    trades = VectorizedBacktester().run(bars, sides)
    return PerformanceMetrics.calculate_metrics(trades, 10000.0, PerformanceMetrics.span_years(bars["timestamp"].to_pandas()))


def print_row(label: str, m: dict) -> None:
    print(
        f"{label:<10} | {m['total_trades']:<8} | {m['win_rate_pct']:>11.2f}% | {m['profit_factor']:>13.4f} | "
        f"{m['sharpe_ratio']:>9.4f} | {m['total_return_pct']:>11.2f}%"
    )


def main() -> None:
    """Parse CLI arguments and run threshold calibration with a time holdout."""
    parser = argparse.ArgumentParser(description="SQA-ML Threshold Calibration")
    parser.add_argument("--symbol", type=str, default=settings.SYMBOLS[0], help="Asset symbol (e.g., ETH/USDT)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe (e.g., 1h)")
    parser.add_argument("--holdout", type=float, default=0.3, help="Fraction of the scored period kept for the final check")
    args = parser.parse_args()

    logger.info("=== INITIALIZING SQA-ML THRESHOLD CALIBRATION ===")

    df_dataset = DatasetBuilder().load_or_build(args.symbol, args.timeframe)
    bars = FeatureStorage().load_features(args.symbol, args.timeframe)
    if df_dataset.is_empty() or bars.is_empty():
        logger.error(f"Dataset or features for {args.symbol} ({args.timeframe}) unavailable.")
        sys.exit(1)

    feature_cols = [col for col in settings.MODEL_FEATURES if col in df_dataset.columns]
    oos_probs = QuantModelTrainer().cross_val_predict_walk_forward(df_dataset, feature_cols=feature_cols)
    p_y1 = oos_probs[:, 1]

    # Only the out-of-sample scored period is simulated; it is split in time into calibration and holdout.
    scored_ts = pd.to_datetime(df_dataset["timestamp"].to_pandas())[~np.isnan(p_y1)]
    bar_ts = pd.to_datetime(bars["timestamp"].to_pandas())
    cutoff = scored_ts.min() + (scored_ts.max() - scored_ts.min()) * (1.0 - args.holdout)
    calib_mask = ((bar_ts >= scored_ts.min()) & (bar_ts < cutoff)).to_numpy()
    hold_mask = (bar_ts >= cutoff).to_numpy()
    calib_bars, hold_bars = bars.filter(calib_mask), bars.filter(hold_mask)

    print("\n" + "=" * 90)
    print(f"  THRESHOLD CALIBRATION ({args.symbol} | {args.timeframe}) | calibration period until {cutoff:%Y-%m-%d}")
    print("=" * 90)
    print(f"{'Threshold':<10} | {'Trades':<8} | {'Win Rate (%)':<12} | {'Profit Factor':<14} | {'Sharpe':<10} | {'Return (%)':<12}")
    print("-" * 90)

    best_threshold, best_score = None, -np.inf
    for th in THRESHOLDS:
        sides = VectorizedBacktester.sides_from_events(bars, df_dataset, p_y1, th)
        m = evaluate(calib_bars, sides[calib_mask])
        print_row(f"{th:.2f}", m)
        score = m["sharpe_ratio"] * 0.5 + min(m["profit_factor"], 3.0) * 0.5
        if m["profit_factor"] >= 1.0 and m["total_trades"] >= 20 and score > best_score:
            best_threshold, best_score = th, score

    print("=" * 90)
    if best_threshold is None:
        logger.warning("No threshold reached Profit Factor >= 1 with 20+ trades in the calibration period.")
        print("\nNo profitable threshold found. Do not trade this model.\n")
        return

    sides = VectorizedBacktester.sides_from_events(bars, df_dataset, p_y1, best_threshold)
    holdout = evaluate(hold_bars, sides[hold_mask])
    print(f"\nHOLDOUT (from {cutoff:%Y-%m-%d}, never used for selection):")
    print_row(f"{best_threshold:.2f}", holdout)
    if holdout["profit_factor"] < 1.0:
        print(f"\nThreshold {best_threshold} did not hold up out of sample (PF {holdout['profit_factor']:.2f}). Do not trade this model.\n")
    else:
        print(f"\nRecommended threshold: {best_threshold} (holdout PF {holdout['profit_factor']:.2f}, {holdout['total_trades']} trades)\n")


if __name__ == "__main__":
    main()
