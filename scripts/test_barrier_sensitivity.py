"""Compare triple-barrier horizons with purged OOS model and continuous backtest metrics."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import polars as pl

HORIZONS_HOURS = (12, 24, 48)
THRESHOLDS = (0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65)
PT_SL = [1.5, 1.0]
HOLDOUT_FRACTION = 0.30


def _evaluate(
    bars: pl.DataFrame,
    sides: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
    bar_timestamps: pd.Series,
    max_holding_hours: int,
    end_timestamp: pd.Timestamp | None = None,
) -> dict:
    from src.backtest.engine import VectorizedBacktester
    from src.backtest.metrics import PerformanceMetrics

    eval_bars = bars
    if end_timestamp is not None:
        eval_bars = bars.filter(pl.col("timestamp") < end_timestamp)
    eval_length = len(eval_bars)
    eval_mask = mask[:eval_length]
    eval_timestamps = bar_timestamps.iloc[:eval_length]
    eval_probabilities = np.where(eval_mask, probabilities[:eval_length], np.nan)
    trades = VectorizedBacktester(max_holding_hours=max_holding_hours, pt_sl=PT_SL).run(
        eval_bars, np.where(eval_mask, sides[:eval_length], 0.0), eval_probabilities
    )
    span = PerformanceMetrics.span_years(eval_timestamps[eval_mask])
    return PerformanceMetrics.calculate_metrics(trades, 10000.0, span)


def analyze_horizon(symbol: str, timeframe: str, max_holding_hours: int) -> dict:
    """Build one horizon's labels, score it walk-forward, and calibrate on the OOS timeline."""
    from src.backtest.engine import VectorizedBacktester
    from src.data.dataset import DatasetBuilder
    from src.features.storage import FeatureStorage
    from src.labeling.triple_barrier import TripleBarrierLabeler
    from src.models.trainer import QuantModelTrainer

    features = FeatureStorage().load_features(symbol, timeframe)
    if features.is_empty():
        raise ValueError(f"Features unavailable for {symbol} ({timeframe}).")

    labeled = TripleBarrierLabeler.get_labels(
        features, pt_sl=PT_SL, max_holding_hours=max_holding_hours
    )
    if labeled.is_empty():
        raise ValueError(f"No resolved labeled events for {symbol} at {max_holding_hours}h.")
    barrier_counts = labeled["barrier_touched"].value_counts()
    counts = {row["barrier_touched"]: row["count"] for row in barrier_counts.to_dicts()}
    total_labels = sum(counts.values())

    dataset = DatasetBuilder().build_dataset(
        symbol, timeframe, pt_sl=PT_SL, max_holding_hours=max_holding_hours
    )
    if dataset.is_empty():
        raise ValueError(f"Training dataset empty for {symbol} at {max_holding_hours}h.")
    feature_cols = QuantModelTrainer.configured_feature_cols(dataset.columns)
    trainer = QuantModelTrainer()
    oos_probabilities = trainer.cross_val_predict_walk_forward(
        dataset, feature_cols=feature_cols, n_splits=5, pct_embargo=0.01
    )
    p_y1 = oos_probabilities[:, 1]
    oos_auc = QuantModelTrainer.oos_roc_auc(dataset, oos_probabilities)
    if not np.isfinite(oos_auc):
        raise ValueError(f"OOS ROC AUC is undefined for {max_holding_hours}h.")

    event_ts = pd.to_datetime(dataset["timestamp"].to_pandas())
    scored = np.isfinite(p_y1)
    if not scored.any():
        raise ValueError(f"No walk-forward OOS predictions for {max_holding_hours}h.")
    scored_ts = event_ts[scored]
    cutoff = scored_ts.min() + (scored_ts.max() - scored_ts.min()) * (1.0 - HOLDOUT_FRACTION)
    bars_ts = pd.to_datetime(features["timestamp"].to_pandas())
    calibration_mask = ((bars_ts >= scored_ts.min()) & (bars_ts < cutoff)).to_numpy()
    # Allow holdout entries to resolve on the remaining candles, but never create entries
    # from events that did not receive an OOS prediction.
    holdout_mask = (bars_ts >= cutoff).to_numpy()

    best_threshold: float | None = None
    best_score = -np.inf
    best_calibration: dict | None = None
    for threshold in THRESHOLDS:
        sides = VectorizedBacktester.sides_from_events(features, dataset, p_y1, threshold)
        metrics = _evaluate(
            features,
            sides,
            VectorizedBacktester.probabilities_from_events(features, dataset, p_y1),
            calibration_mask,
            bars_ts,
            max_holding_hours,
            end_timestamp=cutoff,
        )
        score = 0.5 * metrics["sharpe_ratio"] + 0.5 * min(metrics["profit_factor"], 3.0)
        if metrics["profit_factor"] >= 1.0 and metrics["total_trades"] >= 20 and score > best_score:
            best_threshold, best_score, best_calibration = threshold, score, metrics

    if best_threshold is None or best_calibration is None:
        return {
            "horizon": max_holding_hours,
            "label_counts": counts,
            "label_total": total_labels,
            "oos_auc": oos_auc,
            "threshold": None,
            "oos_metrics": None,
            "holdout_metrics": None,
        }

    sides = VectorizedBacktester.sides_from_events(features, dataset, p_y1, best_threshold)
    holdout_metrics = _evaluate(
        features,
        sides,
        VectorizedBacktester.probabilities_from_events(features, dataset, p_y1),
        holdout_mask,
        bars_ts,
        max_holding_hours,
    )
    return {
        "horizon": max_holding_hours,
        "label_counts": counts,
        "label_total": total_labels,
        "oos_auc": oos_auc,
        "threshold": best_threshold,
        "oos_metrics": best_calibration,
        "holdout_metrics": holdout_metrics,
    }


def _print_results(results: list[dict]) -> None:
    print("\nLABEL DISTRIBUTION")
    print("| Horizon | TP | SL | Vertical | Total |")
    print("|---:|---:|---:|---:|---:|")
    for result in results:
        total = result["label_total"]
        counts = result["label_counts"]
        pct = lambda key: counts.get(key, 0) / total * 100 if total else 0.0
        print(
            f"| {result['horizon']}h | {pct('tp'):.2f}% | {pct('sl'):.2f}% | "
            f"{pct('t1'):.2f}% | {total} |"
        )

    print("\nOOS PERFORMANCE (threshold selected on first 70% of walk-forward OOS predictions)")
    print(
        "| Horizon | OOS ROC AUC | Threshold | OOS PF | OOS Sharpe | OOS win rate | OOS return | "
        "Holdout PF | Holdout Sharpe | Holdout win rate | Holdout return |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        oos = result["oos_metrics"]
        holdout = result["holdout_metrics"]
        if oos is None or holdout is None:
            print(
                f"| {result['horizon']}h | {result['oos_auc']:.4f} | none | no profitable calibration threshold "
                "| — | — | — | — | — | — | — |"
            )
            continue
        print(
            f"| {result['horizon']}h | {result['oos_auc']:.4f} | {result['threshold']:.2f} | "
            f"{oos['profit_factor']:.4f} | {oos['sharpe_ratio']:.4f} | {oos['win_rate_pct']:.2f}% | "
            f"{oos['total_return_pct']:.2f}% | {holdout['profit_factor']:.4f} | "
            f"{holdout['sharpe_ratio']:.4f} | {holdout['win_rate_pct']:.2f}% | "
            f"{holdout['total_return_pct']:.2f}% |"
        )

    sustainable = [
        result
        for result in results
        if result["oos_metrics"] is not None
        and result["oos_metrics"]["profit_factor"] > 1.20
        and result["holdout_metrics"]["profit_factor"] > 1.0
        and result["holdout_metrics"]["total_return_pct"] > 0
    ]
    if sustainable:
        best = max(sustainable, key=lambda result: result["holdout_metrics"]["total_return_pct"])
        print(
            f"\nSustainable positive expectancy: {best['horizon']}h "
            "(OOS PF > 1.20 and positive holdout return with holdout PF > 1.00)."
        )
    else:
        print("\nNo horizon met the sustainability criteria (OOS PF > 1.20 and positive holdout return/PF).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Triple-barrier holding-time sensitivity benchmark")
    from config.settings import settings
    from src.utils.logger import logger

    parser.add_argument("--symbol", type=str, default=settings.SYMBOLS[0])
    parser.add_argument("--timeframe", type=str, default="1h")
    args = parser.parse_args()

    logger.info(
        f"Running triple-barrier sensitivity for {args.symbol} ({args.timeframe}), "
        f"PT/SL={PT_SL}, horizons={HORIZONS_HOURS}."
    )
    results = [analyze_horizon(args.symbol, args.timeframe, hours) for hours in HORIZONS_HOURS]
    _print_results(results)


if __name__ == "__main__":
    from src.utils.logger import logger

    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        logger.error(f"Barrier sensitivity analysis failed: {exc}")
        sys.exit(1)
