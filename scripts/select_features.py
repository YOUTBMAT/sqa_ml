"""Select positive-MDA model features using purged walk-forward OOS folds."""

import argparse
import json
from pathlib import Path

import pandas as pd

from config.settings import settings
from src.data.dataset import DatasetBuilder
from src.features.selection import PurgedFeatureSelector
from src.models.trainer import QuantModelTrainer
from src.validation.cross_validation import PurgedGroupTimeSeriesSplit


def selection_period(data: pd.DataFrame, holdout_fraction: float) -> pd.DataFrame:
    """Limit feature selection to observations strictly before the final time holdout."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1.")
    if "timestamp" not in data.columns or "t1" not in data.columns or data.empty:
        raise ValueError("Feature selection data must be non-empty and contain timestamp and t1 columns.")
    timestamps = pd.to_datetime(data["timestamp"], utc=True)
    event_ends = pd.to_datetime(data["t1"], utc=True)
    start, end = timestamps.min(), timestamps.max()
    if start >= end:
        raise ValueError("Feature selection requires at least two distinct timestamps.")
    cutoff = start + (end - start) * (1.0 - holdout_fraction)
    selected_period = data.loc[(timestamps < cutoff) & (event_ends < cutoff)].copy().reset_index(drop=True)
    if selected_period.empty:
        raise ValueError("No data remains before the reserved holdout.")
    return selected_period


def main() -> None:
    parser = argparse.ArgumentParser(description="Purged walk-forward MDA feature selection")
    parser.add_argument("--symbol", type=str, default="ETH/USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--embargo", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.3,
        help="Final time fraction excluded from feature selection and reserved for holdout evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "selected_features.json",
        help="Output path for the selected feature list.",
    )
    args = parser.parse_args()

    dataset = DatasetBuilder().load_or_build(args.symbol, args.timeframe)
    if dataset.is_empty():
        raise RuntimeError(f"No labeled dataset available for {args.symbol} ({args.timeframe}).")

    feature_cols = [feature for feature in settings.MODEL_FEATURES if feature in dataset.columns]
    missing = [feature for feature in settings.MODEL_FEATURES if feature not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing active model features: {missing}")

    pdf = selection_period(dataset.to_pandas(), args.holdout)
    usable = pdf[feature_cols + ["target_label", "sample_weight"]].notna().all(axis=1)
    pdf = pdf.loc[usable].reset_index(drop=True)
    X = pdf[feature_cols]
    y = pdf["target_label"].astype(int)

    cv = PurgedGroupTimeSeriesSplit(n_splits=args.n_splits, pct_embargo=args.embargo)
    splits = list(cv.split(pdf))
    selector = PurgedFeatureSelector()
    scores = selector.calculate_mda(
        QuantModelTrainer().model,
        X,
        y,
        splits,
        sample_weight=pdf["sample_weight"].astype(float),
        random_state=args.seed,
    )

    print("\n--- PURGED WALK-FORWARD MDA RANKING ---")
    print(scores[["feature", "mean_mda", "std_mda", "folds"]].to_string(index=False))

    selected = scores.loc[scores["mean_mda"] > 0.0, "feature"].tolist()
    eliminated = scores.loc[scores["mean_mda"] <= 0.0, "feature"].tolist()
    if not selected:
        raise RuntimeError(
            "MDA did not identify any positive-importance features; selection file was not changed."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(f"\nSelected {len(selected)} of {len(feature_cols)} active features: {args.output}")
    print("Eliminated (mean MDA <= 0): " + (", ".join(eliminated) if eliminated else "none"))


if __name__ == "__main__":
    main()
