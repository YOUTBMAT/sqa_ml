"""Select positive-MDA model features using purged walk-forward OOS folds."""

import argparse
import json
from pathlib import Path

from config.settings import settings
from src.data.dataset import DatasetBuilder
from src.features.selection import PurgedFeatureSelector
from src.models.trainer import QuantModelTrainer
from src.validation.cross_validation import PurgedGroupTimeSeriesSplit


def main() -> None:
    parser = argparse.ArgumentParser(description="Purged walk-forward MDA feature selection")
    parser.add_argument("--symbol", type=str, default="ETH/USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--embargo", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = DatasetBuilder().load_or_build(args.symbol, args.timeframe)
    if dataset.is_empty():
        raise RuntimeError(f"No labeled dataset available for {args.symbol} ({args.timeframe}).")

    feature_cols = [feature for feature in settings.MODEL_FEATURES if feature in dataset.columns]
    missing = [feature for feature in settings.MODEL_FEATURES if feature not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing active model features: {missing}")

    pdf = dataset.to_pandas()
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

    output_path = Path(__file__).resolve().parents[1] / "config" / "selected_features.json"
    output_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(f"\nSelected {len(selected)} of {len(feature_cols)} active features: {output_path}")
    print("Eliminated (mean MDA <= 0): " + (", ".join(eliminated) if eliminated else "none"))


if __name__ == "__main__":
    main()
