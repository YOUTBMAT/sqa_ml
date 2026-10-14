"""CLI script for training LightGBM quantitative models using dataset artifacts."""

import argparse
import sys
from datetime import datetime, timezone
import polars as pl

from src.data.dataset import DatasetBuilder
from src.models.trainer import QuantModelTrainer
from src.models.storage import ModelStorage
from src.utils.logger import logger


def main() -> None:
    """Parse CLI arguments and run model training pipeline."""
    parser = argparse.ArgumentParser(description="SQA-ML Model Training CLI")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Asset symbol to train on (e.g., BTC/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candle timeframe (e.g., 1h, 1d)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset reserved for validation split",
    )

    args = parser.parse_args()

    logger.info("=== INITIALIZING SQA-ML MODEL TRAINING CLI ===")

    builder = DatasetBuilder()
    storage = ModelStorage()

    # Load dataset, or auto-build if not found
    df_dataset = builder.load_dataset(args.symbol, args.timeframe)
    if df_dataset.is_empty():
        logger.info(f"Dataset for {args.symbol} ({args.timeframe}) not found. Automatically building dataset...")
        df_dataset = builder.build_dataset(args.symbol, args.timeframe)
        if not df_dataset.is_empty():
            builder.save_dataset(df_dataset, args.symbol, args.timeframe)

    if df_dataset.is_empty():
        logger.error(f"Dataset for {args.symbol} ({args.timeframe}) could not be built or loaded.")
        sys.exit(1)

    # Exclude metadata, non-feature columns from feature list
    exclude_cols = {
        "timestamp",
        "symbol",
        "asset_class",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trades_count",
        "target_label",
        "target_return",
        "barrier_touched",
        "sample_weight",
        "ofi",
        "primary_signal",
        "t1",
        "touch_time",
    }
    feature_cols = [col for col in df_dataset.columns if col not in exclude_cols]
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

    print("\n--- EXACT LIST OF FEATURE COLUMNS USED IN TRAINING ---")
    for f in feature_cols:
        print(f"  - {f}")
    print("-" * 50 + "\n")

    logger.info(f"Identified {len(feature_cols)} feature columns for training.")

    # Compute Out-Of-Sample Walk-Forward AUC
    logger.info("Computing Out-Of-Sample Walk-Forward Cross-Validation AUC...")
    from src.validation.cross_validation import PurgedGroupTimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    import numpy as np

    cv = PurgedGroupTimeSeriesSplit(n_splits=5, pct_embargo=0.01)
    pdf = df_dataset.to_pandas()
    oos_probs = np.full((len(df_dataset), 2), [0.5, 0.5])
    tested_indices = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(pdf)):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        train_df = pl.from_pandas(pdf.iloc[train_idx])
        test_df = pl.from_pandas(pdf.iloc[test_idx])

        fold_trainer = QuantModelTrainer(n_estimators=100, max_depth=5, learning_rate=0.05)
        fold_trainer.train(train_df, feature_cols=feature_cols, target_col="target_label", weight_col="sample_weight")

        test_probs = fold_trainer.predict_proba(test_df, feature_cols=feature_cols)
        oos_probs[test_idx] = test_probs
        tested_indices.extend(test_idx)

    if tested_indices:
        y_true = pdf["target_label"].iloc[tested_indices].astype(int)
        y_pred_proba = oos_probs[tested_indices, 1]
        try:
            oos_auc = roc_auc_score(y_true, y_pred_proba)
        except Exception:
            oos_auc = 0.0
    else:
        oos_auc = 0.0

    print(f"\n==================================================")
    print(f" WALK-FORWARD OUT-OF-SAMPLE (OOS) ROC AUC: {oos_auc:.4f}")
    print(f"==================================================\n")

    # Instantiate and train final model on full dataset
    trainer = QuantModelTrainer(n_estimators=100, max_depth=5, learning_rate=0.05)
    
    metrics = trainer.train(
        df=df_dataset,
        feature_cols=feature_cols,
        target_col="target_label",
        weight_col="sample_weight",
    )

    # Display top 10 feature importances
    fi_df = trainer.get_feature_importance()
    print("\n--- TOP 10 FEATURE IMPORTANCES (GAIN) ---")
    print(fi_df.head(10).to_string(index=False))
    print("-" * 42 + "\n")

    # Prepare metadata and save model
    metadata = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df_dataset),
    }

    storage.save_model(trainer, args.symbol, args.timeframe, metadata)
    logger.success("=== MODEL TRAINING & PERSISTENCE COMPLETE ===")


if __name__ == "__main__":
    main()
