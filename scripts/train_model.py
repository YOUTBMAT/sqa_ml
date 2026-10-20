"""CLI script for training LightGBM quantitative models using dataset artifacts."""

import argparse
import sys
from datetime import datetime, timezone

from config.settings import settings
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
        default=settings.SYMBOLS[0],
        help="Asset symbol to train on (e.g., BTC/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candle timeframe (e.g., 1h, 1d)",
    )

    args = parser.parse_args()

    logger.info("=== INITIALIZING SQA-ML MODEL TRAINING CLI ===")

    builder = DatasetBuilder()
    storage = ModelStorage()

    df_dataset = builder.load_or_build(args.symbol, args.timeframe)
    if df_dataset.is_empty():
        logger.error(f"Dataset for {args.symbol} ({args.timeframe}) could not be built or loaded.")
        sys.exit(1)

    feature_cols = QuantModelTrainer.configured_feature_cols(df_dataset.columns)

    print("\n--- EXACT LIST OF FEATURE COLUMNS USED IN TRAINING ---")
    for f in feature_cols:
        print(f"  - {f}")
    print("-" * 50 + "\n")

    # Out-Of-Sample Walk-Forward AUC with the same hyperparameters as the final model
    trainer = QuantModelTrainer()
    oos_probs = trainer.cross_val_predict_walk_forward(df_dataset, feature_cols=feature_cols, n_splits=5, pct_embargo=0.01)
    oos_auc = QuantModelTrainer.oos_roc_auc(df_dataset, oos_probs)

    print("\n==================================================")
    print(f" WALK-FORWARD OUT-OF-SAMPLE (OOS) ROC AUC: {oos_auc:.4f}")
    print("==================================================\n")

    # Train final model on the full dataset (these metrics are in-sample)
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
        "in_sample_metrics": metrics,
        "oos_roc_auc": float(oos_auc),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df_dataset),
    }

    storage.save_model(trainer, args.symbol, args.timeframe, metadata)
    logger.success("=== MODEL TRAINING & PERSISTENCE COMPLETE ===")


if __name__ == "__main__":
    main()
