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
    }
    feature_cols = [col for col in df_dataset.columns if col not in exclude_cols]

    logger.info(f"Identified {len(feature_cols)} feature columns for training.")

    # Instantiate and train model
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
