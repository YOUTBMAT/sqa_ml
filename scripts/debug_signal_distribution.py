"""Diagnostic script for analyzing primary signal distribution and P(y=1) probabilities."""

import numpy as np
import polars as pl
import pandas as pd
from src.data.dataset import DatasetBuilder
from src.models.trainer import QuantModelTrainer
from src.models.storage import ModelStorage
from src.labeling.triple_barrier import TripleBarrierLabeler
from src.features.storage import FeatureStorage
from src.utils.logger import logger


def main() -> None:
    logger.info("=== DEBUGGING SIGNAL DISTRIBUTION & PROBABILITIES ===")
    builder = DatasetBuilder()
    df_dataset = builder.load_dataset("BTC/USDT", "1h")
    if df_dataset.is_empty():
        logger.error("Dataset empty.")
        return

    print(f"\nTotal dataset rows (filtered active signals): {len(df_dataset)}")

    # 1. Base label rate of target_label
    target_labels = df_dataset["target_label"].to_numpy()
    num_ones = (target_labels == 1).sum()
    num_zeros = (target_labels == 0).sum()
    total_labels = len(target_labels)
    print(f"\n--- TRIPLE BARRIER LABEL RATE ---")
    print(f"Total labeled samples: {total_labels}")
    print(f"Class 1 (Success): {num_ones} ({num_ones / total_labels * 100:.2f}%)")
    print(f"Class 0 (Fail): {num_zeros} ({num_zeros / total_labels * 100:.2f}%)")

    # 2. Primary signals across all candles
    feature_storage = FeatureStorage()
    df_features = feature_storage.load_features("BTC/USDT", "1h")
    pdf_features = df_features.to_pandas()
    primary_signals_all = TripleBarrierLabeler.get_primary_signal(pdf_features)
    
    num_long = (primary_signals_all == 1).sum()
    num_short = (primary_signals_all == -1).sum()
    num_zero = (primary_signals_all == 0).sum()
    total_candles = len(primary_signals_all)

    print(f"\n--- PRIMARY SIGNAL COUNTS (ALL CANDLES: {total_candles}) ---")
    print(f"Long signals (+1): {num_long} ({num_long / total_candles * 100:.2f}%)")
    print(f"Short signals (-1): {num_short} ({num_short / total_candles * 100:.2f}%)")
    print(f"Neutral signals (0): {num_zero} ({num_zero / total_candles * 100:.2f}%)")
    print(f"Total active primary signals (!= 0): {num_long + num_short}")

    # 3. Model OOS Probabilities P(y=1) on dataset samples
    storage = ModelStorage()
    try:
        model, meta = storage.load_model("BTC/USDT", "1h")
        feature_cols = meta.get("feature_cols", [])
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    trainer = QuantModelTrainer()
    oos_probs = trainer.cross_val_predict_walk_forward(df_dataset, feature_cols=feature_cols, n_splits=5, pct_embargo=0.01)
    p_y1 = oos_probs[:, 1]

    print(f"\n--- OOS PROBABILITIES P(y=1) DESCRIPTIVE STATISTICS ---")
    print(f"Min: {p_y1.min():.6f}")
    print(f"Max: {p_y1.max():.6f}")
    print(f"Mean: {p_y1.mean():.6f}")
    print(f"Median: {np.median(p_y1):.6f}")
    print(f"75%: {np.percentile(p_y1, 75):.6f}")
    print(f"90%: {np.percentile(p_y1, 90):.6f}")
    print(f"95%: {np.percentile(p_y1, 95):.6f}")
    print(f"99%: {np.percentile(p_y1, 99):.6f}")

    # 4. Sensitivity test: trades generated with threshold = 0.00 (Primary Signal only)
    active_primary = df_dataset["primary_signal"].to_numpy()
    print(f"\n--- SENSITIVITY TEST (Primary Signal Only, threshold = 0.00) ---")
    print(f"Trades generated without ML filter (Primary Signal count): {len(active_primary)}")


if __name__ == "__main__":
    main()
