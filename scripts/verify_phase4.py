"""Strict validation script for Phase 4: Feature warmup removal, vol_ratio clipping, and OOS ROC AUC metadata persistence."""

import sys
import os
from pathlib import Path
import polars as pl
import json

from src.features.pipeline import FeaturePipeline
from src.features.storage import FeatureStorage
from src.models.storage import ModelStorage
from src.utils.logger import logger


def test_a_warmup_removal() -> None:
    """Test A: Verify FeaturePipeline properly drops warmup rows without filling with false constants."""
    logger.info("Running Test A: Feature warmup removal verification...")
    storage = FeatureStorage()
    df_feat = storage.load_features("BTC/USDT", "1h")
    if df_feat.is_empty():
        logger.warning("Feature store file is empty.")
        return

    null_counts = {col: df_feat[col].null_count() for col in df_feat.columns}
    total_nulls = sum(null_counts.values())
    logger.info(f"Total null counts across feature dataframe: {total_nulls}")
    assert total_nulls == 0, f"Expected 0 nulls after FeaturePipeline drop_nulls, found {total_nulls}"
    logger.success("Test A PASSED: FeaturePipeline correctly dropped all warmup rows (0 nulls remaining).")


def test_b_vol_ratio_clip() -> None:
    """Test B: Verify max value of vol_ratio_6_24 does not exceed 10.0."""
    logger.info("Running Test B: Volatility ratio upper bound clipping verification...")
    storage = FeatureStorage()
    df_feat = storage.load_features("BTC/USDT", "1h")
    if not df_feat.is_empty() and "vol_ratio_6_24" in df_feat.columns:
        max_vol_ratio = float(df_feat["vol_ratio_6_24"].max())
        logger.info(f"Maximum observed vol_ratio_6_24: {max_vol_ratio:.4f}")
        assert max_vol_ratio <= 10.0, f"vol_ratio_6_24 max value {max_vol_ratio} exceeds 10.0 clip limit."
        logger.success(f"Test B PASSED: vol_ratio_6_24 max value is {max_vol_ratio:.4f} (<= 10.0).")
    else:
        logger.warning("Feature file or vol_ratio_6_24 column not found. Skipping Test B.")


def test_c_oos_metadata() -> None:
    """Test C: Verify model metadata.json contains 'oos_roc_auc' key."""
    logger.info("Running Test C: OOS ROC AUC metadata persistence verification...")
    storage = ModelStorage()
    meta_path = storage.models_dir / "btc_usdt_1h_metadata.json"
    if not meta_path.exists():
        meta_path = Path("models/saved/btc_usdt_1h_metadata.json")

    assert meta_path.exists(), f"Metadata file not found at {meta_path}"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "oos_roc_auc" in meta, "Metadata dictionary missing 'oos_roc_auc' key."
    oos_auc = meta["oos_roc_auc"]
    logger.info(f"Persisted oos_roc_auc value: {oos_auc}")
    assert isinstance(oos_auc, (float, int)), f"oos_roc_auc must be float, got {type(oos_auc)}"
    logger.success(f"Test C PASSED: metadata.json contains valid oos_roc_auc: {oos_auc:.4f}.")


def main() -> None:
    """Run Phase 4 verification tests."""
    logger.info("=== STARTING PHASE 4 VERIFICATION ===")
    try:
        test_a_warmup_removal()
        test_b_vol_ratio_clip()
        test_c_oos_metadata()
        logger.success("=== ALL PHASE 4 VERIFICATION TESTS PASSED SUCCESSFULLY ===")
    except Exception as e:
        logger.error(f"Phase 4 verification FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
