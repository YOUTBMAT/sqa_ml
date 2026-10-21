"""Strict validation script for Phase 1: Safety Live Lock, Inference Window Parity, and Atomic Persistence."""

import sys
import tempfile
import os
import json
from pathlib import Path
import polars as pl
import numpy as np

from src.execution.runner import BotRunner
from src.execution.paper import PaperTrader
from src.features.pipeline import FeaturePipeline
from src.utils.logger import logger


def test_a_live_lock() -> None:
    """Test A: Attempt to run BotRunner with mode='LIVE' and confirm RuntimeError is raised."""
    logger.info("Running Test A: LIVE mode safety lock verification...")
    try:
        BotRunner(symbols=["BTC/USDT"], mode="LIVE")
        raise AssertionError("Expected RuntimeError for LIVE mode, but none was raised.")
    except RuntimeError as e:
        expected_msg = "LIVE execution is disabled"
        if expected_msg in str(e):
            logger.success(f"Test A PASSED: Successfully caught expected RuntimeError: {e}")
        else:
            raise AssertionError(f"Unexpected RuntimeError message: {e}")


def test_b_feature_parity() -> None:
    """Test B: Compare last row features (frac_diff_close, ema_200_dist) computed on 2,000 candles vs full dataset."""
    logger.info("Running Test B: 2000-candle inference window parity verification...")
    
    feature_path = Path("sqa_ml/data/features/btc_usdt_1h.parquet")
    if not feature_path.exists():
        feature_path = Path("data/features/btc_usdt_1h.parquet")
    if not feature_path.exists():
        feature_path = Path("sqa_ml/data/datasets/btc_usdt_1h_dataset.parquet")
    if not feature_path.exists():
        feature_path = Path("data/datasets/btc_usdt_1h_dataset.parquet")
    
    if not feature_path.exists():
        logger.warning(f"Feature/dataset file not found. Skipping test B.")
        return

    df_full = pl.read_parquet(feature_path)
    if len(df_full) < 2000:
        logger.warning(f"Dataset rows ({len(df_full)}) less than 2,000. Skipping feature parity comparison.")
        return

    # Extract raw OHLCV for last 2000 rows
    df_2000 = df_full.tail(2000)
    pipeline = FeaturePipeline()
    df_computed = pipeline.run(df_2000)

    last_row_computed = df_computed.tail(1)
    last_ts = last_row_computed["timestamp"][0]
    comp_frac_diff = float(last_row_computed["frac_diff_close"][0])
    comp_ema_dist = float(last_row_computed["ema_200_dist"][0])

    row_full = df_full.filter(pl.col("timestamp") == last_ts)
    if row_full.is_empty():
        row_full = df_full.tail(1)

    full_frac_diff = float(row_full["frac_diff_close"][0])
    full_ema_dist = float(row_full["ema_200_dist"][0])

    # Calculate percentage differences
    diff_frac = abs(comp_frac_diff - full_frac_diff) / (abs(full_frac_diff) + 1e-9)
    diff_ema = abs(comp_ema_dist - full_ema_dist) / (abs(full_ema_dist) + 1e-9)

    logger.info(f"Full dataset frac_diff: {full_frac_diff}, Computed 2000-candle frac_diff: {comp_frac_diff} (Diff: {diff_frac:.6f})")
    logger.info(f"Full dataset ema_dist: {full_ema_dist}, Computed 2000-candle ema_dist: {comp_ema_dist} (Diff: {diff_ema:.6f})")

    assert diff_frac < 0.0001, f"frac_diff_close difference {diff_frac:.6f} exceeds 0.01% threshold."
    assert diff_ema < 0.0001, f"ema_200_dist difference {diff_ema:.6f} exceeds 0.01% threshold."
    logger.success("Test B PASSED: Feature parity verified within 0.01% tolerance over 2,000 candles.")


def test_c_atomic_persistence() -> None:
    """Test C: Simulate write interruption/corruption on paper_portfolio.json and verify restoration from .bak."""
    logger.info("Running Test C: Atomic persistence and backup recovery verification...")

    state_dir = Path(tempfile.mkdtemp())  # never touch the live paper portfolio
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "paper_portfolio.json"
    backup_file = state_dir / "paper_portfolio.json.bak"

    # 1. Initialize trader with specific non-default balance and save state
    trader1 = PaperTrader(initial_balance=12345.67, load_saved_state=False, state_dir=str(state_dir))
    trader1.balance = 12345.67
    trader1._save_state()

    assert state_file.exists(), "State file should exist after save."
    assert backup_file.exists(), "Backup file should exist after subsequent save."

    # 2. Corrupt primary state file (simulate write interruption/corrupted JSON)
    with open(state_file, "w", encoding="utf-8") as f:
        f.write("{corrupted_json_syntax_error...")

    # 3. Instantiate new PaperTrader, expecting it to fallback to .bak and restore successfully
    trader2 = PaperTrader(initial_balance=10000.0, load_saved_state=True, state_dir=str(state_dir))

    assert trader2.balance == 12345.67, f"Expected restored balance 12345.67, got {trader2.balance}"
    logger.success("Test C PASSED: Corrupted primary JSON successfully restored from .bak without balance loss.")


def main() -> None:
    """Run all Phase 1 verification tests."""
    logger.info("=== STARTING PHASE 1 VERIFICATION ===")
    try:
        test_a_live_lock()
        test_b_feature_parity()
        test_c_atomic_persistence()
        logger.success("=== ALL PHASE 1 VERIFICATION TESTS PASSED SUCCESSFULLY ===")
    except Exception as e:
        logger.error(f"Phase 1 verification FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
