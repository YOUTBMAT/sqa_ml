"""Strict validation script for Phase 2 & 3: Unified Triple Barrier, Continuous Backtest Engine, 24h Time Limit, and Parameter Parity."""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timezone, timedelta

from config.settings import settings
from src.execution.paper import PaperTrader
from src.backtest.engine import VectorizedBacktester
from src.models.trainer import QuantModelTrainer
from src.utils.logger import logger


def test_a_trade_duration() -> None:
    """Test A: Verify median trade duration in continuous backtest is <= 24 hours."""
    logger.info("Running Test A: Continuous backtest trade duration verification...")
    
    dataset_path = Path("sqa_ml/data/datasets/btc_usdt_1h_dataset.parquet")
    if not dataset_path.exists():
        dataset_path = Path("data/datasets/btc_usdt_1h_dataset.parquet")
    
    if not dataset_path.exists():
        logger.warning("Dataset not found. Skipping Test A (will be verified post dataset rebuild).")
        return

    df = pl.read_parquet(dataset_path)
    n = len(df)
    probs = np.zeros((n, 3))
    probs[:, 1] = 1.0
    for i in range(0, n - 1, 50):
        probs[i, 2] = 0.9

    backtester = VectorizedBacktester(buy_threshold=0.75, sell_threshold=0.25)
    results_df = backtester.run(df, probs)

    positions = results_df["position"].values
    durations = []
    current_duration = 0
    in_trade = False

    for pos in positions:
        if pos != 0:
            if not in_trade:
                in_trade = True
                current_duration = 1
            else:
                current_duration += 1
        else:
            if in_trade:
                durations.append(current_duration)
                in_trade = False
                current_duration = 0
    if in_trade:
        durations.append(current_duration)

    median_dur = float(np.median(durations)) if durations else 0.0
    logger.info(f"Observed median trade duration: {median_dur:.1f} hours (Max allowed: 24h)")
    assert median_dur <= 24.0, f"Median trade duration {median_dur}h exceeds 24h limit."
    logger.success(f"Test A PASSED: Median trade duration is {median_dur:.1f}h (<= 24h).")


def test_b_time_barrier_live() -> None:
    """Test B: Verify PaperTrader closes position after 24h due to TIME_BARRIER_24H."""
    logger.info("Running Test B: Live 24h time barrier execution verification...")

    state_dir = Path("data/paper")
    state_dir.mkdir(parents=True, exist_ok=True)
    trader = PaperTrader(initial_balance=10000.0, load_saved_state=False, state_dir=str(state_dir))
    symbol = "BTC/USDT"
    price = 50000.0

    trader.execute_paper_order(symbol, "buy", 0.1, price, stop_loss=40000.0, take_profit=60000.0)
    assert trader.has_open_position(symbol)

    old_time = datetime.now(timezone.utc) - timedelta(hours=25)
    trader.positions[symbol]["timestamp"] = old_time

    events = trader.update_positions({symbol: price})

    assert len(events) == 1, "Expected 1 triggered event from time barrier."
    assert events[0]["exit_reason"] == "TIME_BARRIER_24H", f"Expected exit_reason 'TIME_BARRIER_24H', got {events[0].get('exit_reason')}"
    assert not trader.has_open_position(symbol), "Position should be closed after time barrier."
    logger.success("Test B PASSED: Position successfully closed by TIME_BARRIER_24H after 25 hours.")


def test_c_model_hyperparameter_parity() -> None:
    """Test C: Validate that QuantModelTrainer and scripts use centralized settings.MODEL_PARAMS."""
    logger.info("Running Test C: Model hyperparameter parity verification...")

    trainer = QuantModelTrainer()
    for param_name, expected_val in settings.MODEL_PARAMS.items():
        actual_val = getattr(trainer.model, param_name, None)
        assert actual_val == expected_val, f"Hyperparameter {param_name} mismatch: expected {expected_val}, got {actual_val}"

    logger.success("Test C PASSED: QuantModelTrainer parameters strictly match config/settings.py MODEL_PARAMS.")


def main() -> None:
    """Run Phase 2 & 3 verification tests."""
    logger.info("=== STARTING PHASE 2 & 3 VERIFICATION ===")
    try:
        test_a_trade_duration()
        test_b_time_barrier_live()
        test_c_model_hyperparameter_parity()
        logger.success("=== ALL PHASE 2 & 3 VERIFICATION TESTS PASSED SUCCESSFULLY ===")
    except Exception as e:
        logger.error(f"Phase 2 & 3 verification FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
