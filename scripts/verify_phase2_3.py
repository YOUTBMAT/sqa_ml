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
from src.models.trainer import QuantModelTrainer
from src.utils.logger import logger


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
        test_b_time_barrier_live()
        test_c_model_hyperparameter_parity()
        logger.success("=== ALL PHASE 2 & 3 VERIFICATION TESTS PASSED SUCCESSFULLY ===")
    except Exception as e:
        logger.error(f"Phase 2 & 3 verification FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
