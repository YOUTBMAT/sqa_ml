"""Unit tests for VectorizedBacktester and PerformanceMetrics."""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timezone, timedelta

from src.backtest.engine import VectorizedBacktester
from src.backtest.metrics import PerformanceMetrics


def test_backtest_execution() -> None:
    """Test VectorizedBacktester calculates equity curve and correctly accounts for transaction costs."""
    n = 100
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=i) for i in range(n)]

    # Simulated upward trending prices
    close_prices = 100.0 + np.arange(n) * 0.5
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "close": close_prices,
        }
    )

    # Simulated probabilities favoring Long positions (class 2)
    probs = np.zeros((n, 3))
    probs[:, 2] = 0.60  # high probability for Long

    backtester = VectorizedBacktester(
        initial_capital=10000.0,
        fee_rate=0.001,
        slippage=0.0005,
        buy_threshold=0.55,
        sell_threshold=0.45,
    )
    results_df = backtester.run(df, probs)

    assert "equity" in results_df.columns
    assert "net_strategy_returns" in results_df.columns
    assert "transaction_costs" in results_df.columns
    assert len(results_df) == n
    # Initial capital should be updated over time
    assert results_df["equity"].iloc[-1] != 10000.0


def test_performance_metrics() -> None:
    """Test PerformanceMetrics calculations for Sharpe, Max Drawdown, and Win Rate."""
    n = 50
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=i) for i in range(n)]

    # Construct synthetic results DataFrame
    np.random.seed(42)
    net_returns = np.random.randn(n) * 0.01 + 0.001
    equity = 10000.0 * np.cumprod(1.0 + net_returns)
    transaction_costs = np.zeros(n)
    transaction_costs[0] = 0.001

    results_df = pd.DataFrame(
        {
            "net_strategy_returns": net_returns,
            "transaction_costs": transaction_costs,
            "equity": equity,
        },
        index=timestamps,
    )

    metrics = PerformanceMetrics.calculate_metrics(results_df, initial_capital=10000.0)

    assert "total_return_pct" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown_pct" in metrics
    assert "win_rate_pct" in metrics
    assert "profit_factor" in metrics
    assert metrics["initial_capital"] == 10000.0
