"""Unit tests for the barrier backtester and per-trade PerformanceMetrics."""

import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timezone, timedelta

from src.backtest.engine import VectorizedBacktester
from src.backtest.metrics import PerformanceMetrics


def _bars(close: np.ndarray) -> pl.DataFrame:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "timestamp": [t0 + timedelta(hours=i) for i in range(len(close))],
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
        }
    )


def test_backtest_runs_on_continuous_bars_and_never_overlaps() -> None:
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, 600)))
    bars = _bars(close)
    sides = np.zeros(len(close))
    sides[150:] = 1  # signal on every bar: trades must still be sequential
    probabilities = np.full(len(close), 0.9)

    bt = VectorizedBacktester(initial_capital=10000.0, fee_rate=0.0005, slippage=0.0002)
    trades = bt.run(bars, sides, probabilities)

    assert len(trades) > 5
    assert (trades["exit_time"] > trades["entry_time"]).all()
    assert (trades["entry_time"].iloc[1:].to_numpy() >= trades["exit_time"].iloc[:-1].to_numpy()).all()
    # Holding never exceeds the 24h vertical barrier in real time.
    assert (trades["exit_time"] - trades["entry_time"]).max() <= pd.Timedelta(hours=24)
    # Sized positions charge actual round-trip costs and compound dollar PnL.
    assert (trades["fees_paid"] > 0).all()
    assert (trades["slippage_cost"] > 0).all()
    assert np.allclose(
        trades["equity"].to_numpy()[1:] - trades["equity"].to_numpy()[:-1],
        trades["net_return"].to_numpy()[1:] * trades["equity"].to_numpy()[:-1],
    )
    assert (
        trades["position_notional"] <= trades["equity_before"] * 0.20 + 1e-8
    ).all()


def test_take_profit_trade_on_trending_bars() -> None:
    close = np.concatenate([100.0 * np.exp(np.cumsum(np.full(150, 0.001) * np.tile([1, -1], 75))), np.linspace(100, 120, 50)])
    bars = _bars(close)
    sides = np.zeros(len(close))
    sides[150] = 1
    probabilities = np.full(len(close), 0.9)
    trades = VectorizedBacktester().run(bars, sides, probabilities)
    assert len(trades) == 1 and trades["barrier"].iloc[0] == "tp" and trades["gross_return"].iloc[0] > 0


def test_sides_from_events_maps_by_timestamp() -> None:
    bars = _bars(np.full(10, 100.0))
    events = pl.DataFrame(
        {
            "timestamp": [bars["timestamp"][2], bars["timestamp"][5], bars["timestamp"][7]],
            "primary_signal": [1, -1, 1],
        }
    )
    sides = VectorizedBacktester.sides_from_events(bars, events, p_y1=np.array([0.7, 0.9, 0.5]), threshold=0.65)
    assert sides.tolist() == [0, 0, 1, 0, 0, -1, 0, 0, 0, 0]


def test_performance_metrics_per_trade() -> None:
    trades = pd.DataFrame({"net_return": [0.02, -0.01, 0.03, -0.01], "fees_paid": [1.0, 1.0, 1.0, 1.0]})
    trades["equity"] = 10000.0 * np.cumprod(1 + trades["net_return"])

    m = PerformanceMetrics.calculate_metrics(trades, initial_capital=10000.0, span_years=1.0)

    assert m["total_trades"] == 4
    assert m["win_rate_pct"] == 50.0
    assert np.isclose(m["profit_factor"], 0.05 / 0.02)
    assert np.isclose(m["max_drawdown_pct"], -1.0)
    assert np.isclose(m["total_fees_paid"], 4.0)
    r = trades["net_return"]
    assert np.isclose(m["sharpe_ratio"], r.mean() / r.std() * np.sqrt(4))


def test_performance_metrics_no_trades() -> None:
    m = PerformanceMetrics.calculate_metrics(pd.DataFrame(columns=["net_return", "fees_paid", "equity"]), 10000.0, 1.0)
    assert m["total_trades"] == 0 and m["final_equity"] == 10000.0
