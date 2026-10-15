"""Tests for the shared triple-barrier geometry used by labels, backtest and paper execution."""

import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timezone, timedelta

from config.settings import settings
from src.labeling.barriers import atr, barrier_prices, check_bar, simulate_trade
from src.labeling.triple_barrier import TripleBarrierLabeler


def test_barrier_prices_long_and_short() -> None:
    tp, sl = barrier_prices(100.0, 1.0, 1, 1.5, 1.0)
    assert np.isclose(tp, 101.5) and np.isclose(sl, 99.0)
    tp, sl = barrier_prices(100.0, 1.0, -1, 1.5, 1.0)
    assert np.isclose(tp, 98.5) and np.isclose(sl, 101.0)


def test_check_bar_stop_wins_ambiguous_bar_and_gaps_fill_at_open() -> None:
    # Bar touches both barriers: stop-loss is assumed first (conservative).
    assert check_bar(1, 101.5, 99.0, 100.0, 102.0, 98.0) == ("sl", 99.0)
    # Gap below the stop: fill at the worse open, not at the stop.
    assert check_bar(1, 101.5, 99.0, 97.0, 97.5, 96.0) == ("sl", 97.0)
    # Short take-profit touched by the low.
    assert check_bar(-1, 98.5, 101.0, 100.0, 100.5, 98.0) == ("tp", 98.5)
    assert check_bar(1, 101.5, 99.0, 100.0, 100.5, 99.5) is None


def test_simulate_trade_vertical_barrier_and_end_of_data() -> None:
    flat = np.full(30, 100.0)
    out = simulate_trade(flat, flat, flat, flat, 0, 1, 1.0, 1.5, 1.0, 24)
    assert out is not None and out.barrier == "t1" and out.exit_idx == 24 and out.exit_price == 100.0
    # Not enough future bars to reach the vertical barrier and nothing touched: unknown outcome.
    assert simulate_trade(flat, flat, flat, flat, 10, 1, 1.0, 1.5, 1.0, 24) is None


def test_simulate_trade_ignores_entry_bar_range() -> None:
    # The entry bar's own high is above TP, but it happened before the entry at its close.
    o = np.full(5, 100.0)
    h = np.array([110.0, 100.0, 100.0, 100.0, 100.0])
    low = np.full(5, 100.0)
    c = np.full(5, 100.0)
    out = simulate_trade(o, h, low, c, 0, 1, 1.0, 1.5, 1.0, 4)
    assert out.barrier == "t1"


def _bars(close: np.ndarray, high_pad: float = 0.0) -> pl.DataFrame:
    n = len(close)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "timestamp": [t0 + timedelta(hours=i) for i in range(n)],
            "open": close,
            "high": close + high_pad,
            "low": close - high_pad,
            "close": close,
            "vol_ratio_6_24": np.full(n, 0.5),
            "vwap_upper_1_dist": np.full(n, 1.0),
            "vwap_lower_1_dist": np.full(n, 1.0),
        }
    )


def test_labels_use_intrabar_touch_and_touch_time_as_t1() -> None:
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 400)))
    df = _bars(close, high_pad=0.1)
    labels = TripleBarrierLabeler.get_labels(df, pt_sl=[1.5, 1.0], num_bars=24)

    assert not labels.is_empty()
    assert set(labels["barrier_touched"].unique().to_list()) <= {"tp", "sl", "t1"}
    assert (labels["target_label"] == (labels["barrier_touched"] == "tp").cast(pl.Int64)).all()
    # t1 is the actual exit time: after entry and at most 24 bars later.
    delta_h = (labels["t1"] - labels["timestamp"]).dt.total_hours()
    assert delta_h.min() >= 1 and delta_h.max() <= 24
    # The last 24 bars cannot be labeled without future data unless a barrier was touched.
    last_ts = df["timestamp"][-1]
    unresolved = labels.filter((pl.col("barrier_touched") == "t1") & (pl.col("t1") > last_ts))
    assert unresolved.is_empty()


def test_label_matches_backtest_outcome() -> None:
    """Label and backtest come from the same simulator, so they must agree trade by trade."""
    rng = np.random.default_rng(1)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 300)))
    df = _bars(close, high_pad=0.2)
    labels = TripleBarrierLabeler.get_labels(df, pt_sl=[1.5, 1.0], num_bars=24).to_pandas()
    pdf = df.to_pandas()
    atr_values = atr(pdf["high"], pdf["low"], pdf["close"], settings.ATR_PERIOD).to_numpy()
    idx = {ts: i for i, ts in enumerate(pd.to_datetime(pdf["timestamp"]))}
    for _, row in labels.head(20).iterrows():
        i = idx[pd.Timestamp(row["timestamp"])]
        out = simulate_trade(
            pdf["open"].to_numpy(), pdf["high"].to_numpy(), pdf["low"].to_numpy(), pdf["close"].to_numpy(),
            i, int(row["primary_signal"]), atr_values[i], 1.5, 1.0, 24,
        )
        assert out.barrier == row["barrier_touched"]
