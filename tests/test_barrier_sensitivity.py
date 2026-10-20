"""Regression tests for deterministic, horizon-bounded triple-barrier labels."""

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from src.labeling.triple_barrier import TripleBarrierLabeler


def _flat_bars(size: int, late_spike_from: int | None = None) -> pl.DataFrame:
    close = np.full(size, 100.0)
    high = np.full(size, 100.1)
    low = np.full(size, 99.9)
    if late_spike_from is not None:
        high[late_spike_from:] = 110.0
        low[late_spike_from:] = 90.0
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(size)],
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "vol_ratio_6_24": np.full(size, 0.5),
            "vwap_upper_1_dist": np.full(size, 1.0),
            "vwap_lower_1_dist": np.full(size, 1.0),
        }
    )


def test_max_holding_hours_sets_vertical_barrier() -> None:
    bars = _flat_bars(80)
    labels_12h = TripleBarrierLabeler.get_labels(bars, max_holding_hours=12)
    labels_24h = TripleBarrierLabeler.get_labels(bars, max_holding_hours=24)

    first_12h = labels_12h.row(0, named=True)
    first_24h = labels_24h.row(0, named=True)
    assert first_12h["barrier_touched"] == "t1"
    assert first_24h["barrier_touched"] == "t1"
    assert (first_12h["t1"] - first_12h["timestamp"]).total_seconds() == 12 * 3600
    assert (first_24h["t1"] - first_24h["timestamp"]).total_seconds() == 24 * 3600


def test_labels_ignore_candles_beyond_selected_horizon() -> None:
    original = _flat_bars(80)
    changed_after_barrier = _flat_bars(80, late_spike_from=14)
    base = TripleBarrierLabeler.get_labels(original, max_holding_hours=12)
    changed = TripleBarrierLabeler.get_labels(changed_after_barrier, max_holding_hours=12)

    # The event at index 0 resolves at index 12 in both datasets; altered candles start at index 14.
    for frame in (base, changed):
        first = frame.row(0, named=True)
        assert first["barrier_touched"] == "t1"
        assert first["t1"] == original["timestamp"][12]
        assert first["target_label"] == 0
    assert base.row(0, named=True)["target_return"] == changed.row(0, named=True)["target_return"]


@pytest.mark.parametrize("hours", [0, -1, 12.0, True])
def test_holding_hours_must_be_positive_integer(hours: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TripleBarrierLabeler.get_labels(_flat_bars(80), max_holding_hours=hours)


def test_conflicting_legacy_and_explicit_horizons_are_rejected() -> None:
    with pytest.raises(ValueError, match="must match"):
        TripleBarrierLabeler.get_labels(
            _flat_bars(80), num_bars=12, max_holding_hours=24
        )
