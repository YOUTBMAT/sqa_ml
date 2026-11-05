"""Single source of truth for triple-barrier geometry.

Convention: entry at the close of the signal bar; horizontal barriers sit at
entry +/- mult * ATR(signal bar); they are scanned on subsequent available
bars. The vertical barrier is defined by elapsed timestamp duration.
"""

import re
from typing import NamedTuple

import numpy as np
import pandas as pd


class TradeOutcome(NamedTuple):
    exit_idx: int
    exit_price: float
    barrier: str  # "tp", "sl" or "t1"


def timeframe_to_seconds(timeframe: str) -> int:
    """Convert a fixed-duration market timeframe such as 15m or 4h to seconds."""
    match = re.fullmatch(r"([1-9]\d*)(m|h|d|w)", timeframe)
    if match is None:
        raise ValueError(f"Unsupported fixed-duration timeframe: {timeframe!r}")
    amount = int(match.group(1))
    unit_seconds = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    return amount * unit_seconds[match.group(2)]


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range (EWM of the true range), known at each bar's close."""
    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return true_range.ewm(span=period).mean()


def barrier_prices(entry_price: float, atr_value: float, side: int, pt_mult: float, sl_mult: float) -> tuple[float, float]:
    """Return (take_profit, stop_loss) prices for side +1 (long) or -1 (short)."""
    take_profit = entry_price + side * pt_mult * atr_value
    stop_loss = entry_price - side * sl_mult * atr_value
    return take_profit, stop_loss


def check_bar(
    side: int, take_profit: float, stop_loss: float, bar_open: float, bar_high: float, bar_low: float
) -> tuple[str, float] | None:
    """Check one bar against the horizontal barriers.

    The stop wins when a bar touches both (intrabar order is unknown, so assume the worst).
    A gap through a barrier fills at the bar open.
    """
    if side == 1:
        if bar_low <= stop_loss:
            return "sl", min(stop_loss, bar_open)
        if bar_high >= take_profit:
            return "tp", max(take_profit, bar_open)
        return None
    if bar_high >= stop_loss:
        return "sl", max(stop_loss, bar_open)
    if bar_low <= take_profit:
        return "tp", min(take_profit, bar_open)
    return None


def simulate_trade(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_idx: int,
    side: int,
    atr_value: float,
    pt_mult: float,
    sl_mult: float,
    max_holding_hours: int,
    timestamps: pd.Series | np.ndarray | None = None,
) -> TradeOutcome | None:
    """Simulate one trade using an elapsed-time vertical barrier.

    The first available bar whose timestamp is at or after
    ``entry_timestamp + max_holding_hours`` is the vertical-barrier bar.
    Missing bars therefore extend the index distance, but never extend the
    real-time holding period.

    ``timestamps`` must contain the timestamp associated with each price.
    """
    if max_holding_hours < 1:
        raise ValueError("max_holding_hours must be positive.")
    if isinstance(max_holding_hours, bool) or not isinstance(max_holding_hours, int):
        raise ValueError("max_holding_hours must be a positive integer.")
    arrays = (open_, high, low, close)
    if any(len(values) != len(close) for values in arrays):
        raise ValueError("All price arrays must have the same length.")
    if side not in (-1, 1):
        raise ValueError("side must be either 1 or -1.")
    if not np.isfinite(atr_value) or atr_value <= 0:
        raise ValueError("atr_value must be finite and positive.")

    if timestamps is None:
        # Backwards-compatible test/helper path; production callers must pass
        # the market timestamps so elapsed time reflects actual data gaps.
        timestamp_values = pd.date_range(
            "1970-01-01", periods=len(close), freq="h"
        )
    else:
        timestamp_values = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps)))
    if len(timestamp_values) != len(close):
        raise ValueError("timestamps must have the same length as the price arrays.")
    if timestamp_values.isna().any() or timestamp_values.has_duplicates:
        raise ValueError("timestamps must be unique and cannot contain NaT.")
    if entry_idx < 0 or entry_idx >= len(close):
        raise IndexError("entry_idx is outside the price arrays.")
    if not timestamp_values.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in ascending order.")

    take_profit, stop_loss = barrier_prices(
        close[entry_idx], atr_value, side, pt_mult, sl_mult
    )
    vertical_deadline = timestamp_values[entry_idx] + pd.Timedelta(
        hours=max_holding_hours
    )

    expiry_idx = timestamp_values.searchsorted(vertical_deadline, side="left")
    if expiry_idx <= entry_idx:
        expiry_idx = entry_idx + 1

    for j in range(entry_idx + 1, min(expiry_idx, len(close))):
        hit = check_bar(
            side,
            take_profit,
            stop_loss,
            open_[j],
            high[j],
            low[j],
        )
        if hit is not None:
            return TradeOutcome(j, float(hit[1]), hit[0])

    if expiry_idx < len(close):
        return TradeOutcome(expiry_idx, float(close[expiry_idx]), "t1")

    return None
