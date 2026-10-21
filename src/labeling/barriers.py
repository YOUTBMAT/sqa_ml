"""Single source of truth for triple-barrier geometry, shared by labeling, backtest and paper execution.

Convention: entry at the close of the signal bar; horizontal barriers sit at entry +/- mult * ATR(signal bar);
they are scanned on the following bars' open/high/low; the vertical barrier exits at the close of bar
entry + max_bars.
"""

from math import ceil
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


def hours_to_bars(hours: int, timeframe: str) -> int:
    """Convert a positive holding duration in hours into a covering number of bars."""
    if isinstance(hours, bool) or not isinstance(hours, int) or hours < 1:
        raise ValueError("Holding duration must be a positive integer number of hours.")
    return ceil(hours * 3600 / timeframe_to_seconds(timeframe))


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
    max_bars: int,
) -> TradeOutcome | None:
    """Simulate one trade entered at close[entry_idx].

    Returns None when the data ends before any barrier is resolved (outcome unknown).
    """
    take_profit, stop_loss = barrier_prices(close[entry_idx], atr_value, side, pt_mult, sl_mult)
    last_idx = entry_idx + max_bars
    for j in range(entry_idx + 1, min(last_idx, len(close) - 1) + 1):
        hit = check_bar(side, take_profit, stop_loss, open_[j], high[j], low[j])
        if hit is not None:
            return TradeOutcome(j, float(hit[1]), hit[0])
    if last_idx < len(close):
        return TradeOutcome(last_idx, float(close[last_idx]), "t1")
    return None
