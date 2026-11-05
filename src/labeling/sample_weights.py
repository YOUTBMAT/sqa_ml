"""Sample weights and concurrency uniqueness module (Marcos López de Prado)."""

import pandas as pd
import numpy as np
from src.utils.logger import logger


class SampleWeights:
    """Class for computing sample concurrency, uniqueness, and return-weighted sample weights."""

    @staticmethod
    def get_num_co_events(close_idx: pd.DatetimeIndex, barrier_events: pd.DataFrame) -> pd.Series:
        """Compute the number of concurrent active events at each timestamp t.

        barrier_events columns required:
        - index: start timestamp (t0)
        - t1: end/touch timestamp (t1)
        """
        logger.info("Computing concurrent active events...")
        if not close_idx.is_monotonic_increasing or close_idx.has_duplicates:
            raise ValueError("close_idx must be sorted and unique.")
        timestamps = pd.DatetimeIndex(pd.to_datetime(close_idx))
        delta = np.zeros(len(timestamps) + 1, dtype=float)
        for start_time, row in barrier_events.iterrows():
            end_time = row["t1"]
            if pd.isna(end_time):
                continue
            start = timestamps.searchsorted(start_time, side="left")
            end = timestamps.searchsorted(end_time, side="right")
            if start < end:
                delta[start] += 1.0
                delta[end] -= 1.0
        return pd.Series(np.cumsum(delta[:-1]), index=close_idx)

    @staticmethod
    def get_sample_uniqueness(barrier_events: pd.DataFrame, num_co_events: pd.Series) -> pd.Series:
        """Compute average uniqueness for each sample over its lifespan [t0, t1]."""
        logger.info("Computing sample uniqueness...")
        uniqueness = pd.Series(index=barrier_events.index, dtype=float)

        if not num_co_events.index.is_monotonic_increasing:
            raise ValueError("num_co_events must be sorted by timestamp.")
        timestamps = pd.DatetimeIndex(num_co_events.index)
        values = num_co_events.to_numpy(dtype=float)
        for start_time, row in barrier_events.iterrows():
            end_time = row["t1"]
            if pd.isna(end_time):
                uniqueness.loc[start_time] = 1.0
                continue

            start = timestamps.searchsorted(start_time, side="left")
            end = timestamps.searchsorted(end_time, side="right")
            co_slice = values[start:end]
            if co_slice.size == 0 or np.all(co_slice == 0):
                uniqueness.loc[start_time] = 1.0
            else:
                uniqueness.loc[start_time] = np.mean(1.0 / co_slice)

        return uniqueness.fillna(1.0)

    @classmethod
    def get_sample_weights(
        cls, barrier_events: pd.DataFrame, uniqueness: pd.Series, target_returns: pd.Series
    ) -> pd.Series:
        """Compute raw sample weights without cross-window normalization."""
        if not barrier_events.index.equals(uniqueness.index):
            raise ValueError("barrier_events and uniqueness must have identical indexes.")
        if not barrier_events.index.equals(target_returns.index):
            raise ValueError("barrier_events and target_returns must have identical indexes.")

        logger.info("Computing raw sample weights...")
        return (
            uniqueness.astype(float)
            * target_returns.abs().astype(float)
        ).replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
