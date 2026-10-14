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
        # Initialize concurrency series with zeros
        concurrency = pd.Series(0.0, index=close_idx)

        # For each event, increment concurrency between start and t1
        for start_time, row in barrier_events.iterrows():
            end_time = row["t1"]
            if pd.isna(end_time):
                continue
            # Find index slice between start_time and end_time
            active_slice = close_idx[(close_idx >= start_time) & (close_idx <= end_time)]
            if not active_slice.empty:
                concurrency.loc[active_slice] += 1.0

        return concurrency

    @staticmethod
    def get_sample_uniqueness(barrier_events: pd.DataFrame, num_co_events: pd.Series) -> pd.Series:
        """Compute average uniqueness for each sample over its lifespan [t0, t1]."""
        logger.info("Computing sample uniqueness...")
        uniqueness = pd.Series(index=barrier_events.index, dtype=float)

        for start_time, row in barrier_events.iterrows():
            end_time = row["t1"]
            if pd.isna(end_time):
                uniqueness.loc[start_time] = 1.0
                continue

            # Get concurrency values during the event lifespan
            co_slice = num_co_events.loc[(num_co_events.index >= start_time) & (num_co_events.index <= end_time)]
            if co_slice.empty or (co_slice == 0).all():
                uniqueness.loc[start_time] = 1.0
            else:
                # Uniqueness at each bar is 1 / co_events
                inv_co = 1.0 / co_slice
                uniqueness.loc[start_time] = inv_co.mean()

        return uniqueness.fillna(1.0)

    @classmethod
    def get_sample_weights(
        cls, barrier_events: pd.DataFrame, uniqueness: pd.Series, target_returns: pd.Series
    ) -> pd.Series:
        """Compute final sample weights combining uniqueness and absolute event returns."""
        logger.info("Computing final sample weights...")
        # Absolute return magnitude as performance weight proxy
        abs_returns = target_returns.abs()
        
        # Raw weights = uniqueness * abs_return
        raw_weights = uniqueness * abs_returns

        # Normalize weights to sum to number of samples (or mean of 1.0)
        if not raw_weights.empty and raw_weights.sum() > 0:
            weights = raw_weights * (len(raw_weights) / raw_weights.sum())
        else:
            weights = pd.Series(1.0, index=barrier_events.index)

        return weights.fillna(1.0)
