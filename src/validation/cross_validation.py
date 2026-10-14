"""Purged Group Time Series Cross-Validation with Embargo (Marcos López de Prado)."""

import numpy as np
import pandas as pd
import polars as pl
from typing import Generator
from src.utils.logger import logger


class PurgedGroupTimeSeriesSplit:
    """Time series cross-validation with purging and embargoing to prevent leakage."""

    def __init__(self, n_splits: int = 5, pct_embargo: float = 0.01) -> None:
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo

    def split(
        self,
        X: pd.DataFrame | pl.DataFrame,
        y: pd.Series | np.ndarray = None,
        groups: pd.Series | np.ndarray = None,
        event_times: pd.DataFrame | pd.Series = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate purged and embargoed train/test indices."""
        logger.info(f"Generating {self.n_splits} purged & embargoed CV splits (embargo={self.pct_embargo})...")

        if isinstance(X, pl.DataFrame):
            X = X.to_pandas()

        if "timestamp" in X.columns and "t1" in X.columns:
            t0 = pd.to_datetime(X["timestamp"])
            t1 = pd.to_datetime(X["t1"])
        elif event_times is not None:
            if isinstance(event_times, pl.DataFrame):
                event_times = event_times.to_pandas()
            t0 = pd.to_datetime(event_times["timestamp"] if "timestamp" in event_times else X.index)
            t1 = pd.to_datetime(event_times["t1"])
        else:
            t0 = pd.to_datetime(X["timestamp"] if "timestamp" in X.columns else X.index)
            t1 = t0

        indices = np.arange(len(X))
        test_folds = np.array_split(indices, self.n_splits)

        for i, test_idx in enumerate(test_folds):
            if len(test_idx) == 0:
                continue

            test_start = t0.iloc[test_idx[0]]
            test_end = t1.iloc[test_idx[-1]]

            # 1. Train indices
            train_idx = np.setdiff1d(indices, test_idx)

            # 2. Purging: remove train samples whose lifespan [t0, t1] overlaps test period
            train_t0 = t0.iloc[train_idx]
            train_t1 = t1.iloc[train_idx]

            overlap_mask = ~((train_t1 < test_start) | (train_t0 > test_end))
            purge_train_idx = train_idx[overlap_mask]
            train_idx = np.setdiff1d(train_idx, purge_train_idx)

            # 3. Embargo: remove train samples immediately following test fold
            if self.pct_embargo > 0 and len(test_idx) > 0:
                embargo_val = int(len(X) * self.pct_embargo)
                test_max_pos = test_idx[-1]
                embargo_end_pos = min(test_max_pos + embargo_val, len(X))
                
                embargo_mask = (train_idx > test_max_pos) & (train_idx <= embargo_end_pos)
                embargo_train_idx = train_idx[embargo_mask]
                train_idx = np.setdiff1d(train_idx, embargo_train_idx)

            logger.info(
                f"Fold {i+1}/{self.n_splits}: Test size = {len(test_idx)}, "
                f"Train size (after purge/embargo) = {len(train_idx)}"
            )
            yield train_idx, test_idx
