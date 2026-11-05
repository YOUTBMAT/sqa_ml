"""Purged Group Time Series Cross-Validation with Embargo (Marcos López de Prado)."""

import numpy as np
import pandas as pd
import polars as pl
from typing import Generator
from src.utils.logger import logger


class PurgedGroupTimeSeriesSplit:
    """Time series cross-validation with purging and embargoing to prevent leakage."""

    def __init__(self, n_splits: int = 5, pct_embargo: float = 0.01) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if not 0 <= pct_embargo < 1:
            raise ValueError("pct_embargo must be in [0, 1).")
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo

    def split(
        self,
        X: pd.DataFrame | pl.DataFrame,
        y: pd.Series | np.ndarray = None,
        groups: pd.Series | np.ndarray = None,
        event_times: pd.DataFrame | pd.Series = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate time-anchored expanding walk-forward train/test indices with purging."""
        logger.info(f"Generating {self.n_splits} strict time-anchored expanding walk-forward splits (embargo={self.pct_embargo})...")

        if isinstance(X, pl.DataFrame):
            X = X.to_pandas()

        if "timestamp" in X.columns and "t1" in X.columns:
            t0 = pd.to_datetime(X["timestamp"])
            t1 = pd.to_datetime(X["t1"])
        elif event_times is not None:
            if isinstance(event_times, pl.DataFrame):
                event_times = event_times.to_pandas()
            if isinstance(event_times, pd.Series):
                t0 = pd.to_datetime(X["timestamp"] if "timestamp" in X.columns else X.index)
                t1 = pd.to_datetime(event_times)
            else:
                t0 = pd.to_datetime(
                    event_times["timestamp"] if "timestamp" in event_times else X.index
                )
                t1 = pd.to_datetime(event_times["t1"])
        else:
            t0 = pd.to_datetime(X["timestamp"] if "timestamp" in X.columns else X.index)
            t1 = t0

        if t0.isna().any() or t1.isna().any():
            raise ValueError("Cross-validation timestamps cannot contain NaT.")
        if not t0.is_monotonic_increasing:
            raise ValueError("Cross-validation input must be sorted by timestamp.")
        if t0.duplicated().any():
            raise ValueError("Cross-validation timestamps must be unique.")

        indices = np.arange(len(X))
        test_folds = np.array_split(indices, self.n_splits)

        for i, test_idx in enumerate(test_folds):
            if len(test_idx) == 0:
                continue

            test_start = t0.iloc[test_idx[0]]
            test_idx_start = test_idx[0]

            # 1. Strict time-anchored expanding walk-forward: train must contain ONLY data prior to test start (train_idx < test_idx_start)
            train_idx = indices[indices < test_idx_start]
            if len(train_idx) == 0:
                logger.warning(f"Fold {i+1}: Train index set is empty (test_idx_start={test_idx_start}). Skipping fold.")
                continue

            # 2. Purging: remove training samples whose vertical barrier t1 invades the test window (t1 >= test_start)
            train_t1 = t1.iloc[train_idx]
            purge_mask = train_t1 >= test_start
            purge_train_idx = train_idx[purge_mask]
            train_idx = np.setdiff1d(train_idx, purge_train_idx)

            # 3. Explicit temporal embargo immediately before the test window.
            # This creates a real time gap between the latest training event and
            # the validation period, even though the walk-forward train set only
            # contains observations before the test window.
            if self.pct_embargo > 0 and len(train_idx) > 0:
                test_end = t0.iloc[test_idx[-1]]
                embargo_delta = (test_end - test_start) * self.pct_embargo
                embargo_start = test_start - embargo_delta
                train_idx = train_idx[t0.iloc[train_idx].to_numpy() < embargo_start]

            logger.info(
                f"Fold {i+1}/{self.n_splits}: Test size = {len(test_idx)}, "
                f"Train size (after time-anchor/purge/embargo) = {len(train_idx)}"
            )
            yield train_idx, test_idx
