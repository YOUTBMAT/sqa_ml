"""Purged walk-forward mean-decrease-accuracy feature selection."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from src.labeling.sample_weights import SampleWeights


class PurgedFeatureSelector:
    """Measure OOS ROC AUC degradation after permuting each feature."""

    @staticmethod
    def calculate_mda(
        model,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        cv_splits: Iterable[tuple[np.ndarray, np.ndarray]],
        sample_weight: pd.Series | np.ndarray | None = None,
        event_metadata: pd.DataFrame | None = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Return mean and standard deviation of per-fold permutation MDA scores.

        The supplied folds must come from the purged walk-forward splitter. Each
        fold model is fitted only on its training indices; permutation is applied
        to a copy of that fold's OOS validation data.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with named feature columns.")
        if X.empty or len(X.columns) == 0:
            raise ValueError("X must contain at least one row and one feature.")
        y_values = np.asarray(y).reshape(-1)
        if len(y_values) != len(X):
            raise ValueError("X and y must have the same number of rows.")
        weights = None if sample_weight is None else np.asarray(sample_weight).reshape(-1)
        if weights is not None and len(weights) != len(X):
            raise ValueError("sample_weight must have the same number of rows as X.")
        if event_metadata is not None:
            required = {"timestamp", "t1", "target_return"}
            if not required.issubset(event_metadata.columns):
                raise ValueError(f"event_metadata must contain {sorted(required)}.")
            if len(event_metadata) != len(X):
                raise ValueError("event_metadata must have the same number of rows as X.")

        scores: dict[str, list[float]] = {feature: [] for feature in X.columns}
        rng = np.random.default_rng(random_state)
        valid_folds = 0
        for train_idx, test_idx in cv_splits:
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)
            if train_idx.size == 0 or test_idx.size == 0:
                continue
            y_train = y_values[train_idx]
            y_test = y_values[test_idx]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                continue

            fold_model = clone(model)
            fit_kwargs = {}
            if event_metadata is not None:
                train_events = event_metadata.iloc[train_idx].copy()
                train_events["timestamp"] = pd.to_datetime(train_events["timestamp"], utc=True)
                train_events["t1"] = pd.to_datetime(train_events["t1"], utc=True)
                train_events = train_events.set_index("timestamp").sort_index()
                concurrency = SampleWeights.get_num_co_events(
                    pd.DatetimeIndex(train_events.index), train_events[["t1"]]
                )
                uniqueness = SampleWeights.get_sample_uniqueness(
                    train_events[["t1"]], concurrency
                )
                fold_weights = SampleWeights.get_sample_weights(
                    train_events[["t1"]],
                    uniqueness,
                    train_events["target_return"],
                )
                mean_weight = float(fold_weights.mean())
                if np.isfinite(mean_weight) and mean_weight > 0:
                    fold_weights = fold_weights / mean_weight
                else:
                    fold_weights = pd.Series(1.0, index=fold_weights.index)
                fit_kwargs["sample_weight"] = fold_weights.to_numpy()
            elif weights is not None:
                fit_kwargs["sample_weight"] = weights[train_idx]
            fold_model.fit(X.iloc[train_idx], y_train, **fit_kwargs)

            baseline = PurgedFeatureSelector._positive_probability(fold_model, X.iloc[test_idx])
            baseline_auc = roc_auc_score(y_test, baseline)
            valid_folds += 1

            for feature in X.columns:
                permuted = X.iloc[test_idx].copy()
                permuted[feature] = rng.permutation(permuted[feature].to_numpy())
                permuted_probability = PurgedFeatureSelector._positive_probability(
                    fold_model, permuted
                )
                scores[feature].append(
                    float(baseline_auc - roc_auc_score(y_test, permuted_probability))
                )

        if valid_folds == 0:
            raise ValueError("No usable CV folds with both target classes in train and OOS.")

        return pd.DataFrame(
            {
                "feature": list(scores),
                "mean_mda": [
                    float(np.mean(values)) if values else float("nan")
                    for values in scores.values()
                ],
                "std_mda": [
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    for values in scores.values()
                ],
                "folds": [len(values) for values in scores.values()],
            }
        ).sort_values("mean_mda", ascending=False, ignore_index=True)

    @staticmethod
    def _positive_probability(model, X: pd.DataFrame) -> np.ndarray:
        probabilities = model.predict_proba(X)
        classes = np.asarray(model.classes_)
        positive_indices = np.flatnonzero(classes == 1)
        if len(positive_indices) != 1:
            raise ValueError("MDA requires a binary classifier with positive class label 1.")
        return np.asarray(probabilities)[:, positive_indices[0]]
