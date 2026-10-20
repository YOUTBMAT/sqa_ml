"""Population Stability Index (PSI) based feature drift detection."""

from collections.abc import Sequence

import numpy as np
import polars as pl


class DataDriftDetector:
    """Calculate PSI for reference and current feature distributions."""

    EPSILON = 1e-4
    SEVERE_DRIFT_THRESHOLD = 0.25

    @staticmethod
    def calculate_psi(
        reference: np.ndarray, current: np.ndarray, num_bins: int = 10
    ) -> float:
        """Calculate PSI using quantile bins defined by the reference population.

        Non-finite observations are excluded. A small epsilon keeps empty bins
        finite and prevents division by zero in the logarithm.
        """
        if num_bins < 1:
            raise ValueError("num_bins must be at least 1.")

        expected_values = np.asarray(reference, dtype=float).reshape(-1)
        actual_values = np.asarray(current, dtype=float).reshape(-1)
        expected_values = expected_values[np.isfinite(expected_values)]
        actual_values = actual_values[np.isfinite(actual_values)]
        if expected_values.size == 0 or actual_values.size == 0:
            raise ValueError("Reference and current samples must contain finite values.")

        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        edges = np.unique(np.quantile(expected_values, quantiles))
        if edges.size == 1:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf

        expected_counts, _ = np.histogram(expected_values, bins=edges)
        actual_counts, _ = np.histogram(actual_values, bins=edges)
        expected_pct = expected_counts / expected_counts.sum()
        actual_pct = actual_counts / actual_counts.sum()

        expected_pct = np.clip(expected_pct, DataDriftDetector.EPSILON, None)
        actual_pct = np.clip(actual_pct, DataDriftDetector.EPSILON, None)
        return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

    @classmethod
    def check_feature_drift(
        cls,
        train_df: pl.DataFrame,
        live_df: pl.DataFrame,
        features: Sequence[str],
    ) -> tuple[dict[str, float], bool]:
        """Return per-feature PSI values and whether any feature is severely shifted."""
        if not features:
            raise ValueError("At least one feature is required for drift detection.")
        missing = [
            f"{source}.{feature}"
            for source, frame in (("train", train_df), ("live", live_df))
            for feature in features
            if feature not in frame.columns
        ]
        if missing:
            raise ValueError(f"Missing drift features: {', '.join(missing)}")

        psi_values = {
            feature: cls.calculate_psi(
                train_df[feature].to_numpy(),
                live_df[feature].to_numpy(),
            )
            for feature in features
        }
        has_severe_drift = any(
            value > cls.SEVERE_DRIFT_THRESHOLD for value in psi_values.values()
        )
        return psi_values, has_severe_drift
