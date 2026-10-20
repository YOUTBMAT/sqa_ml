"""Tests for Population Stability Index drift detection."""

import numpy as np
import polars as pl
import pytest

from src.validation.drift import DataDriftDetector


def test_calculate_psi_is_zero_for_identical_distributions() -> None:
    reference = np.random.default_rng(17).normal(size=2000)

    psi = DataDriftDetector.calculate_psi(reference, reference.copy())

    assert psi == pytest.approx(0.0)


def test_check_feature_drift_detects_moderate_distribution_shift() -> None:
    reference = np.random.default_rng(7).normal(0.0, 1.0, 5000)
    current = np.random.default_rng(8).normal(0.4, 1.0, 5000)

    psi_values, has_severe_drift = DataDriftDetector.check_feature_drift(
        pl.DataFrame({"feature": reference}),
        pl.DataFrame({"feature": current}),
        ["feature"],
    )

    assert 0.10 <= psi_values["feature"] <= 0.25
    assert has_severe_drift is False


def test_check_feature_drift_flags_severe_distribution_shift() -> None:
    reference = np.random.default_rng(7).normal(0.0, 1.0, 5000)
    current = np.random.default_rng(8).normal(0.6, 1.0, 5000)

    psi_values, has_severe_drift = DataDriftDetector.check_feature_drift(
        pl.DataFrame({"feature": reference}),
        pl.DataFrame({"feature": current}),
        ["feature"],
    )

    assert psi_values["feature"] > 0.25
    assert has_severe_drift is True


def test_calculate_psi_handles_nan_and_zero_values() -> None:
    reference = np.array([0.0, 0.0, 1.0, 2.0, np.nan, np.inf])
    current = np.array([0.0, 0.0, 1.0, 2.0, np.nan, -np.inf])

    psi = DataDriftDetector.calculate_psi(reference, current)

    assert psi == pytest.approx(0.0)
    assert np.isfinite(psi)


@pytest.mark.parametrize(
    ("reference", "current"),
    [
        (np.array([np.nan, np.inf]), np.array([1.0])),
        (np.array([1.0]), np.array([np.nan, -np.inf])),
    ],
)
def test_calculate_psi_rejects_samples_without_finite_values(
    reference: np.ndarray, current: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="finite values"):
        DataDriftDetector.calculate_psi(reference, current)
