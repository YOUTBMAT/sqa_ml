"""Tests for purged MDA feature ranking."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from src.features.selection import PurgedFeatureSelector
from src.models.trainer import QuantModelTrainer
from scripts.select_features import selection_period


class SignalOnlyClassifier(ClassifierMixin, BaseEstimator):
    """Small estimator used to verify an independent Gaussian noise column."""

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight=None):
        self.classes_ = np.unique(y)
        self.model_ = LogisticRegression(random_state=0).fit(
            X[["signal"]], y, sample_weight=sample_weight
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[["signal"]])


def test_mda_scores_informative_feature_and_ignores_gaussian_noise() -> None:
    rng = np.random.default_rng(31)
    size = 1200
    signal = rng.normal(size=size)
    noise = rng.normal(size=size)
    target = (signal + rng.normal(scale=0.45, size=size) > 0).astype(int)
    X = pd.DataFrame({"signal": signal, "noise": noise})

    folds = []
    boundaries = np.array_split(np.arange(size), 6)
    for fold_index in range(1, len(boundaries)):
        train = np.concatenate(boundaries[:fold_index])
        test = boundaries[fold_index]
        folds.append((train, test))

    scores = PurgedFeatureSelector.calculate_mda(
        SignalOnlyClassifier(),
        X,
        target,
        folds,
        random_state=13,
    ).set_index("feature")

    assert scores.loc["signal", "mean_mda"] > 0.1
    assert scores.loc["noise", "mean_mda"] == 0.0
    assert scores.loc["noise", "std_mda"] == 0.0


def test_mda_raises_when_no_cv_fold_has_both_classes() -> None:
    X = pd.DataFrame({"signal": np.arange(20, dtype=float)})
    y = np.zeros(20, dtype=int)
    folds = [(np.arange(0, 10), np.arange(10, 20))]

    with pytest.raises(ValueError, match="No usable CV folds"):
        PurgedFeatureSelector.calculate_mda(
            LogisticRegression(), X, y, folds
        )


def test_trainer_uses_validated_selected_feature_list(tmp_path) -> None:
    selected_path = tmp_path / "selected_features.json"
    selected_path.write_text('["signal", "noise"]', encoding="utf-8")

    assert QuantModelTrainer.configured_feature_cols(
        ["signal", "noise", "unused"], selected_features_path=selected_path
    ) == ["signal", "noise"]


def test_trainer_rejects_selected_features_missing_from_dataset(tmp_path) -> None:
    selected_path = tmp_path / "selected_features.json"
    selected_path.write_text('["missing"]', encoding="utf-8")

    with pytest.raises(ValueError, match="missing from data"):
        QuantModelTrainer.configured_feature_cols(
            ["signal"], selected_features_path=selected_path
        )


def test_selection_period_excludes_final_holdout_timestamps() -> None:
    timestamps = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    dataset = pd.DataFrame(
        {
            "timestamp": timestamps,
            "t1": timestamps + pd.Timedelta(hours=6),
            "feature": np.arange(100),
        }
    )

    selected = selection_period(dataset, 0.3)

    assert selected["timestamp"].max() == timestamps[63]
    assert (selected["t1"] < timestamps[70]).all()
    assert timestamps[70] not in set(selected["timestamp"])
