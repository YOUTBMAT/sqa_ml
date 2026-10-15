"""Unit tests for QuantModelTrainer and ModelStorage."""

import pytest
import numpy as np
import polars as pl
from pathlib import Path

from src.models.trainer import QuantModelTrainer
from src.models.storage import ModelStorage


def test_quant_model_trainer_workflow() -> None:
    """Test QuantModelTrainer training, probability prediction, and feature importance."""
    np.random.seed(42)
    n_samples = 200

    # Create synthetic dataset with features, binary target_label (0, 1), and sample_weight
    data = {
        "feat_1": np.random.randn(n_samples),
        "feat_2": np.random.randn(n_samples),
        "target_label": np.random.choice([0, 1], size=n_samples),
        "sample_weight": np.random.rand(n_samples) + 0.5,
    }
    df = pl.DataFrame(data)

    trainer = QuantModelTrainer(n_estimators=10, max_depth=3)
    feature_cols = ["feat_1", "feat_2"]

    metrics = trainer.train(df, feature_cols=feature_cols)

    assert "accuracy" in metrics
    assert "log_loss" in metrics
    assert metrics["train_samples"] == n_samples

    # Test feature importance
    fi_df = trainer.get_feature_importance()
    assert not fi_df.empty
    assert "feature" in fi_df.columns
    assert "importance_gain" in fi_df.columns

    # Test probability prediction
    probs = trainer.predict_proba(df, feature_cols=feature_cols)
    assert probs.shape == (n_samples, 2)  # Binary classification: classes 0 and 1


def test_model_storage_save_load(tmp_path: Path) -> None:
    """Test ModelStorage saving and loading artifacts."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "feat_1": np.random.randn(n_samples),
        "target_label": np.random.choice([0, 1], size=n_samples),
        "sample_weight": np.ones(n_samples),
    }
    df = pl.DataFrame(data)

    trainer = QuantModelTrainer(n_estimators=5, max_depth=2)
    feature_cols = ["feat_1"]
    trainer.train(df, feature_cols=feature_cols)

    storage = ModelStorage(models_dir=str(tmp_path))
    symbol = "TEST/USDT"
    timeframe = "1h"
    metadata = {"symbol": symbol, "timeframe": timeframe, "metrics": {"accuracy": 0.85}}

    storage.save_model(trainer, symbol, timeframe, metadata)

    assert storage.exists(symbol, timeframe) is True

    loaded_model, loaded_meta = storage.load_model(symbol, timeframe)
    assert isinstance(loaded_model, QuantModelTrainer)
    assert loaded_meta["symbol"] == symbol
    assert loaded_meta["metrics"]["accuracy"] == 0.85


def test_walk_forward_leaves_unscored_rows_nan_and_keeps_params() -> None:
    """Rows the walk-forward never scored must stay NaN (not 0.5) so they can never trade."""
    from datetime import datetime, timedelta, timezone

    rng = np.random.default_rng(3)
    n = 300
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts = [t0 + timedelta(hours=i) for i in range(n)]
    df = pl.DataFrame(
        {
            "timestamp": ts,
            "t1": [t + timedelta(hours=2) for t in ts],
            "feat_1": rng.normal(size=n),
            "target_label": rng.integers(0, 2, size=n),
            "sample_weight": np.ones(n),
        }
    )
    trainer = QuantModelTrainer(n_estimators=7, max_depth=2, num_leaves=3)
    probs = trainer.cross_val_predict_walk_forward(df, feature_cols=["feat_1"], n_splits=5)

    scored = ~np.isnan(probs[:, 1])
    assert not scored[:50].any() and scored[-50:].all()
    assert 0.0 <= QuantModelTrainer.oos_roc_auc(df, probs) <= 1.0
    assert trainer.model.get_params()["num_leaves"] == 3
