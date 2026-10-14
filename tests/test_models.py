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
