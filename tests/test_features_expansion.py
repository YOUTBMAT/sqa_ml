"""Regression tests for expanded stationary alpha features."""

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller

from src.features.pipeline import FeaturePipeline

NEW_FEATURES = [
    "roc_3",
    "roc_6",
    "roc_12",
    "roc_24",
    "tsi_25_13",
    "atr_ratio_14",
    "keltner_upper_dist",
    "keltner_lower_dist",
    "bb_width_z_30",
    "normalized_volume_trend_24",
    "obv_momentum_24",
    "volume_weighted_price_deviation_24",
    "return_skew_30",
    "return_kurtosis_30",
]


def _sample_ohlcv(size: int = 1200) -> pl.DataFrame:
    rng = np.random.default_rng(91)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size)))
    high = close * (1.0 + rng.uniform(0.001, 0.02, size))
    low = close * (1.0 - rng.uniform(0.001, 0.02, size))
    open_ = close * np.exp(rng.normal(0.0, 0.003, size))
    volume = rng.lognormal(mean=7.0, sigma=0.35, size=size)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(size)],
            "symbol": ["ETH/USDT"] * size,
            "asset_class": ["CRYPTO"] * size,
            "timeframe": ["1h"] * size,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "trades_count": [100] * size,
        }
    )


def test_expanded_features_are_present_finite_and_warmup_purged() -> None:
    result = FeaturePipeline().run(_sample_ohlcv())

    assert len(NEW_FEATURES) == 14
    assert len([column for column in result.columns if column not in {
        "timestamp", "symbol", "asset_class", "timeframe", "open", "high", "low",
        "close", "volume", "trades_count",
    }]) >= 30
    for feature in NEW_FEATURES:
        values = result[feature].to_numpy()
        assert np.isfinite(values).all(), f"{feature} contains NaN or infinity"
        assert result[feature].null_count() == 0, f"{feature} contains null warmup rows"


def test_expanded_roc_feature_is_stationary() -> None:
    result = FeaturePipeline().run(_sample_ohlcv())
    statistic = adfuller(
        result["roc_12"].to_numpy(), autolag="AIC", result_object=True
    )

    assert statistic.pvalue < 0.05
