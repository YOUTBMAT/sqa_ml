"""Unit tests for quantitative features, fractional differentiation stationarity, and causality (no-lookahead bias)."""

import pytest
import numpy as np
import polars as pl
from datetime import datetime, timezone, timedelta
from statsmodels.tsa.stattools import adfuller

from src.features.stationarity import StationarityTransformer
from src.features.pipeline import FeaturePipeline


def test_fractional_differentiation_stationarity() -> None:
    """Test 1: Fractional differentiation converts non-stationary random walk into stationary series (ADF p-value < 0.05)."""
    np.random.seed(42)
    n = 1500
    random_walk = np.cumsum(np.random.randn(n)) + 100.0

    frac_diff_series = StationarityTransformer.fractional_differentiation_ffd(random_walk, d=0.5, threshold=1e-4)
    valid_series = frac_diff_series[~np.isnan(frac_diff_series)]

    assert len(valid_series) > 50, "Too few valid observations after fractional differentiation."

    adf_result = adfuller(valid_series, autolag="AIC")
    p_value = adf_result[1]

    assert p_value < 0.05, f"Fractional differentiated series is not stationary (p-value: {p_value})"


def test_no_lookahead_bias() -> None:
    """Test 2: Rigorous temporal causality (no lookahead bias)."""
    n = 100
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=i) for i in range(n)]

    np.random.seed(123)
    close_prices = 100.0 + np.cumsum(np.random.randn(n))
    high_prices = close_prices + np.abs(np.random.randn(n))
    low_prices = close_prices - np.abs(np.random.randn(n))
    open_prices = low_prices + (high_prices - low_prices) * np.random.rand(n)
    volumes = 1000.0 + np.abs(np.random.randn(n)) * 100

    df_original = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTC/USDT"] * n,
            "asset_class": ["CRYPTO"] * n,
            "timeframe": ["1h"] * n,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
            "trades_count": [10] * n,
        }
    )

    pipeline = FeaturePipeline(d_fractional=0.4, rolling_window=10)
    features_original = pipeline.run(df_original)

    close_prices_modified = close_prices.copy()
    close_prices_modified[-1] = close_prices_modified[-1] * 2.0

    df_modified = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTC/USDT"] * n,
            "asset_class": ["CRYPTO"] * n,
            "timeframe": ["1h"] * n,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices_modified,
            "volume": volumes,
            "trades_count": [10] * n,
        }
    )

    features_modified = pipeline.run(df_modified)

    row_orig = features_original.row(30, named=True)
    row_mod = features_modified.row(30, named=True)

    for col in features_original.columns:
        if col != "close":
            val_orig = row_orig[col]
            val_mod = row_mod[col]
            if isinstance(val_orig, float) and np.isnan(val_orig):
                assert np.isnan(val_mod)
            else:
                assert val_orig == val_mod, f"Lookahead bias detected in column '{col} at index 30!"


def test_pipeline_execution() -> None:
    """Test 3: End-to-end FeaturePipeline execution with simulated OHLCV data."""
    n = 250
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=i) for i in range(n)]

    np.random.seed(99)
    close_prices = 50000.0 + np.cumsum(np.random.randn(n) * 100)
    high_prices = close_prices + np.abs(np.random.randn(n) * 50)
    low_prices = close_prices - np.abs(np.random.randn(n) * 50)
    open_prices = low_prices + (high_prices - low_prices) * np.random.rand(n)
    volumes = 500.0 + np.abs(np.random.randn(n)) * 50

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTC/USDT"] * n,
            "asset_class": ["CRYPTO"] * n,
            "timeframe": ["1h"] * n,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
            "trades_count": [25] * n,
        }
    )

    pipeline = FeaturePipeline(d_fractional=0.4, rolling_window=20)
    result_df = pipeline.run(df)

    assert not result_df.is_empty()
    expected_cols = [
        "log_return",
        "rolling_z_close",
        "frac_diff_close",
        "vol_parkinson",
        "vol_garman_klass",
        "vol_parkinson_pct",
        "vwap_dist",
        "vwap_upper_1_dist",
        "vwap_lower_1_dist",
        "vwap_upper_2_dist",
        "vwap_lower_2_dist",
        "adx_14",
        "ema_200_dist",
        "hurst_100",
        "rsi_adaptive",
    ]
    for col in expected_cols:
        assert col in result_df.columns, f"Missing expected feature column: {col}"

    for col in expected_cols:
        null_count = result_df[col].null_count()
        assert null_count == 0, f"Found {null_count} null values in feature column '{col}'"
