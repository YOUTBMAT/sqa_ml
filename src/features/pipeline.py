"""Feature pipeline orchestrator integrating stationarity and technical transformations."""

import polars as pl
from src.features.stationarity import StationarityTransformer
from src.features.technical import QuantitativeFeatures
from src.utils.logger import logger


class FeaturePipeline:
    """Orchestrator for computing statistical and technical quantitative features."""

    def __init__(self, d_fractional: float = 0.4, rolling_window: int = 20) -> None:
        self.d_fractional = d_fractional
        self.rolling_window = rolling_window

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        """Execute feature pipeline transformations on OHLCV DataFrame.

        Steps:
        1. Cast numeric columns to Float64 & fill trades_count nulls.
        2. Log returns & Rolling Z-Score.
        3. Fractional Differentiation (FFD).
        4. Volatility estimators (Parkinson, Garman-Klass) & Volatility Percentile.
        5. Dynamic VWAP with Bands.
        6. ADX 14, EMA 200, Hurst Exponent 100.
        7. Adaptive RSI.
        8. Drop warmup period nulls safely using feature subset.
        """
        if df.is_empty():
            logger.warning("Empty DataFrame passed to FeaturePipeline.")
            return df

        logger.info(f"Running FeaturePipeline on {len(df)} rows...")

        # 0. Cast numeric columns explicitly to Float64 and fill trades_count nulls
        numeric_cols = ["open", "high", "low", "close", "volume"]
        cast_exprs = [
            pl.col(c).cast(pl.Float64) for c in numeric_cols if c in df.columns
        ]
        if cast_exprs:
            df = df.with_columns(cast_exprs)

        if "trades_count" in df.columns:
            df = df.with_columns(pl.col("trades_count").fill_null(0))

        # 1. Stationarity: Log returns & Rolling Z-Score
        df = StationarityTransformer.log_returns(df, price_col="close", output_col="log_return")
        df = StationarityTransformer.rolling_z_score(df, value_col="close", window=self.rolling_window, output_col="rolling_z_close")

        # 2. Stationarity: Fractional Differentiation FFD
        df = StationarityTransformer.transform_fractional_diff(
            df, value_col="close", d=self.d_fractional, output_col="frac_diff_close"
        )

        # 3. Technical: Volatilities & Volatility Percentile
        df = QuantitativeFeatures.parkinson_volatility(df, window=self.rolling_window, output_col="vol_parkinson")
        df = QuantitativeFeatures.garman_klass_volatility(df, window=self.rolling_window, output_col="vol_garman_klass")
        df = QuantitativeFeatures.volatility_percentile(df, vol_col="vol_parkinson", window=100, output_col="vol_parkinson_pct")

        # 4. Technical: Dynamic VWAP & Bands
        df = QuantitativeFeatures.dynamic_vwap_bands(df, window=self.rolling_window, num_std=2.0)

        # 5. Technical: ADX 14, EMA 200 Distance, Hurst 100
        df = QuantitativeFeatures.adx(df, period=14, output_col="adx_14")
        df = QuantitativeFeatures.ema_distance(df, span=200, price_col="close", output_col="ema_200_dist")
        df = QuantitativeFeatures.hurst_exponent(df, window=100, price_col="close", output_col="hurst_100")

        # 6. Technical: Adaptive RSI
        df = QuantitativeFeatures.adaptive_rsi(df, base_period=14, vol_window=self.rolling_window, output_col="rsi_adaptive")

        # 7. Technical: Stationary Alpha Features
        df = QuantitativeFeatures.stationary_alpha_features(df)
        df = QuantitativeFeatures.momentum_features(df)
        df = QuantitativeFeatures.volatility_and_band_features(df)
        df = QuantitativeFeatures.volume_flow_features(df, window=24)
        df = QuantitativeFeatures.return_moment_features(df, return_col="log_return", window=30)

        new_feature_cols = [
            *(f"roc_{period}" for period in (3, 6, 12, 24)),
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
        df = df.with_columns(
            [
                pl.when(pl.col(col).is_finite())
                .then(pl.col(col))
                .otherwise(None)
                .alias(col)
                for col in new_feature_cols
            ]
        )

        # Log null counts per column before drop_nulls
        null_counts = {col: df[col].null_count() for col in df.columns}
        logger.info(f"Null counts per column before drop_nulls: {null_counts}")

        # 8. Drop warmup rows using core feature subset
        feature_subset = [
            "close",
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
            "vol_ratio_6_24",
            "volume_momentum_24",
            "volume_price_trend",
            *new_feature_cols,
        ]
        valid_subset = [col for col in feature_subset if col in df.columns]

        initial_len = len(df)
        df = df.drop_nulls(subset=valid_subset).drop_nans(subset=valid_subset)
        dropped_len = initial_len - len(df)

        logger.success(
            f"FeaturePipeline successfully completed. "
            f"Retained {len(df)} rows (dropped {dropped_len} warmup rows)."
        )
        return df
