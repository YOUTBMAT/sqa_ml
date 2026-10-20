"""Advanced technical and quantitative feature engineering module using Polars."""

import numpy as np
import polars as pl
from src.utils.logger import logger


class QuantitativeFeatures:
    """Class implementing advanced quantitative features and volatility estimators."""

    @staticmethod
    def parkinson_volatility(df: pl.DataFrame, window: int = 14, output_col: str = "vol_parkinson") -> pl.DataFrame:
        """Calculate Parkinson Volatility using High and Low prices.

        Formula: sqrt( (1 / (4 * ln(2))) * rolling_mean( (ln(High / Low))^2 ) )
        """
        logger.info(f"Calculating Parkinson Volatility with window={window}...")
        factor = 1.0 / (4.0 * np.log(2.0))
        high = pl.col("high").cast(pl.Float64)
        low = pl.col("low").cast(pl.Float64)
        return df.with_columns(
            (
                factor
                * (high / low).log().pow(2)
            )
            .rolling_mean(window_size=window)
            .sqrt()
            .alias(output_col)
        )

    @staticmethod
    def garman_klass_volatility(df: pl.DataFrame, window: int = 14, output_col: str = "vol_garman_klass") -> pl.DataFrame:
        """Calculate Garman-Klass Volatility using Open, High, Low, and Close prices.

        Formula: sqrt( rolling_mean( 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2 ) )
        """
        logger.info(f"Calculating Garman-Klass Volatility with window={window}...")
        high = pl.col("high").cast(pl.Float64)
        low = pl.col("low").cast(pl.Float64)
        close = pl.col("close").cast(pl.Float64)
        open_ = pl.col("open").cast(pl.Float64)

        gk_term = (
            0.5 * (high / low).log().pow(2)
            - (2.0 * np.log(2.0) - 1.0) * (close / open_).log().pow(2)
        )
        return df.with_columns(
            gk_term.rolling_mean(window_size=window).sqrt().alias(output_col)
        )

    @staticmethod
    def dynamic_vwap_bands(
        df: pl.DataFrame, window: int = 20, num_std: float = 2.0
    ) -> pl.DataFrame:
        """Calculate Rolling VWAP and Dynamic Bands percentage distances relative to VWAP."""
        logger.info(f"Calculating Dynamic VWAP distance bands with window={window}, num_std={num_std}...")
        
        high = pl.col("high").cast(pl.Float64)
        low = pl.col("low").cast(pl.Float64)
        close = pl.col("close").cast(pl.Float64)
        volume = pl.col("volume").cast(pl.Float64)

        tp = (high + low + close) / 3.0
        tp_vol = tp * volume

        sum_tp_vol = tp_vol.rolling_sum(window_size=window)
        sum_vol = volume.rolling_sum(window_size=window)

        vwap = sum_tp_vol / (sum_vol + 1e-8)
        tp_std = tp.rolling_std(window_size=window)

        vwap_upper_1 = vwap + (1.0 * tp_std)
        vwap_lower_1 = vwap - (1.0 * tp_std)
        vwap_upper_2 = vwap + (num_std * tp_std)
        vwap_lower_2 = vwap - (num_std * tp_std)

        return df.with_columns(
            [
                ((close - vwap) / (vwap + 1e-8)).alias("vwap_dist"),
                ((close - vwap_upper_1) / (vwap + 1e-8)).alias("vwap_upper_1_dist"),
                ((close - vwap_lower_1) / (vwap + 1e-8)).alias("vwap_lower_1_dist"),
                ((close - vwap_upper_2) / (vwap + 1e-8)).alias("vwap_upper_2_dist"),
                ((close - vwap_lower_2) / (vwap + 1e-8)).alias("vwap_lower_2_dist"),
            ]
        )

    @staticmethod
    def adaptive_rsi(
        df: pl.DataFrame, base_period: int = 14, vol_window: int = 20, output_col: str = "rsi_adaptive"
    ) -> pl.DataFrame:
        """Calculate Volatility-Adaptive RSI."""
        logger.info(f"Calculating Volatility-Adaptive RSI with base_period={base_period}...")
        close = pl.col("close").cast(pl.Float64)

        ret = close - close.shift(1)
        gain = pl.when(ret > 0).then(ret).otherwise(0)
        loss = pl.when(ret < 0).then(-ret).otherwise(0)

        avg_gain = gain.rolling_mean(window_size=base_period)
        avg_loss = loss.rolling_mean(window_size=base_period)

        rs = avg_gain / (avg_loss + 1e-8)
        rsi_base = 100.0 - (100.0 / (1.0 + rs))

        return df.with_columns(rsi_base.alias(output_col))

    @staticmethod
    def adx(df: pl.DataFrame, period: int = 14, output_col: str = "adx_14") -> pl.DataFrame:
        """Calculate Average Directional Index (ADX)."""
        logger.info(f"Calculating ADX with period={period}...")
        high = pl.col("high").cast(pl.Float64)
        low = pl.col("low").cast(pl.Float64)
        close = pl.col("close").cast(pl.Float64)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
        minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pl.max_horizontal([tr1, tr2, tr3])

        atr = tr.ewm_mean(span=period)
        plus_di = 100.0 * (plus_dm.ewm_mean(span=period) / (atr + 1e-8))
        minus_di = 100.0 * (minus_dm.ewm_mean(span=period) / (atr + 1e-8))

        dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-8)
        adx_val = dx.ewm_mean(span=period)

        return df.with_columns(adx_val.alias(output_col))

    @staticmethod
    def ema_distance(df: pl.DataFrame, span: int = 200, price_col: str = "close", output_col: str = "ema_200_dist") -> pl.DataFrame:
        """Calculate percentage distance from Exponential Moving Average (EMA)."""
        logger.info(f"Calculating EMA distance with span={span}...")
        close = pl.col(price_col).cast(pl.Float64)
        ema = close.ewm_mean(span=span)
        return df.with_columns(((close - ema) / (close + 1e-8)).alias(output_col))

    @staticmethod
    def hurst_exponent(df: pl.DataFrame, window: int = 100, price_col: str = "close", output_col: str = "hurst_100") -> pl.DataFrame:
        """Calculate rolling Hurst Exponent approximation."""
        logger.info(f"Calculating Hurst Exponent with window={window}...")
        close = pl.col(price_col).cast(pl.Float64)
        log_ret = (close / close.shift(1)).log()
        var1 = log_ret.rolling_var(window_size=window // 2)
        var2 = log_ret.rolling_var(window_size=window)
        hurst = 0.5 + (var2 / (var1 + 1e-8)).log() / (2.0 * np.log(2.0))
        return df.with_columns(hurst.clip(0.0, 1.0).alias(output_col))

    @staticmethod
    def volatility_percentile(df: pl.DataFrame, vol_col: str = "vol_parkinson", window: int = 100, output_col: str = "vol_parkinson_pct") -> pl.DataFrame:
        """Calculate rolling percentile rank of Parkinson volatility."""
        logger.info(f"Calculating Volatility Percentile for {vol_col} with window={window}...")
        vol = pl.col(vol_col).cast(pl.Float64)
        rmin = vol.rolling_min(window_size=window)
        rmax = vol.rolling_max(window_size=window)
        pct = (vol - rmin) / (rmax - rmin + 1e-8)
        return df.with_columns(pct.clip(0.0, 1.0).alias(output_col))

    @staticmethod
    def stationary_alpha_features(df: pl.DataFrame) -> pl.DataFrame:
        """Calculate stationary alpha features: vol_ratio_6_24, volume_momentum_24, volume_price_trend."""
        logger.info("Calculating stationary alpha features (vol_ratio_6_24, volume_momentum_24, volume_price_trend)...")
        close = pl.col("close").cast(pl.Float64)
        volume = pl.col("volume").cast(pl.Float64)

        log_ret = (close / close.shift(1)).log()
        vol_6h = log_ret.rolling_std(window_size=6)
        vol_24h = log_ret.rolling_std(window_size=24)
        vol_ratio = (vol_6h / (vol_24h + 1e-8)).clip(0.0, 10.0)

        vol_sma24 = volume.rolling_mean(window_size=24)
        vol_mom = (volume - vol_sma24) / (vol_sma24 + 1e-8)

        vpt = log_ret * vol_mom

        return df.with_columns(
            [
                vol_ratio.alias("vol_ratio_6_24"),
                vol_mom.alias("volume_momentum_24"),
                vpt.alias("volume_price_trend"),
            ]
        )

    @staticmethod
    def momentum_features(df: pl.DataFrame) -> pl.DataFrame:
        """Add multi-horizon rate of change and bounded True Strength Index."""
        close = pl.col("close").cast(pl.Float64)
        momentum = close - close.shift(1)
        smooth_momentum = momentum.ewm_mean(span=25).ewm_mean(span=13)
        smooth_abs_momentum = momentum.abs().ewm_mean(span=25).ewm_mean(span=13)
        expressions = [
            ((close / close.shift(period) - 1.0).clip(-10.0, 10.0)).alias(f"roc_{period}")
            for period in (3, 6, 12, 24)
        ]
        expressions.append(
            (100.0 * smooth_momentum / (smooth_abs_momentum + 1e-8))
            .clip(-100.0, 100.0)
            .alias("tsi_25_13")
        )
        return df.with_columns(expressions)

    @staticmethod
    def volatility_and_band_features(df: pl.DataFrame) -> pl.DataFrame:
        """Add ATR ratio, Keltner channel distances and Bollinger width z-score."""
        high = pl.col("high").cast(pl.Float64)
        low = pl.col("low").cast(pl.Float64)
        close = pl.col("close").cast(pl.Float64)
        previous_close = close.shift(1)
        true_range = pl.max_horizontal(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ]
        )
        atr_14 = true_range.rolling_mean(window_size=14)
        ema_20 = close.ewm_mean(span=20)
        keltner_width = 2.0 * atr_14
        bb_std = close.rolling_std(window_size=20)
        bb_width = (4.0 * bb_std / (close.rolling_mean(window_size=20) + 1e-8)).clip(0.0, 10.0)
        bb_width_mean = bb_width.rolling_mean(window_size=30)
        bb_width_std = bb_width.rolling_std(window_size=30)
        return df.with_columns(
            [
                (atr_14 / (close + 1e-8)).clip(0.0, 10.0).alias("atr_ratio_14"),
                ((close - (ema_20 + keltner_width)) / (close + 1e-8))
                .clip(-10.0, 10.0)
                .alias("keltner_upper_dist"),
                ((close - (ema_20 - keltner_width)) / (close + 1e-8))
                .clip(-10.0, 10.0)
                .alias("keltner_lower_dist"),
                ((bb_width - bb_width_mean) / (bb_width_std + 1e-8))
                .clip(-10.0, 10.0)
                .alias("bb_width_z_30"),
            ]
        )

    @staticmethod
    def volume_flow_features(df: pl.DataFrame, window: int = 24) -> pl.DataFrame:
        """Add normalized volume trend, OBV momentum and volume-weighted deviation."""
        close = pl.col("close").cast(pl.Float64)
        volume = pl.col("volume").cast(pl.Float64)
        volume_mean = volume.rolling_mean(window_size=window)
        normalized_volume = ((volume / (volume_mean + 1e-8)) - 1.0).clip(-10.0, 10.0)
        direction = pl.when(close > close.shift(1)).then(1.0).when(
            close < close.shift(1)
        ).then(-1.0).otherwise(0.0)
        obv_increment = direction * volume
        obv = obv_increment.cum_sum()
        obv_change = obv - obv.shift(window)
        volume_scale = volume.rolling_sum(window_size=window)
        obv_momentum = (obv_change / (volume_scale + 1e-8)).clip(-10.0, 10.0)
        typical_price = (
            pl.col("high").cast(pl.Float64)
            + pl.col("low").cast(pl.Float64)
            + close
        ) / 3.0
        vwap = (typical_price * volume).rolling_sum(window_size=window) / (
            volume.rolling_sum(window_size=window) + 1e-8
        )
        return df.with_columns(
            [
                normalized_volume.alias("normalized_volume_trend_24"),
                obv_momentum.alias("obv_momentum_24"),
                ((close - vwap) / (close + 1e-8))
                .clip(-10.0, 10.0)
                .alias("volume_weighted_price_deviation_24"),
            ]
        )

    @staticmethod
    def return_moment_features(
        df: pl.DataFrame, return_col: str = "log_return", window: int = 30
    ) -> pl.DataFrame:
        """Add rolling skewness and excess kurtosis of log returns."""
        returns = pl.col(return_col).cast(pl.Float64)
        rolling_count = returns.is_not_null().cast(pl.Int64).rolling_sum(window_size=window)
        rolling_std = returns.rolling_std(window_size=window)
        skew = returns.rolling_skew(window_size=window, bias=False)
        kurtosis = returns.rolling_kurtosis(window_size=window, fisher=True, bias=False)
        return df.with_columns(
            [
                pl.when(rolling_count < window)
                .then(None)
                .when(rolling_std <= 1e-12)
                .then(0.0)
                .otherwise(skew)
                .clip(-20.0, 20.0)
                .fill_nan(0.0)
                .alias(f"return_skew_{window}"),
                pl.when(rolling_count < window)
                .then(None)
                .when(rolling_std <= 1e-12)
                .then(0.0)
                .otherwise(kurtosis)
                .clip(-20.0, 100.0)
                .fill_nan(0.0)
                .alias(f"return_kurtosis_{window}"),
            ]
        )
