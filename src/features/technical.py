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
        ).fill_null(0.0)

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
        ).fill_null(0.0)

    @staticmethod
    def dynamic_vwap_bands(
        df: pl.DataFrame, window: int = 20, num_std: float = 2.0
    ) -> pl.DataFrame:
        """Calculate Rolling VWAP and Dynamic Bands (±1σ, ±2σ).

        Typical Price = (High + Low + Close) / 3
        VWAP = sum(Typical Price * Volume) / sum(Volume) over window
        Bands = VWAP ± num_std * rolling_std(Typical Price)
        """
        logger.info(f"Calculating Dynamic VWAP and Bands with window={window}, num_std={num_std}...")
        
        high = pl.col("high").cast(pl.Float64)
        low = pl.col("low").cast(pl.Float64)
        close = pl.col("close").cast(pl.Float64)
        volume = pl.col("volume").cast(pl.Float64)

        # Typical Price
        tp = (high + low + close) / 3.0
        tp_vol = tp * volume

        # Rolling sums
        sum_tp_vol = tp_vol.rolling_sum(window_size=window)
        sum_vol = volume.rolling_sum(window_size=window)

        vwap = sum_tp_vol / (sum_vol + 1e-8)
        tp_std = tp.rolling_std(window_size=window)

        return df.with_columns(
            [
                vwap.fill_null(close).alias("vwap"),
                (vwap + (1.0 * tp_std)).fill_null(close).alias("vwap_upper_1"),
                (vwap - (1.0 * tp_std)).fill_null(close).alias("vwap_lower_1"),
                (vwap + (num_std * tp_std)).fill_null(close).alias("vwap_upper_2"),
                (vwap - (num_std * tp_std)).fill_null(close).alias("vwap_lower_2"),
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

        return df.with_columns(rsi_base.alias(output_col)).fill_null(50.0)

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

        return df.with_columns(adx_val.alias(output_col)).fill_null(20.0)

    @staticmethod
    def ema(df: pl.DataFrame, span: int = 200, price_col: str = "close", output_col: str = "ema_200") -> pl.DataFrame:
        """Calculate Exponential Moving Average (EMA)."""
        logger.info(f"Calculating EMA with span={span}...")
        close = pl.col(price_col).cast(pl.Float64)
        return df.with_columns(close.ewm_mean(span=span).alias(output_col)).fill_null(close)

    @staticmethod
    def hurst_exponent(df: pl.DataFrame, window: int = 100, price_col: str = "close", output_col: str = "hurst_100") -> pl.DataFrame:
        """Calculate rolling Hurst Exponent approximation."""
        logger.info(f"Calculating Hurst Exponent with window={window}...")
        close = pl.col(price_col).cast(pl.Float64)
        log_ret = (close / close.shift(1)).log()
        var1 = log_ret.rolling_var(window_size=window // 2)
        var2 = log_ret.rolling_var(window_size=window)
        hurst = 0.5 + (var2 / (var1 + 1e-8)).log() / (2.0 * np.log(2.0))
        return df.with_columns(hurst.clip(0.0, 1.0).alias(output_col)).fill_null(0.5)

    @staticmethod
    def volatility_percentile(df: pl.DataFrame, vol_col: str = "vol_parkinson", window: int = 100, output_col: str = "vol_parkinson_pct") -> pl.DataFrame:
        """Calculate rolling percentile rank of Parkinson volatility."""
        logger.info(f"Calculating Volatility Percentile for {vol_col} with window={window}...")
        vol = pl.col(vol_col).cast(pl.Float64)
        rmin = vol.rolling_min(window_size=window)
        rmax = vol.rolling_max(window_size=window)
        pct = (vol - rmin) / (rmax - rmin + 1e-8)
        return df.with_columns(pct.clip(0.0, 1.0).alias(output_col)).fill_null(0.5)
