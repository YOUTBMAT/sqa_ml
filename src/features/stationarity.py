"""Stationarity transformer implementing fractional differentiation, log returns, and rolling Z-score."""

import numpy as np
import polars as pl
from src.utils.logger import logger


class StationarityTransformer:
    """Transformer for achieving stationarity via fractional differentiation and normalization."""

    @staticmethod
    def get_fractional_weights(d: float, size: int) -> np.ndarray:
        """Compute weights for Fixed Width Window (FFD) fractional differentiation.

        Formula: w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k
        """
        w = [1.0]
        for k in range(1, size):
            w.append(-w[-1] * (d - k + 1) / k)
        return np.array(w[::-1])

    @classmethod
    def fractional_differentiation_ffd(
        cls, series: np.ndarray, d: float, threshold: float = 1e-3
    ) -> np.ndarray:
        """Apply Fixed Width Window (FFD) Fractional Differentiation to a 1D NumPy array.

        Marcos López de Prado - Advances in Financial Machine Learning, Chapter 5.
        """
        # 1. Compute weights
        weights = [1.0]
        k = 1
        while True:
            w_k = -weights[-1] * (d - k + 1) / k
            if abs(w_k) < threshold:
                break
            weights.append(w_k)
            k += 1

        weights = np.array(weights[::-1])
        drop = len(weights)

        # Safety cap: if drop > 20% of series length, limit window to 15% of series length
        max_window = int(len(series) * 0.15)
        if drop > max_window and max_window > 0:
            logger.warning(f"FFD window size ({drop}) exceeds series limit. Capping window to {max_window} to preserve data.")
            weights = weights[-max_window:]
            drop = len(weights)

        if len(series) <= drop:
            logger.warning("Series length is shorter than fractional differentiation window size.")
            return np.full_like(series, np.nan)

        # 2. Apply weights via convolution / dot product
        res = np.full_like(series, np.nan)
        for i in range(drop - 1, len(series)):
            window = series[i - drop + 1 : i + 1]
            res[i] = np.dot(weights, window)

        return res

    @classmethod
    def transform_fractional_diff(
        cls, df: pl.DataFrame, value_col: str, d: float, threshold: float = 1e-3, output_col: str = "frac_diff"
    ) -> pl.DataFrame:
        """Apply fractional differentiation to a Polars DataFrame column."""
        logger.info(f"Applying fractional differentiation (d={d}) on column '{value_col}'...")
        arr = df[value_col].to_numpy().astype(float)
        diff_arr = cls.fractional_differentiation_ffd(arr, d, threshold)
        return df.with_columns(pl.Series(output_col, diff_arr))

    @staticmethod
    def log_returns(df: pl.DataFrame, price_col: str = "close", output_col: str = "log_return") -> pl.DataFrame:
        """Calculate logarithmic returns: log(P_t / P_{t-1})."""
        logger.info(f"Calculating logarithmic returns on '{price_col}'...")
        return df.with_columns(
            (pl.col(price_col) / pl.col(price_col).shift(1)).log().alias(output_col)
        )

    @staticmethod
    def rolling_z_score(
        df: pl.DataFrame, value_col: str, window: int = 20, output_col: str = "rolling_z"
    ) -> pl.DataFrame:
        """Calculate rolling Z-score: (X_t - rolling_mean) / rolling_std."""
        logger.info(f"Calculating rolling Z-score on '{value_col}' with window={window}...")
        mean_col = f"{value_col}_rolling_mean"
        std_col = f"{value_col}_rolling_std"

        return df.with_columns(
            [
                pl.col(value_col).rolling_mean(window_size=window).alias(mean_col),
                pl.col(value_col).rolling_std(window_size=window).alias(std_col),
            ]
        ).with_columns(
            ((pl.col(value_col) - pl.col(mean_col)) / pl.col(std_col)).alias(output_col)
        ).drop([mean_col, std_col])
