"""Dataset builder module combining features, triple barrier labels, and sample weights."""

from pathlib import Path
import pandas as pd
import polars as pl
from src.features.storage import FeatureStorage
from src.labeling.triple_barrier import TripleBarrierLabeler
from src.labeling.sample_weights import SampleWeights
from src.utils.logger import logger


class DatasetBuilder:
    """Builder for assembling supervised machine learning training datasets with labels and sample weights."""

    def __init__(self, datasets_dir: str = "data/datasets") -> None:
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.feature_storage = FeatureStorage()

    def _get_dataset_filepath(self, symbol: str, timeframe: str) -> Path:
        """Sanitize symbol and return dataset parquet file path."""
        sanitized_symbol = symbol.replace("/", "_").replace("-", "_").lower()
        filename = f"{sanitized_symbol}_{timeframe}_dataset.parquet"
        return self.datasets_dir / filename

    def build_dataset(
        self, symbol: str, timeframe: str, pt_sl: list[float] = [1.5, 1.0], num_bars: int = 24
    ) -> pl.DataFrame:
        """Build supervised ML dataset for a given symbol and timeframe.

        Steps:
        1. Load feature matrix from Parquet via FeatureStorage.
        2. Apply TripleBarrierLabeler to compute labels, returns, and barrier types.
        3. Compute sample concurrency, uniqueness, and sample weights via SampleWeights.
        4. Combine everything into a unified Polars DataFrame.
        """
        logger.info(f"Building ML dataset for {symbol} ({timeframe})...")

        # 1. Load features
        df_features = self.feature_storage.load_features(symbol, timeframe)
        if df_features.is_empty():
            logger.warning(f"No features found for {symbol} ({timeframe}). Cannot build dataset.")
            return pl.DataFrame()

        # 2. Compute Triple Barrier Labels
        df_labeled = TripleBarrierLabeler.get_labels(df_features, pt_sl=pt_sl, num_bars=num_bars)
        if df_labeled.is_empty():
            logger.warning(f"Labeling produced empty dataset for {symbol} ({timeframe}).")
            return pl.DataFrame()

        # 3. Compute Sample Weights & Uniqueness
        pdf = df_labeled.to_pandas()
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
        pdf = pdf.set_index("timestamp").sort_index()

        close_idx = pdf.index
        close_series = pdf["close"]

        # Re-compute volatility and vertical barriers for event concurrency
        vol = TripleBarrierLabeler.get_daily_volatility(close_series, span=100)
        t1 = TripleBarrierLabeler.get_vertical_barriers(close_idx, num_bars=num_bars)

        barrier_events = pd.DataFrame(index=close_idx)
        barrier_events["t1"] = t1
        barrier_events["trgt"] = vol
        barrier_events = barrier_events.dropna(subset=["trgt"])

        # Concurrency & Uniqueness
        num_co = SampleWeights.get_num_co_events(close_idx, barrier_events)
        uniqueness = SampleWeights.get_sample_uniqueness(barrier_events, num_co)
        sample_weights = SampleWeights.get_sample_weights(barrier_events, uniqueness, pdf["target_return"])

        # Add sample_weight to pdf
        pdf["sample_weight"] = sample_weights

        # Drop rows where sample_weight is NaN
        pdf = pdf.dropna(subset=["sample_weight"])
        pdf = pdf.reset_index()

        # 4. Convert back to Polars
        result_df = pl.from_pandas(pdf)
        logger.success(f"Successfully built dataset for {symbol} with {len(result_df)} samples.")
        return result_df

    def save_dataset(self, df: pl.DataFrame, symbol: str, timeframe: str) -> Path:
        """Persist training dataset to Apache Parquet format with snappy compression."""
        if df.is_empty():
            logger.warning(f"Empty dataset for {symbol} ({timeframe}) not saved.")
            return self._get_dataset_filepath(symbol, timeframe)

        filepath = self._get_dataset_filepath(symbol, timeframe)
        logger.info(f"Saving dataset with {len(df)} samples to {filepath}...")

        df.write_parquet(filepath, compression="snappy")
        logger.success(f"Successfully saved dataset to {filepath}")
        return filepath

    def load_dataset(self, symbol: str, timeframe: str) -> pl.DataFrame:
        """Load persisted training dataset from Apache Parquet file."""
        filepath = self._get_dataset_filepath(symbol, timeframe)
        if not filepath.exists():
            logger.warning(f"Dataset file not found at {filepath}")
            return pl.DataFrame()

        logger.info(f"Loading dataset from {filepath}...")
        df = pl.read_parquet(filepath)
        logger.success(f"Loaded {len(df)} samples from {filepath}")
        return df
