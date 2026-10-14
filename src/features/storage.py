"""Feature storage management module using Apache Parquet and Snappy compression."""

from pathlib import Path
import polars as pl
from src.utils.logger import logger


class FeatureStorage:
    """Manager for persisting and loading computed features in Apache Parquet format."""

    def __init__(self, base_dir: str = "data/features") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_filepath(self, symbol: str, timeframe: str) -> Path:
        """Sanitize symbol and return parquet file path."""
        sanitized_symbol = symbol.replace("/", "_").replace("-", "_").lower()
        filename = f"{sanitized_symbol}_{timeframe}.parquet"
        return self.base_dir / filename

    def save_features(self, df: pl.DataFrame, symbol: str, timeframe: str) -> Path:
        """Save feature Polars DataFrame to Apache Parquet with snappy compression."""
        if df.is_empty():
            logger.warning(f"Empty feature DataFrame for {symbol} ({timeframe}) not saved.")
            return self._get_filepath(symbol, timeframe)

        filepath = self._get_filepath(symbol, timeframe)
        logger.info(f"Saving {len(df)} feature rows for {symbol} ({timeframe}) to {filepath}...")

        df.write_parquet(filepath, compression="snappy")
        logger.success(f"Successfully saved features to {filepath}")
        return filepath

    def load_features(self, symbol: str, timeframe: str) -> pl.DataFrame:
        """Load feature matrix from Apache Parquet file into Polars DataFrame."""
        filepath = self._get_filepath(symbol, timeframe)
        if not filepath.exists():
            logger.warning(f"Feature file not found at {filepath}")
            return pl.DataFrame()

        logger.info(f"Loading features for {symbol} ({timeframe}) from {filepath}...")
        df = pl.read_parquet(filepath)
        logger.success(f"Successfully loaded {len(df)} rows from {filepath}")
        return df

    def exists(self, symbol: str, timeframe: str) -> bool:
        """Check if feature parquet file already exists for symbol and timeframe."""
        return self._get_filepath(symbol, timeframe).exists()
