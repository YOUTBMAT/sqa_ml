"""Model storage management module for persisting models via joblib and metadata via JSON."""

import json
from pathlib import Path
import joblib
from src.models.trainer import QuantModelTrainer
from src.utils.logger import logger


class ModelStorage:
    """Manager for saving and loading trained models and metadata artifacts."""

    def __init__(self, models_dir: str = "models/saved") -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _get_paths(self, symbol: str, timeframe: str) -> tuple[Path, Path]:
        """Return file paths for model joblib artifact and metadata JSON."""
        sanitized_symbol = symbol.replace("/", "_").replace("-", "_").lower()
        model_path = self.models_dir / f"{sanitized_symbol}_{timeframe}_model.joblib"
        meta_path = self.models_dir / f"{sanitized_symbol}_{timeframe}_metadata.json"
        return model_path, meta_path

    def save_model(self, model: QuantModelTrainer, symbol: str, timeframe: str, metadata: dict) -> Path:
        """Save model instance using joblib and metadata dictionary using JSON."""
        model_path, meta_path = self._get_paths(symbol, timeframe)
        logger.info(f"Saving model artifact to {model_path}...")

        joblib.dump(model, model_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.success(f"Successfully saved model and metadata for {symbol} ({timeframe})")
        return model_path

    def load_model(self, symbol: str, timeframe: str) -> tuple[QuantModelTrainer, dict]:
        """Load trained model instance and metadata dictionary."""
        model_path, meta_path = self._get_paths(symbol, timeframe)
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Model or metadata not found for {symbol} ({timeframe}) at {self.models_dir}")

        logger.info(f"Loading model artifact from {model_path}...")
        model = joblib.load(model_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        logger.success(f"Successfully loaded model for {symbol} ({timeframe})")
        return model, metadata

    def exists(self, symbol: str, timeframe: str) -> bool:
        """Check if both model and metadata artifacts exist."""
        model_path, meta_path = self._get_paths(symbol, timeframe)
        return model_path.exists() and meta_path.exists()
