"""Configuration management using Pydantic Settings."""

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT_ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_ENV_FILE = Path(__file__).resolve().parent / ".env"
ENV_FILE = ROOT_ENV_FILE if ROOT_ENV_FILE.is_file() else CONFIG_ENV_FILE


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    POSTGRES_USER: str = Field(..., min_length=1)
    POSTGRES_PASSWORD: str = Field(..., min_length=1)
    POSTGRES_HOST: str = Field(..., min_length=1)
    POSTGRES_PORT: int = Field(..., gt=0, le=65535)
    POSTGRES_DB: str = Field(..., min_length=1)
    DATABASE_URL: str = Field(..., min_length=1)

    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    DEFAULT_EXCHANGE: str = "binance"
    EXCHANGE_SANDBOX: bool = True
    EXCHANGE_API_KEY: str = ""
    EXCHANGE_API_SECRET: str = ""

    CALIBRATED_WIN_RATE: float = Field(0.5, ge=0.0, le=1.0)
    RISK_PER_TRADE_PCT: float = Field(0.01, gt=0.0, lt=1.0)

    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    SYMBOLS: list[str] = ["ETH/USDT"]
    PROB_THRESHOLD: float = 0.62
    PT_SL_RATIO: list[float] = [1.5, 1.0]
    MAX_HOLD_HOURS: int = 24
    ATR_PERIOD: int = 14
    MODEL_FEATURES: list[str] = [
        "vol_parkinson_pct",
        "ema_200_dist",
        "frac_diff_close",
        "hurst_100",
        "vol_garman_klass",
        "vol_ratio_6_24",
        "vol_parkinson",
        "rsi_adaptive",
        "vwap_lower_2_dist",
        "adx_14",
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
    MODEL_PARAMS: dict = {
        "max_depth": 3,
        "num_leaves": 7,
        "learning_rate": 0.03,
        "min_child_samples": 50,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
    }

    @model_validator(mode="after")
    def validate_live_exchange_access(self) -> "Settings":
        if not self.EXCHANGE_SANDBOX:
            if self.ENVIRONMENT.lower() != "production":
                raise ValueError(
                    "ENVIRONMENT=production is required for non-sandbox exchange access."
                )
            if not self.EXCHANGE_API_KEY or not self.EXCHANGE_API_SECRET:
                raise ValueError(
                    "EXCHANGE_API_KEY and EXCHANGE_API_SECRET are required "
                    "for non-sandbox exchange access."
                )
        return self

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
