"""Configuration management using Pydantic Settings."""

import os
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "sqadb"
    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5432/sqadb"

    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    DEFAULT_EXCHANGE: str = "binance"
    EXCHANGE_SANDBOX: bool = True
    EXCHANGE_API_KEY: str = "test_api_key_12345"
    EXCHANGE_API_SECRET: str = "test_api_secret_12345"

    # Telegram Notification Settings
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Production Trading & Modeling Configuration
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
    def validate_exchange_credentials(self) -> "Settings":
        if self.ENVIRONMENT.lower() == "production" or not self.EXCHANGE_SANDBOX:
            if not self.EXCHANGE_API_KEY or not self.EXCHANGE_API_SECRET or self.EXCHANGE_API_KEY in ("placeholder_key", ""):
                raise ValueError("EXCHANGE_API_KEY and EXCHANGE_API_SECRET must be provided and non-empty when ENVIRONMENT is production or EXCHANGE_SANDBOX is False.")
        return self

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
