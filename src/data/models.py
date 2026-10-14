"""SQLAlchemy 2.0+ ORM models for TimescaleDB hypertables."""

from datetime import datetime
from sqlalchemy import (
    String,
    Numeric,
    Integer,
    Float,
    DateTime,
    CheckConstraint,
    PrimaryKeyConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy models."""
    pass


class OhlcvCandle(Base):
    """ORM model for OHLCV candlestick time-series data."""

    __tablename__ = "ohlcv_candles"

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True, nullable=False)
    asset_class: Mapped[str] = mapped_column(String(10), nullable=False)  # 'CRYPTO' or 'STOCK'
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)   # e.g., '1m', '1h', '1d'
    open: Mapped[float] = mapped_column(Numeric(precision=18, scale=8), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(precision=18, scale=8), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(precision=18, scale=8), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(precision=18, scale=8), nullable=False)
    volume: Mapped[float] = mapped_column(Numeric(precision=24, scale=8), nullable=False)
    trades_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        CheckConstraint("asset_class IN ('CRYPTO', 'STOCK')", name="check_asset_class"),
        PrimaryKeyConstraint("timestamp", "symbol", name="pk_ohlcv_candles"),
    )

    def __repr__(self) -> str:
        return (
            f"<OhlcvCandle(timestamp={self.timestamp}, symbol='{self.symbol}', "
            f"timeframe='{self.timeframe}', close={self.close})>"
        )


class MarketMetric(Base):
    """ORM model for dynamic market metrics (e.g., funding_rate, open_interest, vwap)."""

    __tablename__ = "market_metrics"

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True, nullable=False)
    metric_name: Mapped[str] = mapped_column(String(50), primary_key=True, nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("timestamp", "symbol", "metric_name", name="pk_market_metrics"),
    )

    def __repr__(self) -> str:
        return (
            f"<MarketMetric(timestamp={self.timestamp}, symbol='{self.symbol}', "
            f"metric_name='{self.metric_name}', value={self.metric_value})>"
        )
