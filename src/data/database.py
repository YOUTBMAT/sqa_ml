"""Database connection and session management using SQLAlchemy 2.0+ and Psycopg."""

from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from config.settings import settings
from src.utils.logger import logger
from src.data.models import Base

# Create SQLAlchemy engine with psycopg driver
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def init_db() -> None:
    """Initialize database tables and convert them to TimescaleDB hypertables if not already done."""
    logger.info("Initializing database schema and TimescaleDB extension...")
    try:
        with engine.begin() as connection:
            # Ensure TimescaleDB extension is enabled
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
            logger.success("TimescaleDB extension verified/enabled.")

        # Create standard tables defined in models
        Base.metadata.create_all(bind=engine)
        logger.success("Database tables created successfully.")

        # Convert tables to TimescaleDB hypertables safely
        with engine.begin() as connection:
            # Check and convert ohlcv_candles
            res_ohlcv = connection.execute(
                text(
                    "SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'ohlcv_candles';"
                )
            ).fetchone()
            if not res_ohlcv:
                connection.execute(
                    text("SELECT create_hypertable('ohlcv_candles', 'timestamp', if_not_exists => TRUE);")
                )
                logger.success("Converted table 'ohlcv_candles' to TimescaleDB hypertable.")
            else:
                logger.info("Table 'ohlcv_candles' is already a hypertable.")

            # Check and convert market_metrics
            res_metrics = connection.execute(
                text(
                    "SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'market_metrics';"
                )
            ).fetchone()
            if not res_metrics:
                connection.execute(
                    text("SELECT create_hypertable('market_metrics', 'timestamp', if_not_exists => TRUE);")
                )
                logger.success("Converted table 'market_metrics' to TimescaleDB hypertable.")
            else:
                logger.info("Table 'market_metrics' is already a hypertable.")

    except Exception as e:
        logger.error(f"Failed to initialize database / hypertables: {e}")
        raise


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session rollback due to error: {e}")
        raise
    finally:
        session.close()
