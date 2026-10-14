"""High-performance database storage and bulk upsert operations using SQLAlchemy and TimescaleDB with batching."""

from datetime import datetime
import polars
import polars as pl
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from src.data.database import engine, get_db_session
from src.data.models import OhlcvCandle, MarketMetric
from src.utils.logger import logger


class DataStorage:
    """Storage manager for high-performance bulk upserts and state retrieval."""

    @staticmethod
    def get_latest_timestamp(symbol: str, timeframe: str) -> datetime | None:
        """Retrieve the latest recorded timestamp for a symbol and timeframe."""
        query = text(
            """
            SELECT MAX(timestamp) 
            FROM ohlcv_candles 
            WHERE symbol = :symbol AND timeframe = :timeframe;
            """
        )
        try:
            with get_db_session() as session:
                result = session.execute(query, {"symbol": symbol, "timeframe": timeframe}).scalar()
                if result:
                    logger.info(f"Latest timestamp for {symbol} ({timeframe}): {result}")
                    return result
                logger.info(f"No existing records found for {symbol} ({timeframe}).")
                return None
        except Exception as e:
            logger.error(f"Failed to get latest timestamp for {symbol}: {e}")
            return None

    @staticmethod
    def bulk_upsert_candles(df: polars.DataFrame, batch_size: int = 1000) -> int:
        """Perform high-performance bulk UPSERT of OHLCV candles into TimescaleDB in batches.

        Uses PostgreSQL ON CONFLICT (timestamp, symbol) DO UPDATE.
        """
        if df.is_empty():
            logger.warning("Empty DataFrame provided for bulk upsert.")
            return 0

        records = df.to_dicts()
        total_records = len(records)
        logger.info(f"Performing bulk upsert for {total_records} OHLCV candles in batches of {batch_size}...")

        total_inserted = 0
        try:
            with get_db_session() as session:
                for i in range(0, total_records, batch_size):
                    batch = records[i : i + batch_size]
                    stmt = insert(OhlcvCandle).values(batch)
                    upsert_stmt = stmt.on_conflict_do_update(
                        index_elements=["timestamp", "symbol"],
                        set_={
                            "open": stmt.excluded.open,
                            "high": stmt.excluded.high,
                            "low": stmt.excluded.low,
                            "close": stmt.excluded.close,
                            "volume": stmt.excluded.volume,
                            "trades_count": stmt.excluded.trades_count,
                        },
                    )
                    session.execute(upsert_stmt)
                    total_inserted += len(batch)
                session.commit()

            logger.success(f"Successfully upserted {total_inserted} OHLCV candles.")
            return total_inserted
        except Exception as e:
            logger.error(f"Bulk upsert failed for OHLCV candles: {e}")
            raise

    @staticmethod
    def bulk_upsert_metrics(df: polars.DataFrame, batch_size: int = 1000) -> int:
        """Perform high-performance bulk UPSERT of market metrics into TimescaleDB in batches."""
        if df.is_empty():
            return 0

        records = df.to_dicts()
        total_records = len(records)
        logger.info(f"Performing bulk upsert for {total_records} market metrics in batches of {batch_size}...")

        total_inserted = 0
        try:
            with get_db_session() as session:
                for i in range(0, total_records, batch_size):
                    batch = records[i : i + batch_size]
                    stmt = insert(MarketMetric).values(batch)
                    upsert_stmt = stmt.on_conflict_do_update(
                        index_elements=["timestamp", "symbol", "metric_name"],
                        set_={
                            "metric_value": stmt.excluded.metric_value,
                        },
                    )
                    session.execute(upsert_stmt)
                    total_inserted += len(batch)
                session.commit()

            logger.success(f"Successfully upserted {total_inserted} market metrics.")
            return total_inserted
        except Exception as e:
            logger.error(f"Bulk upsert failed for market metrics: {e}")
            raise
