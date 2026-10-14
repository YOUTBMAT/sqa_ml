"""ETL pipeline orchestrator for historical and incremental market data ingestion."""

from datetime import datetime, timezone, timedelta
from src.data.collectors.base import BaseCollector
from src.data.validator import DataValidator
from src.data.storage import DataStorage
from src.utils.logger import logger


class ETLPipeline:
    """Orchestrator for ETL pipeline execution (Historical & Incremental)."""

    def __init__(self, collector: BaseCollector, asset_class: str) -> None:
        self.collector = collector
        self.asset_class = asset_class.upper()

    async def run(self, symbol: str, timeframe: str, default_start_date: datetime | None = None) -> int:
        """Execute ETL pipeline for a given symbol and timeframe.

        1. Check latest recorded timestamp in DB.
        2. Determine start_time (latest_timestamp or default_start_date).
        3. Fetch OHLCV data from collector.
        4. Validate data with DataValidator.
        5. Bulk upsert into TimescaleDB via DataStorage.
        6. Fetch and store market metrics (if crypto/derivatives).
        """
        logger.info(f"Starting ETL pipeline for {symbol} ({self.asset_class}) - Timeframe: {timeframe}")
        
        now = datetime.now(timezone.utc)
        
        # 1. Get latest timestamp from DB for incremental sync
        latest_ts = DataStorage.get_latest_timestamp(symbol, timeframe)

        if latest_ts:
            # Incremental sync: start from latest timestamp
            start_time = latest_ts
            logger.info(f"Incremental sync: fetching data from latest DB timestamp {start_time} to {now}")
        else:
            # Historical sync: default to 1 year ago or provided start date
            start_time = default_start_date or (now - timedelta(days=365))
            logger.info(f"Initial historical sync: fetching data from {start_time} to {now}")

        if start_time >= now:
            logger.info(f"Data for {symbol} is already up to date.")
            return 0

        # 2. Fetch OHLCV historical/incremental data
        raw_df = await self.collector.fetch_ohlcv_historical(symbol, timeframe, start_time, now)
        if raw_df.is_empty():
            logger.warning(f"No new data retrieved for {symbol}.")
            return 0

        # 3. Validate data
        validated_df = DataValidator.validate_ohlcv(raw_df)
        if validated_df.is_empty():
            logger.warning(f"All retrieved data for {symbol} was filtered out during validation.")
            return 0

        # 4. Store OHLCV candles
        inserted_candles = DataStorage.bulk_upsert_candles(validated_df)

        # 5. Fetch and store dynamic market metrics (if applicable)
        try:
            metrics_df = await self.collector.fetch_market_metrics(symbol)
            if not metrics_df.is_empty():
                DataStorage.bulk_upsert_metrics(metrics_df)
        except Exception as e:
            logger.warning(f"Failed to process market metrics for {symbol}: {e}")

        logger.success(f"ETL pipeline completed successfully for {symbol}. Processed {inserted_candles} records.")
        return inserted_candles
