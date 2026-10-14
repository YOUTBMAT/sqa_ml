"""Asynchronous cryptocurrency market data collector using CCXT Async and Tenacity."""

from datetime import datetime, timezone
import ccxt.async_support as ccxt_async
import polars as pl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.data.collectors.base import BaseCollector
from src.utils.logger import logger


class CryptoCollector(BaseCollector):
    """Collector for cryptocurrency markets utilizing CCXT Async."""

    def __init__(self, exchange_id: str = "binance", sandbox: bool = False) -> None:
        self.exchange_id = exchange_id
        self.sandbox = sandbox
        exchange_class = getattr(ccxt_async, exchange_id)
        self.exchange = exchange_class(
            {
                "timeout": 30000,
                "enableRateLimit": True,
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                },
                "options": {"defaultType": "future"},  # Default to derivatives for funding/OI
            }
        )
        if sandbox and self.exchange.has.get("sandboxMode"):
            self.exchange.set_sandbox_mode(True)

    async def __aenter__(self) -> "CryptoCollector":
        if not self.exchange.markets:
            try:
                await self.exchange.load_markets()
            except Exception as e:
                logger.warning(f"Failed to load markets on enter: {e}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the exchange underlying HTTP session."""
        if self.exchange:
            await self.exchange.close()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((ccxt_async.NetworkError, ccxt_async.ExchangeNotAvailable, ccxt_async.RateLimitExceeded)),
        reraise=True,
    )
    async def _fetch_ohlcv_page(self, symbol: str, timeframe: str, since: int) -> list:
        """Fetch a single page of OHLCV candles with retry backoff."""
        return await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)

    async def fetch_ohlcv_historical(
        self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime
    ) -> pl.DataFrame:
        """Fetch historical OHLCV candles with pagination and exponential backoff."""
        logger.info(f"[{self.exchange_id.upper()}] Fetching OHLCV for {symbol} ({timeframe}) from {start_time} to {end_time}")
        
        since_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        all_candles = []

        try:
            while since_ms < end_ms:
                candles = await self._fetch_ohlcv_page(symbol, timeframe, since_ms)
                if not candles:
                    break
                
                first_timestamp = candles[0][0]
                # If pagination stalled, break to avoid infinite loop
                if all_candles and first_timestamp <= all_candles[-1][0]:
                    break

                for c in candles:
                    if c[0] <= end_ms:
                        all_candles.append(c)

                since_ms = candles[-1][0] + 1
                
                # If less than limit returned, we reached the end
                if len(candles) < 1000:
                    break

        except Exception as e:
            logger.error(f"Error fetching OHLCV historical for {symbol}: {e}")
            raise

        if not all_candles:
            logger.warning(f"No OHLCV data found for {symbol} in given range.")
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
                    "symbol": pl.String,
                    "asset_class": pl.String,
                    "timeframe": pl.String,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                    "trades_count": pl.Int64,
                }
            )

        # Convert to Polars DataFrame
        df = pl.DataFrame(
            all_candles,
            schema=["timestamp_ms", "open", "high", "low", "close", "volume"],
            orient="row",
        )

        df = df.with_columns(
            [
                pl.from_epoch("timestamp_ms", time_unit="ms").dt.replace_time_zone("UTC").alias("timestamp"),
                pl.lit(symbol).alias("symbol"),
                pl.lit("CRYPTO").alias("asset_class"),
                pl.lit(timeframe).alias("timeframe"),
                pl.lit(None, dtype=pl.Int64).alias("trades_count"),
            ]
        ).select(
            [
                "timestamp",
                "symbol",
                "asset_class",
                "timeframe",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trades_count",
            ]
        )

        logger.success(f"Fetched {len(df)} candles for {symbol}")
        return df

    async def fetch_market_metrics(self, symbol: str) -> pl.DataFrame:
        """Fetch dynamic market metrics (Funding Rate and Open Interest)."""
        logger.info(f"[{self.exchange_id.upper()}] Fetching market metrics for {symbol}")
        metrics = []
        now = datetime.now(timezone.utc)

        try:
            # Fetch Funding Rate if supported
            if self.exchange.has.get("fetchFundingRate"):
                funding = await self.exchange.fetch_funding_rate(symbol)
                rate = funding.get("fundingRate")
                if rate is not None:
                    metrics.append(
                        {
                            "timestamp": now,
                            "symbol": symbol,
                            "metric_name": "funding_rate",
                            "metric_value": float(rate),
                        }
                    )

            # Fetch Open Interest if supported
            if self.exchange.has.get("fetchOpenInterest"):
                oi = await self.exchange.fetch_open_interest(symbol)
                amount = oi.get("openInterestAmount")
                if amount is not None:
                    metrics.append(
                        {
                            "timestamp": now,
                            "symbol": symbol,
                            "metric_name": "open_interest",
                            "metric_value": float(amount),
                        }
                    )
        except Exception as e:
            logger.warning(f"Could not fetch full market metrics for {symbol}: {e}")

        if not metrics:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
                    "symbol": pl.String,
                    "metric_name": pl.String,
                    "metric_value": pl.Float64,
                }
            )

        df = pl.DataFrame(metrics)
        return df
