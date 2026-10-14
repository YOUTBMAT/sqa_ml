"""Healthcheck utility to validate database connection, TimescaleDB extension, and external CCXT APIs."""

import ccxt
from sqlalchemy import text
from src.data.database import engine
from src.utils.logger import logger
from config.settings import settings


def check_database() -> bool:
    """Validate database connectivity and simple query execution."""
    logger.info("Checking database connection...")
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1;")).scalar()
            if result == 1:
                logger.success("Database connection is ACTIVE.")
                return True
            else:
                logger.error("Database query returned unexpected result.")
                return False
    except Exception as e:
        logger.error(f"Database connection FAILED: {e}")
        return False


def check_timescaledb_extension() -> bool:
    """Validate if TimescaleDB extension is installed and active."""
    logger.info("Checking TimescaleDB extension...")
    try:
        with engine.connect() as connection:
            version = connection.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';")
            ).scalar()
            if version:
                logger.success(f"TimescaleDB extension is ACTIVE (version: {version}).")
                return True
            else:
                logger.error("TimescaleDB extension is NOT installed or active.")
                return False
    except Exception as e:
        logger.error(f"Failed to verify TimescaleDB extension: {e}")
        return False


def check_ccxt_api(symbol: str = "BTC/USDT") -> bool:
    """Test public API connectivity to the exchange via CCXT."""
    logger.info(f"Checking CCXT public API connectivity ({settings.DEFAULT_EXCHANGE} - {symbol})...")
    try:
        exchange_class = getattr(ccxt, settings.DEFAULT_EXCHANGE)
        exchange = exchange_class({"enableRateLimit": True})
        ticker = exchange.fetch_ticker(symbol)
        last_price = ticker.get("last")
        logger.success(f"CCXT API connection SUCCESSFUL. {symbol} current price: {last_price}")
        return True
    except Exception as e:
        logger.error(f"CCXT API connection FAILED: {e}")
        return False


def run_healthcheck() -> bool:
    """Run all system healthchecks and return True if all pass, False otherwise."""
    logger.info("=== STARTING SQA-ML SYSTEM HEALTHCHECK ===")
    
    db_status = check_database()
    timescale_status = check_timescaledb_extension() if db_status else False
    ccxt_status = check_ccxt_api()

    all_passed = db_status and timescale_status and ccxt_status

    if all_passed:
        logger.success("=== ALL HEALTHCHECKS PASSED SUCCESSFULLY ===")
    else:
        logger.warning("=== SOME HEALTHCHECKS FAILED ===")

    return all_passed


if __name__ == "__main__":
    passed = run_healthcheck()
    import sys
    sys.exit(0 if passed else 1)
