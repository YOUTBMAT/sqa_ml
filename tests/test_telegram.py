"""Test script for validating Telegram Notifier integration."""

import asyncio
import sys
from src.utils.notifications import TelegramNotifier
from src.utils.logger import logger


async def main() -> None:
    """Test sending a notification message via TelegramNotifier."""
    logger.info("Initializing TelegramNotifier for integration test...")
    notifier = TelegramNotifier()

    if not notifier.enabled:
        logger.warning("TelegramNotifier is currently disabled. Please configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in sqa_ml/config/.env")
    
    logger.info("Sending test message to Telegram...")
    success = await notifier.send_message("🚀 <b>SQA-ML Telegram Integration Verified!</b>")

    if success:
        logger.success("Telegram test notification completed successfully.")
        sys.exit(0)
    else:
        logger.error("Telegram test notification failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
