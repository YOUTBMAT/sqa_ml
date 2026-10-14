"""Structured logging configuration using Loguru."""

import sys
from loguru import logger
from config.settings import settings

def setup_logger() -> None:
    """Configure loguru logger with formatting and log level."""
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

setup_logger()

__all__ = ["logger"]
