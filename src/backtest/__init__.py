"""Backtest package for SQA-ML strategy simulation and performance evaluation."""

from src.backtest.engine import VectorizedBacktester
from src.backtest.metrics import PerformanceMetrics

__all__ = ["VectorizedBacktester", "PerformanceMetrics"]
