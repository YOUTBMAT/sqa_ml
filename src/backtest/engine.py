"""Vectorized backtesting engine simulating strategy returns, transaction costs, and slippage."""

import numpy as np
import pandas as pd
import polars as pl
from src.utils.logger import logger


class VectorizedBacktester:
    """Vectorized backtester for evaluating quantitative trading strategies."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.0005,
        slippage: float = 0.0002,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
    ) -> None:
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def run(self, df: pl.DataFrame, proba_predictions: np.ndarray) -> pd.DataFrame:
        """Run vectorized backtest given price data and model class probabilities.

        proba_predictions: array of shape (N, 3) where columns correspond to classes (-1, 0, 1) or (0, 1, 2).
        Assuming classes are mapped as: 0 -> Short (-1), 1 -> Neutral (0), 2 -> Long (+1).
        """
        logger.info("Running vectorized backtest simulation...")
        pdf = df.select(["timestamp", "close"]).to_pandas()
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
        pdf = pdf.set_index("timestamp").sort_index()

        # Extract probabilities for Short (class 0), Neutral (class 1), Long (class 2)
        if proba_predictions.shape[1] == 3:
            prob_short = proba_predictions[:, 0]
            prob_long = proba_predictions[:, 2]
        else:
            raise ValueError("Expected probability predictions of shape (N, 3).")

        # Determine positions: +1 for Long if prob_long > buy_threshold, -1 for Short if prob_short > sell_threshold else 0
        positions = np.zeros(len(pdf))
        for i in range(len(pdf)):
            if prob_long[i] > self.buy_threshold:
                positions[i] = 1.0
            elif prob_short[i] > self.sell_threshold:
                positions[i] = -1.0
            else:
                positions[i] = 0.0 if i == 0 else positions[i - 1] # hold previous position or flat

        pdf["position"] = positions
        pdf["returns"] = pdf["close"].pct_change().fillna(0.0)
        
        # Strategy returns = position_t * returns_{t+1} (or lagged position)
        pdf["strategy_returns"] = pdf["position"].shift(1).fillna(0.0) * pdf["returns"]

        # Transaction costs & slippage upon position change
        pdf["position_change"] = pdf["position"].diff().abs().fillna(0.0)
        cost_rate = self.fee_rate + self.slippage
        pdf["transaction_costs"] = pdf["position_change"] * cost_rate

        # Net strategy returns after costs
        pdf["net_strategy_returns"] = pdf["strategy_returns"] - pdf["transaction_costs"]

        # Equity curve
        pdf["equity"] = self.initial_capital * (1.0 + pdf["net_strategy_returns"]).cumprod()

        logger.success("Vectorized backtest simulation completed.")
        return pdf
