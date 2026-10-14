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
        """Run backtest simulation with Triple Barrier exits (SL = 1*ATR, TP = 2*ATR, t1 = 24h)."""
        logger.info("Running backtest simulation with Triple Barrier exits...")
        
        cols = [c for c in ["timestamp", "open", "high", "low", "close"] if c in df.columns]
        pdf = df.select(cols).to_pandas()
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
        pdf = pdf.set_index("timestamp").sort_index()

        if proba_predictions.shape[1] == 3:
            prob_short = proba_predictions[:, 0]
            prob_long = proba_predictions[:, 2]
        else:
            raise ValueError("Expected probability predictions of shape (N, 3).")

        # Compute ATR if high and low are available
        if "high" in pdf.columns and "low" in pdf.columns:
            tr1 = pdf["high"] - pdf["low"]
            tr2 = (pdf["high"] - pdf["close"].shift(1)).abs()
            tr3 = (pdf["low"] - pdf["close"].shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.ewm(span=14).mean().fillna(pdf["close"] * 0.01)
        else:
            atr = pdf["close"] * 0.01

        n = len(pdf)
        positions = np.zeros(n)
        strategy_returns = np.zeros(n)
        transaction_costs = np.zeros(n)

        cost_rate = self.fee_rate + self.slippage

        i = 0
        while i < n - 1:
            is_long = prob_long[i] > self.buy_threshold
            is_short = prob_short[i] > self.sell_threshold

            if is_long or is_short:
                side = 1.0 if is_long else -1.0
                entry_idx = i + 1
                if entry_idx >= n:
                    break
                
                entry_price = pdf["close"].iloc[entry_idx]
                current_atr = atr.iloc[entry_idx]
                if pd.isna(current_atr) or current_atr <= 0:
                    current_atr = entry_price * 0.01

                tp_price = entry_price + (1.5 * current_atr * side)
                sl_price = entry_price - (1.0 * current_atr * side)

                exit_idx = entry_idx
                exit_price = entry_price

                for j in range(entry_idx, min(entry_idx + 24, n)):
                    bar_high = pdf["high"].iloc[j] if "high" in pdf.columns else pdf["close"].iloc[j]
                    bar_low = pdf["low"].iloc[j] if "low" in pdf.columns else pdf["close"].iloc[j]
                    bar_close = pdf["close"].iloc[j]

                    if side == 1:  # Long
                        if bar_high >= tp_price:
                            exit_idx = j
                            exit_price = tp_price
                            break
                        elif bar_low <= sl_price:
                            exit_idx = j
                            exit_price = sl_price
                            break
                    else:  # Short
                        if bar_low <= tp_price:
                            exit_idx = j
                            exit_price = tp_price
                            break
                        elif bar_high >= sl_price:
                            exit_idx = j
                            exit_price = sl_price
                            break
                    
                    if j == entry_idx + 23 or j == n - 1:
                        exit_idx = j
                        exit_price = bar_close

                positions[entry_idx:exit_idx+1] = side

                raw_ret = (exit_price - entry_price) / entry_price * side
                net_ret = raw_ret - (2.0 * cost_rate)

                strategy_returns[exit_idx] = net_ret
                transaction_costs[entry_idx] = cost_rate
                transaction_costs[exit_idx] = cost_rate

                i = exit_idx
            else:
                positions[i] = 0.0
                i += 1

        pdf["position"] = positions
        pdf["net_strategy_returns"] = strategy_returns
        pdf["transaction_costs"] = transaction_costs
        pdf["equity"] = self.initial_capital * (1.0 + pdf["net_strategy_returns"]).cumprod()

        logger.success("Backtest simulation with Triple Barrier exits completed.")
        return pdf
