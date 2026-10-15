"""Continuous hourly backtesting engine simulating strategy returns, transaction costs, slippage, and Triple Barrier exits (1.5x ATR TP / 1.0x ATR SL / 24h limit)."""

import numpy as np
import pandas as pd
import polars as pl
from config.settings import settings
from src.utils.logger import logger


class VectorizedBacktester:
    """Continuous hourly backtester for evaluating quantitative trading strategies over continuous price series."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.0005,
        slippage: float = 0.0002,
        buy_threshold: float = 0.75,
        sell_threshold: float = 0.25,
    ) -> None:
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def run(self, df: pl.DataFrame, proba_predictions: np.ndarray) -> pd.DataFrame:
        """Run continuous hourly simulation over all bars with 1.5x ATR TP, 1.0x ATR SL, and 24h time barrier."""
        logger.info("Running continuous hourly backtest simulation over price series...")
        
        cols = [c for c in ["timestamp", "open", "high", "low", "close"] if c in df.columns]
        pdf = df.select(cols).to_pandas()
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
        pdf = pdf.set_index("timestamp").sort_index()

        if proba_predictions.shape[1] == 3:
            prob_short = proba_predictions[:, 0]
            prob_long = proba_predictions[:, 2]
        else:
            raise ValueError("Expected probability predictions of shape (N, 3).")

        # Compute ATR-14
        if "high" in pdf.columns and "low" in pdf.columns:
            tr1 = pdf["high"] - pdf["low"]
            tr2 = (pdf["high"] - pdf["close"].shift(1)).abs()
            tr3 = (pdf["low"] - pdf["close"].shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.ewm(span=settings.ATR_PERIOD).mean().fillna(pdf["close"] * 0.01)
        else:
            atr = pdf["close"] * 0.01

        n = len(pdf)
        positions = np.zeros(n)
        strategy_returns = np.zeros(n)
        transaction_costs = np.zeros(n)
        trade_durations = []

        cost_rate = self.fee_rate + self.slippage
        active_position = None

        for i in range(n):
            current_price = pdf["close"].iloc[i]
            bar_high = pdf["high"].iloc[i] if "high" in pdf.columns else current_price
            bar_low = pdf["low"].iloc[i] if "low" in pdf.columns else current_price
            current_atr = atr.iloc[i]
            if pd.isna(current_atr) or current_atr <= 0:
                current_atr = current_price * 0.01

            # 1. Manage active position bar-by-bar
            if active_position is not None:
                side = active_position["side"]
                entry_price = active_position["entry_price"]
                tp_price = active_position["tp_price"]
                sl_price = active_position["sl_price"]
                entry_idx = active_position["entry_idx"]
                holding_hours = i - entry_idx

                hit = False
                exit_price = current_price

                if side == 1:
                    if bar_high >= tp_price:
                        hit = True
                        exit_price = tp_price
                    elif bar_low <= sl_price:
                        hit = True
                        exit_price = sl_price
                else: # short
                    if bar_low <= tp_price:
                        hit = True
                        exit_price = tp_price
                    elif bar_high >= sl_price:
                        hit = True
                        exit_price = sl_price

                if not hit and holding_hours >= settings.MAX_HOLD_HOURS:
                    hit = True
                    exit_price = current_price

                if hit:
                    raw_ret = (exit_price - entry_price) / entry_price * side
                    net_ret = raw_ret - (2.0 * cost_rate)
                    strategy_returns[i] = net_ret
                    transaction_costs[i] = cost_rate
                    trade_durations.append(holding_hours)
                    active_position = None
                else:
                    positions[i] = side
                    continue

            # 2. Check for new entry signal if no active position
            if active_position is None and i < n - 1:
                is_long = prob_long[i] > self.buy_threshold
                is_short = prob_short[i] > self.sell_threshold

                if is_long or is_short:
                    side = 1.0 if is_long else -1.0
                    entry_price = pdf["close"].iloc[i]
                    pt_mult, sl_mult = settings.PT_SL_RATIO

                    if side == 1:
                        tp_price = entry_price + (pt_mult * current_atr)
                        sl_price = entry_price - (sl_mult * current_atr)
                    else:
                        tp_price = entry_price - (pt_mult * current_atr)
                        sl_price = entry_price + (sl_mult * current_atr)

                    active_position = {
                        "side": side,
                        "entry_price": entry_price,
                        "tp_price": tp_price,
                        "sl_price": sl_price,
                        "entry_idx": i,
                    }
                    positions[i] = side
                    transaction_costs[i] = cost_rate

        pdf["position"] = positions
        pdf["net_strategy_returns"] = strategy_returns
        pdf["transaction_costs"] = transaction_costs
        pdf["equity"] = self.initial_capital * (1.0 + pdf["net_strategy_returns"]).cumprod()

        median_duration = float(np.median(trade_durations)) if trade_durations else 0.0
        logger.success(f"Continuous hourly backtest simulation completed. Total trades: {len(trade_durations)}, Median duration: {median_duration:.1f}h")
        return pdf
