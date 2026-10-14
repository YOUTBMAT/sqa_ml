"""Performance metrics calculation module for quantitative backtesting evaluation."""

import numpy as np
import pandas as pd
from src.utils.logger import logger


class PerformanceMetrics:
    """Calculator for quantitative trading performance and risk metrics."""

    @staticmethod
    def calculate_metrics(results_df: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
        """Calculate comprehensive performance metrics from backtest results DataFrame."""
        logger.info("Calculating performance metrics...")
        equity = results_df["equity"]
        net_returns = results_df["net_strategy_returns"]
        transaction_costs = results_df["transaction_costs"]

        # Total Return
        final_equity = equity.iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital

        # Annualized Sharpe Ratio
        periods_per_year = 365 * 24
        mean_ret = net_returns.mean()
        std_ret = net_returns.std()
        sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(periods_per_year) if std_ret > 0 else 0.0

        # Sortino Ratio (downside deviation)
        negative_returns = net_returns[net_returns < 0]
        downside_std = negative_returns.std()
        sortino = (mean_ret / (downside_std + 1e-8)) * np.sqrt(periods_per_year) if downside_std > 0 else 0.0

        # Maximum Drawdown (MDD)
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar = (total_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0

        # Win Rate & Profit Factor per completed round-trip trade (or fallback to candle returns if no position col)
        if "position" in results_df.columns:
            positions = results_df["position"].values
            trade_returns = []
            current_trade_return = 0.0
            in_trade = False

            for i in range(1, len(positions)):
                prev_pos = positions[i - 1]
                curr_pos = positions[i]
                ret = net_returns.iloc[i] if hasattr(net_returns, "iloc") else net_returns[i]

                if prev_pos == 0 and curr_pos != 0:
                    in_trade = True
                    current_trade_return = ret
                elif in_trade:
                    current_trade_return += ret
                    if curr_pos == 0 or curr_pos != prev_pos:
                        trade_returns.append(current_trade_return)
                        in_trade = False
                        current_trade_return = 0.0

            if in_trade:
                trade_returns.append(current_trade_return)

            trade_returns = np.array(trade_returns)
            winning_trades = trade_returns[trade_returns > 0]
            losing_trades = trade_returns[trade_returns < 0]

            total_trades = len(trade_returns)
            win_rate = (len(winning_trades) / total_trades) if total_trades > 0 else 0.0

            gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0.0
            gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0.0
            profit_factor = (gross_profit / (gross_loss + 1e-8)) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        else:
            winning_trades = net_returns[net_returns > 0]
            losing_trades = net_returns[net_returns < 0]
            total_trades = len(winning_trades) + len(losing_trades)
            win_rate = (len(winning_trades) / total_trades) if total_trades > 0 else 0.0
            gross_profit = winning_trades.sum()
            gross_loss = abs(losing_trades.sum())
            profit_factor = (gross_profit / (gross_loss + 1e-8)) if gross_loss > 0 else 0.0

        # Total Fees Paid
        total_fees = transaction_costs.sum() * initial_capital

        metrics = {
            "initial_capital": float(initial_capital),
            "final_equity": float(final_equity),
            "total_return_pct": float(total_return * 100.0),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown_pct": float(max_drawdown * 100.0),
            "calmar_ratio": float(calmar),
            "win_rate_pct": float(win_rate * 100.0),
            "profit_factor": float(profit_factor) if not np.isnan(profit_factor) else 0.0,
            "total_trades": int(total_trades),
            "total_fees_paid": float(total_fees),
        }

        logger.success(f"Performance metrics calculated successfully: {metrics}")
        return metrics
