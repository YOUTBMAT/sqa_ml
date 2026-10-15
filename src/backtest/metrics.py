"""Performance metrics computed per closed round-trip trade."""

import numpy as np
import pandas as pd
from src.utils.logger import logger


class PerformanceMetrics:
    """Calculator for trade-level performance and risk metrics."""

    @staticmethod
    def calculate_metrics(trades: pd.DataFrame, initial_capital: float, span_years: float) -> dict:
        """Calculate metrics from the backtester's trade list.

        Sharpe/Sortino are per-trade ratios annualized by the observed trades per year.
        Max drawdown is measured on closed-trade equity, starting from the initial capital.
        """
        logger.info("Calculating performance metrics...")
        returns = trades["net_return"].astype(float)
        n_trades = len(returns)
        if n_trades == 0:
            return {
                "initial_capital": float(initial_capital), "final_equity": float(initial_capital),
                "total_return_pct": 0.0, "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "max_drawdown_pct": 0.0,
                "calmar_ratio": 0.0, "win_rate_pct": 0.0, "profit_factor": 0.0, "total_trades": 0,
                "total_fees_paid": 0.0,
            }

        equity = pd.concat([pd.Series([initial_capital]), trades["equity"].astype(float)], ignore_index=True)
        final_equity = float(equity.iloc[-1])
        total_return = (final_equity - initial_capital) / initial_capital

        annualizer = np.sqrt(n_trades / span_years) if span_years > 0 else 0.0
        std = returns.std()
        sharpe = float(returns.mean() / std * annualizer) if n_trades > 1 and std > 0 else 0.0
        downside_dev = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
        sortino = float(returns.mean() / downside_dev * annualizer) if downside_dev > 0 else 0.0

        max_drawdown = float(((equity - equity.cummax()) / equity.cummax()).min())
        calmar = total_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        gross_profit = float(returns[returns > 0].sum())
        gross_loss = float(-returns[returns < 0].sum())
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0

        metrics = {
            "initial_capital": float(initial_capital),
            "final_equity": final_equity,
            "total_return_pct": total_return * 100.0,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_drawdown * 100.0,
            "calmar_ratio": float(calmar),
            "win_rate_pct": float((returns > 0).mean() * 100.0),
            "profit_factor": profit_factor,
            "total_trades": n_trades,
            "total_fees_paid": float(trades["fees_paid"].sum()),
        }
        logger.success(f"Performance metrics calculated successfully: {metrics}")
        return metrics

    @staticmethod
    def span_years(bars_timestamps: pd.Series) -> float:
        """Length of the simulated period in years."""
        ts = pd.to_datetime(bars_timestamps)
        return (ts.max() - ts.min()).total_seconds() / (365.25 * 24 * 3600)
