"""Continuous hourly backtesting engine sharing the labeling triple barrier (1.5x ATR TP / 1.0x ATR SL / 24h limit)."""

import numpy as np
import pandas as pd
import polars as pl
from config.settings import settings
from src.labeling.barriers import atr, hours_to_bars, simulate_trade, timeframe_to_seconds
from src.risk.portfolio import PortfolioManager
from src.utils.logger import logger

TRADE_COLUMNS = [
    "entry_time", "exit_time", "side", "entry_price", "exit_price", "barrier",
    "gross_return", "net_return", "fees_paid", "slippage_cost",
    "position_amount", "position_notional", "equity_before", "equity",
]


class VectorizedBacktester:
    """Backtester simulating one position at a time on continuous bars with the same barriers used for labels."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.0005,
        slippage: float = 0.0002,
        pt_sl: list[float] | None = None,
        max_bars: int | None = None,
        max_holding_hours: int | None = None,
    ) -> None:
        if max_bars is not None and max_holding_hours is not None:
            raise ValueError("Specify max_bars or max_holding_hours, not both.")
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.pt_mult, self.sl_mult = pt_sl or settings.PT_SL_RATIO
        self.max_bars = max_bars
        self.max_holding_hours = max_holding_hours
        self.portfolio_manager = PortfolioManager()

    @staticmethod
    def sides_from_events(
        bars: pl.DataFrame, events: pl.DataFrame, p_y1: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Map confirmed meta-label events (primary side where P(y=1) >= threshold) onto the bar timeline.

        NaN probabilities (events never scored out-of-sample) never trade.
        """
        ts = pd.to_datetime(events["timestamp"].to_pandas())
        confirmed = np.where(np.nan_to_num(p_y1, nan=-1.0) >= threshold, events["primary_signal"].to_numpy(), 0)
        bar_ts = pd.to_datetime(bars["timestamp"].to_pandas())
        return pd.Series(confirmed.astype(float), index=ts).reindex(bar_ts).fillna(0.0).to_numpy()

    @staticmethod
    def probabilities_from_events(
        bars: pl.DataFrame, events: pl.DataFrame, p_y1: np.ndarray
    ) -> np.ndarray:
        """Map event probabilities, including unscored NaNs, onto the bar timeline."""
        if len(events) != len(p_y1):
            raise ValueError("Event probabilities must match the number of events.")
        event_timestamps = pd.to_datetime(events["timestamp"].to_pandas())
        bar_timestamps = pd.to_datetime(bars["timestamp"].to_pandas())
        return (
            pd.Series(np.asarray(p_y1, dtype=float), index=event_timestamps)
            .reindex(bar_timestamps)
            .to_numpy()
        )

    def run(
        self, bars: pl.DataFrame, sides: np.ndarray, signal_probabilities: np.ndarray
    ) -> pd.DataFrame:
        """Simulate trades on continuous hourly bars and return one row per closed trade.

        Each nonzero side and its probability are known at that bar's close.
        Entries happen at that close; a new entry is allowed from the bar where the previous trade exited.
        """
        if len(sides) != len(bars) or len(signal_probabilities) != len(bars):
            raise ValueError("Sides and signal probabilities must match the bar count.")
        logger.info("Running continuous barrier backtest...")
        pdf = bars.select(["timestamp", "open", "high", "low", "close"]).to_pandas()
        open_, high, low, close = (pdf[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
        atr_values = atr(pdf["high"], pdf["low"], pdf["close"], settings.ATR_PERIOD).to_numpy()
        timestamps = pd.to_datetime(pdf["timestamp"])
        timeframe = str(bars["timeframe"][0]) if "timeframe" in bars.columns else "1h"
        max_bars = (
            self.max_bars
            if self.max_bars is not None
            else hours_to_bars(self.max_holding_hours or settings.MAX_HOLD_HOURS, timeframe)
        )
        timeframe_to_seconds(timeframe)

        equity = self.initial_capital
        peak_equity = equity
        day_start_equity = equity
        equity_day = None
        trades = []
        i = 0
        while i < len(close):
            side = int(sides[i])
            if side == 0 or not np.isfinite(atr_values[i]) or atr_values[i] <= 0:
                i += 1
                continue
            probability = float(signal_probabilities[i])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Every active signal must have a finite probability in [0, 1].")
            current_day = timestamps[i].date().isoformat()
            if current_day != equity_day:
                equity_day = current_day
                day_start_equity = equity
            amount = self.portfolio_manager.calculate_position_size(
                signal_prob=probability,
                win_rate=probability,
                win_loss_ratio=self.pt_mult / self.sl_mult,
                current_balance=equity,
                atr=float(atr_values[i]),
                price=float(close[i]),
            )
            if amount <= 0:
                i += 1
                continue
            notional = amount * close[i]
            approved, _ = self.portfolio_manager.check_risk_limits(
                current_equity=equity,
                peak_equity=peak_equity,
                daily_pnl=equity / day_start_equity - 1.0 if day_start_equity > 0 else 0.0,
                total_open_exposure=notional,
            )
            volatility_ok, _ = self.portfolio_manager.check_volatility(
                float(atr_values[i]), float(close[i])
            )
            if not approved or not volatility_ok:
                i += 1
                continue
            outcome = simulate_trade(
                open_, high, low, close, i, side, atr_values[i], self.pt_mult, self.sl_mult, max_bars
            )
            if outcome is None:
                break  # data ends before the trade resolves
            gross = (outcome.exit_price - close[i]) / close[i] * side
            entry_fill = close[i] * (1.0 + side * self.slippage)
            exit_fill = outcome.exit_price * (1.0 - side * self.slippage)
            gross_pnl = amount * (exit_fill - entry_fill) * side
            fees = amount * (abs(entry_fill) + abs(exit_fill)) * self.fee_rate
            slippage_cost = amount * (
                abs(entry_fill - close[i]) + abs(exit_fill - outcome.exit_price)
            )
            net_pnl = gross_pnl - fees
            net = net_pnl / equity
            equity_before = equity
            equity += net_pnl
            peak_equity = max(peak_equity, equity)
            trades.append(
                [
                    timestamps[i], timestamps[outcome.exit_idx], side, close[i], outcome.exit_price,
                    outcome.barrier, gross, net, fees, slippage_cost, amount, notional,
                    equity_before, equity,
                ]
            )
            i = outcome.exit_idx

        durations = [(t[1] - t[0]).total_seconds() / 3600 for t in trades]
        median_duration = float(np.median(durations)) if durations else 0.0
        logger.success(f"Backtest completed. Total trades: {len(trades)}, Median duration: {median_duration:.1f}h")
        return pd.DataFrame(trades, columns=TRADE_COLUMNS)
