"""Continuous hourly backtesting engine sharing the labeling triple barrier (1.5x ATR TP / 1.0x ATR SL / 24h limit)."""

import numpy as np
import pandas as pd
import polars as pl
from config.settings import settings
from src.labeling.barriers import atr, simulate_trade
from src.utils.logger import logger

TRADE_COLUMNS = [
    "entry_time", "exit_time", "side", "entry_price", "exit_price", "barrier",
    "gross_return", "net_return", "fees_paid", "equity",
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
    ) -> None:
        self.initial_capital = initial_capital
        self.cost_rate = fee_rate + slippage
        self.pt_mult, self.sl_mult = pt_sl or settings.PT_SL_RATIO
        self.max_bars = max_bars or settings.MAX_HOLD_HOURS

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

    def run(self, bars: pl.DataFrame, sides: np.ndarray) -> pd.DataFrame:
        """Simulate trades on continuous hourly bars and return one row per closed trade.

        sides[i] in {+1, -1, 0} is the confirmed signal known at the close of bar i.
        Entries happen at that close; a new entry is allowed from the bar where the previous trade exited.
        """
        logger.info("Running continuous barrier backtest...")
        pdf = bars.select(["timestamp", "open", "high", "low", "close"]).to_pandas()
        open_, high, low, close = (pdf[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
        atr_values = atr(pdf["high"], pdf["low"], pdf["close"], settings.ATR_PERIOD).to_numpy()
        timestamps = pd.to_datetime(pdf["timestamp"])

        equity = self.initial_capital
        trades = []
        i = 0
        while i < len(close):
            side = int(sides[i])
            if side == 0 or not np.isfinite(atr_values[i]) or atr_values[i] <= 0:
                i += 1
                continue
            outcome = simulate_trade(
                open_, high, low, close, i, side, atr_values[i], self.pt_mult, self.sl_mult, self.max_bars
            )
            if outcome is None:
                break  # data ends before the trade resolves
            gross = (outcome.exit_price - close[i]) / close[i] * side
            net = gross - 2.0 * self.cost_rate
            fees = equity * 2.0 * self.cost_rate
            equity *= 1.0 + net
            trades.append(
                [timestamps[i], timestamps[outcome.exit_idx], side, close[i], outcome.exit_price,
                 outcome.barrier, gross, net, fees, equity]
            )
            i = outcome.exit_idx

        durations = [(t[1] - t[0]).total_seconds() / 3600 for t in trades]
        median_duration = float(np.median(durations)) if durations else 0.0
        logger.success(f"Backtest completed. Total trades: {len(trades)}, Median duration: {median_duration:.1f}h")
        return pd.DataFrame(trades, columns=TRADE_COLUMNS)
