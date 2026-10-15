"""Meta-Labeling and Triple Barrier Method labeling module (Marcos López de Prado) unified with ATR and settings."""

import numpy as np
import pandas as pd
import polars as pl
from config.settings import settings
from src.labeling.barriers import atr, simulate_trade
from src.utils.logger import logger


class TripleBarrierLabeler:
    """Implementation of Meta-Labeling and Triple Barrier Method for financial ML."""

    @staticmethod
    def get_primary_signal(df: pd.DataFrame) -> pd.Series:
        """Compute primary signal based on Volatility Squeeze & Breakout.

        Squeeze: vol_ratio_6_24 < 0.75
        LONG (+1): squeeze & vwap_upper_1_dist > 0
        SHORT (-1): squeeze & vwap_lower_1_dist < 0
        NEUTRAL (0): otherwise
        """
        close = df["close"]
        vol_ratio = df["vol_ratio_6_24"] if "vol_ratio_6_24" in df.columns else pd.Series(0.5, index=df.index)

        if "vwap_upper_1_dist" in df.columns:
            long_breakout = df["vwap_upper_1_dist"] > 0
            short_breakout = df["vwap_lower_1_dist"] < 0
        else:
            vwap = df["vwap"] if "vwap" in df.columns else close
            std = close.rolling(20).std().fillna(1.0)
            long_breakout = close > vwap + std
            short_breakout = close < vwap - std

        vol_squeeze = vol_ratio < 0.75

        signal = np.zeros(len(df), dtype=int)
        signal[vol_squeeze & long_breakout] = 1
        signal[vol_squeeze & short_breakout] = -1
        return pd.Series(signal, index=df.index)

    @classmethod
    def get_labels(
        cls, df: pl.DataFrame, pt_sl: list[float] = settings.PT_SL_RATIO, num_bars: int = settings.MAX_HOLD_HOURS
    ) -> pl.DataFrame:
        """Label every primary-signal bar with the shared barrier simulator.

        Output keeps only signal bars whose outcome is resolved, with columns:
        primary_signal, target_label (1 = take profit hit first), target_return (gross barrier return),
        barrier_touched ("tp"/"sl"/"t1") and t1 (actual exit timestamp, used for purging and concurrency).
        """
        logger.info("Starting Meta-Labeling pipeline...")
        if df.is_empty():
            return df
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain a 'timestamp' column.")

        pdf = df.sort("timestamp").to_pandas()
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
        open_, high, low, close = (pdf[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
        atr_values = atr(pdf["high"], pdf["low"], pdf["close"], settings.ATR_PERIOD).to_numpy()
        signals = cls.get_primary_signal(pdf).to_numpy()
        pt_mult, sl_mult = pt_sl

        rows: list[tuple[int, int, int, float, str]] = []
        for i in np.flatnonzero(signals):
            if not np.isfinite(atr_values[i]) or atr_values[i] <= 0:
                continue
            outcome = simulate_trade(open_, high, low, close, i, int(signals[i]), atr_values[i], pt_mult, sl_mult, num_bars)
            if outcome is None:
                continue  # data ends before the barrier resolves: label unknown
            ret = (outcome.exit_price - close[i]) / close[i] * signals[i]
            rows.append((i, outcome.exit_idx, int(outcome.barrier == "tp"), ret, outcome.barrier))

        if not rows:
            logger.warning("Meta-labeling produced no resolved signal records.")
            return pl.DataFrame()

        idx, exit_idx, labels, returns, barriers = zip(*rows)
        out = pdf.iloc[list(idx)].copy()
        out["primary_signal"] = signals[list(idx)]
        out["target_label"] = labels
        out["target_return"] = returns
        out["barrier_touched"] = barriers
        out["t1"] = pdf["timestamp"].to_numpy()[list(exit_idx)]

        result_df = pl.from_pandas(out.reset_index(drop=True))
        logger.success(f"Meta-labeling complete. Produced {len(result_df)} active signal records.")
        return result_df
