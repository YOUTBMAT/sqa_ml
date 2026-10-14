"""Meta-Labeling and Triple Barrier Method labeling module (Marcos López de Prado)."""

import pandas as pd
import numpy as np
import polars as pl
from src.utils.logger import logger


class TripleBarrierLabeler:
    """Implementation of Meta-Labeling and Triple Barrier Method for financial ML."""

    @staticmethod
    def get_daily_volatility(close: pd.Series, span: int = 100) -> pd.Series:
        """Compute dynamic volatility using exponentially weighted moving standard deviation of log returns."""
        logger.info(f"Computing daily volatility with span={span}...")
        log_rets = np.log(close / close.shift(1))
        vol = log_rets.ewm(span=span).std()
        return vol.fillna(vol.mean() if not vol.dropna().empty else 0.01)

    @staticmethod
    def get_vertical_barriers(timestamps: pd.Series, num_bars: int = 24) -> pd.Series:
        """Define vertical time barrier timestamps at a fixed number of bars in the future."""
        logger.info(f"Defining vertical barriers {num_bars} bars ahead...")
        t1 = pd.Series(timestamps).shift(-num_bars).values
        t1_series = pd.Series(t1, index=timestamps)
        tz = getattr(timestamps, "tz", None)
        if tz is not None:
            t1_series = pd.to_datetime(t1_series)
            if t1_series.dt.tz is None:
                t1_series = t1_series.dt.tz_localize(tz)
            else:
                t1_series = t1_series.dt.tz_convert(tz)
        else:
            t1_series = pd.to_datetime(t1_series)
        return t1_series

    @staticmethod
    def get_primary_signal(df: pd.DataFrame) -> pd.Series:
        """Compute primary signal based on EMA200 and ADX14 regime filters.

        LONG (+1): close > ema_200 and adx_14 > 20
        SHORT (-1): close < ema_200 and adx_14 > 20
        NEUTRAL (0): adx_14 <= 20
        """
        close = df["close"]
        ema = df["ema_200"] if "ema_200" in df.columns else close.ewm(span=200).mean()
        adx = df["adx_14"] if "adx_14" in df.columns else pd.Series(25, index=df.index)

        signal = np.zeros(len(df), dtype=int)
        long_mask = (close > ema) & (adx > 20)
        short_mask = (close < ema) & (adx > 20)

        signal[long_mask] = 1
        signal[short_mask] = -1
        return pd.Series(signal, index=df.index)

    @staticmethod
    def apply_meta_triple_barrier(
        events: pd.DataFrame, close: pd.Series, primary_signals: pd.Series, pt_sl: list[float]
    ) -> pd.DataFrame:
        """Apply secondary meta-labeling based on primary signals.

        pt_sl: [pt_mult, sl_mult] e.g. [2.0, 1.0]
        y = 1 if Take-Profit reached first; y = 0 if Stop-Loss or t1 reached first.
        """
        logger.info("Applying meta-labeling triple barrier evaluation...")
        out = pd.DataFrame(index=events.index)
        
        pt_mult = pt_sl[0] if len(pt_sl) > 0 else 2.0
        sl_mult = pt_sl[1] if len(pt_sl) > 1 else 1.0

        out["t1"] = events["t1"]
        out["trgt"] = events["trgt"]

        touch_times = []
        meta_labels = []
        returns = []
        barrier_types = []

        close_index_map = {ts: idx for idx, ts in enumerate(close.index)}
        close_tz = close.index.tz

        for loc, t1_time in events["t1"].items():
            sig = primary_signals.loc[loc]
            if sig == 0 or pd.isna(t1_time):
                touch_times.append(loc)
                meta_labels.append(0)
                returns.append(0.0)
                barrier_types.append("neutral")
                continue

            t1_dt = pd.to_datetime(t1_time)
            if close_tz is not None:
                if t1_dt.tz is None:
                    t1_dt = t1_dt.tz_localize(close_tz)
                else:
                    t1_dt = t1_dt.tz_convert(close_tz)

            if loc not in close_index_map:
                touch_times.append(t1_dt)
                meta_labels.append(0)
                returns.append(0.0)
                barrier_types.append("t1")
                continue

            price_start = close.loc[loc]
            target = events.loc[loc, "trgt"]

            if sig == 1:
                tp_limit = price_start * (1.0 + pt_mult * target)
                sl_limit = price_start * (1.0 - sl_mult * target)
            else:
                tp_limit = price_start * (1.0 - pt_mult * target)
                sl_limit = price_start * (1.0 + sl_mult * target)

            future_slice = close.loc[loc:t1_dt]
            if future_slice.empty or len(future_slice) <= 1:
                touch_times.append(t1_dt)
                meta_labels.append(0)
                returns.append(0.0)
                barrier_types.append("t1")
                continue

            future_prices = future_slice.iloc[1:]

            if sig == 1:
                touched_tp = future_prices[future_prices >= tp_limit]
                touched_sl = future_prices[future_prices <= sl_limit]
            else:
                touched_tp = future_prices[future_prices <= tp_limit]
                touched_sl = future_prices[future_prices >= sl_limit]

            t_tp = touched_tp.index[0] if not touched_tp.empty else None
            t_sl = touched_sl.index[0] if not touched_sl.empty else None

            first_touch_time = t1_dt
            barrier_type = "t1"
            meta_label = 0

            if t_tp is not None and t_sl is not None:
                if t_tp < t_sl:
                    first_touch_time = t_tp
                    barrier_type = "tp"
                    meta_label = 1
                else:
                    first_touch_time = t_sl
                    barrier_type = "sl"
                    meta_label = 0
            elif t_tp is not None:
                first_touch_time = t_tp
                barrier_type = "tp"
                meta_label = 1
            elif t_sl is not None:
                first_touch_time = t_sl
                barrier_type = "sl"
                meta_label = 0

            exit_price = close.loc[first_touch_time] if first_touch_time in close.index else future_prices.iloc[-1]
            ret = (exit_price - price_start) / price_start * sig

            touch_times.append(first_touch_time)
            meta_labels.append(meta_label)
            returns.append(ret)
            barrier_types.append(barrier_type)

        out["touch_time"] = touch_times
        out["target_label"] = meta_labels
        out["target_return"] = returns
        out["barrier_touched"] = barrier_types
        out["primary_signal"] = primary_signals

        return out

    @classmethod
    def get_labels(
        cls, df: pl.DataFrame, pt_sl: list[float] = [2.0, 1.0], num_bars: int = 24
    ) -> pl.DataFrame:
        """Main interface to compute Meta-Labels for a Polars feature DataFrame."""
        logger.info("Starting Meta-Labeling pipeline...")
        if df.is_empty():
            return df

        pdf = df.to_pandas()
        if "timestamp" not in pdf.columns:
            raise ValueError("DataFrame must contain a 'timestamp' column.")

        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
        pdf = pdf.set_index("timestamp").sort_index()

        close = pdf["close"]

        vol = cls.get_daily_volatility(close, span=100)
        t1 = cls.get_vertical_barriers(pdf.index, num_bars=num_bars)
        primary_signals = cls.get_primary_signal(pdf)

        events = pd.DataFrame(index=pdf.index)
        events["t1"] = t1
        events["trgt"] = vol

        events = events.dropna(subset=["trgt"])
        events = events[events["trgt"] > 0]

        labels_pdf = cls.apply_meta_triple_barrier(events, close, primary_signals, pt_sl=pt_sl)

        pdf["primary_signal"] = labels_pdf["primary_signal"]
        pdf["target_label"] = labels_pdf["target_label"]
        pdf["target_return"] = labels_pdf["target_return"]
        pdf["barrier_touched"] = labels_pdf["barrier_touched"]
        pdf["t1"] = labels_pdf["t1"]

        pdf = pdf.reset_index()
        result_df = pl.from_pandas(pdf)
        # Filter to active primary signals only
        result_df = result_df.filter(pl.col("primary_signal") != 0)

        logger.success(f"Meta-labeling complete. Produced {len(result_df)} active signal records.")
        return result_df
