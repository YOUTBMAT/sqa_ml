"""Continuous asynchronous trading bot runner executing the live/paper trading loop."""

import asyncio
import signal
from datetime import datetime, timezone, timedelta
import polars as pl

from config.settings import settings

from src.data.collectors.crypto import CryptoCollector
from src.data.dataset import DatasetBuilder
from src.features.pipeline import FeaturePipeline
from src.labeling.barriers import atr
from src.models.storage import ModelStorage
from src.risk.manager import RiskManager
from src.execution.engine import ExecutionEngine
from src.utils.notifications import TelegramNotifier
from src.utils.logger import logger
from src.validation.drift import DataDriftDetector

CRITICAL_DRIFT_FEATURES = (
    "vol_parkinson",
    "frac_diff_close",
    "vwap_dist",
    "rsi_adaptive",
)


class BotRunner:
    """Asynchronous orchestrator for automated trading bot execution loop."""

    def __init__(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        mode: str = "PAPER",
        poll_interval_seconds: int = 60,
        buy_threshold: float = settings.PROB_THRESHOLD,
    ) -> None:
        self.symbols = symbols
        self.timeframe = timeframe
        self.mode = mode.upper()
        if self.mode == "LIVE":
            raise RuntimeError("O modo LIVE esta temporariamente desativado ate a conclusao total da auditoria e re-treino do sistema.")
        self.poll_interval = poll_interval_seconds
        self.buy_threshold = buy_threshold
        self.is_running = False
        self.last_processed_candle_ts: dict[str, datetime] = {}

        self.risk_manager = RiskManager()
        self.notifier = TelegramNotifier()
        self.model_storage = ModelStorage()
        self.feature_pipeline = FeaturePipeline()
        self.dataset_builder = DatasetBuilder()
        self.training_data_cache: dict[str, pl.DataFrame] = {}
        self.protective_mode: dict[str, bool] = {}

    async def start(self) -> None:
        """Start the continuous execution loop."""
        self.is_running = True
        # Graceful shutdown on SIGINT/SIGTERM (systemd stop); needs the running loop, so it is installed here
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except NotImplementedError:
                logger.warning(f"Signal handler for {sig.name} not supported on this platform.")
        logger.info(f"=== BOT RUNNER STARTED [{self.mode}] for symbols: {self.symbols} ===")
        await self.notifier.send_message(f"<b>🤖 SQA-ML BOT STARTED</b>\nMode: <b>{self.mode}</b>\nSymbols: {', '.join(self.symbols)}")

        async with ExecutionEngine(mode=self.mode) as engine:
            while self.is_running:
                for symbol in self.symbols:
                    try:
                        await self._process_symbol(symbol, engine)
                    except Exception as e:
                        logger.error(f"Error processing symbol {symbol}: {e}")

                # Sleep until next poll interval
                await asyncio.sleep(self.poll_interval)

    async def stop(self) -> None:
        """Gracefully stop the bot runner."""
        logger.info("Stopping BotRunner gracefully...")
        self.is_running = False
        await self.notifier.send_message("<b>🛑 SQA-ML BOT STOPPED GRACEFULLY</b>")

    async def _process_symbol(self, symbol: str, engine: ExecutionEngine) -> None:
        """Process a single symbol iteration (Collect -> Feature -> Predict -> Risk -> Execute -> Notify)."""
        logger.info(f"Processing iteration for {symbol}...")

        # 1. Data Collection (fetching recent candles - ~2000 candles for FFD & EMA-200 convergence)
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(days=90)

        async with CryptoCollector() as collector:
            df_candles = await collector.fetch_ohlcv_historical(symbol, self.timeframe, start_time, now)

        if df_candles.is_empty():
            logger.warning(f"No candles retrieved for {symbol}.")
            return

        # Candle Lock & Filtering forming candle
        current_hour_ts = now.replace(minute=0, second=0, microsecond=0)
        df_candles = df_candles.filter(pl.col("timestamp") < current_hour_ts)
        if df_candles.is_empty():
            logger.warning(f"No closed candles available for {symbol} after candle lock filter.")
            return

        # 0. Resolve open paper positions against closed bars with the same barrier rules as labels/backtest
        if self.mode == "PAPER":
            bars = df_candles.select(["timestamp", "open", "high", "low", "close"]).to_dicts()
            engine.paper_trader.update_positions({symbol: bars})
            if engine.paper_trader.has_open_position(symbol):
                logger.debug(f"Position already open for {symbol}. Skipping new entry.")
                return

        # Check if new 1h candle closed
        latest_candle_ts = df_candles["timestamp"][-1]
        if symbol in self.last_processed_candle_ts and latest_candle_ts <= self.last_processed_candle_ts[symbol]:
            logger.debug(f"No new 1h candle closed for {symbol} (Latest: {latest_candle_ts}). Skipping iteration.")
            return

        self.last_processed_candle_ts[symbol] = latest_candle_ts
        logger.info(f"New 1h candle closed for {symbol} at {latest_candle_ts}. Proceeding with pipeline...")

        # 2. Feature Engineering
        df_features = self.feature_pipeline.run(df_candles)
        if df_features.is_empty():
            logger.warning(f"Feature pipeline returned empty DataFrame for {symbol}.")
            return

        # Compare a rolling live window against the feature distribution used for training.
        reference = self.training_data_cache.get(symbol)
        if reference is None:
            reference = self.dataset_builder.load_dataset(symbol, self.timeframe)
            self.training_data_cache[symbol] = reference

        missing_live_features = [
            feature for feature in CRITICAL_DRIFT_FEATURES if feature not in df_features.columns
        ]
        if reference.is_empty() or missing_live_features:
            self.protective_mode[symbol] = True
            logger.warning(
                f"Protective mode enabled for {symbol}: training drift reference unavailable "
                f"or critical live features missing ({missing_live_features})."
            )
            return

        try:
            psi_values, has_severe_drift = DataDriftDetector.check_feature_drift(
                reference,
                df_features.tail(100).select(CRITICAL_DRIFT_FEATURES),
                CRITICAL_DRIFT_FEATURES,
            )
        except ValueError as exc:
            self.protective_mode[symbol] = True
            logger.warning(f"Protective mode enabled for {symbol}: drift check failed: {exc}")
            return

        if has_severe_drift:
            self.protective_mode[symbol] = True
            severe_features = {
                feature: value
                for feature, value in psi_values.items()
                if value > DataDriftDetector.SEVERE_DRIFT_THRESHOLD
            }
            logger.warning(
                f"Protective mode enabled for {symbol}: severe feature drift detected "
                f"(PSI > {DataDriftDetector.SEVERE_DRIFT_THRESHOLD}: {severe_features}). "
                "New entries are paused."
            )
            return
        if self.protective_mode.pop(symbol, False):
            logger.info(f"Protective mode cleared for {symbol}: feature distributions normalized.")

        # 3. Model Prediction & 2-Stage Primary Signal Filtering
        try:
            model, meta = self.model_storage.load_model(symbol, self.timeframe)
            feature_cols = meta.get("feature_cols", [])
        except Exception:
            logger.warning(f"No trained model found for {symbol} ({self.timeframe}). Skipping prediction.")
            return

        # Primary Signal (EMA200 & ADX14)
        pdf_features = df_features.to_pandas()
        from src.labeling.triple_barrier import TripleBarrierLabeler
        primary_signals = TripleBarrierLabeler.get_primary_signal(pdf_features)
        latest_signal = primary_signals.iloc[-1]

        probs = model.predict_proba(df_features, feature_cols=feature_cols)
        latest_prob = probs[-1]  # Probabilities for [Class 0 (Fail), Class 1 (Success)]
        p_y1 = latest_prob[1]

        current_price = float(df_features["close"][-1])
        pdf_candles = df_candles.select(["high", "low", "close"]).to_pandas()
        atr_value = float(atr(pdf_candles["high"], pdf_candles["low"], pdf_candles["close"], settings.ATR_PERIOD).iloc[-1])
        pt_mult, sl_mult = settings.PT_SL_RATIO

        # 4. Risk Evaluation & Sizing (2-Stage Confirmation: Primary != 0 and P(y=1) >= buy_threshold)
        risk_state: dict = {}
        if self.mode == "PAPER":
            paper = engine.paper_trader
            # ponytail: marks only this symbol's price; other open symbols are valued at entry until their own iteration
            equity = paper.mark_equity({symbol: current_price})
            account_balance = equity
            risk_state = {
                "current_equity": equity,
                "peak_equity": paper.peak_equity,
                "daily_pnl_pct": equity / paper.day_start_equity - 1.0 if paper.day_start_equity > 0 else 0.0,
            }
        else:
            balance_info = await engine.get_account_balance()
            account_balance = float(balance_info.get("free", 10000.0) if isinstance(balance_info, dict) else 10000.0)

        side = None
        if latest_signal == 1 and p_y1 >= self.buy_threshold:
            side = "long"
        elif latest_signal == -1 and p_y1 >= self.buy_threshold:
            side = "short"

        if side:
            active_positions = engine.paper_trader.positions if self.mode == "PAPER" else {}
            amount = self.risk_manager.calculate_position_size(
                account_balance=account_balance,
                current_price=current_price,
                win_probability=float(p_y1),
                reward_risk_ratio=pt_mult / sl_mult,
                volatility=atr_value / current_price,
                active_positions=active_positions,
            )

            sl, tp = self.risk_manager.calculate_dynamic_stops(
                current_price, side, atr_value, sl_atr_mult=sl_mult, tp_atr_mult=pt_mult
            )

            # 5. Order Execution & Routing
            try:
                order = await engine.place_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    current_price=current_price,
                    risk_manager=self.risk_manager,
                    **risk_state,
                    stop_loss=sl,
                    take_profit=tp,
                    entry_bar_ts=latest_candle_ts,
                    expires_at=latest_candle_ts + timedelta(hours=settings.MAX_HOLD_HOURS),
                )

                # 6. Notifications
                await self.notifier.notify_order_execution(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    price=current_price,
                    mode=self.mode,
                    sl=sl,
                    tp=tp,
                )
            except PermissionError as pe:
                logger.warning(f"Order blocked by risk management for {symbol}: {pe}")
                if not self.risk_manager.circuit_breaker_notified:
                    peak = risk_state.get("peak_equity", 0.0)
                    drawdown = 1.0 - risk_state["current_equity"] / peak if peak > 0 else 0.0
                    await self.notifier.notify_circuit_breaker(str(pe), current_drawdown=drawdown)
                    self.risk_manager.circuit_breaker_notified = True
            except Exception as ex:
                logger.error(f"Failed to place order for {symbol}: {ex}")
        else:
            logger.info(f"No actionable 2-stage signal for {symbol} (Primary: {latest_signal}, P(y=1): {p_y1:.2f}).")
