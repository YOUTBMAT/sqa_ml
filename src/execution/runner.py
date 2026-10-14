"""Continuous asynchronous trading bot runner executing the live/paper trading loop."""

import asyncio
import signal
from datetime import datetime, timezone, timedelta
import polars as pl

from src.data.collectors.crypto import CryptoCollector
from src.features.pipeline import FeaturePipeline
from src.models.storage import ModelStorage
from src.risk.manager import RiskManager
from src.execution.engine import ExecutionEngine
from src.utils.notifications import TelegramNotifier
from src.utils.logger import logger


class BotRunner:
    """Asynchronous orchestrator for automated trading bot execution loop."""

    def __init__(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        mode: str = "PAPER",
        poll_interval_seconds: int = 60,
        buy_threshold: float = 0.75,
        sell_threshold: float = 0.25,
    ) -> None:
        self.symbols = symbols
        self.timeframe = timeframe
        self.mode = mode.upper()
        self.poll_interval = poll_interval_seconds
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.is_running = False
        self.last_processed_candle_ts: dict[str, datetime] = {}

        self.risk_manager = RiskManager()
        self.notifier = TelegramNotifier()
        self.model_storage = ModelStorage()
        self.feature_pipeline = FeaturePipeline()

        # Handle graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_running_loop().add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except Exception:
                pass  # Signal handlers might not be supported in certain OS threads

    async def start(self) -> None:
        """Start the continuous execution loop."""
        self.is_running = True
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

        # 0. Check if position is already open in paper mode to prevent order stacking
        if self.mode == "PAPER" and engine.paper_trader.has_open_position(symbol):
            logger.debug(f"Position already open for {symbol}. Skipping new entry.")
            return

        # 1. Data Collection (fetching recent candles)
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(days=5)

        async with CryptoCollector() as collector:
            df_candles = await collector.fetch_ohlcv_historical(symbol, self.timeframe, start_time, now)

        if df_candles.is_empty():
            logger.warning(f"No candles retrieved for {symbol}.")
            return

        # Candle Lock: Process only when a new 1h candle is closed
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
        atr = current_price * 0.02

        # 4. Risk Evaluation & Sizing (2-Stage Confirmation: Primary != 0 and P(y=1) >= buy_threshold)
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
                reward_risk_ratio=2.0,
                volatility=0.02,
                active_positions=active_positions,
            )

            sl, tp = self.risk_manager.calculate_dynamic_stops(current_price, side, atr, sl_atr_mult=1.0, tp_atr_mult=2.0)

            # 5. Order Execution & Routing
            try:
                order = await engine.place_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    current_price=current_price,
                    risk_manager=self.risk_manager,
                    stop_loss=sl,
                    take_profit=tp,
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
                    await self.notifier.notify_circuit_breaker(str(pe), current_drawdown=0.05)
                    self.risk_manager.circuit_breaker_notified = True
            except Exception as ex:
                logger.error(f"Failed to place order for {symbol}: {ex}")
        else:
            logger.info(f"No actionable 2-stage signal for {symbol} (Primary: {latest_signal}, P(y=1): {p_y1:.2f}).")
