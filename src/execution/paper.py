"""Virtual broker paper trading simulator with state persistence, slippage, fees, and SL/TP tracking."""

import json
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from config.settings import settings
from src.execution.fsm import OrderFSM, OrderState
from src.labeling.barriers import check_bar
from src.risk.portfolio import PortfolioManager
from src.utils.logger import logger

DATETIME_FIELDS = ("timestamp", "entry_bar_ts", "expires_at")


class PaperTrader:
    """Virtual broker environment simulating real-time order execution and position management with JSON persistence."""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.0005,
        slippage_rate: float = 0.0002,
        state_dir: str = "data/paper",
        load_saved_state: bool = True,
        portfolio_manager: PortfolioManager | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "paper_portfolio.json"

        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.balance = initial_balance
        self.positions: dict[str, dict] = {}
        self.order_history: list[dict] = []
        self.order_fsms: dict[str, OrderFSM] = {}
        self.portfolio_manager = portfolio_manager
        self.peak_equity = initial_balance
        self.day_start_equity = initial_balance
        self.equity_day: str | None = None

        if load_saved_state:
            self._load_state()
        else:
            self._save_state()

    def _save_state(self) -> None:
        """Save paper portfolio state to JSON file atomically using tmp file, fsync, and backup."""
        import os
        import shutil
        try:
            state = {
                "balance": self.balance,
                "initial_balance": self.initial_balance,
                "positions": self.positions,
                "order_history": self.order_history,
                "peak_equity": self.peak_equity,
                "day_start_equity": self.day_start_equity,
                "equity_day": self.equity_day,
            }
            # Serialize datetimes to isoformat
            def _serialize(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")

            temp_file = self.state_file.with_suffix(".json.tmp")
            backup_file = self.state_file.with_suffix(".json.bak")

            # 1. Write to temporary file
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, default=_serialize, indent=4)
                f.flush()
                os.fsync(f.fileno())

            # 2. Create backup of current state file if it exists
            if self.state_file.exists():
                try:
                    shutil.copy(self.state_file, backup_file)
                except Exception as ex:
                    logger.warning(f"Failed to create backup file: {ex}")

            # 3. Atomically replace main file with temporary file
            os.replace(temp_file, self.state_file)
            logger.info("Paper trader state successfully persisted to disk atomically.")
        except Exception as e:
            logger.error(f"Failed to save paper trader state: {e}")
            raise

    def _load_state(self) -> None:
        """Load paper portfolio state from JSON file with fallback to backup. Never overwrite with zero balance if corrupted."""
        import shutil
        backup_file = self.state_file.with_suffix(".json.bak")

        if not self.state_file.exists() and not backup_file.exists():
            logger.info("No existing paper portfolio state found. Initializing fresh state.")
            self._save_state()
            return

        loaded_data = None
        source_loaded = None

        # Try loading primary state file
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                source_loaded = "primary"
            except Exception as e:
                logger.error(f"Primary state file corrupted ({e}). Attempting fallback to backup...")

        # If primary failed or doesn't exist, try backup
        if loaded_data is None and backup_file.exists():
            try:
                with open(backup_file, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                source_loaded = "backup"
                logger.warning("Successfully recovered paper portfolio state from backup file.")
                shutil.copy(backup_file, self.state_file)
            except Exception as e:
                logger.error(f"Backup state file also corrupted ({e}).")

        if loaded_data is None:
            raise RuntimeError("Paper portfolio state file and backup are both missing or corrupted. Refusing to initialize with zero balance.")

        try:
            self.balance = float(loaded_data.get("balance", self.initial_balance))
            self.initial_balance = float(loaded_data.get("initial_balance", self.initial_balance))
            self.peak_equity = float(loaded_data.get("peak_equity", max(self.balance, self.initial_balance)))
            self.day_start_equity = float(loaded_data.get("day_start_equity", self.balance))
            self.equity_day = loaded_data.get("equity_day")
            
            # Restore positions and order history with datetime parsing
            raw_positions = loaded_data.get("positions", {})
            for sym, pos in raw_positions.items():
                for field in DATETIME_FIELDS:
                    if isinstance(pos.get(field), str):
                        pos[field] = datetime.fromisoformat(pos[field])
            self.positions = raw_positions

            raw_orders = loaded_data.get("order_history", [])
            for ord_rec in raw_orders:
                if isinstance(ord_rec.get("timestamp"), str):
                    ord_rec["timestamp"] = datetime.fromisoformat(ord_rec["timestamp"])
                serialized_fsm = ord_rec.get("fsm")
                if isinstance(serialized_fsm, dict):
                    fsm = OrderFSM.from_dict(serialized_fsm)
                    self.order_fsms[fsm.order_id] = fsm
            self.order_history = raw_orders

            logger.success(f"Successfully loaded paper portfolio state from {source_loaded} (Balance: ${self.balance:,.2f})")
        except Exception as e:
            logger.error(f"Failed to parse paper portfolio state data: {e}")
            raise RuntimeError(f"Failed to parse paper portfolio state data: {e}")

    def mark_equity(self, prices: dict[str, float]) -> float:
        """Mark-to-market equity (cash plus long value minus short liability) and update peak and daily anchors.

        Positions without a price in `prices` are valued at their entry price.
        """
        equity = self.balance
        for symbol, pos in self.positions.items():
            sign = 1.0 if pos["side"] == "long" else -1.0
            equity += sign * pos["amount"] * prices.get(symbol, pos["entry_price"])

        today = datetime.now(timezone.utc).date().isoformat()
        if today != self.equity_day or equity > self.peak_equity:
            if today != self.equity_day:
                self.equity_day = today
                self.day_start_equity = equity
            self.peak_equity = max(self.peak_equity, equity)
            self._save_state()
        return equity

    def has_open_position(self, symbol: str) -> bool:
        """Check if there is an active open position for the given symbol."""
        return symbol in self.positions and self.positions[symbol]["amount"] > 0

    def _record_rejected_order(self, fsm: OrderFSM, reason: str) -> None:
        """Persist an order rejected before it could be filled."""
        self.order_history.append(
            {
                "order_id": fsm.order_id,
                "symbol": fsm.symbol,
                "side": fsm.side,
                "amount": fsm.amount,
                "execution_price": fsm.price,
                "timestamp": datetime.now(timezone.utc),
                "status": fsm.state.value.lower(),
                "reason": reason,
                "fsm": fsm.to_dict(),
            }
        )
        self._save_state()

    def execute_paper_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        current_price: float,
        order_type: str = "market",
        stop_loss: float | None = None,
        take_profit: float | None = None,
        entry_bar_ts: datetime | None = None,
        expires_at: datetime | None = None,
        exit_reason: str | None = None,
    ) -> dict:
        """Execute virtual order with slippage, fee deductions, and state persistence supporting LONG and SHORT."""
        logger.info(f"[PAPER] Executing {side.upper()} order for {amount} {symbol} at ~{current_price}...")
        order_id = str(uuid4())
        fsm = OrderFSM(order_id, symbol, side, amount, current_price)
        self.order_fsms[order_id] = fsm
        if (
            not math.isfinite(amount)
            or not math.isfinite(current_price)
            or amount <= 0
            or current_price <= 0
        ):
            reason = "Invalid order amount or price"
            fsm.transition_to(OrderState.REJECTED, reason)
            self._record_rejected_order(fsm, reason)
            raise ValueError("Invalid order amount or price.")

        side_lower = side.lower()
        if side_lower not in {"buy", "long", "sell", "short"}:
            reason = f"Unsupported side: {side}"
            fsm.transition_to(OrderState.REJECTED, reason)
            self._record_rejected_order(fsm, reason)
            raise ValueError(f"Invalid order side: {side}")
        is_buy = side_lower in ["buy", "long"]

        # Apply slippage
        if is_buy:
            execution_price = current_price * (1.0 + self.slippage_rate)
        else:
            execution_price = current_price * (1.0 - self.slippage_rate)
        fsm.price = execution_price

        cost = amount * execution_price
        fee = cost * self.fee_rate

        position = self.positions.get(symbol)
        has_pos = position is not None and position["amount"] > 0

        try:
            if not has_pos:
                # Opening new position
                if self.portfolio_manager is not None:
                    equity = self.mark_equity({})
                    open_notional = sum(
                        float(pos["amount"]) * float(pos["entry_price"])
                        for pos in self.positions.values()
                    )
                    approved, reason = self.portfolio_manager.check_risk_limits(
                        current_equity=equity,
                        peak_equity=self.peak_equity,
                        daily_pnl=(
                            equity / self.day_start_equity - 1.0
                            if self.day_start_equity > 0
                            else 0.0
                        ),
                        total_open_exposure=open_notional + cost,
                    )
                    if not approved:
                        fsm.transition_to(OrderState.REJECTED, reason)
                        raise PermissionError(f"Paper order rejected: {reason}")
                if is_buy:
                    total_required = cost + fee
                    if self.balance < total_required:
                        reason = "Insufficient paper balance"
                        fsm.transition_to(OrderState.REJECTED, reason)
                        raise ValueError(
                            f"Insufficient paper balance: required {total_required:.2f}, available {self.balance:.2f}"
                        )
                    self.balance -= total_required
                else:
                    self.balance += cost - fee
                fsm.transition_to(OrderState.OPEN, "Paper market order accepted")
                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": "long" if is_buy else "short",
                    "amount": amount,
                    "entry_price": execution_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "timestamp": datetime.now(timezone.utc),
                    "entry_bar_ts": entry_bar_ts,
                    "expires_at": expires_at,
                    "entry_order_id": order_id,
                }
            else:
                pos_side = position["side"]
                if not ((pos_side == "long" and not is_buy) or (pos_side == "short" and is_buy)):
                    reason = f"Conflicting order for open {pos_side} position"
                    fsm.transition_to(OrderState.REJECTED, reason)
                    raise ValueError(f"Conflicting order side {side} for existing {pos_side} position on {symbol}.")
                entry_price = position["entry_price"]
                pos_amount = float(position["amount"])
                if amount > pos_amount:
                    reason = "Close order amount exceeds the open position."
                    fsm.transition_to(OrderState.REJECTED, reason)
                    raise ValueError(reason)
                amount = min(amount, pos_amount)
                cost = amount * execution_price
                fee = cost * self.fee_rate
                fsm.amount = amount
                fsm.transition_to(OrderState.OPEN, "Paper market close order accepted")
                if pos_side == "long":
                    pnl = (execution_price - entry_price) * amount
                    self.balance += (entry_price * amount) + pnl - fee
                else:  # short
                    pnl = (entry_price - execution_price) * amount
                    self.balance += pnl - fee - (entry_price * amount)
                remaining_amount = pos_amount - amount
                if remaining_amount <= max(1e-12, pos_amount * 1e-12):
                    del self.positions[symbol]
                else:
                    position["amount"] = remaining_amount

            fsm.transition_to(OrderState.FILLED, exit_reason or "Paper market order filled")
        except Exception as exc:
            if fsm.is_active():
                fsm.transition_to(OrderState.FAILED, "Paper order execution failed")
            self._record_rejected_order(fsm, str(exc))
            raise

        order_record = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "execution_price": execution_price,
            "fee": fee,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "timestamp": datetime.now(timezone.utc),
            "status": "filled",
            "exit_reason": exit_reason,
            "fsm": fsm.to_dict(),
        }
        self.order_history.append(order_record)
        
        # Persist state after order fill
        self._save_state()

        logger.success(f"[PAPER] Order filled: {order_record}")
        return order_record

    def update_positions(self, bars_by_symbol: dict[str, list[dict]]) -> list[dict]:
        """Check open positions against closed OHLC bars (oldest first) for SL/TP and the time barrier.

        Same rules as labeling and backtest: only bars after the entry bar count, the stop wins on
        ambiguous bars, gaps fill at the open, and the time barrier closes at the close of the bar
        starting at expires_at.
        """
        triggered_events = []
        for symbol, pos in list(self.positions.items()):
            exit_hit = self._find_exit(pos, bars_by_symbol.get(symbol, []))
            if exit_hit is None:
                continue
            exit_reason, price = exit_hit
            logger.warning(f"[PAPER] Position {symbol} hit {exit_reason} at price {price}.")
            close_side = "sell" if pos["side"] == "long" else "buy"
            fill = self.execute_paper_order(
                symbol, close_side, pos["amount"], price, exit_reason=exit_reason
            )
            triggered_events.append(fill)

        return triggered_events

    @staticmethod
    def _find_exit(pos: dict, bars: list[dict]) -> tuple[str, float] | None:
        """Return (exit_reason, exit_price) for the first bar that resolves the position, if any."""
        side = 1 if pos["side"] == "long" else -1
        sl, tp = pos.get("stop_loss"), pos.get("take_profit")
        # Positions saved before entry_bar_ts existed: the signal bar is the hour before the fill.
        entry_bar_ts = pos.get("entry_bar_ts") or (
            pos["timestamp"].replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        )
        expires_at = pos.get("expires_at") or entry_bar_ts + timedelta(hours=settings.MAX_HOLD_HOURS)
        reasons = {"sl": "stop_loss", "tp": "take_profit"}

        for bar in bars:
            if bar["timestamp"] <= entry_bar_ts:
                continue
            if sl is not None and tp is not None:
                hit = check_bar(side, tp, sl, bar["open"], bar["high"], bar["low"])
                if hit is not None:
                    return reasons[hit[0]], hit[1]
            if expires_at is not None and bar["timestamp"] >= expires_at:
                duration_hours = (expires_at - entry_bar_ts).total_seconds() / 3600
                return f"TIME_BARRIER_{duration_hours:g}H", bar["close"]
        return None

    def get_account_summary(self) -> dict:
        """Return current virtual account balance and open positions."""
        return {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "open_positions": self.positions,
            "total_orders": len(self.order_history),
        }
