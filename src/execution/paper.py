"""Virtual broker paper trading simulator with state persistence, slippage, fees, and SL/TP tracking."""

import json
from pathlib import Path
from datetime import datetime, timezone
from src.utils.logger import logger


class PaperTrader:
    """Virtual broker environment simulating real-time order execution and position management with JSON persistence."""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.0005,
        slippage_rate: float = 0.0002,
        state_dir: str = "data/paper",
        load_saved_state: bool = True,
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

        if load_saved_state:
            self._load_state()
        else:
            self._save_state()

    def _save_state(self) -> None:
        """Save paper portfolio state to JSON file."""
        try:
            state = {
                "balance": self.balance,
                "initial_balance": self.initial_balance,
                "positions": self.positions,
                "order_history": self.order_history,
            }
            # Serialize datetimes to isoformat
            def _serialize(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, default=_serialize, indent=4)
            logger.info("Paper trader state successfully persisted to disk.")
        except Exception as e:
            logger.error(f"Failed to save paper trader state: {e}")

    def _load_state(self) -> None:
        """Load paper portfolio state from JSON file if it exists."""
        if not self.state_file.exists():
            logger.info("No existing paper portfolio state found. Initializing fresh state.")
            self._save_state()
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            self.balance = float(state.get("balance", self.initial_balance))
            self.initial_balance = float(state.get("initial_balance", self.initial_balance))
            
            # Restore positions and order history with datetime parsing
            raw_positions = state.get("positions", {})
            for sym, pos in raw_positions.items():
                if isinstance(pos.get("timestamp"), str):
                    pos["timestamp"] = datetime.fromisoformat(pos["timestamp"])
            self.positions = raw_positions

            raw_orders = state.get("order_history", [])
            for ord_rec in raw_orders:
                if isinstance(ord_rec.get("timestamp"), str):
                    ord_rec["timestamp"] = datetime.fromisoformat(ord_rec["timestamp"])
            self.order_history = raw_orders

            logger.success(f"Successfully loaded paper portfolio state from {self.state_file} (Balance: ${self.balance:,.2f})")
        except Exception as e:
            logger.error(f"Failed to load paper trader state: {e}. Initializing fresh state.")
            self._save_state()

    def has_open_position(self, symbol: str) -> bool:
        """Check if there is an active open position for the given symbol."""
        return symbol in self.positions and self.positions[symbol]["amount"] > 0

    def execute_paper_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        current_price: float,
        order_type: str = "market",
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict:
        """Execute virtual order with slippage, fee deductions, and state persistence supporting LONG and SHORT."""
        logger.info(f"[PAPER] Executing {side.upper()} order for {amount} {symbol} at ~{current_price}...")
        
        if amount <= 0 or current_price <= 0:
            raise ValueError("Invalid order amount or price.")

        side_lower = side.lower()
        is_buy = side_lower in ["buy", "long"]

        # Apply slippage
        if is_buy:
            execution_price = current_price * (1.0 + self.slippage_rate)
        else:
            execution_price = current_price * (1.0 - self.slippage_rate)

        cost = amount * execution_price
        fee = cost * self.fee_rate

        position = self.positions.get(symbol)
        has_pos = position is not None and position["amount"] > 0

        if not has_pos:
            # Opening new position
            if is_buy:
                # Open LONG
                total_required = cost + fee
                if self.balance < total_required:
                    raise ValueError(f"Insufficient paper balance: required {total_required:.2f}, available {self.balance:.2f}")
                self.balance -= total_required
                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": "long",
                    "amount": amount,
                    "entry_price": execution_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "timestamp": datetime.now(timezone.utc),
                }
            else:
                # Open SHORT
                self.balance += cost - fee
                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": "short",
                    "amount": amount,
                    "entry_price": execution_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "timestamp": datetime.now(timezone.utc),
                }
        else:
            # Closing existing position
            pos_side = position["side"]
            if (pos_side == "long" and not is_buy) or (pos_side == "short" and is_buy):
                entry_price = position["entry_price"]
                pos_amount = position["amount"]
                if pos_side == "long":
                    pnl = (execution_price - entry_price) * pos_amount
                    self.balance += (entry_price * pos_amount) + pnl - fee
                else: # short
                    pnl = (entry_price - execution_price) * pos_amount
                    self.balance -= (cost + fee)
                del self.positions[symbol]
            else:
                raise ValueError(f"Conflicting order side {side} for existing {pos_side} position on {symbol}.")

        order_record = {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "execution_price": execution_price,
            "fee": fee,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "timestamp": datetime.now(timezone.utc),
            "status": "filled",
        }
        self.order_history.append(order_record)
        
        # Persist state after order fill
        self._save_state()

        logger.success(f"[PAPER] Order filled: {order_record}")
        return order_record

    def update_positions(self, current_prices: dict[str, float]) -> list[dict]:
        """Check open positions against current market prices for stop-loss and take-profit hits."""
        triggered_events = []
        for symbol, pos in list(self.positions.items()):
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            sl = pos.get("stop_loss")
            tp = pos.get("take_profit")
            side = pos["side"]

            hit = False
            exit_reason = None

            if side == "long":
                if sl is not None and price <= sl:
                    hit = True
                    exit_reason = "stop_loss"
                elif tp is not None and price >= tp:
                    hit = True
                    exit_reason = "take_profit"
            else: # short
                if sl is not None and price >= sl:
                    hit = True
                    exit_reason = "stop_loss"
                elif tp is not None and price <= tp:
                    hit = True
                    exit_reason = "take_profit"

            if hit:
                logger.warning(f"[PAPER] Position {symbol} hit {exit_reason} at price {price}.")
                close_side = "sell" if side == "long" else "buy"
                fill = self.execute_paper_order(symbol, close_side, pos["amount"], price)
                fill["exit_reason"] = exit_reason
                triggered_events.append(fill)

        return triggered_events

    def get_account_summary(self) -> dict:
        """Return current virtual account balance and open positions."""
        return {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "open_positions": self.positions,
            "total_orders": len(self.order_history),
        }
