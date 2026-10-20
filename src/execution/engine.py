"""Asynchronous execution engine supporting Paper and Live trading modes via CCXT Async and RiskManager."""

from datetime import datetime
from uuid import uuid4
import ccxt.async_support as ccxt_async
from config.settings import settings
from src.execution.fsm import OrderFSM, OrderState
from src.risk.manager import RiskManager
from src.risk.portfolio import PortfolioManager
from src.execution.paper import PaperTrader
from src.utils.logger import logger


class ExecutionEngine:
    """Orchestrator for routing orders to Paper Trader or Live Exchange via CCXT."""

    def __init__(
        self,
        mode: str = "PAPER",
        exchange_id: str = "binance",
        load_saved_state: bool = True,
        state_dir: str = "data/paper",
        is_paper: bool | None = None,
        portfolio_manager: PortfolioManager | None = None,
    ) -> None:
        self.mode = "PAPER" if is_paper is True else mode.upper()
        self.exchange_id = exchange_id
        self.portfolio_manager = portfolio_manager
        self.paper_trader = PaperTrader(
            initial_balance=10000.0,
            load_saved_state=load_saved_state,
            state_dir=state_dir,
            portfolio_manager=portfolio_manager,
        )
        self.exchange = None
        self.order_fsms: dict[str, OrderFSM] = {}
        self.orphan_orders: dict[str, OrderFSM] = {}

    def _get_exchange(self):
        """Create the CCXT exchange only when live trading actually needs it."""
        if self.exchange is None:
            exchange_class = getattr(ccxt_async, self.exchange_id)
            self.exchange = exchange_class({"enableRateLimit": True})
            if settings.EXCHANGE_SANDBOX and self.exchange.has.get("sandboxMode"):
                self.exchange.set_sandbox_mode(True)
        return self.exchange

    async def __aenter__(self) -> "ExecutionEngine":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Close underlying exchange HTTP session."""
        if self.exchange:
            await self.exchange.close()
            self.exchange = None

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        current_price: float,
        risk_manager: RiskManager,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        peak_equity: float | None = None,
        daily_pnl_pct: float | None = None,
        current_equity: float | None = None,
        entry_bar_ts: datetime | None = None,
        expires_at: datetime | None = None,
        atr_value: float | None = None,
    ) -> dict:
        """Validate order with RiskManager circuit breaker and route to Paper or Live exchange."""
        logger.info(f"[{self.mode}] Requesting order placement: {side.upper()} {amount} {symbol}...")

        # 1. Pre-trade Risk Validation (Circuit Breaker check)
        if self.mode == "PAPER":
            if current_equity is None:
                current_equity = self.paper_trader.mark_equity({symbol: current_price})
            if peak_equity is None:
                peak_equity = self.paper_trader.peak_equity
            if daily_pnl_pct is None:
                daily_pnl_pct = (
                    current_equity / self.paper_trader.day_start_equity - 1.0
                    if self.paper_trader.day_start_equity > 0
                    else 0.0
                )
        else:
            current_equity = 10000.0 if current_equity is None else current_equity
            peak_equity = 10000.0 if peak_equity is None else peak_equity
            daily_pnl_pct = 0.0 if daily_pnl_pct is None else daily_pnl_pct
        if not risk_manager.check_circuit_breaker(current_equity, peak_equity, daily_pnl_pct):
            logger.error(f"[{self.mode}] Order REJECTED by RiskManager Circuit Breaker.")
            raise PermissionError("Order rejected: Circuit breaker active due to drawdown limits.")

        if amount <= 0:
            raise ValueError("Order amount must be greater than zero.")

        if self.portfolio_manager is not None:
            positions = self.paper_trader.positions if self.mode == "PAPER" else {}
            open_notional = sum(
                float(position.get("amount", 0.0)) * float(position.get("entry_price", current_price))
                for position in positions.values()
            )
            is_opening_position = self.mode != "PAPER" or symbol not in positions
            proposed_exposure = open_notional + (
                amount * current_price if is_opening_position else 0.0
            )
            approved, reason = self.portfolio_manager.check_risk_limits(
                current_equity=current_equity,
                peak_equity=peak_equity,
                daily_pnl=daily_pnl_pct,
                total_open_exposure=proposed_exposure,
            )
            if not approved:
                raise PermissionError(f"Order rejected by PortfolioManager: {reason}")
            if is_opening_position and atr_value is not None:
                volatility_ok, volatility_reason = self.portfolio_manager.check_volatility(
                    atr_value, current_price
                )
                if not volatility_ok:
                    raise PermissionError(f"Order rejected by PortfolioManager: {volatility_reason}")

        if self.mode == "PAPER":
            order = self.paper_trader.execute_paper_order(
                symbol=symbol,
                side=side,
                amount=amount,
                current_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_bar_ts=entry_bar_ts,
                expires_at=expires_at,
            )
            self.order_fsms[order["order_id"]] = self.paper_trader.order_fsms[order["order_id"]]
            return order

        # 2. Route Execution
        order_id = str(uuid4())
        fsm = OrderFSM(order_id, symbol, side, amount, current_price)
        self.order_fsms[order_id] = fsm
        submission_attempted = False
        try:
            # Live Trading via CCXT Async
            logger.info(f"[LIVE] Submitting order to {self.exchange_id.upper()} API...")
            ccxt_side = side.lower()
            if ccxt_side in ["long", "buy"]:
                ccxt_side = "buy"
            elif ccxt_side in ["short", "sell"]:
                ccxt_side = "sell"

            exchange = self._get_exchange()
            submission_attempted = True
            response = await exchange.create_order(
                symbol=symbol,
                type="market",
                side=ccxt_side,
                amount=amount,
            )
            exchange_order_id = response.get("id") if isinstance(response, dict) else None
            if not exchange_order_id:
                fsm.transition_to(OrderState.FAILED, "Orphaned response: exchange returned no order id")
                self.orphan_orders[order_id] = fsm
                raise RuntimeError("Exchange returned an order response without an id; order is orphaned.")

            fsm.order_id = str(exchange_order_id)
            self.order_fsms[fsm.order_id] = fsm
            self.order_fsms.pop(order_id, None)
            orphaned = self._apply_exchange_order_state(fsm, response)
            if orphaned:
                self.orphan_orders[fsm.order_id] = fsm
                logger.error(f"[LIVE] Order {fsm.order_id} has an unknown exchange status and needs reconciliation.")
                raise RuntimeError(
                    f"Exchange returned unrecognized status for order {fsm.order_id}; order needs reconciliation."
                )
            response["fsm_state"] = fsm.state.value
            response["fsm_history"] = fsm.history
            logger.info(f"[LIVE] Order {fsm.order_id} state: {fsm.state.value}")
            return response

        except Exception as e:
            if fsm.is_active():
                fsm.transition_to(OrderState.FAILED, f"Execution failed: {e}")
            if submission_attempted:
                self.orphan_orders[fsm.order_id] = fsm
            if fsm.order_id not in self.order_fsms:
                self.order_fsms[fsm.order_id] = fsm
            logger.error(f"[{self.mode}] Execution failed for order {symbol}: {e}")
            raise

    @staticmethod
    def _apply_exchange_order_state(fsm: OrderFSM, response: dict) -> bool:
        """Translate CCXT status/fill fields into the local order lifecycle."""
        status = str(response.get("status") or "").lower()
        filled = float(response.get("filled") or 0.0)
        amount = float(response.get("amount") or fsm.amount)

        if status == "pending":
            return False
        if status in {"rejected"}:
            fsm.transition_to(OrderState.REJECTED, "Exchange rejected the order")
        elif status in {"canceled", "cancelled", "expired"}:
            fsm.transition_to(OrderState.EXPIRED, f"Exchange order {status}")
        elif status in {"closed", "filled"} or (amount > 0 and filled >= amount):
            fsm.transition_to(OrderState.FILLED, "Exchange order filled")
        elif status == "open" and filled > 0:
            fsm.transition_to(OrderState.PARTIALLY_FILLED, "Exchange partially filled the order")
        elif status == "open":
            fsm.transition_to(OrderState.OPEN, "Exchange accepted the order")
        else:
            fsm.transition_to(OrderState.FAILED, f"Orphaned/unknown exchange status: {status or 'missing'}")
            return True
        return False

    async def get_account_balance(self) -> dict:
        """Retrieve account balance (Virtual or Live)."""
        if self.mode == "PAPER":
            return {"free": self.paper_trader.balance, "total": self.paper_trader.balance}
        else:
            try:
                exchange = self._get_exchange()
                balance = await exchange.fetch_balance()
                return balance.get("total", {})
            except Exception as e:
                logger.error(f"[LIVE] Failed to fetch exchange balance: {e}")
                raise
