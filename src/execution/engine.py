"""Asynchronous execution engine supporting Paper and Live trading modes via CCXT Async and RiskManager."""

import ccxt.async_support as ccxt_async
from config.settings import settings
from src.risk.manager import RiskManager
from src.execution.paper import PaperTrader
from src.utils.logger import logger


class ExecutionEngine:
    """Orchestrator for routing orders to Paper Trader or Live Exchange via CCXT."""

    def __init__(self, mode: str = "PAPER", exchange_id: str = "binance", load_saved_state: bool = True) -> None:
        self.mode = mode.upper()
        self.exchange_id = exchange_id
        self.paper_trader = PaperTrader(initial_balance=10000.0, load_saved_state=load_saved_state)
        
        exchange_class = getattr(ccxt_async, exchange_id)
        self.exchange = exchange_class({"enableRateLimit": True})
        if settings.EXCHANGE_SANDBOX and self.exchange.has.get("sandboxMode"):
            self.exchange.set_sandbox_mode(True)

    async def __aenter__(self) -> "ExecutionEngine":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Close underlying exchange HTTP session."""
        if self.exchange:
            await self.exchange.close()

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        current_price: float,
        risk_manager: RiskManager,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        peak_equity: float = 10000.0,
        daily_pnl_pct: float = 0.0,
    ) -> dict:
        """Validate order with RiskManager circuit breaker and route to Paper or Live exchange."""
        logger.info(f"[{self.mode}] Requesting order placement: {side.upper()} {amount} {symbol}...")

        # 1. Pre-trade Risk Validation (Circuit Breaker check)
        current_equity = self.paper_trader.balance if self.mode == "PAPER" else 10000.0
        if not risk_manager.check_circuit_breaker(current_equity, peak_equity, daily_pnl_pct):
            logger.error(f"[{self.mode}] Order REJECTED by RiskManager Circuit Breaker.")
            raise PermissionError("Order rejected: Circuit breaker active due to drawdown limits.")

        if amount <= 0:
            raise ValueError("Order amount must be greater than zero.")

        # 2. Route Execution
        try:
            if self.mode == "PAPER":
                order = self.paper_trader.execute_paper_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    current_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )
                return order
            else:
                # Live Trading via CCXT Async
                logger.info(f"[LIVE] Submitting order to {self.exchange_id.upper()} API...")
                order_type = "market"
                ccxt_side = side.lower()
                if ccxt_side in ["long", "buy"]:
                    ccxt_side = "buy"
                elif ccxt_side in ["short", "sell"]:
                    ccxt_side = "sell"

                response = await self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=ccxt_side,
                    amount=amount,
                )
                logger.success(f"[LIVE] Order successfully placed on exchange: {response.get('id')}")
                return response

        except Exception as e:
            logger.error(f"[{self.mode}] Execution failed for order {symbol}: {e}")
            raise

    async def get_account_balance(self) -> dict:
        """Retrieve account balance (Virtual or Live)."""
        if self.mode == "PAPER":
            return {"free": self.paper_trader.balance, "total": self.paper_trader.balance}
        else:
            try:
                balance = await self.exchange.fetch_balance()
                return balance.get("total", {})
            except Exception as e:
                logger.error(f"[LIVE] Failed to fetch exchange balance: {e}")
                raise
