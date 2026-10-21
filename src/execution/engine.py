"""PAPER-only order routing with portfolio-level risk checks."""

from datetime import datetime
import math
from src.execution.fsm import OrderFSM
from src.risk.portfolio import PortfolioManager
from src.execution.paper import PaperTrader
from src.utils.logger import logger


class ExecutionEngine:
    """Route approved orders to the virtual paper broker."""

    def __init__(
        self,
        mode: str = "PAPER",
        load_saved_state: bool = True,
        state_dir: str = "data/paper",
        is_paper: bool | None = None,
        portfolio_manager: PortfolioManager | None = None,
    ) -> None:
        self.mode = "PAPER" if is_paper is True else mode.upper()
        if self.mode != "PAPER":
            raise RuntimeError("ExecutionEngine only permits PAPER mode; LIVE execution is disabled.")
        self.portfolio_manager = portfolio_manager or PortfolioManager()
        self.paper_trader = PaperTrader(
            initial_balance=10000.0,
            load_saved_state=load_saved_state,
            state_dir=state_dir,
            portfolio_manager=self.portfolio_manager,
        )
        self.exchange = None
        self.order_fsms: dict[str, OrderFSM] = {}

    async def __aenter__(self) -> "ExecutionEngine":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the engine; PAPER mode owns no network session."""

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        current_price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        peak_equity: float | None = None,
        daily_pnl_pct: float | None = None,
        current_equity: float | None = None,
        entry_bar_ts: datetime | None = None,
        expires_at: datetime | None = None,
        atr_value: float | None = None,
    ) -> dict:
        """Validate portfolio limits and route a PAPER order."""
        if self.mode != "PAPER":
            raise RuntimeError("Order submission is disabled outside PAPER mode.")
        if (
            not math.isfinite(amount)
            or not math.isfinite(current_price)
            or amount <= 0
            or current_price <= 0
        ):
            raise ValueError("Order amount and price must be positive.")
        logger.info(f"[{self.mode}] Requesting order placement: {side.upper()} {amount} {symbol}...")

        current_equity = (
            self.paper_trader.mark_equity({symbol: current_price})
            if current_equity is None
            else current_equity
        )
        peak_equity = self.paper_trader.peak_equity if peak_equity is None else peak_equity
        if daily_pnl_pct is None:
            daily_pnl_pct = (
                current_equity / self.paper_trader.day_start_equity - 1.0
                if self.paper_trader.day_start_equity > 0
                else 0.0
            )

        positions = self.paper_trader.positions
        position = positions.get(symbol)
        side_lower = side.lower()
        is_closing_position = position is not None and (
            (position["side"] == "long" and side_lower in {"sell", "short"})
            or (position["side"] == "short" and side_lower in {"buy", "long"})
        )
        if not is_closing_position:
            open_notional = sum(
                float(position.get("amount", 0.0)) * float(position.get("entry_price", current_price))
                for position in positions.values()
            )
            approved, reason = self.portfolio_manager.check_risk_limits(
                current_equity=current_equity,
                peak_equity=peak_equity,
                daily_pnl=daily_pnl_pct,
                total_open_exposure=open_notional + amount * current_price,
            )
            if not approved:
                raise PermissionError(f"Order rejected by PortfolioManager: {reason}")
            if atr_value is not None:
                volatility_ok, volatility_reason = self.portfolio_manager.check_volatility(
                    atr_value, current_price
                )
                if not volatility_ok:
                    raise PermissionError(f"Order rejected by PortfolioManager: {volatility_reason}")

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

    async def get_account_balance(self) -> dict:
        """Return the virtual PAPER account balance."""
        if self.mode != "PAPER":
            raise RuntimeError("Account access is disabled outside PAPER mode.")
        return {"free": self.paper_trader.balance, "total": self.paper_trader.balance}
