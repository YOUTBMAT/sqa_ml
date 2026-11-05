"""Asynchronous Telegram notification module for trading alerts and summaries."""

import asyncio
import httpx
from config.settings import settings
from src.utils.logger import logger


class TelegramNotifier:
    """Notifier for sending real-time trading alerts, executions, and summaries via Telegram."""

    def __init__(self, token: str = settings.TELEGRAM_BOT_TOKEN, chat_id: str = settings.TELEGRAM_CHAT_ID) -> None:
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)
        if not self.enabled:
            logger.warning("TelegramNotifier is disabled (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID).")

    async def send_message(self, message: str) -> bool:
        """Send formatted HTML message asynchronously to Telegram chat."""
        if not self.enabled:
            logger.info(f"[Telegram Mock] {message}")
            return True

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            for attempt in range(3):
                try:
                    response = await client.post(url, json=payload)
                except httpx.RequestError as exc:
                    if attempt == 2:
                        logger.error(f"Telegram request failed after retries: {exc}")
                        return False
                    await asyncio.sleep(2**attempt)
                    continue
                if response.status_code == 200:
                    logger.success("Telegram notification sent successfully.")
                    return True
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "1")
                    try:
                        delay = min(60.0, max(1.0, float(retry_after)))
                    except ValueError:
                        delay = 1.0
                    if attempt < 2:
                        await asyncio.sleep(delay)
                        continue
                logger.error(
                    f"Failed to send Telegram notification "
                    f"(status={response.status_code}): {response.text}"
                )
                return False
        return False

    async def notify_order_execution(
        self, symbol: str, side: str, amount: float, price: float, mode: str, sl: float, tp: float
    ) -> bool:
        """Send notification for order execution."""
        msg = (
            f"<b>🚀 SQA-ML ORDER EXECUTED [{mode}]</b>\n\n"
            f"• <b>Symbol:</b> {symbol}\n"
            f"• <b>Side:</b> {side.upper()}\n"
            f"• <b>Amount:</b> {amount}\n"
            f"• <b>Execution Price:</b> ${price:,.2f}\n"
            f"• <b>Stop-Loss:</b> ${sl:,.2f}\n"
            f"• <b>Take-Profit:</b> ${tp:,.2f}"
        )
        return await self.send_message(msg)

    async def notify_circuit_breaker(self, reason: str, current_drawdown: float) -> bool:
        """Send notification when circuit breaker triggers."""
        msg = (
            f"<b>🚨 SQA-ML CIRCUIT BREAKER TRIGGERED 🚨</b>\n\n"
            f"• <b>Reason:</b> {reason}\n"
            f"• <b>Current Drawdown:</b> {current_drawdown*100:.2f}%\n"
            f"• <b>Action:</b> Trading halted for capital protection."
        )
        return await self.send_message(msg)

    async def notify_daily_summary(self, pnl_pct: float, open_positions: int, win_rate: float) -> bool:
        """Send daily performance summary notification."""
        emoji = "📈" if pnl_pct >= 0 else "📉"
        msg = (
            f"<b>📊 SQA-ML DAILY PERFORMANCE SUMMARY</b> {emoji}\n\n"
            f"• <b>Daily PnL:</b> {pnl_pct:+.2f}%\n"
            f"• <b>Open Positions:</b> {open_positions}\n"
            f"• <b>Win Rate:</b> {win_rate*100:.1f}%"
        )
        return await self.send_message(msg)
