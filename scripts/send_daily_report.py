"""CLI script to generate and send daily performance report to Telegram at 00:00 UTC."""

import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import ccxt.async_support as ccxt_async

from src.utils.notifications import TelegramNotifier
from src.utils.logger import logger


async def main() -> None:
    """Generate daily portfolio performance report and send via Telegram."""
    logger.info("=== GENERATING SQA-ML DAILY TELEGRAM REPORT ===")

    state_file = Path("data/paper/paper_portfolio.json")
    if not state_file.exists():
        logger.warning("No paper portfolio state file found. Using default empty state.")
        state = {
            "balance": 10000.0,
            "initial_balance": 10000.0,
            "positions": {},
            "order_history": [],
        }
    else:
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load portfolio state: {e}")
            return

    balance = float(state.get("balance", 10000.0))
    initial_balance = float(state.get("initial_balance", 10000.0))
    positions = state.get("positions", {})
    order_history = state.get("order_history", [])

    # Fetch current prices via CCXT async
    current_prices = {}
    symbols_to_fetch = list(positions.keys()) if positions else ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    exchange = ccxt_async.binance({"enableRateLimit": True})
    try:
        for symbol in symbols_to_fetch:
            try:
                ticker = await exchange.fetch_ticker(symbol)
                current_prices[symbol] = float(ticker["last"])
            except Exception as ex:
                logger.warning(f"Could not fetch live ticker for {symbol}: {ex}. Using fallback price from positions.")
                if symbol in positions:
                    current_prices[symbol] = float(positions[symbol].get("entry_price", 0.0))
                else:
                    current_prices[symbol] = 0.0
    finally:
        await exchange.close()

    # Calculate Floating PnL
    floating_pnl_usd = 0.0
    open_positions_text = ""
    for sym, pos in positions.items():
        side = pos.get("side", "long")
        amount = float(pos.get("amount", 0.0))
        entry_price = float(pos.get("entry_price", 0.0))
        curr_price = current_prices.get(sym, entry_price)
        sl = pos.get("stop_loss")
        tp = pos.get("take_profit")

        if entry_price > 0:
            pnl_direction = 1 if side == "long" else -1
            pos_pnl = (curr_price - entry_price) * amount * pnl_direction
            pos_pnl_pct = ((curr_price - entry_price) / entry_price) * 100 * pnl_direction
            floating_pnl_usd += pos_pnl

            sl_str = f"${sl:,.2f}" if sl is not None else "N/A"
            tp_str = f"${tp:,.2f}" if tp is not None else "N/A"
            open_positions_text += (
                f"• <b>{sym}</b> ({side.upper()})\n"
                f"  - Entry: ${entry_price:,.2f} | Current: ${curr_price:,.2f}\n"
                f"  - PnL: ${pos_pnl:+,.2f} ({pos_pnl_pct:+.2f}%)\n"
                f"  - SL: {sl_str} | TP: {tp_str}\n"
            )

    total_equity = balance + floating_pnl_usd
    total_return_pct = ((total_equity - initial_balance) / initial_balance) * 100

    # Orders in last 24h
    now_utc = datetime.now(timezone.utc)
    cutoff_24h = now_utc - timedelta(hours=24)
    trades_24h = 0
    for ord_rec in order_history:
        ts_str = ord_rec.get("timestamp")
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str)
                if dt >= cutoff_24h:
                    trades_24h += 1
            except Exception:
                pass

    emoji = "📈" if total_return_pct >= 0 else "📉"

    if not open_positions_text:
        open_positions_text = "<i>No open positions currently active.</i>\n"

    message = (
        f"<b>📊 SQA-ML DAILY PERFORMANCE REPORT</b> {emoji}\n"
        f"<i>Timestamp: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}</i>\n\n"
        f"🛡️ <b>Portfolio Overview</b>\n"
        f"• <b>Total Equity:</b> ${total_equity:,.2f} ({total_return_pct:+.2f}%)\n"
        f"• <b>Available Balance:</b> ${balance:,.2f}\n"
        f"• <b>Floating PnL:</b> ${floating_pnl_usd:+,.2f}\n"
        f"• <b>Trades (Last 24h):</b> {trades_24h}\n\n"
        f"📂 <b>Active Positions ({len(positions)})</b>\n"
        f"{open_positions_text}"
    )

    notifier = TelegramNotifier()
    success = await notifier.send_message(message)
    if success:
        logger.success("Daily report sent successfully via Telegram.")
    else:
        logger.error("Failed to send daily report via Telegram.")


if __name__ == "__main__":
    asyncio.run(main())
