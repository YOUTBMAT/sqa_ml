"""Send a periodic PAPER portfolio and service status report to Telegram."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_PATH = PROJECT_ROOT / "data" / "paper" / "paper_portfolio.json"
SAFETY_LOCK_PATH = PROJECT_ROOT / "models" / "saved" / "autotrain_safety_lock.json"
BOT_UNIT = "sqa_ml_bot.service"


def load_configuration() -> tuple[str, str]:
    """Load Telegram credentials from the root .env, falling back to config/.env."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        env_path = PROJECT_ROOT / "config" / ".env"
    load_dotenv(env_path, override=False)
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be configured.")
    return token, chat_id


def read_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read {description} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} at {path} must contain a JSON object.")
    return value


def get_service_status() -> str:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", BOT_UNIT],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    status = result.stdout.strip() or result.stderr.strip()
    return status or f"unknown (systemctl exit {result.returncode})"


def get_psi_status() -> str:
    result = subprocess.run(
        [
            "journalctl",
            "--user",
            "-u",
            BOT_UNIT,
            "--since",
            "30 minutes ago",
            "--no-pager",
            "--output=cat",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"journalctl exit {result.returncode}"
        raise RuntimeError(f"Unable to inspect recent PSI logs: {detail}")

    alerts = [
        line.strip()
        for line in result.stdout.splitlines()
        if "protective mode enabled" in line.lower() or "severe feature drift" in line.lower()
    ]
    if alerts:
        return "ALERT — severe drift detected in recent logs"
    return "No severe-drift alert in the last 30 minutes (not a guarantee of healthy PSI)"


def markdown_escape(value: object) -> str:
    """Escape dynamic text for Telegram MarkdownV2."""
    text = str(value)
    special = "\\_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{char}" if char in special else char for char in text)


def format_status_message(
    portfolio: dict,
    service_status: str,
    psi_status: str,
    safety_lock: dict | None,
) -> str:
    balance = float(portfolio["balance"])
    initial_balance = float(portfolio["initial_balance"])
    if initial_balance <= 0:
        raise RuntimeError("Portfolio initial_balance must be greater than zero.")
    pnl_pct = ((balance - initial_balance) / initial_balance) * 100
    positions = portfolio.get("positions", {})
    if not isinstance(positions, dict):
        raise RuntimeError("Portfolio positions must be a JSON object.")

    if safety_lock is None:
        lock_status = "INACTIVE"
    elif safety_lock.get("active") is True:
        reason = safety_lock.get("reason", "no reason provided")
        reason_text = str(reason)
        if len(reason_text) > 180:
            reason_text = reason_text[:177] + "..."
        lock_status = f"ACTIVE — {reason_text}"
    else:
        lock_status = "INACTIVE"

    lines = [
        "*📊 SQA\\-ML STATUS REPORT*",
        f"`{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        "",
        "*Portfolio*",
        f"• Initial balance: `${initial_balance:,.2f}`",
        f"• Current balance: `${balance:,.2f}`",
        f"• Accumulated PnL: *{markdown_escape(f'{pnl_pct:+.2f}%')}*",
        "",
        "*System*",
        f"• Bot service: {markdown_escape(service_status)}",
        f"• PSI status: {markdown_escape(psi_status)}",
        f"• Safety lock: {markdown_escape(lock_status)}",
        "",
        f"*Open positions \\({len(positions)}\\)*",
    ]
    if positions:
        for symbol, position in positions.items():
            if not isinstance(position, dict):
                raise RuntimeError(f"Position for {symbol} must be a JSON object.")
            side = position.get("side", "long")
            amount = position.get("amount", 0)
            entry_price = position.get("entry_price", 0)
            lines.append(
                f"• {markdown_escape(symbol)} — {markdown_escape(side)}; "
                f"amount {markdown_escape(amount)}; entry `${float(entry_price):,.2f}`"
            )
    else:
        lines.append("• None")
    return "\n".join(lines)


def send_message(token: str, chat_id: str, text: str) -> None:
    payload = json.dumps(
        {"chat_id": chat_id, "text": text, "parse_mode": "MarkdownV2"}
    ).encode("utf-8")
    request = Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Telegram status report request failed: HTTP {exc.code}: {detail}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Telegram status report request failed: {exc}") from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError("Telegram API rejected the status report.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report SQA-ML PAPER status to Telegram")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the formatted report without contacting Telegram",
    )
    args = parser.parse_args()

    try:
        portfolio = read_json(PORTFOLIO_PATH, "paper portfolio")
        safety_lock = (
            read_json(SAFETY_LOCK_PATH, "autotrain safety lock")
            if SAFETY_LOCK_PATH.exists()
            else None
        )
        message = format_status_message(
            portfolio,
            get_service_status(),
            get_psi_status(),
            safety_lock,
        )
        if args.dry_run:
            print(message)
            return 0
        token, chat_id = load_configuration()
        send_message(token, chat_id, message)
        print("Telegram status report sent.")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
