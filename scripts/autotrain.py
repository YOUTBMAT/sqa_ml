"""Safely retrain, calibrate, and promote a model without replacing production on failure."""

import argparse
import asyncio
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone

from src.data.collectors.crypto import CryptoCollector
from src.features.pipeline import FeaturePipeline
from src.features.storage import FeatureStorage
from src.utils.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "models" / "saved" / "autotrain_safety_lock.json"
RUN_LOCK_PATH = PROJECT_ROOT / "models" / "saved" / "autotrain_running.lock"


def set_safety_lock(reason: str) -> None:
    """Atomically activate the runner's no-new-entries lock."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = LOCK_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"active": True, "reason": reason, "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, LOCK_PATH)


def acquire_run_lock(run_id: str) -> int | None:
    """Acquire a kernel-released lock so crashes cannot leave a stale running marker."""
    RUN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(RUN_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    os.ftruncate(descriptor, 0)
    os.write(descriptor, json.dumps({"pid": os.getpid(), "run_id": run_id}).encode("utf-8"))
    os.fsync(descriptor)
    return descriptor


async def download_and_build_features(symbol: str, timeframe: str, days: int, exchange_id: str) -> int:
    """Fetch recent exchange candles and persist their engineered feature matrix."""
    now = datetime.now(timezone.utc)
    async with CryptoCollector(exchange_id=exchange_id) as collector:
        candles = await collector.fetch_ohlcv_historical(
            symbol, timeframe, now - timedelta(days=days), now
        )
    if candles.is_empty():
        raise RuntimeError(f"Exchange returned no OHLCV candles for {symbol} ({timeframe}).")
    match = re.fullmatch(r"(\d+)(m|h|d)", timeframe)
    if match:
        unit_seconds = {"m": 60, "h": 3600, "d": 86400}[match.group(2)]
        candle_seconds = int(match.group(1)) * unit_seconds
        completed_until = datetime.fromtimestamp(
            int(now.timestamp()) // candle_seconds * candle_seconds, tz=timezone.utc
        )
        candles = candles.filter(candles["timestamp"] < completed_until)
    if candles.is_empty():
        raise RuntimeError(f"No completed candles available for {symbol} ({timeframe}).")
    features = FeaturePipeline().run(candles)
    if features.is_empty():
        raise RuntimeError(f"Feature generation returned no rows for {symbol} ({timeframe}).")
    FeatureStorage().save_features(features, symbol, timeframe)
    return len(features)


def run_step(label: str, command: list[str], environment: dict[str, str]) -> None:
    """Run one pipeline subprocess and raise on failure."""
    logger.info(f"Autotrain step: {label}")
    logger.info("Command: " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    logger.success(f"Autotrain step complete: {label}")


def promote_staged_model(
    symbol: str,
    timeframe: str,
    staged_models: Path,
    selected_features: Path,
    calibration_result: dict,
) -> None:
    """Atomically replace individual production artifacts while the safety lock remains active."""
    threshold = calibration_result.get("threshold")
    if calibration_result.get("status") != "approved":
        raise ValueError("Refusing to promote a model without approved calibration.")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Approved calibration must include a finite probability threshold in [0, 1].")

    sanitized = symbol.replace("/", "_").replace("-", "_").lower()
    production_models = PROJECT_ROOT / "models" / "saved"
    production_models.mkdir(parents=True, exist_ok=True)
    for suffix in ("model.joblib", "metadata.json"):
        name = f"{sanitized}_{timeframe}_{suffix}"
        source = staged_models / name
        if not source.is_file():
            raise FileNotFoundError(f"Staged model artifact missing: {source}")
    metadata_path = staged_models / f"{sanitized}_{timeframe}_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["calibrated_threshold"] = float(threshold)
    metadata["calibration_status"] = "approved"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    replacements = [
        (staged_models / f"{sanitized}_{timeframe}_model.joblib",
         production_models / f"{sanitized}_{timeframe}_model.joblib"),
        (metadata_path, production_models / f"{sanitized}_{timeframe}_metadata.json"),
        (selected_features, PROJECT_ROOT / "config" / "selected_features.json"),
    ]
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for source, destination in replacements:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            shutil.copy2(source, temporary)
            temporary_paths.append((temporary, destination))
        for temporary, destination in temporary_paths:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in temporary_paths:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely retrain and calibrate the SQA-ML model")
    parser.add_argument("--symbol", type=str, default="ETH/USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--days", type=int, default=365, help="Recent candle history to download")
    parser.add_argument("--exchange", type=str, default="binance")
    args = parser.parse_args()
    if args.days < 30:
        parser.error("--days must be at least 30 to provide adequate training history.")

    run_id = uuid.uuid4().hex
    staging_root = PROJECT_ROOT / "models" / "saved" / f"autotrain-staging-{run_id}"
    staged_models = staging_root / "models"
    staged_features = staging_root / "config" / "selected_features.json"
    result_path = staging_root / "calibration.json"
    python = sys.executable
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PROJECT_ROOT), existing_pythonpath) if part
    )

    run_lock_fd = acquire_run_lock(run_id)
    if run_lock_fd is None:
        logger.error(f"Autotrain already running (lock: {RUN_LOCK_PATH}); refusing concurrent run.")
        return 1

    try:
        set_safety_lock("Automated retraining/calibration is in progress.")
        logger.warning(f"Production entry lock active for retraining {args.symbol} ({args.timeframe}).")
        logger.info(f"Downloading candles and generating features for {args.symbol} ({args.timeframe}).")
        count = asyncio.run(
            download_and_build_features(args.symbol, args.timeframe, args.days, args.exchange)
        )
        logger.success(f"Persisted feature rows: {count}.")
        run_step(
            "rebuilding labeled dataset",
            [python, "scripts/build_dataset.py", "--symbol", args.symbol, "--timeframe", args.timeframe],
            environment,
        )
        run_step(
            "Purged MDA feature selection",
            [
                python, "scripts/select_features.py", "--symbol", args.symbol, "--timeframe", args.timeframe,
                "--output", str(staged_features),
            ],
            environment,
        )
        run_step(
            "staged LightGBM training",
            [
                python, "scripts/train_model.py", "--symbol", args.symbol, "--timeframe", args.timeframe,
                "--models-dir", str(staged_models), "--selected-features", str(staged_features),
            ],
            environment,
        )
        run_step(
            "staged threshold calibration and holdout validation",
            [
                python, "scripts/calibrate_thresholds.py", "--symbol", args.symbol, "--timeframe", args.timeframe,
                "--models-dir", str(staged_models), "--result-json", str(result_path),
                "--require-positive-holdout",
            ],
            environment,
        )
        calibration = json.loads(result_path.read_text(encoding="utf-8"))
        if calibration.get("status") != "approved":
            raise RuntimeError("Calibration did not approve a positive threshold and holdout.")

        promote_staged_model(
            args.symbol, args.timeframe, staged_models, staged_features, calibration
        )
        LOCK_PATH.unlink(missing_ok=True)
        logger.success(
            f"Retrained model promoted for {args.symbol}; calibrated threshold "
            f"{calibration['threshold']:.2f}. Production entry lock cleared."
        )
        return 0
    except Exception as exc:
        try:
            set_safety_lock(f"Autotrain failed; no new entries allowed: {exc}")
        except OSError as lock_error:
            logger.critical(f"Failed to persist autotrain safety lock: {lock_error}")
        logger.exception(f"Autotrain failed; production model was not approved and the safety lock remains active: {exc}")
        return 1
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        os.close(run_lock_fd)


if __name__ == "__main__":
    sys.exit(main())
