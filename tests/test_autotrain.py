"""Safety tests for staged autotrain promotion and its persistent entry lock."""

import json

import pytest

import scripts.autotrain as autotrain


def test_safety_lock_records_reason_and_is_atomic(tmp_path, monkeypatch) -> None:
    lock_path = tmp_path / "autotrain_safety_lock.json"
    monkeypatch.setattr(autotrain, "LOCK_PATH", lock_path)

    autotrain.set_safety_lock("Calibration failed.")

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["active"] is True
    assert payload["reason"] == "Calibration failed."
    assert not lock_path.with_suffix(".json.tmp").exists()


def test_staged_promotion_updates_model_and_calibrated_threshold(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    staged_models = tmp_path / "staging" / "models"
    staged_models.mkdir(parents=True)
    staged_features = tmp_path / "staging" / "selected_features.json"
    staged_features.write_text(json.dumps(["roc_24"]) + "\n", encoding="utf-8")
    (staged_models / "eth_usdt_1h_model.joblib").write_bytes(b"trained-model")
    (staged_models / "eth_usdt_1h_metadata.json").write_text(
        json.dumps({"feature_cols": ["roc_24"], "oos_roc_auc": 0.55}),
        encoding="utf-8",
    )
    monkeypatch.setattr(autotrain, "PROJECT_ROOT", project_root)

    autotrain.promote_staged_model(
        "ETH/USDT",
        "1h",
        staged_models,
        staged_features,
        {"status": "approved", "threshold": 0.62},
    )

    production = project_root / "models" / "saved"
    assert (production / "eth_usdt_1h_model.joblib").read_bytes() == b"trained-model"
    metadata = json.loads(
        (production / "eth_usdt_1h_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["calibrated_threshold"] == 0.62
    assert metadata["calibration_status"] == "approved"
    assert json.loads(
        (project_root / "config" / "selected_features.json").read_text(encoding="utf-8")
    ) == ["roc_24"]


def test_staged_promotion_refuses_missing_model_without_touching_production(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    production = project_root / "models" / "saved"
    production.mkdir(parents=True)
    existing_model = production / "eth_usdt_1h_model.joblib"
    existing_model.write_bytes(b"production-model")
    staged_models = tmp_path / "staging" / "models"
    staged_models.mkdir(parents=True)
    staged_features = tmp_path / "staging" / "selected_features.json"
    staged_features.write_text('["roc_24"]\n', encoding="utf-8")
    monkeypatch.setattr(autotrain, "PROJECT_ROOT", project_root)

    with pytest.raises(FileNotFoundError, match="Staged model artifact missing"):
        autotrain.promote_staged_model(
            "ETH/USDT",
            "1h",
            staged_models,
            staged_features,
            {"status": "approved", "threshold": 0.62},
        )

    assert existing_model.read_bytes() == b"production-model"


def test_staged_promotion_rejects_failed_calibration_before_writing(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    production = project_root / "models" / "saved"
    production.mkdir(parents=True)
    existing_model = production / "eth_usdt_1h_model.joblib"
    existing_model.write_bytes(b"production-model")
    monkeypatch.setattr(autotrain, "PROJECT_ROOT", project_root)

    with pytest.raises(ValueError, match="without approved calibration"):
        autotrain.promote_staged_model(
            "ETH/USDT",
            "1h",
            tmp_path / "staging" / "models",
            tmp_path / "staging" / "selected_features.json",
            {"status": "rejected", "threshold": 0.5},
        )

    assert existing_model.read_bytes() == b"production-model"
