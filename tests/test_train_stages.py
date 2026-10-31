from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from omegaconf import DictConfig

from src import train


def test_stage_raw_missing_raises_guide(tmp_path: Path, capsys):
    missing_path = tmp_path / "nonexistent_raw.csv"
    cfg = DictConfig({"paths": {"raw_csv": str(missing_path)}, "data": {"profile": "pilot"}})
    with pytest.raises(FileNotFoundError, match="Missing raw dataset"):
        train.run_stage_raw(cfg, raw_path=missing_path)

    captured = capsys.readouterr()
    assert "Hướng dẫn thu thập dữ liệu RecipeNLG" in captured.err
    assert "fetch_artifacts.py" in captured.err


def test_stage_raw_existing_succeeds(built_fixture, fixture_csv):
    cfg, _, _ = built_fixture
    verified = train.run_stage_raw(cfg, raw_path=fixture_csv)
    assert verified.exists()


def test_stage_processed_caching(built_fixture, fixture_csv, capsys):
    cfg, _, out = built_fixture
    # First call: already built by fixture, should hit cache
    ret_dir = train.run_stage_processed(cfg, raw_path=fixture_csv, force=False)
    assert ret_dir.exists()
    captured = capsys.readouterr()
    assert "[CACHED]" in captured.out


def test_stage_pairs_builds_and_caches(built_fixture, capsys):
    cfg, _, _ = built_fixture
    pairs_file = train.run_stage_pairs(cfg, mode="ingredients", force=False)
    assert pairs_file.exists()

    # Second call should be cached
    capsys.readouterr()
    train.run_stage_pairs(cfg, mode="ingredients", force=False)
    captured = capsys.readouterr()
    assert "[CACHED]" in captured.out


def test_stage_baseline_and_cache(built_fixture, tmp_path, capsys):
    cfg, _, _ = built_fixture
    bundle = train.run_stage_baseline(cfg, out_models_dir=tmp_path, force=False)
    assert bundle.exists()
    assert (bundle / "manifest.json").exists()

    # Second run should be cached
    capsys.readouterr()
    train.run_stage_baseline(cfg, out_models_dir=tmp_path, force=False)
    captured = capsys.readouterr()
    assert "[CACHED]" in captured.out


def test_train_cli_stage_flag(built_fixture, fixture_csv, tmp_path):
    cfg, _, out = built_fixture
    cfg_path = tmp_path / "test_cfg.yaml"
    cfg_path.write_text(
        f"seed: 42\n"
        f"paths:\n  raw_csv: {fixture_csv.as_posix()}\n  processed_dir: {out.as_posix()}\n"
        f"data:\n  profile: main\n"
    )

    # Test running single stage: raw
    ret = train.main(["--config", str(cfg_path), "--raw-path", str(fixture_csv), "--stage", "raw"])
    assert ret == 0


def test_registry_rollback(tmp_path: Path):
    uri = f"sqlite:///{(tmp_path / 'registry_test.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient(uri)

    model_name = "test_rollback_model"
    # Create empty registered model
    client.create_registered_model(model_name)
    # Create 2 dummy runs & model versions
    with mlflow.start_run() as r1:
        mlflow.log_param("version", "1")
    with mlflow.start_run() as r2:
        mlflow.log_param("version", "2")

    client.create_model_version(model_name, source=f"runs:/{r1.info.run_id}/model", run_id=r1.info.run_id)
    client.create_model_version(model_name, source=f"runs:/{r2.info.run_id}/model", run_id=r2.info.run_id)

    # Set initial aliases: v2 is production, v1 is staging
    client.set_registered_model_alias(model_name, "production", "2")
    client.set_registered_model_alias(model_name, "staging", "1")

    assert str(client.get_registered_model(model_name).aliases["production"]) == "2"

    # Execute rollback: roll production back to version 1
    res = train.rollback_model_registry(
        model_name=model_name,
        target_version=1,
        alias="production",
        tracking_uri=uri,
    )
    assert str(res["previous_version"]) == "2"
    assert str(res["current_version"]) == "1"
    assert str(client.get_registered_model(model_name).aliases["production"]) == "1"
