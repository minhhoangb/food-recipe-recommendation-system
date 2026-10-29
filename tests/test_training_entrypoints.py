import mlflow
import numpy as np
import pytest

from src import finetune, train
from src.config import processed_dir
from src.utils import read_json


class TrainingReached(Exception):
    pass


def stop_before_training(pairs, **kwargs):
    assert pairs and all(isinstance(pair, dict) for pair in pairs)
    assert all(pair["split"] == "train" for pair in pairs)
    raise TrainingReached


def test_finetune_cli_builds_train_pairs(built_fixture, tmp_path, monkeypatch):
    cfg, _, out = built_fixture
    path = tmp_path / "fixture.yaml"
    path.write_text(f"paths:\n  processed_dir: {out.as_posix()}\ndata:\n  profile: main\n")
    monkeypatch.setattr(finetune, "finetune_model", stop_before_training)
    with pytest.raises(TrainingReached):
        finetune.main(["--config", str(path)])


def test_training_pipeline_accepts_pair_builder_result(built_fixture, monkeypatch):
    cfg, _, _ = built_fixture
    train_ids = set(read_json(processed_dir(cfg) / "splits.json")["train"])

    def check_pairs(pairs, **kwargs):
        assert all(pair["recipe_id"] in train_ids for pair in pairs)
        stop_before_training(pairs)

    monkeypatch.setattr(train, "finetune_model", check_pairs)
    with pytest.raises(TrainingReached):
        train.run_training_pipeline(cfg, log_to_mlflow=False)


def test_training_logs_ranking_metrics_to_mlflow(built_fixture, tmp_path):
    from src.embeddings import save_bundle

    cfg, _, _ = built_fixture
    bundle = save_bundle(
        np.ones((2, 8), dtype=np.float32) / np.sqrt(8),
        ["r1", "r2"],
        mode="ingredients",
        profile="main",
        cfg=cfg,
        corpus_sha256="test-corpus",
        truncated=0,
        encode_seconds=0.01,
        model_name="minilm_finetuned",
        out_dir=tmp_path,
    )
    uri = f"sqlite:///{(tmp_path / 'training.db').as_posix()}"
    result = {
        "run_id": bundle.name,
        "mode": "ingredients",
        "train_summary": {
            "base_model": "fixture",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 2e-5,
            "n_examples": 2,
            "train_seconds": 0.01,
        },
        "metrics": {"mrr@10": 0.5, "recall@10": 1.0},
        "encode_seconds": 0.01,
        "peak_rss_mb": 1.0,
    }
    run_id = train._log_and_register_mlflow(
        result=result,
        cfg=cfg,
        bundle_dir=bundle,
        tracking_uri=uri,
        experiment="training-fixture",
        model_registry_name="fixture_model",
        promote_production=False,
    )
    metrics = mlflow.MlflowClient(uri).get_run(run_id).data.metrics
    assert metrics["mrr_at_10"] == 0.5 and metrics["recall_at_10"] == 1.0
