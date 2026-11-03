from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src import train
from src.utils import read_json, write_json
from tests.test_embeddings import FakeEncoder


def test_all_stages_cache_and_invalidate_training_parameters(
    built_fixture, fixture_csv, tmp_path, monkeypatch
):
    import sentence_transformers

    cfg, _, _ = built_fixture
    monkeypatch.setattr(train, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *a, **kw: FakeEncoder())
    loaded_encoders = []

    def load_encoder(**kwargs):
        loaded_encoders.append(kwargs["model_name_or_path"])
        return FakeEncoder()

    monkeypatch.setattr("src.retrieval.load_model", load_encoder)
    training_calls = []

    def fit(pairs, *, output_path, **kwargs):
        training_calls.append(kwargs)
        output_path.mkdir(parents=True, exist_ok=True)
        for name in ("modules.json", "model.safetensors", "config.json"):
            (output_path / name).write_text("fixture model")
        summary = {
            "base_model": "fixture",
            "epochs": kwargs["epochs"],
            "batch_size": kwargs["batch_size"],
            "learning_rate": kwargs["learning_rate"],
            "n_examples": len(pairs),
            "device": kwargs["device"],
            "train_seconds": 0.123,
        }
        return FakeEncoder(), summary

    monkeypatch.setattr("src.finetune.finetune_model", fit)
    models = tmp_path / "models"
    result = train.run_pipeline(cfg, raw_path=fixture_csv, models_dir=models, log_to_mlflow=False)
    assert set(result) == set(train.STAGES)
    assert len(training_calls) == 1 and training_calls[0]["device"] == "cpu"
    assert result["evaluate"]["train_summary"]["train_seconds"] == 0.123
    assert loaded_encoders[-1] == str(
        (result["evaluate"]["bundle"] and Path(result["evaluate"]["bundle"]) / "encoder").resolve()
    )
    train.run_pipeline(cfg, raw_path=fixture_csv, models_dir=models, log_to_mlflow=False)
    assert len(training_calls) == 1
    train.run_stage_finetune(cfg, out_models_dir=models, learning_rate=1e-5)
    assert len(training_calls) == 2
    # Changing a cached artifact also forces a rebuild, even when inputs are unchanged.
    (models / "checkpoint_main_ingredients" / "model.safetensors").write_text("tampered")
    train.run_stage_finetune(cfg, out_models_dir=models, learning_rate=1e-5)
    assert len(training_calls) == 3


def test_processed_cache_rejects_changed_config(built_fixture):
    cfg, _, _ = built_fixture
    changed = OmegaConf.merge(cfg, {"seed": 99})
    with pytest.raises(ValueError, match="hash mismatch"):
        train.run_stage_processed(changed)


def test_cache_verifies_inputs_and_artifact_hashes(tmp_path):
    file = tmp_path / "artifact.json"
    write_json(file, {"value": 1})
    cache = tmp_path / "stage_cache.json"
    train._record_cache(cache, {"seed": 42}, [file])
    assert train._cache_matches(cache, {"seed": 42})
    assert not train._cache_matches(cache, {"seed": 43})
    write_json(file, {"value": 2})
    assert not train._cache_matches(cache, {"seed": 42})


def test_stage_evaluate_logs_observed_metadata_only(built_fixture, tmp_path, monkeypatch):
    import numpy as np

    from src.embeddings import save_bundle

    cfg, _, _ = built_fixture
    bundle = save_bundle(
        np.eye(2, 8, dtype=np.float32),
        ["r1", "r2"],
        mode="ingredients",
        profile="main",
        cfg=cfg,
        corpus_sha256="fixture",
        truncated=0,
        encode_seconds=0.246,
        model_name="minilm_finetuned",
        out_dir=tmp_path,
    )
    monkeypatch.setattr("src.retrieval.DenseRetriever.from_bundle", lambda *a, **kw: object())
    monkeypatch.setattr("src.evaluation.load_split", lambda *a: ([], []))
    monkeypatch.setattr("src.evaluation.evaluate_retriever", lambda *a, **kw: {"mrr@10": 0.5})
    recorded = []
    monkeypatch.setattr(train, "_log_and_register_mlflow", lambda **kw: recorded.append(kw) or "fixture")
    train.run_stage_evaluate(cfg, bundle_dir=bundle)
    assert recorded[-1]["result"]["train_summary"] == {}
    assert recorded[-1]["result"]["encode_seconds"] == 0.246
    write_json(bundle / "train_summary.json", {"n_examples": 27, "train_seconds": 2.345})
    train.run_stage_evaluate(cfg, bundle_dir=bundle)
    assert recorded[-1]["result"]["train_summary"] == read_json(bundle / "train_summary.json")
