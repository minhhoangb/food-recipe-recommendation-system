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


def test_processed_cache_ignores_serving_download_and_profile_changes(built_fixture, fixture_csv):
    cfg, _, directory = built_fixture
    before = {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    changed = OmegaConf.merge(
        cfg,
        {
            "serving": {"bundle": "models/another-bundle"},
            "artifacts": {"url": "https://example.com/data.zip", "sha256": "a" * 64},
            "data": {"profile": "pilot"},
        },
    )
    assert train.run_stage_processed(changed, raw_path=fixture_csv) == directory / "pilot"
    assert {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize(
    "overrides",
    [{"data": {"max_ingredients": 39}}, {"queries": {"max_query_size": 5}}],
)
def test_processed_cache_rejects_changed_data_parameters(built_fixture, overrides):
    cfg, _, _ = built_fixture
    with pytest.raises(ValueError, match="hash mismatch"):
        train.run_stage_processed(OmegaConf.merge(cfg, overrides))


def test_legacy_processed_cache_verifies_its_original_config(built_fixture, fixture_csv, tmp_path, monkeypatch):
    cfg, _, directory = built_fixture
    # Copy the fixture so the session's original manifest stays untouched.
    import shutil

    copied = tmp_path / "processed"
    shutil.copytree(directory, copied)
    source_config = tmp_path / "source.yaml"
    OmegaConf.save(cfg, source_config)
    monkeypatch.setattr(train, "LEGACY_DATA_CONFIG", source_config, raising=False)
    manifest_path = copied / "main" / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    manifest.pop("data_config_sha256", None)
    write_json(manifest_path, manifest)
    changed = OmegaConf.merge(
        cfg,
        {"paths": {"processed_dir": str(copied)}, "artifacts": {"url": "https://example.com/new.zip"}},
    )
    before = manifest_path.read_bytes()
    assert train.run_stage_processed(changed, raw_path=fixture_csv) == copied / "main"
    assert manifest_path.read_bytes() == before
    with pytest.raises(ValueError, match="hash mismatch"):
        train.run_stage_processed(OmegaConf.merge(changed, {"queries": {"max_query_size": 5}}))
    manifest["config_sha256"] = "0" * 64
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="hash mismatch"):
        train.run_stage_processed(changed)


@pytest.mark.parametrize("tamper", ["corpus", "normalization", "profile"])
def test_processed_cache_still_rejects_modified_artifacts(built_fixture, tmp_path, tamper):
    import shutil

    cfg, _, directory = built_fixture
    copied = tmp_path / "processed"
    shutil.copytree(directory, copied)
    if tamper == "corpus":
        with (copied / "main" / "corpus.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("{}\n")
    else:
        path = copied / "main" / "dataset_manifest.json"
        manifest = read_json(path)
        if tamper == "normalization":
            manifest["normalization"]["fingerprint"] = "0" * 64
        else:
            manifest["profile"] = "pilot"
        write_json(path, manifest)
    changed = OmegaConf.merge(cfg, {"paths": {"processed_dir": str(copied)}})
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
