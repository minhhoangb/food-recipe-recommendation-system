"""Tests for the config-driven bundle selection (`serving.bundle`).

Every bundle kind must be loadable through the same entry point, and a bundle whose
hashes do not match its manifest must stop the app before it serves anything.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from api.app import app
from src.config import load_config
from src.serving import BundleError, bundle_manifest, load_retriever
from src.utils import read_json, read_jsonl, write_json, write_jsonl


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    """A TestClient on a small, real corpus (the fixture pipeline, not a mock)."""
    processed = tmp_path_factory.mktemp("serving_processed")
    profile_dir = processed / "api"
    profile_dir.mkdir(parents=True)
    write_jsonl(
        profile_dir / "corpus.jsonl",
        [
            {
                "recipe_id": "r1",
                "group_id": "g1",
                "title": "Chicken Rice",
                "ingredients_raw": ["chicken", "rice"],
                "ingredients_normalized": ["chicken", "rice"],
                "directions": ["Cook the rice."],
                "link": "https://example.com/r1",
                "source": "test",
                "ner_raw": ["chicken", "rice"],
            },
            {
                "recipe_id": "r2",
                "group_id": "g2",
                "title": "Rice Bowl",
                "ingredients_raw": ["rice"],
                "ingredients_normalized": ["rice"],
                "directions": [],
                "link": "https://example.com/r2",
                "source": "test",
                "ner_raw": ["rice"],
            },
        ],
    )
    config = processed / "config.yaml"
    config.write_text(
        f"data:\n  profile: api\npaths:\n  processed_dir: {processed}\n", encoding="utf-8"
    )
    os.environ["API_CONFIG"] = str(config)
    os.environ["API_REQUEST_LOG"] = str(processed / "requests.db")
    os.environ.pop("API_BACKEND", None)
    os.environ.pop("API_BUNDLE", None)
    with TestClient(app) as client:
        yield client, processed
    for name in ("API_CONFIG", "API_REQUEST_LOG", "API_BACKEND", "API_BUNDLE"):
        os.environ.pop(name, None)


def _config_with_bundle(processed: Path, bundle: Path, path: Path) -> Path:
    """A config that serves `bundle` and reads the corpus from `processed`."""
    config = path / "config_with_bundle.yaml"
    config.write_text(
        f"data:\n  profile: api\npaths:\n  processed_dir: {processed}\n"
        f"serving:\n  bundle: {bundle}\n",
        encoding="utf-8",
    )
    return config


def _lexical_bundle(tmp_path: Path, processed: Path | None = None, corpus_sha: str = "") -> Path:
    """A tfidf bundle on disk, written the way `src.baseline` writes it.

    With `processed` given, the bundle is built on the corpus already in that folder, so it
    matches the corpus the app serves.
    """
    from src.baseline import fit_retriever, save_bundle

    if processed is None:
        processed = tmp_path / "processed"
    profile_dir = processed / "api"
    profile_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = profile_dir / "corpus.jsonl"
    if not corpus_file.exists():
        write_jsonl(
            corpus_file,
            [
                {
                    "recipe_id": "r1",
                    "title": "Chicken Rice",
                    "ingredients_normalized": ["chicken", "rice"],
                },
                {"recipe_id": "r2", "title": "Rice Bowl", "ingredients_normalized": ["rice"]},
            ],
        )
    corpus = list(read_jsonl(corpus_file))
    train_ids = {r["recipe_id"] for r in corpus}
    cfg = load_config(
        "configs/default.yaml",
        {"data": {"profile": "api"}, "paths": {"processed_dir": str(processed)}},
    )
    retriever, info = fit_retriever("tfidf", corpus, train_ids)
    bundle = save_bundle(retriever, info, cfg, tmp_path)
    if corpus_sha:
        manifest = read_json(bundle / "manifest.json")
        write_json(bundle / "manifest.json", {**manifest, "corpus_sha256": corpus_sha})
    return bundle


def test_a_lexical_bundle_loads_through_the_shared_entry_point(tmp_path):
    retriever = load_retriever(_lexical_bundle(tmp_path))
    assert retriever.name == "tfidf"
    assert retriever.search(["rice"], 1)[0].recipe_id in {"r1", "r2"}


def test_a_dense_bundle_loads_through_the_shared_entry_point(tmp_path, monkeypatch):
    import src.retrieval as retrieval
    from src.embeddings import save_bundle

    class FakeEncoder:
        def encode(self, sentences, **kwargs):
            vectors = np.zeros((len(sentences), 8), dtype=np.float32)
            for row, text in enumerate(sentences):
                codes = [ord(c) for c in text[:8]]
                codes += [0] * (8 - len(codes))
                vector = np.array(codes, dtype=np.float32)
                norm = np.linalg.norm(vector)
                vectors[row] = vector / norm if norm else vector
            return vectors

    monkeypatch.setattr(retrieval, "load_model", lambda: FakeEncoder())
    vectors = np.eye(2, 8, dtype=np.float32)
    bundle = save_bundle(
        vectors,
        ["r1", "r2"],
        mode="ingredients",
        profile="api",
        cfg=OmegaConf.create({"seed": 42, "data": {"profile": "api"}}),
        corpus_sha256="deadbeef",
        truncated=0,
        encode_seconds=1.0,
        out_dir=tmp_path,
    )
    retriever = load_retriever(bundle)
    assert retriever.name == "minilm_pretrained"
    assert {hit.recipe_id for hit in retriever.search(["rice"], 2)} == {"r1", "r2"}


def test_a_folder_that_is_no_bundle_is_reported(tmp_path):
    with pytest.raises(BundleError, match="neither"):
        load_retriever(tmp_path)


def test_the_manifest_is_read_from_the_bundle(tmp_path):
    bundle = _lexical_bundle(tmp_path)
    manifest = bundle_manifest(bundle)
    assert manifest["model_name"] == "tfidf"
    assert manifest["corpus_sha256"] == bundle_manifest(bundle)["corpus_sha256"]


def test_a_tampered_bundle_is_rejected(tmp_path):
    bundle = _lexical_bundle(tmp_path)
    with open(bundle / "model.joblib", "ab") as handle:
        handle.write(b"x")
    with pytest.raises(Exception, match="sha256"):
        load_retriever(bundle)


def test_the_config_selects_the_bundle(api_client, tmp_path):
    """The bundle in the config is the one that serves; the env var overrides it."""
    _, processed = api_client
    bundle = _lexical_bundle(tmp_path, processed)
    config = _config_with_bundle(processed, bundle, tmp_path)
    os.environ["API_CONFIG"] = str(config)
    try:
        with TestClient(app) as client:
            body = client.get("/model-info").json()
            assert body["model"] == "tfidf"
            assert body["backend"] == "lexical"
            assert body["bundle"] == str(bundle)
            assert body["corpus_sha256"] == bundle_manifest(bundle)["corpus_sha256"]
    finally:
        os.environ.pop("API_CONFIG", None)


def test_a_bundle_of_another_corpus_does_not_start(api_client, tmp_path):
    """A bundle built on another corpus must stop the app: its ids mean nothing here."""
    _, processed = api_client
    bundle = _lexical_bundle(tmp_path, corpus_sha="a" * 64)
    config = _config_with_bundle(processed, bundle, tmp_path)
    os.environ["API_CONFIG"] = str(config)
    try:
        with pytest.raises(Exception, match="another corpus"):
            with TestClient(app):
                pass
    finally:
        os.environ.pop("API_CONFIG", None)


def test_a_bundle_with_a_wrong_hash_does_not_start(api_client, tmp_path):
    """A bundle whose file hash does not match its manifest must stop the app."""
    _, processed = api_client
    bundle = _lexical_bundle(tmp_path, processed)
    with open(bundle / "model.joblib", "ab") as handle:
        handle.write(b"x")
    config = _config_with_bundle(processed, bundle, tmp_path)
    os.environ["API_CONFIG"] = str(config)
    try:
        with pytest.raises(Exception, match="sha256"):
            with TestClient(app):
                pass
    finally:
        os.environ.pop("API_CONFIG", None)


def test_the_ui_is_served(api_client):
    client, _ = api_client
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>What can I cook?</title>" in response.text
