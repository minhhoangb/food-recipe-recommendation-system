"""Tests for the FastAPI app, driven by TestClient (httpx).

The app is built on the real processed corpus when it is there; otherwise the tests
build a tiny corpus with the shared fixture pipeline, so they never need the network
and never need the 50k main corpus.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app
from src.config import load_config, processed_dir
from src.utils import write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    """A TestClient on a small, real corpus (the fixture pipeline, not a mock)."""
    processed = tmp_path_factory.mktemp("api_processed")
    profile_dir = processed / "api"
    profile_dir.mkdir(parents=True)
    corpus = profile_dir / "corpus.jsonl"
    write_jsonl(
        corpus,
        [
            {
                "recipe_id": "r1",
                "group_id": "g1",
                "title": "Chicken Rice",
                "ingredients_raw": ["chicken", "rice", "soy sauce"],
                "ingredients_normalized": ["chicken", "rice", "soy sauce"],
                "directions": ["Cook the rice."],
                "link": "https://example.com/r1",
                "source": "test",
                "ner_raw": ["chicken", "rice", "soy sauce"],
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
            {
                "recipe_id": "r3",
                "group_id": "g3",
                "title": "Chicken Soup",
                "ingredients_raw": ["chicken", "water"],
                "ingredients_normalized": ["chicken", "water"],
                "directions": [],
                "link": "https://example.com/r3",
                "source": "test",
                "ner_raw": ["chicken", "water"],
            },
        ],
    )
    config = processed / "config.yaml"
    config.write_text(
        f"data:\n  profile: api\npaths:\n  processed_dir: {processed}\n", encoding="utf-8"
    )
    os.environ["API_BACKEND"] = "reference"
    os.environ["API_CONFIG"] = str(config)
    os.environ["API_REQUEST_LOG"] = str(processed / "requests.db")
    with TestClient(app) as client:
        yield client
    for name in ("API_BACKEND", "API_CONFIG", "API_REQUEST_LOG"):
        os.environ.pop(name, None)


def test_health_reports_ok(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_describes_the_serving_model(api_client):
    response = api_client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "tfidf-reference"
    assert body["corpus_recipes"] == 3


def test_predict_returns_the_matched_and_missing_ingredients(api_client):
    response = api_client.post("/predict", json={"ingredients": ["chicken", "rice"], "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "tfidf-reference"
    assert len(body["results"]) == 2
    first = body["results"][0]
    assert set(first["matched_ingredients"]) == {"chicken", "rice"}
    assert first["directions"] == ["Cook the rice."]
    assert first["link"].startswith("https://")


def test_predict_normalises_the_input(api_client):
    response = api_client.post("/predict", json={"ingredients": ["1 cup chicken", "RICE"], "top_k": 1})
    assert response.status_code == 200
    assert response.json()["results"][0]["matched_ingredients"] == ["chicken", "rice"]


@pytest.mark.parametrize("top_k", [True, 0, 21, -1])
def test_an_invalid_top_k_is_rejected_with_a_description(api_client, top_k):
    response = api_client.post("/predict", json={"ingredients": ["rice"], "top_k": top_k})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_an_empty_ingredient_list_is_rejected(api_client):
    response = api_client.post("/predict", json={"ingredients": [], "top_k": 3})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_ingredients_that_normalise_to_nothing_are_rejected(api_client):
    response = api_client.post("/predict", json={"ingredients": ["!!!"], "top_k": 3})
    assert response.status_code == 422
    assert "normalisation" in str(response.json()["detail"]).lower()


def test_a_blank_ingredient_is_rejected(api_client):
    response = api_client.post("/predict", json={"ingredients": ["rice", "  "], "top_k": 3})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_an_unknown_field_is_rejected(api_client):
    response = api_client.post("/predict", json={"ingredients": ["rice"], "top_k": 3, "nope": 1})
    assert response.status_code == 422


def test_batch_predict_keeps_the_order_of_the_instances(api_client):
    payload = {
        "instances": [
            {"ingredients": ["rice"], "top_k": 1},
            {"ingredients": ["chicken"], "top_k": 2},
            {"ingredients": ["chicken", "rice"], "top_k": 3},
        ]
    }
    response = api_client.post("/batch-predict", json=payload)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    assert predictions[0]["results"][0]["recipe_id"] == "r2"
    assert predictions[2]["results"][0]["recipe_id"] == "r1"


def test_a_batch_of_21_instances_is_rejected(api_client):
    payload = {"instances": [{"ingredients": ["rice"], "top_k": 1} for _ in range(21)]}
    response = api_client.post("/batch-predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]


def test_an_empty_batch_is_rejected(api_client):
    response = api_client.post("/batch-predict", json={"instances": []})
    assert response.status_code == 422


def test_the_requests_are_logged(api_client, tmp_path_factory):
    api_client.post("/predict", json={"ingredients": ["chicken"], "top_k": 1})
    log_path = Path(os.environ["API_REQUEST_LOG"])
    import sqlite3

    with sqlite3.connect(log_path) as connection:
        rows = connection.execute(
            "SELECT endpoint, model, ingredients, top_k, status_code FROM request_log"
        ).fetchall()
    assert rows, "the request log must contain the prediction"
    endpoint, model, ingredients, top_k, status_code = rows[-1]
    assert endpoint == "/predict"
    assert model == "tfidf-reference"
    assert ingredients == "chicken"
    assert top_k == 1
    assert status_code == 200


def test_a_rejected_request_is_logged_with_its_status(api_client):
    api_client.post("/predict", json={"ingredients": ["rice"], "top_k": 99})
    log_path = Path(os.environ["API_REQUEST_LOG"])
    import sqlite3

    with sqlite3.connect(log_path) as connection:
        rows = connection.execute(
            "SELECT status_code FROM request_log WHERE status_code = 422"
        ).fetchall()
    assert rows, "a rejected request must be logged with its status code"


def test_the_api_works_against_the_real_processed_corpus():
    """The same app, on the real pilot corpus when it is present."""
    cfg = load_config("configs/pilot.yaml")
    corpus = processed_dir(cfg) / "corpus.jsonl"
    if not corpus.exists():
        pytest.skip(f"{corpus} not built")
    os.environ["API_BACKEND"] = "reference"
    os.environ["API_CONFIG"] = "configs/pilot.yaml"
    os.environ["API_REQUEST_LOG"] = str(REPO_ROOT / "logs" / "requests-test.db")
    with TestClient(app) as client:
        response = client.post("/predict", json={"ingredients": ["chicken", "rice"], "top_k": 3})
    assert response.status_code == 200
    assert response.json()["results"]
    for name in ("API_BACKEND", "API_CONFIG", "API_REQUEST_LOG"):
        os.environ.pop(name, None)


def test_the_api_serves_a_dense_bundle(tmp_path, monkeypatch):
    """The same app, with the MiniLM bundle as the backend (the encoder is faked)."""
    import numpy as np
    from omegaconf import OmegaConf

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

    profile_dir = tmp_path / "api"
    profile_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        profile_dir / "corpus.jsonl",
        [
            {
                "recipe_id": "r1",
                "title": "One",
                "directions": [],
                "link": "https://example.com/r1",
                "ingredients_normalized": ["rice"],
            },
            {
                "recipe_id": "r2",
                "title": "Two",
                "directions": [],
                "link": "https://example.com/r2",
                "ingredients_normalized": ["rice", "salt"],
            },
        ],
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"data:\n  profile: api\npaths:\n  processed_dir: {tmp_path}\n", encoding="utf-8"
    )

    os.environ["API_BACKEND"] = "dense"
    os.environ["API_BUNDLE"] = str(bundle)
    os.environ["API_CONFIG"] = str(config)
    os.environ["API_REQUEST_LOG"] = str(tmp_path / "requests.db")
    try:
        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["model"] == "minilm_pretrained"
            response = client.post("/predict", json={"ingredients": ["rice"], "top_k": 2})
            assert response.status_code == 200
            body = response.json()
            assert body["model"] == "minilm_pretrained"
            assert {result["recipe_id"] for result in body["results"]} == {"r1", "r2"}
    finally:
        for name in ("API_BACKEND", "API_BUNDLE", "API_CONFIG", "API_REQUEST_LOG"):
            os.environ.pop(name, None)
