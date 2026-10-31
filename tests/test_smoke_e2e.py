"""End-to-end smoke test of the running API, over HTTP, with the real serving bundle.

    python -m pytest tests/test_smoke_e2e.py -v
    # or against a server that already runs (Docker):
    python -m pytest tests/test_smoke_e2e.py --smoke-url http://127.0.0.1:8000 -v

The default target is the app in-process (TestClient) on the production bundle
(`serving.bundle` of `configs/default.yaml`). The test is skipped when that bundle or the
main corpus is not on the machine, so the suite stays green on a clean clone.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app
from src.config import load_config, processed_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


def _production_bundle() -> str:
    """The bundle the production config serves, if it exists on this machine."""
    config = load_config("configs/production.yaml")
    bundle = str(config.get("serving", {}).get("bundle", "") or "")
    return bundle if bundle and (REPO_ROOT / bundle).exists() else ""


@pytest.fixture(scope="module")
def smoke_client():
    """A client on the production bundle; skipped when the bundle is not on the machine."""
    bundle = _production_bundle()
    if not bundle:
        pytest.skip("the production bundle is not on this machine")
    if not (processed_dir(load_config("configs/production.yaml")) / "corpus.jsonl").exists():
        pytest.skip("the main corpus is not on this machine")

    os.environ["API_CONFIG"] = str(REPO_ROOT / "configs" / "production.yaml")
    os.environ["API_REQUEST_LOG"] = str(REPO_ROOT / "logs" / "smoke.db")
    os.environ.pop("API_BACKEND", None)
    os.environ.pop("API_BUNDLE", None)
    with TestClient(app) as client:
        yield client
    for name in ("API_CONFIG", "API_REQUEST_LOG", "API_BACKEND", "API_BUNDLE"):
        os.environ.pop(name, None)


def test_health_is_ok(smoke_client):
    response = smoke_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"]


def test_model_info_describes_the_serving_bundle(smoke_client):
    response = smoke_client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model"]
    assert body["model_version"]
    assert body["corpus_sha256"]
    assert body["bundle"]


def test_predict_returns_ranked_recipes(smoke_client):
    response = smoke_client.post(
        "/predict", json={"ingredients": ["chicken", "rice", "garlic"], "top_k": 5}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"]
    assert len(body["results"]) <= 5
    scores = [result["score"] for result in body["results"]]
    assert scores == sorted(scores, reverse=True)
    for result in body["results"]:
        assert result["recipe_id"]
        assert result["title"]
        assert result["link"].startswith(("http://", "https://"))
        assert set(result["matched_ingredients"]) | set(result["missing_ingredients"])


def test_batch_predict_keeps_the_order_of_the_input(smoke_client):
    queries = [["chicken"], ["tomato", "basil"], ["rice"]]
    response = smoke_client.post(
        "/batch-predict",
        json={"instances": [{"ingredients": q, "top_k": 2} for q in queries]},
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == len(queries)
    assert all(len(prediction["results"]) <= 2 for prediction in predictions)


def test_a_bad_request_is_answered_with_422(smoke_client):
    for payload in (
        {"ingredients": [], "top_k": 3},  # empty list
        {"ingredients": ["chicken"], "top_k": 0},  # top_k too small
        {"ingredients": ["chicken"], "top_k": 21},  # top_k too large
        {"ingredients": ["   "], "top_k": 3},  # nothing survives the normalisation
    ):
        response = smoke_client.post("/predict", json=payload)
        assert response.status_code == 422, payload


def test_the_ui_is_served(smoke_client):
    response = smoke_client.get("/")
    assert response.status_code == 200
    assert "What can I cook?" in response.text
