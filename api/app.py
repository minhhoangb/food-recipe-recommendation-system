"""FastAPI app: /health, /model-info, /predict, /batch-predict.

    uvicorn api.app:app --reload

The retriever and the corpus metadata are loaded once in the lifespan (not per
request). Two backends are supported:

  * `reference` (default): `src.reference.TfidfReference` over the processed corpus --
    no model download, works out of the box.
  * `dense`: `src.retrieval.DenseRetriever` over a MiniLM bundle under `models/`.

The backend and the paths are read from the environment so Docker and a local run
behave the same:

    API_BACKEND=reference|dense
    API_CONFIG=configs/pilot.yaml
    API_BUNDLE=models/minilm_pretrained-E04-pilot-<sha8>
    API_REQUEST_LOG=logs/requests.db
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.request_log import RequestLogger
from src.config import load_config, processed_dir
from src.contracts import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    RequestLogRecord,
)
from src.normalize import normalize_ingredients
from src.predict import load_corpus_metadata, recommend

DEFAULT_CONFIG = "configs/pilot.yaml"
DEFAULT_LOG = "logs/requests.db"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the retriever and the corpus metadata once, before the first request."""
    backend = _env("API_BACKEND", "reference")
    config_path = _env("API_CONFIG", DEFAULT_CONFIG)
    cfg = load_config(config_path)
    corpus_path = processed_dir(cfg) / "corpus.jsonl"

    if backend == "dense":
        from src.retrieval import DenseRetriever

        bundle = _env("API_BUNDLE", "")
        if not bundle:
            raise RuntimeError(
                "API_BACKEND=dense needs API_BUNDLE=<models/run_id>; "
                "build one with: python -m src.embeddings --config <config> --mode ingredients"
            )
        app.state.retriever = DenseRetriever.from_bundle(bundle)
    else:
        from src.reference import TfidfReference

        app.state.retriever = TfidfReference.from_corpus(corpus_path)

    app.state.metadata = load_corpus_metadata(corpus_path)
    app.state.backend = backend
    app.state.logger = RequestLogger(_env("API_REQUEST_LOG", DEFAULT_LOG))
    app.state.config_path = config_path
    yield


app = FastAPI(
    title="Food Recipe Recommendation API",
    version="1",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, error: RequestValidationError) -> JSONResponse:
    """422 with the reason, and the rejected request is logged like every other one."""
    logger = getattr(app.state, "logger", None)
    if logger is not None:
        body = error.body if isinstance(error.body, dict) else {}
        raw = body.get("ingredients") or []
        logger.log(
            RequestLogRecord(
                request_id=uuid.uuid4().hex,
                timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                endpoint=request.url.path,
                model=getattr(app.state.retriever, "name", "unknown"),
                model_version=getattr(app.state.retriever, "version", "unknown"),
                ingredients=[str(item) for item in raw] if isinstance(raw, list) else [],
                top_k=int(body.get("top_k", 0) or 0),
                result_ids=[],
                scores=[],
                latency_ms=0.0,
                status_code=422,
            )
        )
    return JSONResponse(status_code=422, content=jsonable_encoder({"detail": error.errors()}))


def _predict(payload: PredictRequest) -> PredictResponse:
    started = time.perf_counter()
    results = recommend(
        payload.ingredients,
        payload.top_k,
        retriever=app.state.retriever,
        metadata=app.state.metadata,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    note = None if results else "no recipe shares an ingredient with the query"
    return PredictResponse(
        model=app.state.retriever.name,
        model_version=app.state.retriever.version,
        latency_ms=latency_ms,
        results=results,
        note=note,
    )


def _log(payload: PredictRequest, response: PredictResponse, status_code: int, endpoint: str) -> None:
    app.state.logger.log(
        RequestLogRecord(
            request_id=uuid.uuid4().hex,
            timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            endpoint=endpoint,
            model=response.model,
            model_version=response.model_version,
            ingredients=normalize_ingredients(payload.ingredients),
            top_k=payload.top_k,
            result_ids=[result.recipe_id for result in response.results],
            scores=[result.score for result in response.results],
            latency_ms=response.latency_ms,
            status_code=status_code,
        )
    )


@app.get("/health")
def health() -> dict:
    """Liveness probe: the app is up and its retriever is loaded."""
    retriever = app.state.retriever
    return {"status": "ok", "model": retriever.name, "model_version": retriever.version}


@app.get("/model-info")
def model_info() -> dict:
    """Which model is serving, and where its data comes from."""
    retriever = app.state.retriever
    return {
        "model": retriever.name,
        "model_version": retriever.version,
        "backend": app.state.backend,
        "config": app.state.config_path,
        "corpus_recipes": len(app.state.metadata),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Recommend recipes for one ingredient list.

    The input is validated by the pydantic schema (`src/contracts.py`), so a bad
    request never reaches this function: FastAPI answers 422 with the reason.
    """
    response = _predict(payload)
    _log(payload, response, 200, "/predict")
    return response


@app.post("/batch-predict", response_model=BatchPredictResponse)
def batch_predict(payload: BatchPredictRequest) -> BatchPredictResponse:
    """Recommend recipes for several ingredient lists; the order of the input is kept."""
    predictions: list[PredictResponse] = []
    for item in payload.instances:
        response = _predict(item)
        _log(item, response, 200, "/batch-predict")
        predictions.append(response)
    return BatchPredictResponse(predictions=predictions)


__all__ = ["app", "lifespan"]
