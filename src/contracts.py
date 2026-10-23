"""Team contract v1 in code: record schemas, the Retriever interface, API and manifest models.

Every module talks to the others only through these types. Changing anything here needs a PR that
the other two members review (see CONTRIBUTING.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from src.normalize import normalize_ingredients

SCHEMA_VERSION = "1"
Split = Literal["train", "validation", "test"]
ModelName = Literal["tfidf-reference", "tfidf", "bm25", "lsa", "minilm_pretrained", "minilm_finetuned"]

MAX_INGREDIENTS = 50
MAX_INGREDIENT_CHARS = 100
MAX_TOP_K = 20
MAX_BATCH = 20


# ---------------------------------------------------------------------------------------------
# Data records (JSONL files under data/processed/<profile>/)
# ---------------------------------------------------------------------------------------------
class RecipeRecord(BaseModel):
    """One line of corpus.jsonl."""

    model_config = ConfigDict(extra="forbid")

    recipe_id: str  # "recipenlg:<raw index>", stable across runs
    group_id: str  # recipes in one group always land in the same split
    title: str
    ingredients_raw: list[str]
    ingredients_normalized: list[str]  # output of normalize_ingredients, used by every model
    directions: list[str]
    link: str  # always http(s)
    source: str
    ner_raw: list[str]


class QueryRecord(BaseModel):
    """One line of queries.jsonl."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    split: Split
    ingredients: list[str]  # already normalised
    origin_recipe_ids: list[str]
    origin_group_ids: list[str]
    label_source: Literal["synthetic", "manual"] = "synthetic"


class QrelRecord(BaseModel):
    """One line of qrels.jsonl. relevance: 0/1/2 (synthetic labels are always 1)."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    recipe_id: str
    relevance: Annotated[int, Field(ge=0, le=2)]
    label_source: Literal["synthetic_origin", "manual"] = "synthetic_origin"


# ---------------------------------------------------------------------------------------------
# Retrieval interface
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchHit:
    recipe_id: str
    score: float  # similarity/ranking score; NOT a probability and not comparable across models


@runtime_checkable
class Retriever(Protocol):
    """Every model (jaccard, tfidf, bm25, lsa, minilm_*) exposes exactly this.

    Rules (checked by tests/contract_utils.py):
      * input are already-normalised ingredients (use src.normalize.normalize_ingredients);
      * at most top_k hits, sorted by score desc, ties broken by recipe_id ascending;
      * no duplicate recipe_ids; may return fewer hits (e.g. zero lexical overlap -> []).
    """

    name: str
    version: str

    def search(self, ingredients: list[str], top_k: int) -> list[SearchHit]: ...


# ---------------------------------------------------------------------------------------------
# API schemas (FastAPI request/response)
# ---------------------------------------------------------------------------------------------
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ingredients: list[str] = Field(min_length=1, max_length=MAX_INGREDIENTS)
    top_k: StrictInt = Field(default=5, ge=1, le=MAX_TOP_K)

    @field_validator("ingredients")
    @classmethod
    def _check_items(cls, items: list[str]) -> list[str]:
        for item in items:
            if not item.strip():
                raise ValueError("ingredients must not contain blank strings")
            if len(item) > MAX_INGREDIENT_CHARS:
                raise ValueError(f"each ingredient must be at most {MAX_INGREDIENT_CHARS} chars")
        return items

    @model_validator(mode="after")
    def _non_empty_after_normalisation(self) -> PredictRequest:
        if not normalize_ingredients(self.ingredients):
            raise ValueError("no usable ingredient left after normalisation")
        return self


class RecipeResult(BaseModel):
    recipe_id: str
    title: str
    score: float
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    directions: list[str]
    link: str


class PredictResponse(BaseModel):
    model: str
    model_version: str
    latency_ms: float
    results: list[RecipeResult]
    note: str | None = None  # e.g. "no recipe shares an ingredient with the query"


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[PredictRequest] = Field(min_length=1, max_length=MAX_BATCH)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


# ---------------------------------------------------------------------------------------------
# Artifact manifests
# ---------------------------------------------------------------------------------------------
class BundleManifest(BaseModel):
    """models/<run_id>/manifest.json — what a serving bundle must declare."""

    schema_version: str = SCHEMA_VERSION
    run_id: str
    model_name: ModelName
    revision: str | None = None  # HF revision / checkpoint hash
    corpus_sha256: str
    recipe_ids_sha256: str
    normalization_fingerprint: str
    config_sha256: str
    embedding_dim: int | None = None
    embedding_dtype: str | None = None
    normalized_vectors: bool | None = None
    created_utc: str
    library_versions: dict[str, str] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)  # relative path -> sha256


class RequestLogRecord(BaseModel):
    """One API request in the request log (schema v1 shared by the API logger and monitor.py)."""

    request_id: str
    timestamp_utc: str
    endpoint: str
    model: str
    model_version: str
    ingredients: list[str]  # normalised input
    top_k: int
    result_ids: list[str]
    scores: list[float]
    latency_ms: float
    status_code: int
