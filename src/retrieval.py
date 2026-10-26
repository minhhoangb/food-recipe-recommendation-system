"""Dense retriever: cosine similarity over precomputed MiniLM embeddings.

    from src.retrieval import DenseRetriever
    retriever = DenseRetriever.from_bundle("models/minilm_pretrained-E04-pilot-<sha>")
    retriever.search(["chicken", "rice"], 5)

The corpus is never re-encoded: `from_bundle` reads `embeddings.npy` and `recipe_ids.json`
and only the *query* goes through the model. Both sides are L2-normalised, so the dot
product is the cosine similarity and `sentence_transformers.util.semantic_search` ranks by
that cosine.

`semantic_search` returns its own top-k, which can drop candidates that are tied at the
cut-off. The retriever therefore asks for more candidates than it needs (a small multiple
of top_k, capped by the corpus size) and re-sorts them with the contract order
(score descending, ties by recipe_id ascending).

A query has no title, so it is always encoded as an ingredient list. With an E05 bundle
(title + ingredients) the query text is therefore not formatted like the documents; see
`docs/HUNG_IMPLEMENTATION_PLAN.md` for that known limitation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.contracts import SearchHit
from src.embeddings import (
    EMBEDDING_DIM,
    MODEL_NAME,
    BundleError,
    format_query,
    load_model,
    validate_bundle,
)

CANDIDATE_FACTOR = 4


class DenseRetriever:
    """`Retriever` implementation on top of a stored embedding bundle."""

    name = "minilm_pretrained"
    version = "1"

    def __init__(
        self,
        model: Any,
        vectors: np.ndarray,
        recipe_ids: list[str],
        *,
        embedding_dim: int | None = None,
    ) -> None:
        self._model = model
        self._vectors = vectors
        self._recipe_ids = recipe_ids
        self._ids = np.array(recipe_ids)
        self._embedding_dim = embedding_dim or int(vectors.shape[1])

    # -- construction -------------------------------------------------------------------
    @classmethod
    def from_bundle(cls, bundle: str | Path, model: Any | None = None) -> DenseRetriever:
        """Load a validated bundle. The corpus is read from disk, never re-encoded."""
        manifest, vectors, recipe_ids = validate_bundle(bundle)
        if manifest.embedding_dim is not None and vectors.shape[1] != manifest.embedding_dim:
            raise BundleError(
                f"embedding dimension {vectors.shape[1]} differs from the manifest "
                f"({manifest.embedding_dim})"
            )
        if manifest.normalized_vectors is False:
            raise BundleError("bundle vectors are not normalised; rebuild the bundle")
        encoder = model if model is not None else load_model()
        retriever = cls(encoder, vectors, recipe_ids, embedding_dim=int(vectors.shape[1]))
        retriever.name = manifest.model_name
        return retriever

    @classmethod
    def cold_load(cls, bundle: str | Path, model: Any | None = None) -> tuple[DenseRetriever, float]:
        """Load a bundle in a fresh retriever and report the wall-clock time in seconds."""
        started = time.perf_counter()
        retriever = cls.from_bundle(bundle, model=model)
        return retriever, time.perf_counter() - started

    # -- search ------------------------------------------------------------------------
    def encode_query(self, ingredients: list[str]) -> np.ndarray:
        """Encode one query into a normalised vector."""
        vector = np.asarray(
            self._model.encode(
                [format_query(ingredients)],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        if vector.shape[0] != 1:
            raise BundleError(f"encoder returned {vector.shape} for one query")
        if vector.shape[1] != self._embedding_dim:
            raise BundleError(
                f"query vector has dimension {vector.shape[1]}, corpus has {self._embedding_dim}"
            )
        return vector[0]

    def search(self, ingredients: list[str], top_k: int) -> list[SearchHit]:
        if not ingredients or top_k < 1:
            return []
        query = self.encode_query(ingredients)
        wanted = min(max(top_k * CANDIDATE_FACTOR, top_k), len(self._ids))

        # cosine search over the stored vectors (both sides are L2-normalised)
        hits = semantic_search(
            query_embeddings=torch.from_numpy(query.reshape(1, -1)),
            corpus_embeddings=torch.from_numpy(self._vectors),
            top_k=wanted,
        )[0]
        scored = [(float(hit["score"]), str(self._ids[int(hit["corpus_id"])])) for hit in hits]

        # semantic_search keeps only `wanted` candidates, so a recipe tied with the last kept
        # one can be missing. The boundary score is looked up in the full score vector (a
        # matrix product, not a full sort) and every tied recipe is added.
        boundary = sorted((score for score, _ in scored), reverse=True)[min(top_k, len(scored)) - 1]
        scores = self._vectors @ query
        tied = np.flatnonzero(np.isclose(scores, boundary, rtol=0, atol=1e-6))
        known = {recipe_id for _, recipe_id in scored}
        for index in tied:
            recipe_id = str(self._ids[index])
            if recipe_id not in known:
                scored.append((float(scores[index]), recipe_id))
                known.add(recipe_id)

        # contract order: score descending, ties by recipe_id ascending
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [SearchHit(recipe_id, score) for score, recipe_id in scored[:top_k]]


def semantic_search(*args: Any, **kwargs: Any) -> list[list[dict[str, Any]]]:
    """Thin wrapper around `sentence_transformers.util.semantic_search` (imported lazily)."""
    from sentence_transformers.util import semantic_search as _semantic_search

    return _semantic_search(*args, **kwargs)


__all__ = [
    "CANDIDATE_FACTOR",
    "BundleError",
    "DenseRetriever",
    "EMBEDDING_DIM",
    "MODEL_NAME",
    "semantic_search",
]
