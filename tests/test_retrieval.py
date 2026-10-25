"""Tests for the dense retriever. The encoder is a fake, so no model download happens.

The bundle is built by `src.embeddings.save_bundle` with fake vectors; the retriever must
load it, encode only the query and rank by cosine similarity with the contract ordering.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.embeddings import BundleError, save_bundle
from src.retrieval import DenseRetriever
from tests.contract_utils import assert_retriever_contract


class FakeEncoder:
    """Maps a text to a deterministic unit vector; records every call."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, sentences, **kwargs):
        self.calls.append(list(sentences))
        vectors = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for row, text in enumerate(sentences):
            codes = [ord(c) for c in text[: self.dim]]
            codes += [0] * (self.dim - len(codes))
            vector = np.array(codes, dtype=np.float32)
            norm = np.linalg.norm(vector)
            vectors[row] = vector / norm if norm else vector
        return vectors


@pytest.fixture
def bundle(tmp_path: Path):
    """A 6-recipe bundle with distinct, normalised vectors."""
    cfg = OmegaConf.create({"seed": 42, "data": {"profile": "pilot"}})
    ids = [f"recipenlg:{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(6, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    directory = save_bundle(
        vectors,
        ids,
        mode="ingredients",
        profile="pilot",
        cfg=cfg,
        corpus_sha256="deadbeef",
        truncated=0,
        encode_seconds=1.0,
        out_dir=tmp_path,
    )
    return directory, vectors, ids


def test_retriever_loads_the_bundle_without_encoding_the_corpus(bundle):
    directory, vectors, ids = bundle
    encoder = FakeEncoder()
    retriever = DenseRetriever.from_bundle(directory, model=encoder)
    assert encoder.calls == [], "loading must not encode the corpus"
    assert retriever._recipe_ids == ids
    assert np.allclose(retriever._vectors, vectors)


def test_search_encodes_only_the_query(bundle):
    directory, _, _ = bundle
    encoder = FakeEncoder()
    retriever = DenseRetriever.from_bundle(directory, model=encoder)
    retriever.search(["garlic"], 3)
    assert encoder.calls == [["garlic"]]


def test_query_vector_is_normalised(bundle):
    directory, _, _ = bundle
    retriever = DenseRetriever.from_bundle(directory, model=FakeEncoder())
    vector = retriever.encode_query(["garlic"])
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_search_ranks_by_cosine_similarity(bundle):
    directory, vectors, ids = bundle
    retriever = DenseRetriever.from_bundle(directory, model=FakeEncoder())
    query = vectors[2]
    retriever._model = type(
        "Fixed", (), {"encode": lambda self, sentences, **kw: query.reshape(1, -1)}
    )()
    hits = retriever.search(["garlic"], 3)
    assert hits[0].recipe_id == ids[2]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[1].score >= hits[2].score


def test_top_k_is_respected(bundle):
    directory, _, _ = bundle
    retriever = DenseRetriever.from_bundle(directory, model=FakeEncoder())
    assert len(retriever.search(["garlic"], 2)) == 2
    assert len(retriever.search(["garlic"], 20)) == 6


def test_empty_query_returns_no_hits(bundle):
    directory, _, _ = bundle
    retriever = DenseRetriever.from_bundle(directory, model=FakeEncoder())
    assert retriever.search([], 5) == []
    assert retriever.search(["garlic"], 0) == []


def test_ties_are_broken_by_recipe_id(tmp_path: Path):
    """Identical vectors -> identical scores -> the smallest recipe_id must come first."""
    cfg = OmegaConf.create({"seed": 42, "data": {"profile": "pilot"}})
    ids = ["recipenlg:3", "recipenlg:1", "recipenlg:2"]
    vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (3, 1))
    directory = save_bundle(
        vectors,
        ids,
        mode="ingredients",
        profile="pilot",
        cfg=cfg,
        corpus_sha256="deadbeef",
        truncated=0,
        encode_seconds=1.0,
        out_dir=tmp_path,
    )
    retriever = DenseRetriever.from_bundle(directory, model=FakeEncoder(dim=2))
    hits = retriever.search(["garlic"], 3)
    assert [hit.recipe_id for hit in hits] == sorted(ids)
    assert hits[0].score == pytest.approx(hits[-1].score)


def test_a_tie_at_the_top_k_boundary_is_not_lost(tmp_path: Path):
    """Four recipes with identical scores; top_k=2 must return the two smallest ids.

    semantic_search keeps an arbitrary subset of the tied candidates, so the retriever has
    to look the boundary score up in the full score vector.
    """
    cfg = OmegaConf.create({"seed": 42, "data": {"profile": "pilot"}})
    ids = ["recipenlg:5", "recipenlg:1", "recipenlg:4", "recipenlg:2"]
    vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (4, 1))
    directory = save_bundle(
        vectors,
        ids,
        mode="ingredients",
        profile="pilot",
        cfg=cfg,
        corpus_sha256="deadbeef",
        truncated=0,
        encode_seconds=1.0,
        out_dir=tmp_path,
    )
    retriever = DenseRetriever.from_bundle(directory, model=FakeEncoder(dim=2))
    hits = retriever.search(["garlic"], 2)
    assert [hit.recipe_id for hit in hits] == ["recipenlg:1", "recipenlg:2"]


def test_retriever_follows_the_contract(bundle):
    directory, _, ids = bundle
    retriever = DenseRetriever.from_bundle(directory, model=FakeEncoder())
    assert_retriever_contract(retriever, ids, [["garlic", "rice"], ["no such thing"]])


def test_a_wrong_hash_is_reported(bundle):
    directory, _, _ = bundle
    with open(directory / "embeddings.npy", "ab") as handle:
        handle.write(b"x")
    with pytest.raises(BundleError, match="sha256"):
        DenseRetriever.from_bundle(directory, model=FakeEncoder())


def test_a_dimension_mismatch_is_reported(bundle):
    directory, _, _ = bundle
    payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    payload["embedding_dim"] = 384
    (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BundleError, match="dimension"):
        DenseRetriever.from_bundle(directory, model=FakeEncoder())


def test_a_recipe_id_mismatch_is_reported(bundle):
    directory, _, _ = bundle
    (directory / "recipe_ids.json").write_text('["b", "a"]', encoding="utf-8")
    with pytest.raises(BundleError, match="recipe id order"):
        DenseRetriever.from_bundle(directory, model=FakeEncoder())


def test_a_missing_artifact_is_reported(bundle):
    directory, _, _ = bundle
    (directory / "recipe_ids.json").unlink()
    with pytest.raises(BundleError, match="missing"):
        DenseRetriever.from_bundle(directory, model=FakeEncoder())


def test_a_missing_bundle_is_reported(tmp_path: Path):
    with pytest.raises(BundleError, match="manifest.json"):
        DenseRetriever.from_bundle(tmp_path / "nothing", model=FakeEncoder())


def test_a_query_with_the_wrong_dimension_is_reported(bundle):
    directory, _, _ = bundle

    class WrongDimension:
        def encode(self, sentences, **kwargs):
            return np.zeros((1, 4), dtype=np.float32)

    retriever = DenseRetriever.from_bundle(directory, model=FakeEncoder())
    retriever._model = WrongDimension()
    with pytest.raises(BundleError, match="dimension"):
        retriever.search(["garlic"], 3)


def test_cold_load_returns_the_elapsed_time(bundle):
    directory, _, _ = bundle
    retriever, seconds = DenseRetriever.cold_load(directory, model=FakeEncoder())
    assert retriever.name == "minilm_pretrained"
    assert seconds > 0
