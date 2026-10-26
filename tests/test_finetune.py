from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.contracts import BundleManifest
from src.embeddings import save_bundle, validate_bundle
from src.finetune import finetune_model, prepare_examples
from src.retrieval import DenseRetriever


def test_prepare_examples_pairs_and_triplets():
    pairs = [
        {"query": "chicken", "positive": "chicken soup"},
        {"query": "beef", "positive": "beef stew", "negative": "ice cream"},
    ]
    examples = prepare_examples(pairs)
    assert len(examples) == 2
    assert examples[0].texts == ["chicken", "chicken soup"]
    assert examples[1].texts == ["beef", "beef stew", "ice cream"]


def test_finetune_model_empty_pairs_raises_error():
    with pytest.raises(ValueError, match="pairs list must not be empty"):
        finetune_model([])


def test_finetuned_bundle_creation_and_contract(tmp_path: Path):
    from src.config import load_config
    cfg = load_config("configs/pilot.yaml")

    dim = 8
    n_docs = 4
    # Create normalized vectors
    raw = np.ones((n_docs, dim), dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vectors = raw / norms
    recipe_ids = [f"recipenlg:{i}" for i in range(n_docs)]

    bundle = save_bundle(
        vectors,
        recipe_ids,
        mode="ingredients",
        profile="pilot",
        cfg=cfg,
        corpus_sha256="abc123def456",
        truncated=0,
        encode_seconds=0.05,
        model_name="minilm_finetuned",
        out_dir=tmp_path,
    )

    manifest, loaded_vecs, loaded_ids = validate_bundle(bundle)
    assert isinstance(manifest, BundleManifest)
    assert manifest.model_name == "minilm_finetuned"
    assert manifest.embedding_dim == dim
    assert loaded_ids == recipe_ids
    assert np.allclose(loaded_vecs, vectors)

    # Test retriever reflects model name
    class MockEncoder:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), dim), dtype=np.float32) / np.sqrt(dim)

    retriever = DenseRetriever.from_bundle(bundle, model=MockEncoder())
    assert retriever.name == "minilm_finetuned"
    hits = retriever.search(["chicken"], 2)
    assert len(hits) == 2
    assert hits[0].recipe_id == "recipenlg:0"
