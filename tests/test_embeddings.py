"""Tests for the embedding pipeline. No model download: a fake encoder stands in.

The fake encoder produces deterministic vectors that are already normalised, so the tests
check the pipeline (formatting, batching, normalisation, alignment, truncation, bundle
files) and not the pretrained weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.contracts import BundleManifest, RecipeRecord
from src.embeddings import (
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    BundleError,
    count_truncated,
    encode_corpus,
    format_document,
    format_query,
    load_corpus,
    save_bundle,
    validate_bundle,
)


class FakeTokenizer:
    """Whitespace tokenizer; one 'token' per word."""

    def __call__(self, texts, add_special_tokens=True, truncation=False):
        return {"input_ids": [text.split() for text in texts]}


class FakeEncoder:
    """Deterministic unit vectors; the text characters are padded/truncated to `dim`.

    The real SentenceTransformer batches internally, so the fake does the same: it splits
    the sentences into chunks of `batch_size` and records one call per chunk.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.tokenizer = FakeTokenizer()
        self.calls: list[list[str]] = []

    def encode(self, sentences, **kwargs):
        batch_size = kwargs.get("batch_size") or len(sentences)
        vectors = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for start in range(0, len(sentences), batch_size):
            chunk = list(sentences[start : start + batch_size])
            self.calls.append(chunk)
            for offset, text in enumerate(chunk):
                codes = [ord(c) for c in text[: self.dim]]
                codes += [0] * (self.dim - len(codes))
                vector = np.array(codes, dtype=np.float32)
                norm = np.linalg.norm(vector)
                vectors[start + offset] = vector / norm if norm else vector
        return vectors


def make_record(index: int, title: str = "Garlic Chicken", ingredients=("garlic", "rice")):
    return RecipeRecord(
        recipe_id=f"recipenlg:{index}",
        group_id=f"group:{index}",
        title=title,
        ingredients_raw=list(ingredients),
        ingredients_normalized=list(ingredients),
        directions=["Cook."],
        link="https://example.com/r",
        source="Gathered",
        ner_raw=list(ingredients),
    )


# -- formatting -------------------------------------------------------------------------------
def test_ingredients_only_format_has_no_title():
    record = make_record(1, title="Garlic Chicken", ingredients=("garlic", "rice"))
    assert format_document(record, "ingredients") == "garlic, rice"


def test_title_plus_ingredients_format_contains_both():
    record = make_record(1, title="Garlic Chicken", ingredients=("garlic", "rice"))
    text = format_document(record, "title_ingredients")
    assert "Garlic Chicken" in text
    assert "garlic" in text and "rice" in text


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        format_document(make_record(1), "ingredients_only")


def test_query_format_matches_ingredient_documents():
    assert format_query(["garlic", "rice"]) == format_document(
        make_record(1, ingredients=("garlic", "rice")), "ingredients"
    )


# -- truncation ------------------------------------------------------------------------------
def test_truncated_documents_are_counted():
    tokenizer = FakeTokenizer()
    texts = ["a b c", "w " * (MAX_SEQ_LENGTH + 5)]
    assert count_truncated(texts, tokenizer, MAX_SEQ_LENGTH) == 1


def test_no_truncation_when_all_documents_are_short():
    assert count_truncated(["a b", "c"], FakeTokenizer(), MAX_SEQ_LENGTH) == 0


# -- encoding --------------------------------------------------------------------------------
def test_embeddings_have_the_right_shape_dtype_and_are_normalised():
    records = [make_record(i) for i in range(5)]
    encoder = FakeEncoder()
    vectors, ids, truncated = encode_corpus(encoder, records, "ingredients")
    assert vectors.shape == (5, 8)
    assert vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert truncated == 0
    assert ids == [record.recipe_id for record in records]


def test_encoding_is_done_in_batches():
    records = [make_record(i) for i in range(7)]
    encoder = FakeEncoder()
    vectors, ids, _ = encode_corpus(encoder, records, "ingredients", batch_size=3)
    # one call per batch (the fake encoder records every call)
    assert len(encoder.calls) == 3
    assert [len(call) for call in encoder.calls] == [3, 3, 1]
    assert sum(len(call) for call in encoder.calls) == 7
    assert vectors.shape[0] == len(ids)


def test_row_order_matches_recipe_id_order():
    records = [make_record(i) for i in (9, 3, 7)]
    encoder = FakeEncoder()
    vectors, ids, _ = encode_corpus(encoder, records, "title_ingredients")
    assert ids == ["recipenlg:9", "recipenlg:3", "recipenlg:7"]
    # the vector of a row is the vector of that record's text
    expected = FakeEncoder().encode([format_document(r, "title_ingredients") for r in records])
    assert np.allclose(vectors, expected)


def test_encoder_shape_mismatch_is_reported():
    class Broken:
        def encode(self, sentences, **kwargs):
            return np.zeros((len(sentences) + 1, 4), dtype=np.float32)

        tokenizer = FakeTokenizer()

    with pytest.raises(RuntimeError, match="shape"):
        encode_corpus(Broken(), [make_record(0)], "ingredients")


# -- bundle ----------------------------------------------------------------------------------
def test_bundle_is_written_and_validates(tmp_path: Path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"seed": 42, "data": {"profile": "pilot"}})
    records = [make_record(i) for i in range(4)]
    encoder = FakeEncoder()
    vectors, ids, truncated = encode_corpus(encoder, records, "ingredients")

    bundle = save_bundle(
        vectors,
        ids,
        mode="ingredients",
        profile="pilot",
        cfg=cfg,
        corpus_sha256="deadbeef",
        truncated=2,
        encode_seconds=1.25,
        out_dir=tmp_path,
    )
    assert (bundle / "embeddings.npy").exists()
    assert (bundle / "recipe_ids.json").exists()
    assert (bundle / "manifest.json").exists()

    manifest, loaded_vectors, loaded_ids = validate_bundle(bundle)
    assert isinstance(manifest, BundleManifest)
    assert manifest.model_name == "minilm_pretrained"
    assert manifest.embedding_dim == 8
    assert manifest.normalized_vectors is True
    assert loaded_ids == ids
    assert np.allclose(loaded_vectors, vectors)

    payload = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert payload["embedding_mode"] == "ingredients"
    assert payload["max_seq_length"] == MAX_SEQ_LENGTH
    assert payload["n_truncated"] == 2
    assert payload["encoder_model"] == MODEL_NAME


def test_bundle_detects_a_tampered_embedding_file(tmp_path: Path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"seed": 42, "data": {"profile": "pilot"}})
    vectors = np.eye(3, dtype=np.float32)
    bundle = save_bundle(
        vectors,
        ["a", "b", "c"],
        mode="ingredients",
        profile="pilot",
        cfg=cfg,
        corpus_sha256="deadbeef",
        truncated=0,
        encode_seconds=1.0,
        out_dir=tmp_path,
    )
    with open(bundle / "embeddings.npy", "ab") as handle:
        handle.write(b"garbage")
    with pytest.raises(BundleError, match="sha256"):
        validate_bundle(bundle)


def test_bundle_detects_a_recipe_id_mismatch(tmp_path: Path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"seed": 42, "data": {"profile": "pilot"}})
    bundle = save_bundle(
        np.eye(2, dtype=np.float32),
        ["a", "b"],
        mode="ingredients",
        profile="pilot",
        cfg=cfg,
        corpus_sha256="deadbeef",
        truncated=0,
        encode_seconds=1.0,
        out_dir=tmp_path,
    )
    (bundle / "recipe_ids.json").write_text('["b", "a"]', encoding="utf-8")
    with pytest.raises(BundleError, match="recipe id order"):
        validate_bundle(bundle)


def test_bundle_detects_a_missing_file(tmp_path: Path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"seed": 42, "data": {"profile": "pilot"}})
    bundle = save_bundle(
        np.eye(2, dtype=np.float32),
        ["a", "b"],
        mode="ingredients",
        profile="pilot",
        cfg=cfg,
        corpus_sha256="deadbeef",
        truncated=0,
        encode_seconds=1.0,
        out_dir=tmp_path,
    )
    (bundle / "embeddings.npy").unlink()
    with pytest.raises(BundleError, match="missing"):
        validate_bundle(bundle)


def test_bundle_detects_a_malformed_manifest(tmp_path: Path):
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(BundleError, match="valid JSON"):
        validate_bundle(directory)


def test_bundle_detects_a_dimension_mismatch(tmp_path: Path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"seed": 42, "data": {"profile": "pilot"}})
    bundle = save_bundle(
        np.eye(2, dtype=np.float32),
        ["a", "b"],
        mode="ingredients",
        profile="pilot",
        cfg=cfg,
        corpus_sha256="deadbeef",
        truncated=0,
        encode_seconds=1.0,
        out_dir=tmp_path,
    )
    payload = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    payload["embedding_dim"] = 99
    (bundle / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BundleError, match="dimension"):
        validate_bundle(bundle)


# -- corpus ----------------------------------------------------------------------------------
def test_load_corpus_rejects_duplicate_ids(tmp_path: Path):
    directory = tmp_path / "pilot"
    directory.mkdir()
    lines = [
        json.dumps(make_record(1).model_dump()),
        json.dumps(make_record(1).model_dump()),
    ]
    (directory / "corpus.jsonl").write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(directory)


def test_load_corpus_reports_invalid_records(tmp_path: Path):
    directory = tmp_path / "pilot"
    directory.mkdir()
    (directory / "corpus.jsonl").write_text(json.dumps({"recipe_id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid record"):
        load_corpus(directory)


# -- build (CLI body) ------------------------------------------------------------------------
def test_build_writes_a_bundle(tmp_path: Path, monkeypatch):
    """The CLI body, with the model replaced by the fake encoder (no download)."""
    from omegaconf import OmegaConf

    from src import embeddings as module

    directory = tmp_path / "pilot"
    directory.mkdir()
    records = [make_record(i) for i in range(3)]
    (directory / "corpus.jsonl").write_text(
        "\n".join(json.dumps(r.model_dump()) for r in records), encoding="utf-8"
    )
    cfg = OmegaConf.create(
        {"seed": 42, "data": {"profile": "pilot"}, "paths": {"processed_dir": str(tmp_path)}}
    )
    monkeypatch.setattr(module, "load_model", lambda device="cpu": FakeEncoder())
    monkeypatch.setattr(
        module, "processed_dir", lambda cfg, profile=None: directory
    )

    report = module.build(cfg, "title_ingredients", out_dir=tmp_path / "models")
    assert report["documents"] == 3
    assert report["dim"] == 8
    assert report["truncated"] == 0
    assert report["encode_seconds"] >= 0
    manifest, vectors, ids = validate_bundle(report["bundle"])
    assert ids == [r.recipe_id for r in records]
    assert vectors.shape == (3, 8)


def test_build_reports_a_missing_corpus(tmp_path: Path):
    from omegaconf import OmegaConf

    from src import embeddings as module

    cfg = OmegaConf.create(
        {"seed": 42, "data": {"profile": "pilot"}, "paths": {"processed_dir": str(tmp_path)}}
    )
    with pytest.raises(FileNotFoundError, match="corpus.jsonl"):
        module.build(cfg, "ingredients", out_dir=tmp_path / "models")
