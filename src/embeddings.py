"""Embed the pilot corpus with the pretrained MiniLM encoder (E04 / E05).

    python -m src.embeddings --config configs/pilot.yaml --mode ingredients
    python -m src.embeddings --config configs/pilot.yaml --mode title_ingredients

Reads `data/processed/<profile>/corpus.jsonl` and writes `models/<run_id>/` with
`embeddings.npy` (L2-normalised float32 rows, one per recipe, in corpus order),
`recipe_ids.json` (the same order) and `manifest.json` (a `BundleManifest`).

CPU only: the model is loaded with device="cpu" and torch is never asked for CUDA.
The corpus is encoded in batches with `max_seq_length=128`; documents whose token
count exceeds that limit are counted and reported (their tails are cut off by the
tokenizer). Nothing is dropped: the row order of `embeddings.npy` always matches the
order of `recipe_ids.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from omegaconf import DictConfig

from src.config import fingerprint, load_config, processed_dir
from src.contracts import BundleManifest, RecipeRecord
from src.normalize import NORMALIZATION_FINGERPRINT
from src.utils import REPO_ROOT, read_jsonl, sha256_file, sha256_obj, write_json

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_MODELS_DIR = REPO_ROOT / "models"

EmbeddingMode = Literal["ingredients", "title_ingredients"]
MODES: tuple[EmbeddingMode, ...] = ("ingredients", "title_ingredients")
MODE_LABELS = {"ingredients": "E04", "title_ingredients": "E05"}


class BundleError(RuntimeError):
    """The bundle on disk is missing, corrupted or does not match its manifest."""


class Encoder(Protocol):
    """The slice of SentenceTransformer this module uses (keeps the tests offline)."""

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray: ...

    def tokenize(self, texts: list[str]) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------------------------
# Document formatting
# ---------------------------------------------------------------------------------------------
def format_document(record: RecipeRecord, mode: EmbeddingMode) -> str:
    """Text that is embedded for one recipe.

    E04 "ingredients": the normalised ingredient names only.
    E05 "title_ingredients": the recipe title followed by the normalised ingredient names.
    """
    ingredients = ", ".join(record.ingredients_normalized)
    if mode == "ingredients":
        return ingredients
    if mode == "title_ingredients":
        return f"{record.title}. Ingredients: {ingredients}" if ingredients else record.title
    raise ValueError(f"unknown embedding mode {mode!r}; expected one of {MODES}")


def format_query(ingredients: list[str]) -> str:
    """A query is encoded exactly like an E04 document so both live in the same space."""
    return ", ".join(ingredients)


# ---------------------------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------------------------
def count_truncated(texts: list[str], tokenizer: Any, max_seq_length: int = MAX_SEQ_LENGTH) -> int:
    """How many documents are longer than `max_seq_length` tokens (they will be cut off).

    Counted on the token ids without the special tokens, exactly like the tokenizer does
    when it truncates.
    """
    if not texts:
        return 0
    lengths = [
        len(ids)
        for ids in tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
    ]
    return sum(1 for length in lengths if length > max_seq_length)


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------
def load_model(device: str = "cpu") -> Encoder:
    """Load the pretrained MiniLM on CPU (downloads the weights once)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "sentence-transformers is not installed. Run: pip install -r requirements.txt"
        ) from error

    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def library_versions() -> dict[str, str]:
    versions = {}
    for name in ("sentence-transformers", "torch", "numpy"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - depends on the environment
            versions[name] = "not installed"
    return versions


# ---------------------------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------------------------
def load_corpus(directory: Path) -> list[RecipeRecord]:
    """Read the corpus in file order; invalid records are reported, never silently dropped."""
    records: list[RecipeRecord] = []
    problems: list[str] = []
    for number, row in enumerate(read_jsonl(directory / "corpus.jsonl"), start=1):
        try:
            records.append(RecipeRecord(**row))
        except Exception as error:  # pydantic ValidationError and friends
            problems.append(f"line {number}: {error}")
    if problems:
        raise ValueError(
            f"{len(problems)} invalid record(s) in {directory / 'corpus.jsonl'}; first: {problems[0]}"
        )
    ids = [record.recipe_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("corpus contains duplicate recipe_id values")
    return records


def encode_corpus(
    model: Encoder,
    records: list[RecipeRecord],
    mode: EmbeddingMode,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> tuple[np.ndarray, list[str], int]:
    """Encode every recipe. Returns (embeddings, recipe_ids, n_truncated).

    The vectors are L2-normalised, so a dot product is a cosine similarity.
    """
    texts = [format_document(record, mode) for record in records]
    tokenizer = getattr(model, "tokenizer", None)
    truncated = count_truncated(texts, tokenizer, max_seq_length) if tokenizer is not None else 0

    vectors = np.asarray(
        model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(records):
        raise RuntimeError(f"encoder returned shape {vectors.shape} for {len(records)} documents")
    return vectors, [record.recipe_id for record in records], truncated


# ---------------------------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------------------------
def bundle_run_id(mode: EmbeddingMode, profile: str, corpus_sha256: str) -> str:
    return f"minilm_pretrained-{MODE_LABELS[mode]}-{profile}-{corpus_sha256[:8]}"


def save_bundle(
    vectors: np.ndarray,
    recipe_ids: list[str],
    *,
    mode: EmbeddingMode,
    profile: str,
    cfg: DictConfig,
    corpus_sha256: str,
    truncated: int,
    encode_seconds: float,
    out_dir: Path = DEFAULT_MODELS_DIR,
) -> Path:
    """Write embeddings.npy, recipe_ids.json and manifest.json; return the bundle folder."""
    run_id = bundle_run_id(mode, profile, corpus_sha256)
    bundle = Path(out_dir) / run_id
    bundle.mkdir(parents=True, exist_ok=True)

    np.save(bundle / "embeddings.npy", vectors)
    write_json(bundle / "recipe_ids.json", recipe_ids)

    manifest = BundleManifest(
        run_id=run_id,
        model_name="minilm_pretrained",
        revision=None,
        corpus_sha256=corpus_sha256,
        recipe_ids_sha256=sha256_obj(recipe_ids),
        normalization_fingerprint=NORMALIZATION_FINGERPRINT,
        config_sha256=fingerprint(cfg),
        embedding_dim=int(vectors.shape[1]),
        embedding_dtype=str(vectors.dtype),
        normalized_vectors=True,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        library_versions=library_versions(),
        files={
            "embeddings.npy": sha256_file(bundle / "embeddings.npy"),
            "recipe_ids.json": sha256_file(bundle / "recipe_ids.json"),
        },
    )
    payload = manifest.model_dump()
    # BundleManifest has no dedicated field for these; they stay optional and additive.
    payload["embedding_mode"] = mode
    payload["max_seq_length"] = MAX_SEQ_LENGTH
    payload["batch_size"] = int(cfg.get("embeddings", {}).get("batch_size", DEFAULT_BATCH_SIZE))
    payload["encoder_model"] = MODEL_NAME
    payload["n_documents"] = int(vectors.shape[0])
    payload["n_truncated"] = int(truncated)
    payload["encode_seconds"] = round(float(encode_seconds), 3)
    write_json(bundle / "manifest.json", payload)
    return bundle


def validate_bundle(bundle: str | Path) -> tuple[BundleManifest, np.ndarray, list[str]]:
    """Load a bundle and check every invariant the manifest claims.

    Raises `BundleError` with a message that names the broken invariant.
    """
    directory = Path(bundle)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise BundleError(f"{manifest_path} is missing")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BundleError(f"{manifest_path} is not valid JSON: {error}") from error
    try:
        manifest = BundleManifest(**payload)
    except Exception as error:
        raise BundleError(f"{manifest_path} does not match BundleManifest: {error}") from error

    ids_path = directory / "recipe_ids.json"
    vectors_path = directory / "embeddings.npy"
    for path in (ids_path, vectors_path):
        if not path.exists():
            raise BundleError(f"{path} is missing (declared in manifest.json)")

    # The id order is checked before the file hashes, so a reordered recipe_ids.json reports
    # the ordering problem instead of a generic hash mismatch.
    recipe_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if not isinstance(recipe_ids, list) or not all(isinstance(i, str) for i in recipe_ids):
        raise BundleError("recipe_ids.json must be a list of strings")
    if sha256_obj(recipe_ids) != manifest.recipe_ids_sha256:
        raise BundleError("recipe id order differs from the manifest (recipe_ids_sha256 mismatch)")

    for name, expected in manifest.files.items():
        path = directory / name
        if not path.exists():
            raise BundleError(f"{path} is missing (declared in manifest.json)")
        actual = sha256_file(path)
        if actual != expected:
            raise BundleError(f"{name} does not match the sha256 in manifest.json")

    vectors = np.load(vectors_path)
    if vectors.ndim != 2:
        raise BundleError(f"embeddings.npy must be 2-D, got shape {vectors.shape}")
    if vectors.shape[0] != len(recipe_ids):
        raise BundleError(
            f"embeddings.npy has {vectors.shape[0]} rows but recipe_ids.json has {len(recipe_ids)} ids"
        )
    if manifest.embedding_dim is not None and vectors.shape[1] != manifest.embedding_dim:
        raise BundleError(
            f"embedding dimension {vectors.shape[1]} differs from the manifest ({manifest.embedding_dim})"
        )
    if not np.isfinite(vectors).all():
        raise BundleError("embeddings.npy contains non-finite values")
    return manifest, vectors, recipe_ids


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def build(cfg: DictConfig, mode: EmbeddingMode, out_dir: Path = DEFAULT_MODELS_DIR) -> dict:
    """Encode the configured corpus and write the bundle. Returns a small report dict."""
    directory = processed_dir(cfg)
    corpus_path = directory / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found. Build or fetch the processed data first "
            "(python -m src.data_collection / python scripts/fetch_artifacts.py)."
        )
    records = load_corpus(directory)
    model = load_model()
    batch_size = int(cfg.get("embeddings", {}).get("batch_size", DEFAULT_BATCH_SIZE))

    started = time.perf_counter()
    vectors, recipe_ids, truncated = encode_corpus(model, records, mode, batch_size=batch_size)
    encode_seconds = time.perf_counter() - started

    bundle = save_bundle(
        vectors,
        recipe_ids,
        mode=mode,
        profile=cfg.data.profile,
        cfg=cfg,
        corpus_sha256=sha256_file(corpus_path),
        truncated=truncated,
        encode_seconds=encode_seconds,
        out_dir=out_dir,
    )
    return {
        "bundle": bundle,
        "mode": mode,
        "documents": len(records),
        "dim": int(vectors.shape[1]),
        "truncated": truncated,
        "encode_seconds": round(encode_seconds, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--mode", default="ingredients", choices=list(MODES))
    parser.add_argument("--out", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"documents per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args(argv)

    overrides = {"embeddings": {"batch_size": int(args.batch_size)}} if args.batch_size else None
    cfg = load_config(args.config, overrides)
    report = build(cfg, args.mode, Path(args.out))
    print(f"[{MODE_LABELS[report['mode']]}] {report['mode']}")
    print(f"  bundle          {report['bundle']}")
    print(f"  documents       {report['documents']}")
    print(f"  dimension       {report['dim']}")
    print(f"  truncated       {report['truncated']} (max_seq_length={MAX_SEQ_LENGTH})")
    print(f"  encode seconds  {report['encode_seconds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
