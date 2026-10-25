"""Lexical baselines as Retrievers: TF-IDF, BM25 (rank-bm25) and LSA (TruncatedSVD on TF-IDF).

    python -m src.baseline --model tfidf --config configs/pilot.yaml

TF-IDF and LSA are fitted on train recipes only and then applied to the whole candidate corpus.
BM25 (rank-bm25) computes IDF over the corpus it indexes, so it indexes all candidates
(transductive; recorded in metrics.json). Each run writes models/<run_id>/ with manifest.json,
model.joblib and metrics.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from omegaconf import DictConfig
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import linear_kernel

from src.config import fingerprint, load_config, processed_dir
from src.contracts import BundleManifest, Retriever, SearchHit
from src.normalize import NORMALIZATION_FINGERPRINT, normalize_ingredients
from src.pipeline import build_lsa_pipeline, build_text_pipeline
from src.utils import REPO_ROOT, read_json, read_jsonl, sha256_file, sha256_obj, write_json

MODELS = ("tfidf", "bm25", "lsa")
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


class BundleError(RuntimeError):
    """The bundle on disk does not match its manifest or the current code/data."""


def rank_top_k(ids: np.ndarray, scores: np.ndarray, top_k: int) -> list[SearchHit]:
    """Contract ordering: score descending, ties by recipe_id ascending; zero-overlap -> no hits."""
    order = np.lexsort((ids, -scores))[:top_k]
    return [SearchHit(str(ids[i]), float(scores[i])) for i in order if scores[i] > 0]


class VectorRetriever:
    """Cosine similarity between a query vector and precomputed document vectors (TF-IDF or LSA)."""

    version = "1"

    def __init__(self, name: str, pipeline, ids: np.ndarray, matrix):
        self.name, self.pipeline, self.ids, self.matrix = name, pipeline, ids, matrix

    def search(self, ingredients: list[str], top_k: int) -> list[SearchHit]:
        if not ingredients or top_k < 1:
            return []
        scores = linear_kernel(self.pipeline.transform([ingredients]), self.matrix).ravel()
        return rank_top_k(self.ids, scores, top_k)


class Bm25Retriever:
    name = "bm25"
    version = "1"

    def __init__(self, ids: np.ndarray, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.ids = ids
        self.index = BM25Okapi(docs, k1=k1, b=b)

    def search(self, ingredients: list[str], top_k: int) -> list[SearchHit]:
        query = normalize_ingredients(ingredients)
        if not query or top_k < 1:
            return []
        return rank_top_k(self.ids, self.index.get_scores(query), top_k)


def fit_retriever(model: str, corpus: list[dict], train_ids: set[str]) -> tuple[Retriever, dict]:
    """Fit on train, index every recipe in `corpus`. Returns (retriever, metrics.json content)."""
    corpus = sorted(corpus, key=lambda r: r["recipe_id"])
    ids = np.array([r["recipe_id"] for r in corpus])
    docs = [r["ingredients_normalized"] for r in corpus]
    train_docs = [d for r, d in zip(corpus, docs, strict=True) if r["recipe_id"] in train_ids]
    started = time.perf_counter()

    if model == "bm25":
        retriever: Retriever = Bm25Retriever(ids, [normalize_ingredients(d) for d in docs])
        info = {"idf_corpus": "all candidate recipes (transductive)"}
    else:
        if model == "tfidf":
            pipeline = build_text_pipeline()
        else:
            vocabulary = len({ingredient for doc in train_docs for ingredient in doc})
            pipeline = build_lsa_pipeline(n_components=min(256, max(1, vocabulary - 1)))
        pipeline.fit(train_docs)
        retriever = VectorRetriever(model, pipeline, ids, pipeline.transform(docs))
        info = {"fit_on": "train recipes only", "vocabulary_size": len(pipeline["tfidf"].vocabulary_)}

    info.update(
        fit_seconds=round(time.perf_counter() - started, 3), n_docs=len(docs), n_train_docs=len(train_docs)
    )
    return retriever, info


def save_bundle(
    retriever: Retriever, info: dict, cfg: DictConfig, out_dir: Path = DEFAULT_MODELS_DIR
) -> Path:
    directory = processed_dir(cfg)
    corpus_sha = sha256_file(directory / "corpus.jsonl")
    run_id = f"{retriever.name}-{cfg.data.profile}-{corpus_sha[:8]}"
    bundle = Path(out_dir) / run_id
    bundle.mkdir(parents=True, exist_ok=True)
    joblib.dump(retriever, bundle / "model.joblib")
    manifest = BundleManifest(
        run_id=run_id,
        model_name=retriever.name,  # type: ignore[arg-type]
        corpus_sha256=corpus_sha,
        recipe_ids_sha256=sha256_obj([str(i) for i in retriever.ids]),  # type: ignore[attr-defined]
        normalization_fingerprint=NORMALIZATION_FINGERPRINT,
        config_sha256=fingerprint(cfg),
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        library_versions={name: version(name) for name in ("scikit-learn", "rank-bm25", "numpy")},
        files={"model.joblib": sha256_file(bundle / "model.joblib")},
    )
    write_json(bundle / "manifest.json", manifest.model_dump())
    write_json(bundle / "metrics.json", info)
    return bundle


def load_bundle(path: str | Path) -> tuple[Retriever, BundleManifest]:
    """Load model.joblib after checking hashes; refuse a bundle built with another normaliser."""
    bundle = Path(path)
    manifest = BundleManifest(**read_json(bundle / "manifest.json"))
    for name, expected in manifest.files.items():
        if sha256_file(bundle / name) != expected:
            raise BundleError(f"{name} does not match the sha256 in manifest.json")
    if manifest.normalization_fingerprint != NORMALIZATION_FINGERPRINT:
        raise BundleError("bundle was built with a different ingredient normalisation; rebuild it")
    retriever = joblib.load(bundle / "model.joblib")
    if sha256_obj([str(i) for i in retriever.ids]) != manifest.recipe_ids_sha256:
        raise BundleError("recipe id order differs from the manifest")
    return retriever, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument(
        "--out", default=str(DEFAULT_MODELS_DIR), help="folder that receives models/<run_id>/"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    directory = processed_dir(cfg)
    corpus = list(read_jsonl(directory / "corpus.jsonl"))
    train_ids = set(read_json(directory / "splits.json")["train"])
    retriever, info = fit_retriever(args.model, corpus, train_ids)
    bundle = save_bundle(retriever, info, cfg, Path(args.out))
    print(f"{args.model}: {info}\nbundle: {bundle}")
    return 0


if __name__ == "__main__":
    # Re-import so pickled classes are stored as src.baseline.*, not __main__.* (loadable elsewhere).
    from src.baseline import main as _main

    sys.exit(_main())
