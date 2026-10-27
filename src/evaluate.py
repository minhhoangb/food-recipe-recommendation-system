"""Evaluate a bundle on one split with ranx and log the run to MLflow.

    python -m src.evaluate --bundle models/tfidf-pilot-xxxx --split validation --config configs/pilot.yaml

Metrics (ranx): Recall@5, Recall@10, MRR@10 against the synthetic qrels (origin-recipe proxy).
`test` is refused unless --allow-test is given: read it once, after the configuration is frozen.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import mlflow
from ranx import Qrels, Run, evaluate

from src.baseline import load_bundle
from src.config import fingerprint, load_config, processed_dir
from src.contracts import Retriever
from src.utils import read_json, read_jsonl, sha256_file

METRICS = ("recall@5", "recall@10", "mrr@10")
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "recipe-retrieval"


def evaluate_retriever(
    retriever: Retriever,
    queries: Sequence[dict],
    qrels: dict[str, dict[str, int]],
    top_k: int = 10,
    metrics: Sequence[str] = METRICS,
) -> dict[str, float]:
    """Run every query and score with ranx. Returns metrics plus n_queries and mean latency (ms).

    Hits are converted to rank-based scores so ranx keeps the retriever's own ordering, including
    the recipe_id tie-break; a query with no hits is an empty result (scored 0, not skipped).
    """
    run: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    for query in queries:
        hits = retriever.search(query["ingredients"], top_k)
        run[query["query_id"]] = {hit.recipe_id: float(len(hits) - rank) for rank, hit in enumerate(hits)}
    latency_ms = 1000 * (time.perf_counter() - started) / max(len(queries), 1)

    if not any(run.values()):
        return {**{m: 0.0 for m in metrics}, "n_queries": len(queries), "latency_ms": latency_ms}

    scores = evaluate(Qrels(qrels), Run(run), list(metrics), make_comparable=True)
    scores = scores if isinstance(scores, dict) else {metrics[0]: scores}
    return {**{k: float(v) for k, v in scores.items()}, "n_queries": len(queries), "latency_ms": latency_ms}


def load_split(directory: Path, split: str) -> tuple[list[dict], dict[str, dict[str, int]]]:
    queries = [q for q in read_jsonl(directory / "queries.jsonl") if q["split"] == split]
    wanted = {q["query_id"] for q in queries}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in read_jsonl(directory / "qrels.jsonl"):
        if row["query_id"] in wanted:
            qrels[row["query_id"]][row["recipe_id"]] = row["relevance"]
    return queries, dict(qrels)


def log_to_mlflow(
    results: dict, manifest, bundle: Path, cfg, split: str, top_k: int, uri: str, experiment: str
) -> str:
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    fit_info = read_json(bundle / "metrics.json") if (bundle / "metrics.json").exists() else {}
    with mlflow.start_run(run_name=f"{manifest.model_name}-{cfg.data.profile}-{split}") as run:
        mlflow.log_params(
            {
                "model": manifest.model_name,
                "bundle": manifest.run_id,
                "profile": cfg.data.profile,
                "split": split,
                "top_k": top_k,
                "corpus_sha256": manifest.corpus_sha256[:12],
                "config_sha256": fingerprint(cfg)[:12],
                "seed": cfg.seed,
            }
        )
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in results.items()})  # MLflow forbids "@"
        if "fit_seconds" in fit_info:
            mlflow.log_metric("fit_seconds", fit_info["fit_seconds"])
        mlflow.set_tags({k: str(v) for k, v in fit_info.items() if k != "fit_seconds"})
        return run.info.run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--bundle", required=True, help="models/<run_id>/ written by src.baseline")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--allow-test", action="store_true", help="needed for --split test (frozen config only)"
    )
    args = parser.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        parser.error("--split test needs --allow-test: read the test split once, after freezing the config")

    cfg = load_config(args.config)
    directory = processed_dir(cfg)
    bundle = Path(args.bundle)
    retriever, manifest = load_bundle(bundle)
    if manifest.corpus_sha256 != sha256_file(directory / "corpus.jsonl"):
        parser.error("bundle was built from a different corpus than the configured profile")

    queries, qrels = load_split(directory, args.split)
    results = evaluate_retriever(retriever, queries, qrels, args.top_k)
    print(f"{manifest.model_name} | {cfg.data.profile} | {args.split} | {int(results['n_queries'])} queries")
    for name in METRICS:
        print(f"  {name:<10} {results[name]:.4f}")
    print(f"  latency    {results['latency_ms']:.2f} ms/query")
    if not args.no_mlflow:
        run_id = log_to_mlflow(
            results, manifest, bundle, cfg, args.split, args.top_k, args.tracking_uri, args.experiment
        )
        print(f"mlflow run: {run_id} ({args.tracking_uri})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
