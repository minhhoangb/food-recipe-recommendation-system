"""Embed the pilot corpus and evaluate E04 / E05 on the validation split.

    python -m src.evaluation --mode both --config configs/pilot.yaml --split validation

E04 embeds the ingredients only, E05 the title + ingredients; everything else (corpus,
queries, qrels, model checkpoint, retrieval settings) is identical so the two numbers are
comparable. Metrics come from `ranx` (Recall@5, Recall@10, MRR@10) with the conventions
locked in `tests/test_ranking_metrics.py`.

`src/evaluate.py` belongs to Person 1 (Hoàng). When it is merged this module delegates to
it (`evaluate_retriever`, `load_split` keep the same signatures), so the two implementations
cannot drift apart.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ranx import Qrels, Run, evaluate

from src.config import fingerprint, load_config, processed_dir
from src.contracts import BundleManifest
from src.memory import peak_rss_mb
from src.utils import read_json, read_jsonl, sha256_file

METRICS = ("recall@5", "recall@10", "mrr@10")
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "recipe-retrieval"
MODES = ("ingredients", "title_ingredients")


def _shared() -> Any | None:
    """Person 1's `src.evaluate` when it is merged; None otherwise.

    Both modules implement the same metric conventions, so the shared one is preferred as
    soon as it exists and the local copy is only the fallback until then.
    """
    try:
        import src.evaluate as shared
    except ModuleNotFoundError:
        return None
    return shared if hasattr(shared, "evaluate_retriever") else None


# ---------------------------------------------------------------------------------------------
# Metric helpers (shared with src/evaluate.py once it is merged)
# ---------------------------------------------------------------------------------------------
def evaluate_retriever(
    retriever: Any,
    queries: Sequence[dict],
    qrels: dict[str, dict[str, int]],
    top_k: int = 10,
    metrics: Sequence[str] = METRICS,
) -> dict[str, float]:
    """Run every query and score with ranx. Returns the metrics plus n_queries and latency.

    Hits become rank-based scores so ranx keeps the retriever's own ordering, including
    the recipe_id tie-break; a query with no hits is an empty result (scored 0, not skipped).
    """
    shared = _shared()
    if shared is not None:
        return shared.evaluate_retriever(retriever, queries, qrels, top_k, metrics)

    run: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    for query in queries:
        hits = retriever.search(query["ingredients"], top_k)
        run[query["query_id"]] = {
            hit.recipe_id: float(len(hits) - rank) for rank, hit in enumerate(hits)
        }
    latency_ms = 1000 * (time.perf_counter() - started) / max(len(queries), 1)

    if not any(run.values()):  # ranx cannot build a Run with no document at all
        return {
            **{name: 0.0 for name in metrics},
            "n_queries": len(queries),
            "latency_ms": latency_ms,
        }
    scores = evaluate(Qrels(qrels), Run(run), list(metrics), make_comparable=True)
    scores = scores if isinstance(scores, dict) else {metrics[0]: scores}
    return {**{k: float(v) for k, v in scores.items()}, "n_queries": len(queries), "latency_ms": latency_ms}


def load_split(directory: Path, split: str) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """The queries of one split and the matching qrels (query_id -> recipe_id -> relevance)."""
    shared = _shared()
    if shared is not None:
        return shared.load_split(directory, split)

    queries = [q for q in read_jsonl(directory / "queries.jsonl") if q["split"] == split]
    wanted = {q["query_id"] for q in queries}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in read_jsonl(directory / "qrels.jsonl"):
        if row["query_id"] in wanted:
            qrels[row["query_id"]][row["recipe_id"]] = row["relevance"]
    return queries, dict(qrels)


# ---------------------------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------------------------
def log_to_mlflow(
    results: dict,
    manifest: BundleManifest,
    bundle: Path,
    cfg: Any,
    split: str,
    top_k: int,
    uri: str,
    experiment: str,
    extra_params: dict[str, Any] | None = None,
) -> str:
    """Log one experiment run. Returns the MLflow run id."""
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    bundle_info = read_json(bundle / "manifest.json")
    params = {
        "model": manifest.model_name,
        "bundle": manifest.run_id,
        "profile": cfg.data.profile,
        "split": split,
        "top_k": top_k,
        "corpus_sha256": manifest.corpus_sha256[:12],
        "config_sha256": fingerprint(cfg)[:12],
        "seed": cfg.seed,
        "embedding_mode": bundle_info.get("embedding_mode", "unknown"),
        "encoder_model": bundle_info.get("encoder_model", "unknown"),
        "max_seq_length": bundle_info.get("max_seq_length", "unknown"),
        "batch_size": bundle_info.get("batch_size", "unknown"),
        "n_documents": bundle_info.get("n_documents", "unknown"),
        "embedding_dim": manifest.embedding_dim or "unknown",
    }
    params.update({k: str(v) for k, v in (extra_params or {}).items()})
    with mlflow.start_run(run_name=f"{manifest.model_name}-{cfg.data.profile}-{split}") as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in results.items()})
        mlflow.set_tags({"embedding_mode": str(params["embedding_mode"]), "split": split})
        return run.info.run_id


# ---------------------------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------------------------
def run_experiment(
    mode: str,
    cfg: Any,
    *,
    split: str = "validation",
    top_k: int = 10,
    models_dir: Path,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment: str = DEFAULT_EXPERIMENT,
    log: bool = True,
) -> dict:
    """Embed (if needed), retrieve and score one embedding mode. Returns a result record."""
    from src.embeddings import build, bundle_run_id
    from src.retrieval import DenseRetriever

    directory = processed_dir(cfg)
    corpus_sha = sha256_file(directory / "corpus.jsonl")
    run_id = bundle_run_id(mode, cfg.data.profile, corpus_sha)
    bundle = Path(models_dir) / run_id

    if not (bundle / "embeddings.npy").exists():
        report = build(cfg, mode, Path(models_dir))
        bundle = report["bundle"]
    bundle_info = read_json(bundle / "manifest.json")
    encode_seconds = float(bundle_info.get("encode_seconds", 0.0))

    retriever, cold_load_seconds = DenseRetriever.cold_load(bundle)
    queries, qrels = load_split(directory, split)
    results = evaluate_retriever(retriever, queries, qrels, top_k)

    manifest = BundleManifest(**bundle_info)
    record = {
        "mode": mode,
        "bundle": bundle,
        "run_id": run_id,
        "metrics": results,
        "encode_seconds": encode_seconds,
        "cold_load_seconds": round(cold_load_seconds, 3),
        "peak_rss_mb": peak_rss_mb(),
        "split": split,
    }
    if log:
        record["mlflow_run_id"] = log_to_mlflow(
            results,
            manifest,
            bundle,
            cfg,
            split,
            top_k,
            tracking_uri,
            experiment,
            extra_params={
                "encode_seconds": round(encode_seconds, 3),
                "cold_load_seconds": round(cold_load_seconds, 3),
                "peak_rss_mb": record["peak_rss_mb"],
                "n_truncated": bundle_info.get("n_truncated", "unknown"),
                "experiment": f"E{4 if mode == 'ingredients' else 5}",
            },
        )
    return record


def compare(records: list[dict]) -> str:
    """Markdown comparison table with the measured numbers only."""
    lines = [
        "| Experiment | Representation | Recall@5 | Recall@10 | MRR@10 | Queries | "
        "Encode s | Cold load s | Peak RSS MB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        metrics = record["metrics"]
        lines.append(
            "| {label} | {mode} | {r5:.4f} | {r10:.4f} | {mrr:.4f} | {n} | {enc} | {cold} | {rss} |".format(
                label="E04" if record["mode"] == "ingredients" else "E05",
                mode=record["mode"],
                r5=metrics["recall@5"],
                r10=metrics["recall@10"],
                mrr=metrics["mrr@10"],
                n=metrics["n_queries"],
                enc=record["encode_seconds"],
                cold=record["cold_load_seconds"],
                rss=record["peak_rss_mb"],
            )
        )
    return "\n".join(lines)


def library_versions() -> dict[str, str]:
    versions = {}
    for name in ("mlflow", "ranx", "numpy"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover
            versions[name] = "not installed"
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--mode", default="both", choices=[*MODES, "both"])
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--out", default=None, help="write the comparison table to this file")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    modes = list(MODES) if args.mode == "both" else [args.mode]
    records = []
    for mode in modes:
        record = run_experiment(
            mode,
            cfg,
            split=args.split,
            top_k=args.top_k,
            models_dir=Path(args.models_dir),
            tracking_uri=args.tracking_uri,
            experiment=args.experiment,
            log=not args.no_mlflow,
        )
        records.append(record)
        metrics = record["metrics"]
        print(
            f"[{'E04' if mode == 'ingredients' else 'E05'}] {mode} | {cfg.data.profile} | "
            f"{args.split} | {metrics['n_queries']} queries"
        )
        for name in METRICS:
            print(f"  {name:<10} {metrics[name]:.4f}")
        print(f"  encode      {record['encode_seconds']} s")
        print(f"  cold load   {record['cold_load_seconds']} s")
        print(f"  peak RSS    {record['peak_rss_mb']} MB")
        if "mlflow_run_id" in record:
            print(f"  mlflow run  {record['mlflow_run_id']} ({args.tracking_uri})")

    table = compare(records)
    print()
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
