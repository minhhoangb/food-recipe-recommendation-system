"""Evaluate a bundle on one split with ranx and log the run to MLflow; build manual judging pools.

    python -m src.evaluate --bundle models/tfidf-pilot-xxxx --split validation --config configs/pilot.yaml
    python -m src.evaluate pool --config configs/default.yaml --split dev \
        --system E01=models/tfidf-main-1f1f4ce5 --system E08=models/minilm_finetuned-E04-main-1f1f4ce5 \
        --encoder E08=models/minilm_finetuned-main-cpu1ep

Metrics (ranx): Recall@5, Recall@10, MRR@10 against the synthetic qrels (origin-recipe proxy).
`test` is refused unless --allow-test is given: read it once, after the configuration is frozen.
The pool (reports/manual_eval/annotation_guide.md) holds the top 5 of every system per manual
query, merged and shuffled, with no system name; the system/rank key goes to a separate file.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import mlflow
from ranx import Qrels, Run, evaluate

from src.baseline import load_bundle
from src.config import fingerprint, load_config, processed_dir
from src.contracts import BundleManifest, Retriever
from src.utils import REPO_ROOT, read_json, read_jsonl, sha256_file

METRICS = ("recall@5", "recall@10", "mrr@10")
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "recipe-retrieval"
MANUAL_DIR = REPO_ROOT / "reports" / "manual_eval"
POOL_COLUMNS = [  # = reports/manual_eval/labels_template.csv (checked by a test)
    "query_id", "query_text", "query_ingredients", "item_id", "recipe_id", "title",
    "ingredients", "directions", "link", "relevance", "annotator", "notes",
]  # fmt: skip
ANNOTATORS = ("hoang", "hung", "hieu")
DOUBLE_SHARE = 0.2  # share of each query's items labelled independently by a second annotator


def evaluate_retriever(
    retriever: Retriever,
    queries: Sequence[dict],
    qrels: dict[str, dict[str, int]],
    top_k: int = 10,
    metrics: Sequence[str] = METRICS,
) -> dict[str, float]:
    """Run every query and score with ranx. Returns metrics plus n_queries and mean latency (ms)."""
    run, latency_ms = run_retriever(retriever, queries, top_k)
    return {**score_run(run, qrels, metrics), "n_queries": len(queries), "latency_ms": latency_ms}


def run_retriever(
    retriever: Retriever, queries: Sequence[dict], top_k: int = 10
) -> tuple[dict[str, dict[str, float]], float]:
    """query_id -> {recipe_id: score} for every query, plus the mean latency in ms.

    Hits are converted to rank-based scores so ranx keeps the retriever's own ordering, including
    the recipe_id tie-break; a query with no hits is an empty result (scored 0, not skipped).
    """
    run: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    for query in queries:
        hits = retriever.search(query["ingredients"], top_k)
        run[query["query_id"]] = {hit.recipe_id: float(len(hits) - rank) for rank, hit in enumerate(hits)}
    return run, 1000 * (time.perf_counter() - started) / max(len(queries), 1)


def score_run(
    run: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]], metrics: Sequence[str] = METRICS
) -> dict[str, float]:
    if not any(run.values()):  # ranx cannot build a Run with no document at all
        return {m: 0.0 for m in metrics}
    scores = evaluate(Qrels(qrels), Run(run), list(metrics), make_comparable=True)
    scores = scores if isinstance(scores, dict) else {metrics[0]: scores}
    return {k: float(v) for k, v in scores.items()}


def load_split(directory: Path, split: str) -> tuple[list[dict], dict[str, dict[str, int]]]:
    queries = [q for q in read_jsonl(directory / "queries.jsonl") if q["split"] == split]
    wanted = {q["query_id"] for q in queries}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in read_jsonl(directory / "qrels.jsonl"):
        if row["query_id"] in wanted:
            qrels[row["query_id"]][row["recipe_id"]] = row["relevance"]
    return queries, dict(qrels)


def log_to_mlflow(
    results: dict,
    manifest,
    bundle: Path,
    cfg,
    split: str,
    top_k: int,
    uri: str,
    experiment: str,
    tags: dict | None = None,
) -> str:
    """tags["experiment"] (E01, E02, ...) also prefixes the run name, so E01/E02 runs stay apart."""
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    fit_info = read_json(bundle / "metrics.json") if (bundle / "metrics.json").exists() else {}
    tags = tags or {}
    label = f"{tags['experiment']}-" if "experiment" in tags else ""
    with mlflow.start_run(run_name=f"{label}{manifest.model_name}-{cfg.data.profile}-{split}") as run:
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
        mlflow.set_tags({**{k: str(v) for k, v in fit_info.items() if k != "fit_seconds"}, **tags})
        return run.info.run_id


def load_system(bundle: Path, encoder: str | Path | None = None):
    """(retriever, manifest, fit seconds, what the fit time measures) for a lexical or dense bundle.

    encoder: the checkpoint a fine-tuned bundle was encoded with. Queries must go through the same
    model; DenseRetriever.from_bundle alone would encode them with the pretrained MiniLM.
    """
    if (bundle / "model.joblib").exists():
        retriever, manifest = load_bundle(bundle)
        info = read_json(bundle / "metrics.json")
        what = "index build" if manifest.model_name == "bm25" else "fit on train + transform corpus"
        return retriever, manifest, info["fit_seconds"], what
    from src.embeddings import load_model  # torch only when a dense bundle is used
    from src.retrieval import DenseRetriever

    payload = read_json(bundle / "manifest.json")
    if payload["model_name"] == "minilm_finetuned" and encoder is None:
        raise ValueError(f"{bundle} was encoded by a fine-tuned model: pass its checkpoint as encoder")
    model = load_model(model_name_or_path=str(encoder)) if encoder else None
    retriever = DenseRetriever.from_bundle(bundle, model=model)
    return retriever, BundleManifest(**payload), payload["encode_seconds"], "corpus encoding (no training)"


def build_pool(
    systems: dict[str, Retriever],
    queries: Sequence[dict],
    corpus: dict[str, dict],
    depth: int = 5,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Blind judging pool: the top `depth` of every system per query, merged and shuffled.

    queries are rows of reports/manual_eval/queries.csv. Returns (items, key): items have the
    POOL_COLUMNS and nothing that names a system; key maps item_id to the systems and ranks.
    """
    rng = random.Random(seed)
    items, key = [], []
    for query in queries:
        found: dict[str, list[str]] = defaultdict(list)
        for label, retriever in systems.items():
            for rank, hit in enumerate(retriever.search(query["ingredients"].split(";"), depth), 1):
                found[hit.recipe_id].append(f"{label}@{rank}")
        recipe_ids = sorted(found)
        rng.shuffle(recipe_ids)  # pool order must not reveal which system ranked what
        for n, recipe_id in enumerate(recipe_ids, 1):
            item_id, recipe = f"{query['query_id']}-{n:02d}", corpus[recipe_id]
            items.append(
                {
                    "query_id": query["query_id"],
                    "query_text": query["text"],
                    "query_ingredients": query["ingredients"],
                    "item_id": item_id,
                    "recipe_id": recipe_id,
                    "title": recipe["title"],
                    "ingredients": "; ".join(recipe["ingredients_raw"]),
                    "directions": " ".join(f"{i}. {step}" for i, step in enumerate(recipe["directions"], 1)),
                    "link": recipe["link"],
                    "relevance": "",
                    "annotator": "",
                    "notes": "",
                }
            )
            key.append({"item_id": item_id, "query_id": query["query_id"], "recipe_id": recipe_id,
                        "systems": " ".join(found[recipe_id])})  # fmt: skip
    return items, key


def assign_annotators(
    items: Sequence[dict],
    annotators: Sequence[str] = ANNOTATORS,
    double_share: float = DOUBLE_SHARE,
    seed: int = 42,
) -> list[dict]:
    """One row per (item, annotator): queries go round-robin over the annotators, and at least
    `double_share` of each query's items also go to the next annotator for an independent label."""
    rng = random.Random(seed)
    by_query: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_query[item["query_id"]].append(item)
    rows = []
    for n, query_id in enumerate(sorted(by_query)):
        first, second = annotators[n % len(annotators)], annotators[(n + 1) % len(annotators)]
        query_items = by_query[query_id]
        doubled = set(rng.sample(range(len(query_items)), math.ceil(double_share * len(query_items))))
        for index, item in enumerate(query_items):
            rows.append({**item, "annotator": first})
            if index in doubled:
                rows.append({**item, "annotator": second})
    return rows


def pool_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.evaluate pool", description=build_pool.__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="dev", choices=["dev", "final"])
    parser.add_argument("--system", action="append", required=True, help="LABEL=models/<run_id>")
    parser.add_argument("--encoder", action="append", default=[], help="LABEL=<checkpoint> (fine-tuned)")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--out", default=str(MANUAL_DIR / "pool"))
    args = parser.parse_args(argv)

    out = Path(args.out)
    for existing in out.glob(f"pool_{args.split}*.csv"):
        with existing.open(encoding="utf-8-sig", newline="") as handle:
            if any(
                row.get("relevance", "").strip() or row.get("notes", "").strip()
                for row in csv.DictReader(handle)
            ):
                parser.error(f"{existing} contains labels or notes; use a new --out directory")

    cfg = load_config(args.config)
    directory = processed_dir(cfg)
    corpus_sha = sha256_file(directory / "corpus.jsonl")
    encoders = dict(spec.split("=", 1) for spec in args.encoder)
    systems = {}
    for spec in args.system:
        label, path = spec.split("=", 1)
        retriever, manifest, _, _ = load_system(Path(path), encoders.get(label))
        if manifest.corpus_sha256 != corpus_sha:
            parser.error(f"{label}: bundle was built from a different corpus than the configured profile")
        systems[label] = retriever
    with open(MANUAL_DIR / "queries.csv", encoding="utf-8", newline="") as handle:
        queries = [q for q in csv.DictReader(handle) if q["split"] == args.split]
    corpus = {r["recipe_id"]: r for r in read_jsonl(directory / "corpus.jsonl")}

    items, key = build_pool(systems, queries, corpus, args.depth, cfg.seed)
    rows = assign_annotators(items, seed=cfg.seed)
    out.mkdir(parents=True, exist_ok=True)
    for annotator in ANNOTATORS:
        mine = [row for row in rows if row["annotator"] == annotator]
        write_csv(out / f"pool_{args.split}_{annotator}.csv", mine, POOL_COLUMNS)
        print(f"{annotator}: {len(mine)} rows, queries {sorted({r['query_id'] for r in mine})}")
    write_csv(out / f"pool_{args.split}_key.csv", key, ["item_id", "query_id", "recipe_id", "systems"])
    print(f"{len(items)} pooled items for {len(queries)} queries from {sorted(systems)} -> {out}")
    return 0


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    # utf-8-sig so Excel opens accents correctly; the pandas/csv readers skip the BOM
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["pool"]:
        return pool_main(argv[1:])
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
