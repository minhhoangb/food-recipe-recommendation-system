"""Freeze artifacts, run E10 once, and prepare a blind final judging pool.

    python -m src.final_evaluation freeze --system E03=models/bm25-... --selected E03
    python -m src.final_evaluation run
    python -m src.final_evaluation pool

The exclusive E10 ledger is created before reading test queries. A failed run is
preserved for audit and cannot be restarted by this command. No fitting occurs.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from ranx import Qrels, Run, compare

from src.compare import MAX_P, significance
from src.config import fingerprint, load_config, processed_dir
from src.evaluate import (
    ANNOTATORS,
    DOUBLE_SHARE,
    MANUAL_DIR,
    METRICS,
    POOL_COLUMNS,
    assign_annotators,
    build_pool,
    load_split,
    load_system,
    run_retriever,
    score_run,
    write_csv,
)
from src.utils import REPO_ROOT, read_json, read_jsonl, sha256_file, write_json

FINAL_DIR = REPO_ROOT / "reports" / "final"
CODE_FILES = (
    "src/final_evaluation.py",
    "src/evaluate.py",
    "src/retrieval.py",
    "src/embeddings.py",
    "src/baseline.py",
    "src/normalize.py",
    "src/config.py",
    "src/contracts.py",
    "src/pipeline.py",
    "configs/production.yaml",
)


def tree_hashes(path: Path) -> dict[str, str]:
    return {p.relative_to(path).as_posix(): sha256_file(p) for p in sorted(path.rglob("*")) if p.is_file()}


def freeze(
    config: str,
    systems: dict[str, str],
    encoders: dict[str, str],
    selected: str,
    out: Path,
    expected_queries: int = 1000,
) -> dict:
    if selected not in systems:
        raise ValueError("Selected deployment model must be one of the frozen systems")
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "frozen_config.json"
    if destination.exists() or (out / "e10_execution.json").exists():
        raise FileExistsError("Freeze already exists; do not overwrite it")
    cfg = load_config(config)
    directory = processed_dir(cfg)
    corpus_sha = sha256_file(directory / "corpus.jsonl")
    models = []
    for label, value in systems.items():
        bundle = Path(value)
        if not bundle.is_absolute():
            bundle = REPO_ROOT / bundle
        manifest = read_json(bundle / "manifest.json")
        if manifest["corpus_sha256"] != corpus_sha:
            raise ValueError(f"{label}: corpus hash mismatch")
        for name, digest in manifest["files"].items():
            if sha256_file(bundle / name) != digest:
                raise ValueError(f"{label}: modified artifact {name}")
        encoder = Path(encoders[label]).resolve() if label in encoders else None
        if manifest["model_name"] == "minilm_pretrained" and encoder is None:
            raise ValueError(f"{label}: pin the local pretrained encoder snapshot before freezing")
        models.append(
            {
                "label": label,
                "bundle": bundle.relative_to(REPO_ROOT).as_posix()
                if bundle.is_relative_to(REPO_ROOT)
                else str(bundle),
                "model_name": manifest["model_name"],
                "files": tree_hashes(bundle),
                "encoder": str(encoder) if encoder else None,
                "encoder_files": tree_hashes(encoder) if encoder else {},
            }
        )
    frozen = {
        "status": "frozen",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "decision_source": "User confirmed seven systems and BM25 deployment before any E10 score",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "config": config,
        "config_sha256": fingerprint(cfg),
        "profile": str(cfg.data.profile),
        "data_dir": str(directory),
        "data_files": {
            name: sha256_file(directory / name)
            for name in (
                "corpus.jsonl",
                "splits.json",
                "queries.jsonl",
                "qrels.jsonl",
                "dataset_manifest.json",
            )
        },
        "code_files": {name: sha256_file(REPO_ROOT / name) for name in CODE_FILES},
        "systems": models,
        "selected_deployment": selected,
        "selection_basis": "Highest historical validation MRR@10; test scores cannot change this choice",
        "top_k": 10,
        "rerank": "none",
        "test_queries_expected": expected_queries,
        "pool_depth": 5,
        "double_label_share": DOUBLE_SHARE,
        "seed": int(cfg.seed),
        "significance": {
            "library": "ranx.compare",
            "test": "student",
            "max_p": MAX_P,
            "correction": None,
            "limitation": "Pairwise tests, no multiplicity correction",
        },
        "python": platform.python_version(),
    }
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(frozen, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return frozen


def verify_freeze(path: Path) -> dict:
    frozen = read_json(path)
    if frozen["status"] != "frozen" or fingerprint(load_config(frozen["config"])) != frozen["config_sha256"]:
        raise ValueError("Configuration changed after freezing")
    for name, digest in frozen["code_files"].items():
        if sha256_file(REPO_ROOT / name) != digest:
            raise ValueError(f"Evaluation code changed after freezing: {name}")
    for name, digest in frozen["data_files"].items():
        if sha256_file(Path(frozen["data_dir"]) / name) != digest:
            raise ValueError(f"Frozen data changed: {name}")
    for model in frozen["systems"]:
        bundle = Path(model["bundle"])
        if not bundle.is_absolute():
            bundle = REPO_ROOT / bundle
        if tree_hashes(bundle) != model["files"]:
            raise ValueError(f"Frozen bundle changed: {model['label']}")
        if model["encoder"] and tree_hashes(Path(model["encoder"])) != model["encoder_files"]:
            raise ValueError(f"Frozen encoder changed: {model['label']}")
    return frozen


def run_final(path: Path, tracking_uri: str = "sqlite:///mlflow.db") -> pd.DataFrame:
    out = path.parent
    ledger = out / "e10_execution.json"
    if ledger.exists():
        raise FileExistsError("E10 already started; use its saved outputs, never rerun test retrieval")
    frozen = verify_freeze(path)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("recipe-retrieval")
    previous = mlflow.search_runs(search_all_experiments=True, filter_string="params.split = 'test'")
    if not previous.empty:
        raise ValueError("Prior test runs exist in this MLflow store; audit them before E10")
    state = {
        "status": "running",
        "freeze_sha256": sha256_file(path),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "completed_systems": [],
    }
    with ledger.open("x", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    try:
        queries, qrels = load_split(Path(frozen["data_dir"]), "test")
        if len(queries) != frozen["test_queries_expected"]:
            raise ValueError("Test query count differs from frozen protocol")
        rows, runs = [], []
        with mlflow.start_run(
            run_name="E10-frozen-main-test",
            tags={"experiment": "E10", "freeze_sha256": state["freeze_sha256"]},
        ) as parent:
            state["mlflow_parent_run"] = parent.info.run_id
            for model in frozen["systems"]:
                label = model["label"]
                bundle = Path(model["bundle"])
                if not bundle.is_absolute():
                    bundle = REPO_ROOT / bundle
                retriever, manifest, _, _ = load_system(bundle, model["encoder"])
                ranking, latency = run_retriever(retriever, queries, frozen["top_k"])
                raw = out / "runs" / f"{label}_test_run.json"
                write_json(raw, ranking)
                metrics = {**score_run(ranking, qrels), "latency_ms": latency, "n_queries": len(queries)}
                # Each system has its own auditable run under the single E10 parent.
                with mlflow.start_run(run_name=f"E10-{label}", nested=True) as child:
                    mlflow.log_params(
                        {
                            "split": "test",
                            "profile": frozen["profile"],
                            "system": label,
                            "bundle": manifest.run_id,
                            "top_k": frozen["top_k"],
                            "freeze_sha256": state["freeze_sha256"],
                        }
                    )
                    mlflow.set_tags(
                        {
                            "experiment": "E10",
                            "selected_before_test": str(label == frozen["selected_deployment"]).lower(),
                        }
                    )
                    mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items()})
                    mlflow.log_artifact(str(raw), "rankings")
                    run_id = child.info.run_id
                rows.append(
                    {
                        "experiment": label,
                        **{f"test_{k}": metrics[k] for k in METRICS},
                        "test_n_queries": len(queries),
                        "mlflow_test_run": run_id,
                        "selected_deployment": label == frozen["selected_deployment"],
                    }
                )
                runs.append(Run({q: hits for q, hits in ranking.items() if hits}, name=label))
                state["completed_systems"].append(label)
                write_json(ledger, state)
                print(f"E10 {label}: {metrics}", flush=True)
            report = compare(
                Qrels(qrels), runs, list(METRICS), max_p=MAX_P, stat_test="student", make_comparable=True
            )
            report_dict = report.to_dict()
            for row in rows:
                row["test_mrr_significantly_better_than"] = significance(report_dict, row["experiment"])
            frame = pd.DataFrame(rows)
            frame.to_csv(out / "comparison_test.csv", index=False)
            write_json(out / "significance_test.json", report_dict)
            (out / "comparison_ranx.md").write_text(
                "# Frozen E10 test comparison\n\nPaired Student t-test; p < 0.01, "
                "no multiplicity correction. Deployment was chosen on validation before test.\n\n```\n"
                + str(report)
                + "\n```\n",
                encoding="utf-8",
            )
            per_query = [
                {"query_id": q, "experiment": r.name, **{m: r.scores[m][q] for m in METRICS}}
                for r in runs
                for q in qrels
            ]
            pd.DataFrame(per_query).to_csv(out / "comparison_test_per_query.csv", index=False)
            mlflow.log_artifacts(str(out), "final_evaluation")
        state.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(ledger, state)
        return frame
    except BaseException as error:
        state.update(status="failed_no_automatic_retry", error=str(error))
        write_json(ledger, state)
        raise


def final_pool(path: Path) -> dict:
    frozen = verify_freeze(path)
    out = MANUAL_DIR / "final"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Final pool already exists; preserve assignments and independent labels")
    with (MANUAL_DIR / "queries.csv").open(encoding="utf-8-sig", newline="") as handle:
        queries = [q for q in csv.DictReader(handle) if q["split"] == "final"]
    if len(queries) != 20:
        raise ValueError("Expected exactly 20 final manual queries")
    systems, system_runs = {}, {}
    for model in frozen["systems"]:
        bundle = Path(model["bundle"])
        if not bundle.is_absolute():
            bundle = REPO_ROOT / bundle
        retriever = load_system(bundle, model["encoder"])[0]
        systems[model["label"]] = retriever
        normalized = [
            {"query_id": q["query_id"], "ingredients": q["ingredients"].split(";")} for q in queries
        ]
        system_runs[model["label"]] = run_retriever(retriever, normalized, 10)[0]
    corpus = {r["recipe_id"]: r for r in read_jsonl(Path(frozen["data_dir"]) / "corpus.jsonl")}
    items, key = build_pool(systems, queries, corpus, depth=frozen["pool_depth"], seed=frozen["seed"])
    rows = assign_annotators(items, double_share=frozen["double_label_share"], seed=frozen["seed"])
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "pool_final.csv", items, POOL_COLUMNS)
    for name in ANNOTATORS:
        write_csv(out / f"pool_final_{name}.csv", [r for r in rows if r["annotator"] == name], POOL_COLUMNS)
    # Keep system identities separate from the judging folder.
    private = path.parent / "pool_keys"
    private.mkdir(exist_ok=True)
    write_csv(private / "pool_final_key.csv", key, ["item_id", "query_id", "recipe_id", "systems"])
    write_json(private / "system_runs_top10.json", system_runs)
    metadata = {
        "freeze_sha256": sha256_file(path),
        "n_queries": len(queries),
        "n_items": len(items),
        "n_assignments": len(rows),
        "assignments": {n: sum(r["annotator"] == n for r in rows) for n in ANNOTATORS},
        "n_double_label_items": len(rows) - len(items),
        "depth": frozen["pool_depth"],
        "label_status": "awaiting independent human labels",
        "n_labels": 0,
    }
    write_json(out / "pool_final_metadata.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run", "pool"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--system", action="append", default=[])
    parser.add_argument("--encoder", action="append", default=[])
    parser.add_argument("--selected", default="E03")
    parser.add_argument("--out", type=Path, default=FINAL_DIR)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    path = args.out / "frozen_config.json"
    if args.action == "freeze":
        if not args.system:
            parser.error("freeze needs explicit --system declarations")
        freeze(
            args.config,
            dict(s.split("=", 1) for s in args.system),
            dict(s.split("=", 1) for s in args.encoder),
            args.selected,
            args.out,
        )
    elif args.action == "run":
        run_final(path, args.tracking_uri)
    else:
        print(final_pool(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
