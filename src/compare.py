"""Validation comparison of E01-E05 (+ LSA) with ranx.compare significance tests.

    python -m src.compare --config configs/default.yaml \
        --system E01=models/tfidf-main-1f1f4ce5 --system E02=models/tfidf_tuned-main-1f1f4ce5 \
        --system E03=models/bm25-main-1f1f4ce5 --system LSA=models/lsa-main-1f1f4ce5 \
        --system E04=models/minilm_pretrained-E04-main-1f1f4ce5 \
        --system E05=models/minilm_pretrained-E05-main-1f1f4ce5

Every system is scored on the same corpus and the same queries: `train` (train proxy) and
`validation`; each (system, split) is one MLflow run. CV comes from src.tune (reports/*_cv5.json).
Writes reports/comparison.csv, reports/comparison_per_query.csv (validation, one row per query and
system) and reports/comparison_ranx.md (paired Student t-test, p < 0.01, ranx default).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd
from ranx import Qrels, Run, compare

from src.config import load_config, processed_dir
from src.evaluate import (
    DEFAULT_EXPERIMENT,
    DEFAULT_TRACKING_URI,
    METRICS,
    load_split,
    load_system,
    log_to_mlflow,
    run_retriever,
    score_run,
)
from src.utils import REPO_ROOT, read_json, sha256_file

MAX_P = 0.01
# Why a column is N/A: the Master Comparison Table only allows N/A with a reason.
CV_NOT_APPLICABLE = {
    "bm25": "N/A: rank-bm25 indexes every candidate (transductive IDF), nothing is fitted on train",
    "minilm_pretrained": "N/A: pretrained encoder, nothing is fitted on our data",
    "minilm_finetuned": "N/A: one fine-tuning run; checkpoint chosen on validation, not cross-validated",
}


def cv_columns(label: str, model_name: str, reports: Path) -> dict:
    """CV mean/std from src.tune output: E01 = default grid point, E02 = best, LSA = its own file."""
    files = {"E01": ("tfidf_cv5.json", "default_E01"), "E02": ("tfidf_cv5.json", "best")}
    files["LSA"] = ("lsa_cv5.json", "default")
    if label in files and (reports / files[label][0]).exists():
        row = read_json(reports / files[label][0])[files[label][1]]
        return {"cv_mrr@10_mean": row["mean_cv_mrr@10"], "cv_mrr@10_std": row["std_cv_mrr@10"], "cv_note": ""}
    note = CV_NOT_APPLICABLE.get(model_name, "N/A: CV not run (see reports/*_cv5.json)")
    return {"cv_mrr@10_mean": None, "cv_mrr@10_std": None, "cv_note": note}


def significance(report_dict: dict, label: str, metric: str = "mrr@10") -> str:
    """Systems this one beats on `metric` with p < MAX_P."""
    mine = report_dict[label]
    better = [
        other
        for other, p_values in mine["comparisons"].items()
        if p_values[metric] < MAX_P and mine["scores"][metric] > report_dict[other]["scores"][metric]
    ]
    return " ".join(sorted(better))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--system", action="append", required=True, help="LABEL=models/<run_id>")
    parser.add_argument("--encoder", action="append", default=[], help="LABEL=<checkpoint> (fine-tuned)")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    directory = processed_dir(cfg)
    corpus_sha = sha256_file(directory / "corpus.jsonl")
    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    splits = {split: load_split(directory, split) for split in ("train", "validation")}

    encoders = dict(spec.split("=", 1) for spec in args.encoder)
    rows, validation_runs = [], []
    for spec in args.system:
        label, _, path = spec.partition("=")
        bundle = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
        retriever, manifest, fit_seconds, fit_what = load_system(bundle, encoders.get(label))
        if manifest.corpus_sha256 != corpus_sha:
            parser.error(f"{label}: bundle was built from a different corpus than the configured profile")
        row = {"experiment": label, "model": manifest.model_name, "bundle": manifest.run_id}
        tuned = {"E02": "yes (GroupKFold CV5)"}.get(label, "no")
        row["tuned"] = "fine-tuned (MNRL)" if manifest.model_name == "minilm_finetuned" else tuned
        for split, (queries, qrels) in splits.items():
            run, latency_ms = run_retriever(retriever, queries, args.top_k)
            results = {**score_run(run, qrels), "n_queries": len(queries), "latency_ms": latency_ms}
            prefix = "train" if split == "train" else "val"
            row.update({f"{prefix}_{m}": results[m] for m in METRICS})
            row[f"{prefix}_n_queries"] = len(queries)
            if split == "validation":
                row["latency_ms"] = round(latency_ms, 2)
                validation_runs.append(Run({q: hits for q, hits in run.items() if hits}, name=label))
            if not args.no_mlflow:
                row[f"mlflow_{prefix}_run"] = log_to_mlflow(
                    results, manifest, bundle, cfg, split, args.top_k, args.tracking_uri, args.experiment,
                    tags={"experiment": label},
                )
            print(f"{label:<4} {split:<10} " + "  ".join(f"{m} {results[m]:.4f}" for m in METRICS))
        row.update(cv_columns(label, manifest.model_name, out))
        row.update(fit_seconds=fit_seconds, fit_measures=fit_what)
        rows.append(row)

    _, val_qrels = splits["validation"]
    report = compare(Qrels(val_qrels), validation_runs, list(METRICS), max_p=MAX_P, make_comparable=True)
    report_dict = report.to_dict()
    for row in rows:
        row["mrr@10_significantly_better_than"] = significance(report_dict, row["experiment"])
        row["corpus_sha256"] = corpus_sha[:12]

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "comparison.csv", index=False)
    per_query = [
        {"query_id": query_id, "experiment": run.name, **{m: run.scores[m][query_id] for m in METRICS}}
        for run in validation_runs
        for query_id in val_qrels
    ]
    pd.DataFrame(per_query).to_csv(out / "comparison_per_query.csv", index=False)
    (out / "comparison_ranx.md").write_text(
        f"# ranx.compare on {cfg.data.profile}/validation ({len(val_qrels)} queries)\n\n"
        f"Paired Student t-test, superscript = significantly better than that system at p < {MAX_P}.\n\n"
        f"```\n{report}\n```\n",
        encoding="utf-8",
    )
    print(f"\n{report}\n\n-> {out / 'comparison.csv'}")
    if not args.no_mlflow:
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=f"comparison-{cfg.data.profile}-validation"):
            for name in ("comparison.csv", "comparison_per_query.csv", "comparison_ranx.md"):
                mlflow.log_artifact(str(out / name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
