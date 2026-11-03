"""Validate final human labels, preserve disagreements, then compute pooled metrics.

    python -m src.manual_evaluation

No metric is published while labels, independence declarations or adjudications
are incomplete. System keys are read only after the original labels are locked.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from ranx import Qrels, Run, evaluate
from sklearn.metrics import cohen_kappa_score

from src.evaluate import ANNOTATORS, write_csv
from src.utils import REPO_ROOT, read_json, sha256_file, write_json


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate_labels(directory: Path, keys: Path, out: Path) -> dict:
    master = read_csv(directory / "pool_final.csv")
    by_id = {r["item_id"]: r for r in master}
    if len(by_id) != len(master):
        raise ValueError("Duplicate item IDs in master pool")
    originals = {n: read_csv(directory / f"pool_final_{n}.csv") for n in ANNOTATORS}
    snapshots = {n: sha256_file(directory / f"pool_final_{n}.csv") for n in ANNOTATORS}
    lock = out / "original_labels_lock.json"
    if lock.exists():
        previous = read_json(lock)
        if previous["label_files_sha256"] != snapshots:
            raise ValueError("Original labels changed after locking; adjudicate separately")
        if previous["annotation_log_sha256"] != sha256_file(directory / "annotation_log.csv"):
            raise ValueError("Original annotation provenance changed after locking")
    missing = {n: sum(r["relevance"].strip() == "" for r in rows) for n, rows in originals.items()}
    status = {
        "status": "awaiting_independent_labels",
        "missing_labels": missing,
        "n_items": len(master),
        "metrics_published": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manual_eval_status.json", status)
    by_item = defaultdict(dict)
    for name, rows in originals.items():
        for row in rows:
            item = by_id.get(row["item_id"])
            if item is None or row["annotator"] != name or row["recipe_id"] != item["recipe_id"]:
                raise ValueError("An assignment changed identity or annotator")
            for column in (
                "query_id",
                "query_text",
                "query_ingredients",
                "title",
                "ingredients",
                "directions",
            ):
                if row[column] != item[column]:
                    raise ValueError(f"Assignment content changed: {row['item_id']}/{column}")
            if name in by_item[row["item_id"]]:
                raise ValueError("Duplicate assignment")
            label = row["relevance"].strip()
            if label and label not in ("0", "1", "2"):
                raise ValueError("Relevance must be 0, 1 or 2")
            by_item[row["item_id"]][name] = label
    if any(missing.values()):
        return status
    if set(by_item) != set(by_id):
        raise ValueError("Some pool items have no judgments")
    metadata = read_json(directory / "pool_final_metadata.json")
    if any(len(originals[n]) != metadata["assignments"][n] for n in ANNOTATORS):
        raise ValueError("Assignment row count changed")
    provenance = {r["annotator"]: r for r in read_csv(directory / "annotation_log.csv")}
    for name in ANNOTATORS:
        row = provenance.get(name, {})
        if (
            row.get("label_source") != "human"
            or row.get("independent") != "yes"
            or row.get("exposure") != "none"
            or not row.get("completed_utc")
        ):
            raise ValueError(f"{name}: independent human provenance is incomplete")
    per_query = defaultdict(list)
    for item_id, item in by_id.items():
        per_query[item["query_id"]].append(item_id)
    for query, ids in per_query.items():
        if sum(len(by_item[i]) >= 2 for i in ids) < math.ceil(0.2 * len(ids)):
            raise ValueError(f"{query}: fewer than 20% independently double-labelled items")
    write_json(
        lock,
        {
            "label_files_sha256": snapshots,
            "annotation_log_sha256": sha256_file(directory / "annotation_log.csv"),
        },
    )
    # Only now read disagreements and system rankings; do not expose them to unfinished judging.
    disagreements = []
    final = {}
    pair_labels = defaultdict(lambda: ([], []))
    for item_id, labels in by_item.items():
        for first, second in combinations(sorted(labels), 2):
            a, b = pair_labels[(first, second)]
            a.append(int(labels[first]))
            b.append(int(labels[second]))
        if len(set(labels.values())) == 1:
            final[item_id] = int(next(iter(labels.values())))
        else:
            disagreements.append(
                {
                    "item_id": item_id,
                    "original_labels": str(labels),
                    "relevance": "",
                    "adjudicator": "",
                    "reason": "",
                }
            )
    adjudication = directory / "adjudication.csv"
    if disagreements and not adjudication.exists():
        write_csv(adjudication, disagreements, list(disagreements[0]))
    resolved = {r["item_id"]: r for r in read_csv(adjudication)} if adjudication.exists() else {}
    unresolved = []
    for conflict in disagreements:
        row = resolved.get(conflict["item_id"], {})
        if row.get("relevance") not in ("0", "1", "2") or not row.get("adjudicator") or not row.get("reason"):
            unresolved.append(conflict["item_id"])
        else:
            final[conflict["item_id"]] = int(row["relevance"])
    if unresolved:
        status.update(status="awaiting_adjudication", unresolved_items=unresolved)
        write_json(out / "manual_eval_status.json", status)
        return status
    agreement = []
    for pair, (a, b) in sorted(pair_labels.items()):
        kappa = float(cohen_kappa_score(a, b, labels=[0, 1, 2]))
        agreement.append(
            {
                "annotators": list(pair),
                "n_double_labels": len(a),
                "cohen_kappa": kappa if math.isfinite(kappa) else None,
                "raw_agreement": sum(x == y for x, y in zip(a, b, strict=True)) / len(a),
            }
        )
    qrels = defaultdict(dict)
    for item_id, label in final.items():
        row = by_id[item_id]
        qrels[row["query_id"]][row["recipe_id"]] = label
    system_runs = read_json(keys / "system_runs_top10.json")
    scores = []
    for label, ranking in system_runs.items():
        if set(ranking) != set(qrels):
            raise ValueError(f"{label}: ranking query IDs differ from judged pool")
        scored = float(evaluate(Qrels(dict(qrels)), Run(ranking), "ndcg@10", make_comparable=True))
        coverage_by_query = {}
        for query, hits in ranking.items():
            coverage_by_query[query] = sum(rid in qrels[query] for rid in hits) / max(1, len(hits))
        scores.append(
            {
                "system": label,
                "pooled_ndcg_at_10": scored,
                "mean_top10_judging_coverage": sum(coverage_by_query.values()) / len(coverage_by_query),
                "coverage_by_query": coverage_by_query,
            }
        )
    labels_out = [
        {
            "item_id": item,
            "query_id": by_id[item]["query_id"],
            "recipe_id": by_id[item]["recipe_id"],
            "relevance": value,
        }
        for item, value in sorted(final.items())
    ]
    write_csv(out / "final_labels_adjudicated.csv", labels_out, list(labels_out[0]))
    status.update(
        status="complete",
        metrics_published=True,
        agreement=agreement,
        scores=scores,
        n_adjudicated=len(disagreements),
        metric_convention=(
            "ranx ndcg@10 uses linear gain; unjudged results count as zero, not proven irrelevant"
        ),
    )
    write_json(out / "manual_eval_status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=REPO_ROOT / "reports/manual_eval/final")
    parser.add_argument("--keys", type=Path, default=REPO_ROOT / "reports/final/pool_keys")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports/final")
    args = parser.parse_args()
    print(evaluate_labels(args.labels, args.keys, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
