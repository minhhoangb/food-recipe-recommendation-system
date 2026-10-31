"""Production monitoring for the recommendation API from request logs.

    python -m src.monitor --log-path logs/requests.db --out reports/monitoring_report.md

Reads the `RequestLogRecord` rows of the API's request log (SQLite by default, the format
`api/request_log.py` writes; a JSONL file is accepted too) and computes:
  - Request volume & throughput
  - Success and error rates
  - Latency distribution (p50, p90, p95, p99)
  - Zero-hit rate (queries returning 0 results)
  - Model & endpoint breakdowns
  - Ingredient frequency & distribution drift
Logs metrics to MLflow or exports a markdown report.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import RequestLogRecord
from src.utils import REPO_ROOT, read_jsonl

DEFAULT_LOG_PATH = REPO_ROOT / "logs" / "requests.db"
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "monitoring_report.md"


def load_request_logs(path: Path | str) -> list[RequestLogRecord]:
    """Load and validate RequestLogRecords from the request log.

    The API writes SQLite (`API_REQUEST_LOG`, see `api/request_log.py`); a JSONL file is
    also accepted, because both hold the same `RequestLogRecord` schema.
    """
    p = Path(path)
    if not p.exists():
        return []
    if p.suffix in {".db", ".sqlite", ".sqlite3"}:
        return _load_request_logs_sqlite(p)
    records: list[RequestLogRecord] = []
    for line in read_jsonl(p):
        try:
            records.append(RequestLogRecord.model_validate(line))
        except Exception:
            continue
    return records


def _load_request_logs_sqlite(path: Path) -> list[RequestLogRecord]:
    """Read the rows of the API's SQLite request log."""
    import sqlite3

    from api.request_log import COLUMNS, TABLE

    records: list[RequestLogRecord] = []
    with sqlite3.connect(path, timeout=30.0) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(f"SELECT {', '.join(COLUMNS)} FROM {TABLE}").fetchall()
        except sqlite3.Error:
            return []
    for row in rows:
        record = dict(row)
        record["ingredients"] = _split(record["ingredients"])
        record["result_ids"] = _split(record["result_ids"])
        record["scores"] = [float(s) for s in _split(record["scores"])]
        try:
            records.append(RequestLogRecord.model_validate(record))
        except Exception:
            continue
    return records


def _split(value: Any) -> list[str]:
    """The log stores list fields as comma-separated strings."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or value == "":
        return []
    return [part for part in str(value).split(",") if part]


def compute_monitoring_metrics(records: list[RequestLogRecord]) -> dict[str, Any]:
    """Compute production health, latency percentiles, and drift metrics."""
    if not records:
        return {
            "total_requests": 0,
            "error_rate": 0.0,
            "success_rate": 0.0,
            "latency_ms": {},
            "zero_hit_rate": 0.0,
            "endpoint_counts": {},
            "model_counts": {},
            "top_ingredients": [],
        }

    total = len(records)
    errors = sum(1 for r in records if r.status_code >= 400)
    latencies = [r.latency_ms for r in records]

    # Latency percentiles
    latencies_sorted = sorted(latencies)

    def percentile(p: float) -> float:
        idx = int(p * (len(latencies_sorted) - 1))
        return round(latencies_sorted[idx], 2)

    latency_stats = {
        "mean": round(statistics.mean(latencies), 2),
        "median": round(statistics.median(latencies), 2),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "min": round(min(latencies), 2),
        "max": round(max(latencies), 2),
    }

    # Zero hits for predict endpoints
    predict_records = [r for r in records if "predict" in r.endpoint]
    zero_hits = sum(1 for r in predict_records if not r.result_ids)
    zero_hit_rate = round(zero_hits / len(predict_records), 4) if predict_records else 0.0

    # Top hit scores
    all_scores = [s for r in records for s in r.scores]
    top_scores = [r.scores[0] for r in records if r.scores]

    # Endpoint & model breakdown
    endpoints = Counter(r.endpoint for r in records)
    models = Counter(r.model for r in records)

    # Ingredient frequency
    ingredient_counter: Counter[str] = Counter()
    for r in records:
        ingredient_counter.update(r.ingredients)

    return {
        "total_requests": total,
        "error_count": errors,
        "error_rate": round(errors / total, 4),
        "success_rate": round((total - errors) / total, 4),
        "latency_ms": latency_stats,
        "predict_requests": len(predict_records),
        "zero_hit_count": zero_hits,
        "zero_hit_rate": zero_hit_rate,
        "mean_top_score": round(statistics.mean(top_scores), 4) if top_scores else None,
        "mean_score": round(statistics.mean(all_scores), 4) if all_scores else None,
        "endpoint_counts": dict(endpoints),
        "model_counts": dict(models),
        "top_ingredients": ingredient_counter.most_common(10),
    }


def format_monitoring_report(metrics: dict[str, Any]) -> str:
    """Format monitoring metrics as a readable markdown document."""
    lines = [
        "# API & Model Monitoring Report",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        "",
        "## Summary",
        "",
        f"- **Total Requests:** {metrics['total_requests']}",
        f"- **Success Rate:** {metrics['success_rate'] * 100:.2f}%",
        f"- **Error Rate:** {metrics['error_rate'] * 100:.2f}% ({metrics.get('error_count', 0)} errors)",
        f"- **Zero-Hit Rate:** {metrics['zero_hit_rate'] * 100:.2f}% "
        f"({metrics.get('zero_hit_count', 0)} queries)",
        "",
        "## Latency Distribution (ms)",
        "",
    ]

    lat = metrics.get("latency_ms", {})
    if lat:
        lines.extend(
            [
                "| Metric | Latency (ms) |",
                "| --- | --- |",
                f"| Mean | {lat.get('mean')} |",
                f"| Median (p50) | {lat.get('median')} |",
                f"| p90 | {lat.get('p90')} |",
                f"| p95 | {lat.get('p95')} |",
                f"| p99 | {lat.get('p99')} |",
                f"| Min | {lat.get('min')} |",
                f"| Max | {lat.get('max')} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Request Breakdown by Endpoint",
            "",
            "| Endpoint | Count |",
            "| --- | --- |",
        ]
    )
    for ep, count in metrics.get("endpoint_counts", {}).items():
        lines.append(f"| `{ep}` | {count} |")

    lines.extend(
        [
            "",
            "## Request Breakdown by Model",
            "",
            "| Model | Count |",
            "| --- | --- |",
        ]
    )
    for m, count in metrics.get("model_counts", {}).items():
        lines.append(f"| `{m}` | {count} |")

    lines.extend(
        [
            "",
            "## Top 10 Requested Ingredients",
            "",
            "| Ingredient | Frequency |",
            "| --- | --- |",
        ]
    )
    for ing, count in metrics.get("top_ingredients", []):
        lines.append(f"| {ing} | {count} |")

    return "\n".join(lines)


def log_monitoring_to_mlflow(
    metrics: dict[str, Any],
    *,
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment: str = "api-monitoring",
) -> str | None:
    """Log monitoring snapshot metrics to MLflow."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

        run_name = f"monitoring-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            flat_metrics: dict[str, float] = {
                "total_requests": float(metrics["total_requests"]),
                "error_rate": float(metrics["error_rate"]),
                "success_rate": float(metrics["success_rate"]),
                "zero_hit_rate": float(metrics["zero_hit_rate"]),
            }
            for k, v in metrics.get("latency_ms", {}).items():
                if isinstance(v, (int, float)):
                    flat_metrics[f"latency_{k}"] = float(v)
            if metrics.get("mean_top_score") is not None:
                flat_metrics["mean_top_score"] = float(metrics["mean_top_score"])

            mlflow.log_metrics(flat_metrics)
            return run.info.run_id
    except Exception as e:
        print(f"warning: MLflow monitoring logging failed: {e}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--out", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--mlflow", action="store_true", help="Log monitoring metrics to MLflow")
    args = parser.parse_args(argv)

    records = load_request_logs(args.log_path)
    if not records:
        print(f"info: No log records found at {args.log_path}")
        return 0

    metrics = compute_monitoring_metrics(records)
    report = format_monitoring_report(metrics)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Monitoring report saved to {out_path}")

    if args.mlflow:
        run_id = log_monitoring_to_mlflow(metrics)
        if run_id:
            print(f"Monitoring logged to MLflow run: {run_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
