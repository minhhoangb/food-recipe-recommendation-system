"""Benchmark the serving bundles: latency, RAM, startup time and bundle size.

    python -m src.benchmark --config configs/default.yaml --out reports/benchmark.json

Every bundle under `models/` (or the ones given with `--bundle`) is loaded, warmed up and then
timed: at least 20 warmup queries and 100 measured queries, one after the other, with
`time.perf_counter`. The machine (CPU, RAM) is written down, because the numbers are only
valid for that machine.

Each bundle is measured in its own child process, because peak RSS is a *process* high-water
mark: measuring several bundles in one process would report the largest of them all as the
cost of each. The queries are the validation queries of the configured profile, and a bundle
built from another corpus is skipped rather than timed on queries that do not belong to it.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.config import load_config, processed_dir
from src.evaluation import load_split
from src.memory import peak_rss_mb
from src.serving import bundle_kind, bundle_manifest, load_retriever
from src.utils import REPO_ROOT, read_json, sha256_file, write_json

WARMUP_QUERIES = 20
MEASURED_QUERIES = 100
DEFAULT_TOP_K = 10


def machine_info() -> dict:
    """CPU, RAM and OS of the machine the numbers were measured on."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import psutil

        info["cpu_logical_cores"] = psutil.cpu_count(logical=True)
        info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
        info["cpu_max_mhz"] = psutil.cpu_freq().max if psutil.cpu_freq() else None
    except Exception:
        pass
    return info


def percentile(values: list[float], fraction: float) -> float:
    """The `fraction` percentile of `values` (nearest-rank, no interpolation)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(fraction * len(ordered))) - 1))
    return ordered[index]


def bundle_size_mb(bundle: Path) -> float:
    """Size of the whole bundle folder in MB."""
    return round(sum(f.stat().st_size for f in bundle.rglob("*") if f.is_file()) / 1024**2, 2)


def find_bundles(models_dir: Path, explicit: list[str]) -> list[Path]:
    """The bundles to benchmark: the ones asked for, or every bundle folder that has a manifest."""
    if explicit:
        return [Path(p) if Path(p).is_absolute() else REPO_ROOT / p for p in explicit]
    return sorted(
        bundle
        for bundle in models_dir.iterdir()
        if bundle.is_dir() and (bundle / "manifest.json").exists()
    )


def benchmark_bundle(bundle: Path, queries: list[dict], top_k: int = DEFAULT_TOP_K,
                     warmup: int = WARMUP_QUERIES, measured: int = MEASURED_QUERIES) -> dict:
    """Load one bundle, time it, and report what the machine did (this process only)."""
    started = time.perf_counter()
    retriever = load_retriever(bundle)
    startup_seconds = time.perf_counter() - started

    ingredient_lists = [q["ingredients"] for q in queries]
    for ingredients in ingredient_lists[:warmup]:
        retriever.search(ingredients, top_k)

    latencies: list[float] = []
    for ingredients in ingredient_lists[:measured]:
        query_started = time.perf_counter()
        retriever.search(ingredients, top_k)
        latencies.append((time.perf_counter() - query_started) * 1000)

    manifest = bundle_manifest(bundle)
    metrics_path = bundle / "metrics.json"
    return {
        "bundle": bundle.name,
        "model": manifest.get("model_name"),
        "model_version": retriever.version,
        "backend": bundle_kind(bundle),
        "corpus_sha256": manifest.get("corpus_sha256"),
        "startup_seconds": round(startup_seconds, 3),
        "peak_rss_mb": peak_rss_mb(),
        "bundle_size_mb": bundle_size_mb(bundle),
        "top_k": top_k,
        "n_warmup_queries": min(warmup, len(ingredient_lists)),
        "n_measured_queries": len(latencies),
        "latency_ms_p50": round(percentile(latencies, 0.50), 3),
        "latency_ms_p95": round(percentile(latencies, 0.95), 3),
        "latency_ms_mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "latency_ms_min": round(min(latencies), 3) if latencies else 0.0,
        "latency_ms_max": round(max(latencies), 3) if latencies else 0.0,
        "fit_seconds": read_json(metrics_path).get("fit_seconds") if metrics_path.exists() else None,
    }


def _benchmark_in_child(bundle: Path, config: str, split: str, top_k: int, warmup: int,
                        measured: int) -> dict:
    """Run `benchmark_bundle` in a fresh interpreter so peak RSS belongs to that bundle alone."""
    command = [
        sys.executable, "-m", "src.benchmark",
        "--config", config, "--split", split, "--top-k", str(top_k),
        "--warmup", str(warmup), "--queries", str(measured),
        "--bundle", str(bundle), "--child",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"{bundle.name}: benchmark child failed\n{completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def markdown_table(records: list[dict]) -> str:
    """The measured numbers as a markdown table (the same values as in benchmark.json)."""
    lines = [
        "| Model | Bundle | Backend | p50 latency (ms) | p95 latency (ms) | Startup (s) | "
        "Peak RSS (MB) | Bundle size (MB) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| {model} | `{bundle}` | {backend} | {p50:.2f} | {p95:.2f} | {startup:.2f} | "
            "{rss} | {size:.2f} |".format(
                model=record["model"],
                bundle=record["bundle"],
                backend=record["backend"],
                p50=record["latency_ms_p50"],
                p95=record["latency_ms_p95"],
                startup=record["startup_seconds"],
                rss=record["peak_rss_mb"],
                size=record["bundle_size_mb"],
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--bundle", action="append", default=[], help="models/<run_id> (repeatable)")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--warmup", type=int, default=WARMUP_QUERIES)
    parser.add_argument("--queries", type=int, default=MEASURED_QUERIES)
    parser.add_argument("--out", default="reports/benchmark.json")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    directory = processed_dir(cfg)
    queries, _ = load_split(directory, args.split)
    if not queries:
        print(f"no {args.split} queries in {directory}")
        return 1

    corpus_sha = sha256_file(directory / "corpus.jsonl")
    models_dir = Path(args.models_dir) if Path(args.models_dir).is_absolute() else REPO_ROOT / args.models_dir

    if args.child:  # one bundle, in its own process, printed as one JSON line for the parent
        bundle = Path(args.bundle[0])
        print(json.dumps(benchmark_bundle(bundle, queries, args.top_k, args.warmup, args.queries)))
        return 0

    records = []
    for bundle in find_bundles(models_dir, args.bundle):
        if bundle_manifest(bundle).get("corpus_sha256") != corpus_sha:
            print(f"skip {bundle.name}: built from another corpus than {args.config}")
            continue
        record = _benchmark_in_child(bundle, args.config, args.split, args.top_k, args.warmup, args.queries)
        records.append(record)
        print(
            f"{record['model']:<20} p50 {record['latency_ms_p50']:>8.2f} ms  "
            f"p95 {record['latency_ms_p95']:>8.2f} ms  "
            f"startup {record['startup_seconds']:>6.2f} s  "
            f"peak RSS {str(record['peak_rss_mb']):>7} MB  "
            f"bundle {record['bundle_size_mb']:>7} MB  ({record['bundle']})"
        )

    payload = {
        "machine": machine_info(),
        "config": args.config,
        "profile": str(cfg.data.profile),
        "split": args.split,
        "top_k": args.top_k,
        "n_queries_available": len(queries),
        "n_warmup_queries": args.warmup,
        "n_measured_queries": args.queries,
        "bundles": records,
    }
    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    write_json(out, payload)
    table_path = out.with_suffix(".md")
    machine = payload["machine"]
    table_path.write_text(
        f"# Benchmark ({payload['profile']}/{payload['split']}, top_k={args.top_k})\n\n"
        f"Measured on {machine['platform']}, {machine['processor']}, "
        f"{machine.get('cpu_physical_cores', '?')} physical cores, "
        f"{machine.get('ram_total_gb', '?')} GB RAM, Python {machine['python']}. "
        f"{args.warmup} warmup queries, {args.queries} measured queries, one after the other.\n\n"
        + markdown_table(records)
        + "\n",
        encoding="utf-8",
    )
    print(f"\n-> {out}\n-> {table_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
