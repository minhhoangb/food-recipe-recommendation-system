"""Check that the native run and the Docker container serve the same results.

    python scripts/check_parity.py --native http://127.0.0.1:8010 --docker http://127.0.0.1:8020

The same queries are sent to both servers and the answers are compared: the same recipe ids
in the same order, and the same scores within a tolerance (the scores of two runs on
different machines are never bit-identical). The result is written to
`reports/parity.json` and printed as a table.

The log of the check is the report itself: every query, both servers and the difference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import REPO_ROOT, write_json  # noqa: E402

DEFAULT_QUERIES = [
    ["chicken", "rice", "garlic"],
    ["pasta", "tomato", "basil"],
    ["egg", "milk", "flour"],
    ["beef", "onion", "carrot"],
    ["rice", "soy sauce"],
]
DEFAULT_TOLERANCE = 1e-6


def predict(base_url: str, ingredients: list[str], top_k: int) -> dict:
    """One /predict call, with the answer the server gave (or the error it gave)."""
    import httpx

    response = httpx.post(
        f"{base_url.rstrip('/')}/predict",
        json={"ingredients": ingredients, "top_k": top_k},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def compare(native: dict, docker: dict, tolerance: float) -> dict:
    """Are the two answers the same? (ids in the same order, scores within the tolerance)."""
    native_results = native.get("results", [])
    docker_results = docker.get("results", [])
    native_ids = [r["recipe_id"] for r in native_results]
    docker_ids = [r["recipe_id"] for r in docker_results]
    score_diffs = [
        abs(a["score"] - b["score"]) for a, b in zip(native_results, docker_results, strict=False)
    ]
    max_diff = max(score_diffs, default=0.0)
    return {
        "same_ids": native_ids == docker_ids,
        "n_results": len(native_ids),
        "max_score_diff": max_diff,
        "within_tolerance": max_diff <= tolerance,
        "ok": native_ids == docker_ids and max_diff <= tolerance,
    }


def model_info(base_url: str) -> dict:
    """What the server says it is serving."""
    import httpx

    return httpx.get(f"{base_url.rstrip('/')}/model-info", timeout=120.0).json()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--native", default="http://127.0.0.1:8010")
    parser.add_argument("--docker", default="http://127.0.0.1:8020")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--out", default="reports/parity.json")
    args = parser.parse_args(argv)

    native_info = model_info(args.native)
    docker_info = model_info(args.docker)

    rows = []
    for ingredients in DEFAULT_QUERIES:
        native = predict(args.native, ingredients, args.top_k)
        docker = predict(args.docker, ingredients, args.top_k)
        row = {
            "ingredients": ingredients,
            "native_ids": [r["recipe_id"] for r in native.get("results", [])],
            "docker_ids": [r["recipe_id"] for r in docker.get("results", [])],
            **compare(native, docker, args.tolerance),
        }
        rows.append(row)
        status = "OK " if row["ok"] else "DIFF"
        print(
            f"{status} {', '.join(ingredients):<28} {row['n_results']} results, "
            f"max score diff {row['max_score_diff']:.3e}"
        )

    payload = {
        "native": {"url": args.native, "model_info": native_info},
        "docker": {"url": args.docker, "model_info": docker_info},
        "top_k": args.top_k,
        "tolerance": args.tolerance,
        "queries": rows,
        "all_ok": all(row["ok"] for row in rows),
    }
    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    write_json(out, payload)
    print(f"\n-> {out}")
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
