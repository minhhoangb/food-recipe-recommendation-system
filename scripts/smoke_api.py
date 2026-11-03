"""Exercise the actual API over HTTP, including inside the running Docker container."""
from __future__ import annotations

import argparse
import json
import platform
import urllib.error
import urllib.request
from datetime import datetime, timezone


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--expected-model", default="bm25")
    args = parser.parse_args(argv)

    def request(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(args.url.rstrip("/") + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)

    health = request("/health")
    info = request("/model-info")
    assert health["status"] == "ok", health
    assert health["model"] == info["model"] == args.expected_model, info
    query = {"ingredients": ["chicken", "rice", "garlic"], "top_k": 5}
    prediction = request("/predict", query)
    assert prediction["model"] == args.expected_model, prediction
    assert len(prediction["results"]) == 5, prediction
    ids = [r["recipe_id"] for r in prediction["results"]]
    assert len(set(ids)) == 5
    batch = request("/batch-predict", {"instances": [query, {**query, "top_k": 2}]})
    assert len(batch["predictions"]) == 2, batch
    assert [r["recipe_id"] for r in batch["predictions"][0]["results"]] == ids
    assert [r["recipe_id"] for r in batch["predictions"][1]["results"]] == ids[:2]
    try:
        request("/predict", {**query, "top_k": 0})
    except urllib.error.HTTPError as error:
        assert error.code == 422, error.code
    else:
        raise AssertionError("Invalid top_k did not return 422")
    print(json.dumps({"all_ok": True, "verified_utc": datetime.now(timezone.utc).isoformat(),
                      "platform": platform.platform(), "url": args.url, "health": health,
                      "model_info": info, "predict_ids": ids, "batch_sizes": [5, 2],
                      "invalid_top_k_status": 422}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
