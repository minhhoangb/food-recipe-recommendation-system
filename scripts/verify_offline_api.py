"""Verify real HTTP API serving with the MLflow tracking endpoint unavailable."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import httpx

from src.config import load_config, processed_dir
from src.serving import bundle_manifest
from src.utils import REPO_ROOT, read_jsonl, sha256_file, write_json


def verify_offline_api(bundle: Path, config: str = "configs/default.yaml") -> dict:
    """Start a fresh API, send HTTP requests, then stop it even if verification fails."""
    manifest = bundle_manifest(bundle)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    logs = REPO_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    environment = {
        **os.environ,
        "API_BACKEND": "bundle",
        "API_BUNDLE": str(bundle.resolve()),
        "API_CONFIG": config,
        "API_REQUEST_LOG": str(logs / "offline_api.db"),
        "API_TRACKING_URI": "http://127.0.0.1:1",
        "MLFLOW_TRACKING_URI": "http://127.0.0.1:1",
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0",
        "MLFLOW_HTTP_REQUEST_TIMEOUT": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    with (logs / "offline_api_server.log").open("w", encoding="utf-8") as output:
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.app:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=REPO_ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15.0) as client:
                deadline = time.monotonic() + 90
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("API startup failed; see logs/offline_api_server.log")
                    try:
                        health = client.get("/health")
                        health.raise_for_status()
                        break
                    except httpx.TransportError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("API did not become healthy") from None
                        time.sleep(0.25)
                prediction = client.post(
                    "/predict", json={"ingredients": ["chicken", "rice", "garlic"], "top_k": 3}
                )
                prediction.raise_for_status()
                model_info = client.get("/model-info")
                model_info.raise_for_status()
                body = prediction.json()
                if not body["results"] or body["model"] != manifest["model_name"]:
                    raise RuntimeError("Offline API returned no results or used the wrong model")
                directory = processed_dir(load_config(config))
                queries = islice(
                    (q for q in read_jsonl(directory / "queries.jsonl") if q["split"] == "validation"), 19
                )
                requests = [
                    {
                        "ingredients": ["chicken", "rice", "garlic"],
                        "status": prediction.status_code,
                        "n_results": len(body["results"]),
                    }
                ]
                for query in queries:
                    response = client.post("/predict", json={"ingredients": query["ingredients"], "top_k": 3})
                    response.raise_for_status()
                    requests.append(
                        {
                            "ingredients": query["ingredients"],
                            "status": response.status_code,
                            "n_results": len(response.json()["results"]),
                        }
                    )
                return {
                    "success": True,
                    "checked_utc": datetime.now(timezone.utc).isoformat(),
                    "transport": "HTTP to a fresh uvicorn subprocess",
                    "tracking_uri": environment["API_TRACKING_URI"],
                    "bundle": bundle.name,
                    "manifest_sha256": sha256_file(bundle / "manifest.json"),
                    "encoder_weights_sha256": manifest.get("revision"),
                    "health": health.json(),
                    "model_info": model_info.json(),
                    "predict_status": prediction.status_code,
                    "prediction": body,
                    "prediction_requests": requests,
                }
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    evidence = verify_offline_api(args.bundle, args.config)
    path = REPO_ROOT / "reports" / "offline_api_verification.json"
    write_json(path, evidence)
    print(path)
