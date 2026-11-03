"""Measured, controlled MiniLM ablation; only AdamW weight_decay changes.

python -m scripts.run_regularization_ablation --device cuda
The fixed train/validation probes use eval mode and identical full batch sizes.
This is a single-seed loss experiment, not a significance or retrieval-quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import mlflow
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, losses

from src.build_pairs import NoDuplicateBatchSampler, build_pairs
from src.config import load_config, processed_dir
from src.embeddings import MODEL_NAME
from src.utils import sha256_file, sha256_obj, write_json, write_jsonl


def batches(pairs, batch_size, seed):
    return [b for b in NoDuplicateBatchSampler(pairs, batch_size, seed=seed, drop_last=True)
            if len(b) == batch_size]


def features(model, pairs, indices, device):
    return [{k: v.to(device) for k, v in model.tokenize([pairs[i][field] for i in indices]).items()}
            for field in ("query", "positive")]


def measured_loss(model, loss, pairs, probe_batches, device):
    model.eval()
    with torch.no_grad():
        values = [float(loss(features(model, pairs, b, device), None)) for b in probe_batches]
    if not values or not all(np.isfinite(values)):
        raise ValueError("Loss probes must contain full batches with finite measurements")
    return sum(values) / len(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-pairs", type=int, default=1024)
    parser.add_argument("--val-pairs", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--out", default="reports/regularization_ablation")
    args = parser.parse_args(argv)
    if min(args.train_pairs, args.val_pairs, args.epochs, args.batch_size, args.eval_every, args.threads) <= 0:
        parser.error("Pair counts, epochs, batch size, eval interval and threads must be positive")
    if args.batch_size < 2 or min(args.train_pairs, args.val_pairs) < args.batch_size:
        parser.error("MNRL requires at least one batch of two or more pairs per split")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    cfg = load_config(args.config)
    directory = processed_dir(cfg)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    train, _ = build_pairs(directory, split="train", max_pairs=args.train_pairs, seed=args.seed,
                           queries_per_recipe=1)
    val, _ = build_pairs(directory, split="validation", max_pairs=args.val_pairs, seed=args.seed,
                         queries_per_recipe=1)
    if {p["group_id"] for p in train} & {p["group_id"] for p in val}:
        raise ValueError("Training and validation groups overlap")
    train_probe = batches(train[:len(val)], args.batch_size, args.seed)
    val_probe = batches(val, args.batch_size, args.seed)
    n_probe_batches = min(len(train_probe), len(val_probe))
    if not n_probe_batches:
        raise ValueError("Train/validation probes must contain full batches")
    # Group deduplication can leave different numbers of full batches in each split.
    train_probe, val_probe = train_probe[:n_probe_batches], val_probe[:n_probe_batches]
    schedule = [batches(train, args.batch_size, args.seed + epoch) for epoch in range(args.epochs)]
    if not all(schedule):
        raise ValueError("No complete training batches")
    write_jsonl(out / "train_pairs.jsonl", train)
    write_jsonl(out / "validation_pairs.jsonl", val)
    hardware = {"platform": platform.platform(), "python": platform.python_version(),
                "device": args.device, "colab": bool(os.environ.get("COLAB_RELEASE_TAG")),
                "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                "libraries": {p: version(p) for p in ("torch", "sentence-transformers", "mlflow")}}
    controls = {k: v for k, v in vars(args).items() if k not in ("out", "tracking_uri")}
    controls.update(base_model=MODEL_NAME, max_seq_length=128, optimizer="AdamW",
                    corpus_sha256=sha256_file(directory / "corpus.jsonl"),
                    splits_sha256=sha256_file(directory / "splits.json"),
                    train_pairs_sha256=sha256_file(out / "train_pairs.jsonl"),
                    validation_pairs_sha256=sha256_file(out / "validation_pairs.jsonl"),
                    batch_schedule_sha256=sha256_obj(schedule), loss="MultipleNegativesRankingLoss")
    controls.update(probe_batches=n_probe_batches, train_probe_sha256=sha256_obj(train_probe),
                    validation_probe_sha256=sha256_obj(val_probe))
    write_json(out / "controls.json", controls)
    write_json(out / "hardware.json", hardware)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("recipe-regularization-ablation")
    results = []
    initial_hash = None
    with mlflow.start_run(run_name="controlled-weight-decay") as parent:
        mlflow.log_params(controls)
        mlflow.set_tags({"controlled_ablation": "true", "environment": "colab" if hardware["colab"] else "local"})
        for wd in (0.0, 0.01):
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.device == "cuda":
                torch.cuda.manual_seed_all(args.seed)
            model = SentenceTransformer(MODEL_NAME, device=args.device)
            model.max_seq_length = 128
            digest = hashlib.sha256()
            for name, tensor in model.state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.cpu().contiguous().numpy().tobytes())
            state_hash = digest.hexdigest()
            if initial_hash is not None and state_hash != initial_hash:
                raise ValueError("Initial model weights differ between arms")
            initial_hash = state_hash
            loss = losses.MultipleNegativesRankingLoss(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd)
            history = []
            started = time.perf_counter()
            with mlflow.start_run(run_name=f"weight-decay-{wd}", nested=True) as run:
                mlflow.log_params({**controls, "weight_decay": wd, "initial_weights_sha256": state_hash})
                mlflow.set_tags({"controlled_ablation": "true", "environment": "colab" if hardware["colab"] else "local"})
                def evaluate(step, model=model, loss=loss, history=history, wd=wd):
                    row = {"step": step,
                           "train_loss": measured_loss(model, loss, train, train_probe, args.device),
                           "val_loss": measured_loss(model, loss, val, val_probe, args.device)}
                    history.append(row)
                    mlflow.log_metrics({k: v for k, v in row.items() if k != "step"}, step=step)
                    print(f"wd={wd} {row}", flush=True)
                evaluate(0)
                step = 0
                for epoch_batches in schedule:
                    for batch in epoch_batches:
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        value = loss(features(model, train, batch, args.device), None)
                        if not torch.isfinite(value):
                            raise ValueError("Non-finite training loss")
                        value.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        step += 1
                        mlflow.log_metric("optimization_loss", float(value.detach()), step=step)
                        if step % args.eval_every == 0:
                            evaluate(step)
                if history[-1]["step"] != step:
                    evaluate(step)
                model.save(str(out / f"checkpoint-wd{wd}"))
                row = {"weight_decay": wd, "run_id": run.info.run_id, "history": history,
                       "train_seconds": time.perf_counter() - started, "initial_weights_sha256": state_hash}
                write_json(out / f"wd{wd}.json", row)
                mlflow.log_metric("train_seconds", row["train_seconds"])
                mlflow.log_artifact(str(out / f"wd{wd}.json"))
                results.append(row)
            del evaluate, optimizer, loss, model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        report = {"parent_run_id": parent.info.run_id, "controls": controls, "hardware": hardware,
                  "arms": results, "scope": "Single-seed controlled loss experiment; no test-set evaluation."}
        write_json(out / "summary.json", report)
        for name in ("summary.json", "controls.json", "hardware.json", "train_pairs.jsonl", "validation_pairs.jsonl"):
            mlflow.log_artifact(str(out / name))
    print(f"Evidence: {out / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
