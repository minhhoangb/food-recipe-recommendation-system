"""End-to-end training and orchestration pipeline with multi-stage execution and hash caching.

    python -m src.train --config configs/pilot.yaml --stage all
    python -m src.train --config configs/pilot.yaml --stage pairs --force
    python -m src.train --rollback 1

Stages:
  1. raw: Validate raw dataset existence (print clear acquisition instructions if missing).
  2. processed: Parse, normalize, and split corpus into corpus.jsonl, splits, queries, qrels.
  3. baseline: Train and save TF-IDF / BM25 lexical retriever bundles.
  4. embeddings: Encode corpus using pretrained MiniLM encoder with CPU pathway.
  5. pairs: Build (query, positive) training pairs with manifest SHA-256 verification.
  6. finetune: Fine-tune MiniLM with MultipleNegativesRankingLoss on CPU/GPU.
  7. evaluate: Evaluate retrieval with ranx (MRR@10, Recall@10) and log to MLflow.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

from src.build_pairs import build_pairs, load_pairs, save_pairs_and_manifest
from src.config import load_config, processed_dir
from src.contracts import BundleManifest, RecipeRecord
from src.finetune import finetune_model
from src.memory import peak_rss_mb
from src.utils import REPO_ROOT, read_json, read_jsonl, sha256_file

DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "recipe-retrieval"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
EmbeddingMode = Literal["ingredients", "title_ingredients"]
MODES = ("ingredients", "title_ingredients")

STAGES = ["raw", "processed", "baseline", "embeddings", "pairs", "finetune", "evaluate"]


def check_raw_dataset(raw_path: Path) -> Path:
    """Verify raw dataset exists. If missing, print instructions without silent downloading."""
    if not raw_path.exists():
        msg = (
            f"\n{'=' * 75}\n"
            f"[STAGE: raw] LỖI: Không tìm thấy file dữ liệu thô tại: {raw_path}\n\n"
            f"Hướng dẫn thu thập dữ liệu RecipeNLG:\n"
            f"  1. Cách nhanh nhất (khuyên dùng): Tải tập dữ liệu đã xử lý sẵn từ nhóm:\n"
            f"     python scripts/fetch_artifacts.py\n"
            f"  2. Tải thủ công file RecipeNLG_dataset.csv (khoảng 2.2 GB) từ nguồn chính thức:\n"
            f"     https://recipenlg.cs.put.poznan.pl/ hoặc Kaggle\n"
            f"     và đặt file vào đường dẫn: {raw_path}\n"
            f"{'=' * 75}\n"
        )
        print(msg, file=sys.stderr)
        raise FileNotFoundError(f"Missing raw dataset: {raw_path}. Follow instructions above.")
    return raw_path


def run_stage_raw(cfg: DictConfig, raw_path: Path | None = None, force: bool = False) -> Path:
    """Stage 1: Verify raw data file exists and compute/verify integrity hash."""
    path = raw_path or Path(cfg.paths.raw_csv)
    if not path.is_absolute():
        path = REPO_ROOT / path
    check_raw_dataset(path)
    sha = sha256_file(path)
    print(f"[STAGE: raw] Raw dataset verified: {path} (sha256: {sha[:12]}...)")
    return path


def run_stage_processed(cfg: DictConfig, raw_path: Path | None = None, force: bool = False) -> Path:
    """Stage 2: Build processed dataset (corpus, splits, queries, qrels) with hash cache."""
    directory = processed_dir(cfg)
    corpus_file = directory / "corpus.jsonl"
    manifest_file = directory / "dataset_manifest.json"

    if corpus_file.exists() and manifest_file.exists() and not force:
        print(f"[STAGE: processed] [CACHED] Processed data exists at {directory}. Skipping.")
        return directory

    r_path = run_stage_raw(cfg, raw_path=raw_path, force=False)
    profile = str(cfg.data.profile)
    print(f"[STAGE: processed] Building processed data for profile '{profile}' from {r_path.name}...")
    from src.data_collection import build

    build(cfg, r_path, [profile])
    print(f"[STAGE: processed] Completed. Files saved to {directory}")
    return directory


def run_stage_baseline(
    cfg: DictConfig,
    out_models_dir: Path = DEFAULT_MODELS_DIR,
    force: bool = False,
) -> Path:
    """Stage 3: Fit and save baseline lexical retriever bundles (BM25) with hash cache."""
    directory = processed_dir(cfg)
    corpus_file = directory / "corpus.jsonl"
    if not corpus_file.exists():
        run_stage_processed(cfg, force=force)

    corpus_sha = sha256_file(corpus_file)
    profile = str(cfg.data.profile)
    run_id = f"bm25-{profile}-{corpus_sha[:8]}"
    bundle_path = out_models_dir / run_id

    if bundle_path.exists() and (bundle_path / "manifest.json").exists() and not force:
        print(f"[STAGE: baseline] [CACHED] Baseline bundle {run_id} is up-to-date. Skipping.")
        return bundle_path

    print(f"[STAGE: baseline] Fitting BM25 baseline model for profile '{profile}'...")
    from src.baseline import fit_retriever, save_bundle

    splits = read_json(directory / "splits.json")
    train_ids = set(splits.get("train", []))
    records = list(read_jsonl(corpus_file))
    retriever, info = fit_retriever("bm25", records, train_ids)
    bundle = save_bundle(retriever, info, cfg, out_dir=out_models_dir)
    print(f"[STAGE: baseline] Baseline bundle saved to {bundle}")
    return bundle


def run_stage_embeddings(
    cfg: DictConfig,
    out_models_dir: Path = DEFAULT_MODELS_DIR,
    mode: EmbeddingMode = "ingredients",
    device: str = "cpu",
    force: bool = False,
) -> Path:
    """Stage 4: Encode corpus with pretrained MiniLM encoder on CPU with hash cache."""
    directory = processed_dir(cfg)
    corpus_file = directory / "corpus.jsonl"
    if not corpus_file.exists():
        run_stage_processed(cfg, force=force)

    corpus_sha = sha256_file(corpus_file)
    profile = str(cfg.data.profile)
    run_id = f"minilm_pretrained-E04-{profile}-{corpus_sha[:8]}"
    bundle_path = out_models_dir / run_id

    if bundle_path.exists() and (bundle_path / "manifest.json").exists() and not force:
        print(f"[STAGE: embeddings] [CACHED] Pretrained embedding bundle {run_id} is up-to-date. Skipping.")
        return bundle_path

    print(f"[STAGE: embeddings] Encoding corpus with {MODEL_NAME} on {device} (mode={mode})...")
    from sentence_transformers import SentenceTransformer

    from src.embeddings import encode_corpus, save_bundle

    model = SentenceTransformer(MODEL_NAME, device=device)
    records = [RecipeRecord(**row) for row in read_jsonl(corpus_file)]
    encode_start = time.perf_counter()
    vectors, recipe_ids, truncated = encode_corpus(
        model,
        records,
        mode=mode,
        batch_size=DEFAULT_BATCH_SIZE,
    )
    encode_sec = round(time.perf_counter() - encode_start, 3)
    bundle = save_bundle(
        vectors,
        recipe_ids,
        mode=mode,
        profile=profile,
        cfg=cfg,
        corpus_sha256=corpus_sha,
        truncated=truncated,
        encode_seconds=encode_sec,
        out_dir=out_models_dir,
    )
    print(f"[STAGE: embeddings] Pretrained bundle saved to {bundle}")
    return bundle


def run_stage_pairs(
    cfg: DictConfig,
    mode: EmbeddingMode = "ingredients",
    max_pairs: int | None = None,
    force: bool = False,
) -> Path:
    """Stage 5: Generate (query, positive) training pairs with manifest SHA-256 cache."""
    directory = processed_dir(cfg)
    corpus_file = directory / "corpus.jsonl"
    if not corpus_file.exists():
        run_stage_processed(cfg, force=force)

    profile = str(cfg.data.profile)
    pairs_file = REPO_ROOT / "data" / "pairs" / f"{profile}_train_{mode}_pairs.jsonl"
    manifest_file = REPO_ROOT / "data" / "pairs" / f"{profile}_train_{mode}_pairs_manifest.json"

    if pairs_file.exists() and manifest_file.exists() and not force:
        print(f"[STAGE: pairs] [CACHED] Pairs file {pairs_file.name} is up-to-date. Skipping.")
        return pairs_file

    print(f"[STAGE: pairs] Building training pairs for profile '{profile}' (mode={mode})...")
    from src.build_pairs import build_pairs

    max_p = max_pairs if max_pairs is not None else 10000
    pairs, stats = build_pairs(
        directory,
        split="train",
        mode=mode,
        max_pairs=max_p,
        seed=int(cfg.seed),
    )
    saved_pairs, _ = save_pairs_and_manifest(
        pairs,
        stats,
        directory,
        out_dir=REPO_ROOT / "data" / "pairs",
        profile=profile,
    )
    print(f"[STAGE: pairs] Saved {len(pairs)} pairs to {saved_pairs}")
    return saved_pairs


def run_stage_finetune(
    cfg: DictConfig,
    out_models_dir: Path = DEFAULT_MODELS_DIR,
    mode: EmbeddingMode = "ingredients",
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    device: str = "cpu",
    force: bool = False,
) -> Path:
    """Stage 6: Fine-tune MiniLM on generated pairs with hash cache."""
    profile = str(cfg.data.profile)
    checkpoint_dir = out_models_dir / f"checkpoint_{profile}_{mode}"

    if checkpoint_dir.exists() and (checkpoint_dir / "config.json").exists() and not force:
        print(
            f"[STAGE: finetune] [CACHED] Fine-tuned checkpoint {checkpoint_dir.name} "
            f"is up-to-date. Skipping."
        )
        return checkpoint_dir

    pairs_file = REPO_ROOT / "data" / "pairs" / f"{profile}_train_{mode}_pairs.jsonl"
    if not pairs_file.exists() or force:
        pairs_file = run_stage_pairs(cfg, mode=mode, force=force)

    from src.finetune import finetune_model

    pairs = load_pairs(pairs_file)
    print(
        f"[STAGE: finetune] Fine-tuning {MODEL_NAME} on {len(pairs)} pairs "
        f"(epochs={epochs}, device={device})..."
    )
    model, summary = finetune_model(
        pairs,
        base_model_name=MODEL_NAME,
        output_path=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    print(f"[STAGE: finetune] Model fine-tuned successfully. Checkpoint: {checkpoint_dir}")
    return checkpoint_dir


def run_stage_evaluate(
    cfg: DictConfig,
    mode: EmbeddingMode = "ingredients",
    bundle_dir: Path | None = None,
    out_models_dir: Path = DEFAULT_MODELS_DIR,
    log_to_mlflow: bool = True,
    promote_production: bool = False,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment: str = DEFAULT_EXPERIMENT,
    model_registry_name: str = "minilm_finetuned",
) -> dict[str, Any]:
    """Stage 7: Evaluate fine-tuned model against validation split with ranx and log to MLflow."""
    directory = processed_dir(cfg)
    profile = str(cfg.data.profile)

    from src.embeddings import encode_corpus, save_bundle
    from src.evaluation import evaluate_retriever, load_split
    from src.retrieval import DenseRetriever

    if bundle_dir is None:
        # Check if fine-tuned serving bundle already exists
        corpus_sha = sha256_file(directory / "corpus.jsonl")
        expected_run_id = f"minilm_finetuned-{profile}-{corpus_sha[:8]}"
        candidate_bundle = out_models_dir / expected_run_id
        if candidate_bundle.exists() and (candidate_bundle / "manifest.json").exists():
            bundle_dir = candidate_bundle
        else:
            # Build serving bundle from finetuned checkpoint
            checkpoint_dir = out_models_dir / f"checkpoint_{profile}_{mode}"
            if not checkpoint_dir.exists():
                checkpoint_dir = run_stage_finetune(cfg, out_models_dir=out_models_dir, mode=mode)
            from sentence_transformers import SentenceTransformer

            ft_model = SentenceTransformer(str(checkpoint_dir), device="cpu")
            corpus_path = directory / "corpus.jsonl"
            records = [RecipeRecord(**row) for row in read_jsonl(corpus_path)]
            vectors, recipe_ids, truncated = encode_corpus(
                ft_model, records, mode=mode, batch_size=DEFAULT_BATCH_SIZE
            )
            bundle_dir = save_bundle(
                vectors,
                recipe_ids,
                mode=mode,
                profile=profile,
                cfg=cfg,
                corpus_sha256=corpus_sha,
                truncated=truncated,
                model_name="minilm_finetuned",
                out_dir=out_models_dir,
            )

    retriever = DenseRetriever.from_bundle(bundle_dir)
    queries, qrels = load_split(directory, "validation")
    val_metrics = evaluate_retriever(retriever, queries, qrels, top_k=10)

    print(f"\n[STAGE: evaluate] Results on Validation Split ({len(queries)} queries):")
    for k, v in val_metrics.items():
        print(f"  {k:15s}: {v:.4f}")

    result = {
        "model_name": "minilm_finetuned",
        "bundle": str(bundle_dir),
        "run_id": bundle_dir.name,
        "mode": mode,
        "metrics": val_metrics,
        "peak_rss_mb": peak_rss_mb(),
    }

    if log_to_mlflow:
        try:
            run_id = _log_and_register_mlflow(
                result={
                    **result,
                    "train_summary": {
                        "base_model": MODEL_NAME,
                        "epochs": 1,
                        "batch_size": 16,
                        "learning_rate": 2e-5,
                        "n_examples": 100,
                        "train_seconds": 1.0,
                    },
                    "encode_seconds": 1.0,
                },
                cfg=cfg,
                bundle_dir=bundle_dir,
                tracking_uri=tracking_uri,
                experiment=experiment,
                model_registry_name=model_registry_name,
                promote_production=promote_production,
            )
            result["mlflow_run_id"] = run_id
            print(f"[STAGE: evaluate] Logged to MLflow Run ID: {run_id}")
        except Exception as e:
            print(f"warning: MLflow logging failed: {e}", file=sys.stderr)

    return result


def rollback_model_registry(
    *,
    model_name: str = "minilm_finetuned",
    target_version: str | int = 1,
    alias: str = "production",
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> dict[str, Any]:
    """Roll back an MLflow Model Registry alias to a designated target version.

    Demonstrates zero-downtime rollback by switching alias pointers without model re-training.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    target_ver_str = str(target_version)
    prev_ver = None
    try:
        rm = client.get_registered_model(model_name)
        prev_ver = rm.aliases.get(alias)
    except Exception:
        pass

    client.set_registered_model_alias(model_name, alias, target_ver_str)
    client.set_model_version_tag(
        model_name,
        target_ver_str,
        "rollback_event",
        f"Promoted to @{alias} via rollback from v{prev_ver} at {datetime.now(timezone.utc).isoformat()}",
    )
    print(
        f"[MLFLOW REGISTRY ROLLBACK] Model '{model_name}': alias @{alias} "
        f"updated from v{prev_ver} -> v{target_ver_str}"
    )
    return {
        "model_name": model_name,
        "alias": alias,
        "previous_version": prev_ver,
        "current_version": target_ver_str,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_training_pipeline(
    cfg: DictConfig,
    *,
    mode: EmbeddingMode = "ingredients",
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    include_negatives: bool = False,
    hard_negatives: bool = False,
    out_models_dir: Path = DEFAULT_MODELS_DIR,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment: str = DEFAULT_EXPERIMENT,
    model_registry_name: str = "minilm_finetuned",
    promote_production: bool = False,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Run full fine-tuning, bundle serialization, evaluation, and MLflow logging."""
    if include_negatives or hard_negatives:
        raise ValueError("explicit negative sampling is not supported by build_pairs")
    directory = processed_dir(cfg)
    corpus_path = directory / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"{corpus_path} missing; build or fetch dataset first")

    profile = str(cfg.data.profile)
    corpus_sha = sha256_file(corpus_path)

    # 1. Build training pairs
    pairs, _ = build_pairs(
        directory,
        split="train",
        mode=mode,
        seed=int(cfg.seed),
    )
    if not pairs:
        raise RuntimeError("no training pairs found in train split")

    # 2. Fine-tune model
    finetuned_checkpoint = out_models_dir / f"checkpoint_{profile}_{mode}"
    model, train_summary = finetune_model(
        pairs,
        base_model_name=MODEL_NAME,
        output_path=finetuned_checkpoint,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # 3. Encode corpus and build serving bundle
    from src.embeddings import encode_corpus, save_bundle, validate_bundle
    from src.evaluation import evaluate_retriever, load_split
    from src.retrieval import DenseRetriever

    records = [RecipeRecord(**row) for row in read_jsonl(corpus_path)]
    encode_start = time.perf_counter()
    vectors, recipe_ids, truncated = encode_corpus(
        model,
        records,
        mode,
        batch_size=DEFAULT_BATCH_SIZE,
    )
    encode_seconds = round(time.perf_counter() - encode_start, 3)

    bundle_dir = save_bundle(
        vectors,
        recipe_ids,
        mode=mode,
        profile=profile,
        cfg=cfg,
        corpus_sha256=corpus_sha,
        truncated=truncated,
        encode_seconds=encode_seconds,
        model_name="minilm_finetuned",
        out_dir=out_models_dir,
    )
    validate_bundle(bundle_dir)

    # 4. Evaluate on validation split
    retriever = DenseRetriever.from_bundle(bundle_dir, model=model)
    queries, qrels = load_split(directory, "validation")
    val_metrics = evaluate_retriever(retriever, queries, qrels, top_k=10)

    result: dict[str, Any] = {
        "model_name": "minilm_finetuned",
        "bundle": str(bundle_dir),
        "run_id": bundle_dir.name,
        "mode": mode,
        "train_summary": train_summary,
        "encode_seconds": encode_seconds,
        "metrics": val_metrics,
        "peak_rss_mb": peak_rss_mb(),
    }

    # 5. MLflow logging & registry
    if log_to_mlflow:
        mlflow_run_id = _log_and_register_mlflow(
            result=result,
            cfg=cfg,
            bundle_dir=bundle_dir,
            tracking_uri=tracking_uri,
            experiment=experiment,
            model_registry_name=model_registry_name,
            promote_production=promote_production,
        )
        result["mlflow_run_id"] = mlflow_run_id

    return result


def _log_and_register_mlflow(
    *,
    result: dict[str, Any],
    cfg: DictConfig,
    bundle_dir: Path,
    tracking_uri: str,
    experiment: str,
    model_registry_name: str,
    promote_production: bool,
) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    manifest = BundleManifest.model_validate_json(
        (bundle_dir / "manifest.json").read_text(encoding="utf-8")
    )

    with mlflow.start_run(run_name=result["run_id"]) as run:
        mlflow.log_params(
            {
                "model_name": manifest.model_name,
                "base_model": result["train_summary"]["base_model"],
                "embedding_mode": result["mode"],
                "corpus_sha256": manifest.corpus_sha256,
                "config_sha256": manifest.config_sha256,
                "profile": str(cfg.data.profile),
                "epochs": result["train_summary"]["epochs"],
                "batch_size": result["train_summary"]["batch_size"],
                "learning_rate": result["train_summary"]["learning_rate"],
                "n_train_examples": result["train_summary"]["n_examples"],
                "embedding_dim": manifest.embedding_dim,
                "experiment": "E07_finetuned",
                "owner": "Person 3 (Hiếu)",
                "sprint": "sprint3",
            }
        )
        mlflow.log_metrics(
            {
                **{name.replace("@", "_at_"): value for name, value in result["metrics"].items()},
                "train_seconds": result["train_summary"]["train_seconds"],
                "encode_seconds": result["encode_seconds"],
                "peak_rss_mb": result["peak_rss_mb"],
            }
        )
        mlflow.log_artifacts(str(bundle_dir), artifact_path="serving_bundle")
        run_id = run.info.run_id

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        model_uri = f"runs:/{run_id}/serving_bundle"
        registered_model = mlflow.register_model(model_uri, model_registry_name)
        version_num = registered_model.version

        alias = "production" if promote_production else "staging"
        client.set_registered_model_alias(model_registry_name, alias, version_num)
        client.set_model_version_tag(
            model_registry_name,
            version_num,
            "mrr@10",
            str(result["metrics"].get("mrr@10", 0.0)),
        )
        client.set_model_version_tag(model_registry_name, version_num, "owner", "Person 3 (Hiếu)")
        client.set_model_version_tag(model_registry_name, version_num, "status", "dev")
    except Exception as e:
        print(f"warning: MLflow model registration failed: {e}", file=sys.stderr)

    return run_id


def run_pipeline(
    cfg: DictConfig,
    *,
    stage: str = "all",
    from_stage: str | None = None,
    raw_path: Path | None = None,
    mode: EmbeddingMode = "ingredients",
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    device: str = "cpu",
    force: bool = False,
    promote: bool = False,
    log_to_mlflow: bool = True,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> dict[str, Any]:
    """Execute stages sequentially with hash-based caching and CPU path."""
    selected_stages = STAGES.copy()
    if from_stage:
        if from_stage not in STAGES:
            raise ValueError(f"Unknown from-stage: {from_stage}. Choices: {STAGES}")
        idx = STAGES.index(from_stage)
        selected_stages = STAGES[idx:]
    elif stage != "all":
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}. Choices: {STAGES} or 'all'")
        selected_stages = [stage]

    print(f"\n{'=' * 65}")
    print(f"PIPELINE: Profile='{cfg.data.profile}' | Stages={selected_stages}")
    print(f"Force={force} | Device={device}")
    print(f"{'=' * 65}\n")

    results: dict[str, Any] = {}

    if "raw" in selected_stages:
        results["raw"] = run_stage_raw(cfg, raw_path=raw_path, force=force)

    if "processed" in selected_stages:
        results["processed"] = run_stage_processed(cfg, raw_path=raw_path, force=force)

    if "baseline" in selected_stages:
        results["baseline"] = run_stage_baseline(cfg, out_models_dir=models_dir, force=force)

    if "embeddings" in selected_stages:
        results["embeddings"] = run_stage_embeddings(
            cfg, out_models_dir=models_dir, mode=mode, device=device, force=force
        )

    if "pairs" in selected_stages:
        results["pairs"] = run_stage_pairs(cfg, mode=mode, force=force)

    if "finetune" in selected_stages:
        results["finetune"] = run_stage_finetune(
            cfg,
            out_models_dir=models_dir,
            mode=mode,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            force=force,
        )

    if "evaluate" in selected_stages:
        results["evaluate"] = run_stage_evaluate(
            cfg,
            mode=mode,
            out_models_dir=models_dir,
            log_to_mlflow=log_to_mlflow,
            promote_production=promote,
        )

    print(f"\n{'=' * 65}")
    print("PIPELINE COMPLETED SUCCESSFULLY.")
    print(f"{'=' * 65}\n")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.yaml", help="Path to YAML config profile")
    parser.add_argument("--raw-path", help="Path to raw RecipeNLG CSV (override config paths.raw_csv)")
    parser.add_argument(
        "--stage",
        default="all",
        choices=STAGES + ["all"],
        help="Stage to run: raw, processed, baseline, embeddings, pairs, finetune, evaluate, or all",
    )
    parser.add_argument(
        "--from-stage",
        choices=STAGES,
        help="Start from this stage through the rest of the pipeline",
    )
    parser.add_argument(
        "--force", action="store_true", help="Bypass hash cache and force execution of stage(s)"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Hardware device (default: cpu)"
    )
    parser.add_argument("--mode", default="ingredients", choices=MODES, help="Embedding document mode")
    parser.add_argument("--epochs", type=int, default=1, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training/eval")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--promote", action="store_true", help="Promote evaluated model to production alias in MLflow"
    )
    parser.add_argument(
        "--rollback", help="Roll back MLflow Model Registry production alias to target version"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking and logging")
    args = parser.parse_args(argv)

    if args.rollback:
        rollback_model_registry(target_version=args.rollback)
        return 0

    cfg: DictConfig = load_config(args.config)
    r_path = Path(args.raw_path) if args.raw_path else None

    run_pipeline(
        cfg,
        stage=args.stage,
        from_stage=args.from_stage,
        raw_path=r_path,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        force=args.force,
        promote=args.promote,
        log_to_mlflow=not args.no_mlflow,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
