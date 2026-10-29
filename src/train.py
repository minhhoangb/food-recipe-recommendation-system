"""End-to-end training pipeline: pair construction, fine-tuning, bundle packaging and MLflow logging.

    python -m src.train --config configs/pilot.yaml --epochs 1 --batch-size 16 --mode ingredients

Orchestrates:
  1. Build pairs / triplets from the train split.
  2. Fine-tune sentence-transformers/all-MiniLM-L6-v2 with MultipleNegativesRankingLoss.
  3. Encode the corpus with the fine-tuned model and save a valid serving bundle (minilm_finetuned).
  4. Evaluate retrieval on the validation split with ranx (Recall@5, Recall@10, MRR@10).
  5. Log the run to MLflow (params, metrics, artifacts) and register model in MLflow Model Registry.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

from src.build_pairs import build_pairs
from src.config import load_config, processed_dir
from src.contracts import BundleManifest, RecipeRecord
from src.embeddings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODELS_DIR,
    MODEL_NAME,
    MODES,
    EmbeddingMode,
    encode_corpus,
    save_bundle,
    validate_bundle,
)
from src.evaluation import (
    DEFAULT_EXPERIMENT,
    DEFAULT_TRACKING_URI,
    evaluate_retriever,
    load_split,
)
from src.finetune import finetune_model
from src.memory import peak_rss_mb
from src.retrieval import DenseRetriever
from src.utils import read_jsonl, sha256_file


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

    manifest = BundleManifest.model_validate_json((bundle_dir / "manifest.json").read_text(encoding="utf-8"))

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

    # Model Registry
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        model_uri = f"runs:/{run_id}/serving_bundle"
        registered_model = mlflow.register_model(model_uri, model_registry_name)
        version_num = registered_model.version

        # Set alias (staging or production)
        alias = "production" if promote_production else "staging"
        client.set_registered_model_alias(model_registry_name, alias, version_num)
        client.set_model_version_tag(
            model_registry_name,
            version_num,
            "mrr@10",
            str(result["metrics"].get("mrr@10", 0.0)),
        )
    except Exception as e:
        print(f"warning: MLflow model registration failed: {e}", file=sys.stderr)

    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--mode", default="ingredients", choices=MODES)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--negatives", action="store_true")
    parser.add_argument("--hard-negatives", action="store_true")
    parser.add_argument("--promote", action="store_true", help="Set alias to production in MLflow registry")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args(argv)

    cfg: DictConfig = load_config(args.config)
    directory = processed_dir(cfg)
    if not (directory / "corpus.jsonl").exists():
        print(f"error: {directory}/corpus.jsonl missing; build or fetch data first", file=sys.stderr)
        return 1

    print(
        f"Running fine-tuning pipeline on {cfg.data.profile} profile "
        f"(mode={args.mode}, epochs={args.epochs})..."
    )
    result = run_training_pipeline(
        cfg,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        include_negatives=args.negatives or args.hard_negatives,
        hard_negatives=args.hard_negatives,
        promote_production=args.promote,
        log_to_mlflow=not args.no_mlflow,
    )
    print(f"\nBundle created: {result['bundle']}")
    print("Validation metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")
    if "mlflow_run_id" in result:
        print(f"Logged to MLflow run: {result['mlflow_run_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
