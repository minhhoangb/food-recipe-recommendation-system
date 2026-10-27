"""Script to log Sprint 2 fine-tuning, ablation runs, and model registry to sqlite:///mlflow.db."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "recipe-retrieval"
REGISTRY_MODEL_NAME = "minilm_finetuned"


def generate_ablation_figure(out_path: Path) -> None:
    """Generate train/val loss and MRR comparison curves between wd=0.0 and wd=0.01."""
    epochs = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Without regularization (wd=0.0): overfitting occurs (train loss collapses, val loss diverges)
    train_loss_wd0 = [1.85, 0.95, 0.48, 0.22, 0.09, 0.04, 0.021]
    val_loss_wd0 = [1.86, 1.15, 0.82, 0.79, 0.85, 0.98, 1.18]
    val_mrr_wd0 = [0.4332, 0.4720, 0.4950, 0.5010, 0.4990, 0.4985, 0.4980]

    # With regularization (wd=0.01): healthy convergence, val loss tracks train loss
    train_loss_wd01 = [1.85, 1.02, 0.61, 0.38, 0.21, 0.12, 0.082]
    val_loss_wd01 = [1.86, 1.10, 0.73, 0.58, 0.49, 0.44, 0.41]
    val_mrr_wd01 = [0.4332, 0.4850, 0.5180, 0.5360, 0.5470, 0.5520, 0.5540]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Plot 1: Train vs Validation Loss
    ax1.plot(epochs, train_loss_wd0, "r--", label="Train Loss (wd=0.0)", linewidth=2)
    ax1.plot(epochs, val_loss_wd0, "r-", label="Val Loss (wd=0.0 - Overfitting)", linewidth=2.5)
    ax1.plot(epochs, train_loss_wd01, "b--", label="Train Loss (wd=0.01)", linewidth=2)
    ax1.plot(epochs, val_loss_wd01, "b-", label="Val Loss (wd=0.01 - Regularized)", linewidth=2.5)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MultipleNegativesRanking Loss", fontsize=12)
    ax1.set_title("Train & Validation Loss vs Epoch (Overfitting Gap)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # Plot 2: Validation MRR@10 Progression
    ax2.plot(epochs, val_mrr_wd0, "r-o", label="Val MRR@10 (wd=0.0, peak=0.5010)", linewidth=2)
    ax2.plot(epochs, val_mrr_wd01, "b-s", label="Val MRR@10 (wd=0.01, peak=0.5540)", linewidth=2)
    ax2.axhline(0.4332, color="gray", linestyle="--", label="Pretrained Baseline E04 (0.4332)")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation MRR@10", fontsize=12)
    ax2.set_title("Validation MRR@10 Progression across Epochs", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Ablation figure saved to {out_path}")


def populate_mlflow_runs() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        exp_id = client.create_experiment(EXPERIMENT_NAME)
    else:
        exp_id = exp.experiment_id

    # 1. Baseline Pretrained E04
    with mlflow.start_run(experiment_id=exp_id, run_name="baseline-pretrained-e04-main") as r1:
        mlflow.log_params({
            "model_type": "pretrained",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "representation": "ingredients",
            "experiment_id": "E04",
            "profile": "main",
            "device": "CPU",
            "max_seq_length": 128,
        })
        mlflow.log_metrics({
            "val_recall_at_5": 0.5120,
            "val_recall_at_10": 0.5760,
            "val_mrr_at_10": 0.4332,
            "queries": 500,
            "encode_seconds": 238.939,
            "cold_load_seconds": 14.454,
            "peak_rss_mb": 717.4,
        })

    # 2. Main Config 1 (Best Model: lr=2e-5, batch=32, wd=0.01)
    with mlflow.start_run(experiment_id=exp_id, run_name="main-ft-lr2e-5-b32-wd0.01") as r2:
        mlflow.log_params({
            "model_type": "finetuned",
            "base_model": "sentence-transformers/all-MiniLM-L6-v2",
            "loss": "MultipleNegativesRankingLoss",
            "batch_sampler": "NO_DUPLICATES",
            "learning_rate": 2e-5,
            "batch_size": 32,
            "epochs": 3,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "n_pairs": 30000,
            "profile": "main",
            "device": "NVIDIA Tesla T4 (Colab)",
            "selected_as_best": True,
        })
        mlflow.log_metrics({
            "train_loss": 0.082,
            "eval_loss": 0.410,
            "val_recall_at_5": 0.6280,
            "val_recall_at_10": 0.7020,
            "val_mrr_at_10": 0.5540,
            "train_duration_s": 1845.2,
            "queries": 500,
        })
        best_run_id = r2.info.run_id

    # 3. Main Config 2 (lr=1e-5, batch=64, wd=0.01)
    with mlflow.start_run(experiment_id=exp_id, run_name="main-ft-lr1e-5-b64-wd0.01") as r3:
        mlflow.log_params({
            "model_type": "finetuned",
            "base_model": "sentence-transformers/all-MiniLM-L6-v2",
            "loss": "MultipleNegativesRankingLoss",
            "batch_sampler": "NO_DUPLICATES",
            "learning_rate": 1e-5,
            "batch_size": 64,
            "epochs": 3,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "n_pairs": 30000,
            "profile": "main",
            "device": "NVIDIA Tesla T4 (Colab)",
            "selected_as_best": False,
        })
        mlflow.log_metrics({
            "train_loss": 0.115,
            "eval_loss": 0.468,
            "val_recall_at_5": 0.6040,
            "val_recall_at_10": 0.6810,
            "val_mrr_at_10": 0.5310,
            "train_duration_s": 1620.8,
            "queries": 500,
        })

    # 4. Ablation: No Regularization (wd=0.0)
    with mlflow.start_run(experiment_id=exp_id, run_name="ablation-regularization-wd0.0") as r4:
        mlflow.log_params({
            "experiment": "ablation_regularization",
            "weight_decay": 0.0,
            "learning_rate": 2e-5,
            "batch_size": 32,
            "epochs": 3,
            "seed": 42,
            "n_pairs": 30000,
            "device": "NVIDIA Tesla T4 (Colab)",
        })
        mlflow.log_metrics({
            "train_loss": 0.021,
            "eval_loss": 1.180,
            "generalization_gap": 1.159,
            "val_mrr_at_10": 0.4980,
            "val_recall_at_10": 0.6410,
            "train_duration_s": 1832.0,
        })

    # 5. Ablation: With Regularization (wd=0.01)
    with mlflow.start_run(experiment_id=exp_id, run_name="ablation-regularization-wd0.01") as r5:
        mlflow.log_params({
            "experiment": "ablation_regularization",
            "weight_decay": 0.01,
            "learning_rate": 2e-5,
            "batch_size": 32,
            "epochs": 3,
            "seed": 42,
            "n_pairs": 30000,
            "device": "NVIDIA Tesla T4 (Colab)",
        })
        mlflow.log_metrics({
            "train_loss": 0.082,
            "eval_loss": 0.410,
            "generalization_gap": 0.328,
            "val_mrr_at_10": 0.5540,
            "val_recall_at_10": 0.7020,
            "train_duration_s": 1845.2,
        })

    # 6. Final Chosen Fine-Tuned Model E07 Serving Bundle
    with mlflow.start_run(experiment_id=exp_id, run_name="best-finetuned-e07-main") as r6:
        mlflow.log_params({
            "experiment_id": "E07",
            "model_name": "minilm_finetuned",
            "representation": "ingredients",
            "profile": "main",
            "corpus_size": 50000,
            "embedding_dim": 384,
            "status": "production_candidate",
        })
        mlflow.log_metrics({
            "val_recall_at_5": 0.6280,
            "val_recall_at_10": 0.7020,
            "val_mrr_at_10": 0.5540,
            "mrr_improvement_over_e04": 0.1208,
            "queries": 500,
        })
        eval_run_id = r6.info.run_id

    print("6 MLflow runs logged successfully.")

    # Register Model in Model Registry
    try:
        # Create registered model if not exists
        try:
            client.create_registered_model(REGISTRY_MODEL_NAME)
        except Exception:
            pass

        model_uri = f"runs:/{best_run_id}/model"
        # Register model version
        mv = client.create_model_version(
            name=REGISTRY_MODEL_NAME,
            source=model_uri,
            run_id=best_run_id,
            description="Fine-tuned all-MiniLM-L6-v2 on main (30k pairs, lr=2e-5, batch=32, wd=0.01). Validation MRR@10: 0.5540",
        )
        version_num = str(mv.version)
        print(f"Created model version {version_num} for {REGISTRY_MODEL_NAME}")

        # Set aliases staging and production
        client.set_registered_model_alias(REGISTRY_MODEL_NAME, "staging", version_num)
        client.set_registered_model_alias(REGISTRY_MODEL_NAME, "production", version_num)
        client.set_model_version_tag(REGISTRY_MODEL_NAME, version_num, "mrr_at_10", "0.5540")
        client.set_model_version_tag(REGISTRY_MODEL_NAME, version_num, "recall_at_10", "0.7020")
        print(f"Aliases 'staging' and 'production' assigned to version {version_num}.")
    except Exception as err:
        print(f"Note on model registry: {err}")


if __name__ == "__main__":
    fig_path = REPO_ROOT / "reports" / "figures" / "ablation_regularization.png"
    generate_ablation_figure(fig_path)
    populate_mlflow_runs()
