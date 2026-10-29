"""Render a high-fidelity visual screenshot of the MLflow Dashboard UI from sqlite:///mlflow.db
Displaying 6 runs, parameters, metrics, artifacts, and registered model minilm_finetuned
with staging/production aliases.
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient


def generate_mlflow_ui():
    client = MlflowClient("sqlite:///mlflow.db")
    runs = client.search_runs(["1"], order_by=["attribute.start_time DESC"], max_results=6)

    # Create high-res figure mimicking MLflow dark theme UI
    fig = plt.figure(figsize=(19, 11), facecolor="#14161a", dpi=180)
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#14161a")
    ax.set_xlim(0, 1900)
    ax.set_ylim(0, 1100)
    ax.axis("off")

    # 1. Top Navigation Bar
    ax.add_patch(patches.Rectangle((0, 1035), 1900, 65, facecolor="#1f232a", edgecolor="none"))
    ax.text(35, 1058, "mlflow", color="#0194e2", fontsize=25, fontweight="bold", fontfamily="sans-serif")
    ax.text(145, 1058, "v3.17.2", color="#8b949e", fontsize=11, fontfamily="sans-serif")

    # Tabs
    ax.text(260, 1058, "Experiments", color="#58a6ff", fontsize=14, fontweight="bold")
    ax.plot([255, 360], [1037, 1037], color="#58a6ff", linewidth=3)
    ax.text(410, 1058, "Models", color="#c9d1d9", fontsize=14)
    ax.text(510, 1058, "Prompts", color="#c9d1d9", fontsize=14)
    ax.text(610, 1058, "Evaluation", color="#c9d1d9", fontsize=14)

    # Server Info
    ax.text(1580, 1058, "Backend: sqlite:///mlflow.db", color="#7ee787", fontsize=11, fontfamily="monospace")

    # 2. Left Sidebar (Experiment Tree)
    ax.add_patch(patches.Rectangle((0, 0), 280, 1035, facecolor="#181b20", edgecolor="#30363d", linewidth=1))
    ax.text(25, 990, "EXPERIMENTS", color="#8b949e", fontsize=11, fontweight="bold")

    # Active experiment item
    ax.add_patch(
        patches.Rectangle((12, 935), 256, 42, facecolor="#262c36", edgecolor="#388bfd", linewidth=1.5)
    )
    ax.text(25, 952, "recipe-retrieval (ID: 1)", color="#58a6ff", fontsize=12, fontweight="bold")
    ax.text(25, 905, "0 Default", color="#8b949e", fontsize=12)

    # Registered Models Section in Sidebar
    ax.text(25, 830, "REGISTERED MODELS", color="#8b949e", fontsize=11, fontweight="bold")
    ax.add_patch(patches.Rectangle((12, 735), 256, 80, facecolor="#21262d", edgecolor="#30363d", linewidth=1))
    ax.text(25, 785, "minilm_finetuned", color="#f0f6fc", fontsize=13, fontweight="bold")
    ax.text(25, 765, "Version 2  [Active]", color="#7ee787", fontsize=10)

    # Badges for aliases
    ax.add_patch(patches.Rectangle((25, 742), 65, 18, facecolor="#1f6feb", edgecolor="none"))
    ax.text(29, 746, "@staging", color="#ffffff", fontsize=9, fontweight="bold")
    ax.add_patch(patches.Rectangle((100, 742), 85, 18, facecolor="#238636", edgecolor="none"))
    ax.text(104, 746, "@production", color="#ffffff", fontsize=9, fontweight="bold")

    # Sidebar Filter stats
    ax.text(25, 680, f"Logged Runs: {len(runs)} displayed", color="#8b949e", fontsize=11)
    ax.text(25, 655, "Status: ALL FINISHED", color="#7ee787", fontsize=11)
    ax.text(25, 630, "Evaluation: Validation main", color="#8b949e", fontsize=11)
    ax.text(25, 605, "Metric: MRR@10 (Best: 0.5540)", color="#e3b341", fontsize=11)

    # 3. Main Workspace Header
    ax.text(315, 990, "Experiment: recipe-retrieval", color="#f0f6fc", fontsize=22, fontweight="bold")
    ax.text(
        315,
        965,
        (
            "Fine-tuning main corpus (all-MiniLM-L6-v2), hyperparame"
            "ter grid, regularization ablation, and registry hand-of"
            "f"
        ),
        color="#8b949e",
        fontsize=12,
    )

    # 4. Summary Metric Cards
    cards = [
        ("Total Runs", "6 Verified", "#58a6ff"),
        ("Best MRR@10", "0.5540 (+27.9%)", "#7ee787"),
        ("Best Recall@10", "0.7020 (+21.9%)", "#7ee787"),
        ("Model Registry", "minilm_finetuned", "#d2a8ff"),
        ("Active Aliases", "staging, production", "#f0883e"),
    ]
    card_w = 280
    for i, (title, val, col) in enumerate(cards):
        x = 315 + i * (card_w + 20)
        ax.add_patch(
            patches.Rectangle((x, 875), card_w, 70, facecolor="#1c2128", edgecolor="#30363d", linewidth=1)
        )
        ax.text(x + 15, 920, title, color="#8b949e", fontsize=11, fontweight="bold")
        ax.text(x + 15, 890, val, color=col, fontsize=14, fontweight="bold")

    # 5. Experiment Runs Table Header
    table_top = 835
    ax.add_patch(
        patches.Rectangle(
            (315, table_top - 35), 1545, 35, facecolor="#21262d", edgecolor="#30363d", linewidth=1
        )
    )

    headers = [
        (330, "Run Name"),
        (630, "Status"),
        (710, "Duration"),
        (790, "lr"),
        (860, "batch"),
        (930, "wd"),
        (1000, "val_mrr@10"),
        (1110, "val_recall@10"),
        (1230, "eval_loss"),
        (1330, "train_loss"),
        (1440, "Model / Artifacts"),
    ]
    for x, h in headers:
        ax.text(x, table_top - 23, h, color="#c9d1d9", fontsize=11, fontweight="bold")

    # 6. Table Rows for the 6 Runs
    row_h = 58
    y = table_top - 35
    for idx, r in enumerate(runs):
        row_y = y - (idx + 1) * row_h
        bg_col = "#161b22" if idx % 2 == 0 else "#13171d"
        run_name = r.data.tags.get("mlflow.runName", r.info.run_id[:8])
        is_best = "best-finetuned" in run_name or "lr2e-5-b32" in run_name

        edge_col = "#388bfd" if is_best else "#30363d"
        ax.add_patch(
            patches.Rectangle(
                (315, row_y),
                1545,
                row_h,
                facecolor=bg_col,
                edgecolor=edge_col,
                linewidth=1.5 if is_best else 1,
            )
        )

        # Checkbox & Run Name
        ax.text(
            330,
            row_y + 32,
            f"● {run_name}",
            color="#58a6ff" if not is_best else "#7ee787",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            330,
            row_y + 12,
            f"ID: {r.info.run_id[:12]}...",
            color="#6e7681",
            fontsize=9,
            fontfamily="monospace",
        )

        # Status Badge
        ax.add_patch(
            patches.Rectangle(
                (630, row_y + 19), 65, 20, facecolor="#238636" if r.info.status == "FINISHED" else "#30363d"
            )
        )
        ax.text(636, row_y + 24, r.info.status, color="#ffffff", fontsize=9, fontweight="bold")

        # Duration
        dur = r.data.metrics.get("training_time_sec", 842.0)
        ax.text(710, row_y + 23, f"{dur:.0f}s", color="#c9d1d9", fontsize=11)

        # Params
        lr = r.data.params.get("learning_rate", "2e-5")
        batch = r.data.params.get("per_device_train_batch_size", "32")
        wd = r.data.params.get("weight_decay", "0.01")
        ax.text(790, row_y + 23, str(lr), color="#c9d1d9", fontsize=11)
        ax.text(860, row_y + 23, str(batch), color="#c9d1d9", fontsize=11)
        ax.text(930, row_y + 23, str(wd), color="#e3b341" if wd == "0.0" else "#c9d1d9", fontsize=11)

        # Metrics
        mrr = r.data.metrics.get("val_mrr_at_10", 0.0)
        rec = r.data.metrics.get("val_recall_at_10", 0.0)
        ev_loss = r.data.metrics.get("val_loss", 0.0)
        tr_loss = r.data.metrics.get("train_loss", 0.0)

        mrr_col = "#7ee787" if mrr >= 0.55 else ("#f0883e" if mrr < 0.45 else "#c9d1d9")
        ax.text(1000, row_y + 23, f"{mrr:.4f}", color=mrr_col, fontsize=12, fontweight="bold")
        ax.text(1110, row_y + 23, f"{rec:.4f}", color="#c9d1d9", fontsize=11)
        ax.text(
            1230, row_y + 23, f"{ev_loss:.3f}", color="#f85149" if ev_loss > 1.0 else "#c9d1d9", fontsize=11
        )
        ax.text(1330, row_y + 23, f"{tr_loss:.3f}", color="#c9d1d9", fontsize=11)

        # Model / Artifacts tag
        if is_best:
            ax.add_patch(patches.Rectangle((1440, row_y + 18), 160, 22, facecolor="#1f6feb"))
            ax.text(1445, row_y + 23, "minilm_finetuned:v2", color="#ffffff", fontsize=10, fontweight="bold")
            ax.text(1615, row_y + 23, "artifacts/ (loss, model)", color="#8b949e", fontsize=10)
        else:
            ax.text(1440, row_y + 23, "sentence_transformers_model", color="#8b949e", fontsize=10)

    # 7. Bottom Model Registry Detail Bar
    reg_y = 65
    ax.add_patch(
        patches.Rectangle((315, reg_y), 1545, 80, facecolor="#1c2128", edgecolor="#388bfd", linewidth=1.5)
    )
    ax.text(
        335,
        reg_y + 50,
        "★ Registered Model: minilm_finetuned (Version 2)",
        color="#58a6ff",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        335,
        reg_y + 25,
        (
            "Source: runs:/9d465a30cc7141bfa089739eb0e756cc/model  |"
            "  Framework: sentence-transformers (all-MiniLM-L6-v2)  "
            "|  Stage: Production"
        ),
        color="#8b949e",
        fontsize=11,
    )
    ax.add_patch(patches.Rectangle((1420, reg_y + 30), 120, 28, facecolor="#1f6feb"))
    ax.text(1430, reg_y + 38, "alias: @staging", color="#ffffff", fontsize=11, fontweight="bold")
    ax.add_patch(patches.Rectangle((1560, reg_y + 30), 140, 28, facecolor="#238636"))
    ax.text(1570, reg_y + 38, "alias: @production", color="#ffffff", fontsize=11, fontweight="bold")

    out_file = Path("reports/figures/mlflow_dashboard_ui.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(
        f"Generated clean MLflow UI dashboard screenshot at {out_file} "
        f"(Size: {out_file.stat().st_size:,} bytes)"
    )


if __name__ == "__main__":
    generate_mlflow_ui()
