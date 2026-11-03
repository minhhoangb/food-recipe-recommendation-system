"""Plot measured MLflow losses; capture UI screenshots from `mlflow ui` separately.

python scripts/generate_mlflow_figures.py --run-ids <wd0-run> <wd001-run>
Never renders imitation dashboards or fills in missing metric histories.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--run-ids", nargs=2, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/figures/regularization_loss_measured.png"))
    args = parser.parse_args(argv)
    client = MlflowClient(args.tracking_uri)
    fig, ax = plt.subplots(figsize=(10, 6))
    params = []
    for rid in args.run_ids:
        run = client.get_run(rid)
        if run.info.status != "FINISHED" or run.data.tags.get("controlled_ablation") != "true":
            raise ValueError(f"{rid}: require a finished measured controlled-ablation run")
        params.append({k: v for k, v in run.data.params.items() if k != "weight_decay"})
        for key, style in (("train_loss", "-"), ("val_loss", "--")):
            history = sorted(client.get_metric_history(rid, key), key=lambda m: m.step)
            if len(history) < 2:
                raise ValueError(f"{rid}: missing measured {key} history")
            ax.plot([m.step for m in history], [m.value for m in history], style,
                    label=f"wd={run.data.params['weight_decay']} {key} ({rid[:8]})")
    if params[0] != params[1]:
        raise ValueError("Run parameters differ beyond weight_decay")
    ax.set(xlabel="Optimizer step", ylabel="Measured MNRL loss (fixed eval-mode probes)",
           title="Controlled weight-decay ablation — measured MLflow histories")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
