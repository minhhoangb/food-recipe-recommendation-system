# Controlled weight-decay ablation: measured on Colab

Executed on 2026-11-03 in [the Colab notebook](https://colab.research.google.com/drive/178UsXUi0OhmEOM7PgDbyIY4UGHYX9L4M),
using a Tesla T4, Python 3.13.16, torch 2.11.0+cu130,
sentence-transformers 5.1.2 and MLflow 3.17.2.

Both arms use all-MiniLM-L6-v2 with identical initial weights (SHA-256
`c13c0900ac73c2de46340ec049fd09d33205a6a07265308320e79e283ece248d`),
seed 42, the same 1,024 training pairs and 256 validation pairs, batch size 16,
AdamW, learning rate 2e-5, maximum sequence length 128, and three epochs.
Only `weight_decay` changes. The script verifies train/validation group isolation
and hashes the data, pairs, initial weights and batch schedule. Duplicate groups
are excluded within batches; dropping incomplete batches gives 190 optimizer steps.

Losses are measured at step 0, every 32 steps and the last step. Fixed probes use
eval mode, no gradients, and 15 complete batches (240 pairs) per split so MNRL
losses use equal numbers of in-batch negatives. `optimization_loss` is logged
separately during training. No test query is evaluated, and no serving bundle or
registry alias is promoted by this experiment.

| weight_decay | Final train probe loss | Final validation probe loss | Measured train/probe/save seconds |
| --- | --- | --- | --- |
| 0.0 | 0.0136506967 | 0.0263719251 | 23.4188 |
| 0.01 | 0.0136520094 | 0.0263708937 | 22.2684 |

The absolute validation difference is about 0.00000103. This small, single-seed
experiment does **not** demonstrate a meaningful regularization benefit,
statistical significance or retrieval-quality improvement. It supersedes the old
hard-coded curves and their unsupported overfitting claims. The existing 30k-pair
CPU checkpoint and E08 retrieval results are separate experiments.

Original Colab parent run: `727a5f2969154b0591842fefc161ca18`.
Original child runs: `d6288a44961f46cc84e3053324855d4f` (0.0) and
`dd5a0f6c678546eaa4d76d18c4676ec0` (0.01).
Local imported runs: `6b4b0facbbfb4082be15b94f3977c805` and
`6178d21f4c064d019dc8fab00efd384a`, experiment `recipe-regularization-ablation`.
Import preserves every metric value, step and timestamp; source run IDs and the
notebook URL remain explicit tags.

- Raw exported measurements: [regularization_ablation/colab_export.json](regularization_ablation/colab_export.json).
- Summary and original/local ID mapping: [regularization_ablation/summary.json](regularization_ablation/summary.json).
- Actual UI screenshot: [figures/mlflow_ablation_curves.png](figures/mlflow_ablation_curves.png).
- Screenshot provenance: [figures/mlflow_screenshots.json](figures/mlflow_screenshots.json).
- Measured-only plot: [figures/regularization_loss_measured.png](figures/regularization_loss_measured.png).

Re-run with `python -m scripts.run_regularization_ablation --device cuda`, or use
`notebooks/04_regularization_controlled_colab.ipynb`. `src.finetune` now forwards
weight decay through the dedicated `fit(weight_decay=...)` argument: setting it
only inside `optimizer_params` was overridden by fit's parameter groups.
