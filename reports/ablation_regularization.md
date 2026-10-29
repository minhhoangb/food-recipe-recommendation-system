# Weight-decay ablation: not yet measured

The previous fixed result tables and plotted loss curves were not backed by
measured runs. They have been removed and must not be cited as experimental
evidence. No controlled weight-decay comparison or causal regularization effect
has been established.

The measured CPU one-epoch checkpoint and its validation results are documented
in [e07_e08_finetune_main.md](e07_e08_finetune_main.md), with per-system run IDs
in [comparison.csv](comparison.csv). A single training configuration cannot
establish the effect of weight decay.

To complete the ablation, run two actual training experiments with the same
corpus and pair hashes, split, seed, model, batch size, learning rate, schedule
and hardware, changing only weight decay. Record measured losses, validation
retrieval metrics, duration, checkpoint hashes and MLflow run IDs. Plot curves
from those run histories and select the checkpoint using validation only.

`notebooks/03_finetune_main_ablation.ipynb` contains the training workflow;
unmeasured sections remain explicitly pending. Checkpoint export requires
completed training results and selects the best measured validation MRR@10.
Registry aliases require a separate reviewed promotion decision.

This cleanup removes the scripts that inserted fabricated runs or rendered a
dashboard illustration. It does not delete existing MLflow runs or registry
aliases. Existing local tracking history must be audited against source logs
before any historical result is trusted.
