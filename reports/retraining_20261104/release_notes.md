# GPU retraining artifacts — 2026-11-04

Fresh RTX 3060 Laptop GPU run using PyTorch 2.9.1+cu128. Seven systems were rebuilt
and compared on the fixed 500-query validation set and 50,000-recipe corpus.
The original frozen E10 test evaluation and production selection are unchanged.

- Main MiniLM fine-tuning: 30,000 train pairs, one epoch, batch 32, seed 42;
  126.791 seconds, peak allocated CUDA memory 1244.13 MiB.
- Validation MRR@10: BM25 0.7583134921; fine-tuned MiniLM 0.6228761905.
  These are synthetic origin-recipe metrics; independent final human labels are pending.
- Controlled weight-decay ablation, pair manifests, per-system bundles, checkpoints,
  training logs, environment freeze and a consistent MLflow SQLite snapshot are included.
- Source/configuration hashes and pre-commit working-tree state are preserved in
  `run_status.json`. The runner contains original workstation paths.

Archive: `gpu-retraining-20261104-9496b9b06fde.zip` (679,961,804 bytes).

SHA-256:

```text
11eb9c4758e5f813f58b8ad7d4c886133b9f5c65c38af949e9b85f1d96a42cfd
```

Download the ZIP and matching `.sha256` asset. In PowerShell, verify with
`Get-FileHash .\gpu-retraining-20261104-9496b9b06fde.zip -Algorithm SHA256`
before extracting. The archive creates `gpu-retraining-20261104/`; extract into a
separate directory to preserve the historical production models. Training again
is not required to inspect or use the saved bundles.

The repository's `reports/retraining_20261104/artifacts.json` lists every archived
file and its hash. The MLflow database retains original artifact URIs; inspecting
metrics works, but artifact paths may need remapping on another workstation.
Raw RecipeNLG CSV, virtual environments, credentials and the newly generated
PPTX, Word and ACM LaTeX report deliverables are excluded.
