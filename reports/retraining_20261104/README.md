# GPU retraining on 4 November 2026

Seven systems were rebuilt and evaluated on the fixed 500-query validation set,
with 50,000 candidate recipes, top-k 10 and no reranking. MiniLM was fine-tuned
from the cached pretrained snapshot on the RTX 3060 Laptop GPU, not resumed from
the historical CPU checkpoint. The run completed at 10:18 UTC (17:18 Bangkok).

| System | Validation MRR@10 |
|---|---:|
| E01 TF-IDF | 0.6567388889 |
| E02 selected TF-IDF configuration | 0.6567388889 |
| E03 BM25 | 0.7583134921 |
| LSA | 0.3628523810 |
| E04 MiniLM ingredients | 0.4331531746 |
| E05 MiniLM title and ingredients | 0.1518833333 |
| FT MiniLM fine-tuned | 0.6228761905 |

The main FT run used 30,000 train pairs, one epoch, batch 32, learning rate 2e-5,
50 warmup steps and weight decay 0.01. Fine-tuning took 126.791 seconds. Peak
CUDA tensor allocation was 1244.13 MiB, with a peak reserved allocation of
1500 MiB. Training used the snapshot's 256-token limit; encoding and retrieval
used the project's 128-token limit. These settings are recorded separately.

The controlled 1,024/256-pair weight-decay experiment was also rerun on CUDA.
Final validation probe losses were 0.0261550910 and 0.0261529131. This single-seed
result does not establish a meaningful regularization or relevance improvement.

These are synthetic origin-recipe metrics. No new test evaluation was performed.
The original E10 execution ledger, CPU checkpoint and production configuration
were preserved. BM25 remains the selected system on validation. Final independent
human labels are still pending.

## Evidence

- `comparison_validation.csv` includes each MLflow run ID.
- `run_status.json` records the actual dirty working tree, code/config hashes,
  hardware and completion time. Its `git_head` is the pre-commit run provenance.
- `training_summary.json`, `training.log` and per-system validation rankings retain
  the measured training and retrieval outputs.
  The repository log trims trailing console whitespace; the release preserves the original log.
- `regularization_ablation/` contains the new measured loss histories and controls.
- `requirements-gpu-freeze.txt` records the CUDA environment.
- `pytest.log` records the 302 passing tests run immediately after retraining.
- `run_retraining.py.txt` preserves the original task-local runner without edits.
  It contains workstation-specific paths and is historical execution evidence,
  not a portable training entrypoint.

## Large artifacts

Weights, embeddings, indexes, training pairs and the new MLflow database are
shared separately in the
[GPU artifact release](https://github.com/minhhoangb/food-recipe-recommendation-system/releases/tag/model-artifacts-gpu-20261104).
Use `artifacts.json` for the archive URL, SHA-256 and per-file inventory. Extract
the ZIP into a separate directory; it restores `gpu-retraining-20261104/` and
does not replace the historical `models/` tree or select a serving bundle.

Verify the downloaded archive against the `.sha256` asset before extraction.
The MLflow database retains original artifact URIs; it is a tracking snapshot,
not a shared portable tracking server. Metrics and parameters can be inspected,
while artifact locations may need remapping on another workstation.

The newly generated presentation, Word report and ACM LaTeX deliverables are
intentionally kept outside this repository at the user's request.
