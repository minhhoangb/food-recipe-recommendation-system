# Khung 3 verification for Hưng and Hiếu

Verified locally on 2026-11-03 against dev, starting at dcc076a. The requested fixes
leave Hoàng's label-status documentation and all three assignment CSVs unchanged.

## Hưng

- API/benchmark/evaluator now load the encoder packaged inside a fine-tuned bundle.
  Missing checkpoints and modified checkpoint files are rejected. A relocated bundle
  is covered by a regression test. The existing locked FT checkpoint was packaged
  without changing corpus vectors or retraining weights.
- `benchmark.json` and `benchmark.md` were regenerated on this machine for TF-IDF,
  BM25, LSA, MiniLM E04/E05 and FT: 20 warmup and 100 measured validation requests
  per bundle, measured in separate processes. FT p50/p95: 24.66/30.74 ms (see JSON
  for the exact recorded values and hardware).
- `figures/ui_390.png` and `figures/ui_1440.png` were recaptured from the real FT API
  with explicit Playwright CSS viewports. Both have scroll width equal to viewport
  width; the Results field is visible, and Tab from ingredients reaches that field.
  `ui_viewports.json` records the checks. Capture script on this workstation:
  `D:/MDS301/outputs/khung3-verification/capture_ui.cjs`.

## Hiếu

- Processed caches verify manifest/config/raw hashes. Other stages verify input
  and artifact hashes; training parameters and device invalidate the checkpoint
  cache. A processed-data mismatch stops with instructions instead of rebuilding
  shared artifacts silently. Existing legacy artifacts with different configuration
  hashes require verification before use; this pass did not regenerate them.
- All seven stages run through a fixture integration test, including cache hits,
  changed learning rate and altered weights. The CPU device reaches training.
  Fixture encoders stand in for large models in this test; this is not a new pilot
  fine-tuning experiment.
- Evaluate uses observed `train_summary.json` and measured encoding time.
  Unknown historical training measurements are omitted from MLflow instead of
  replaced with constants.
- API logs record missing-ingredient counts for each result. SQLite migration and
  old-record handling are tested. Missing historical values remain unknown.
  `monitoring_khung3_ft.json`/`.html` summarize actual HTTP request logs, including
  63 result counts. Simulation reports were regenerated. References exclude test
  and final queries, and a query-only reference no longer invents latency values.
- A real registry alias rollback 2 → 1 → 2 was executed and verified. The initially
  empty local registry was bootstrapped with two revisions of the same locked FT
  checkpoint. This verifies alias operations, not a quality change between models.
  The original production/staging aliases were preserved after the trial.
- A fresh uvicorn subprocess served `/health`, `/predict` and `/model-info` with
  tracking set to unavailable `http://127.0.0.1:1` and Hugging Face offline mode.
  Twenty prediction requests succeeded. Evidence: `offline_api_verification.json`
  and `mlflow_rollback_trial.json`. The subprocess was stopped afterwards.

## Validation

- `python -m pytest --cov=src.baseline --cov=src.features --cov=src.evaluate
  --cov-report=term-missing --cov-fail-under=70`: 276 passed, coverage 97.09%.
- `python -m ruff check .`: passed.
- Four existing library/resource warnings remain. A complete new pilot training
  run, a Docker rebuild, remote CI and PR reviews were not performed in this pass.
