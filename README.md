# Food Recipe Recommendation System

Type the ingredients you have, get recipes that use them, with what you already have and what is still missing. Ingredient-list retrieval on the RecipeNLG corpus (English only), comparing TF-IDF / BM25 / LSA baselines with a pretrained and a fine-tuned `all-MiniLM-L6-v2` encoder, served by a FastAPI app with a small web UI.

**Production model:** BM25 (E03), selected on validation before the final test run (E10). Test MRR@10 0.7766, Recall@10 0.909 on 1,000 test queries.

## Team

| Member | Student ID | Role |
| --- | --- | --- |
| Bùi Minh Hoàng | SE193392 | Baselines, evaluation, analysis |
| Trần Tuấn Hưng | SE183526 | Embeddings, API, UI, Docker |
| Hồ Minh Hiếu | SE193051 | Fine-tuning, MLflow, monitoring |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest

python scripts/fetch_artifacts.py      # processed corpus (GitHub Release, sha256-checked)
python scripts/verify_data.py
python -m src.baseline --config configs/default.yaml --model bm25   # builds models/bm25-main-1f1f4ce5/
uvicorn api.app:app --host 127.0.0.1 --port 8000                    # UI on http://127.0.0.1:8000/
```

- Python 3.11 or 3.12. Everything runs on CPU; CUDA is never required.
- `requirements.txt` is the full stack (serving + training: `torch` 2.9.1, `sentence-transformers` 5.1.2, `fastapi`, `mlflow`, `rank-bm25`, `accelerate`, `datasets`). `requirements-serve.txt` is the serving subset used by the Docker image.
- On Linux, install the CPU torch wheel first to avoid pulling CUDA: `pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu`. The default wheel on Windows/macOS is already CPU-only.
- The first MiniLM use downloads ~90 MB of weights into the Hugging Face cache.
- The full `models/` directory is available as an optional [GitHub Release download](docs/shared_model_artifacts.md).
- **Windows:** if pip reports *"An Application Control policy has blocked this file"*, run `Get-ChildItem .venv -Recurse -Include *.pyd,*.dll | Unblock-File`.

## Dataset

- **RecipeNLG** (Bien et al., INLG 2020), ~2.23M recipes: https://recipenlg.cs.put.poznan.pl/. The raw CSV (2.2 GB) goes to `data/raw/RecipeNLG_dataset.csv` (or `--raw-path` / `RECIPENLG_RAW`) and is never modified or redistributed.
- Seeded random subsets: **main** (50,000 recipes) and **pilot** (its first 5,000, same splits).
- Rebuild from the raw CSV instead of downloading (about 1 minute, identical hashes): `python -m src.data_collection --raw-path data/raw/RecipeNLG_dataset.csv`.

Pipeline (`src/data_collection.py`, `src/normalize.py`):

1. Safe parsing of stringified lists (JSON, then `ast.literal_eval`); dropped rows are counted per reason in `dataset_manifest.json`.
2. Ingredient normalisation with `ingredient-parser-nlp` + `inflect` (quantities, units and preparation words removed; multi-word names kept).
3. Exact duplicates removed; recipes with the same ingredient set or title are grouped, and groups never cross splits.
4. `GroupShuffleSplit` (seed 42): ~72% train, 8% validation, 20% test.
5. Synthetic queries: 2–6 ingredients from a recipe, at least one hidden. The only positive is the origin recipe, so Recall/MRR are an *origin-recipe retrieval proxy*, not human relevance.

## Models and experiments

All retrievers implement `Retriever.search(ingredients, top_k) -> list[SearchHit]` (`src/contracts.py`). Metrics are computed with `ranx` and logged to MLflow (experiment `recipe-retrieval`).

| ID | System | How it is built |
| --- | --- | --- |
| E01 | TF-IDF | `python -m src.baseline --model tfidf` (sklearn defaults, ingredient tokens) |
| E02 | TF-IDF tuned | `python -m src.tune` — GridSearchCV + GroupKFold(5), 24 grid points; the best point equals the E01 defaults |
| E03 | **BM25** (selected) | `python -m src.baseline --model bm25` (`rank-bm25`) |
| LSA | TF-IDF + TruncatedSVD | `python -m src.baseline --model lsa` (up to 256 components) |
| E04 | MiniLM pretrained, ingredients | `python -m src.embeddings --mode ingredients` |
| E05 | MiniLM pretrained, title + ingredients | `python -m src.embeddings --mode title_ingredients` |
| E06 | Missing-first rerank (ablation) | off by default; costs 34% relative Recall@10 on main ([report](reports/e04_e05_main_benchmark.md)) |
| E07 | Fine-tuning on pilot | `notebooks/02_finetune_colab.ipynb` (MultipleNegativesRankingLoss, Colab GPU) |
| E08 / FT | Fine-tuning on main | `python -m src.finetune` — 30,000 pairs, 1 epoch, batch 32, lr 2e-5 on CPU ([report](reports/e07_e08_finetune_main.md)) |
| E09 | Regularisation ablation | weight decay 0.0 vs 0.01 ([report](reports/ablation_regularization.md)) |
| E10 | Final test evaluation | one frozen run over all seven systems, 1,000 test queries ([evidence](reports/final)) |

### Results (main corpus, MRR@10)

| Model | Train | Validation | Test | CV (GroupKFold 5) |
|---|---:|---:|---:|---|
| E01 TF-IDF | 0.6998 | 0.6567 | 0.6649 | 0.6896 ± 0.0063 |
| E02 TF-IDF tuned | 0.6998 | 0.6567 | 0.6649 | 0.6896 ± 0.0063 |
| **E03 BM25** | 0.7687 | **0.7583** | **0.7766** | N/A (no fitting) |
| LSA | 0.3402 | 0.3629 | 0.3424 | 0.3560 ± 0.0095 |
| E04 MiniLM pretrained | 0.4575 | 0.4342 | 0.4752 | N/A (pretrained) |
| E05 MiniLM title + ingr. | 0.1808 | 0.1519 | 0.1989 | N/A (pretrained) |
| FT MiniLM fine-tuned | 0.6450 | 0.6209 | 0.6368 | N/A (one checkpoint) |

- Fine-tuning lifts MiniLM by +43% MRR@10 on validation (0.434 → 0.621), but BM25 stays ahead on this proxy.
- Recall, run IDs and timings: [reports/comparison.csv](reports/comparison.csv) and [reports/final/master_comparison.csv](reports/final/master_comparison.csv). Significance tests: [reports/final/significance_test_export.json](reports/final/significance_test_export.json). Error analysis: [reports/error_analysis.md](reports/error_analysis.md).
- The human relevance study (388 query–recipe pairs, 20 queries) is prepared but the labels are still pending; see [docs/final_annotation_workflow.md](docs/final_annotation_workflow.md).

### Dense retrieval and fine-tuning

- `src/embeddings.py` encodes the corpus on CPU (`max_seq_length=128`, L2-normalised) into a bundle: `embeddings.npy`, `recipe_ids.json`, `manifest.json`. Loading a bundle checks every hash and invariant and raises `BundleError` on a mismatch.
- `src/build_pairs.py` builds training pairs from `train` recipes only. Fine-tuned bundles package their `encoder/`, so the API and the evaluator load the fine-tuned weights:

```bash
python -m src.build_pairs --config configs/default.yaml --split train --mode ingredients --max-pairs 30000 --seed 42
python -m src.embeddings --config configs/default.yaml --model-path models/minilm_finetuned-main-cpu1ep
python -m src.evaluation --mode both --split validation
```

- End-to-end training with caching: `python src/train.py --config configs/pilot.yaml` (`--stage`, `--from-stage`, `--force`, `--device cpu`).

## Configs

| File | Corpus | Serving bundle |
| --- | --- | --- |
| `configs/default.yaml` | main | none (reference TF-IDF) |
| `configs/pilot.yaml` | pilot | none |
| `configs/production.yaml` | main | `models/bm25-main-1f1f4ce5` (E03) |

`default.yaml` is the base; the other profiles only override what changes. Seed 42 everywhere.

## API

`api/app.py` loads the retriever and corpus metadata once at startup. The bundle folder decides the loader (`model.joblib` = lexical, `embeddings.npy` = dense); a bundle from another corpus or with a wrong hash stops the startup.

| Variable | Default | Meaning |
| --- | --- | --- |
| `API_CONFIG` | `configs/production.yaml` | config (corpus + bundle) to serve |
| `API_BUNDLE` | - | overrides `serving.bundle` |
| `API_BACKEND` | from config | `reference` (TF-IDF) or `bundle` |
| `API_REQUEST_LOG` | `logs/requests.db` | SQLite request log (`""` disables it) |

| Endpoint | Method | What it does |
| --- | --- | --- |
| `/health` | GET | liveness, used by the Docker health check |
| `/model-info` | GET | model, bundle, corpus hash and validation metrics |
| `/predict` | POST | one ingredient list → `top_k` recipes (1–20) with matched/missing ingredients |
| `/batch-predict` | POST | up to 20 instances, order preserved |
| `/` | GET | web UI |
| `/docs` | GET | Swagger UI |

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["chicken", "rice", "garlic"], "top_k": 3}'
```

Invalid input returns **422** with the reason. Verified BM25 responses: [reports/clean_clone_smoke.json](reports/clean_clone_smoke.json).

**Web UI** (`fe/index.html`): static page, no build step; keyboard accessible, `?q=chicken,rice&k=3` links straight to a search. Screenshots: `reports/figures/ui_390.png`, `ui_1440.png`.

## Monitoring and MLflow

- Every request is logged to SQLite (`RequestLogRecord`). `src/monitor.py` reports KS-test drift on query length, OOV ingredient rate, zero-hit rate, error rate and p50/p95/p99 latency, and returns `INSUFFICIENT_DATA` for small samples.

```bash
python -m src.monitor --log-path logs/requests.db --out reports/monitoring_report.md
python scripts/simulate_drift.py        # no-drift vs drift scenarios, see reports/drift_simulation.md
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

- The repository includes an `mlflow.db` snapshot. Fine-tuned models are registered as `minilm_finetuned` with `staging` / `production` aliases; the API serves from bundles and does not need a tracking server.
- Rollback in one command: `python -m src.train --rollback 1 --alias production` ([trial](reports/mlflow_rollback_trial.md)).

## Benchmark and Docker

```bash
python -m src.benchmark --config configs/default.yaml --out reports/benchmark.json   # p50/p95 latency, peak RAM, startup, size
docker compose up --build              # API on http://127.0.0.1:8080
docker compose --profile mlflow up     # optional MLflow UI on :5000
```

- CPU-only image. `init-bundles` copies the host `models/` into a named volume; `data/` is mounted read-only; request logs persist in `api_logs`.
- Native vs Docker parity: `scripts/check_parity.py` — 5/5 queries identical, zero score difference ([verification](reports/docker_verification.md), 2026-11-03).

## Reproducibility

- `python -m src.data_collection` reproduces the processed files exactly (hashes in the manifest).
- `python -m pytest` runs offline in seconds; tests that need real data skip when it is absent.
- GPU retraining run (2026-11-04, RTX 3060, CUDA 12.8): [results](reports/retraining_20261104/README.md) and [model release](https://github.com/minhhoangb/food-recipe-recommendation-system/releases/tag/model-artifacts-gpu-20261104).
- Collaboration rules: [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository structure

```
api/        FastAPI app and request log
configs/    default, pilot, production
data/       raw (git-ignored) and processed corpus, splits, queries, qrels
fe/         static web UI
models/     serving bundles (git-ignored)
notebooks/  EDA, Colab fine-tuning, ablations
reports/    results, final evaluation, monitoring, benchmarks, figures
scripts/    data fetch/verify, smoke and parity checks, report builders
src/        data pipeline, retrievers, evaluation, fine-tuning, serving, monitoring
tests/      pytest suite
```
