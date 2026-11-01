# Food Recipe Recommendation System

Type the ingredients you have, get recipes that use them, with what you already have and what is still missing. Ingredient-list retrieval on the RecipeNLG corpus (English only), comparing TF-IDF / BM25 / LSA baselines with a pretrained and a fine-tuned `all-MiniLM-L6-v2` encoder, served by a FastAPI app with a small web UI.

> Status: the shared foundation (data, contracts, metrics, tooling) is done. Sections marked **TODO(owner)** are filled in by their owner as the work lands. Do not put numbers here that do not come from an MLflow run.

## Team

| Member | Student ID | Role |
| --- | --- | --- |
| TODO | TODO | Person 1: baselines, evaluation, analysis |
| TODO | TODO | Person 2: embeddings, API, UI, Docker |
| TODO | TODO | Person 3: fine-tuning, MLflow, monitoring |

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

`requirements.txt` has everything, the training stack included. `requirements-serve.txt` is
the same list without the training extras (`accelerate`, `datasets`): those two cannot be
installed next to `transformers` — `datasets>=5.1` needs `huggingface-hub>=1.31` while
`transformers` needs `<1.0` — so the Docker image installs the serving list instead.

```bash
pip install -r requirements-serve.txt     # only what the API needs
```

Python 3.11 or 3.12 is recommended for the model work. Serving and training dependencies are appended to `requirements.txt` by their owners after testing (**TODO(owner)**).

### CPU-only torch (needed for the MiniLM encoder)

`requirements.txt` pins `torch` and `sentence-transformers`. On Linux the default `torch`
wheel bundles CUDA; to keep the environment small and CPU-only, install torch from the CPU
index **before** the rest of the requirements:

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

On Windows and macOS the default torch wheel is already CPU-only, so `pip install -r
requirements.txt` is enough. Everything in this project runs on CPU; CUDA is never
required. Google Colab works the same way:

```bash
%pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
%pip install -r requirements.txt
```

The first run downloads the pretrained weights (~90 MB) into the Hugging Face cache; after
that the pipeline is offline. `src/embeddings.py` and `src/retrieval.py` never ask for
CUDA (`device="cpu"`), so Colab's CPU runtime is enough.

## Dataset

- **RecipeNLG** (Bien et al., INLG 2020), about 2.23 million recipes, 7 columns: `title, ingredients, directions, link, source, NER` plus an index. Project page: https://recipenlg.cs.put.poznan.pl/ (also mirrored on Kaggle). Read the dataset's own terms of use before redistributing anything; this repository does not redistribute the raw file and no licence has been assumed here.
- The raw CSV (2.2 GB) goes to `data/raw/RecipeNLG_dataset.csv` and is never modified. You can also point to it with `--raw-path` or the `RECIPENLG_RAW` environment variable.
- We use a seeded random subset: **50,000 recipes** (`main`) and its first **5,000** (`pilot`, a nested subset with identical splits). Sampling is a seeded permutation of every raw id over the *whole* file; the file is sorted by source, so taking the first rows would be biased.

Get the processed data one of two ways:

```bash
# A) download the shared zip (URL + sha256 are set in configs/default.yaml by the data owner)
python scripts/fetch_artifacts.py
# B) build it yourself from the raw CSV (about 1 minute); corpus/splits/queries/qrels are identical
#    (compare the file hashes in dataset_manifest.json; only its created_utc differs)
python -m src.data_collection --raw-path data/raw/RecipeNLG_dataset.csv
python scripts/verify_data.py
```

### What the pipeline does

1. Safe parsing of the stringified lists (JSON, then `ast.literal_eval`; never `eval`). Rows with missing or malformed fields are dropped and counted per reason in `dataset_manifest.json`.
2. Ingredient normalisation (`src/normalize.py`): `ingredient-parser-nlp` removes quantities, units and preparation words, `inflect` singularises, multi-word names are kept (`chicken stock` is not `chicken breast`). The NER column is the primary source, raw lines are the fallback. First import downloads one small NLTK resource (needs internet once; in Docker run `python -c "import ingredient_parser"` at build time).
3. Exact duplicate removal, then grouping recipes with an identical ingredient set or identical title (scipy connected components). Groups never cross splits. Limitation: no all-pairs similarity, so near-duplicates with different titles and ingredients stay separate.
4. Group-wise split with scikit-learn `GroupShuffleSplit`: about 20% test, 8% validation, 72% train (seed 42; fractions apply to groups, so recipe shares are approximate).
5. Synthetic queries: 2-6 ingredients sampled from a recipe with at least one hidden, always containing at least one non-common ingredient (document frequency <= 2%). The only known positive is the origin recipe, so Recall/MRR on these queries are an *origin-recipe retrieval proxy*, not a relevance score.

Current sizes are in `data/processed/<profile>/dataset_manifest.json`.

## ML pipeline and models

**TODO(Person 1/2/3)**: preprocessing/feature description, model list (E01-E10), tuning, results. Interfaces are fixed in `src/contracts.py` (`Retriever.search(ingredients, top_k) -> list[SearchHit]`).

### Pretrained MiniLM encoder and dense retriever (Person 2)

`src/embeddings.py` encodes the corpus with `sentence-transformers/all-MiniLM-L6-v2` on CPU,
in batches, with `max_seq_length=128` and L2-normalised vectors. Two document
representations are supported:

| Mode | Label | Embedded text |
| --- | --- | --- |
| `ingredients` | E04 | `"garlic, rice, olive oil"` |
| `title_ingredients` | E05 | `"Garlic Chicken. Ingredients: garlic, rice, olive oil"` |

```bash
# E04 bundle -> models/minilm_pretrained-E04-pilot-<corpus-sha8>/
python -m src.embeddings --config configs/pilot.yaml --mode ingredients
# E05 bundle
python -m src.embeddings --config configs/pilot.yaml --mode title_ingredients
```

Each bundle holds `embeddings.npy` (float32, one row per recipe, L2-normalised),
`recipe_ids.json` (the same order) and `manifest.json` (a `BundleManifest` plus the
embedding mode, `max_seq_length`, the truncated-document count and the measured encode
time). The bundles live under `models/` and are **not** committed to git.

`src/retrieval.py` loads a bundle and searches it by cosine similarity
(`sentence_transformers.util.semantic_search` over normalised vectors). The corpus is
never re-encoded: only the query goes through the model. Results follow the Retriever
contract: at most `top_k` hits, score descending, ties broken by `recipe_id` ascending.

```python
from src.retrieval import DenseRetriever

retriever = DenseRetriever.from_bundle("models/minilm_pretrained-E04-pilot-<sha8>")
for hit in retriever.search(["chicken", "rice"], 5):
    print(hit.recipe_id, hit.score)
```

Loading a bundle verifies every invariant the manifest declares (file hashes, recipe id
order, vector count, embedding dimension, normalisation) and raises a `BundleError` that
names the broken invariant.

### E04 / E05 evaluation

`src/evaluation.py` embeds (or reuses a bundle), retrieves and scores the validation split
with `ranx` (Recall@5, Recall@10, MRR@10), then logs the run to MLflow with the model,
the embedding mode, the corpus id, the measured encode time, the CPU cold-load time and
the peak RAM.

```bash
python -m src.evaluation --mode both --config configs/pilot.yaml --split validation
python -m src.evaluation --mode both --split validation --out reports/e04_e05_pilot.md
```

MLflow runs are stored in `mlflow.db` (git-ignored); open them with `mlflow ui`. Person 1's
`src/evaluate.py` owns the shared baseline evaluation; the metric conventions are the ones
locked in `tests/test_ranking_metrics.py`.

### Fine-tuning and training pipeline (Person 3)

`src/build_pairs.py` builds training pairs from `train` recipes using `src.queries.make_query_ingredients` across multiple seeds (up to 10,000 pairs for pilot, 30,000 for main, seed 42) and Hưng's document representation (`src.embeddings.format_document`), outputting a manifest with SHA256 hashes. It also provides `NoDuplicateBatchSampler` ensuring no batch contains duplicate recipes or duplicate groups.

- **Pilot Fine-Tuning**: `notebooks/02_finetune_colab.ipynb` runs pilot fine-tuning on Colab GPU with `SentenceTransformerTrainer` + `MultipleNegativesRankingLoss`, evaluates every epoch with `InformationRetrievalEvaluator` (MRR@10), demonstrates `save_steps` checkpointing and `resume_from_checkpoint`, and exports the model.
- **Main CPU run (2026-10-29)**: `src/finetune.py` trained one epoch on 30,000 train pairs (batch 32, lr 2e-5). The checkpoint is `models/minilm_finetuned-main-cpu1ep`; measured validation results and MLflow IDs are in [reports/e07_e08_finetune_main.md](reports/e07_e08_finetune_main.md). The current training loader shuffles pairs; it does not use the available `NoDuplicateBatchSampler`.
- **Main ablation notebook**: `notebooks/03_finetune_main_ablation.ipynb` provides a workflow for measured hyperparameter experiments and exports the checkpoint with the best validation MRR@10. A controlled weight-decay ablation is still pending; see [the experiment status](reports/ablation_regularization.md).

#### Main validation: pretrained vs CPU one-epoch fine-tuned MiniLM

Measured on `main` validation (500 queries over 50,000 recipes) using synthetic origin-recipe qrels. FT queries and corpus vectors use the same checkpoint. No test retrieval metrics were computed.

| Exp ID | Pipeline Description | Recall@5 | Recall@10 | MRR@10 | Delta vs Baseline |
|---|---|---|---|---|---|
| **E04** | Pretrained MiniLM, ingredients | 0.5120 | 0.5760 | 0.4342 | Baseline |
| **FT (E08)** | Fine-tuned MiniLM on main, ingredients, CPU one epoch | **0.7340** | **0.7980** | **0.6209** | **+43.0% MRR@10 (+0.1867)** |
| **E05** | Pretrained MiniLM, title + ingredients | 0.2140 | 0.2620 | 0.1519 | Different representation |

Experiment ids follow the team contract (E07 = pilot fine-tuning, E08 = main fine-tuning). No fine-tuned title + ingredients bundle and no hybrid pipeline were evaluated in this run. BM25 remains above FT on this proxy (MRR@10 0.7583). See [the full comparison](reports/comparison.csv), [error analysis](reports/error_analysis.md) and [dev judging assignments](reports/manual_eval/README.md).

#### MLflow Tracking & Model Registry

Evaluation runs, the imported historical training log, and report artifacts are stored in the local SQLite backend:

```bash
# Launch MLflow UI to inspect runs and model registry
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

- **Experiment:** `recipe-retrieval`; per-system evaluation run IDs are in `reports/comparison.csv`.
- **Checkpoint provenance:** [reports/finetune_cpu1ep_provenance.json](reports/finetune_cpu1ep_provenance.json). This evaluation does not register or promote a model alias.

#### Hand-off to Person 2 (Hưng): Indexing with Fine-Tuned Checkpoint

Pass the exported fine-tuned checkpoint bundle via `--model-path`:

```bash
# 1. Build training pairs on main (30,000 pairs, seed 42)
python -m src.build_pairs --config configs/default.yaml --split train --mode ingredients --max-pairs 30000 --seed 42

# 2. Build index bundle using the CPU one-epoch checkpoint (FT / E08)
python -m src.embeddings --config configs/default.yaml --model-path models/minilm_finetuned-main-cpu1ep
```

### Production API & Model Monitoring (Person 3)

`src/monitor.py` calculates statistical drift via SciPy Kolmogorov-Smirnov (KS 2-sample test, $\alpha=0.05$) and operational metrics over `RequestLogRecord` entries from API request logs:
- **Query Length & Token Drift:** KS test comparing query length distributions against reference queries.
- **OOV (Out-Of-Vocabulary) Ingredients:** Percentage of requested ingredients not in training vocabulary (drift threshold > 5%).
- **Missing Ingredients & Latency:** p50/p95/p99 latency (ms), zero-hit rate, and system error rate (threshold > 5%).
- **Sample Sufficiency:** Returns `INSUFFICIENT_DATA` if sample size < `min_samples` (default: 10) instead of false positives.
- **HTML & Markdown Reporting:** Generates a standalone, dark-themed HTML report with simulation disclaimers:

```bash
# Monitor drift against reference queries and export HTML report
python -m src.monitor --reference data/processed/pilot/queries.jsonl --requests logs/simulated_requests_drift.jsonl --out reports/monitoring_report.html

# Run simulated drift scenarios (no-drift vs drifted)
python scripts/simulate_drift.py
```

*See simulation reports and analysis in [reports/drift_simulation.md](reports/drift_simulation.md), [reports/monitoring_nodrift_simulation.html](reports/monitoring_nodrift_simulation.html), and [reports/monitoring_drift_simulation.html](reports/monitoring_drift_simulation.html).*

#### MLflow Model Registry & Instant Rollback

The model registry uses aliases to manage deployments cleanly without redeploying code:
- `@production`: Active model serving live user traffic (Version 2).
- `@staging`: Candidate model undergoing validation/QA (Version 1).
- **Offline Serving Decoupling:** Serving bundles (`models/<bundle_name>`) hold static `embeddings.npy` and `manifest.json`. The API (`src.serving` / `api.app`) runs independently without requiring an active MLflow tracking server.
- **Single-command Rollback:**
  ```bash
  # Rollback @production alias to Version 1
  python -m src.train --rollback 1 --alias production
  ```
*Detailed trial execution and SOP documented in [reports/mlflow_rollback_trial.md](reports/mlflow_rollback_trial.md).*

## Experiment results (Master Comparison Table)

**TODO(Person 1)**: filled from the MLflow runs after the frozen final evaluation.

## API usage

The FastAPI app lives in `api/app.py`. The retriever and the corpus metadata are loaded
**once** at startup (the lifespan), not per request.

```bash
# reference backend (TF-IDF over the processed corpus; no model download)
uvicorn api.app:app --reload

# a bundle: the model is chosen in the config (serving.bundle), not in the code
set API_CONFIG=configs/default.yaml
uvicorn api.app:app --reload

# the env var overrides the config
set API_BUNDLE=models/minilm_pretrained-E04-main-1f1f4ce5
uvicorn api.app:app --reload
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `API_BACKEND` | from the config | `reference` (TF-IDF) or `bundle` (the bundle of `serving.bundle`) |
| `API_CONFIG` | `configs/pilot.yaml` | which processed profile to serve |
| `API_BUNDLE` | - | bundle folder; overrides `serving.bundle` from the config |
| `API_REQUEST_LOG` | `logs/requests.db` | SQLite request log (`""` disables it) |

### Choosing the model with the config

`serving.bundle` in `configs/default.yaml` (or in any profile file) says which bundle the
API serves. The folder decides the loader: `model.joblib` is a lexical bundle (tfidf, bm25,
lsa, written by `src.baseline`), `embeddings.npy` is a dense one (MiniLM, written by
`src.embeddings`). Both are checked against their manifest before the app answers a single
request, so a bundle with a wrong hash stops the startup instead of serving wrong results.

```yaml
serving:
  bundle: models/bm25-main-1f1f4ce5
```

`/model-info` then reports the model, the bundle, the corpus hash and the validation
metrics of the run that produced the bundle (read from the manifest or from MLflow).

Endpoints (`src/contracts.py` owns the request/response schemas):

| Endpoint | Method | What it does |
| --- | --- | --- |
| `/health` | GET | `{"status": "ok", ...}` — used by the Docker health check |
| `/model-info` | GET | the serving model, the bundle, the corpus hash and the metrics |
| `/predict` | POST | one ingredient list -> `top_k` recipes with matched/missing ingredients |
| `/batch-predict` | POST | up to 20 instances; the order of the input is preserved |
| `/` | GET | the web UI (`fe/index.html`) |

A bad request returns **422** with the reason (e.g. `top_k` out of range, an ingredient
list that normalises to nothing). `top_k` is an integer in `[1, 20]`, a batch holds at
most 20 instances.

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["chicken", "rice"], "top_k": 3}'

curl -X POST http://127.0.0.1:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"ingredients": ["chicken"], "top_k": 3}, {"ingredients": ["rice"], "top_k": 3}]}'
```

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"ingredients": ["chicken", "rice"], "top_k": 3},
    timeout=30,
)
for result in response.json()["results"]:
    print(result["title"], result["matched_ingredients"], result["missing_ingredients"])
```

Every request is appended to the SQLite request log
(`RequestLogRecord` from `src/contracts.py`): request id, timestamp, endpoint, model,
normalised ingredients, `top_k`, result ids, scores, latency and status code. SQLite is
used because it is safe with several workers writing at the same time (WAL mode + a busy
timeout).

### recommend() and the E06 rerank

`src/predict.py` holds the recommendation logic used by the API:

```python
from src.predict import load_corpus_metadata, recommend

metadata = load_corpus_metadata("data/processed/main/corpus.jsonl")
results = recommend(["chicken", "rice"], 5, retriever=retriever, metadata=metadata)
```

`matched_ingredients` are the query ingredients the recipe uses, `missing_ingredients`
the recipe ingredients the user does not have (both on the normalised ingredient sets).

The **missing-first rerank** (E06 ablation) ranks the top 100 candidates by the number
of missing ingredients (fewest first). It is **off by default**; `reports/e04_e05_main_benchmark.md`
shows that it costs 34% of the relative Recall@10 on the main corpus (0.576 -> 0.380),
because the synthetic label is the origin recipe, which is exactly the recipe with the
most missing ingredients.

## Web UI

`fe/index.html` is a static page (HTML/CSS/JS, no build step, no dependency) that the API
serves through `StaticFiles`. It asks for ingredients and a number of results, and shows for
every recipe the title, the ingredients you have, the ones you are missing, the directions
(behind a toggle) and the link to the source recipe.

- The page is in English, works with the keyboard only (labels, visible focus, `aria-live`
  status) and shows a loading, empty, error and retry state.
- Recipe content is written with `textContent`, never with `innerHTML`, and a link only
  becomes clickable when its protocol is `http:` or `https:`.
- `?q=chicken,rice,garlic&k=3` fills the form and searches, so a result page can be linked.

```bash
uvicorn api.app:app --reload
# open http://127.0.0.1:8000/
```

The screenshots of the task (390 px and 1440 px, real results from the API) are in
`reports/figures/ui_390.png` and `reports/figures/ui_1440.png`. They are taken with
headless Chrome:

```bash
python scripts/shoot_ui.py --url http://127.0.0.1:8000/ --ingredients "chicken, rice, garlic"
```

## Benchmark

`src/benchmark.py` measures the serving bundles: p50/p95 latency, peak RAM, startup time
and bundle size. Every bundle is measured in its **own process** (peak RSS is a process
high-water mark), with 20 warmup queries and 100 measured queries, one after the other.

```bash
python -m src.benchmark --config configs/default.yaml --out reports/benchmark.json
```

It writes `reports/benchmark.json` (the numbers plus the CPU and the RAM of the machine) and
`reports/benchmark.md` (the same numbers as a table). A bundle built from another corpus is
skipped instead of being timed on queries that do not belong to it.

## Docker

The image is CPU-only: torch comes from the CPU wheel index, so no CUDA is pulled in.

```bash
docker compose up --build          # API on http://127.0.0.1:8000
docker compose --profile mlflow up  # optional MLflow UI on http://127.0.0.1:5000
```

- `data/` and `models/` are mounted read-only; the request log goes to the `api_logs`
  volume.
- The health check (`HEALTHCHECK` in the `Dockerfile`) calls `/health` every 30 s.
- To serve a bundle, point the config at it (or set `API_BUNDLE`):

```bash
API_BUNDLE=models/minilm_pretrained-E04-main-1f1f4ce5 \
  API_CONFIG=configs/default.yaml docker compose up --build
```

The image size is reported in `reports/e04_e05_main_benchmark.md` after the build
(torch CPU wheels are the bulk of it).

The image bakes in the ingredient-parser NLTK resource **and** the MiniLM weights, so the
dense backend also starts without network access. The corpus and the bundles are still
mounted at runtime (`data/`, `models/`), never baked in.

### Native vs Docker parity

The same bundle must give the same answers on the host and in the container.
`scripts/check_parity.py` sends the same queries to both servers and compares the recipe
ids (same order) and the scores (within a tolerance); the log of the check is
`reports/parity.json`.

```bash
# native
API_CONFIG=configs/default.yaml API_BUNDLE=models/bm25-main-1f1f4ce5 \
  uvicorn api.app:app --port 8010
# container
API_BUNDLE=models/bm25-main-1f1f4ce5 API_CONFIG=configs/default.yaml \
  docker compose up -d --build
python scripts/check_parity.py --native http://127.0.0.1:8010 --docker http://127.0.0.1:8000
```

## MLflow

MLflow tracks all baseline runs, pretrained evaluations, and fine-tuning experiments under `mlflow.db`:

```bash
# Launch the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Experiments are recorded in `recipe-retrieval`. Serving bundles are saved as artifacts, and fine-tuned models are registered in the MLflow Model Registry under `minilm_finetuned` with aliases (`staging`, `production`).

## Repository structure

```
api/            FastAPI app                                  (Person 2)
configs/        default.yaml, pilot.yaml
data/raw/       original CSV, never modified (git-ignored)
data/processed/ corpus.jsonl, splits.json, queries.jsonl, qrels.jsonl, manifest per profile
fe/             static web UI (index.html)                   (Person 2)
models/         serving bundles (git-ignored)
notebooks/      00_colab_bootstrap.ipynb, 02_finetune_colab.ipynb, EDA
reports/        figures, final report, monitoring_report.md, benchmark.json
scripts/        verify_data.py, package_artifacts.py, fetch_artifacts.py
                shoot_ui.py, check_parity.py                  (Person 2)
src/            contracts, config, utils, normalize, preprocess, data_collection, queries, reference
                embeddings, retrieval, evaluation, memory, serving, benchmark  (Person 2)
                build_pairs, finetune, train, monitor        (Person 3)
tests/          pytest suite, tests/fixtures (hand-written, not RecipeNLG)
docs/           implementation plans per owner
```

## Reproducibility

- Seed 42 everywhere; `configs/default.yaml` holds every data parameter and is hashed into each manifest.
- `python -m src.data_collection` reproduces the processed files exactly (hashes are in the manifest; `tests/test_data_collection.py::test_build_is_reproducible` checks it on a fixture).
- `python -m pytest` runs offline in seconds. Tests that read the real processed data skip themselves if it is absent.
- Training from scratch: run `python src/train.py --config configs/pilot.yaml`. Supports multi-stage orchestration (`raw`, `processed`, `baseline`, `embeddings`, `pairs`, `finetune`, `evaluate`) with hash-based caching to avoid redundant computation.
  - `--stage <name>`: Execute a single stage (e.g., `--stage pairs`).
  - `--from-stage <name>`: Run sequentially starting from a specific stage.
  - `--force`: Bypass hash caching and force re-computation.
  - `--device cpu`: Guaranteed CPU pathway for environments without GPU.
  - Missing raw dataset: Exits cleanly with clear data acquisition instructions (never performs silent background downloads).
- Collaboration rules and file ownership: [CONTRIBUTING.md](CONTRIBUTING.md).
