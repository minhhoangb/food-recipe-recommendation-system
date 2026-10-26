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

## Experiment results (Master Comparison Table)

**TODO(Person 1)**: filled from the MLflow runs after the frozen final evaluation.

## API usage

**TODO(Person 2)**: endpoints `GET /health`, `GET /model-info`, `POST /predict`, `POST /batch-predict`, curl and Python examples. Request/response schemas are already defined in `src/contracts.py`.

## Docker

**TODO(Person 2)**: `docker build`, `docker compose up --build`, health check.

### Fine-tuning and training pipeline (Person 3)

`src/build_pairs.py` builds training pairs from `train` recipes using `src.queries.make_query_ingredients` across multiple seeds (up to 10,000 pairs for pilot, seed 42) and Hưng's document representation (`src.embeddings.format_document`), outputting a manifest with SHA256 hashes. It also provides `NoDuplicateBatchSampler` ensuring no batch contains duplicate recipes or duplicate groups.

`notebooks/02_finetune_colab.ipynb` runs fine-tuning on Colab GPU with `SentenceTransformerTrainer` + `MultipleNegativesRankingLoss`, evaluates every epoch with `InformationRetrievalEvaluator` (MRR@10), demonstrates `save_steps` checkpointing and `resume_from_checkpoint`, and exports the model.

Hand-off to Person 2 (Hưng): pass the exported fine-tuned checkpoint via `--model-path`:

```bash
# 1. Build training pairs with verified manifest
python -m src.build_pairs --config configs/pilot.yaml --split train --mode ingredients --max-pairs 10000 --seed 42

# 2. Fine-tune model directly or end-to-end with MLflow logging
python -m src.train --config configs/pilot.yaml --epochs 1 --batch-size 16 --promote

# 3. Person 2 (Hưng) hand-off: build serving bundle using the fine-tuned model checkpoint
python -m src.embeddings --config configs/pilot.yaml --model-path models/minilm_finetuned_checkpoint
```

### Production API & Model Monitoring (Person 3)

`src/monitor.py` parses `RequestLogRecord` entries from `logs/requests.jsonl` to compute request throughput, error rates, p90/p95/p99 latency, zero-hit rate, and vocabulary drift:

```bash
python -m src.monitor --log-path logs/requests.jsonl --out reports/monitoring_report.md --mlflow
```

## Experiment results (Master Comparison Table)

**TODO(Person 1)**: filled from the MLflow runs after the frozen final evaluation.

## API usage

**TODO(Person 2)**: endpoints `GET /health`, `GET /model-info`, `POST /predict`, `POST /batch-predict`, curl and Python examples. Request/response schemas are already defined in `src/contracts.py`.

## Docker

**TODO(Person 2)**: `docker build`, `docker compose up --build`, health check.

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
fe/             static web UI                                (Person 2)
models/         serving bundles (git-ignored)
notebooks/      00_colab_bootstrap.ipynb, 02_finetune_colab.ipynb, EDA
reports/        figures, final report, monitoring_report.md
scripts/        verify_data.py, package_artifacts.py, fetch_artifacts.py
src/            contracts, config, utils, normalize, preprocess, data_collection, queries, reference
                embeddings, retrieval, evaluation, memory    (Person 2)
                build_pairs, finetune, train, monitor        (Person 3)
tests/          pytest suite, tests/fixtures (hand-written, not RecipeNLG)
docs/           implementation plans per owner
```

## Reproducibility

- Seed 42 everywhere; `configs/default.yaml` holds every data parameter and is hashed into each manifest.
- `python -m src.data_collection` reproduces the processed files exactly (hashes are in the manifest; `tests/test_data_collection.py::test_build_is_reproducible` checks it on a fixture).
- `python -m pytest` runs offline in seconds. Tests that read the real processed data skip themselves if it is absent.
- Training from scratch: run `python -m src.train --config configs/pilot.yaml`. Runs on CPU by default; automatically leverages CUDA when available (e.g. on Colab GPU).
- Collaboration rules and file ownership: [CONTRIBUTING.md](CONTRIBUTING.md).
