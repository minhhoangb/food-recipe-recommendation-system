# Food Recipe Recommendation System

Type the ingredients you have, get recipes that use them, with what you already have and what is still missing. Ingredient-list retrieval on the RecipeNLG corpus (English only), comparing TF-IDF / BM25 / LSA baselines with a pretrained and a fine-tuned `all-MiniLM-L6-v2` encoder, served by a FastAPI app with a small web UI.

> Status: the shared foundation (data, contracts, metrics, tooling) is done. Sections marked **TODO(owner)** are filled in by their owner as the work lands. Do not put numbers here that do not come from a run record.

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
pip install -r requirements-dev.txt
python -m pytest
```

Python 3.11 or 3.12 is recommended for the model work. Serving and training dependencies are added to `requirements.txt` / `requirements-train.txt` by their owners after testing (**TODO(owner)**).

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

## Experiment results (Master Comparison Table)

**TODO(Person 1)**: filled from `reports/runs/` after the frozen final evaluation.

## API usage

**TODO(Person 2)**: endpoints `GET /health`, `GET /model-info`, `POST /predict`, `POST /batch-predict`, curl and Python examples. Request/response schemas are already defined in `src/contracts.py`.

## Docker

**TODO(Person 2)**: `docker build`, `docker compose up --build`, health check.

## MLflow

**TODO(Person 3)**: launching the UI, experiments, registry (`staging` / `production`).

## Repository structure

```
api/            FastAPI app                                  (Person 2)
configs/        default.yaml, pilot.yaml
data/raw/       original CSV, never modified (git-ignored)
data/processed/ corpus.jsonl, splits.json, queries.jsonl, qrels.jsonl, manifest per profile
fe/             static web UI                                (Person 2)
models/         serving bundles (git-ignored)
notebooks/      00_colab_bootstrap.ipynb, EDA, fine-tuning
reports/        figures, runs/ (JSON run records), final report
scripts/        verify_data.py, package_artifacts.py, fetch_artifacts.py
src/            contracts, config, normalize, preprocess, data_collection, queries, reference
tests/          pytest suite, tests/fixtures (hand-written, not RecipeNLG)
```

## Reproducibility

- Seed 42 everywhere; `configs/default.yaml` holds every data parameter and is hashed into each manifest.
- `python -m src.data_collection` reproduces the processed files exactly (hashes are in the manifest; `tests/test_data_collection.py::test_build_is_reproducible` checks it on a fixture).
- `python -m pytest` runs offline in seconds. Tests that read the real processed data skip themselves if it is absent.
- Training from scratch (`python src/train.py`) and CPU fallback: **TODO(Person 3)**.
- Collaboration rules and file ownership: [CONTRIBUTING.md](CONTRIBUTING.md).
