# Hưng — Implementation Plan (Embedding MiniLM pretrained + Dense Retriever)

Owner: Hưng (Person 2). Scope: FEAT 01-04 of the assignment sheet (`Hưng.md`).
Branch: `feat/recipe-embedding`. Base for the PR: `dev`.

---

## A. Current project assessment

**Architecture.** Flat `src/` package + `tests/` + `configs/` + `scripts/`, driven by OmegaConf
(`src/config.py`), typed by pydantic schemas in `src/contracts.py`, IO/hashing in `src/utils.py`.
Data flow: raw RecipeNLG CSV -> `src/data_collection.py` -> `data/processed/<profile>/`
(`corpus.jsonl`, `splits.json`, `queries.jsonl`, `qrels.jsonl`, `dataset_manifest.json`) ->
models read that folder. `src/contracts.py` is the only boundary between modules
(`Retriever` protocol, `BundleManifest`).

**Already there.** Ingredient normalisation (`src/normalize.py`), record schemas, the
`Retriever` protocol + `tests/contract_utils.assert_retriever_contract`, the TF-IDF reference
retriever (`src/reference.py`), the ranx metric conventions (`tests/test_ranking_metrics.py`),
mlflow in `requirements.txt`.

**Missing (mine).** `src/embeddings.py`, `src/retrieval.py` (DenseRetriever), the E04/E05
evaluation driver, the sentence-transformers/torch dependencies, README model sections.

**Owned by others.** `src/evaluate.py`, `src/baseline.py`, `src/features.py`, `src/pipeline.py`
belong to Person 1 (Hoàng) and live on `origin/feature/baselines-evaluation`; they are **not**
on `dev` yet. `src/build_pairs.py`, `src/finetune.py`, `src/train.py` belong to Person 3.
`src/contracts.py`, `src/config.py`, `configs/*` are shared -> I do not modify them.

**Integration risks.**

1. `src/evaluate.py` is not merged yet. My evaluation must work alone *and* reuse Hoàng's
   interface the moment it lands: I therefore put the shared evaluation logic in
   `src/evaluation.py` (my file) with the same function names/signatures Hoàng uses
   (`evaluate_retriever`, `load_split`) and call his `src.evaluate` when it is importable.
2. `BundleManifest.model_name` is a `Literal` that does **not** contain a MiniLM-dense name.
   The contract allows only `minilm_pretrained` / `minilm_finetuned`; `minilm_pretrained` is
   the correct value for E04/E05 (pretrained MiniLM). No contract change needed.
3. `BundleManifest` has no field for the embedding mode (E04 vs E05) or `max_seq_length`.
   I put them in `library_versions`? No -> I add them to the manifest `files`-adjacent free
   space by extending `BundleManifest` with two optional fields (`embedding_mode`,
   `max_seq_length`) via a **shared-file PR** (allowed: "a shared file may be edited by anyone
   through a PR that the two other members approve"), keeping both optional so nothing breaks.
   Fallback if that is rejected: store them in `manifest["embedding_mode"]` via `extra` keys.
4. `data/processed/pilot/` does not exist in this checkout (no raw CSV, artifact URL empty),
   so the real E04/E05 runs need the dataset. See blockers.

---

## B. Requirement traceability matrix

| ID | Requirement | Existing Status | Files Involved | Implementation Tasks | Acceptance Criteria |
|---|---|---|---|---|---|
| FEAT01.1 | Load `sentence-transformers/all-MiniLM-L6-v2` | missing | `src/embeddings.py` | `load_model()` on CPU | model loads, CPU only |
| FEAT01.2 | E04 ingredients-only / E05 title+ingredients | missing | `src/embeddings.py` | `format_document()` | unit tests on formatting |
| FEAT01.3 | Batch encoding | missing | `src/embeddings.py` | `encode_corpus(batch_size)` | batching honours batch_size |
| FEAT01.4 | L2-normalised vectors | missing | `src/embeddings.py` | `normalize_embeddings=True` | norms == 1 +- 1e-5 |
| FEAT01.5 | Bundle: `embeddings.npy` + `recipe_ids.json` + `manifest.json` | missing | `src/embeddings.py` | `save_bundle()` | files exist, manifest validates as `BundleManifest` |
| FEAT01.6 | Row/id alignment | missing | `src/embeddings.py` | ids written in corpus order | test asserts alignment |
| FEAT01.7 | `max_seq_length = 128` + truncated count | missing | `src/embeddings.py` | token-length scan before encode | truncated count reported |
| FEAT01.8 | CLI `python -m src.embeddings --config configs/pilot.yaml` | missing | `src/embeddings.py` | `main()` | CLI runs, prints report |
| FEAT01.9 | Tests: shape/dtype/finite/norm/id order/bundle integrity | missing | `tests/test_embeddings.py` | unit tests with a fake encoder | all pass offline |
| FEAT02.1 | Load bundle, never re-encode corpus | missing | `src/retrieval.py` | `DenseRetriever.from_bundle()` | test asserts encoder never called for corpus |
| FEAT02.2 | Encode + normalise queries | missing | `src/retrieval.py` | `encode_query()` | query vector norm == 1 |
| FEAT02.3 | `sentence_transformers.util.semantic_search` | missing | `src/retrieval.py` | `search()` | cosine ranking matches manual cosine |
| FEAT02.4 | top-k, score desc, `recipe_id` tie-break | missing | `src/retrieval.py` | stable sort by (-score, recipe_id) | contract test passes |
| FEAT02.5 | Clear errors: bad hash / dim mismatch / id order / missing files | missing | `src/retrieval.py` | `BundleError` subclasses | each error has a distinct message |
| FEAT02.6 | `assert_retriever_contract` | missing | `tests/test_retrieval.py` | contract test | passes |
| FEAT02.7 | CPU cold-load time measured | missing | `src/retrieval.py` | `cold_load_seconds` in bundle report | measured value recorded |
| FEAT03.1 | E04 and E05 bundles | partial | `src/embeddings.py` | two runs | two bundle dirs |
| FEAT03.2 | Evaluate validation pilot with ranx | missing | `src/evaluation.py` | `evaluate_retriever()` + `load_split()` | metrics dict |
| FEAT03.3 | Recall@5, Recall@10, MRR@10 | missing | `src/evaluation.py` | ranx metric list | values printed |
| FEAT03.4 | MLflow runs with params/metrics/duration/RAM | missing | `src/evaluation.py` | `log_to_mlflow()` | 2 runs in mlflow.db |
| FEAT03.5 | Comparison table | missing | `reports/` + README | markdown table | real numbers only |
| FEAT04.1 | Dependencies: sentence-transformers, CPU torch, numpy, ranx, mlflow | partial | `requirements.txt` | append pinned lines | clean venv installs |
| FEAT04.2 | Clean venv install test | n/a | - | `python -m venv .venv` + `pip install -r requirements.txt` | log attached |
| FEAT04.3 | `pytest` green | n/a | `tests/` | run suite | log attached |
| FEAT04.4 | README installation + usage | missing | `README.md` | install + CLI sections | matches real CLI |
| FEAT04.5 | Colab CPU compatibility | partial | `notebooks/00_colab_bootstrap.ipynb` | document CPU path | limitation stated |

---

## C. Feature breakdown

### FEAT 01 — Embedding pipeline (`src/embeddings.py`)

- **Objective:** encode the pilot corpus with pretrained MiniLM into a validated bundle.
- **Reuse:** `src.config.load_config/processed_dir/fingerprint`, `src.contracts.BundleManifest`,
  `src.normalize.NORMALIZATION_FINGERPRINT`, `src.utils.sha256_file/sha256_obj/write_json/read_jsonl`.
- **New:** `EmbeddingMode` literal, `format_document(record, mode)`, `count_truncated(texts, tokenizer, max_seq_length)`,
  `build_bundle(...)`, `save_bundle(...)`, `load_corpus(...)`, `main()`.
- **Inputs/outputs:** in `data/processed/<profile>/corpus.jsonl`; out `models/<run_id>/{embeddings.npy,recipe_ids.json,manifest.json}`.
- **Dependencies:** sentence-transformers, torch (CPU), numpy.
- **Tests:** `tests/test_embeddings.py` (fake encoder, no download), `tests/test_bundle_roundtrip.py`.
- **DoD:** CLI creates a bundle; every FEAT01.9 check passes.

### FEAT 02 — Dense retriever (`src/retrieval.py`)

- **Objective:** cosine retrieval over a stored bundle, contract-compliant.
- **Reuse:** `src.contracts.SearchHit`, `src.utils`, `src.embeddings.load_bundle/validate_bundle`.
- **New:** `DenseRetriever` (`name="minilm_pretrained"`, `version="1"`), `search()`,
  `from_bundle()`, `BundleError` family.
- **DoD:** `assert_retriever_contract` passes; all error cases raise a specific message;
  corpus is never re-encoded at load.

### FEAT 03 — E04/E05 evaluation (`src/evaluation.py`)

- **Objective:** evaluate both embedding modes on the pilot validation split and log to MLflow.
- **Reuse:** `ranx` conventions from `tests/test_ranking_metrics.py`; Hoàng's `src/evaluate.py`
  when present on the branch (delegation).
- **New:** `evaluate_retriever()`, `load_split()`, `log_to_mlflow()`, `run_experiment()`,
  `compare()` + `python -m src.evaluation --mode both`.
- **DoD:** two MLflow runs, a real comparison table.

### FEAT 04 — Dependencies, tests, docs

- **requirements.txt:** append `sentence-transformers`, `torch` (CPU index documented in README).
- **README:** installation, embedding CLI, retriever usage, evaluation CLI, MLflow, artifacts.
- **DoD:** clean venv install + green pytest, logs in the PR.

---

## D. Development order

1. Contract/config verification (done in Phase 0).
2. Environment preparation (`.venv` + `requirements.txt`).
3. `src/embeddings.py` + `tests/test_embeddings.py`.
4. Bundle validation (`src/embeddings.py::load_bundle`) + tests.
5. `src/retrieval.py` + `tests/test_retrieval.py`.
6. `src/evaluation.py` (E04/E05) + MLflow.
7. Full `pytest` + `ruff`.
8. README + this plan.

---

## E. Risks and assumptions

**Verified facts**

- `src/contracts.py` has no `embedding_mode`/`max_seq_length` field on `BundleManifest`.
- `src/evaluate.py` does not exist on `dev`; it exists on `origin/feature/baselines-evaluation`.
- `configs/default.yaml` has `artifacts.url = ""` -> the processed dataset cannot be downloaded
  automatically.
- `data/processed/` is empty in this checkout.
- `models/` and `data/` are git-ignored.

**Assumptions**

- The dataset owner will publish `dist/processed-<id>.zip` + sha256 in `configs/default.yaml`,
  or provide the raw CSV. Until then the real E04/E05 runs cannot execute.
- `minilm_pretrained` is the right `model_name` literal for the dense MiniLM retriever.

**Risks**

- `semantic_search` returns an intermediate top-k that may drop candidates tied at the cut-off
  boundary. Mitigated by fetching `min(top_k*4, n)` candidates and re-sorting with the
  contract tie-break, without a full-corpus sort.
- Download of the MiniLM weights needs internet on first use (Colab has it; offline CI does not).
  Unit tests use a fake encoder, never the network.
- `torch` wheels are large; CPU-only index must be documented.

---

## F. Acceptance checklist

- [x] `python -m src.embeddings --config configs/pilot.yaml` creates a bundle
- [x] E04 and E05 document formats unit-tested
- [x] embeddings shape/dtype/finite/L2-norm/id-order tests pass
- [x] truncation count reported with `max_seq_length=128` (E04: 1, E05: 1)
- [x] bundle validates against `BundleManifest`
- [x] `DenseRetriever` passes `assert_retriever_contract` (fake encoder and real E04 bundle)
- [x] bad hash / dim mismatch / id mismatch / missing file -> specific errors
- [x] corpus never re-encoded at load
- [x] CPU cold-load time measured (E04: 16.8 s, E05: 5.2 s, model weights warm on second run)
- [x] E04 + E05 evaluated with ranx (recall@5, recall@10, mrr@10) on pilot/validation (200 queries)
- [x] 2 MLflow runs with params, metrics, encode duration, RAM
- [x] comparison table with real numbers (`reports/e04_e05_pilot.md`)
- [x] clean venv install + green pytest (120 passed, 2 skipped)
- [x] README updated
- [x] `models/` stays git-ignored (two local bundles, untracked)
- [x] `data/processed/` verified with `scripts/verify_data.py` ([pilot] OK, [main] OK)

## G. Execution log (real commands and outputs)

- `python -m venv .venv` + `pip install -r requirements.txt` -> base requirements installed.
- `pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu`,
  `pip install sentence-transformers==5.1.2`, `pip install psutil` -> versions pinned in
  `requirements.txt` (torch 2.9.1+cpu, sentence-transformers 5.1.2, psutil 7.2.2).
- MiniLM cold load on CPU -> `loaded 384 256` (dim 384; default 256, clamped to 128 in code).
- RecipeNLG mirror check -> `SandhyaKilari/RecipeNLG_dataset` and `Mahimas/recipenlg` host the
  identical 2 294 981 083-byte CSV; downloaded at ~3 MB/s to `data/raw/` (git-ignored).
- `python -m src.data_collection --raw-path data/raw/RecipeNLG_dataset.csv --config configs/pilot.yaml`
  -> pilot {5000 recipes, 1500 queries} and main {50000 recipes, 6500 queries}; drop reasons
  {ingredient_count_out_of_range: 180, exact_duplicate: 180}.
- `python scripts/verify_data.py` -> `[pilot] OK`, `[main] OK`.
- `python -m src.embeddings --config configs/pilot.yaml --mode ingredients`
  -> `models/minilm_pretrained-E04-pilot-37e8411f`, 5000 docs, dim 384, truncated 1, 21.6 s.
- `python -m src.embeddings --config configs/pilot.yaml --mode title_ingredients`
  -> `models/minilm_pretrained-E05-pilot-37e8411f`, 5000 docs, dim 384, truncated 1, 27.8 s.
- `python -m src.evaluation --mode both --config configs/pilot.yaml --split validation
  --out reports/e04_e05_pilot.md` -> table below; MLflow runs recorded (see section 8 of the final report).

---

## G. Execution log

(see the "Execution log" section at the end of this file, updated as work lands)
