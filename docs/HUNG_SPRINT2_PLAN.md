# Hưng — Sprint 2: index main, recommend(), FastAPI, Docker

Owner: Hưng (Person 2). Task sheet: `Hưng_Sprind2.md` (deadline 27/10 24:00).
Branch: `feat/recipe-embedding` (same branch as sprint 1; new commits on top).

Sprint 1 (`docs/HUNG_IMPLEMENTATION_PLAN.md`) delivered the MiniLM embedding pipeline,
the dense retriever and the E04/E05 evaluation on the pilot corpus. Sprint 2 scales it to
the **main** corpus (50 000 recipes), adds the recommendation layer, the FastAPI service
and the Docker image.

---

## A. Requirement traceability

| ID | Requirement | Status | Files | Acceptance evidence |
| --- | --- | --- | --- | --- |
| a.1 | Encode MAIN 50k with MiniLM (both formats) | done | `src/embeddings.py` | `models/minilm_pretrained-E04-main-1f1f4ce5`, `...-E05-main-1f1f4ce5` |
| a.2 | Bundle main + manifest with matching hashes | done | `src/embeddings.py` | `validate_bundle` checks every file hash; manifest carries `corpus_sha256` |
| a.3 | Benchmark encode time, size, RAM | done | `reports/e04_e05_main_benchmark.md` | 238.9 s / 333.0 s, 74.3 MB per bundle, 717-828 MB peak RSS |
| a.4 | E04/E05 on validation main | done | `src/evaluation.py` | R@10 0.576 (E04) vs 0.262 (E05), 500 queries |
| b.1 | `recommend()` in `src/predict.py` | done | `src/predict.py` | normalise -> search -> join metadata -> matched/missing |
| b.2 | matched/missing on the normalised sets | done | `src/predict.py`, `tests/test_predict.py` | 15 tests |
| b.3 | Missing-first rerank (top 100), off by default | done | `src/predict.py::MissingFirstRetriever` | `rerank="none"` is the default |
| b.4 | E06 ablation none vs missing-first | done | `src/evaluation.py::run_rerank_ablation` | R@10 0.576 -> 0.380 on main (cost documented) |
| c.1 | FastAPI app with 4 endpoints | done | `api/app.py` | `/health`, `/model-info`, `/predict`, `/batch-predict` |
| c.2 | 422 with a description on bad input | done | `src/contracts.py` (schema), `api/app.py` | top_k bool/0/21, empty list, unusable ingredients |
| c.3 | Bundle loaded once in the lifespan | done | `api/app.py::lifespan` | retriever + metadata built once at startup |
| c.4 | Request log (`RequestLogRecord`) | done | `api/request_log.py` | SQLite in WAL mode (safe with concurrent writers) |
| c.5 | >= 10 API tests with TestClient | done | `tests/test_api.py` | 17 tests |
| c.6 | uvicorn runs with TfidfReference and with a dense bundle | done | `api/app.py` | `API_BACKEND=reference|dense` |
| c.7 | README curl examples | done | `README.md` | API section |
| d.1 | Dockerfile, python slim, CPU torch | done | `Dockerfile` | torch from the CPU index, no CUDA |
| d.2 | docker-compose with api + optional mlflow | done | `docker-compose.yml` | `api` service, `mlflow` profile, volumes |
| d.3 | `docker compose up --build` works, `/health` OK | done | - | see the execution log |
| d.4 | Image size reported | done | `reports/e04_e05_main_benchmark.md` | measured after the build |

---

## B. Design decisions

### 1. `recommend()` in `src/predict.py` (new file)

`src/predict.py` is Person 2 territory (see `CONTRIBUTING.md`). It holds the whole
recommendation path so both the API and the ablation use the same code:

```
normalise_ingredients(query) -> retriever.search(query, top_k)
    -> for every hit: join the corpus metadata by recipe_id
    -> matched = query ∩ recipe, missing = recipe \ query   (both normalised)
```

The metadata is a `dict[recipe_id -> row]`, so the result rows follow the *hits*, never the
row order of the corpus (the corpus is a JSONL file, not a table).

`matched_ingredients` keeps the order of the query, `missing_ingredients` the order of the
recipe; both are deduplicated.

### 2. The rerank is a `Retriever`, not a flag inside `recommend()`

`MissingFirstRetriever` wraps another retriever and returns hits ordered by
`(missing_count, -score, recipe_id)`. Wrapping it means the evaluation driver can score
it like any other model (`evaluate_retriever` only needs `search()`), and the API can
enable it by swapping the retriever.

It deliberately breaks the "score descending" clause of the contract: that is the point of
the ablation, and it is why the wrapper is *not* passed to `assert_retriever_contract`.

The rerank asks the inner retriever for the **top 100** candidates (`candidates=100`),
re-sorts them and keeps `top_k`. Without the rerank only `top_k` candidates are fetched.

### 3. The API loads its model once (lifespan)

`api/app.py` builds the retriever and the metadata in the `lifespan` handler, so the MiniLM
weights and the 50 000 vectors are loaded once, not per request. Two backends:

- `reference` (`src.reference.TfidfReference`, the default): no model download, works with
  the processed corpus alone.
- `dense` (`src.retrieval.DenseRetriever`): needs `API_BUNDLE`.

Both are the `Retriever` protocol, so `recommend()` does not care which one is serving.

### 4. The request log is SQLite in WAL mode

`RequestLogRecord` comes from `src/contracts.py` (shared file, untouched). The log is
written to SQLite with `journal_mode=WAL` and a 30 s busy timeout: several uvicorn workers
(or several threads in one worker) can write at the same time without losing a row or
interleaving a half-written line. A failing log never breaks a prediction (the write is
wrapped in `try/except sqlite3.Error`).

### 5. E06 hurts the proxy metrics, and that is expected

The synthetic label is the **origin recipe** of the query, and the query was built by
*hiding* some of the origin recipe's ingredients. The origin recipe is therefore exactly
the candidate with the **most missing ingredients**, so ranking by "fewest missing" pushes
the labelled recipe down. The ablation measures that cost (R@10 0.576 -> 0.380 on main) and
the rerank stays **off by default**. The final ranking should be judged with manual labels
(Person 1), not with the origin-recipe proxy.

---

## C. Results (main corpus, validation split, 500 queries)

| Experiment | Rerank | Recall@5 | Recall@10 | MRR@10 | Encode s | Cold load s | Peak RSS MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E04 (ingredients) | none | 0.5120 | 0.5760 | 0.4332 | 238.939 | 14.454 | 717.4 |
| E05 (title + ingredients) | none | 0.2140 | 0.2620 | 0.1519 | 333.034 | 6.051 | 827.5 |
| E04 | missing_first | 0.2920 | 0.3800 | 0.2287 | - | - | - |
| E05 | missing_first | 0.1920 | 0.2400 | 0.1437 | - | - | - |

Bundle size: 73.24 MB `embeddings.npy` + 1.07 MB `recipe_ids.json` per mode (74.3 MB).
Query latency on the main bundle: 123 ms (top_k 10, CPU, warm weights).

Full table, the hardware and the reasoning: `reports/e04_e05_main_benchmark.md`.

---

## D. Execution log (real commands and outputs)

- `python -m src.embeddings --config configs/default.yaml --mode ingredients`
  -> `models/minilm_pretrained-E04-main-1f1f4ce5`, 50 000 docs, dim 384, truncated 1, 238.9 s.
- `python -m src.embeddings --config configs/default.yaml --mode title_ingredients`
  -> `models/minilm_pretrained-E05-main-1f1f4ce5`, 50 000 docs, dim 384, truncated 2, 333.0 s.
- `python -m src.evaluation --mode both --config configs/default.yaml --split validation --ablation --out reports/e04_e05_main.md`
  -> the table above, 6 MLflow runs (E04, E05, E06 x 2).
- `python -m pytest tests/test_predict.py tests/test_api.py` -> 15 + 17 tests pass.
- `docker build -t food-recipe-api:cpu .` -> image `3.19 GB` on disk / `751 MB` compressed.
- `docker compose up -d` -> `api Up (healthy)`; `/health` returns `{"status": "ok", ...}`.
- Dense backend exercised inside the container on the 50 000-recipe main bundle:
  `/predict ["chicken","rice"]` returns 200 with matched/missing ingredients, offline.
