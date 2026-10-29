# Hưng — Sprint 3: giao diện web, benchmark, chọn bundle qua config

Owner: Hưng (Person 2). Task sheet: `Hung_Sprint3.md` (deadline 29/10 24:00).
Branch: `feat/UI-web`.

Sprint 2 (`docs/HUNG_SPRINT2_PLAN.md`) delivered the main corpus index, `recommend()`, the
FastAPI service and the Docker image. Sprint 3 adds the **web UI**, a **benchmark** of every
serving bundle, the **config-driven model choice** and the **native vs Docker parity** check.

---

## A. Requirement traceability

| ID | Requirement | Status | Files | Acceptance evidence |
| --- | --- | --- | --- | --- |
| a.1 | Static UI `fe/index.html` (HTML/CSS/JS, no build step) | done | `fe/index.html` | served by FastAPI `StaticFiles` at `/` |
| a.2 | Form: ingredients + top-k | done | `fe/index.html` | `top_k` clamped to [1, 20], the API validates too |
| a.3 | Result list: title, matched/missing, expandable directions, source link | done | `fe/index.html` | `<details>` for the directions, the link is rendered only for http/https |
| a.4 | Recipe content escaped (no `innerHTML`) | done | `fe/index.html` | every text node is written with `textContent` |
| a.5 | loading / empty / error / retry states | done | `fe/index.html` | `aria-live` status, the retry button replays the last payload |
| a.6 | Usable with the keyboard only | done | `fe/index.html` | labels, visible focus, no mouse-only control |
| a.7 | Checked at 390 px and 1440 px, with pictures | done | `reports/figures/ui_390.png`, `ui_1440.png` | taken with headless Chrome on the real API |
| b.1 | `src/benchmark.py`: p50/p95 latency | done | `src/benchmark.py` | 20 warmup + 100 measured queries, one after the other |
| b.2 | RAM (peak RSS), startup time, bundle size | done | `src/benchmark.py` | `psutil` peak RSS, `time.perf_counter`, folder size |
| b.3 | Every model: TF-IDF, BM25, LSA, MiniLM pretrained (E04/E05), fine-tuned | done | `reports/benchmark.json` | 5 bundles of the main corpus measured |
| b.4 | The real CPU and RAM of the machine in the report | done | `reports/benchmark.json` | `machine` block: CPU, cores, RAM, Python |
| b.5 | Re-runnable with one command | done | `src/benchmark.py` | `python -m src.benchmark --config configs/default.yaml` |
| c.1 | The model is chosen by the config (`serving.bundle`) | done | `configs/default.yaml`, `api/app.py` | `API_BUNDLE` overrides the config |
| c.2 | All bundle types load: tfidf, bm25, lsa, minilm pretrained, fine-tuned | done | `src/serving.py` | `model.joblib` -> lexical, `embeddings.npy` -> dense |
| c.3 | `/model-info`: version, corpus hash, metrics (manifest or MLflow) | done | `api/app.py`, `src/serving.py` | the metrics come from the MLflow run of the bundle |
| c.4 | A bundle with a wrong hash does not start | done | `src/serving.py`, `tests/test_serving.py` | the manifest hashes are checked at startup |
| d.1 | Blind pool for the dev annotation | done | `scripts/build_pool.py` | `reports/manual_eval/pool_dev.csv` (278 rows, 20 queries, 4 systems) |
| d.2 | The dev labels of Person 2 | to do | `reports/manual_eval/pool_dev.csv` | the annotator fills `relevance` and `annotator` |
| d.3 | Parity native vs Docker (same top-k, scores within a tolerance) | done | `scripts/check_parity.py`, `reports/parity.json` | 5/5 queries, max score difference 0.0 |

---

## B. Design decisions

### Every bundle is measured in its own process

The peak RSS is the high-water mark of a process, so measuring several bundles in one
process reports the sum of them. `src/benchmark.py` therefore runs the measurement of each
bundle in a child process (`--child`) and reads the peak RSS of that child.

### The bundle decides how it is loaded

The loader does not need a config flag per model type: a bundle with `model.joblib` is a
lexical one (TF-IDF, BM25, LSA, written by `src.baseline`), a bundle with `embeddings.npy`
is a dense one (MiniLM, written by `src.embeddings`). `src/serving.py` looks at the files
of the folder and loads the right retriever, then checks every hash of the manifest.

### The requirements are split for the Docker image

`requirements.txt` cannot be resolved: `datasets>=5.1` (the training stack, only used in
the notebooks) needs `huggingface-hub>=1.31` while `transformers` needs `<1.0`.
`requirements-serve.txt` has the same list without the training extras and is what the
Docker image installs. `requirements.txt` stays the full list for the notebooks.

### The parity is checked on BM25

BM25 is exact and deterministic, so a difference of 0.0 is a real proof that the container
serves the same thing as the host. The dense bundles would only differ by the float noise
of the BLAS libraries.

---

## C. Results

### Benchmark (main corpus, validation split, 500 queries)

| Model | Bundle | p50 (ms) | p95 (ms) | Startup (s) | Peak RSS (MB) | Size (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tfidf | `tfidf-main-1f1f4ce5` | 13.61 | 17.90 | 0.09 | 546.6 | 8.44 |
| bm25 | `bm25-main-1f1f4ce5` | 43.39 | 69.84 | 0.68 | 558.1 | 5.94 |
| lsa | `lsa-main-1f1f4ce5` | 80.14 | 116.79 | 0.40 | 662.7 | 124.47 |
| minilm_pretrained (E04) | `minilm_pretrained-E04-main-1f1f4ce5` | 63.74 | 75.28 | 20.77 | 732.5 | 74.32 |
| minilm_pretrained (E05) | `minilm_pretrained-E05-main-1f1f4ce5` | 63.91 | 83.46 | 15.70 | 750.6 | 74.32 |

Machine: Intel Core Ultra 7 155H (8 physical cores), 7.71 GB RAM, Windows 11, Python 3.12.10.
The numbers above are one run; the table in `reports/benchmark.md` is written by the run
itself, so it always shows the numbers of the machine that ran it.

Reading: TF-IDF is the fastest but the least accurate (R@10 0.826); BM25 is the most
accurate (R@10 0.910) but the slowest per query; MiniLM E04 is both fast after the
26 s startup and accurate (R@10 0.576 on the synthetic labels); LSA costs 124 MB for a
worse score than BM25.

### Parity native vs Docker (BM25, top_k = 5)

| Query | Same ids | Max score difference |
| --- | --- | --- |
| chicken, rice, garlic | yes | 0.0 |
| pasta, tomato, basil | yes | 0.0 |
| egg, milk, flour | yes | 0.0 |
| beef, onion, carrot | yes | 0.0 |
| rice, soy sauce | yes | 0.0 |

The log is `reports/parity.json`.

---

## D. Still open

- The dev labels of Person 2 (`reports/manual_eval/pool_dev.csv`): the pool is built and
  ready, the `relevance` and `annotator` columns are still empty.
- The fine-tuned bundle was not measured: `minilm_finetuned` has no bundle on the main
  corpus yet (Person 3 owns the fine-tuning run). The benchmark picks it up as soon as the
  bundle exists in `models/`.
