# Working agreement

The task list per person is in the allocation sheet shared by the team lead (not part of this repo). This file only covers *how* we work in the repo.

## What already exists (do not rebuild it)

| Need | Use |
| --- | --- |
| Clean ingredient names (query and recipe) | `src.normalize.normalize_ingredients` |
| Record / API / manifest types | `src.contracts` |
| Paths, seed, caps | `src.config.load_config("configs/pilot.yaml")` |
| Real data: corpus, splits, queries, qrels | `data/processed/{pilot,main}/` (see README) |
| Ranking metrics | `ranx`: `evaluate(Qrels(...), Run(...), ["recall@10", "mrr@10", "ndcg@10"], make_comparable=True)`; conventions locked in `tests/test_ranking_metrics.py` |
| Group-aware CV / splits | `sklearn.model_selection.GroupKFold` with `group_id` from the corpus |
| Experiment tracking | `mlflow` directly (`sqlite:///mlflow.db`); no wrapper |
| A working retriever for UI/API/monitor work | `src.reference.TfidfReference` (scikit-learn TF-IDF cosine) |
| "Is my retriever valid?" | `tests/contract_utils.assert_retriever_contract` |

**Working rule: use a library or existing code before writing new code.** Check the table above, then the installed packages, then PyPI; write code only for project-specific logic, and give it a test.

Data is built once by the data owner. Everyone else only reads it. Never regenerate it on your own machine with different settings: ids, splits and hashes must be the same for the whole team (`python scripts/verify_data.py` checks it).

## File ownership

| Owner | Files |
| --- | --- |
| Person 1 | `src/features.py`, `src/pipeline.py`, `src/baseline.py`, `src/evaluate.py`, EDA notebooks, comparison and error analysis, manual labels |
| Person 2 | `src/embeddings.py`, `src/retrieval.py`, `src/predict.py`, `src/benchmark.py`, `api/`, `fe/`, `Dockerfile`, `docker-compose.yml`, README run/API/Docker sections |
| Person 3 | `src/build_pairs.py`, `src/finetune.py`, `src/train.py`, MLflow registry/aliases, `src/monitor.py`, Colab fine-tune notebook |
| Shared (needs both other reviews) | `src/contracts.py`, `src/config.py`, `src/normalize.py`, `configs/*`, `src/data_collection.py`, `src/queries.py`, `src/preprocess.py` |

A shared file may be edited by anyone through a PR that the two other members approve. If a change alters data output (normalization, sampling, splits), the data owner rebuilds and republishes the artifacts, and the PR says so.

## Git

- One feature branch per task, named after the feature (`feature/<topic>`, e.g. `feature/baselines-evaluation`), merged into `main` by pull request with at least one reviewer who is not the author. `main` must always run (`python -m pytest` green).
- Commit messages: `<type>: <description>` with type in `feat fix docs test refactor chore perf ci data exp`. Enable the check once per clone:

```bash
git config core.hooksPath .githooks
```

- At least 20 meaningful commits in total are required by the course, so commit small and often.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest
python scripts/verify_data.py
```

Python 3.11 or 3.12 is recommended for the model/serving work (torch, sentence-transformers wheels). The shared foundation itself is tested on 3.14.

## Rules that protect the results

- A model may only be *fit* on `train`. `validation` is for tuning. `test` is read once, after the configuration is frozen.
- Quote only numbers that come from an MLflow run. Mark simulated or fixture results as such.
- Fixtures are for tests. Never show fixture output as a result.
- Do not commit raw CSV, embeddings, model weights, or large indexes.
