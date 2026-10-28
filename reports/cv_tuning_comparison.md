# Khung 2 (a, b): TF-IDF CV tuning and validation comparison on MAIN

All numbers come from MLflow runs in `sqlite:///mlflow.db` (experiment `recipe-retrieval`) or from
the files next to this one. Corpus `main`, sha256 `1f1f4ce5...` (50 000 recipes), seed 42.
Labels are synthetic origin-recipe qrels, so every metric is an *origin-recipe retrieval proxy*.

## (a) E02: GridSearchCV + GroupKFold(5) on outer train

```bash
python -m src.tune --config configs/default.yaml               # E02, MLflow dac82cb580e8403082cd9d0fabb40448
python -m src.tune --config configs/default.yaml --model lsa   # LSA CV, MLflow d46efe09f7dd4835841f66b860d4d433
```

- Rows: 35 896 outer-train recipes, groups = `group_id` (28 705 groups); a group is never split.
- Each fold refits the vectorizer on four folds; all 35 896 train recipes stay candidates, so the
  held-out origin recipes can be retrieved. Queries: the 5 000 synthetic `train` queries, scored
  in the fold that holds their origin recipe. Scorer: MRR@10 (ranx).
- Fold isolation is tested in `tests/test_tune.py` (fold vocabulary has no held-out-only token).
- Grid (24 points, 939 s on CPU), reason per parameter in `src/tune.py::PARAM_GRID`:
  `tokens` {whole ingredient names, + words inside names} (stands in for `ngram_range`, which
  TfidfVectorizer ignores with a callable analyzer), `min_df` {1, 2, 5}, `sublinear_tf`, `norm` {l2, None}.

| Grid point | CV MRR@10 (mean ± std) | Train-fold MRR@10 |
| --- | --- | --- |
| **best = sklearn default (E01 params)**: ingredient, min_df 1, l2 | **0.6896 ± 0.0063** | 0.7280 |
| ingredient, min_df 1, norm None | 0.6765 ± 0.0073 | 0.7364 |
| ingredient, min_df 2, l2 | 0.6561 ± 0.0103 | 0.6762 |
| ingredient_and_words, min_df 1, l2 | 0.6376 ± 0.0076 | 0.6456 |

Full grid: `tfidf_cv5.csv`. Findings: the default wins; splitting names into words hurts (a query
"chicken" then matches every "chicken *" recipe), raising `min_df` drops rare ingredients that pin
a recipe down, `sublinear_tf` changes nothing with whole-name tokens (no token repeats in a recipe).
The best values are stored in `configs/default.yaml` (`baselines.tfidf_tuned`), so **E02 = E01**;
the E02 bundle is built through the same path (`python -m src.baseline --model tfidf --tuned`).

## (b) Validation comparison (500 queries) with ranx.compare

```bash
python -m src.compare --config configs/default.yaml \
  --system E01=models/tfidf-main-1f1f4ce5 --system E02=models/tfidf_tuned-main-1f1f4ce5 \
  --system E03=models/bm25-main-1f1f4ce5 --system LSA=models/lsa-main-1f1f4ce5 \
  --system E04=models/minilm_pretrained-E04-main-1f1f4ce5 \
  --system E05=models/minilm_pretrained-E05-main-1f1f4ce5
```

| Exp | Model | Tuned? | Train proxy MRR@10 (5 000 q) | CV MRR@10 | Val R@5 | Val R@10 | Val MRR@10 | Fit / index s | Latency ms/q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E01 | TF-IDF | no | 0.6998 | 0.6896 ± 0.0063 | 0.776 | 0.826 | 0.6567 | 28.0 | 16.4 |
| E02 | TF-IDF | yes (CV5) | 0.6998 | 0.6896 ± 0.0063 | 0.776 | 0.826 | 0.6567 | 34.0 | 15.5 |
| E03 | BM25 | no | 0.7687 | N/A (1) | **0.856** | **0.910** | **0.7583** | 25.7 (index) | 57.7 |
| -   | LSA (256) | no | 0.3402 | 0.3560 ± 0.0095 | 0.456 | 0.536 | 0.3629 | 51.6 | 44.5 |
| E04 | MiniLM, ingredients | no | 0.4575 | N/A (2) | 0.512 | 0.576 | 0.4342 | 172.1 (encode) | 30.3 |
| E05 | MiniLM, title + ingr. | no | 0.1808 | N/A (2) | 0.214 | 0.262 | 0.1519 | 445.9 (encode) | 32.2 |

(1) rank-bm25 computes IDF over every candidate it indexes (transductive), nothing is fitted on
train to cross-validate; k1 = 1.5, b = 0.75 defaults. (2) Pretrained encoder, nothing is fitted on
our data. Fit time: TF-IDF/LSA = fit on train + transform 50 000 recipes; BM25 = index build;
E04/E05 = corpus encoding (no training). E05 encode time was measured while other jobs shared the
CPU (Hưng measured 333 s alone). Latency: mean over the 500 validation queries, one CPU, warm.

Paired Student t-test (ranx, p < 0.01) on MRR@10: **E03 beats every other system**; E01/E02 beat
LSA, E04 and E05; E04 beats LSA and E05; LSA beats E05. Full report: `comparison_ranx.md`.

Files: `comparison.csv` (one row per system, with MLflow run ids for the train and validation runs),
`comparison_per_query.csv` (500 queries x 6 systems), `comparison_ranx.md`, `tfidf_cv5.*`, `lsa_cv5.*`.

Notes:
- E04/E05 bundles were rebuilt here with Hưng's command; R@5/R@10 equal his report, E04 MRR@10 is
  0.4342 vs 0.4332 (encoder numerics differ slightly between machines).
- Train proxy (full model on all 5 000 train queries over 50 000 candidates) is not the same number
  as the CV train-fold score (4/5 of train, 35 896 candidates).
