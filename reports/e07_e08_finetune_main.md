# Fine-tuned MiniLM: main corpus CPU one-epoch evaluation

Measured on 2026-10-29. This report replaces the earlier E07/E08 table: its FT
values and registry claims were not tied to this checkpoint. No E08 (fine-tuned
title + ingredients) bundle was evaluated in this run.

## Setup and provenance

- Corpus: 50,000 recipes; 500 validation queries; origin-recipe synthetic qrels.
- Checkpoint: `models/minilm_finetuned-main-cpu1ep`.
- Serving bundle: `models/minilm_finetuned-E04-main-1f1f4ce5` (ingredients only).
- Both corpus and query embeddings use the fine-tuned checkpoint; 384 dimensions,
  normalized vectors, maximum sequence length 128. One corpus document is truncated.
- Existing training: 30,000 train pairs, seed 42 for pair sampling, one epoch,
  batch 32, learning rate 2e-5, configured weight decay 0.01, warmup 50 steps.
- Historical log: 871.336 seconds, peak RSS
  2787.1 MB, mean training loss
  0.079213; corpus encoding 97.079 seconds.
- Training was not repeated. Its existing log was imported into MLflow with an
  explicit historical-log tag. The generated model card lists default optimizer
  settings; the settings above come from the training invocation/log and source.
- All 30,000 pair recipe IDs were verified against train; their groups do not
  overlap validation/test groups. Checkpoint and input hashes are recorded in
  [finetune_cpu1ep_provenance.json](finetune_cpu1ep_provenance.json).

## Validation results

| System | Recall@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| E01 TF-IDF | 0.7760 | 0.8260 | 0.6567 |
| E02 tuned TF-IDF | 0.7760 | 0.8260 | 0.6567 |
| E03 BM25 | 0.8560 | 0.9100 | 0.7583 |
| LSA | 0.4560 | 0.5360 | 0.3629 |
| E04 pretrained ingredients | 0.5120 | 0.5760 | 0.4342 |
| E05 pretrained title + ingredients | 0.2140 | 0.2620 | 0.1519 |
| FT CPU one epoch (E07) | 0.7340 | 0.7980 | 0.6209 |

FT improves MRR@10 over E04 by **+0.1867 (+43.0%)** and Recall@10 by
**+0.2220**. It improves reciprocal rank on
196 queries, regresses on 30,
and ties on 274. BM25 and TF-IDF still score higher on this proxy.

These numbers are backed by the validation error-analysis MLflow run below.
The full ranx comparison, per-query scores and train-validation gap are recorded
in [comparison.csv](comparison.csv), [comparison_per_query.csv](comparison_per_query.csv)
and [comparison_ranx.md](comparison_ranx.md).

## Train-validation gap and significance

| System | Train proxy MRR@10 (5,000 queries) | Validation MRR@10 (500 queries) | Train - validation |
| --- | ---: | ---: | ---: |
| E04 pretrained | 0.4575 | 0.4342 | +0.0233 |
| FT CPU one epoch | 0.6450 | 0.6209 | +0.0241 |

The train scores are retrieval proxies over the full candidate corpus, not training
losses. The observed gap alone does not establish absence of overfitting.
`ranx.compare` finds FT better than E04, E05 and LSA on MRR@10 at p < 0.01
(paired Student t-test). BM25 is significantly better than FT. The lower FT score
relative to TF-IDF is not significant at that threshold. These tests compare queries
within one training run; they do not measure variation across training seeds.
Comparison timing columns are descriptive, not a dedicated inference benchmark.

## Error analysis and manual judging

- [error_analysis.md](error_analysis.md) contains pattern tables and 28 deterministic
  validation cases, with per-query ranks in [error_analysis_queries.csv](error_analysis_queries.csv).
- FT remains weakest on short queries and large origin recipes. The tables show
  where the pretrained-to-FT improvement remains below BM25.
- For 77 queries, FT's top recipe contains every normalized
  query ingredient but is not the synthetic origin. These are candidates for
  manual review, not established relevance judgments.
- [manual_eval/README.md](manual_eval/README.md) describes the 396-item dev pool
  and 484 assignments, including 88 independent second labels. All labels are blank.

## Limits

- Test metrics and the final manual pool remain untouched. Human agreement,
  pooled nDCG@10 and judging coverage require completed manual labels.
- Synthetic MRR/Recall measure recovery of origin recipes, not all useful recipes.
- There is only one training run/seed. Pair sampling uses full-corpus ingredient
  document frequencies; positive recipes and groups are train-only. The existing
  training loader does not use `NoDuplicateBatchSampler`, so same-group in-batch
  false negatives remain possible. These protocol limits need consideration before
  a future training experiment.

## MLflow evidence (`sqlite:///mlflow.db`)

- Imported historical training log: `0437841237734100a7c1a9b26725d908`.
- Validation error analysis: `d069e2ae76454826ae848f148eade5f7`.
- Dev pool and assignments: `1c83d1d32d814732a9f0907ebdb1f50c`.
- The full comparison CSV includes a separate run ID for every system and split.

- FT train evaluation: `90c9dfb87aff42668483433d8216ce2f`.
- FT validation evaluation: `0444aa538a9948f8aecc25d5c42bb365`.
