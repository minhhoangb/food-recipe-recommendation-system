# Main corpus dev judging pool

Generated on 2026-10-29 from the top 5 results of E01, E02, E03, LSA, E04,
E05 and the CPU one-epoch FT checkpoint. The 20 dev queries produce 396 unique
query-recipe pairs. Every query's items are shuffled with seed 42 before assignment.

Use the existing [annotation guide](annotation_guide.md) without changing its scale.
Fill only `relevance` and `notes` in your assigned file; `annotator` is already set.
The three human assignment files still have empty relevance labels.

**Before independent labels are locked, do not open `pool_dev_key.csv`,
`pool_dev_systems.jsonl`, or any files under `reports/ai_annotations/` (including
AI labels, summaries and examples).** They reveal system information or suggested
answers and can influence your judgments. Read only your assignment and the guide.
If you have already seen any of them, record that exposure; those judgments must
not be reported as blind independent labels. Moving the files cannot undo exposure.

| Annotator | File | Rows |
| --- | --- | ---: |
| Hoang | [pool_dev_hoang.csv](pool_dev_hoang.csv) | 172 |
| Hung | [pool_dev_hung.csv](pool_dev_hung.csv) | 161 |
| Hieu | [pool_dev_hieu.csv](pool_dev_hieu.csv) | 151 |

There are 484 assignments: 396 first labels and 88 independent second labels
(at least 20% of each query's items). Keep both original labels when resolving
disagreements, as required by the annotation guide.

`pool_dev.csv` is the unassigned master copy. Both system keys refer to the same
item IDs as the assigned files.
`pool_dev_metadata.json` records the checkpoint hash, corpus hash, query hash and
MLflow run ID for this pool.

The final queries have not been pooled or judged. Human agreement, pooled nDCG@10
and top-10 judging coverage remain pending until manual labels are complete.

## Rebuilding pools

`python -m src.evaluate pool` is the single pool implementation for both splits.
Supply each frozen bundle as `--system LABEL=path` and the matching FT checkpoint
as `--encoder LABEL=path`. For example, from the repository root:

```bash
python -m src.evaluate pool --config configs/default.yaml --split dev --depth 5 --system E04=models/minilm_pretrained-E04-main-1f1f4ce5 --system FT=models/minilm_finetuned-E04-main-1f1f4ce5 --encoder FT=models/minilm_finetuned-main-cpu1ep --out reports/manual_eval/pool
```

This example selects two systems; use all agreed frozen systems for the actual
pool. Use `--split final` only after the configuration is frozen. The command
refuses to overwrite assignment CSVs containing relevance labels or notes.
