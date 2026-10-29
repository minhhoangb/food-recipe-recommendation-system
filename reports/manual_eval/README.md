# Main corpus dev judging pool

Generated on 2026-10-29 from the top 5 results of E01, E02, E03, LSA, E04,
E05 and the CPU one-epoch FT checkpoint. The 20 dev queries produce 396 unique
query-recipe pairs. Every query's items are shuffled with seed 42 before assignment.

Use the existing [annotation guide](annotation_guide.md) without changing its scale.
Fill only `relevance` and `notes` in your assigned file; `annotator` is already set.
All relevance labels are currently empty.

| Annotator | File | Rows |
| --- | --- | ---: |
| Hoang | [pool_dev_hoang.csv](pool_dev_hoang.csv) | 172 |
| Hung | [pool_dev_hung.csv](pool_dev_hung.csv) | 161 |
| Hieu | [pool_dev_hieu.csv](pool_dev_hieu.csv) | 151 |

There are 484 assignments: 396 first labels and 88 independent second labels
(at least 20% of each query's items). Keep both original labels when resolving
disagreements, as required by the annotation guide.

`pool_dev.csv` is the unassigned master copy. Annotators must not open
`pool_dev_key.csv` or `pool_dev_systems.jsonl` during judging: those files reveal
system names and ranks. Both keys refer to the same item IDs as the assigned files.
`pool_dev_metadata.json` records the checkpoint hash, corpus hash, query hash and
MLflow run ID for this pool.

The final queries have not been pooled or judged. Human agreement, pooled nDCG@10
and top-10 judging coverage remain pending until manual labels are complete.
