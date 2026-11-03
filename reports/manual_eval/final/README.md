# Final judging instructions

This is the final pool for 20 natural-language queries from the pre-existing final
query split. Follow `reports/manual_eval/annotation_guide.md`; its 0/1/2 scale is
unchanged. The deployment choice and model hashes were frozen before judging.

1. Hoàng reads only `reports/manual_eval/final/pool_final_hoang.csv`; Hưng reads only
   `pool_final_hung.csv`; Hiếu reads only `pool_final_hieu.csv`. Model names/ranks
   are absent. The files include independent second assignments covering at
   least 20% of each query's pool. Labels and notes are initially empty.
2. Each person fills `relevance` (0, 1 or 2) and optional `notes`. Preserve row IDs,
   order, recipe data and annotator. Judge from the actual query intent, recipe
   ingredients and directions. Mark uncertainty in notes; do not invent content.
3. Before all independent labels are complete, do not open other people's files,
   `reports/final/pool_keys/`, AI suggestions or per-item model comparisons. Do not
   use an AI label as an independent human label. Record any prior exposure;
   a later file move cannot restore independence.
4. Fill your row of `annotation_log.csv` with actual start/completion UTC times,
   `label_source=human`, `independent=yes`, `exposure=none` only if true. If exposed,
   document it and arrange a fresh independent judgment. Do not sign for another person.
5. Hoàng runs `python -m src.manual_evaluation` after collecting all assignments.
   Incomplete inputs produce a status report and no kappa/nDCG. Once complete,
   original label files are hash-locked. Never overwrite the original judgments.
6. The command writes `adjudication.csv` only for disagreements, after all original
   judgments are locked. Discuss these items together; fill its adjudicated
   `relevance`, actual adjudicator and reason. Keep both original labels untouched.
7. Run the same command again to compute pairwise Cohen's kappa on the original
   independent overlap, raw agreement, and pooled nDCG@10 on the adjudicated labels.
   Outputs are in `reports/final/manual_eval_status.json` and
   `reports/final/final_labels_adjudicated.csv`.

Kappa is unweighted for the three labels. If its denominator is zero, it is reported
as undefined rather than replaced with a flattering number. nDCG uses ranx's linear
gain. Top-10 items outside the depth-5 pool remain unjudged; the scorer assigns them
zero gain for this pooled estimate, which does not establish irrelevance. Each model
must be reported with its judging coverage, including per-query coverage.

The synthetic E10 test and this manual study answer different questions. Do not
select a new deployment model from final human or E10 scores. Any later change
requires a public protocol deviation and a new evaluation dataset.
