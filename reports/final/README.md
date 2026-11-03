# Hoàng khung 4 evidence

Branch: `final-evaluation-report`. This work covers Hoàng's final evaluation,
manual-study coordination, model-development report and rubric/PR review.
Hưng's Docker work and Hiếu's training/report corrections are outside this branch.

| Item | Status | Evidence |
|---|---|---|
| a Frozen E10 and Master Table | Complete | frozen_config.json, e10_execution.json, comparison_test.csv, master_comparison.csv/.md |
| b Blind final pool and scoring workflow | Prepared; independent labels/adjudication pending | ../manual_eval/final/, manual_eval_status.json, ../../docs/final_annotation_workflow.md |
| c Model-development report >=6 pages | Complete: 7 pages | ../report_part2_models.md and ../report_part2_models.pdf |
| d Hiếu PR review and rubric map | Complete | ../../docs/reviews/hieu-khung4-pr20.md and ../../docs/rubric_checklist.md |

## Final evaluation

- User confirmed all seven systems and BM25 selection before test.
- Main corpus: 50,000 recipes. E10: 1,000 test queries, top-k 10, rerank none.
- Parent run: `16e4dfb977434b2a87a06418c8f2e5a1` in local `sqlite:///mlflow.db`.
- One execution completed seven systems. The ledger refuses a second execution.
- BM25 test MRR@10: 0.7765793651; Recall@5: 0.8600; Recall@10: 0.9090.
- Every system has a child run, ranking JSON and per-query metrics. Train and
  validation numbers refer to the existing historical runs. CV values and full
  run IDs are in the master CSV. Table generation verifies the copied values
  against MLflow, including CV mean/std.
- `significance_test.json` preserves ranx's raw NaN p-values for identical paired
  scores. Use `significance_test_export.json` for strict JSON; undefined values
  become null and are never interpreted as significant differences.

Do not rerun E10 to regenerate a report. Read the saved outputs. Reporting commands:

```bash
python -m scripts.build_final_tables
# Use the artifact Python with reportlab for this report builder:
python scripts/build_model_report.py
```

The bundle/data/encoder paths in the freeze record identify this workstation.
Models, raw data and the local tracking DB remain external artifacts, not Git content.

## Final human study

The pool contains 388 unique query-recipe pairs across 20 final queries. There are
473 assignments and 85 double-labelled items (21.9% overall, at least 20% per query).
Hoàng has 160 assignments, Hưng 177 and Hiếu 136. All labels and attestations are
blank; the task does not fabricate human judgments or kappa/nDCG scores.

Judges read only their assigned CSV and the guide. Do not open `pool_keys/` or the
other annotators' labels before completing independent judgments. After all three
files and provenance declarations are complete:

```bash
python -m src.manual_evaluation
# Resolve any generated adjudication.csv, preserving original labels, then run again.
```

## Validation

- 283 tests passed; baseline/features/evaluate coverage 97.09%.
- Ruff checks passed.
- The seven-page PDF was rendered with Poppler and every page visually checked.
- Frozen input/model/code hashes remained unchanged after E10.
- The retrospective review was submitted to PR #20 as COMMENTED; it does not
  approve unsupported MLflow images or the unmeasured regularization ablation.
