# Review of Hiếu's khung 4 PR

Reviewed 2026-11-03: [PR #20](https://github.com/minhhoangb/food-recipe-recommendation-system/pull/20),
head `aa0945d618eaa65e06a855485e81d8182a81df7d`, merged into dev on 2026-11-01.
[PR #22](https://github.com/minhhoangb/food-recipe-recommendation-system/pull/22)
adds dev labels; it is not a replacement implementation of the reporting work.
This is a retrospective review of the merged changes, not an approval.
The review was [submitted on GitHub](https://github.com/minhhoangb/food-recipe-recommendation-system/pull/20#pullrequestreview-5747898537)
as COMMENTED, with the findings below.

## Findings that still need correction by Hiếu

1. **[P1] Replace constructed MLflow images and fixed ablation curves with measured evidence.**
   `scripts/generate_mlflow_figures.py:47` embeds run names, IDs and metrics in a
   literal table, then draws a facsimile of the MLflow UI. Lines 219-223 define
   train/validation loss arrays directly. `reports/report_part2_dl_mlops.md:130`
   presents those values as a controlled weight-decay experiment. A picture drawn
   from constants does not establish that these runs occurred. The existing
   `reports/ablation_regularization.md` explicitly says that the controlled
   comparison has not been measured. Remove the experimental claims or replace
   them with actual run histories and genuine UI captures. Do not use these
   three generated images as screenshots or evidence in the final report.

2. **[P2] Describe the monitoring workload as scripted function calls, not user/API traffic.**
   `scripts/run_real_api_demo.py:125` loops over fixed inputs, calls recommendation
   code and manually writes `RequestLogRecord`, including its own validation
   branches. It does not exercise FastAPI routing/validation over HTTP.
   `reports/monitoring_real_report.md:16` describes inputs as naturally collected
   user requests and labels the environment production/ASGI. Correct the scope
   and cite the log artifact, or run the real API. The later evidence in
   `reports/offline_api_verification.json` does exercise a fresh uvicorn process;
   it is still a scripted smoke workload, not user traffic.

3. **[P2] Align experiment identifiers and model descriptions with the source artifacts.**
   `reports/finetune_analysis_and_manifest.md` calls main FT E07, describes E02 as
   sublinear TF and LSA as 100 dimensions. The team contract/main CPU provenance
   identifies main FT as E08, the selected TF-IDF settings have sublinear_tf=false,
   and `reports/lsa_cv5.json` records 256 components. Correct those descriptions
   and reference full MLflow IDs before merging the final combined report.

## Verified improvements already in dev

Commit `2c8e460` fixes packaged FT encoder selection, stage cache validation,
observed training metadata, missing-ingredient logging and genuine offline HTTP
verification. Its full suite passed 276 tests. The current final-evaluation work
passed 281 tests before freezing. These fixes do not validate the older generated
ablation/MLflow images and do not supply a missing controlled regularization run.

## Recommendation

Keep the locked, hash-verified CPU one-epoch FT checkpoint in E10. Use its measured
train/validation results and record the single-seed/no-controlled-ablation limits.
Hiếu owns replacement of the three misleading images and the regularization
evidence. Hoàng's new report must not repeat the unsupported claims.
