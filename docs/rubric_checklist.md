# Rubric evidence checklist

Source: `D:/MDS301/Assessments/Project.docx`, sections 4-6. Checked locally on
2026-11-03. Owners: Hoàng (data/evaluation/report), Hưng (API/UI/Docker), Hiếu
(training/DL/MLflow/monitoring). "Implemented" is not a clean-clone or remote-CI
verification. Missing evidence remains assigned below rather than assumed done.

| Requirement | Evidence | Status and remaining work | Owner |
|---|---|---|---|
| Phase 1 (10): problem and user benefit | reports/proposal.pdf; report_part1.md sections 1-2 | Documented | Hoàng |
| P1.1 ML formulation and quantitative success criteria | reports/proposal.html; report_part1.md | Retrieval/NLP formulation and proxy limitations documented | Hoàng |
| P1.1 dataset URL, size, quality and licensing | README Dataset; reports/data_audit.md; report_part1.md | Documented; no redistribution licence assumed | Hoàng |
| P1.1 weekly project plan and responsibilities | reports/proposal.html; CONTRIBUTING.md | Documented; instructor registration/approval evidence is not in repo | Hoàng |
| Dataset suitability bounds and original domain registration | 50k manifest; engineered NLP features | NLP dimensionality interpretation and portal registration need instructor confirmation | Hoàng |
| P1.2 raw immutable, scripted collection, large-data download | src/data_collection.py; scripts/fetch_artifacts.py; .gitignore | Code exists; artifacts.url is empty, so clean-clone download is not fully operational | Hoàng |
| Phase 2a (15): sklearn Pipeline/ColumnTransformer | src/pipeline.py; tests/test_features_pipeline.py | Implemented and tested | Hoàng |
| P2.1 missing handling with justification | src/features.py; report_part1.md | Missing source handled as unknown; text empties normalized, documented | Hoàng |
| P2.1 scaling and categorical encoding | StandardScaler/OneHotEncoder in src/pipeline.py | Implemented and tested; retrieval uses sparse text representations | Hoàng |
| P2.1 two original engineered features | ingredient_count, mean_ingredient_rarity in src/features.py | Implemented; train-only fitting tested | Hoàng |
| Phase 2b (20): at least three appropriate models | TF-IDF, BM25, LSA, pretrained MiniLM, FT in comparison.csv | Implemented; main FT is E08, not E07 | Hoàng/Hiếu |
| P2.2 identical seeded outer 80/20 split | data/processed/main/splits.json; dataset_manifest.json | Same grouped outer split, seed 42; validation held out within outer train | Hoàng |
| P2.2 CV >=5 and tuning | reports/tfidf_cv5.json; lsa_cv5.json; src/tune.py | GroupKFold(5); E02 selected from 24 configurations | Hoàng |
| P2.2 train/test metrics and measured duration | reports/comparison.csv; reports/final/comparison_test.csv; master_comparison.csv | E10 outputs and full run IDs provide final test evidence; no retuning after test | Hoàng |
| P2.2 bias-variance analysis and deployment selection | reports/report_part2_models.md; final/frozen_config.json | BM25 selected on validation before test; single-seed limits explicit | Hoàng |
| Phase 2c (15): appropriate DL model and architecture | src/finetune.py; finetune_cpu1ep_provenance.json; new part 2 report | MiniLM/MNRL and one measured CPU checkpoint | Hiếu |
| P2.3 genuine train/validation loss curves | reports/ablation_regularization.md; docs/reviews/hieu-khung4-pr20.md | MISSING: existing generated ablation curves are constants, not measured histories | Hiếu |
| P2.3 controlled regularization effect | notebooks/03_finetune_main_ablation.ipynb | MISSING: run an actual controlled comparison; configured weight decay alone proves no effect | Hiếu |
| P2.4 importable src and >=5 unit tests | src/; tests/; pyproject.toml | 281 tests passed before E10; 97.09% coverage for baseline/features/evaluate | All |
| Phase 3a (10): serialization and four REST endpoints | src/baseline.py; api/app.py; tests/test_api.py | Implemented and tested; selected BM25 uses joblib | Hưng |
| P3.1 descriptive malformed-input errors | contracts.py; tests/test_api.py | 422 cases tested | Hưng |
| P3.1 persistent per-prediction logging | api/request_log.py; src/monitor.py; monitoring_khung3_ft.json | SQLite migration and missing counts tested; real HTTP smoke evidence exists | Hưng/Hiếu |
| P3.1 startup routine and API examples | README API usage; docs/demo_runbook.md | Present; recheck selected BM25 from clean clone | Hưng |
| Phase 3b (10): Dockerfile/Compose/base image/port | Dockerfile; docker-compose.yml | Present; this branch has not rebuilt the image | Hưng |
| P3.2 same endpoint tests and health in container | reports/parity.json; tests/test_smoke_e2e.py | Earlier E04 parity exists; MISSING selected-BM25 clean-clone/container evidence | Hưng |
| P3.2 README build/run/health instructions | README Docker | Present; clean-clone execution remains required | Hưng |
| Phase 3c (5): MLflow run params/train/test/durations | comparison.csv; final/e10_execution.json; final/comparison_test.csv | Historical train/val IDs retained and E10 IDs logged; unknown times are not invented | Hoàng/Hiếu |
| P3.3 artifacts and Staging/Production versioning | models/; mlflow_rollback_trial.json | Local FT trial revisions share one locked checkpoint; no claim of different model quality | Hiếu |
| P3.3 genuine MLflow UI screenshot | docs/reviews/hieu-khung4-pr20.md | MISSING: replace generated UI facsimiles with actual UI captures | Hiếu |
| Monitoring strategy | src/monitor.py; drift_simulation.md; monitoring_khung3_ft.json | Simulations labelled; real smoke workload separated from user traffic; no relevance-decay claim | Hiếu |
| Phase 4a (5): required directory structure | README tree; src/api/tests/configs/reports/models/data/notebooks | Present, large artifacts ignored | All |
| P4.2 >=20 meaningful conventional commits | git history (93 commits before this branch) | Count met; each reviewer must assess meaningfulness | All |
| P4.2 branches/PR/reviewer and runnable main | GitHub PRs #19/#20/#22; current PR | dev workflow present; reviewer approvals and runnable main need remote confirmation | All |
| Phase 4b (5): title/problem/domain | README opening | Present | Hoàng |
| P4.3 team names and student IDs | README Team | MISSING: TODO rows remain; actual student IDs must be supplied | Hoàng |
| P4.3 installation and dataset download | README Installation/Dataset | Commands present; complete artifact URL and clean-clone check still needed | Hưng/Hoàng |
| P4.3 preprocessing/features/architecture | README ML pipeline; reports/report_part2_models.md | Present | Hoàng/Hiếu |
| P4.3 Master Comparison Table | README Experiment results; reports/final/master_comparison.csv | Added with Train/Test/CV/time/tuning/notes and validation-selected star | Hoàng |
| P4.3 API/native/Docker/examples | README API/Docker | Present; final BM25 clone/container check pending | Hưng |
| P4.3 MLflow/tree/reproducibility/seed | README MLflow/Repository structure/Reproducibility | Present; legacy processed-cache config mismatch needs deliberate verification | All |
| P4.4 formal PDF 15-25 pages plus executive summary | report_part1.pdf; report_part2_models.pdf; report_part2_dl_mlops.pdf | Components exist; final combined report, corrected DL evidence and references remain khung 5 work | Hoàng/Hiếu |
| P4.4 model development/Master Table/discussion/ethics | reports/report_part2_models.md/.pdf | At least six substantive pages, with data provenance and explicit limits | Hoàng |
| Phase 5 (5): 15-20 slides and all members speak | No final deck evidence found | MISSING final deck and speaking allocation | All |
| P5 live demo, 20min +10min Q&A | docs/demo_runbook.md | Runbook exists; MISSING timed group rehearsal and Q&A record | All |
| 5.2 reproducible one-command pipeline | src/train.py; tests/test_stage_regressions.py | Full fixture stages tested; MISSING complete pilot run/clean-clone record | Hiếu/Hưng |
| 5.2 pinned dependencies and CPU fallback | requirements.txt; README | Core pins exist; accelerate/datasets remain bounded rather than exact pins | Hiếu |
| 5.2 notebooks clean-kernel execution | notebooks/; tests/test_notebook_export.py | Static checks exist; MISSING complete clean-kernel execution logs | Each notebook owner |
| Final manual labels/20% overlap/kappa/nDCG/coverage | reports/manual_eval/final/; src/manual_evaluation.py | Awaiting independent human labels and adjudication; no made-up scores | All, coordinated by Hoàng |

The rubric awards 100 points across the ten scored components above. This checklist
is an evidence map, not an inferred grade or a claim that every component is complete.
