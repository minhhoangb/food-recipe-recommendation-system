# AI annotations for the dev judging pool

Generated on 2026-10-29 after a request to annotate the dev pool. These are a single AI pass, not three independent human judgments.

**Post-judging review only. Do not read this report, its CSVs or summaries before
independent human labels are locked.** Prior exposure must be recorded and prevents
claiming the affected judgments were blind and independent.

Inputs: `reports/manual_eval/pool_dev.csv` and the annotation guide as it stood
at annotation time. Their recorded hashes remain in `ai_label_metadata.json`.
The guide later gained a dated independence clarification; its label scale was
not changed. Moving these artifacts leaves the original AI decisions unchanged.

## Outputs

- `pool_dev_ai.csv`: 396 rows, the original pool columns, `annotator=ai`, relevance 0/1/2 and a reason per row.
- `ai_label_summary.csv`: label counts for each of the 20 dev queries.
- `ai_label_metadata.json`: source, hashes, method, limitations and MLflow run ID.

| Relevance | Items |
| --- | ---: |
| 0 | 116 |
| 1 | 208 |
| 2 | 72 |

## Review method

The fixed annotation guide was applied to recipe titles, raw ingredient lists and preparation excerpts. Full instructions were inspected for selected incomplete records. The system/rank key was not consulted when assigning labels. Decisions were recorded by item ID before exporting the CSV. All 20 dev queries were included; final queries were not annotated.

Staples were handled as defined in the guide. For five or more listed ingredients, the guide permits one missing key ingredient for label 2 when the extra-ingredient and intent conditions also hold. Different foods were not treated as synonyms: chicken broth is not chicken breast, black-eyed peas are not green peas, and gochugaru is not gochujang. Brand names and composite foods can be ambiguous; 20 rows carry an explicit `unsure` flag.

## Examples

| Item | Label | Reason |
| --- | ---: | --- |
| m14-12 | 2 | Tahini, lemon and garlic are used; chickpeas are one extra. |
| m16-20 | 1 | All three key ingredients occur, but lettuce, perilla, garlic and leek namul exceed two extras. |
| m12-08 | 0 | Baking violates the explicit no-bake request. |
| m17-07 | 0 | A one-line preparation fragment provides no usable cooking method. |
| m18-13 | 0 | The directions describe a pet-food supplement, not a human dish. |

## Per-query counts

| Query | Items | 0 | 1 | 2 | Unsure |
| --- | ---: | ---: | ---: | ---: | ---: |
| m01 | 21 | 3 | 17 | 1 | 3 |
| m02 | 16 | 1 | 10 | 5 | 2 |
| m03 | 21 | 4 | 13 | 4 | 0 |
| m04 | 19 | 2 | 11 | 6 | 2 |
| m05 | 16 | 1 | 13 | 2 | 0 |
| m06 | 15 | 2 | 8 | 5 | 4 |
| m07 | 26 | 6 | 20 | 0 | 0 |
| m08 | 15 | 3 | 12 | 0 | 1 |
| m09 | 26 | 7 | 19 | 0 | 0 |
| m10 | 19 | 8 | 11 | 0 | 0 |
| m11 | 21 | 8 | 13 | 0 | 0 |
| m12 | 24 | 10 | 13 | 1 | 2 |
| m13 | 19 | 0 | 12 | 7 | 0 |
| m14 | 20 | 8 | 6 | 6 | 0 |
| m15 | 18 | 0 | 2 | 16 | 4 |
| m16 | 21 | 20 | 1 | 0 | 0 |
| m17 | 20 | 5 | 5 | 10 | 0 |
| m18 | 18 | 6 | 3 | 9 | 2 |
| m19 | 20 | 12 | 8 | 0 | 0 |
| m20 | 21 | 10 | 11 | 0 | 0 |

## Interpretation and remaining work

Human verification remains false. No inter-annotator kappa, agreement percentage, human relevance score or model-quality improvement is claimed from this single pass. Ingredient forms, optional components, extra-ingredient counting and dish intent require verification. The human assignment files and unassigned master were preserved.

Anyone reviewing labels should keep the original AI source and record their own decisions separately. The strict requirement that all three people finish independent labels is still open. Human judgments and adjudication must exist before human agreement or pooled relevance metrics are reported.

MLflow run: `6c1bbc06ef8b4ae0b713b321cb2accea` in `sqlite:///mlflow.db`.
