"""Build Hoàng's seven-page model-development report from stored final evidence.

Run with the bundled artifact Python (reportlab). This script reads saved scores;
it never executes a retriever, changes a model, or opens unlabelled system keys.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build():
    final = REPORTS / "final"
    state = read(final / "e10_execution.json")
    if state["status"] != "complete":
        raise ValueError("The report requires completed E10 outputs")
    with (final / "master_comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by = {r["experiment"]: r for r in rows}
    manual = read(final / "manual_eval_status.json") if (final / "manual_eval_status.json").exists() else {}
    train = read(REPORTS / "finetune_cpu1ep_provenance.json")

    def f(label, split="test"):
        return f"{float(by[label][split + '_mrr@10']):.4f}"

    pages = []

    def page(title, paragraphs, table=None, after=None):
        pages.append((title, paragraphs, table, after or []))

    page(
        "1 Evaluation scope and frozen protocol",
        [
            "This part reports model development, a frozen final comparison and the resulting deployment decision "
            "for English ingredient-based recipe retrieval. It complements Part 1, which covers problem formulation, "
            "EDA, preprocessing and engineered features. The candidate corpus contains 50,000 RecipeNLG recipes. "
            "The system ranks existing recipes; it does not predict cooking time, nutrition, allergy safety or whether "
            "a particular person will enjoy a dish. The final result should therefore be read as evidence about a "
            "defined retrieval task, not a general guarantee of culinary usefulness.",
            "Every evaluated system uses the same grouped outer split and seed 42. Near-duplicate groups remain "
            "within one split. The outer test partition is approximately 20% of recipes; validation is held out "
            "within the remaining outer-training partition. Text vectorizers and fine-tuning positives use training "
            "recipes. All 50,000 recipes remain retrieval candidates, including held-out origin recipes. This "
            "candidate-corpus convention makes the task a retrieval benchmark rather than ordinary classification. "
            "BM25 also computes index statistics over the candidate collection; this transductive convention is "
            "disclosed rather than described as a train-only vocabulary.",
            "Synthetic queries are ingredient subsets of known recipes. Their relevance judgments identify origin "
            "recipes and any matched group positives. Recall@5 and Recall@10 measure recovery of these positives; "
            "MRR@10 measures the reciprocal rank of the first positive within ten results. A non-origin recipe "
            "can still be useful to a cook, so these scores are origin-recipe proxies. They do not replace the "
            "separate 0/1/2 human study. Empty retrieval results contribute zero rather than disappearing from the mean.",
            "The team decision in this chat froze E01, E02, E03, LSA, E04, E05 and the CPU one-epoch FT checkpoint. "
            "BM25 (E03) was selected using validation before opening E10 scores. The protocol fixes top-k=10 and "
            "no missing-first reranking. It records hashes of corpus, splits, queries, qrels, configurations, "
            "model files, evaluation code and pinned pretrained encoder snapshot. The FT query encoder is "
            "packaged with its corpus embeddings and checked against file hashes.",
            f"E10 ran once on 1,000 main test queries. Parent MLflow run: {state['mlflow_parent_run']}. "
            "The exclusive execution ledger was written before test queries were loaded. A repeat invocation "
            "is rejected, including after failure; per-system rankings are preserved. Report generation reads "
            "those rankings and scores only. No hyperparameter, reranking or deployment choice was changed after "
            "test inspection. Hash and timestamp evidence is in reports/final/frozen_config.json and e10_execution.json.",
        ],
    )

    page(
        "2 Models and training procedure",
        [
            "TF-IDF represents each recipe as a sparse vector of canonical ingredient names. Multi-word ingredients "
            "remain atomic tokens, so chicken stock and chicken breast are distinct. The vocabulary and inverse "
            "document frequencies are fitted on training recipes, while every candidate recipe is transformed with "
            "that fitted vectorizer. Query and document vectors use cosine similarity. E01 is the default model; "
            "E02 uses parameters selected by grouped cross-validation. Serialization stores the fitted pipeline "
            "and corpus alignment, allowing the serving code to load a bundle without refitting it.",
            "BM25 is a lexical ranking baseline implemented with rank-bm25. It combines ingredient overlap with "
            "inverse document frequency, term-frequency saturation and recipe-length normalization. The indexed "
            "candidate collection supplies its document statistics. This differs from TF-IDF's train-only fit, "
            "so the report names its timing as index construction rather than supervised training. BM25 is "
            "attractive when rare ingredient names sharply identify recipes; common or short ingredient queries "
            "remain ambiguous. A query with no ingredient overlap yields no lexical recommendation.",
            "LSA applies TruncatedSVD to the train-fitted TF-IDF representation. The evaluated model uses 256 "
            "components, not the 100 dimensions mentioned in an older draft. The low-dimensional representation "
            "can connect co-occurring ingredients but may lose the exact rare terms needed to recover an origin "
            "recipe. It is evaluated as a separate model, not assumed superior because its representation is dense.",
            "E04 and E05 use sentence-transformers/all-MiniLM-L6-v2, a six-layer transformer with 384-dimensional "
            "sentence embeddings. Text is truncated at 128 tokens in this project, pooled and L2-normalized. "
            "Queries contain normalized ingredients. E04 embeds ingredients only; E05 embeds title plus ingredients. "
            "Cosine search is implemented with sentence-transformers utilities and deterministic recipe-ID tie "
            "breaking. The encoder is pinned to the local snapshot recorded in the frozen manifest. These "
            "pretrained baselines have no training time on our recipe dataset; corpus encoding is a separate cost.",
            "The FT system is the main-data E08 checkpoint, not pilot E07. It uses the ingredients-only format "
            "and MultipleNegativesRankingLoss on 30,000 train-only positive pairs. The historical measured run "
            "used one epoch, batch 32, learning rate 2e-5, weight decay 0.01 and 50 warmup steps on CPU. "
            f"Observed training duration was {train['observed_training']['train_seconds']:.3f} seconds. "
            "The loss treats other positives in a batch as negatives. The existing loader does not enforce "
            "the available no-duplicate group sampler, so same-group false negatives are a known limitation. "
            "This report evaluates the locked checkpoint without retraining it or claiming a controlled ablation.",
        ],
    )

    page(
        "3 Tuning and bias variance analysis",
        [
            "TF-IDF tuning used GridSearchCV with GroupKFold(5) on outer-training recipes. Each fold refitted the "
            "vectorizer using only its training groups. A held-out query was scored in the fold holding its origin "
            "recipe, and held-out recipes remained candidates. The grid compared whole-ingredient tokens with "
            "ingredient-plus-word tokens, min_df in {1,2,5}, sublinear term frequency and normalization choices. "
            "There were 24 candidate configurations. Tests verify that held-out-only terms do not enter a fold's "
            "fitted vocabulary. Using GroupKFold addresses group leakage; it does not eliminate all corpus bias.",
            "The selected TF-IDF settings equal E01 defaults: whole-ingredient tokens, min_df=1, "
            "sublinear_tf=false and L2 normalization. CV MRR@10 was 0.6896 with standard deviation 0.0063. "
            "E02 therefore has the same Train, Validation and Test rankings as E01. This is a valid negative "
            "tuning result. The report does not invent a gain or describe E02 as a sublinear-TF model. Splitting "
            "multi-word names can increase broad matches, while a larger minimum document frequency removes "
            "rare identifiers; the CV results, rather than those intuitions alone, determine the selected setting.",
            "LSA's grouped five-fold score is 0.3560 ± 0.0095 at the evaluated 256-component setting. This is a "
            "measured fixed-setting CV result, not a searched optimum. BM25 has no supervised fit in this setup, "
            "and the pretrained encoders are not fitted on recipe labels, so their CV entries are N/A with that "
            "reason. FT has one fixed checkpoint and no cross-validated retraining. N/A is not silently replaced "
            "with zero and must not be used in an average over model quality.",
            f"Train-proxy versus validation MRR@10 is {f('E01', 'train')} versus {f('E01', 'val')} for TF-IDF, "
            f"{f('E03', 'train')} versus {f('E03', 'val')} for BM25, and {f('FT', 'train')} versus {f('FT', 'val')} "
            "for FT. These retrieval gaps are descriptive; train-proxy retrieval scores are not training losses. "
            "The single FT run and seed do not estimate training variance. A small observed gap cannot establish "
            "absence of overfitting, nor does a better test score justify retrospectively choosing a checkpoint.",
            "The validation error analysis provides more specific evidence: 28 deterministic cases compare top-five "
            "results from all seven systems. It separates short queries, common-ingredient share, rare terms, "
            "singular/plural variants and large origin recipes. FT improved over pretrained E04 on 196 validation "
            "queries, regressed on 30 and tied on 274. These are cases from a fixed validation comparison, not "
            "new test-derived tuning rules. No test-case inspection was used to modify the frozen systems.",
            "A separate controlled weight-decay loss experiment was completed on Colab T4 on 2026-11-03; "
            "reports/ablation_regularization.md links its original runs, measured histories and actual MLflow "
            "UI captures. The matched wd=0.0/0.01 arms differ by about 0.00000103 in final validation loss, "
            "which does not establish a meaningful regularization benefit. This experiment does not change "
            "the locked checkpoint, frozen deployment choice or E10 evaluation.",
        ],
    )

    master = [["System", "Train MRR", "Test MRR", "CV MRR", "Train/index s", "Tuned", "Notes"]]
    for row in rows:
        cv = (
            f"{float(row['cv_mrr@10_mean']):.4f} ± {float(row['cv_mrr@10_std']):.4f}"
            if row["cv_mrr@10_mean"]
            else "N/A"
        )
        master.append(
            [
                row["experiment"] + (" *" if row["deployment_marker"] else ""),
                f"{float(row['train_mrr@10']):.4f}",
                f"{float(row['test_mrr@10']):.4f}",
                cv,
                f"{float(row['training_seconds']):.3f}",
                "CV5" if row["experiment"] == "E02" else ("MNRL" if row["experiment"] == "FT" else "No"),
                {"E01": "Defaults", "E02": "CV best=default", "E03": "Selected pretest; index",
                 "LSA": "256 components", "E04": "Ingredients", "E05": "Title + ingredients",
                 "FT": "E08; one CPU epoch"}[row["experiment"]],
            ]
        )
    wins = by["E03"]["test_mrr_significantly_better_than"]
    page(
        "4 Master Comparison Table and final results",
        [
            "All rows below use the frozen main corpus. Train scores are historical retrieval proxies over 5,000 "
            "train queries; Test scores come from the single E10 execution over 1,000 test queries. CV scores "
            "use grouped folds of the outer-training set. The asterisk marks the deployment decision made on "
            "validation before E10. It is not assigned by sorting this test table.",
        ],
        master,
        [
            "Notes: E01 and E02 have identical selected parameters. LSA's CV is a fixed-setting assessment, not "
            "tuning. BM25's time is index build; TF-IDF/LSA time includes fit and corpus transform. FT's duration "
            "is the historical CPU fine-tuning run. E04/E05 show zero dataset training seconds because their "
            "weights are pretrained; their nonzero corpus-encoding durations are stored separately. Different "
            "machines and timing definitions prevent a fair speed ranking from this training-time column.",
            f"BM25 test Recall@5 is {float(by['E03']['test_recall@5']):.4f}, Recall@10 is "
            f"{float(by['E03']['test_recall@10']):.4f}, and MRR@10 is {f('E03')}. "
            f"FT test MRR@10 is {f('FT')}, compared with {f('E04')} for pretrained E04 and "
            f"{f('E05')} for E05. The result supports the frozen lexical deployment choice on this proxy; "
            "it does not establish that lexical systems satisfy every natural-language cooking intent.",
            "The significance analysis uses ranx.compare and a two-sided paired Student t-test over per-query "
            f"scores, p < 0.01. For test MRR@10, BM25 is significantly better than: {wins}. "
            "The full per-metric comparisons and per-query scores are saved alongside the table. Pairwise "
            "tests are not corrected for multiplicity; the threshold must not be interpreted as a familywise "
            "error guarantee. E01/E02's identical ranks may produce undefined pairwise p-values rather than a win.",
            "The final scores, raw test rankings and full MLflow run IDs are in reports/final. Table generation "
            "checks each copied Train/Validation/Test metric against its source run. It does not score a retriever "
            "again. The ledger records completion of all seven systems, and the selected model remains E03.",
        ],
    )

    page(
        "5 Deployment choice and practical limits",
        [
            "The selection rule was highest validation MRR@10 among the fixed systems, with performance and "
            "operational limitations disclosed. BM25's validation score of 0.7583 exceeded TF-IDF's 0.6567 and "
            "FT's 0.6209. The user confirmed E03 before E10. The production configuration now identifies the "
            "same BM25 main bundle. The subsequent evidence addendum verifies the fresh local clone and "
            "a rebuilt local Docker image; reports/docker_verification.json records healthy startup, "
            "container smoke and five native/container queries with identical IDs, order and scores.",
            "BM25's serialized joblib bundle is small and does not require loading transformer weights for "
            "inference. The serving benchmark is a separate experiment: 20 warmup queries and 100 sequential "
            "validation queries per bundle, with startup, latency, RAM and artifact size recorded. These values "
            "describe one machine and workload. They cannot prove an end-user SLA, concurrent throughput or "
            "cold-start performance on a different host. E10 mean query latency also should not be substituted "
            "for the dedicated benchmark's p50/p95 measurements.",
            "The underlying labels reward origin recovery. A different recipe containing the requested "
            "ingredients may be a better real meal than the origin. Conversely, ingredient overlap can reward "
            "recipes that violate an explicit intent such as vegetarian or no oven. The API has no reliable "
            "dietary constraint model, personalization, nutrition or availability knowledge. Human judgments "
            "must therefore consider intent, extra ingredients and usability rather than reproduce the proxy.",
            "Preprocessing normalizes ingredient names and singular/plural variants, but parsing errors, "
            "ambiguous synonyms and specific ingredient substitutions remain. Treating stock as meat, ignoring "
            "a preparation requirement or merging distinct ingredients can mislead retrieval. The EDA audit "
            "and error-analysis cases document these issues; a high average MRR does not remove them. "
            "Deterministic recipe IDs, manifests and tie breaking make results reproducible, not automatically correct.",
            "The model freeze includes artifact and encoder hashes and code provenance. The command refuses "
            "changed hashes or an existing E10 ledger. This supports auditability within this repository. "
            "It is not an enforcement mechanism against someone deleting audit files or using a separate "
            "tracking store. If a model must change after final test inspection, disclose the deviation "
            "and evaluate on newly reserved data rather than treating another pass as the same untouched test.",
            "Potential future work includes an independently labelled evaluation set, better intent and "
            "substitution handling, safer no-overlap responses and multiple training seeds. These are proposals, "
            "not features implemented or gains demonstrated here. They should be chosen using future development "
            "data, not by tuning to the final query pool or revisiting E10 failures.",
        ],
    )

    page(
        "6 Human evaluation and ethical considerations",
        [
            "The natural-language study uses the existing 20 final queries, separate from the 20 development "
            "queries. The frozen systems contribute their top-five recipes per query; duplicates are merged "
            "and item order is shuffled with seed 42. Assigned CSVs contain query, recipe, ingredients and "
            "directions, but no model identity or rank. Queries are allocated round-robin to Hoàng, Hưng and "
            "Hiếu, with at least 20% of each query's items independently assigned to a second person.",
            "The original annotation guide defines relevance 0/1/2 before the final pool is generated. Judges "
            "must work independently, avoid suggested AI labels and avoid system keys. Each person records "
            "actual timing, source and exposure in annotation_log.csv. A declaration is not signed by another "
            "person or filled by this report. Independent originals are hash-locked before disagreements are "
            "shown. Discussion is recorded in a separate adjudication table; original labels remain unchanged.",
            f"Current final-study status: {manual.get('status', 'awaiting_independent_labels')}. "
            "The pool and scoring implementation are ready, but independent final human labels have not yet "
            "been supplied. This report therefore publishes no human kappa or pooled nDCG score. It does not "
            "reinterpret blank labels as zero relevance, reuse dev judgments as final judgments, or present "
            "AI-generated judgments as agreement between the three team members.",
            "After all originals are complete, the scoring command computes pairwise unweighted Cohen's "
            "kappa and raw agreement on the independent overlap. It then uses adjudicated labels to compute "
            "ranx nDCG@10 with linear relevance gains. Items outside the pool remain unjudged. The pooled "
            "calculation assigns them zero gain, a scoring convention that does not prove irrelevance. Every "
            "system is reported with mean and per-query top-ten judging coverage. A query with no positive "
            "pooled label and an undefined kappa denominator are explicitly limited cases, not flattering estimates.",
            "Allergy and dietary safety require particular care. The source corpus does not provide verified "
            "allergen or cross-contamination information, and normalization can erase distinctions. Users should "
            "check the original ingredient list and personal restrictions; the system must not claim that a "
            "recommendation is safe for an allergy or a medical diet. The report's evaluation does not certify "
            "those properties. Deploying such a promise would require additional data, validation and controls.",
            "Recipe sources and language coverage can bias the candidate collection. English-only recipes, "
            "popular ingredients and duplicated source styles may dominate the learned representation and "
            "judgment pool. The dataset licence has not been assumed or reassigned: cite RecipeNLG and source "
            "links, check redistribution rights and keep raw data/model artifacts out of source control. "
            "Prediction logs should be minimized and retained deliberately because ingredient histories can "
            "reveal personal habits. These limits travel with the reported scores.",
        ],
    )

    provenance = [["System", "Train run", "Validation run", "E10 test run"]]
    for row in rows:
        provenance.append(
            [row["experiment"], row["mlflow_train_run"], row["mlflow_val_run"], row["mlflow_test_run"]]
        )
    page(
        "7 Provenance and rubric review",
        [
            "The table below gives the full MLflow run IDs for all copied retrieval scores. Resolve them in "
            "the local sqlite:///mlflow.db tracking store, experiment recipe-retrieval. Each E10 child run "
            "contains its system label, frozen-manifest hash, metrics and saved test ranking. The parent run "
            "contains the aggregated evidence. Full-precision CSVs remain the numerical authority when printed "
            "tables round to four decimal places.",
        ],
        provenance,
        [
            f"FT training-time provenance: {train['mlflow_run_id']}; measured time "
            f"{train['observed_training']['train_seconds']:.3f} seconds imported from the historical CPU log, "
            "not a new training run. TF-IDF grouped CV run: dac82cb580e8403082cd9d0fabb40448. "
            "LSA CV run: d46efe09f7dd4835841f66b860d4d433. The CV JSONs give means, standard deviations, "
            "fold counts and corpus hashes. The master CSV links timing and CV fields to these sources.",
            "The rubric checklist maps all five phases and their subrequirements to files, runs and responsible "
            "owners. The evidence addendum completes measured regularization, actual MLflow UI captures, "
            "artifact-download configuration and the native BM25 fresh-clone check. The selected-BM25 "
            "container rebuild, health, smoke and parity check also passed after the Windows restart. "
            "Final human judgments remain pending "
            "for the group. The checklist also identifies incomplete student IDs, "
            "clean-kernel notebook evidence and final presentation/rehearsal deliverables. It is an evidence "
            "map, not a guessed course grade.",
            "The retrospective review of Hiếu's PR #20 records three findings: generated UI facsimiles and "
            "constant ablation loss arrays cannot establish measured experiments; scripted recommendation "
            "function calls are not naturally collected API user traffic; and experiment/model descriptions "
            "must match main FT E08, TF-IDF settings and 256-component LSA. Those findings are reported, not "
            "treated as satisfactory historical evidence. The separate evidence addendum documents the "
            "subsequent measured ablation and replacement UI captures.",
            "References and primary artifacts: RecipeNLG dataset and source links (README Dataset); "
            "scikit-learn TF-IDF/TruncatedSVD/GroupKFold; rank-bm25; sentence-transformers MiniLM and "
            "MultipleNegativesRankingLoss; ranx metrics and paired comparison; MLflow tracking. Repository "
            "implementations are src/baseline.py, tune.py, finetune.py, evaluate.py and final_evaluation.py. "
            "Human-study conventions are locked by tests/test_ranking_metrics.py and annotation_guide.md. "
            "The companion Master Comparison CSV, frozen manifest and execution ledger are included in reports/final.",
        ],
    )

    # Author Markdown first; keep the same content and tables in the PDF.
    md = [
        "# Model Development and Final Evaluation",
        "",
        "Technical report Part 2 | Hoàng | 3 November 2026 | Frozen main test E10",
        "",
    ]
    for title, paragraphs, table, after in pages:
        md.extend(["## " + title, "", *[p + "\n" for p in paragraphs]])
        if table:
            md.append("| " + " | ".join(table[0]) + " |")
            md.append("| " + " | ".join("---" for _ in table[0]) + " |")
            md.extend("| " + " | ".join(row) + " |" for row in table[1:])
            md.append("")
        md.extend(p + "\n" for p in after)
        md.append("<!-- PAGEBREAK -->\n")
    (REPORTS / "report_part2_models.md").write_text("\n".join(md), encoding="utf-8")

    regular = Path("C:/Windows/Fonts/segoeui.ttf")
    bold = Path("C:/Windows/Fonts/segoeuib.ttf")
    pdfmetrics.registerFont(TTFont("Report", str(regular)))
    pdfmetrics.registerFont(TTFont("ReportBold", str(bold)))
    body = ParagraphStyle(
        "ReportBody", fontName="Report", fontSize=9.4, leading=13.6, spaceAfter=8, alignment=TA_LEFT
    )
    heading = ParagraphStyle(
        "ReportHeading", parent=body, fontName="ReportBold", fontSize=16, leading=20, spaceAfter=14
    )
    small = ParagraphStyle("ReportSmall", parent=body, fontSize=7.3, leading=10.2, spaceAfter=0)
    story = []
    for index, (title, paragraphs, table, after) in enumerate(pages):
        if index:
            story.append(PageBreak())
        else:
            story.append(Paragraph("Model Development and Final Evaluation", heading))
            story.append(Paragraph("Technical report Part 2 | Bui Minh Hoang | 3 November 2026", body))
            story.append(Spacer(1, 10))
        story.append(Paragraph(html.escape(title), heading))
        story.extend(Paragraph(html.escape(p), body) for p in paragraphs)
        if table:
            values = [[Paragraph(html.escape(value), small) for value in row] for row in table]
            widths = [40, 146, 146, 146] if index == 6 else [43, 55, 55, 100, 78, 55, 92]
            grid = Table(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
            grid.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf1")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#536372")),
                        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#c7cdd3")),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([grid, Spacer(1, 11)])
        story.extend(Paragraph(html.escape(p), body) for p in after)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Report", 8)
        canvas.setFillColor(colors.HexColor("#536372"))
        canvas.drawString(18 * mm, 13 * mm, "Food Recipe Recommendation System | Model Development")
        canvas.drawRightString(192 * mm, 13 * mm, str(doc.page))
        canvas.restoreState()

    pdf = REPORTS / "report_part2_models.pdf"
    document = SimpleDocTemplate(
        str(pdf),
        pagesize=(210 * mm, 297 * mm),
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=21 * mm,
        title="Model Development and Final Evaluation",
        author="Bui Minh Hoang",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(pdf)


if __name__ == "__main__":
    build()
