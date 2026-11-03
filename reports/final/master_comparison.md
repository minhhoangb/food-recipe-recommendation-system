| Model | Train MRR@10 | Test MRR@10 | CV MRR@10 | Training/index s | Tuned? | Notes |
|---|---:|---:|---|---:|---|---|
| E01 tfidf | 0.6998 | 0.6649 | 0.6896 ± 0.0063 | 28.024 | no |  |
| E02 tfidf | 0.6998 | 0.6649 | 0.6896 ± 0.0063 | 33.988 | yes (GroupKFold CV5) |  |
| E03* bm25 | 0.7687 | 0.7766 | N/A | 25.658 | no | Selected before test; N/A: rank-bm25 indexes every candidate (transductive IDF), nothing is fitted on train |
| LSA lsa | 0.3402 | 0.3424 | 0.3560 ± 0.0095 | 51.638 | no |  |
| E04 minilm_pretrained | 0.4575 | 0.4752 | N/A | 0.000 | no | N/A: pretrained encoder, nothing is fitted on our data |
| E05 minilm_pretrained | 0.1808 | 0.1989 | N/A | 0.000 | no | N/A: pretrained encoder, nothing is fitted on our data |
| FT minilm_finetuned | 0.6450 | 0.6368 | N/A | 871.336 | fine-tuned (MNRL) | N/A: one fixed one-epoch checkpoint, not cross-validated |

* = selected for deployment using validation before E10. Train and test are origin-recipe retrieval proxies, not human relevance. Train/Validation/Test, training-time and CV run IDs are in `master_comparison.csv`. Training/index/encoding times have different meanings and hardware; do not rank their speed.
