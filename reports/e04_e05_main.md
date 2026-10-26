| Experiment | Representation | Recall@5 | Recall@10 | MRR@10 | Queries | Encode s | Cold load s | Peak RSS MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E04 | ingredients | 0.5120 | 0.5760 | 0.4332 | 500 | 238.939 | 14.454 | 717.4 |
| E05 | title_ingredients | 0.2140 | 0.2620 | 0.1519 | 500 | 333.034 | 6.051 | 827.5 |

| Experiment | Rerank | Recall@5 | Recall@10 | MRR@10 | Queries | Wall s |
| --- | --- | --- | --- | --- | --- | --- |
| E04 | none | 0.5120 | 0.5760 | 0.4332 | 500 | 24.555 |
| E04 | missing_first | 0.2920 | 0.3800 | 0.2287 | 500 | 29.24 |
| E05 | none | 0.2140 | 0.2620 | 0.1519 | 500 | 26.057 |
| E05 | missing_first | 0.1920 | 0.2400 | 0.1437 | 500 | 25.818 |
