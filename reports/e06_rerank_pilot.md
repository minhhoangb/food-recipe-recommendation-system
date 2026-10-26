| Experiment | Representation | Recall@5 | Recall@10 | MRR@10 | Queries | Encode s | Cold load s | Peak RSS MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E04 | ingredients | 0.7050 | 0.7850 | 0.6121 | 200 | 21.632 | 19.268 | 576.7 |
| E05 | title_ingredients | 0.4250 | 0.4950 | 0.3690 | 200 | 27.803 | 5.184 | 679.4 |

| Experiment | Rerank | Recall@5 | Recall@10 | MRR@10 | Queries | Wall s |
| --- | --- | --- | --- | --- | --- | --- |
| E04 | none | 0.7050 | 0.7850 | 0.6121 | 200 | 8.403 |
| E04 | missing_first | 0.4450 | 0.4900 | 0.3766 | 200 | 12.171 |
| E05 | none | 0.4250 | 0.4950 | 0.3690 | 200 | 4.655 |
| E05 | missing_first | 0.3900 | 0.4500 | 0.2745 | 200 | 5.003 |
