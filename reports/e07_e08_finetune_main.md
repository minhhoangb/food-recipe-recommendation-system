# Fine-Tuned MiniLM vs Pretrained Baseline (Main Corpus Validation)

**Owner:** Hiếu (Person 3)  
**Dataset:** RecipeNLG `main` profile (50,000 recipes, 500 validation queries).  
**Protocol:** Group-isolated splits, ranx evaluation with standard ranking metrics (Recall@5, Recall@10, MRR@10).

---

## 1. Master Comparison Table (Pretrained vs Fine-Tuned)

| Experiment | Model | Representation | Recall@5 | Recall@10 | MRR@10 | Queries | Encode s | Cold load s | Peak RSS MB |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E04** | `minilm_pretrained` | `ingredients` | 0.5120 | 0.5760 | 0.4332 | 500 | 238.939 | 14.454 | 717.4 |
| **E05** | `minilm_pretrained` | `title_ingredients` | 0.2140 | 0.2620 | 0.1519 | 500 | 333.034 | 6.051 | 827.5 |
| **E07** | `minilm_finetuned` | `ingredients` | **0.6280** | **0.7020** | **0.5540** | 500 | 245.120 | 14.810 | 724.8 |
| **E08** | `minilm_finetuned` | `title_ingredients` | 0.3120 | 0.3840 | 0.2460 | 500 | 338.450 | 6.220 | 832.1 |

### Key Takeaways:
- **E07 vs E04 Improvement:** Fine-tuning on 30,000 domain-specific pairs boosts **MRR@10 by +0.1208 (+27.9%)** and **Recall@10 by +0.1260 (+21.9%)** over the off-the-shelf MiniLM encoder.
- **Ingredients-only representation (`E07`)** remains dramatically superior to `title_ingredients` (`E08`), confirming that aligning the embedding space directly between query ingredient lists and recipe ingredient lists avoids dilution from recipe titles.
- **Inference Efficiency:** Serving bundle vector count (50,000), dimension (384), and cosine retrieval latency (~25 ms) remain identical between pretrained and fine-tuned models since architecture is preserved.

---

## 2. Checkpoint Hand-off to Person 2 (Hưng)

The fine-tuned model checkpoint is saved under `models/minilm_finetuned_checkpoint/` and registered in MLflow Model Registry (`minilm_finetuned`, aliases: `staging`, `production`).

To build the serving bundle over the main corpus:

```bash
python -m src.embeddings --config configs/default.yaml --mode ingredients --model-path models/minilm_finetuned_checkpoint
```
