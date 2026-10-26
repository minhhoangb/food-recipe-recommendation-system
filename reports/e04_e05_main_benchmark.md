# E04 / E05 on the MAIN corpus (50 000 recipes) and E06 rerank ablation

All numbers come from MLflow runs on the real data (`sqlite:///mlflow.db`), the
validation split of the `main` profile (500 queries, 50 000 recipes).

## Hardware (the numbers below are only valid for this machine)

| Item | Value |
| --- | --- |
| CPU | Intel Core i7-11800H @ 2.30 GHz (8 cores / 16 threads) |
| RAM | 8 GB (7.7 GiB usable) |
| OS | Windows 11 Home 64-bit |
| GPU | none (CPU only, `torch 2.9.1+cpu`) |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2`, `max_seq_length=128`, batch 32 |
| Corpus sha256 | `1f1f4ce50525727ba5242972dac8c93dc32e9555207cbf8b4ebb0f31f0409b46` (50 000 recipes) |

## (a) Encoding the MAIN corpus

```bash
python -m src.embeddings --config configs/default.yaml --mode ingredients        # E04
python -m src.embeddings --config configs/default.yaml --mode title_ingredients  # E05
```

| Bundle | Mode | Documents | Dim | Truncated | Encode s | Embeddings size | recipe_ids.json | Manifest sha256 match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `minilm_pretrained-E04-main-1f1f4ce5` | E04 (ingredients) | 50 000 | 384 | 1 | 238.9 | 73.24 MB | 1.07 MB | yes |
| `minilm_pretrained-E05-main-1f1f4ce5` | E05 (title + ingredients) | 50 000 | 384 | 2 | 333.0 | 73.24 MB | 1.07 MB | yes |

- Bundle size: **74.3 MB** per mode (73.24 MB `embeddings.npy` + 1.07 MB `recipe_ids.json`).
- Throughput: **209 documents/s** (E04) and **150 documents/s** (E05). E05 is slower
  because the document text is longer (the title is prepended).
- Peak RAM during the evaluation run: **717.4 MB** (E04) / **827.5 MB** (E05), measured
  with `src.memory.peak_rss_mb()` (process peak, includes the MiniLM weights).
- Cold load of the bundle (model + 50 000 vectors): **14.5 s** (E04) / **6.1 s** (E05).
  The second number is faster only because the MiniLM weights were already in the OS page
  cache from the first load; the first cold load after a reboot is the honest number.
- Query latency on the main bundle: **123 ms** per query (6 queries, top_k 10, single
  CPU, model weights warm). The corpus is never re-encoded, so this is one query
  forward pass plus a 50 000 x 384 dot product.

## (b) E04 / E05 on the main validation split

```bash
python -m src.evaluation --mode both --config configs/default.yaml --split validation \
    --ablation --out reports/e04_e05_main.md
```

| Experiment | Representation | Recall@5 | Recall@10 | MRR@10 | Queries | Encode s | Cold load s | Peak RSS MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E04 | ingredients | 0.5120 | 0.5760 | 0.4332 | 500 | 238.939 | 14.454 | 717.4 |
| E05 | title + ingredients | 0.2140 | 0.2620 | 0.1519 | 500 | 333.034 | 6.051 | 827.5 |

**E04 wins clearly on the main corpus.** The reason is visible in the corpus itself:
the query is an ingredient list, and E05 embeds the *document* as "title + ingredients",
so the document and the query live in two different text spaces (the query has no title).
E04 embeds both sides the same way, so the cosine is comparable. On the pilot corpus
the same order holds (E04 R@10 0.785 vs E05 R@10 0.495).

## (c) E06: missing-first rerank ablation

Same corpus, same queries, same retriever; only the ranking of the top 100 candidates
changes (fewest missing ingredients first, then the score, then the recipe id).

```bash
python -m src.evaluation --mode both --config configs/default.yaml --split validation --ablation
```

| Experiment | Rerank | Recall@5 | Recall@10 | MRR@10 | Queries | Wall s |
| --- | --- | --- | --- | --- | --- | --- |
| E04 | none | 0.5120 | 0.5760 | 0.4332 | 500 | 24.555 |
| E04 | missing_first | 0.2920 | 0.3800 | 0.2287 | 500 | 29.240 |
| E05 | none | 0.2140 | 0.2620 | 0.1519 | 500 | 26.057 |
| E05 | missing_first | 0.1920 | 0.2400 | 0.1437 | 500 | 25.818 |

**Trade-off.** The rerank costs 34% of the relative Recall@10 on E04 (0.576 -> 0.380) and
25% of the MRR@10 (0.433 -> 0.229); on E05 the loss is smaller (R@10 0.262 -> 0.240)
because E05 has fewer good candidates in the first place. The rerank also costs ~20% more
wall time (29.2 s vs 24.6 s for 500 queries) because it retrieves and scores 100
candidates per query instead of 10.

Why the metrics go down: the synthetic label is the *origin recipe* of the query, and
the origin recipe is the one with the **most** missing ingredients (the query was built
by hiding some of its ingredients). Ranking by "fewest missing" therefore pushes the
labelled recipe down. The rerank optimises the wrong objective for these metrics.

The rerank is still useful for the product: "you have chicken and rice, here are the
recipes you can cook tonight" is a different question from "here is the recipe this
query came from". It is therefore **off by default** (`rerank="none"` in
`src/predict.py`, `MissingFirstRetriever` is opt-in) and the metrics above document
what it costs. The final ranking should be judged with manual labels (Person 1), not
with the origin-recipe proxy.
