# Benchmark (main/validation, top_k=10)

Measured on Windows-11-10.0.26200-SP0, Intel64 Family 6 Model 141 Stepping 1, GenuineIntel, 8 physical cores, 7.71 GB RAM, Python 3.12.10. 20 warmup queries, 100 measured queries, one after the other.

| Model | Bundle | Backend | p50 latency (ms) | p95 latency (ms) | Startup (s) | Peak RSS (MB) | Bundle size (MB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | `bm25-main-1f1f4ce5` | lexical | 43.39 | 69.84 | 0.68 | 558.1 | 5.94 |
| lsa | `lsa-main-1f1f4ce5` | lexical | 80.14 | 116.79 | 0.40 | 662.7 | 124.47 |
| minilm_pretrained | `minilm_pretrained-E04-main-1f1f4ce5` | dense | 63.74 | 75.28 | 20.77 | 732.5 | 74.32 |
| minilm_pretrained | `minilm_pretrained-E05-main-1f1f4ce5` | dense | 63.91 | 83.46 | 15.70 | 750.6 | 74.32 |
| tfidf | `tfidf-main-1f1f4ce5` | lexical | 13.61 | 17.90 | 0.09 | 546.6 | 8.44 |
