# Benchmark (main/validation, top_k=10)

Measured on Windows-11-10.0.26200-SP0, Intel64 Family 6 Model 154 Stepping 3, GenuineIntel, 14 physical cores, 15.75 GB RAM, Python 3.14.7. 20 warmup queries, 100 measured queries, one after the other.

| Model | Bundle | Backend | p50 latency (ms) | p95 latency (ms) | Startup (s) | Peak RSS (MB) | Bundle size (MB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| tfidf | `tfidf-main-1f1f4ce5` | lexical | 13.87 | 19.10 | 0.17 | 578.7 | 8.44 |
| bm25 | `bm25-main-1f1f4ce5` | lexical | 31.44 | 45.29 | 0.73 | 588.8 | 5.94 |
| lsa | `lsa-main-1f1f4ce5` | lexical | 33.08 | 36.29 | 0.51 | 693.6 | 124.47 |
| minilm_pretrained | `minilm_pretrained-E04-main-1f1f4ce5` | dense | 22.05 | 24.30 | 7.46 | 824.0 | 74.32 |
| minilm_pretrained | `minilm_pretrained-E05-main-1f1f4ce5` | dense | 21.14 | 23.37 | 7.28 | 824.8 | 74.32 |
| minilm_finetuned | `minilm_finetuned-E04-main-1f1f4ce5` | dense | 24.66 | 30.74 | 8.69 | 821.0 | 161.89 |
