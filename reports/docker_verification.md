# Selected-BM25 Docker verification

Verified on 2026-11-03 using the actual local Docker Desktop Linux engine after
the Windows virtualization restart. Docker client and server: **29.7.2**.

The image was built from the current workspace with `docker compose build api`,
then started with `docker compose up -d --no-build --force-recreate api`.
The init service populated a new bundles volume from the host models; the app
mounted corpus and model files read-only. No existing named volume was deleted.

- Image: `food-recipe-api:cpu`.
- Image ID: `sha256:a934b170afe7d161d25d8ea94c0ec80e85e0dd756b22e97b7b17c0cba8218020`.
- Image created: `2026-11-03T11:14:54.378803006Z`.
- Docker-reported image size: 752,052,796 bytes.
- Published endpoint: `http://127.0.0.1:8080` (container port 8000).
- Docker healthcheck: **healthy**.
- `/health`: `{"status":"ok","model":"bm25","model_version":"1"}`.
- Serving bundle: `models/bm25-main-1f1f4ce5`, 50,000 recipes.
- Corpus SHA-256: `1f1f4ce50525727ba5242972dac8c93dc32e9555207cbf8b4ebb0f31f0409b46`.

The smoke command ran **inside the running container**:

```bash
docker compose exec -T api python scripts/smoke_api.py --expected-model bm25
```

It checked `/health`, `/model-info`, five unique `/predict` results,
`/batch-predict` with five/two results and preserved order, and HTTP 422 for
invalid `top_k`. All assertions passed. These requests are test traffic.

Native/container parity used a freshly started Windows native server on port
8010 and the container's published port 8080:

```bash
python scripts/check_parity.py --native http://127.0.0.1:8010 --docker http://127.0.0.1:8080 --expected-model bm25
```

| Ingredients | Compared results | Recipe IDs and order | Maximum score difference |
| --- | --- | --- | --- |
| chicken, rice, garlic | 5 | Identical | 0 |
| pasta, tomato, basil | 5 | Identical | 0 |
| egg, milk, flour | 5 | Identical | 0 |
| beef, onion, carrot | 5 | Identical | 0 |
| rice, soy sauce | 5 | Identical | 0 |

Both endpoints reported BM25, the same bundle and the same corpus hash.
`reports/parity.json` now records **all_ok=true** for BM25; the earlier E04
report remains in `reports/parity_e04_historical.json`.

Machine-readable evidence:
[image/container state and source hashes](docker_verification.json),
[container smoke](docker_smoke.json), [native smoke](native_bm25_smoke.json),
[every parity response ID and score difference](parity.json).
Build/startup logs are retained locally in `logs/docker-bm25-build.log` and
`logs/docker-bm25-startup.log` (git-ignored).

The fresh-clone result is documented separately in
[clean_clone_verification.json](clean_clone_verification.json): it used a fresh
local clone with the working patch and a new venv. Source changes remain local,
uncommitted and unpushed; this is not a claim about the current GitHub main branch.
