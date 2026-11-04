# Docker verification completed after the Windows restart

Completed on 2026-11-03: the selected-BM25 Docker image was rebuilt, the container
became healthy, internal endpoint smoke passed, and all five native/container
queries matched with zero score difference. Evidence:
`reports/docker_verification.json`, `reports/docker_verification.md`,
`reports/docker_smoke.json` and `reports/parity.json`.

The instructions below are retained for reproducing the checks. Before the restart,
Windows Virtual Machine Platform was disabled;
Docker Desktop's Linux engine returned HTTP 500, and WSL reported
`Wsl/CallMsi/Install/REGDB_E_CLASSNOTREG`. After the user enabled virtualization
and restarted, the actual local Docker Linux engine was verified.

Completed evidence is saved locally:

- Actual Colab T4 wd=0.0/0.01 experiment: `reports/regularization_ablation/`.
- All three actual MLflow UI PNGs: `reports/figures/mlflow_*.png` and their JSON provenance.
- Public processed-data ZIP URL and SHA-256: `configs/default.yaml`.
- Fresh local clone with the working patch, fresh venv, verified data, BM25 and API:
  `D:/MDS301/clean-clone-regularization-20261103`;
  `reports/clean_clone_verification.json` and `reports/clean_clone_smoke.json`.
- `reports/parity_e04_historical.json` preserves the old E04 result.
  `reports/parity.json` now contains the passing selected-BM25 comparison.

Code changes are uncommitted and have not been pushed. The data and experiment
source ZIPs have been published to the public GitHub Release
`processed-1f1f4ce505`. Preserve the user's pre-existing untracked report/log files.

From PowerShell, verify the Linux daemon first:

```powershell
Set-Location D:\MDS301\food-recipe-recommendation-system
docker version
docker info
docker compose up -d --build --force-recreate api
docker compose ps
docker compose logs --tail 50 api
docker compose exec -T api python scripts/smoke_api.py --expected-model bm25 |
  Set-Content -Encoding utf8 reports/docker_smoke.json
```

Start a native BM25 server in a separate terminal (it must be restarted after reboot):

```powershell
Set-Location D:\MDS301\food-recipe-recommendation-system
$env:API_CONFIG = 'configs/production.yaml'
$env:API_REQUEST_LOG = ''
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8010
```

Then run and inspect the actual parity output:

```powershell
.\.venv\Scripts\python.exe scripts/check_parity.py `
  --native http://127.0.0.1:8010 --docker http://127.0.0.1:8080 --expected-model bm25
```

Record image ID, build output, container health and smoke output with the report.
P3.2 was marked complete after Docker actually served BM25 and parity passed.
Keep named volumes and request logs; no `docker compose down -v` is needed.

MLflow UI can be reopened without retraining:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

The imported ablation is experiment 2. Local child run IDs:
`6b4b0facbbfb4082be15b94f3977c805` and `6178d21f4c064d019dc8fab00efd384a`.
The validation-loss difference is approximately 0.00000103; do not claim a
meaningful regularization improvement or change the locked E10 model selection.
