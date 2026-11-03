# Shared models and MLflow snapshot

The full existing `models/` directory is published as a ZIP on
[GitHub Release model-artifacts-20261103](https://github.com/minhhoangb/food-recipe-recommendation-system/releases/tag/model-artifacts-20261103).
It contains 67 files, including BM25/TF-IDF/LSA indexes, pretrained embeddings,
the fine-tuned encoder, and the rollback-trial snapshot. The archive contains
only paths under `models/`. The `data/` directory is excluded from this upload.

- [Download models-20261103-da6b6d75ae.zip](https://github.com/minhhoangb/food-recipe-recommendation-system/releases/download/model-artifacts-20261103/models-20261103-da6b6d75ae.zip).
- ZIP size: 681,917,816 bytes (about 650 MiB); extracted size: 759,977,629 bytes.
- SHA-256: `09245386be5639017a04df4f709497f5c97c8d62bec44e0cf5cb0b060d810efc`.
- Per-file sizes and hashes: [reports/model_artifacts.json](../reports/model_artifacts.json).

The release preserves the existing models; it does not retrain or select a new
deployment model. The production config continues to select BM25. You can still
build that bundle from the processed data using the README instead of downloading
all the archived models.

To restore the archive, run this Python code from a clone's root. Use a directory
without existing model files, since extraction restores the archived paths.

```python
import hashlib
import urllib.request
import zipfile
from pathlib import Path

url = "https://github.com/minhhoangb/food-recipe-recommendation-system/releases/download/model-artifacts-20261103/models-20261103-da6b6d75ae.zip"
archive = Path("models.zip")
urllib.request.urlretrieve(url, archive)
with archive.open("rb") as handle:
    digest = hashlib.file_digest(handle, "sha256").hexdigest()
assert digest == "09245386be5639017a04df4f709497f5c97c8d62bec44e0cf5cb0b060d810efc"
with zipfile.ZipFile(archive) as bundle:
    bundle.extractall(".")
```

The root `mlflow.db` is a checked SQLite snapshot with 52 recorded runs, including
the imported controlled Colab ablation. Open the history with the pinned project
environment:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

The database retains its original records and artifact URIs. `mlruns/` was not
uploaded, so some artifact links still reference files on the original machine.
Metrics, parameters and tags can be inspected from the database; the exported
Colab measurements and actual UI screenshots are also versioned under `reports/`.
The ZIP contains the saved model files, not the MLflow artifact-store layout.

The database is a snapshot rather than a shared live server. Creating new local
runs changes it; intentionally publish a fresh snapshot when it is time to share
updated history.
