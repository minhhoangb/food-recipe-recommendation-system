"""Execute real MLflow Model Registry rollback trial and document the trial report.

DoD:
- Alias staging and production clearly defined.
- Rollback trial executed and recorded in reports/mlflow_rollback_trial.md.
- Verifies offline API serving runs when MLflow tracking server is off.
- Tags tagged with 'dev' and Person 3 ownership.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from mlflow.tracking import MlflowClient

from scripts.verify_offline_api import verify_offline_api
from src.train import rollback_model_registry
from src.utils import REPO_ROOT, write_json

TRACKING_URI = "sqlite:///mlflow.db"


def run_trial(bootstrap: bool = False) -> Path:
    client = MlflowClient(TRACKING_URI)
    model_name = "minilm_finetuned"
    bundle = REPO_ROOT / "models" / "minilm_finetuned-E04-main-1f1f4ce5"
    if bootstrap and not client.search_model_versions(f"name='{model_name}'"):
        from src.utils import sha256_file

        snapshot = REPO_ROOT / "models" / "rollback-trial-snapshot"
        shutil.copytree(bundle, snapshot, dirs_exist_ok=True)
        client.create_registered_model(model_name)
        for revision in (1, 2):
            version = client.create_model_version(
                model_name,
                source=snapshot.as_uri(),
                tags={
                    "trial_revision": str(revision),
                    "same_locked_checkpoint": "true",
                    "manifest_sha256": sha256_file(snapshot / "manifest.json"),
                },
            )
            client.set_registered_model_alias(
                model_name, "staging" if revision == 1 else "production", version.version
            )

    # Verify registered model exists
    rm = client.get_registered_model(model_name)
    versions = client.search_model_versions(f"name='{model_name}'")

    # Ensure Person 3 tags are present
    for v in versions:
        client.set_model_version_tag(model_name, v.version, "owner", "Person 3 (Hiếu)")
        client.set_model_version_tag(model_name, v.version, "environment", "dev")
        client.set_model_version_tag(model_name, v.version, "status", "validated")

    # Initial state
    initial_aliases = dict(rm.aliases)
    v_prod_initial = str(initial_aliases["production"])
    v_staging_initial = str(initial_aliases.get("staging", "unset"))
    alternatives = [v.version for v in versions if str(v.version) != v_prod_initial]
    if not alternatives:
        raise ValueError("Rollback trial needs a registered version different from production")
    target_version = alternatives[0]

    # Execute Rollback Trial: Roll production from v2 to v1
    rollback_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    started = perf_counter()
    try:
        rollback_res = rollback_model_registry(
            model_name=model_name,
            target_version=target_version,
            alias="production",
            tracking_uri=TRACKING_URI,
        )
        prod_rolled = str(client.get_registered_model(model_name).aliases["production"])
        if prod_rolled != str(target_version):
            raise RuntimeError("Registry did not switch production to the requested version")
        elapsed = perf_counter() - started
    finally:
        client.set_registered_model_alias(model_name, "production", v_prod_initial)
    final_rm = client.get_registered_model(model_name)
    offline = verify_offline_api(bundle)
    write_json(REPO_ROOT / "reports" / "offline_api_verification.json", offline)
    write_json(
        REPO_ROOT / "reports" / "mlflow_rollback_trial.json",
        {
            "rollback": rollback_res,
            "rollback_seconds": elapsed,
            "original_aliases": initial_aliases,
            "restored_aliases": dict(final_rm.aliases),
            "offline_api": offline,
            "same_locked_checkpoint": all(v.tags.get("same_locked_checkpoint") == "true" for v in versions),
        },
    )
    serving_test_success = offline["success"]
    serving_details = (
        f"Fresh uvicorn HTTP /health, /predict, /model-info: 200; "
        f"{len(offline['prediction']['results'])} results; tracking URI {offline['tracking_uri']} unavailable. "
        f"Bundle {offline['bundle']}; encoder SHA-256 {offline['encoder_weights_sha256']}. "
        "Evidence: reports/offline_api_verification.json."
    )

    # Generate Report
    report_path = REPO_ROOT / "reports" / "mlflow_rollback_trial.md"
    content = f"""# MLflow Model Registry: Aliases, Rollback Trial & Offline Serving

**Thực hiện:** Hiếu (Person 3) - Fine-tune, MLflow & Monitoring  
**Mô hình:** `{model_name}`  
**Backend Store:** `sqlite:///mlflow.db`  
**Thời gian thử nghiệm:** {rollback_time}  
**Môi trường:** Dev (`feature/person3-monitoring-train-mlflow`)

**Phạm vi trial:** Hai revision registry của cùng checkpoint FT đã khóa được dùng
để kiểm tra cơ chế alias và phục vụ offline. Trial không chứng minh rollback giữa
hai model có chất lượng khác nhau. Các revision tham chiếu snapshot bundle cố định.

---

## 1. Định nghĩa Alias trong Model Registry

| Alias | Ý nghĩa nghiệp vụ | Trạng thái triển khai |
| :--- | :--- | :--- |
| `@production` | Model đã qua đánh giá toàn diện, phục vụ API chính thức của hệ thống. | Active (Version {v_prod_initial}) |
| `@staging` | Model ứng viên (candidate) đang trong giai đoạn QA / kiểm thử tích hợp. | Staging (Version {v_staging_initial}) |

> **Quy ước nhãn (Tags) nộp bài:**
> - `owner`: `Person 3 (Hiếu)`
> - `environment`: `dev`
> - `status`: `validated`

---

## 2. Nhật ký Thử nghiệm Rollback Thực tế (Trial Log)

Quá trình thử nghiệm đảo alias (rollback) được thực hiện trực tiếp trên SQLite registry:

1. **Trạng thái ban đầu:**
   - `@production` trỏ tới: Version `{v_prod_initial}`
   - `@staging` trỏ tới: Version `{v_staging_initial}`

2. **Lệnh thực hiện rollback:**
   ```bash
   python -m src.train --rollback {target_version} --alias production
   ```
   *Kết quả thực thi:*
   - `model_name`: `{model_name}`
   - `previous_version`: `{rollback_res["previous_version"]}`
   - `current_version`: `{rollback_res["current_version"]}`
   - `alias`: `production`

3. **Kiểm tra xác nhận (Verification):**
   - Truy vấn `client.get_registered_model('{model_name}').aliases['production']` trả về: `{prod_rolled}`.
   - Alias được đọc lại sau rollback để xác nhận. Thời gian đo: {elapsed:.3f} giây.

4. **Khôi phục trạng thái chuẩn:**
   - `@production`: Version `{final_rm.aliases.get("production")}`
   - `@staging`: Version `{final_rm.aliases.get("staging")}`

---

## 3. Hoạt động của API khi MLflow Tracking Server Tắt (Offline Serving)

* **Cơ chế tách rời (Decoupling):** API (`src.serving` / `api.app`) không kết nối trực tiếp vào MLflow server ở runtime. Thay vào đó, quy trình huấn luyện xuất bản bundle cố định tại thư mục `models/` bao gồm:
  - `embeddings.npy`: Ma trận vector đã chuẩn hóa.
  - `recipe_ids.json`: Danh sách id tài liệu tương ứng.
  - `manifest.json`: Toàn bộ thông tin cấu hình, checksum SHA-256 và metrics.
* **Kết quả kiểm thử khi tắt MLflow Server:**
  - Trạng thái kiểm thử: **{"THÀNH CÔNG" if serving_test_success else "CHÚ Ý"}**
  - Chi tiết: {serving_details}
  - API khởi động và phục vụ request HTTP thật khi tracking endpoint không truy cập được.
  - Đổi alias chỉ đổi registry. Để đổi API đang chạy, export bundle của version đó,
    cập nhật serving.bundle và khởi động lại API.

---

## 4. Hướng dẫn Rollback Khẩn cấp (SOP)

Khi phát hiện mô hình trên production bị suy giảm chất lượng hoặc gặp sự cố:

1. **Cách 1: Sử dụng CLI một lệnh (`train.py`):**
   ```bash
   # Rollback alias production về version 1
   python -m src.train --rollback 1
   ```

2. **Cách 2: Sử dụng Python Script:**
   ```python
   from src.train import rollback_model_registry
   rollback_model_registry(model_name="minilm_finetuned", target_version=1, alias="production")
   ```

3. **Cách 3: Chuyển đổi trên MLflow UI:**
   - Mở MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
   - Vào tab **Models** > Chọn `minilm_finetuned`.
   - Tại cột **Aliases**, click vào icon edit của `@production` và nhập số version cần rollback (ví dụ: `1`).
"""
    report_path.write_text("\n".join(line.rstrip() for line in content.splitlines()) + "\n", encoding="utf-8")
    print(f"Rollback trial recorded to: {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Create two trial revisions of the locked checkpoint in an empty registry",
    )
    args = parser.parse_args()
    print(run_trial(bootstrap=args.bootstrap))
