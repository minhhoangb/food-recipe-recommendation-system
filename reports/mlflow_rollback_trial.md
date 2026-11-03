# MLflow Model Registry: Aliases, Rollback Trial & Offline Serving

**Thực hiện:** Hiếu (Person 3) - Fine-tune, MLflow & Monitoring
**Mô hình:** `minilm_finetuned`
**Backend Store:** `sqlite:///mlflow.db`
**Thời gian thử nghiệm:** 2026-11-03 05:49:18 UTC
**Môi trường:** Dev (`feature/person3-monitoring-train-mlflow`)

**Phạm vi trial:** Hai revision registry của cùng checkpoint FT đã khóa được dùng
để kiểm tra cơ chế alias và phục vụ offline. Trial không chứng minh rollback giữa
hai model có chất lượng khác nhau. Các revision tham chiếu snapshot bundle cố định.

---

## 1. Định nghĩa Alias trong Model Registry

| Alias | Ý nghĩa nghiệp vụ | Trạng thái triển khai |
| :--- | :--- | :--- |
| `@production` | Model đã qua đánh giá toàn diện, phục vụ API chính thức của hệ thống. | Active (Version 2) |
| `@staging` | Model ứng viên (candidate) đang trong giai đoạn QA / kiểm thử tích hợp. | Staging (Version 1) |

> **Quy ước nhãn (Tags) nộp bài:**
> - `owner`: `Person 3 (Hiếu)`
> - `environment`: `dev`
> - `status`: `validated`

---

## 2. Nhật ký Thử nghiệm Rollback Thực tế (Trial Log)

Quá trình thử nghiệm đảo alias (rollback) được thực hiện trực tiếp trên SQLite registry:

1. **Trạng thái ban đầu:**
   - `@production` trỏ tới: Version `2`
   - `@staging` trỏ tới: Version `1`

2. **Lệnh thực hiện rollback:**
   ```bash
   python -m src.train --rollback 1 --alias production
   ```
   *Kết quả thực thi:*
   - `model_name`: `minilm_finetuned`
   - `previous_version`: `2`
   - `current_version`: `1`
   - `alias`: `production`

3. **Kiểm tra xác nhận (Verification):**
   - Truy vấn `client.get_registered_model('minilm_finetuned').aliases['production']` trả về: `1`.
   - Alias được đọc lại sau rollback để xác nhận. Thời gian đo: 0.070 giây.

4. **Khôi phục trạng thái chuẩn:**
   - `@production`: Version `2`
   - `@staging`: Version `1`

---

## 3. Hoạt động của API khi MLflow Tracking Server Tắt (Offline Serving)

* **Cơ chế tách rời (Decoupling):** API (`src.serving` / `api.app`) không kết nối trực tiếp vào MLflow server ở runtime. Thay vào đó, quy trình huấn luyện xuất bản bundle cố định tại thư mục `models/` bao gồm:
  - `embeddings.npy`: Ma trận vector đã chuẩn hóa.
  - `recipe_ids.json`: Danh sách id tài liệu tương ứng.
  - `manifest.json`: Toàn bộ thông tin cấu hình, checksum SHA-256 và metrics.
* **Kết quả kiểm thử khi tắt MLflow Server:**
  - Trạng thái kiểm thử: **THÀNH CÔNG**
  - Chi tiết: Fresh uvicorn HTTP /health, /predict, /model-info: 200; 3 results; tracking URI http://127.0.0.1:1 unavailable. Bundle minilm_finetuned-E04-main-1f1f4ce5; encoder SHA-256 ecac4fdf79dce1a1e16fd1ace100d794103e4a1f15f9fe14da4f909f36a4cdd3. Evidence: reports/offline_api_verification.json.
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
