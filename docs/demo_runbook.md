# Demo runbook — 5 phút

Bài thuyết trình của Hưng (Person 2): Docker từ clone sạch, API, UI và kiểm chứng.

Mọi lệnh trong runbook này **đã chạy thật** trên máy dự án (Windows, PowerShell). Trên
Linux/macOS thay `set X=...` bằng `export X=...` và dùng `\` thay vì dấu gạch nối dòng.

Bố cục 5 phút:

| Thời gian | Bước | Mục đích |
| --- | --- | --- |
| 0:00-0:45 | A. Dựng container từ clone sạch | Docker chạy được, health OK |
| 0:45-2:15 | B. Dựng đồ thị: health → predict → input sai → batch → model-info | API trả lời đúng, có kiểm chứng |
| 2:15-3:00 | C. Giao diện web | UI dùng được, không cần thêm gì |
| 3:00-3:45 | D. Log và monitoring | thấy được request log thật |
| 3:45-4:30 | E. Offline backup | demo không phụ thuộc mạng |
| 4:30-5:00 | F. Smoke test + bug đã sửa | đóng gói bằng chứng |

---

## A. Dựng container từ clone sạch (0:45)

Trên một máy sạch (không có `models/` hay `data/processed/`):

```bash
git clone https://github.com/minhhoangb/food-recipe-recommendation-system.git
cd food-recipe-recommendation-system

# dữ liệu + bundle (không có sẵn trong git)
python scripts/fetch_artifacts.py                 # corpus đã xử lý
python -m src.embeddings --config configs/default.yaml --mode ingredients

# build + chạy
docker compose up --build
```

**Cần biết trước khi lên sân khấu:** lệnh `docker compose up --build` mất khoảng
3-6 phút lần đầu (tải torch CPU ~200 MB). Nếu trình bày tại chỗ, **build trước ở nhà**
và chỉ `docker compose up` trên sân khấu.

Những gì `docker compose up --build` làm, theo thứ tự:

1. `init-bundles` copy `models/` của máy vào volume `bundles` (nếu volume đã có thì thôi).
2. `api` khởi động **sau khi** copy xong, đọc bundle từ volume ở chế độ read-only.
3. `data/` bind-mount read-only; request log nằm trong volume `api_logs` nên không mất khi
   `docker compose down`.

Chứng minh nó chạy:

```bash
docker compose ps              # STATUS phải là "healthy", không phải "starting"
curl http://127.0.0.1:8000/health
```

Kết quả thật:

```json
{"status":"ok","model":"minilm_pretrained","model_version":"1"}
```

> Nếu `STATUS` kẹt ở `starting`, xem `docker compose logs api`: thường là do bundle chưa
> có trên máy hoặc bundle khác corpus với `data/processed/` (API cố ý không khởi động
> để không trả về sai món).

---

## B. Dựng đồ thị trên API (1:30)

### B1. Health

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok","model":"minilm_pretrained","model_version":"1"}
```

### B2. Model-info — model nào đang phục vụ, chấm điểm ra sao

```bash
curl http://127.0.0.1:8000/model-info
```

```json
{
  "model": "minilm_pretrained",
  "model_version": "1",
  "backend": "dense",
  "config": "configs/production.yaml",
  "corpus_recipes": 50000,
  "bundle": "models/minilm_pretrained-E04-main-1f1f4ce5",
  "corpus_sha256": "1f1f4ce50525727ba5242972dac8c93dc32e9555207cbf8b4ebb0f31f0409b46",
  "metrics": {"recall@5": 0.292, "recall@10": 0.38, "mrr@10": 0.22871746031746035, "n_queries": 500.0}
}
```

Điểm cần nói: `corpus_sha256` là bằng chứng rằng bundle và corpus là của **cùng một**
bộ dữ liệu. API từ chối khởi động nếu hai bên lệch nhau.

### B3. Predict — câu hỏi thật

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["chicken", "rice", "garlic"], "top_k": 2}'
```

```json
{
  "model": "minilm_pretrained",
  "model_version": "1",
  "latency_ms": 845.99,
  "results": [
    {
      "recipe_id": "recipenlg:7642",
      "title": "Chicken And Rice",
      "score": 0.8863415718078613,
      "matched_ingredients": ["chicken", "rice", "garlic"],
      "missing_ingredients": ["onion", "salt"],
      "directions": ["Boil chicken with skin on until done...", "Cut up onion put into broth.", "Add minced garlic; cook for 5 minutes."],
      "link": "https://www.cookbooks.com/Recipe-Details.aspx?id=709588"
    },
    {
      "recipe_id": "recipenlg:1553265",
      "title": "Tastier Brown Rice Pilaf",
      "score": 0.8497336506843567,
      "matched_ingredients": ["chicken", "garlic"],
      "missing_ingredients": ["butter", "onion", "brown rice", "salt"],
      "directions": ["Melt butter in a large sauce pan..."],
      "link": "https://tastykitchen.com/recipes/sidedishes/tastier-brown-rice-pilaf/"
    }
  ],
  "note": null
}
```

Điểm cần nói: `matched_ingredients` là nguyên liệu người dùng **đã có**,
`missing_ingredients` là nguyên liệu **còn thiếu** — đây là điều người dùng cần để đi
chợ. `score` là điểm cosine, không phải xác suất.

### B4. Input sai — API từ chối rõ ràng (quan trọng)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ingredients": [], "top_k": 3}'
```

```json
{"detail":[{"type":"too_short","loc":["body","ingredients"],"msg":"List should have at least 1 item after validation, not 0","input":[],"ctx":{"field_type":"List","min_length":1,"actual_length":0}}]}
```

HTTP status là **422**, không phải 500. Các trường hợp khác cũng trả 422: `top_k` ngoài
khoảng `[1, 20]`, danh sách nguyên liệu toàn khoảng trắng, thừa trường không hợp lệ.

### B5. Batch — giữ nguyên thứ tự input

```bash
curl -X POST http://127.0.0.1:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"ingredients": ["chicken", "rice"], "top_k": 1}, {"ingredients": ["tomato", "basil"], "top_k": 1}]}'
```

```json
{
  "predictions": [
    {"model": "minilm_pretrained", "model_version": "1", "latency_ms": 203.56,
     "results": [{"recipe_id": "recipenlg:2199670", "title": "Shaun the Sheep Character Bento", "score": 0.7940338850021362,
                  "matched_ingredients": ["rice"], "missing_ingredients": ["cheese"],
                  "directions": ["Make 2 bento sized rice balls."],
                  "link": "https://cookpad.com/us/recipes/151775-shaun-the-sheep-character-bento"}],
     "note": null},
    {"model": "minilm_pretrained", "model_version": "1", "latency_ms": 128.63,
     "results": [{"recipe_id": "recipenlg:2116245", "title": "My Best Sandwich Ever", "score": 0.845429003238678,
                  "matched_ingredients": ["tomato", "basil"], "missing_ingredients": ["salad"],
                  "directions": ["On a frying pan, cook the steak a few minutes on each side..."],
                  "link": "https://tastykitchen.com/recipes/main-courses/my-best-sandwich-ever/"}],
     "note": null}
  ]
}
```

Điểm cần nói: `predictions[i]` ứng với `instances[i]` — thứ tự được giữ, không bị đảo.

### B6. Trang /docs của FastAPI

Mở http://127.0.0.1:8000/docs trong trình duyệt: Swagger UI với schema thật của từng
endpoint. Bấm "Try it out" ngay trên trang để gọi API mà không cần gõ curl.

---

## C. Giao diện web (0:45)

Mở http://127.0.0.1:8000/

- Gõ nguyên liệu vào ô, chọn số kết quả, bấm **Search**.
- Mỗi món hiện tên, nguyên liệu **có** / **thiếu**, nút mở *directions*, và link tới nguồn.
- Link truyền tham số: http://127.0.0.1:8000/?q=chicken,rice,garlic&k=3 — mở là tự tìm.

Ảnh chụp màn hình (390 px và 1440 px, kết quả thật từ API) nằm ở
`reports/figures/ui_390.png` và `reports/figures/ui_1440.png`. Chụp lại bằng:

```bash
python scripts/shoot_ui.py --url http://127.0.0.1:8000/ --ingredients "chicken, rice, garlic"
```

---

## D. Log và monitoring (0:45)

Mọi request đều được ghi vào SQLite (`logs/requests.db` trong container, volume
`api_logs`):

```bash
docker compose exec api python -c "
import sqlite3
with sqlite3.connect('/app/logs/requests.db') as c:
    for row in c.execute('SELECT timestamp_utc, endpoint, status_code, latency_ms FROM request_log ORDER BY timestamp_utc DESC LIMIT 5'):
        print(row)
"
```

```text
('2026-10-30T14:04:07Z', '/predict', 200, 845.99)
('2026-10-30T14:04:02Z', '/batch-predict', 200, 203.56)
('2026-10-30T14:03:58Z', '/predict', 422, 0.0)
```

Đọc log ra báo cáo monitoring (trên máy, không cần trong container):

```bash
python -m src.monitor --log-path logs/requests.db --out reports/monitoring_report.md
```

Báo cáo có: tổng số request, tỉ lệ lỗi, phân vối latency (p50/p90/p95/p99), tỉ lệ
query không ra kết quả, và phân rã theo endpoint.

---

## E. Backup để dùng offline (0:45)

Bản sao lưu nằm ngoài thư mục repo, trong OneDrive nên tự đồng bộ lên cloud:

```text
C:\Users\tuanh\OneDrive\Desktop\MDS_FINAL\artifacts_backup\
  models/minilm_pretrained-E04-main-1f1f4ce5/   embeddings.npy, recipe_ids.json, manifest.json
  processed/main/                               corpus.jsonl, splits.json, queries.jsonl, qrels.jsonl, manifest
  MANIFEST.txt                                  sha256 của từng file
```

Tổng 137 MB. Tạo lại hoặc kiểm tra:

```bash
python scripts/backup_artifacts.py --out "C:\Users\tuanh\OneDrive\Desktop\MDS_FINAL\artifacts_backup"
python scripts/backup_artifacts.py --out "C:\Users\tuanh\OneDrive\Desktop\MDS_FINAL\artifacts_backup" --verify
```

Kết quả `--verify`:

```text
the backup in C:\Users\tuanh\OneDrive\Desktop\MDS_FINAL\artifacts_backup matches the working copy (8 files)
```

**Offline nghĩa là gì:** image Docker đã bake sẵn NLTK resource của ingredient-parser và
trọng số MiniLM, nên container khởi động được khi không có mạng. Corpus và bundle
lấy từ backup ở trên. Nếu mạng chết giữa chừng, demo vẫn chạy.

---

## F. Smoke test và các lỗi đã sửa (0:30)

Smoke test end-to-end (pytest + httpx) chạy với bundle thật:

```bash
python -m pytest tests/test_smoke_e2e.py -v
```

```text
tests/test_smoke_e2e.py::test_health_is_ok PASSED
tests/test_smoke_e2e.py::test_model_info_describes_the_serving_bundle PASSED
tests/test_smoke_e2e.py::test_predict_returns_ranked_recipes PASSED
tests/test_smoke_e2e.py::test_batch_predict_keeps_the_order_of_the_input PASSED
tests/test_smoke_e2e.py::test_a_bad_request_is_answered_with_422 PASSED
tests/test_smoke_e2e.py::test_the_ui_is_served PASSED
============================== 6 passed, 2 warnings in 25.32s ==============================
```

Các lỗi đã tìm ra và sửa trong sprint này (chi tiết trong PR):

1. **API phục vụ bundle của corpus khác.** `configs/pilot.yaml` là config mặc định cũ
   (corpus pilot, 5.000 món) nhưng `serving.bundle` trỏ tới bundle `main` (50.000 món).
   Vì id của kết quả là **vị trí** trong corpus của bundle, app trả về món sai mà không
   báo lỗi. Sửa: `src/serving.py` có `check_bundle_corpus()`, và API gọi nó trước khi phục
   vụ; app không khởi động được nếu lệch. Thêm test
   `test_a_bundle_of_another_corpus_does_not_start`.
2. **`src.monitor` đọc log sai định dạng.** API ghi SQLite (`logs/requests.db`) còn
   monitor đọc JSONL (`logs/requests.jsonl`), nên báo cáo monitoring luôn rỗng. Sửa:
   `load_request_logs()` nhận cả SQLite lẫn JSONL; mặc định đổi sang `logs/requests.db`.
3. **Docker mount sai trên clone sạch.** `./models` bind-mount nghĩa là clone sạch không có
   `models/` thì container không có bundle. Sửa: volume `bundles` + service `init-bundles`
   copy từ `./models` trước khi `api` khởi động.
4. **Bundle không có trên máy thì API crash không giải thích.** Sửa: thông báo rõ phải
   build bundle bằng lệnh nào, hoặc dùng `API_BACKEND=reference`.

---

## Phụ lục: chạy không cần Docker

Khi sân khấu không có mạng hoặc Docker hỏng:

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Cùng mọi lệnh ở phần B, C, D. `API_CONFIG` mặc định là `configs/production.yaml`
(bundle E04 + corpus main). Thêm `--reload` khi cần sửa code.
