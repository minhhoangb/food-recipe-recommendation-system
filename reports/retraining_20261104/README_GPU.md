# Lần train mới ngày 04 tháng 11 năm 2026

Đã chuyển môi trường `D:/MDS301/food-recipe-recommendation-system/.venv` sang `torch 2.9.1+cu128` và xác nhận CUDA hoạt động trên RTX 3060 Laptop 6 GB. `pip check` không báo dependency hỏng; toàn bộ 302 test của dự án pass.

Bảy cấu hình được fit, lập chỉ mục hoặc mã hóa lại. MiniLM được fine-tune từ pretrained trên 30.000 cặp train, 1 epoch, batch 32, learning rate 2e-5, weight decay 0.01, seed 42. Fine-tune mất 126.791 giây, CUDA allocated cao nhất 1244.13 MB. Model dùng giới hạn 256 token trong train; mã hóa và truy hồi dùng 128 token theo code hiện tại.

| Cấu hình | MRR@10 validation mới |
|---|---:|
| TF-IDF | 0.6567388889 |
| TF-IDF đã chọn | 0.6567388889 |
| BM25 | 0.7583134921 |
| LSA | 0.3628523810 |
| MiniLM nguyên liệu | 0.4331531746 |
| MiniLM thêm tiêu đề | 0.1518833333 |
| MiniLM fine-tune | 0.6228761905 |

Đánh giá dùng 500 query validation và 50.000 ứng viên, top-k 10, không rerank. Đây là benchmark truy tìm công thức gốc bằng qrels tổng hợp. Không chạy lại E10; không gán metric test cũ cho checkpoint mới. Chưa thay cấu hình hoặc bundle production.

Thí nghiệm weight decay mới nằm ở `regularization_ablation/summary.json`. Hai nhánh có validation loss cuối 0.0261550910 và 0.0261529131. Chênh lệch nhỏ, một seed, chưa chứng minh cải thiện relevance có ý nghĩa.

## Mở lịch sử MLflow mới

Trong thư mục repo, dùng môi trường `.venv` và chạy:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///D:/MDS301/outputs/retrain-20261104/mlflow.db --port 5051
```

Các experiment mới là `recipe-retraining-20261104-gpu` và `recipe-regularization-ablation`. Database cũ trong repo được giữ nguyên. Danh sách package thực tế nằm ở `requirements-gpu-freeze.txt`.

## Kiểm tra GPU

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Checkpoint mới nằm ở `models/minilm_finetuned-checkpoint`. Bundle FT đóng gói cả encoder nằm ở `models/minilm_finetuned-E04-main-1f1f4ce5`. Bảng đầy đủ là `comparison_validation.csv`; từng dòng có run ID. Script `run_retraining.py` từ chối ghi đè thư mục model đã tồn tại. Muốn chạy thêm một lần, sao chép script vào một thư mục đầu ra mới rồi chạy bằng Python CUDA của repo.
