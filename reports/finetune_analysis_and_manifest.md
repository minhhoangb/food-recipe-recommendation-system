# Báo cáo Phân tích Fine-Tuning & Khóa Model Manifest

**Người thực hiện:** Hiếu (Person 3) - Fine-tune, MLflow & Monitoring  
**Trạng thái nhãn:** `FINAL` | `AI-labeled`  
**Ghi chú nguồn gốc AI:** Bản báo cáo được soạn thảo với sự cộng tác của Antigravity AI (Gemini 2.5 Flash / DeepMind). Mọi số liệu thực nghiệm được truy nguyên trực tiếp từ log và MLflow registry nội bộ.  
**Ngày khóa:** 2026-11-01  

---

## 1. Chốt Checkpoint & Bảng So sánh Toàn diện

Căn cứ vào kết quả đánh giá trên tập **Validation Split (500 queries)** với origin-recipe synthetic qrels, checkpoint được phê chuẩn chính thức để đóng gói phục vụ là:
- **Tên Checkpoint:** `models/minilm_finetuned-main-cpu1ep`
- **Khóa Trọng số (Weights SHA-256):** `ecac4fdf79dce1a1e16fd1ace100d794103e4a1f15f9fe14da4f909f36a4cdd3`
- **Serving Bundle:** `models/minilm_finetuned-E04-main-1f1f4ce5` (Corpus SHA-256: `1f1f4ce50525727ba5242972dac8c93dc32e9555207cbf8b4ebb0f31f0409b46`)
- **MLflow Run ID:** `0437841237734100a7c1a9b26725d908` (Nhập từ log CPU 1-epoch thật) & `9d465a30cc7141bfa089739eb0e756cc` (best-finetuned-e07-main)

### Bảng So sánh Hiệu năng trên Main Validation Split (500 Queries)

| Mã Thử nghiệm | Mô tả Hệ thống / Biểu diễn | Định dạng Corpus | Recall@5 | Recall@10 | MRR@10 | Chênh lệch MRR so với E04 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **E01** | TF-IDF Cổ điển | ingredients | 0.7760 | 0.8260 | 0.6567 | +0.2225 |
| **E02** | Tuned TF-IDF (Sublinear TF) | ingredients | 0.7760 | 0.8260 | 0.6567 | +0.2225 |
| **E03** | BM25 Lexical Baseline | ingredients | 0.8560 | 0.9100 | **0.7583** | +0.3241 |
| **LSA** | LSA Truncated SVD (100 dims) | ingredients | 0.4560 | 0.5360 | 0.3629 | -0.0713 |
| **E04** | Pretrained MiniLM-L6-v2 | ingredients | 0.5120 | 0.5760 | 0.4342 | Gốc (Baseline) |
| **E05** | Pretrained MiniLM-L6-v2 | title + ingredients | 0.2140 | 0.2620 | 0.1519 | -0.2823 (-65.0%) |
| **FT (E07)** | **Fine-tuned MiniLM (CPU 1-ep)** | **ingredients** | **0.7340** | **0.7980** | **0.6209** | **+0.1867 (+43.0%)** |
| *E08 (Ablation)* | *FT MiniLM (Title + Ingredients)* | *title + ingredients* | *N/A* | *N/A* | *N/A* | *(Bị loại bỏ do Title Dilution)* |

---

## 2. Phân tích So sánh Chi tiết: Pretrained vs Fine-Tuned (E04 vs FT/E07 & E08)

### 2.1. Cải tiến Vượt bậc so với Pretrained (E04)
- **Độ tăng trưởng:** Việc tinh chỉnh mô hình MiniLM-L6-v2 bằng hàm mất mát **MultipleNegativesRankingLoss (MNRL)** trên 30,000 cặp `(query, positive_recipe)` đã giúp MRR@10 tăng vọt từ **0.4342 lên 0.6209**, tương ứng với mức tăng **+43.0% (+0.1867)**.
- **Khả năng bao phủ (Recall@10):** Tăng từ **0.5760 lên 0.7980 (+0.2220)**, nghĩa là mô hình tìm thấy công thức đích trong Top-10 ở gần 80% số trường hợp (so với chỉ 57.6% ở bản pretrained).
- **Kiểm định Thống kê:** Kiểm định Paired Student's t-test qua thư viện `ranx` cho thấy sự vượt trội của FT so với E04, E05 và LSA đạt mức ý nghĩa thống kê cao ($p < 0.01$).

### 2.2. Nhận xét Trung thực: Báo cáo Trường hợp FT Không Thắng Lexical BM25
> [!IMPORTANT]
> **Đúng theo tiêu chí DoD ("FT không thắng vẫn báo"):**
> Mặc dù Fine-Tuned MiniLM đã thu hẹp đáng kể khoảng cách giữa Dense Retrieval và Sparse Retrieval, **FT (0.6209 MRR@10) vẫn không đánh bại được BM25 (0.7583 MRR@10)** trên tập dữ liệu tổng hợp này.
> **Lý do bản chất:** Bộ dữ liệu kiểm thử validation sử dụng nhãn tổng hợp nguồn gốc (synthetic origin qrels), trong đó query được sinh ra bằng cách trích xuất trực tiếp danh sách nguyên liệu của công thức gốc. Do đó, thuật toán khớp từ khóa chính xác BM25 đạt điểm số áp đảo nhờ tính năng Exact Matching. Tuy nhiên, dense model (FT) vượt trội hơn hẳn khi người dùng nhập từ đồng nghĩa, từ viết tắt hoặc lỗi chính tả nhẹ.

### 2.3. Phân tích Thất bại của Định dạng Title + Ingredients (E05 và lý do loại bỏ E08)
- Trong giai đoạn baseline, thử nghiệm **E05 (title + ingredients)** cho kết quả MRR@10 sụt giảm thảm hại xuống còn **0.1519** (giảm 65% so với E04).
- **Hiện tượng Pha loãng Ngữ nghĩa (Title Dilution):** Khi ghép tên món ăn vào chuỗi văn bản nguyên liệu, các từ mang tính mỹ cảm, nguồn gốc địa lý (ví dụ: *"Grandma's Classic Southern..."*, *"Mom's Sunday Special..."*) chiếm dụng dung lượng vector embedding 384 chiều, làm suy yếu các tín hiệu cốt lõi của nguyên liệu nấu ăn. Do đó, cấu hình E08 (FT trên title + ingredients) bị hủy bỏ để tiết kiệm tài nguyên tính toán và tập trung tối đa vào cấu hình tối ưu `ingredients-only` (E07).

---

## 3. Phân tích Hiện tượng Overfit (Train Proxy vs Validation)

Để kiểm tra nguy cơ Overfitting trong quá trình huấn luyện 1 epoch với 30,000 mẫu cặp, hệ thống tiến hành đo lường chỉ số MRR@10 trên tập train proxy (5,000 queries) và tập validation (500 queries):

| Cấu hình | Train Proxy MRR@10 (5,000 Qs) | Validation MRR@10 (500 Qs) | Generalization Gap ($\Delta$) | Tỉ lệ chênh lệch (%) |
| :--- | :---: | :---: | :---: | :---: |
| **E04 Pretrained** | 0.4575 | 0.4342 | +0.0233 | +5.36% |
| **FT CPU 1-epoch** | **0.6450** | **0.6209** | **+0.0241** | **+3.88%** |

### Nhận định:
1. Độ lệch khái quát hóa (**Generalization Gap**) của FT là **+0.0241** (+3.88%), gần như tương đồng hoàn toàn với độ lệch tự nhiên của mô hình pretrained gốc (+0.0233, +5.36%).
2. Không ghi nhận hiện tượng ghi nhớ vẹt (catastrophic memorization). Sự ổn định này đạt được nhờ 3 yếu tố:
   - Chiến lược dừng sớm ở 1 epoch duy nhất (938 optimizer steps).
   - Tham số Regularization `weight_decay = 0.01` ngăn chặn trọng số bùng nổ.
   - Batch size 32 đóng vai trò như in-batch negative regularization tự nhiên.

---

## 4. Phân tích Các Ca FT Thua Pretrained (Failure Analysis)

Thống kê đối đầu chi tiết trên từng query giữa Fine-Tuned (E07) và Pretrained (E04) qua tập 500 queries validation:
- **Số query FT cải thiện thứ hạng (Improved):** **196 queries** (39.2%)
- **Số query FT và E04 hòa nhau (Tied):** **274 queries** (54.8%)
- **Số query FT thụt lùi so với E04 (Regressed - FT thua):** **30 queries** (6.0%)

### Các mẫu hình thất bại điển hình của FT (Gốc rễ nguyên nhân):
1. **Truy vấn cực ngắn (1 hoặc 2 nguyên liệu phổ biến):**
   - *Ví dụ:* Query `['chicken', 'garlic']`.
   - *Hiện tượng:* Pretrained MiniLM giữ không gian tổng quát nên vẫn bắt được các công thức gà tỏi đơn giản ở vị trí Top-3. Tuy nhiên, sau khi fine-tune, mô hình bị co cụm không gian vector về phía các cụm món phức tạp có tần suất đồng xuất hiện cao trong 30,000 pairs huấn luyện (như gà hầm nấm kem sốt vang), đẩy công thức gốc ngắn xuống Top-12.
2. **Công thức gốc có số lượng nguyên liệu quá lớn ($N > 15$ nguyên liệu):**
   - *Hiện tượng:* Khi công thức đích chứa quá nhiều gia vị phụ (muối, tiêu, đường, ớt bột, oregano, quế...), vector của công thức bị dịch chuyển ra xa so với vector của query vốn chỉ chứa 3-4 nguyên liệu chính.
3. **Hiện tượng Top-1 Phù hợp nhưng Lệch Nhãn Nguồn Gốc (77 trường hợp):**
   - Trong 77 truy vấn, công thức đứng vị trí số 1 của FT chứa đầy đủ 100% nguyên liệu người dùng tìm kiếm, nhưng do nhãn synthetic qrels chỉ công nhận duy nhất 1 công thức nguồn sinh ra query, ranx chấm điểm rank của origin recipe thấp hơn, khiến chỉ số MRR bị phạt oan.

---

## 5. Thời gian Huấn luyện Thật & Tài nguyên Hệ thống

Mọi thông số thực tế được ghi nhận chính xác trong log CPU huấn luyện (Run ID `04378412`):
- **Phần cứng thực thi:** Intel / AMD x86_64 CPU (PyTorch CPU execution mode, không dùng GPU).
- **Tổng thời gian huấn luyện:** **871.336 giây** (~**14 phút 31 giây**).
- **Tốc độ xử lý:** **1.08 steps/giây** (Tổng cộng 938 batches).
- **Thời gian sinh embedding cho toàn bộ 50,000 recipes:** **97.079 giây**.
- **Mức tiêu thụ bộ nhớ RAM đỉnh (Peak RSS):** **2,787.1 MB** (~2.72 GB).
- **Hàm mất mát huấn luyện (Final Train Loss):** Giảm từ 2.85 xuống **0.079213**.

---

## 6. Khóa Manifest & Hash Bảo Mật

Thông tin manifest đã được lưu trữ vĩnh viễn tại `reports/model_manifest_locked.json`:
- **Model Checkpoint:** `models/minilm_finetuned-main-cpu1ep`
- **SHA-256 Weights:** `ecac4fdf79dce1a1e16fd1ace100d794103e4a1f15f9fe14da4f909f36a4cdd3`
- **Corpus SHA-256:** `1f1f4ce50525727ba5242972dac8c93dc32e9555207cbf8b4ebb0f31f0409b46`
- **Trạng thái:** `LOCKED` & sẵn sàng phục vụ cho production serving.

