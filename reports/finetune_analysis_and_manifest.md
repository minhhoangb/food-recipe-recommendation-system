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
- **MLflow training run:** `0437841237734100a7c1a9b26725d908` (nhập từ log CPU 1-epoch thật); **validation evaluation run:** `0444aa538a9948f8aecc25d5c42bb365` theo `comparison.csv`. FT main là **E08**; E07 là pilot.

### Bảng So sánh Hiệu năng trên Main Validation Split (500 Queries)

| Mã Thử nghiệm | Mô tả Hệ thống / Biểu diễn | Định dạng Corpus | Recall@5 | Recall@10 | MRR@10 | Chênh lệch MRR so với E04 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **E01** | TF-IDF Cổ điển | ingredients | 0.7760 | 0.8260 | 0.6567 | +0.2225 |
| **E02** | Tuned TF-IDF (GroupKFold CV5) | ingredients | 0.7760 | 0.8260 | 0.6567 | +0.2225 |
| **E03** | BM25 Lexical Baseline | ingredients | 0.8560 | 0.9100 | **0.7583** | +0.3241 |
| **LSA** | LSA Truncated SVD (256 dims) | ingredients | 0.4560 | 0.5360 | 0.3629 | -0.0713 |
| **E04** | Pretrained MiniLM-L6-v2 | ingredients | 0.5120 | 0.5760 | 0.4342 | Gốc (Baseline) |
| **E05** | Pretrained MiniLM-L6-v2 | title + ingredients | 0.2140 | 0.2620 | 0.1519 | -0.2823 (-65.0%) |
| **FT (E08)** | **Fine-tuned MiniLM (CPU 1-ep)** | **ingredients** | **0.7340** | **0.7980** | **0.6209** | **+0.1867 (+43.0%)** |

---

## 2. Phân tích So sánh Chi tiết: Pretrained vs Fine-Tuned (E04 vs FT/E08)

### 2.1. Cải tiến Vượt bậc so với Pretrained (E04)
- **Độ tăng trưởng:** Việc tinh chỉnh mô hình MiniLM-L6-v2 bằng hàm mất mát **MultipleNegativesRankingLoss (MNRL)** trên 30,000 cặp `(query, positive_recipe)` đã giúp MRR@10 tăng vọt từ **0.4342 lên 0.6209**, tương ứng với mức tăng **+43.0% (+0.1867)**.
- **Khả năng bao phủ (Recall@10):** Tăng từ **0.5760 lên 0.7980 (+0.2220)**, nghĩa là mô hình tìm thấy công thức đích trong Top-10 ở gần 80% số trường hợp (so với chỉ 57.6% ở bản pretrained).
- **Kiểm định Thống kê:** Kiểm định Paired Student's t-test qua thư viện `ranx` cho thấy sự vượt trội của FT so với E04, E05 và LSA đạt mức ý nghĩa thống kê cao ($p < 0.01$).

### 2.2. Nhận xét Trung thực: Báo cáo Trường hợp FT Không Thắng Lexical BM25
> [!IMPORTANT]
> **Đúng theo tiêu chí DoD ("FT không thắng vẫn báo"):**
> Mặc dù Fine-Tuned MiniLM đã thu hẹp đáng kể khoảng cách giữa Dense Retrieval và Sparse Retrieval, **FT (0.6209 MRR@10) vẫn không đánh bại được BM25 (0.7583 MRR@10)** trên tập dữ liệu tổng hợp này.
> **Giới hạn nhãn:** Validation dùng synthetic origin qrels: query được trích từ nguyên liệu recipe nguồn, phù hợp với khớp từ khóa. Ưu thế của FT với từ đồng nghĩa, từ viết tắt hoặc lỗi chính tả cần bộ đánh giá riêng, chưa được chứng minh bằng bảng này.

### 2.3. Định dạng Title + Ingredients của Baseline E05
- Trong giai đoạn baseline, thử nghiệm **E05 (title + ingredients)** cho kết quả MRR@10 sụt giảm thảm hại xuống còn **0.1519** (giảm 65% so với E04).
- Chưa có đối chứng xác định cơ chế “title dilution”. Không có fine-tuned title + ingredients bundle được đánh giá trong lần chạy này; không thể suy ra chất lượng cấu hình đó từ E05. E08 là FT main ingredients, không phải thí nghiệm title bị loại bỏ.

---

## 3. Phân tích Hiện tượng Overfit (Train Proxy vs Validation)

Để kiểm tra nguy cơ Overfitting trong quá trình huấn luyện 1 epoch với 30,000 mẫu cặp, hệ thống tiến hành đo lường chỉ số MRR@10 trên tập train proxy (5,000 queries) và tập validation (500 queries):

| Cấu hình | Train Proxy MRR@10 (5,000 Qs) | Validation MRR@10 (500 Qs) | Generalization Gap ($\Delta$) | Tỉ lệ chênh lệch (%) |
| :--- | :---: | :---: | :---: | :---: |
| **E04 Pretrained** | 0.4575 | 0.4342 | +0.0233 | +5.36% |
| **FT CPU 1-epoch** | **0.6450** | **0.6209** | **+0.0241** | **+3.88%** |

### Nhận định:
1. Độ lệch khái quát hóa (**Generalization Gap**) của FT là **+0.0241** (+3.88%), gần như tương đồng hoàn toàn với độ lệch tự nhiên của mô hình pretrained gốc (+0.0233, +5.36%).
2. Một checkpoint và chênh lệch giữa hai tập query không chứng minh mô hình không overfit hoặc xác định nguyên nhân do weight decay, batch size hay số epoch. Thí nghiệm đối chứng Colab riêng dùng 1,024 train / 256 validation pairs, batch 16, ba epoch, seed 42 cho final validation probe loss 0.0263719251 (wd=0.0) và 0.0263708937 (wd=0.01). Chênh lệch khoảng 0.00000103 không chứng minh lợi ích regularization đáng kể, ý nghĩa thống kê hoặc tăng retrieval MRR. Xem `reports/ablation_regularization.md` và `reports/regularization_ablation/summary.json`; không trộn loss này với optimization loss của checkpoint CPU 30k pairs.

---

## 4. Phân tích Các Ca FT Thua Pretrained (Failure Analysis)

Thống kê đối đầu chi tiết trên từng query giữa Fine-Tuned (E08) và Pretrained (E04) qua tập 500 queries validation:
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
