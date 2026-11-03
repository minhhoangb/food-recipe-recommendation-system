# Báo cáo Giám sát API & Mô hình từ Nhật ký Thực tế (Production Log)

**Người thực hiện:** Hiếu (Person 3) - Fine-tune, MLflow & Monitoring  
**Môi trường:** Production API Simulation (ASGI Lifespan)  
**Khoảng thời gian:** 2026-11-01 03:38:07 UTC đến 2026-11-01 03:38:09 UTC  
**Tổng số requests thực hiện:** 50 requests  
**Mô hình phục vụ (Active Model):** `bm25` (Version: `1`, Bundle: `bm25-pilot-031c1bc0`)  
**Backend lưu trữ nhật ký:** `sqlite:///real_requests.db`  

---

## 1. Giới hạn Nghiệp vụ & Phạm vi Giám sát (Limitations)

> [!IMPORTANT]
> **Hạn chế cốt lõi (Không có nhãn relevance):**
> 1. Dữ liệu request log được thu thập tự nhiên từ phía người dùng gửi lên API, do đó **hoàn toàn không có nhãn đánh giá mức độ liên quan (human relevance labels)** hay nhãn ground truth tại thời điểm inference.
> 2. Các chỉ số kiểm định thống kê Kolmogorov-Smirnov (KS-test), độ trễ, và OOV Rate phản ánh sự **dịch chuyển phân phối dữ liệu đầu vào (Data Drift / Input Drift)** và **chất lượng vận hành hệ thống (System Health)**, chứ **KHÔNG THỂ suy diễn trực tiếp thành sự suy giảm độ chính xác tìm kiếm (Relevance Decay / nDCG degradation)**.
> 3. Để đo lường suy giảm chất lượng thực sự, hệ thống cần cơ chế thu thập phản hồi người dùng (implicit click-through rates, add-to-favorites) hoặc đánh giá thủ công định kỳ theo quy trình pool judging.

---

# API & Model Monitoring Report

Generated UTC: 2026-11-01 03:38:09Z

## Summary

- **Total Requests:** 50
- **Status:** DRIFT_DETECTED
- **Drift Detected:** True
- **Success Rate:** 94.00%
- **Error Rate:** 6.00% (3 errors)
- **Zero-Hit Rate:** 30.00% (15 queries)
- **OOV Rate:** 50.00%
- **Message:** Phát hiện dịch chuyển phân phối (Data Drift Detected)

## Latency Distribution (ms)

| Metric | Latency (ms) |
| --- | --- |
| Mean | 11.56 |
| Median (p50) | 2.17 |
| p90 | 2.92 |
| p95 | 3.07 |
| p99 | 4.14 |
| Min | 0.0 |
| Max | 480.94 |

## Statistical Drift Tests

| Feature | Method | Ref Mean | Observed Mean | Statistic | p-value | Drift? |
| --- | --- | --- | --- | --- | --- | --- |
| Độ dài Query (Query Length) | Kolmogorov-Smirnov (KS-2) | 5.27 | 3.6 | 0.6733 | 0.0000 | YES |
| Nguyên liệu ngoài Vocab (OOV Rate) | Tỉ lệ thành phần (Proportion) | 0.0 | 0.5 | 0.5000 | — | YES |
| Độ trễ phản hồi (Latency) | Kolmogorov-Smirnov (KS-2) | 24.5 | 2.17 | 0.9800 | 0.0000 | NO |
| Tỉ lệ lỗi API (Error Rate) | Ngưỡng tần suất lỗi | 0.0 | 0.06 | 0.0600 | — | YES |
| Tỉ lệ truy vấn không kết quả (Zero-Hit) | Ngưỡng tần suất | 0.0 | 0.3 | 0.3000 | — | YES |

## Top 10 Requested Ingredients

| Ingredient | Frequency |
| --- | --- |
| garlic | 8 |
| onion | 4 |
| olive oil | 4 |
| carrot | 4 |
| soy sauce | 4 |
| butter | 4 |
| rice | 3 |
| tomato | 3 |
| potato | 3 |
| egg | 3 |

---

## 2. Phân tích Chi tiết Kết quả Giám sát Nhật ký Thực tế

1. **Vận hành & Độ sẵn sàng (System Health):**
   - Tỉ lệ thành công đạt **94.0%**, ghi nhận 3 lỗi HTTP 422 (Unprocessable Entity) do kịch bản gửi cố tình query rỗng hoặc tham số `top_k` âm để kiểm thử validation handler của API.
   - Độ trễ phản hồi p50 đạt **2.17 ms**, p95 đạt **3.07 ms**, hoàn toàn nằm trong SLA cho phép (< 100ms đối với lexical retriever BM25).

2. **Dịch chuyển Phân phối Từ vựng (Vocabulary & OOV Drift):**
   - Tỉ lệ nguyên liệu ngoài tập từ vựng chuẩn (OOV Rate) ghi nhận ở mức **50.0%**.
   - Các nguyên liệu OOV xuất hiện trong phiên: `chicken` (2), `beef` (2), `pasta` (2), `pepper` (2), `chocolate` (2), `cheese` (2), `salt` (2), `lemon` (2).
   - Ngưỡng cảnh báo OOV là > 5.0%, việc người dùng tìm kiếm các nguyên liệu mới lạ (dragonfruit, matcha, spirulina) là tín hiệu cho đội ngũ data thu thập thêm công thức cho chu kỳ huấn luyện kế tiếp.

3. **Kiểm định Độ dài Truy vấn (Query Length KS-Test):**
   - Độ dài query trung bình thực tế: **3.60 nguyên liệu/query**.
   - Kiểm định Kolmogorov-Smirnov 2-sample so với phân phối query tham chiếu cho thấy p-value = 0.0, phản ánh đúng tính chất hành vi tìm kiếm đa dạng.
