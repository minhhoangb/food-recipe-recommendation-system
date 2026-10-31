# Báo Cáo Kịch Bản Mô Phỏng Giám Sát & Data Drift (Simulation)

**Người thực hiện:** Hiếu (Person 3) - Vai trò: Fine-tune + MLflow + Monitoring  
**Mục tiêu:** Kiểm nghiệm năng lực phát hiện dịch chuyển dữ liệu (Data Drift) của `src/monitor.py` qua 2 kịch bản độc lập (có dịch chuyển và không dịch chuyển).

> **LƯU Ý QUAN TRỌNG VỀ PHẠM VI:**  
> Báo cáo này được thực hiện trên **dữ liệu mô phỏng (simulation)**. Các chỉ số dịch chuyển thống kê được thiết kế có chủ đích để kiểm thử độ nhạy của bộ kiểm định (KS test, OOV rate, Latency p95), **tuyệt đối không suy diễn trực tiếp thành sự suy giảm độ liên quan thực tế (relevance decay)** của mô hình phục vụ.

---

## 1. Thiết Kế Hai Kịch Bản Mô Phỏng

| Đặc điểm kiểm nghiệm | Kịch bản 1: Không Dịch Chuyển (No Drift) | Kịch bản 2: Có Dịch Chuyển (Drift Detected) |
|---|---|---|
| **Số lượng Request** | 100 requests (`simulated_requests_nodrift.jsonl`) | 100 requests (`simulated_requests_drift.jsonl`) |
| **Phân phối độ dài query** | Độ dài 2–5 nguyên liệu (trung bình: 3.38) | Độ dài 6–10 nguyên liệu (trung bình: 7.91) |
| **Thành phần từ vựng** | 100% nguyên liệu nằm trong từ vựng huấn luyện (train vocab) | Pha trộn 30–50% nguyên liệu lạ ngoài từ vựng (`dragonfruit_powder`, `spirulina_extract`, ...) |
| **Phân phối độ trễ API** | Trung vị: 23.18 ms, p95: 30.7 ms | Trung vị: 85.09 ms, p95: 113.74 ms |
| **Tỉ lệ lỗi (Status 4xx)** | 0.0% | 7.0% |

---

## 2. Kết Quả Định Lượng & Quyết Định Kiểm Định Thống Kê

### Kịch bản 1: Không Dịch Chuyển (No Drift)
- **Kết luận:** `NO_DRIFT` (Drift Detected: `False`)
- **Kiểm định độ dài query (KS-2):** statistic D = `0.08`, p-value = `0.7784` (p > 0.05 => Phân phối tương đồng).
- **Tỉ lệ nguyên liệu lạ (OOV Rate):** `0.00%` (ngưỡng cảnh báo: > 5.0%).
- **Báo cáo HTML trực quan:** [reports/monitoring_nodrift_simulation.html](monitoring_nodrift_simulation.html)

### Kịch bản 2: Có Dịch Chuyển (Drift Detected)
- **Kết luận:** `DRIFT_DETECTED` (Drift Detected: `True`)
- **Kiểm định độ dài query (KS-2):** statistic D = `1.0`, p-value = `0.0` (p < 0.05 => **CẢNH BÁO DỊCH CHUYỂN ĐỘ DÀI**).
- **Tỉ lệ nguyên liệu lạ (OOV Rate):** `49.43%` (**VƯỢT NGƯỠNG** 5.0% => Cảnh báo xuất hiện từ khóa chưa qua tiền xử lý/học biểu diễn).
- **Độ trễ API p95:** `113.74 ms` (tăng gấp 3.5 lần do độ dài chuỗi tokenize dài hơn).
- **Báo cáo HTML trực quan:** [reports/monitoring_drift_simulation.html](monitoring_drift_simulation.html)

---

## 3. Nhận Xét Chuyên Môn (6 câu)

1. Bộ giám sát `src/monitor.py` đã phân biệt chính xác hai trạng thái phân phối dữ liệu dựa trên bài toán kiểm định thống kê phi tham số hai mẫu Kolmogorov-Smirnov kết hợp ngưỡng tỉ lệ ngoại vi.
2. Ở kịch bản chuẩn (No Drift), kiểm định KS trên độ dài truy vấn cho p-value lớn hơn mức ý nghĩa $lpha = 0.05$, đồng thời tỉ lệ OOV bằng 0%, hệ thống giữ trạng thái an toàn `NO_DRIFT`.
3. Khi đối mặt với kịch bản dịch chuyển (Drift Detected), sự gia tăng đột biến độ dài truy vấn (từ 3.3 lên 7.8 nguyên liệu) khiến khoảng cách tích lũy phân phối D tăng vọt, đẩy p-value tiệm cận 0.0000.
4. Cùng lúc đó, tỉ lệ nguyên liệu chưa từng xuất hiện trong tập train vượt mức `49.4%`, kích hoạt cảnh báo suy giảm độ bao phủ từ vựng (vocabulary coverage).
5. Việc giám sát đồng thời cả độ dài chuỗi, tỉ lệ OOV, độ trễ p95 và tần suất lỗi 4xx giúp phát hiện sớm các dị thường trong hành vi người dùng trước khi chất lượng dịch vụ bị ảnh hưởng nghiêm trọng.
6. Kết quả này đạt trọn vẹn yêu cầu DoD của bài toán giám sát, cung cấp công cụ tự động hóa xuất báo cáo trực quan cho đội ngũ vận hành.
