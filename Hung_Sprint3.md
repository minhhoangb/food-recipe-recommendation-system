## Hưng | Giao diện web, benchmark và chọn bundle qua config

**Người nhận:** Hưng | **Vai trò:** Embedding + API + giao diện + Docker

**Gửi:** tối Thứ 3 27/10 | **HẠN NỘP: 24:00 Thứ 5 29/10 (0h sáng Thứ 6 30/10)**

**Việc:** Giao diện web, benchmark và chọn bundle qua config

| **Mục** | **Chi tiết**                                                                                                                                                                                                                                                                                      | **DoD (xong khi)**                                                                                               |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **a**   | Giao diện tĩnh fe/index.html (HTML/CSS/JS) phục vụ qua FastAPI StaticFiles, tiếng Anh: form nguyên liệu và top-k, danh sách kết quả (title, matched/missing, directions mở rộng được, link nguồn chỉ http/https, escape nội dung), trạng thái loading/empty/error/retry, dùng được bằng bàn phím. | Chạy end-to-end với API. Kiểm tra ở 390px và 1440px có ảnh chụp. Không có ảnh, thời gian nấu hay dinh dưỡng bịa. |
| **b**   | Viết src/benchmark.py: p50/p95 latency (tối thiểu 20 lần warmup và 100 query tuần tự), RAM, thời gian khởi động, dung lượng bundle, cho TF-IDF, BM25, LSA, MiniLM pretrained và FT. Dùng psutil và time.perf_counter.                                                                             | benchmark.json và bảng. Ghi CPU và RAM máy thật. Chạy lại được bằng một lệnh.                                    |
| **c**   | Chọn model bằng config (serving.bundle) cho cả tfidf, bm25, lsa, minilm, ft. /model-info trả version, hash corpus và metric đọc từ manifest hoặc MLflow.                                                                                                                                          | Đổi bundle chỉ bằng config. Mỗi bundle có test. Bundle sai hash không cho khởi động.                             |
| **d**   | Gán nhãn dev phần của bạn. Kiểm tra parity: cùng bundle chạy native và Docker cho cùng top-k và score trong sai số.                                                                                                                                                                               | Nhãn nộp đúng hạn. Log parity native và Docker.                                                                  |

**Đầu ra nộp:** PR vào dev; fe/; benchmark.json; test chọn bundle; ảnh UI.

**Thư viện nên dùng:** FastAPI, psutil, httpx.

**Làm song song / phụ thuộc:** UI dựng trên API sẵn có, không chờ model mới. Benchmark chạy khi có đủ bundle của Hoàng và Hiếu.