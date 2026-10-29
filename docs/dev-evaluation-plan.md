# Kế hoạch gán nhãn dev, phân tích lỗi và báo cáo

Nhánh: `error-analysis-labels`. PR đích: `dev`. Người thực hiện: Hoàng;
reviewer: Hiếu. Hạn nộp: 24:00 ngày 29/10/2026 (00:00 ngày 30/10/2026).

## Phase 0 — Sửa điều kiện chạy và kiểm tra tích hợp

- Sửa lint toàn repo; giữ nguyên các quy tắc kiểm tra của CI.
- Sửa CLI fine-tune và pipeline train nhận đúng `(pairs, stats)` từ `build_pairs`.
- Thêm test tái hiện lỗi trước khi sửa; chạy toàn bộ pytest và đo coverage.
- Không chạy lại fine-tune hoặc thay đổi corpus/checkpoint hiện có.

## Phase 1 — Pool và nhãn dev

- Đã có pool top-5 của bảy hệ thống, 20 query, 396 mục sau gộp trùng.
- Có 484 lượt chấm, gồm 88 mục được chấm độc lập hai lần.
- Hoàng: `reports/manual_eval/pool_dev_hoang.csv`, 172 dòng.
- Hưng: `reports/manual_eval/pool_dev_hung.csv`, 161 dòng.
- Hiếu: `reports/manual_eval/pool_dev_hieu.csv`, 151 dòng.
- Query chính được chia 7/7/6; mỗi người còn nhận một phần chấm kép.
- Đọc `reports/manual_eval/annotation_guide.md`; không mở file key khi chấm.
- Lệnh tạo pool từ chối ghi đè CSV đã có relevance hoặc notes.
- Đã có một lượt chấm AI riêng cho đủ 396 mục trong
  `reports/manual_eval/pool_dev_ai.csv`, gồm nhãn 0/1/2, nguồn `ai` và lý do.
  Có 116 nhãn 0, 208 nhãn 1, 72 nhãn 2; 20 mục đánh dấu `unsure`.
  Metadata và MLflow ghi rõ chưa được người kiểm chứng, không tính kappa.
- **Còn chờ:** cả ba chấm relevance thật, ghi nhận nhãn ban đầu, đối chiếu
  bất đồng sau khi chấm độc lập, lưu nhãn thống nhất và nguồn người chấm.
- Chỉ khi có nhãn mới tính agreement, pooled nDCG và coverage. Không điền
  nhãn tự động vào tên người chấm. Không gửi tin nhắn cho thành viên từ repo.

## Phase 2 — Phân tích lỗi

- Đã có `reports/error_analysis.md` và bảng từng query đi kèm.
- Có 28 ca validation thật, recipe_id và top-5 của từng hệ thống.
- Có số liệu theo độ dài query, nguyên liệu phổ biến/hiếm, số ít/số nhiều,
  kích thước recipe và train–validation gap; bao gồm query còn một nguyên liệu.
- MRR@10 FT: train 0.644959; validation 0.620852. Kết quả là proxy tìm lại
  recipe gốc, không thay thế đánh giá relevance của người.

## Phase 3 — Báo cáo phần 1

- Nguồn: `reports/report_part1.md`; bản nộp: `reports/report_part1.pdf`.
- Gồm problem formulation, data analysis, preprocessing, feature engineering;
  sử dụng sáu hình từ notebook EDA và tối thiểu sáu trang nội dung.
- `reports/report_part1_provenance.json` ghi hash, số liệu corpus đã đo lại
  và run MLflow. Audit raw cũ được chỉ rõ là nguồn kế thừa, chưa chạy lại.
- Không dùng bảng ablation hoặc ảnh dashboard minh họa làm bằng chứng thực nghiệm.

## Phase 4 — Test, commit và PR

- Đã có ít nhất 11 test mới so với `dev` trước các regression test bổ sung.
- Chạy `ruff check .` và toàn bộ `pytest` với pytest-cov như workflow CI.
- Yêu cầu từng module baseline, features và evaluate có coverage ít nhất 70%.
- Commit code, test và báo cáo; loại log tạm, dữ liệu, weights và file skill mới.
  Giữ nguyên các file đã được Git theo dõi từ trước.
- Push nhánh, mở PR vào `dev`, chờ CI và Hiếu review trước khi merge.
- DoD chỉ đóng khi nhãn người hoàn tất và CI trên PR xanh; kết quả local
  không được ghi thành kết quả GitHub Actions.

## Kết quả kiểm tra local ngày 29/10/2026

- `ruff check .`: đạt, không bỏ qua lỗi.
- Toàn bộ pytest: 243 passed. Coverage baseline 97.20%, features 100%, evaluate 96.69%.
- Trên Windows sandbox, TestClient bị chặn ở socketpair; bộ test chạy đạt ngoài sandbox.
- Ba file nhãn của người vẫn trống; bộ nhãn AI được lưu riêng và không thay thế
  ba lượt chấm độc lập. PR #14 đã được merge; các thay đổi tiếp nối được nộp riêng.
