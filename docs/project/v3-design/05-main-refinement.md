# 05 — Hoàn thiện đặc tả (bổ sung của main agent sau khi review bản nháp GLM)

> **Tác giả:** main agent (deepseek-v4-flash, provider OpenCode Go) — phiên làm việc `sess_1c356bbb-d126-4ea6-9508-ad490bca4ef1`
> **Ngày:** 12/08/2026
> **Mục đích:** bản nháp GLM (00–04) đã tốt về khung. File này chốt 6 điểm còn bỏ ngỏ, theo chỉ đạo chủ đầu tư ("GLM viết thiết kế chưa tốt — main hoàn thiện, sau đó Kimi K3 chốt kế hoạch final 1 lần").

---

## 1. VISION CỦA CHỦ ĐẦU TƯ (bắt buộc ghi vào mọi quyết định)

> **"Nâng nghiên cứu lên v3 HOÀN CHỈNH để sử dụng cho 2 dự án"** (chủ đầu tư, 12/08/2026).

V3 KHÔNG phải "bản vá" cho dashboard. V3 là **nghiên cứu hoàn chỉnh** với 2 đầu ra:
1. **Nghiên cứu ChartPatternscan** — dữ liệu + chuẩn đo lường đủ để xuất chapter PDF theo chuẩn Bulkowski (publication-final).
2. **Dashboard market_stats_v2** (tab Lớp mẫu hình phụ trợ) — hiển thị dữ liệu đáng tin, đủ điều kiện nghiệm thu.

Hệ quả thiết kế: mọi chuẩn (lookahead, failure, target, cỡ mẫu) phải đặt ở TẦNG DỮ LIỆU/SPEC dùng chung — không có "chuẩn riêng cho dashboard".

## 2. CHỐT: cơ chế benchmark trước/sau (bổ sung cho R4.1)

Đặc tả GLM nói "snapshot profile cũ trước" nhưng không có cơ chế cụ thể. Chốt:

- **Trước mỗi mốc M1–M4:** sao chép toàn bộ `web/stock_pattern_*.json` + `web/profiles|setups|personality/` sang `backups/v3_benchmark/<mốc>/<timestamp>/` (kèm `events.csv` cũ).
- **Sau mỗi mốc:** sinh bảng so sánh `before → after` (median MFE/MAE, failure rate, target_hit, n, thứ hạng top-10) — file `backups/v3_benchmark/<mốc>/before_after_report.md`.
- **Quy tắc:** KHÔNG rebuild lần 2 khi chưa có snapshot lần 1. Đây là checkpoint bắt buộc, không bỏ qua.

## 3. CHỐT: đóng băng hiển thị dashboard tới M3

R4.4 nói "có thể đóng băng" — chốt thành BẮT BUỘC:

- **Từ giờ tới khi M3 xong:** dashboard KHÔNG tự cập nhật giao diện/số liệu mới. Lý do: tránh chủ đầu tư thấy "số liệu nửa vời" (lookahead mới nhưng failure cũ) giữa các mốc.
- **Cách làm:** pipeline mới (M1–M4) build vào thư mục riêng `web_v3/` (staging) — không đè `web/`. Chỉ khi M3 xong + cross-check PASS → chuyển `web_v3/` → `web/` (1 bước bật V3).
- **Ngoại lệ:** nếu chủ đầu tư muốn xem tiến độ, mở staging qua đường dẫn riêng (không phải trang chính).

## 4. CHỐT: quyền đọc PDF gốc Bulkowski (Lớp C / M5)

- **Chủ đầu tư ĐÃ duyệt** (12/08/2026, yêu cầu "GLM dùng vision đọc tài liệu gốc Thomas Bulkowski"): GLM được đọc 3 file PDF trong `references/` để đối chiếu chuẩn.
- Cách đọc: GLM dùng công cụ đọc file (chrome-devtools hoặc tương đương) mở đúng trang pattern; ghi `pdf_path + book_pages_checked + pdf_pages_checked + failure/target/sample trích dẫn`.
- Không trích dài nội dung PDF vào artifact public — chỉ trích con số chuẩn (số liệu, không phải văn bản).

## 5. CHỐT: mối quan hệ Lớp A (đang sửa số) vs registry tập trung (đặc tả 02 §2)

- **Lớp A đang thi công** = đổi tham số lookahead trong từng detector (số cứng → số đúng chuẩn). Đây là đúng hướng, không bỏ dở.
- **Registry tập trung** (`lookahead_registry.py`) = tối ưu hóa SAU, khi M1 xong: gom chuẩn 1 nơi để build profile + dashboard cùng đọc (tránh lệch chuẩn lần nữa).
- **Quyết định:** M1 đạt khi số liệu đúng chuẩn (bảng đối chiếu PASS). Registry làm trong M1 nếu rẻ (≤ vài giờ), nếu không → gộp vào M4 (tầng điều chỉnh). KHÔNG block M2 vì registry.

## 6. CHỐT: nghiệm thu từng mốc (ai kiểm chứng)

| Mốc | Người thi công | Người verify độc lập | Chủ đầu tư |
|---|---|---|---|
| M1 | agent thi công (deepseek-v4-flash) | GLM-5.2 (đối chiếu bảng lookahead + events) | xem báo cáo before/after |
| M2 | agent thi công | GLM-5.2 (audit failure_busted vs spec) | xem báo cáo |
| M3 | agent thi công | GLM-5.2 (UI render thật trên chrome-devtools) | **ký nghiệm thu tab mẫu hình** |
| M4 | agent thi công | GLM-5.2 (chạy 1 lệnh end-to-end) | xem báo cáo |
| M5 | GLM-5.2 (đọc PDF) | main agent (đối chiếu số) | xem bảng đối chiếu |

- Cross-check GLM độc lập KHÔNG đọc kết quả thi công trước khi review xong (đúng quy trình 2 model).
- Chủ đầu tư ký chính thức 1 lần cuối (tab Lớp mẫu hình) — giữa các mốc chỉ xem báo cáo.

## 7. Bổ sung nhỏ (từ review)

- **Non-advice boundary GIỮ NGUYÊN** mọi nơi ("Hồ sơ hành vi lịch sử; không phải tín hiệu mua bán") — kể cả khi pattern lên Nấc 3.
- **Ngôn ngữ hiển thị:** mọi cảnh báo/chú thích mới trên dashboard bằng tiếng Việt đời thường (người dùng non-code).
- **Rollback plan:** mỗi mốc có `backups/v3_benchmark/` → nếu mốc sau lỗi, khôi phục snapshot mốc trước ≤ 15 phút.
- **Nhãn version:** metadata mỗi artifact ghi `pipeline_version` (v3-m1…v3-m4) + `generated_at` — dashboard cảnh báo stale >7 ngày.
- **Nấc 1 ẩn mặc định + toggle "hiện bản nháp"** — giữ nguyên đề xuất GLM (file 01 §4.2).

---

## Kết luận (gửi K3 chốt)

Bản nháp hoàn chỉnh = file 00 + 01 + 02 + 03 + 04 (GLM) + file 05 này (main). Yêu cầu K3: review toàn bộ, chốt kế hoạch triển khai FINAL (điều chỉnh gì nếu cần — đặc biệt: thứ tự mốc, phạm vi M1–M5, cách đóng băng UI, rủi ro nào phải xử trước), ra quyết định dứt khoát — KHÔNG để ngỏ câu nào.
