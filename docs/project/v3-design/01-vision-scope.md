# 01 — Tầm nhìn, phạm vi, tiêu chí phát hành

> Đọc kèm `00-overview.md`. File này trả lời 3 câu hỏi: V3 muốn đạt gì? khác V2 thế nào? khi nào một pattern "đủ chuẩn phát hành"?

---

## 1. Tầm nhìn V3

**Một câu:** Biến lớp dữ liệu "mẫu hình giá" từ **bản nháp không thể nghiệm thu** thành **dữ liệu nghiên cứu có minh bạch nguồn + chuẩn đo lường + nhãn độ tin cậy**, phục vụ đồng thời 2 dự án:

1. **Nghiên cứu ChartPatternscan** — xuất chapter PDF theo chuẩn Bulkowski (đã có source-grounded publication gate).
2. **Dashboard `market_stats_v2`** (tab "Lớp mẫu hình phụ trợ") — hiển thị hồ sơ hành vi lịch sử theo mã, **đủ tin cậy để nghiệm thu**.

### Nguyên tắc thiết kế (3 nguyên tắc cốt lõi)

1. **Nguồn gốc trước, số liệu sau** (bài học session thẩm định 12/08/2026)
   - Mọi con số trên dashboard phải truy ngược được về: digitized spec nào → detector nào → events.csv nào → scan ngày nào.
   - Không có số liệu "trôi nổi" không có lookahead + định nghĩa failure đi kèm.

2. **Đúng việc đúng chuẩn** — mỗi pattern có 1 chuẩn đo lường riêng (lookahead, failure threshold, target method). Không dùng 1 lookahead/định nghĩa chung cho mọi pattern.

3. **Minh bạch phân cấp** (triết lý #11 của chủ đầu tư: mọi phân loại phải có ngưỡng số ngay dưới bảng)
   - Dashboard không chỉ show số, mà show thêm: con số này đáng tin cỡ nào? (cỡ mẫu, đã kiểm định hay nháp, lookahead bao nhiêu, target lớn cỡ nào).

---

## 2. Phạm vi V3

### 2.1. Trong phạm vi (in-scope)

| Hạng mục | Mô tả |
|---|---|
| **24 detector v2** | Sửa lookahead + failure/target theo spec digitized (Lớp A đang thi công) |
| **31 family digitized** | Nguồn chuẩn cho lookahead/failure/target (xem `03-measurement-standards.md` bảng đầy đủ) |
| **~55 pattern key** trên artifact web | Mở rộng publication status cho toàn bộ (hiện chỉ ~14 có status) |
| **Pipeline build → split → dashboard** | Tự động hoá, bỏ thao tác chạy tay |
| **Hiển thị dashboard** | Nhãn publication + cảnh báo cỡ mẫu/lookahead + chú thích định nghĩa |
| **Chất lượng dữ liệu** | Filter MAE>80%, xử lý survivorship, minh bạch tỉ lệ mã mất |

### 2.2. Ngoài phạm vi (out-of-scope)

- **Không** thiết kế lại kiến trúc detector từ đầu (giữ family detector hiện có, chỉ sửa chuẩn đo lường).
- **Không** thêm pattern mới ngoài 31 family digitized + biến thể đã có.
- **Không** thay đổi schema DB OHLCV (chỉ filter/label ở tầng build profile).
- **Không** nâng `tradable-final-95` (lớp entry/exit/cost/slippage/portfolio) — V3 chỉ nâng **publication-final** (tài liệu tham khảo). Lớp tradable vẫn `not_tested`.
- **Không** đọc nội dung PDF gốc trong đặc tả này (bản quyền; Lớp C sẽ làm ở M5).

### 2.3. Tích hợp 3 lớp đang triển khai

V3 **không thay thế** 3 lớp A/B/C đang chạy, mà **đóng gói + nâng chuẩn** chúng:

| Lớp | Đang làm | V3 đóng góp |
|---|---|---|
| **A — sửa lookahead 24 detector** | thi công | V3 cung cấp bảng chuẩn lookahead 31 family (file 03) + tiêu chí nghiệm thu "lookahead khớp spec" |
| **B — publication status → artifact** | kế tiếp | V3 định nghĩa pipeline gắn nhãn + hiển thị dashboard (file 02, 04) |
| **C — đọc PDF gốc Bulkowski** | kế tiếp | V3 định nghĩa pattern ưu tiên + cách số liệu PDF đối chiếu với digitized (M5, file 04) |

---

## 3. V3 khác V2 ở đâu

| Khía cạnh | V2 (hiện tại) | V3 (đề xuất) |
|---|---|---|
| **Lookahead** | Hardcode 60 (inside_day) / 120 (pipes mặc định) cho mọi pattern | Đọc từ digitized spec: 10–252 tuỳ pattern |
| **Định nghĩa failure** | `failure = MFE < 5%` (move không đạt 5%) | Theo spec: "giá vượt đáy/đỉnh pattern trước khi chạm target", threshold 1–5% tuỳ pattern |
| **target_hit** | Show % đơn lẻ | Luôn kèm `target_dist_pct` (độ lớn mục tiêu) + cảnh báo nếu target quá nhỏ |
| **Cap events** | `max_events_per_symbol` cố định (12/14/18) → bão hoà | Giữ cap nhưng **minh bạch**: đánh dấu "n = cap" + frequency_score không vượt 100 khi n chạm cap |
| **Dữ liệu bẩn MAE>80%** | Không filter → lẫn vào stats | Filter + label `data_quality_flag=suspect_split`; report tỉ lệ bị loại |
| **Split artifact** | Chạy tay `split_stock_history_artifacts.mjs` | Tự động sau mỗi scan |
| **Publication** | Trộn ~14 final + ~41 draft không nhãn | Nhãn rõ từng pattern: `publication_final` / `candidate` / `draft` + cảnh báo |
| **Survivorship** | Không xử lý (62/1715 mã <2024) | Minh bạch: metadata ghi "đã loại N mã delisted" + option ẩn mã delisted |
| **Chú thích định nghĩa** | Không có | Mỗi bảng có chú thích: "lookahead=X phiên, failure=Y%, target=Z" ngay dưới bảng |
| **Cỡ mẫu** | Có `sample_label` nhưng không cảnh báo nổi | Cảnh báo nổi "mẫu mỏng" khi n<30 + ẩn/loại pattern n<5 khỏi "best_historical" |

---

## 4. Tiêu chí "đủ chuẩn phát hành" cho từng pattern

V3 chia pattern thành **3 nấc độ tin cậy** hiển thị trên dashboard. Nấc nào cũng phải đáp ứng **6 điều kiện kỹ thuật** trước, rồi mới xét nấc.

### 4.1. 6 điều kiện kỹ thuật (bắt buộc cho mọi nấc)

| # | Điều kiện | Cách kiểm chứng | File tham chiếu |
|---|---|---|---|
| K1 | **Lookahead khớp spec** — `evaluated_bars` median ± 5% so với `lookahead_bars` trong digitized spec | So sánh `events.csv[evaluated_bars].median()` vs `spec.post_breakout_measurement.lookahead_bars` | 03 §1 |
| K2 | **Failure tính đúng định nghĩa** — không còn dùng `MFE<5%`, dùng định nghĩa spec (xem 03 §2) | Audit code detector: `failure` flag tính từ path rows, không phải từ MFE | 03 §2 |
| K3 | **target_hit kèm target_dist_pct** — mọi event có cả 2 trường, dashboard hiển thị cả 2 | Check `events.csv` có cột `target_dist_pct` không null | 03 §3 |
| K4 | **Không còn event MAE>80%** (dữ liệu bẩn split chưa điều chỉnh) | Filter ở build: drop event `mae_pct > 80`; report số bị loại | 03 §5 |
| K5 | **Cỡ mẫu minh bạch** — metadata ghi `n`, `sample_label`, `sample_warning` | Đã có trong build.py, giữ nguyên + làm nổi trên UI | 02 §4 |
| K6 | **publication_status từ manifest** — gắn nhãn từ `pattern_family_manifest.json` | Build đọc manifest, gắn `publication_status` vào profile | 04 §1 |

### 4.2. 3 nấc độ tin cậy

| Nấc | Tên hiển thị | Điều kiện bổ sung | Hành vi UI |
|---|---|---|---|
| **🟢 Nấc 3** | **"Đã kiểm định"** | K1–K6 + `publication_status ∈ {publication_final, active}` + `n ≥ 30` + đã qua source-grounded publication gate (`source_notes.status=PASS`, ≥6 source_rules) | Hiển thị đầy đủ, không cảnh báo |
| **🟡 Nấc 2** | **"Ứng viên — đang kiểm tra"** | K1–K6 + `publication_status ∈ {publication_candidate, active_candidate, branch_headline_candidate, candidate_built_review_required}` | Hiển thị + nhãn vàng "đang kiểm tra" |
| **🔴 Nấc 1** | **"Bản nháp — chưa kiểm định"** | Còn lại (draft / pending / in_progress / không có status) | Hiển thị nhưng **ẩn mặc định**, cần user click "hiện bản nháp" + cảnh báo đỏ |

> **Lưu ý:** pattern n<5 LUÔN ở Nấc 1 bất kể publication_status (cỡ mẫu quá mỏng không kết luận được).

### 4.3. Mapping publication_status hiện tại → nấc V3

Dựa trên `pattern_family_manifest.json` (đọc 12/08/2026):

| publication_status (manifest) | Nấc V3 | Số pattern ước tính |
|---|---|---|
| `active` (flag_family) | 🟢 Nấc 3 | 2 (bull_flags, bear_flags) |
| `publication_final` (flag_like: pennants, high_tight_flags) | 🟢 Nấc 3 | 3 |
| `publication_candidate` (triangles_ascending) | 🟡 Nấc 2 | 1 |
| `branch_headline_candidate` (triangles_descending/symmetrical) | 🟡 Nấc 2 | 2 |
| `active_candidate` (wedge, double — family-level) | 🟡 Nấc 2 | 2 family |
| `candidate_built_review_required` (double AA) | 🟡 Nấc 2 | 1 |
| `in_progress` (wedges_falling/rising) | 🔴 Nấc 1 | 2 |
| `next_source_grounded_variant`, `pending` (double AE/EA/EE) | 🔴 Nấc 1 | 3 |
| **Không có trong manifest** (~41 pattern: scallop, horn, pipe, diamond, dead_cat, rectangle, HSB/HST, broadening ×6, measured_move ×2, gap ×4, island ×2, rounding ×2, inside_day, three_methods ×2, three_peaks/valleys ×2, triple ×2, bump_and_run ×2, cup_with_handle ×2) | 🔴 Nấc 1 | ~41 |

**Kết luận:** hiện chỉ **5 pattern Nấc 3** (đã kiểm định), ~6 pattern Nấc 2, ~41 pattern Nấc 1. V3 cần mở rộng publication để đưa thêm pattern lên Nấc 2/3 theo thời gian (xem roadmap file 04).

---

## 5. Định nghĩa "nghiệm thu dashboard V3"

Dashboard tab "Lớp mẫu hình phụ trợ" được nghiệm thu khi:

1. **Mã bất kỳ** mở lên → hồ sơ pattern chỉ hiển thị pattern Nấc 1 khi user chủ động "hiện bản nháp".
2. Mỗi pattern có nhãn nấc rõ + chú thích định nghĩa (lookahead, failure, target) ngay dưới bảng.
3. Không còn event MAE>80% trong dữ liệu nền.
4. target_hit luôn kèm target_dist_pct.
5. Metadata ghi: nguồn (digitized spec + detector), ngày scan, số event, số event bị loại (dữ liệu bẩn), số mã delisted.
6. Chạy 1 lệnh → toàn pipeline rebuild + split + dashboard refresh (không thao tác tay).

Tiêu chí này là **điều kiện cần**, chưa phải đủ — chủ đầu tư duyệt lần cuối trên UI thực tế.
