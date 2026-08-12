# 04 — Publication governance + Roadmap + Rủi ro

> File này: (1) mở rộng publication governance cho 55 pattern (tận dụng source-grounded-publication-gate hiện có), (2) 5 mốc triển khai có đầu ra kiểm chứng, (3) rủi ro + giảm thiểu.

---

## 1. Publication governance V3 — mở rộng 55 pattern

### 1.1. Tận dụng gate hiện có

Repo đã có 2 gate mạnh (không cần viết lại):
- `docs/project/source-grounded-publication-gate.md` — 11 nguyên tắc (đọc nguồn trước, không tự suy target, 2 trục chấm điểm publication-final vs tradable-final-95...).
- `scanner/v2/pattern_family_manifest.json` — manifest theo dõi publication status từng pattern/family.

V3 **không thay thế** 2 gate này, mà **mở rộng áp dụng** từ ~14 pattern hiện có sang 55 pattern.

### 1.2. Quy trình 3 nấc: draft → candidate → final

| Nấc | Tên | Điều kiện lên nấc | Tương đương manifest hiện tại |
|---|---|---|---|
| **draft** | Bản nháp | Detector chạy ra events.csv, chưa qua source audit | (không có status / `pending` / `in_progress` / `next_source_grounded_variant`) |
| **candidate** | Ứng viên | + `source_notes.status=PASS` + ≥6 `source_rules` + digitized spec có lookahead/failure/target + pattern qua K1–K6 (file 01 §4.1) + visual validation (premium scored_n≥30, median≥4, pass_rate≥70) | `publication_candidate` / `active_candidate` / `branch_headline_candidate` / `candidate_built_review_required` |
| **final** | Đã kiểm định | + `direct_pdf_review.status=PASS` (đọc PDF gốc, có pdf_path + book_pages_checked) + target calibration PASS + semantic audit PASS (`publication_semantic_gate_v1`) + governance matrix cập nhật | `publication_final` / `active` |

> **Lưu ý quan trọng (nguyên tắc 9 source-gate):** không tuyên bố `final` nếu chưa kiểm PDF gốc trực tiếp. Digitized spec chỉ là chỉ mục. Đây là lý do M5 (đọc PDF) cần thiết cho pattern lên Nấc 3.

### 1.3. Mapping 55 pattern → nấc hiện tại + kế hoạch nâng

Dựa trên manifest (đọc 12/08/2026) + artifact EVENT_SOURCES (55 key):

#### Nấc 3 (final) — hiện 5 pattern
| Pattern | Đã có? | Ghi chú |
|---|---|---|
| bull_flags | ✅ active | tham chiếu |
| bear_flags | ✅ active | defensive |
| bull_pennants | ✅ publication_final | |
| bear_pennants | ✅ publication_final | |
| high_tight_flags | ✅ publication_final | |

#### Nấc 2 (candidate) — hiện ~6 pattern
| Pattern | Status | Việc cần làm lên Nấc 3 |
|---|---|---|
| triangles_ascending | publication_candidate | direct_pdf_review + calibration |
| triangles_descending | branch_headline_candidate | branch headline + PDF review |
| triangles_symmetrical | branch_headline_candidate | direction-first headline + PDF |
| wedges_falling | in_progress → candidate | source contract + visual |
| wedges_rising | in_progress → candidate | source contract + visual |
| double_bottoms_adam_adam | candidate_built_review_required | source audit |

#### Nấc 1 (draft) — ~44 pattern còn lại
Nhóm cần source contract + digitized spec đầy đủ + visual validation:
- **Reversal lớn (lookahead 252):** cup_with_handle (+inverted), HSB (+complex), HST (+complex), broadening (6 biến thể), bump_and_run (2), diamond (2), double AE/EA/EE (6), rectangle (2), rounding (2), scallop (4), three_peaks/valleys (2), triple (2), dead_cat_bounce (2).
- **Ngắn/trung (lookahead 10–63):** inside_day, rising/falling_three_methods, area/breakaway/continuation/exhaustion_gaps, island_reversals, islands_long, horn_bottoms/tops, pipe_bottoms/tops, measured_move_up/down.

### 1.4. Thứ tự ưu tiên nâng nấc (đề xuất)

Ưu tiên theo **tần suất xuất hiện trên VN + độ phủ spec**:

| Ưu tiên | Pattern | Lý do |
|---|---|---|
| **P0** (làm đầu) | inside_day, pipe_bottoms/tops, horn_bottoms/tops | tần suất cao trên VN, đã có detector, chỉ cần sửa lookahead + failure |
| **P1** | cup_with_handle, HSB, scallop, rectangle | reversal phổ biến, đã có detector + digitized spec đầy đủ |
| **P2** | double variants (AE/EA/EE), broadening, diamond | cần source contract per-variant |
| **P3** | gaps (4 loại), islands, measured_move, three_methods, three_peaks/valleys, triple, bump, dead_cat, rounding | tần suất thấp hơn hoặc spec chưa đầy đủ |

### 1.5. Tích hợp vào build profile + dashboard

1. Build profile đọc manifest → gắn `publication_status` + `publication_narrative_tier` (Nấc 1/2/3) vào mỗi pattern.
2. Dashboard hiển thị nhãn nấc (xem file 02 §5).
3. Khi manifest thay đổi (pattern lên nấc) → rebuild profile → dashboard tự cập nhật nhãn.

---

## 2. Cỡ mẫu publication chapter (riêng vs hồ sơ mã)

> Phân biệt rõ: §4 file 03 nói cỡ mẫu cho **hồ sơ 1 mã** (n≥30). Phần này nói cỡ mẫu cho **publication chapter toàn thị trường** (để pattern lên Nấc 3).

| Loại thống kê | Cỡ mẫu tối thiểu (toàn thị trường VN) | Ghi chú |
|---|---:|---|
| Stats headline (median MFE/MAE, failure rate) | ≥ 100 events | VN có ~1500 mã hoạt động, pattern phổ biến dễ đạt |
| Target calibration | ≥ 50 events per target band | 4 band: 0.46x/0.5x/0.75x/1.0x |
| Variant split (up/down, AA/AE...) | ≥ 30 events per variant | tránh variant mỏng |
| Visual validation (premium) | scored_n ≥ 30, median ≥ 4, pass_rate ≥ 70 | đã có trong triangle gate |

Pattern nào không đạt cỡ mẫu publication → stays ở Nấc 2 (candidate) + dashboard ghi "cỡ mẫu chưa đủ cho publication chapter".

---

## 3. Roadmap — 5 mốc triển khai

> Thứ tự phụ thuộc Lớp A (đang thi công) → B → C. Mỗi mốc có đầu ra kiểm chứng được.

### M1 — Sửa lookahead 24 detector theo spec digitized (Lớp A)

**Mục tiêu:** mọi detector đọc lookahead từ registry, không hardcode.

**Đầu ra kiểm chứng:**
- Bảng đối chiếu 24 detector: `pattern | spec_lookahead | before | after | chênh %`.
- Sample kiểm: inside_day median MFE giảm từ ~15% → ~3% (khoảng ±2%); scallop `evaluated_bars` tăng từ ~120 → ~252.
- events.csv có cột `expected_lookahead`, `actual_evaluated_bars`.

**Phụ thuộc:** Lớp A đang thi công (agent khác). V3 cung cấp bảng chuẩn (file 03 §1).

**Rủi ro:** đổi lookahead làm thay đổi MFE/MAE/failure → cần benchmark before/after (xem §4).

---

### M2 — Định nghĩa lại failure/target chuẩn Bulkowski

**Mục tiêu:** thêm `failure_busted` (chuẩn), đổi tên `failure_5pct` → `weak_move_5pct`, target kèm `target_dist_pct`.

**Đầu ra kiểm chứng:**
- Report đối chiếu: `pattern | failure_busted_rate_V3 | weak_move_5pct_rate_cũ | spec_failure_rate`.
- bull_flags: failure_busted_rate ≈ 5.5% (spec), weak_move_5pct ≈ 25% (cũ).
- inside_day: failure_busted_rate ≈ 15% (spec at_5pct).
- cup_with_handle: failure_busted_rate ≈ 5% (spec).
- Dashboard hiển thị cả 2 + target_dist_pct.

**Phụ thuộc:** M1 (cần lookahead đúng trước khi tính failure).

---

### M3 — Publication status từ manifest → artifact web (Lớp B)

**Mục tiêu:** mỗi pattern trên dashboard có nhãn Nấc 1/2/3 rõ + cảnh báo.

**Đầu ra kiểm chứng:**
- Build profile đọc manifest, gắn `publication_narrative_tier` cho 55 pattern.
- Bảng: `pattern_key | publication_status_manifest | nấc_V3`.
- Dashboard: mở mã bất kỳ → hồ sơ pattern có nhãn nấc + màu + chú thích định nghĩa.
- Nấc 1 ẩn mặc định, có toggle "hiện bản nháp".
- target_hit kèm target_dist_pct; cảnh báo target quá nhỏ.

**Phụ thuộc:** M1 + M2 (cần số liệu đúng trước khi gắn nhãn tin cậy).

---

### M4 — Tự động hoá rebuild + split + filter dữ liệu bẩn

**Mục tiêu:** 1 lệnh refresh toàn pipeline; dữ liệu bẩn MAE>80% bị loại + report.

**Đầu ra kiểm chứng:**
- `scripts/refresh_pattern_pipeline.sh` (hoặc Makefile target) chạy end-to-end: detector → build → split → timestamp.
- Report mỗi pattern: `events_total | events_dropped_split_suspect | drop_rate_pct | delisted_symbols`.
- inside_day: drop_rate ≈ 2.8% (277/9847).
- Dashboard đọc `metadata.generated_at` → cảnh báo stale nếu >7 ngày.
- Hồ sơ mã có cảnh báo "dữ liệu cần rà corporate action" nếu >3 event suspect.

**Phụ thuộc:** M1–M3.

---

### M5 — Đọc PDF gốc Bulkowski nâng chuẩn pattern ưu tiên (Lớp C)

**Mục tiêu:** bổ sung failure/target/sample từ PDF gốc cho pattern ưu tiên (P0 + P1) + pattern chưa có trong digitized (dead_cat, high_tight_flags chi tiết, broadening_right_angled, triple, three_peaks/valleys).

**Đầu ra kiểm chứng:**
- Bảng: `pattern | pdf_path | book_pages_checked | pdf_pages_checked | failure_rate_pdf | target_pdf | sample_pdf`.
- Pattern P0 + P1 có `direct_pdf_review.status=PASS` → đủ điều kiện lên Nấc 3.
- Đối chiếu: số liệu V3 (sau M1–M4) vs số liệu PDF — chênh lớn → audit hoặc ghi "khả thi cho thị trường VN khác US".

**Phụ thuộc:** M2 (cần định nghĩa failure đúng trước khi đối chiếu PDF).

**Lưu ý bản quyền:** đọc PDF trong repo `references/` cho nghiên cứu nội bộ (theo AGENTS.md). Không trích dài vào artifact public.

---

### 3.1. Tóm tắt thứ tự + thời gian ước tính

```
M1 (lookahead)  ──┐
                  ├─→ M2 (failure/target) ──→ M3 (publication UI) ──┐
                  │                                                  ├─→ nghiệm thu dashboard V3
                  └──────────────────────────→ M4 (auto + filter) ──┘
                                                                     │
                                              M5 (PDF) ──────────────┴─→ nâng Nấc 3 cho P0/P1
```

Thời gian ước tính (để tham khảo, không cam kết):
- M1: đang làm (Lớp A).
- M2: sau M1 ~ vài ngày.
- M3: song song M2 (~ vài ngày).
- M4: sau M3 (~ vài ngày).
- M5: sau M2, kéo dài (đọc PDF nhiều pattern).

---

## 4. Rủi ro + giảm thiểu

### R4.1. Đổi lookahead làm đảo thứ hạng pattern (NGHIÊM TRỌNG)

**Triệu chứng:** inside_day hiện "mạnh" (median MFE 15%) sẽ tụt xuống "yếu" (MFE ~3%) khi sửa lookahead=10. Scallop hiện "chưa hết move" sẽ thay đổi. Thứ hạng "best_historical_patterns" trên dashboard đảo.

**Giảm thiểu:**
- **Benchmark before/after bắt buộc** ở M1: snapshot profile cũ trước, so sánh sau.
- Đánh dấu "số liệu đã cập nhật V3" trên dashboard (timestamp) để người đọc biết thứ hạng mới là bản đúng.
- Giải thích thay đổi trong note phát hành (chứ không "im lặng" đảo hạng).

### R4.2. Số events giảm mạnh khi filter + giảm cap

**Triệu chứng:** filter MAE>80% + sửa lookahead (đoạn đầu đo ít bar hơn) + giảm cap bão hoà → một số pattern n giảm từ vài trăm → vài chục. Pattern hiếm có thể n<5 → bị ẩn.

**Giảm thiểu:**
- Report minh bạch: `n_before | n_after | drop_reason`.
- Không ẩn vội — chỉ ẩn pattern n<5, còn lại show + nhãn "mẫu mỏng".
- Pattern n<30 không đưa vào "best_historical" nhưng vẫn show trong hồ sơ.

### R4.3. Định nghĩa failure mới không khớp số liệu literature

**Triệu chứng:** bull_flags failure_busted_rate V3 có thể ra 8–12% (VN data) vs Bulkowski 5.5% (US data). Chênh có thể do: (a) thị trường VN khác, (b) detector nhận diện sai, (c) định nghĩa vẫn lệch.

**Giảm thiểu:**
- Đối chiếu M5 (PDF) bắt buộc cho pattern lên Nấc 3.
- Chênh ≤ 2× → chấp nhận "khác thị trường", ghi rõ.
- Chênh > 2× → audit detector + sample event thủ công (xem chart 10 event random).

### R4.4. Thay đổi pipeline gây stale dữ liệu tạm thời

**Triệu chứng:** giữa M1 và M3, dashboard có thể hiển thị số liệu "nửa vời" (lookahead mới nhưng failure cũ).

**Giảm thiểu:**
- Mỗi mốc rebuild toàn pipeline + ghi `pipeline_version` (v3-m1, v3-m2...) vào metadata.
- Dashboard cảnh báo nếu `pipeline_version` không nhất quán (ví dụ lookahead V3 nhưng failure chưa V3).
- Có thể "đóng băng" dashboard ở phiên bản V2 cho đến khi M3 xong → chỉ bật V3 khi đủ.

### R4.5. Publication nấc làm user nhầm "đã kiểm định" = "khả sinh lời"

**Triệu chứng:** chủ đầu tư non-code có thể hiểu "🟢 Đã kiểm định" = "mua được". Thực ra publication-final chỉ = tài liệu tham khảo đủ chuẩn, chưa phải tradable-final-95.

**Giảm thiểu:**
- Giữ nguyên non-advice boundary: "Hồ sơ hành vi lịch sử; không phải tín hiệu mua bán."
- Nhãn rõ: Nấc 3 = "đã kiểm định **như tài liệu tham khảo**", KHÔNG phải "đã kiểm định **khả sinh lời**".
- Mọi pattern V3 ghi `tradable_status = not_tested` (theo nguyên tắc 11 source-gate).

### R4.6. Agent song song sửa `scanner/v2/*.py` trong khi V3 đặc tả

**Triệu chứng:** đặc tả V3 ghi vào `docs/project/v3-design/` (read-only phần còn lại), nhưng Lớp A đang sửa detector. Có thể đặc tả và code thi công lệch nhau.

**Giảm thiểu:**
- V3 là **spec dẫn đường**, không phải code. Agent thi công đọc đặc tả → triển khai → báo cáo lệch (nếu có).
- Sau mỗi mốc, đối chiếu đặc tả vs code thực tế; cập nhật đặc tả nếu phát hiện spec sai.
- Không ghi đè code của agent thi công (quy tắc phạm vi trong prompt task).

---

## 5. Tóm tắt nghiệm thu V3 (tổng)

V3 được coi "hoàn thành đặc tả" khi 5 file này tồn tại + nhất quán:
- `00-overview.md` ✅
- `01-vision-scope.md` ✅
- `02-architecture-bottlenecks.md` ✅
- `03-measurement-standards.md` ✅
- `04-governance-roadmap.md` ✅ (file này)

V3 được coi "hoàn thành triển khai" khi:
- M1–M4 PASS (M5 là mở rộng dần).
- Dashboard nghiệm thu theo 6 tiêu chí ở `01-vision-scope.md` §5.
- Bảng đối chiếu before/after có sẵn cho chủ đầu tư duyệt.

Đặc tả này là **bản draft** — chủ đầu tư duyệt → agent thi công (deepseek-v4-flash hoặc tương đương) hiện thực hoá theo mốc.
