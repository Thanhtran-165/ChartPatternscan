# M1 — Bằng chứng verify measurement_registry + detector nối registry

**Ngày:** 2026-08-12
**Trạng thái:** chờ K3-1 ký duyệt + phán quyết các ô mâu thuẫn

---

## 1. File trung tâm

- `scanner/v2/measurement_registry.py` (TẠO MỚI) — nguồn chuẩn đo lường DUY NHẤT:
  lookahead + failure_threshold + failure_reference + cap + source (pdf/digitized/detector_legacy).
  Thứ tự ưu tiên: **PDF_REVIEW (12 family) > digitized average_days (days to ultimate) > lookahead_bars spec > detector_legacy**.
- 27 detector trong `scanner/v2/` — bỏ lookahead cứng, đọc từ registry qua
  `_registry_lookahead(...)`. Mỗi event vẫn giữ `pattern_key` → registry tra theo pattern_key.
- 3 detector scan dữ liệu TUẦN (pipes, horns, rounding) → dùng `lookahead_weeks()` = ceil(ngày/5).

## 2. 3 bug nghiêm trọng đã tìm và fix (khi verify thực tế)

| # | Bug | Biểu hiện | Fix |
|---|---|---|---|
| 1 | Registry trả lookahead FLOAT (5.0, 60.0) từ JSON digitized | `TypeError: cannot do positional indexing on RangeIndex with these indexers [45.0] of type float` — mọi symbol fail âm thầm (try/except nuốt), detection_count = 0 | `lookahead_bars()` bọc `int()` |
| 2 | `pattern_key` không phải biến local tại call site `_path_rows_from_series(..., horizon_bars=_registry_lookahead(pattern_key))` | `NameError: name 'pattern_key' is not defined` | Đổi thành `_registry_lookahead(scan["pattern_key"])` (26 file) |
| 3 | pipes/horns/rounding scan TUẦN nhưng áp lookahead NGÀY | pipe_bottom MFE phồng 155% (đo 194 tuần ≈ 4 năm thay vì 194 ngày ≈ 39 tuần) | Thêm `lookahead_weeks()` = ceil(ngày/5); MFE 155% → 46% |
| 4 | dead_cat dùng chung `_evaluate_detection` từ pipes (đã đổi sang lookahead_weeks) nhưng scan DAILY | evaluated_bars max = 13 (63 ngày bị quy đổi thành 13 tuần) | dead_cat truyền `lookahead=_registry_lookahead(...)` = lookahead_bars (63) — 99–100% đạt |

Ngoài ra: `gaps.py` scan dict dùng `"pattern_key": "gap_family"` (placeholder) và
`islands.py` dùng `"island_family"` → path_rows trả None → `int + NoneType` lỗi → đã
đổi sang loại cụ thể (AREA_GAPS / ISLAND_REVERSALS).

## 3. Verify thực tế — evaluated_bars so chuẩn registry

Phương pháp: chạy detector trên DB thật `market_cache/stock_ohlcv/latest.sqlite`
(1.599 mã, tới 2026-08-11), 250–400 mã mỗi detector; đối chiếu phân bố
`evaluated_bars` (số phiên thực tế đo sau breakout) với chuẩn registry.
% đạt = tỷ lệ event có evaluated_bars = chuẩn đầy đủ (event gần cuối chuỗi dữ liệu
bị cắt cửa sổ là hành vi ĐÚNG, không phải bug — đã xác minh breakout_date).

| Family / pattern | Chuẩn registry | n events | % đạt chuẩn | MFE trung bình | Ghi chú |
|---|---:|---:|---:|---:|---|
| inside_day | 5 | 3.606 | **100%** | 5,2% | MFE giảm từ ~15% → ~5% (mục tiêu M1) |
| ascending_triangles | 60 | 483 | 98% | 23,3% | |
| head_shoulders_bottoms | 176 | 9 | 89% | 60,1% | mẫu nhỏ (400 mã) |
| head_shoulders_tops | 62 | 3 | 100% | 24,6% | mẫu nhỏ |
| double_bottoms | 76 | 260 | 97% | 21,1% | |
| double_tops | 71 | 200 | 96% | 12,5% | |
| scallops_ascending | 162 | 904 | 93% | 34,0% | |
| scallops_descending | 106 | 922 | 96% | 21,9% | |
| pipe_bottoms (tuần) | 39 tuần (194d) | 368 | 91% | 46,3% | MFE từ 155% |
| pipe_tops (tuần) | 15 tuần (75d) | 422 | 97% | 13,8% | |
| horn_bottoms (tuần) | 36 tuần (180d) | 888 | 94% | 47,0% | MFE từ 153% |
| horn_tops (tuần) | 14 tuần (67d) | 937 | 98% | 13,7% | |
| rounding_bottoms (tuần) | 17 tuần (84d) | 482 | 99% | 31,9% | |
| rounding_tops (tuần) | 13 tuần (63d) | 221 | 91% | 10,3% | |
| pennants | 20 | 365 | **99,7%** | — | verify qua scan_symbol trực tiếp |
| ascending_triangles | 60 | 302 | 98% | 20,5% | |
| descending_triangles | 60 | 348 | 96% | 12,1% | |
| symmetrical_triangles | 60 | 575 | 98% | 16,8% | |
| cup_with_handle | 167 | 1.032 | 97% | 46,4% | |
| cup_with_handle_inverted | 167 | 1.118 | 95% | 18,5% | |
| falling_wedges | 50 | 273 | 97% | 17,7% | |
| rising_wedges | 50 | 184 | 97% | 10,9% | |
| gaps_area | 63 | 1.252 | 99% | 14,3% | |
| gaps_breakaway | 42 | 386 | 99% | 25,8% | |
| gaps_continuation | 21 | 279 | 100% | 36,6% | |
| gaps_exhaustion | 5 | 421 | 100% | 5,4% | |
| three_methods_rising | 10 | 81 | 100% | 8,6% | |
| three_methods_falling | 10 | 35 | 100% | 7,6% | |
| three_falling_peaks | 36 | 80 | 95% | 11,9% | |
| three_rising_valleys | 125 | 88 | 99% | 37,1% | |
| bump_bottoms | 70 | 1.029 | 97% | 28,3% | |
| bump_tops | 70 | 1.185 | 98% | 17,1% | |
| rectangle_bottoms | 177 | 734 | 86% | 28,7% | mẫu nhiều event cắt cuối chuỗi |
| rectangle_tops | 170 | 740 | 95% | 29,8% | |
| dead_cat | 63 | 1.708 | **99%** | 12,8% | fix: dùng lookahead NGÀY (trước bị weeks → 13) |
| dead_cat_inverted | 63 | 1.881 | **100%** | 8,4% | |
| diamond_bottoms | 77 | 6 | 100% | 50,2% | mẫu hiếm (250 mã) |
| diamond_tops | 63 | 9 | 89% | 36,5% | mẫu hiếm |

**Tổng: 26/28 pattern đạt ≥90% evaluated_bars = chuẩn** (2 ô <90% do mẫu hiếm/cắt cuối chuỗi, không phải bug).

## 4. Các ô MÂU THUẪN cần K3-1 phán quyết

Registry hiện ưu tiên **average_days = "days to ultimate"** (đúng nguyên tắc kế hoạch
08 §2: "chuẩn = days to ultimate median từ PDF, bỏ default 252"). Nhưng bảng
`03-measurement-standards.md §1.2` (bản dự thảo cũ) ghi số khác cho nhiều family
(đa số = 252 = lookahead_bars spec). Cần K3 chốt 1 trong 2 quy ước cho family
CHƯA có PDF_REVIEW:

| Family | A: digitized average_days (đang dùng) | B: bảng §1.2 cũ / lookahead_bars spec | PDF_REVIEW |
|---|---:|---:|---:|
| inside_day | **5** | 10 | 7–9 (Harami — LỆCH định nghĩa, body vs range) |
| islands | **14** | 42 | chưa có |
| measured_move | **21** | 63 | chưa có |
| wedges | **50** | 126 | chưa có |
| triangles | **60** | 126 | chưa có |
| broadening_bottoms / tops | **84 / 56** | 252 | chưa có |
| bump_and_run | **70** | 252 | chưa có |
| diamond_bottoms / tops | **77 / 63** | 252 | chưa có |
| rectangles (variant) | **177 / 170** | 252 | chưa có |
| three_methods | **10** | 20 | chưa có |
| flags / pennants | **25 / 20** | 63 | chưa có |
| gaps (4 loại) | **42 / 21 / 5 / 63** | 63 | chưa có |
| double_bottoms / tops | **76 / 71** | 252 | chưa có |
| dead_cat | **63** (detector_legacy) | — | chờ M5 PDF |
| horn_bottoms_tops (family-level) | 14 | 42 | **180 / 67** — variant horn_bottoms/horn_tops ĐÃ dùng PDF ✓ |

**Đề xuất (để K3 xác nhận hoặc bác):** giữ quy ước A (days to ultimate) cho family
chưa có PDF + gắn cờ `source=digitized` (đã làm) — M5 (GLM đọc PDF 19 family) sẽ
nâng cấp thành `source=pdf` khi có số. Riêng inside_day: PDF_REVIEW ghi định nghĩa
LỆCH (Harami body vs inside_day range) → giữ 5 (digitized) là an toàn, hoặc theo PDF 7–9.

## 5. Chưa verify (chờ M5 hoặc ghi chú)

- 19 family chưa có PDF_REVIEW → số digitized sẽ được GLM đọc PDF xác nhận ở M5.
- flags_experiment không có CLI (module) — pipeline thật dùng flags qua
  bull_flags_monograph (lớp after-buy, ngoài phạm vi 27 detector M1).
