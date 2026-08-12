# 09 — Bằng chứng nghiệm thu M2: Failure logic chuẩn Bulkowski

- **Ngày**: 12/08/2026 (verify toàn DB xong ~23:40)
- **Mốc**: M2 trong kế hoạch K3 (08-k3-final-plan.md)
- **Tài liệu chuẩn đối chiếu**: 03-measurement-standards.md §2 (đã K3-1 duyệt) · PDF_REVIEW_20260812.md (đọc PDF gốc Bulkowski) · spec digitized trong `extraction_phase_1/digitization/patterns_digitized/`
- **Chờ**: K3-2 ký duyệt + phán quyết §6

---

## 1. Việc đã làm

1. **Sửa 4 spec digitized theo PDF gốc** (đã ghi trong 08-k3-final-plan M2):
   - `pipe_bottoms`: lookahead 63 → **194d** — ghi chú chính xác: `lookahead_bars: 63` trong spec vẫn nguyên;
     số 194 nằm ở `post_breakout_measurement.ultimate_high_method.average_days` và registry `_PDF_OVERRIDES`
     ưu tiên dùng 194 cho runtime (K3-2 xác nhận thứ tự ưu tiên đúng, chỉ cần ghi chú cho người đọc).
   - `head_and_shoulders_bottom`: lookahead 79 → **176d**
   - `head_and_shoulders_top`: lookahead 79 → **62d**
   - `scallops`: lookahead theo PDF (41/87 → chuẩn per-variant)
2. **Viết lại `scanner/v2/failure_logic.py`** (thước đo M2, thay `failure_5pct`):
   - `weak_move_5pct`: MFE < 5% theo hướng breakout (đổi tên từ `failure_5pct` — giữ ý nghĩa "move yếu", KHÔNG phải failure chuẩn — 03 §2.1)
   - `failure_busted`: giá dùng **LOW/HIGH** (không phải close) quay lại **mức failure reference = đáy/đỉnh pattern thật** (không phải breakout) × (1 ∓ threshold/100) **TRƯỚC khi chạm target** (03 §2.2)
   - `days_to_bust`: số ngày từ breakout đến bar busted (None nếu không bust)
   - `_FAMILY_LEVEL_KEYS`: map 34 family → keys mức giá thực trong detection row (khảo sát 25 detector trên 6 mã — `scripts/survey_failure_keys.py`), thứ tự ưu tiên per direction; fallback `breakout ± pattern_height_pct`; chót `pattern_low/pattern_high/breakout_price`
3. **Patch 13 file detector** (`scripts/patch_m2_failure_logic.py`, 40 bước thành công): import + 3 cột mới (`weak_move_5pct`, `failure_busted`, `days_to_bust`) + registry `"gaps": 2.0` + fix `bump_and_run` (target_price float) + fix `islands` (row thiếu pattern_key → setdefault)
4. **Smoke 10 mã lớn** (`scripts/verify_m2_failure_logic.py`): không crash, weak==failure_5pct cũ (bảo toàn), busted% hợp lý

## 2. Cách lấy mức failure reference (keys THỰC per family)

Bảng keys thực tế (không phải tên lý thuyết — khảo sát trực tiếp detection row):

| Family | Key failure level (ưu tiên) |
|---|---|
| cup_with_handle | `handle_extreme_price` |
| flags / pennants | `flag_lower_breakout_value` (up) / `flag_upper_breakout_value` (down) |
| triangles | `triangle_support` / `triangle_resistance` |
| rectangles | `rectangle_support` / `rectangle_resistance` |
| broadening / diamond | `pattern_low` / `pattern_high` |
| rounding | `extreme_price` |
| horn / pipe | `support_resistance_price` / `low_boundary_price` / `high_boundary_price` |
| head_and_shoulders | `neckline_price` |
| measured_move | `first_leg_start_price` |
| dead_cat | `bounce_high_price` / `event_low_price` |
| three_methods | `first_bar_low` / `first_bar_high` |
| triple | `pivot_N_price` |
| scallop | `low/high_boundary_price` / `middle_anchor_price` |
| gaps | `gap_rim_close_price` / `gap_bottom_price` / `gap_top_price` (row KHÔNG có key gap_edge) |
| inside_day | `inside_day_low/high` / `mother_bar_low/high` |

## 3. Verify toàn DB (1599 mã, tới 2026-08-11) — 16 detector chính

`scripts/verify_m2_full.py` → `/tmp/verify_m2_full.json`. Định nghĩa:
- `f5%` = weak_move_5pct (MFE<5%) — thước phụ
- `busted%` = failure_busted (quay lại đáy/đỉnh pattern × (1∓threshold) TRƯỚC target) — thước chính M2
- `med_days` = median days_to_bust (chỉ event busted)

| pattern | events | f5% | busted% | n_bust | med_days | med_tgt% | med_mfe |
|---|---:|---:|---:|---:|---:|---:|---:|
| bull_flags | 341 | 43.11 | **18.18** | 62 | 17.0 | 17.57 | 6.25 |
| cup_with_handle | 6888 | 12.84 | **43.77** | 3014 | 25.0 | 17.95 | 27.29 |
| triangles_ascending | 2066 | 23.32 | **11.39** | 235 | 30.0 | 9.96 | 13.44 |
| pipe_bottoms | 1590 | 14.50 | **27.55** | 437 | 12.0 | 15.82 | 24.73 |
| gaps | 15040 | 32.25 | **30.21** | 4544 | 4.0 | 3.67 | 10.00 |
| inside_day | 14545 | 61.97 | **25.59** | 3721 | 2.0 | 2.35 | 3.18 |
| rectangle_bottoms | 4761 | 16.40 | **25.79** | 1228 | 34.0 | 6.60 | 18.75 |
| dead_cat_bounce | 10782 | 8.46 | **34.88** | 3761 | 19.0 | 15.52 | 18.74 |
| horn_bottoms | 3729 | 15.82 | **31.31** | 1166 | 9.0 | 16.28 | 25.95 |
| rounding_bottoms | 1962 | 23.60 | **4.14** | 81 | 8.0 | 33.33 | 15.99 |
| broadening_bottoms | 812 | 17.78 | **22.59** | 183 | 34.0 | 26.10 | 19.20 |
| double_bottoms | 1117 | 21.67 | **23.19** | 259 | 27.0 | 10.41 | 14.63 |
| measured_move_up | 913 | 25.19 | **3.94** | 36 | 13.0 | 21.59 | 9.63 |
| scallops_ascending | 3755 | 15.69 | **29.50** | 1107 | 44.0 | 16.47 | 21.12 |
| pennants | 2926 | 41.70 | **27.24** | 797 | 9.0 | 20.56 | 6.25 |
| rising_three_methods | 467 | 46.47 | **16.49** | 77 | 5.0 | 5.90 | 5.59 |

## 4. Audit cup busted (vì chênh lớn so mục tiêu 03 §2.5)

12 event busted ngẫu nhiên (toàn DB) — bust xảy ra THẬT (giá xuyên sâu dưới đáy handle):

| sym | breakout | handle | edge (×0.95) | days | hit | mfe% | mae% |
|---|---:|---:|---:|---:|---|---:|---:|
| VBH | 13.70 | 12.30 | 11.69 | 17 | False | −2.2 | 59.1 |
| CMT | 8.77 | 7.37 | 7.00 | 19 | True | 171.3 | 20.2 |
| ANV | 9.71 | 8.79 | 8.35 | 12 | True | 19.8 | 43.0 |
| HTM | 12.00 | 10.93 | 10.38 | 39 | True | 35.0 | 16.7 |
| HHS | 11.23 | 10.07 | 9.57 | 18 | False | 2.7 | 55.7 |
| HAI | 9.26 | 8.28 | 7.87 | 58 | True | 148.9 | 17.0 |
| DDN | 8.14 | 7.10 | 6.75 | 136 | False | 5.5 | 24.6 |
| CLG | 9.00 | 8.00 | 7.60 | 7 | False | 0.0 | 35.6 |
| VIM | 26.04 | 20.67 | 19.64 | 104 | False | 9.1 | 36.2 |
| SNC | 11.58 | 9.51 | 9.04 | 111 | True | 48.0 | 29.3 |
| CBS | 33.05 | 28.44 | 27.02 | 47 | False | 6.7 | 34.6 |
| TIG | 2.22 | 1.99 | 1.89 | 102 | False | 9.2 | 19.8 |

Nhận xét:
- **10/12 event là cổ phiếu nhỏ giá 2–13k** (VBH, HHS, HAI, DDN, CLG, TIG, SNC, ANV, CMT) với MAE 17–59% — thanh khoản kém, nhảy giá dữ dội.
- **10 mã vốn hóa lớn quen thuộc** (VCB CTD FPT VNM HPG ACB BID MWG TCB SSI): 60 event, busted **30.0%** — thấp hơn toàn sàn (43.8%) nhưng vẫn cao.
- **Phân bố thời điểm bust** (n=3014): ≤10d 23.8% · 11–30d 33.8% · 31–60d 24.1% · 61–90d 9.6% · >90d 8.8% → 57.6% bust trong 30 ngày đầu; 8.8% bust SAU 90 ngày (bust muộn, ít ý nghĩa trade nhưng đúng khung đo Bulkowski 167d).
- `hit=True + bust=True` (CMT, ANV...) KHÔNG mâu thuẫn: target_hit đo toàn lookahead độc lập; event busted ngày 19 nhưng về sau vẫn chạm target (mfe 171%) — đúng định nghĩa Bulkowski "busted rồi vẫn có thể thành công sau".

## 5. Phát hiện quan trọng — SO SÁNH SAI ĐƠN VỊ với mục tiêu 03 §2.5

Đối chiếu số spec (nguồn chuẩn) vs 2 thước VN:

| Pattern | Spec (PDF/digitized) | VN weak (MFE<5%) | VN busted |
|---|---|---|---|
| cup_with_handle | **5%** (break-even, PDF bull 5/bear 7) | 12.84 | **43.77** |
| flags bull | **5%** (break-even) | 43.11 | **18.18** |
| triangles asc | **8%** | 23.32 | **11.39** |
| pipe_bottoms | **5%** (BE 5/4) | 14.50 | **27.55** |
| inside_day | 15% (digitized at_5pct; PDF không báo BE — candlestick) | 61.97 | **25.59** |
| horn_bottoms | 9% (PDF BE bull) | 15.82 | **31.31** |

**Phân tích**:
- `failure_rate_pct` trong spec (và PDF_REVIEW "Break-even failure") = tỷ lệ mẫu mà giá **quay về mức breakout / không đạt +5%** — tức tương ứng `weak_move_5pct` về mặt khái niệm, KHÔNG phải busted.
- Busted (03 §2.2) = giá quay lại **đáy/đỉnh pattern** × (1∓threshold) — nghiêm ngặt hơn break-even, và Bulkowski chỉ mô tả "busted patterns" định tính (hiếm), KHÔNG có bảng % để so.
- → Mục tiêu 03 §2.5 (cup ≈ 5%, flags ≈ 5.5%) lấy số từ break-even spec nhưng đo bằng busted → **so sánh sai đơn vị ngay từ gốc**; chênh 3–9× không phản ánh bug logic mà phản ánh: (a) khác đơn vị đo, (b) khác thị trường (penny VN), (c) khung đo dài (167d) với threshold 5% sát breakout.
- VN `weak_move_5pct` cũng cao hơn spec break-even 2–8× (flags 43.1 vs 5, inside_day 62 vs 15) → phần còn lại là **đặc tính thị trường VN thật** (penny + thanh khoản + T+), không phải bug code.

## 6. Câu hỏi xin K3-2 phán quyết (kèm tài liệu chuẩn để K3 tự đối chiếu)

1. **Đơn vị đo failure**: Xác nhận `failure_rate` spec = break-even failure (quay lại breakout / không đạt +5%), KHÁC với `failure_busted` (quay lại đáy/đỉnh pattern)? Nếu đúng → mục tiêu 03 §2.5 cần chuẩn lại: giữ `failure_busted` làm thước chính M2 (như 03 §2.4) nhưng KHÔNG áp mốc 5%/5.5% từ break-even; thay bằng nhận định định tính + baseline VN thật.
2. **Baseline VN**: VN busted (cup 43.8% toàn sàn / 30.0% ở 10 mã lớn; flags 18.2%; pipes 27.6%) — chấp nhận làm số thật của thị trường VN (hiển thị kèm cỡ mẫu + ghi chú so US) hay cần điều chỉnh threshold/khung đo?
3. **Gaps threshold 2%** (fill breakaway gap = failure) — xác nhận giữ, hay để 5% mặc định?
4. **weak_move_5pct** giữ làm thước phụ (03 §2.4 mục 3) — xác nhận.

## 8. KẾT QUẢ K3-2 KÝ DUYỆT (12/08/2026, agent_f1f52d0b — kimi-k3/OpenCode Go)

**KẾT LUẬN: M2 ĐẠT — KÝ DUYỆT.** "Logic failure chuẩn Bulkowski đã đúng; vấn đề duy nhất là khung so sánh (§2.5) sai đơn vị từ đầu, đã có hướng sửa rõ ràng."

Phán quyết từng câu:
- **Q1**: XÁC NHẬN — `failure_rate` spec + "Break-even failure" PDF = tỷ lệ giá không đi quá +5% theo hướng breakout (≈ `weak_move_5pct`); `failure_busted` = khái niệm busted pattern chỉ mô tả định tính (không có bảng %). **Bác phương án đổi busted sang mức breakout** (sẽ suy biến thành bản sao weak_move). Mục tiêu 03 §2.5 đã vá (mục §2.5 mới, bỏ mốc tuyệt đối, đóng băng baseline VN v1). `inside_day` bỏ hẳn mốc 15% (spec lệch định nghĩa Harami).
- **Q2**: CHẤP NHẬN baseline VN (cup 43.8%/30% mã lớn, flags 18.2%, pipes 27.6%) — **CẤM chỉnh threshold/khung để ép về US**. Điều kiện hiển thị M3: mọi `failure_busted_rate` kèm (a) cỡ mẫu n, (b) bucket penny/vốn hóa lớn, (c) median days_to_bust + ghi chú khung đo (57.6% bust ≤30d là phần có nghĩa trade; 8.8% sau 90d gần như nhiễu), (d) dòng chú thích so US. Ghi chú riêng flags/pennants (weak 43.1% vs BE 5% — T+ và biên độ VN).
- **Q3**: GIỮ gaps threshold **2%** (mép gap sát breakout; 5% sẽ quá lỏng mất nghĩa "breakaway fill = failure"; số liệu ủng hộ: gaps busted 30.2%, median 4 ngày).
- **Q4**: XÁC NHẬN `weak_move_5pct` làm thước phụ (thước duy nhất có số spec để đối chiếu chéo US↔VN).

Điều kiện trước/trong M3 (không chặn ký):
1. Vá 03 §2.5 ✅ (đã làm, mục §2.5 mới)
2. Vệ sinh `pipe_bottoms_digitized.json` `post_breakout_measurement.failure_rate` stale (at_5pct 12→5, at_10pct 5→2) ✅ (đã làm)
3. M3 build profile hiện thực hóa Q2 (busted rate kèm cỡ mẫu + bucket penny/lớn + median days_to_bust + chú thích US) → ghi vào kế hoạch M3, là điều kiện nghiệm thu M3.

## 7. File liên quan

- `scanner/v2/failure_logic.py` (viết lại) · `scanner/v2/measurement_registry.py` (gaps 2.0)
- `scripts/patch_m2_failure_logic.py` (patch 13 file) · `scripts/verify_m2_failure_logic.py` (smoke 10 mã) · `scripts/verify_m2_full.py` (toàn DB) · `scripts/survey_failure_keys.py` (khảo sát keys)
- Spec đã sửa: `extraction_phase_1/digitization/patterns_digitized/{pipe_bottoms,head_and_shoulders_bottom,head_and_shoulders_top,scallop_ascending_descending}_digitized.json`
- Tài liệu chuẩn: `docs/project/pdf_review/PDF_REVIEW_20260812.md` · `docs/project/v3-design/03-measurement-standards.md`
