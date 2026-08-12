# BẢNG AUDIT DETECTOR — M0 (12/08/2026, sau commit Lớp A 3b52652)

| # | Detector (file) | Lookahead HIỆN TẠI | Nguồn (file:line) | Chuẩn spec digitized (average_days) | Chênh | Ghi chú |
|---|---|---|---|---|---|---|
| 1 | ascending_triangles.py | **126** | def local:353 | 60 | +66 | |
| 2 | broadening_patterns.py | **252** | call:471 | ? | ⚠️ ? | |
| 3 | bump_and_run.py | **252** | def local horizon:296 | ? | ⚠️ ? | |
| 4 | cup_with_handle.py | **252** | call:467 | 168 | +84 | |
| 5 | dead_cat_bounce.py | **63** | import pipes:51 (không truyền) | ? | ⚠️ ? | |
| 6 | descending_triangles.py | **126** | def local:356 | 60 | +66 | |
| 7 | diamonds.py | **252** | call:318 | ? | ⚠️ ? | |
| 8 | double_patterns.py | **252** | def local:424 | ? | ⚠️ ? | |
| 9 | falling_wedges.py | **126** | call:245 | ? | ⚠️ ? | |
| 10 | flags_experiment.py | **63** | def local:403 | 25 | +38 | |
| 11 | gaps.py | **63** | config.evaluation_bars:63 | ? | ⚠️ ? | |
| 12 | head_shoulders.py | **252** | def local:308 | ? | ⚠️ ? | |
| 13 | high_tight_flags.py | **63** | import flags_experiment:261 (không truyền) | ? | ⚠️ ? | |
| 14 | horns.py | **42** | call:338 | ? | ⚠️ ? | |
| 15 | inside_days.py | **10** | call:231 | 5 | +5 | |
| 16 | islands.py | **42** | config.evaluation_bars:57 | ? | ⚠️ ? | |
| 17 | measured_moves.py | **63** | def local:409 | ? | ⚠️ ? | |
| 18 | pennants.py | **63** | call:294 | 20 | +43 | |
| 19 | pipes.py | **63** | def local:429 | ? | ⚠️ ? | |
| 20 | rectangles.py | **252** | def local:345 | ? | ⚠️ ? | |
| 21 | rising_wedges.py | **126** | call:246 | ? | ⚠️ ? | |
| 22 | rounding.py | **252** | call:313 | ? | ⚠️ ? | |
| 23 | scallops.py | **252** | def local:448 | ? | ⚠️ ? | |
| 24 | symmetrical_triangles.py | **126** | def local:377 | 60 | +66 | |
| 25 | three_methods.py | **20** | call:389 | ? | ⚠️ ? | |
| 26 | three_peaks_valleys.py | **252** | call:273 | ? | ⚠️ ? | |
| 27 | triple_patterns.py | **252** | call:284 | ? | ⚠️ ? | |

**Kết luận audit:** 27 detector kiểm kê; 7 detector lệch so spec digitized (xem chi tiết M1).

### Các "?" cần bù (spec thiếu average_days — GLM đọc PDF bổ sung trong M5):
- Broadening Formations, Right-Angled and Ascending
- Broadening Formations, Right-Angled and Descending
- Broadening Wedges, Ascending and Descending
- Gaps (Breakaway, Common, Continuation, Exhaustion)
- Rounding Bottoms and Tops
- Scallop Ascending and Descending
- Three Falling Peaks
- Three Rising Valleys

### Ghi chú đặc biệt:
- dead_cat_bounce: import _evaluate_detection từ pipes (63) nhưng event-driven 3-phase — K3 chốt N/A, không dùng lookahead chuẩn
- high_tight_flags: import từ flags_experiment (63) nhưng spec mới pdfreview = 32 bars → sẽ sửa ở M2 khi nạp spec
- inside_days: Lớp A đặt 10 (chuẩn digitized ghi 5; PDF Harami 7-9) → M2 nạp spec PDF quyết
- horns: 42 (digitized 14; PDF 7-9) → M2 nạp spec PDF quyết
- triangles: 126 (digitized 60) → M1 registry so chuẩn PDF
- head_shoulders: 252 (digitized 79/73; PDF 107-176) → M1/M2 quyết
