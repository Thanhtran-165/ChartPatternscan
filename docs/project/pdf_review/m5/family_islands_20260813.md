# M5 — Trích số liệu PDF gốc: family ISLANDS

**Ngày trích:** 2026-08-13
**Model / Provider:** GLM-5.2 / builtin:zai-coding-plan (Z.AI Coding Plan)
**Session mẹ:** `sess_0fb10a3f-28e7-42c6-8bdd-6aca4b147362`
**Nguồn PDF:** `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` (ECP 2nd ed., 1.035 trang)
**Phương pháp:** `pdftotext -layout` (Poppler 26.04.0). Offset PDF ↔ sách in: **+23 trang**. Tự kiểm: chương 30 "Island Reversals" sách p464 → PDF p487; chương 31 "Islands, Long" sách p480 → PDF p503 (khớp offset +23).

---

## ⚠️ Phát hiện quan trọng (family ISLANDS có 2 chương riêng)

Bulkowski tách islands thành **2 chương riêng biệt** trong ECP:

| Chương | Pattern | Trang sách | Trang PDF | Đặc điểm |
|---|---|---|---|---|
| **30** | Island Reversals | p464-479 | p487-502 | 2 gap cùng giá (gaps share same price) — đảo chiều |
| **31** | Islands, Long | p480-495 | p503-518 | 2 gap khác giá (gaps at unequal prices) — tiếp tục |

File `islands_digitized.json` **KHÔNG phân biệt** 2 variant này — gộp chung "island top/bottom". Đây là thiếu sót vì 2 pattern có performance KHÁC NHAU (Island Reversals yếu hơn Islands Long về average rise).

---

## Bảng 1 — ISLAND REVERSALS (ECP chapter 30, PDF p487-502 / sách p464-479)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | |
| **pages_checked** | PDF **487-502** (16 trang) | sách p464-479 |
| **sample (Number of formations, Table 30.2)** | **345** (bull/Up) + **118** (bear/Up) + **289** (bull/Down) + **165** (bear/Down) = **917 islands** | Bulkowski: "917 islands without really trying", 500 stocks × 2 period (1991-96, 2000-02 bear) |
| **Reversal/Continuation** | 100% Reversal (345R / 118R / 289R / 165R) | Theo định nghĩa, đảo chiều |
| **failure break-even BE% (Table 30.3, mốc 5%)** | **18%** (bull/Up, 62/345) · **10%** (bear/Up, 12/118) · **17%** (bull/Down, 49/289) · **5%** (bear/Down, 8/165) | Best = bear/Down (5%); worst = bull/Up (18%) |
| **Failure rate đầy đủ (Table 30.3)** | 5%=18/10/17/5% · 10%=36/29/41/23% · 15%=47/47/55/33% · 20%=57/60/69/47% · 25%=63/66/75/56% · 30%=68/71/83/67% · 35%=72/75/88/78% · 50%=83/90/97/95% · 75%=92/99/100/100% | Thứ tự cột: bull/Up · bear/Up · bull/Down · bear/Down |
| **% meeting price target (Results Snapshot)** | Island Top (Downward): **62%** (bull) / **46%** (bear) · Island Bottom (Upward): **69%** (bull) / **49%** (bear) | Bulkowski: pattern mediocre performer |
| **Average rise/decline (Table 30.2)** | **+23%** (bull/Up) · **+21%** (bear/Up) · **−17%** (bull/Down) · **−23%** (bear/Down) | Đo breakout → ultimate high/low |
| **Days to ultimate high/low (Table 30.2)** | **128** (bull/Up) · **53** (bear/Up) · **53** (bull/Down) · **34** (bear/Down) | Bull/Up lâu gấp 4 lần bear/Down |
| **Measure rule** | Tính formation height = highest high − lowest low trong island. Add vào breakout point (hoặc subtract cho downtrend). **Ví dụ sách (Fig 30.5):** high 24.63 − low 17.50 = 7.13; target = 24.63 + 7.13 = **31.75** | Bulkowski: target thường KHÔNG đạt (pattern mediocre) |
| **Performance rank (Results Snapshot)** | Top: 21/21 (bull) · 13/21 (bear) · Bottom: 23/23 (bull) · 11/19 (bear) | Rank tồi — gần đáy danh sách |
| **Throwbacks/Pullbacks (Table 30.4)** | 70% (bull/Up) · 75% (bear/Up) · 65% (bull/Down) · 59% (bear/Down) | Rất cao — đảo chiều test island |
| **Avg time to throwback/pullback ends** | 9d / 6d / 8d / 8d | |
| **Avg rise/decline WITHOUT throwback/pullback** | 38% / 28%ᵃ / −23% / −30% | |
| **Avg rise/decline WITH throwback/pullback** | 18% / 19% / −14% / −19% | Throwback/pullback giảm performance 2x |
| **Busted pattern performance** | 42% / 23%ᵃ / −24% / −26%ᵃ | Bulkowski: "look elsewhere" cho busted |
| **Change after trend ends** | −28% / −37% / 45% / 46% | |

### Đối chiếu với `islands_digitized.json`

| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `post_breakout_measurement.failure_rate.at_5pct` | 18 | 18% (bull/Up) · 10% (bear/Up) · 17% (bull/Down) · 5% (bear/Down) — trung bình 4 tổ hợp = **12.5%** | 🟡 LỆCH nhẹ (18 vs trung bình 12.5) — digitized lấy 1 giá trị, không tách 4 tổ hợp |
| `post_breakout_measurement.failure_rate.at_10pct` | 10 | 10%=36/29/41/23% — trung bình **32.25%** | 🔴 LỆCH LỚN (10 vs 32.25) |
| `post_breakout_measurement.failure_definition.threshold_pct` | 2.0 | 5% (breakeven trong Table 30.3) | 🔴 LỆCH (2 vs 5) |
| `post_breakout_measurement.average_move.island_top_decline_pct` | 10 | **−17%** (bull/Down) · **−23%** (bear/Down) | 🔴 LỆCH LỚN (10 vs 17-23) |
| `post_breakout_measurement.average_move.island_bottom_rise_pct` | 10 | **+23%** (bull/Up) · **+21%** (bear/Up) | 🔴 LỆCH LỚN (10 vs 21-23) |
| `post_breakout_measurement.average_move.overall_pct` | 10 | Trung bình = (23+21+17+23)/4 = **21%** | 🔴 LỆCH LỚN (10 vs 21) |
| `post_breakout_measurement.ultimate_high_method.average_days` | 14 | **128** (bull/Up) · **53** (bear/Up) | 🔴 LỆCH LỚN (14 vs 53-128) |
| `post_breakout_measurement.ultimate_low_method.average_days` | 14 | **53** (bull/Down) · **34** (bear/Down) | 🔴 LỆCH LỚN (14 vs 34-53) |
| `post_breakout_measurement.lookahead_bars` | 42 | Trung bình days to ultimate = (128+53+53+34)/4 = **67** | 🟡 LỆCH nhẹ (42 vs 67) |
| `post_breakout_measurement.target_calculation.method` | `gap_height` | **pattern_height** = highest high − lowest low (KHÔNG phải gap height) | 🔴 **SAI METHOD** — digitized dùng gap height, PDF dùng formation height |
| `post_breakout_measurement.target_calculation.formula_island_top` | `mainland_after − (island_price − mainland_before)` | `target = breakout_point + (highest_high − lowest_low)` (up) / đối xứng (down) | 🔴 **SAI CÔNG THỨC** |
| `performance_statistics.failure_rate_5pct` | 18 | Trung bình 12.5% | 🟡 LỆCH nhẹ |
| `performance_statistics.failure_rate_10pct` | 10 | Trung bình 32.25% | 🔴 LỆCH LỚN |
| `performance_statistics.average_move_pct` | 10 | **21%** (trung bình 4 tổ hợp) | 🔴 LỆCH LỚN (10 vs 21) |
| `performance_statistics.median_move_pct` | 8 | **KHÔNG TÌM THẤY** (PDF không báo median) | 🔴 **BỊA** |
| `performance_statistics.time_to_completion_days` | 5 | **KHÔNG TÌM THẤY** (PDF chỉ có "days to ultimate" = 34-128) | 🔴 **BỊA** |
| `performance_statistics.throwback_rate_pct` | 30 | 70% / 75% / 65% / 59% — trung bình **67%** | 🔴 LỆCH LỚN (30 vs 67) |
| `performance_statistics.gap_size_effect` | "Larger gaps (>2%) 75% success vs 60%" | **KHÔNG TÌM THẤY** (PDF không báo % success theo gap size) | 🔴 **BỊA** |
| `performance_statistics.duration_effect` | "Single-day islands 70% success vs 55% multi-day" | **KHÔNG TÌM THẤY** | 🔴 **BỊA** |
| `variant_handling.variants[].island_top` | có | PDF: Island Top = Downward breakout (matches) | 🟢 KHỚP (semantic) |
| `variant_handling.variants[].island_bottom` | có | PDF: Island Bottom = Upward breakout (matches) | 🟢 KHỚP |
| Sample size | KHÔNG ghi | **917** islands | 🔴 THIẾU hoàn toàn |
| `% meeting price target` | KHÔNG có trường | **62/46/69/49%** | 🔴 THIẾU hoàn toàn |

**Tóm tắt lệch Island Reversals:** 🔴 **LỆCH NGHIÊM TRỌNG** về (1) average move (10% vs 21% thực — thấp gấp đôi), (2) days to ultimate (14 vs 34-128), (3) measure rule method (gap height vs formation height — sai concept), (4) throwback rate (30 vs 67%), (5) sample (không ghi 917). Nhiều trường BỊA (median_move, time_to_completion, gap_size_effect, duration_effect).

---

## Bảng 2 — ISLANDS, LONG (ECP chapter 31, PDF p503-518 / sách p480-495)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | như trên | |
| **pages_checked** | PDF **503-518** (16 trang) | sách p480-495 |
| **sample (Number of formations, Table 31.2)** | **255** (bull/Up) + **206** (bear/Up) + **193** (bull/Down) + **266** (bear/Down) = **920 long islands** | Bulkowski: "found the pattern in 220 stocks in two groups" |
| **Reversal/Continuation** | Hỗn hợp (cả R và C) — Table 31.2 chi tiết | Khác Island Reversals (100% R) — Long island thường tiếp tục trend |
| **failure break-even BE% (Table 31.3, mốc 5%)** | **11%** (bull/Up, 27/255) · **4%** (bear/Up, 9/206) · **5%** (bull/Down, 9/193) · **2%** (bear/Down, 5/266) | Best = bear/Down (2%); Island Long BE thấp hơn Island Reversals |
| **Average rise/decline (Table 31.2)** | **+31%** (bull/Up) · **+25%** (bear/Up) · **−22%** (bull/Down) · **−26%** (bear/Down) | Long island perform tốt hơn Island Reversals |
| **Days to ultimate high/low (Table 31.2)** | **67** (bull/Up) · **33** (bear/Up) · **27** (bull/Down) · **26** (bear/Down) | Nhanh hơn Island Reversals |
| **% meeting price target (Results Snapshot)** | Upward: **82%** (bull) / **72%** (bear) — dùng half formation height · Downward: **78%** (bull) / **76%** (bear) | Long island đạt target tốt hơn Island Reversals nhiều |
| **Throwbacks/Pullbacks (Results Snapshot)** | 67% / 74% (Up) · 54% / 54% (Down) | |
| **Measure rule** | Dùng **half formation height** (không phải full height như Island Reversals) — vì long island dài hơn, full height target khó đạt | Bulkowski chỉ rõ "using half the formation height" |

### Đối chiếu Islands Long với Island Reversals (cùng family)

| Metric | Island Reversals (ch.30) | Islands Long (ch.31) | Nhận xét |
|---|---|---|---|
| **Sample** | 917 | 920 | Gần bằng nhau |
| **BE failure rate (trung bình 4 tổ hợp)** | (18+10+17+5)/4 = **12.5%** | (11+4+5+2)/4 = **5.5%** | Long island an toàn hơn (BE thấp hơn) |
| **Average rise bull/Up** | +23% | +31% | Long island perform tốt hơn |
| **Average decline bear/Down** | −23% | −26% | Long island sâu hơn chút |
| **Days to ultimate (trung bình)** | (128+53+53+34)/4 = **67d** | (67+33+27+26)/4 = **38d** | Long island nhanh hơn |
| **% meeting target** | 62/46/69/49% | 82/72/78/76% | Long island đạt target cao hơn rõ rệt |
| **Measure rule** | Full formation height | **Half formation height** | Khác method — long island dài nên dùng nửa |

**Kết luận so sánh:** **Islands Long perform tốt hơn Island Reversals trên mọi metric** (BE thấp hơn, average rise cao hơn, %target cao hơn, nhanh hơn). File digitized gộp chung 2 pattern → mất thông tin quan trọng này.

---

## Bằng chứng verbatim (số liệu thô, ≤3 dòng, bản quyền)

### Island Reversals Results Snapshot (PDF p487-489 / sách p464-466)
```
Tops (Downward Breakouts):   BE 17%/5%, decline 17%/23%, pullback 65%/59%, %target 62%/46%
Bottoms (Upward Breakouts):  BE 18%/10%, rise 23%/21%, throwback 70%/75%, %target 69%/49%
```

### Island Reversals Table 30.2 (PDF p494 / sách p471)
```
Number of formations         345    118    289    165
Average rise or decline      23%    21%    -17%   -23%
Days to ultimate high/low    128    53     53     34
```

### Island Reversals Table 30.3 Failure Rates (PDF p495 / sách p472)
```
5 (breakeven)   62 or 18%   12 or 10%   49 or 17%   8 or 5%
10              125 or 36%  34 or 29%   119 or 41%  38 or 23%
```

### Islands Long Table 31.2 (PDF p510 / sách p487)
```
Number of formations         255    206    193    266
Average rise or decline      31%    25%    -22%   -26%
Days to ultimate high/low    67     33     27     26
```

### Islands Long Results Snapshot (PDF p503-505 / sách p480-482)
```
Upward:   BE 11%/4%, rise 31%/25%, %target (half height) 82%/72%
Downward: BE 5%/2%, decline 22%/26%, %target (half height) 78%/76%
```

---

## Reproducer

```bash
PDF="/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet/Nghiên cứu mô hình nến/references/encyclopedia-of-chart-patterns-2nbsped-9786468600-3175723993-9780471668268-0471668265_compress.pdf"

# Island Reversals (ch.30, PDF p487-502)
pdftotext -layout -f 487 -l 489 "$PDF" - | sed -n '/RESULTS SNAPSHOT/,/Tour/p'
pdftotext -layout -f 494 -l 495 "$PDF" - | sed -n '/Table 30.2/,/Table 30.3/p'
pdftotext -layout -f 495 -l 496 "$PDF" - | sed -n '/Failure Rates/,/Table 30.4/p'

# Islands Long (ch.31, PDF p503-518)
pdftotext -layout -f 503 -l 505 "$PDF" - | sed -n '/RESULTS SNAPSHOT/,/Tour/p'
pdftotext -layout -f 510 -l 511 "$PDF" - | sed -n '/Table 31.2/,/Table 31.3/p'
```

---

## Khuyến nghị (không thực hiện trong task này)

1. **Tách islands thành 2 variant riêng**: Island Reversals (ch.30, 2 gap cùng giá) vs Islands Long (ch.31, 2 gap khác giá). Hiện digitized gộp chung — mất thông tin performance khác biệt (Long perform tốt hơn Reversals).
2. **Sửa average_move**: digitized ghi 10% — PDF: Reversals 21% (avg 4 tổ hợp), Long 26% (avg). Lệch gấp đôi.
3. **Sửa days to ultimate**: digitized ghi 14 — PDF: Reversals 34-128d, Long 26-67d. Lệch lớn.
4. **Sửa measure rule**: digitized dùng `gap_height` — PDF dùng **formation height** (highest high − lowest low). Islands Long dùng **half formation height**.
5. **Sửa throwback rate**: digitized ghi 30% — PDF: Reversals 59-75%, Long 54-74%. Lệch lớn.
6. **Bổ sung sample**: Reversals 917, Long 920.
7. **Bổ sung % meeting target**: Reversals 46-69%, Long 72-82%.
8. **Xóa các trường BỊA**: median_move, time_to_completion, gap_size_effect, duration_effect — không có trong PDF.

---

**Hết file.**
