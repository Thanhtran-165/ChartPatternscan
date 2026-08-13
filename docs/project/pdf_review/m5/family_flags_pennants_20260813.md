# M5 — Trích số liệu PDF gốc: family FLAGS + PENNANTS

**Ngày trích:** 2026-08-13
**Model / Provider:** GLM-5.2 / Z.AI (zai-coding-plan)
**Session mẹ:** `sess_subagent_agent_b2b052bd-6a1c-44c2-889e-c6aef5f5390f`
**Nguồn PDF:** `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` (1.035 trang)
**Phương pháp:** `pdftotext -layout` (Poppler 26.04.0). Offset PDF ↔ sách in: **+23 trang** (số in + 23 = số PDF).

---

## ⚠️ Phát hiện quan trọng (áp dụng cho CẢ 2 family)

Bulkowski đo performance Flags + Pennants **KHÁC** với mọi pattern khác trong ECP:

> *"The performance statistics do not use the usual ultimate high or low method (waiting for a 20% trend change). Instead, I looked at the beginning and ending of the price trend (usually the nearest minor high or low mirrored across the pattern)."* — ECP p341 (Flags), p526-527 (Pennants)

Hệ quả:
- **"Days to trend high/low" KHÔNG PHẢI "Days to ultimate high/low"** như các pattern khác. Bulkowski dừng ở minor high/low gần nhất, không đợi 20% reversal.
- Số ngày ngắn hơn nhiều (Flags 17d, Pennants 16-22d) so với pattern dùng ultimate method (H&S 41-176d, Cup 63-167d).
- **KHÔNG SO SÁNH** số performance của Flags/Pennants với pattern khác — chỉ so sánh được Flags ↔ Pennants với nhau.

---

## Bảng 1 — FLAGS (ECP chapter 21, PDF p362-374 / sách p339-351)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | |
| **pages_checked** | PDF **362-374** (13 trang) | sách in p339-351 |
| **sample (Number of formations, Table 21.2)** | **149** (bull/UA) + **133** (bear/UA) + **103** (bull/UD) + **138** (bear/UD) = **523 flags** | Bulkowski nói "523 flags without intensive search"; continuation only |
| **Reversal/Continuation** | 100% Continuation (149C / 133C / 103C / 138C) | Chỉ đo continuation, không đo reversal |
| **failure break-even BE% (Table 21.3, mốc 5%)** | **4%** (bull/UA, 6/149) · **3%** (bear/UA, 4/133) · **2%** (bull/UD, 2/103) · **0%** (bear/UD, 0/138) | Best = bear/UD; worst = bull/UA |
| **Failure rate đầy đủ (Table 21.3)** | 5%=4/3/2/0% · 10%=20/21/24/10% · 15%=36/47/49/25% · 20%=55/65/72/42% · 25%=66/76/79/56% · 30%=81/85/88/70% · 35%=85/93/93/77% · 50%=93/98/100/90% · 75%=98/100/100/100% | Thứ tự cột: bull/UA · bear/UA · bull/UD · bear/UD |
| **% meeting price target** | **KHÔNG có bảng riêng**; chỉ thấy 1 số context: bull/UD ≈ **47%** (sample trade, p348) | Bulkowski: "more than half hit target" — không có số chính xác cho 4 tổ hợp |
| **Average rise/decline (Table 21.2)** | +23% (bull/UA) · +17% (bear/UA) · −16% (bull/UD) · −25% (bear/UD) | Đo từ breakout → trend end, KHÔNG phải ultimate |
| **Days to trend high/low (Table 21.2)** | **17 ngày** cho cả 4 tổ hợp | Bulkowski double-check vì trùng; lưu ý: KHÔNG phải ultimate |
| **Measure rule (Table 21.8)** | Tính trend height = (trend start A) − (low tại formation start B). Trừ giá trị này từ high tại formation end (C) → target price. **Ví dụ sách (Fig 21.6):** A=47.50, B=42.75, diff=4.75; C=43 → target=38.25 | **KHÁC digitized** |
| **Throwbacks/pullbacks (Table 21.4)** | 43% / 53% / 46% / 44% | Bull/UA · bear/UA · bull/UD · bear/UD |
| **Avg time to throwback/pullback ends** | 14d / 12d / 15d / 12d | |
| **Avg formation length** | 10d / 9d / 11d / 10d | Median 8/7/9/7 days |
| **Busted pattern performance** | +30% (bull/UA, n<30) · N/A · −7% (bull/UD, n<30) · −19% (bear/UD, n<30) | Sample quá nhỏ, Bulkowski bảo "ignore" |

### Đối chiếu với `flags_digitized.json` — `ghi_chu_lenh_digitized`

| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `failure_rate.bull_flags_bull_market` | 5% | 4% (BE bull/UA) | 🟡 LỆCH nhẹ (5% vs 4%) |
| `failure_rate.bear_flags_bear_market` | 6% | 0% (BE bear/UD) | 🔴 LỆCH LỚN (6% vs 0%) — digitized cao gấp ∞ |
| `failure_rate.overall_average` | 5.5% | Trung bình 4 tổ hợp = (4+3+2+0)/4 = **2.25%** | 🔴 LỆCH (5.5% vs 2.25%) |
| `failure_definition.threshold_pct` | 5% | 5% (breakeven) | 🟢 KHỚP |
| `performance_statistics.bull_flags.average_rise_pct` | "Similar to flagpole gain" (không có số) | **+23%** (bull/UA, đo trend-end) | 🔴 THIẾU số thực |
| `performance_statistics.bear_flags.average_decline_pct` | "Similar to flagpole loss" | **−25%** (bear/UD) | 🔴 THIẾU số thực |
| `performance_statistics.bull_flags.failure_rate_pct` | 5% | 4% (BE bull/UA) | 🟡 LỆCH nhẹ |
| `performance_statistics.high_tight_flags.average_rise_pct` | 47% | **69%** (HTF chương 22, đã trích 12/8) | 🔴 LỆCH (47% vs 69%) |
| `performance_statistics.high_tight_flags.failure_rate_pct` | 5% | **0%** (HTF chương 22) | 🔴 LỆCH (5% vs 0%) |
| `post_breakout_measurement.ultimate_high_method.average_days` | 25 | **17** (days to trend high, KHÔNG phải ultimate) | 🟡 LỆCH + SAI SEMANTIC — digitized dùng "ultimate" nhưng PDF đo trend-end |
| `post_breakout_measurement.ultimate_low_method.average_days` | 22 | **17** (days to trend low) | 🟡 LỆCH + SAI SEMANTIC |
| `post_breakout_measurement.lookahead_bars` | 63 | 17 ngày (PDF) — ~17 bars daily | 🟡 LỆCH (63 vs ~17) |
| `post_breakout_measurement.target_calculation.method` | `flagpole_addition` (target = breakout + flagpole height) | **KHÁC**: target = formation_end − (trend_start − formation_start_low) | 🔴 **SAI METHOD** — digitized measure sai completely |
| `post_breakout_measurement.target_calculation.formula` | `target_price = breakout_price + (flagpole_top - flagpole_bottom)` | `target = formation_end_high − (trend_start − formation_start_low)` cho downtrend; đối xứng cho uptrend | 🔴 **SAI CÔNG THỨC** |
| `breakout_confirmation.throwback_pullback_rate_pct` | 45% | 43% (bull/UA) / 53% (bear/UA) / 46% (bull/UD) / 44% (bear/UD) | 🟢 GẦN KHỚP (45 vs 43-53) |
| `geometry_constraints.width_optimal_bars` | 12 | Median 7-9d, avg 9-11d | 🟡 LỆCH (12 vs 9-11) |
| `duration_constraints.optimal_bars` | 12 | Avg 9-11d | 🟡 LỆCH nhẹ |
| Sample size | KHÔNG ghi | **523** flags | 🔴 THIẾU hoàn toàn |

**Tóm tắt lệch Flags:** 🔴 **LỆCH NGHIÊM TRỌNG** về (1) measure rule method (digitized sai công thức), (2) sample (không ghi), (3) failure rate (5.5% vs 2.25% thực), (4) average rise (không có số thực), (5) HTF variant (47% vs 69%, 5% vs 0%).

---

## Bảng 2 — PENNANTS (ECP chapter 34, PDF p545-564 / sách p522-541)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | |
| **pages_checked** | PDF **545-564** (20 trang) | sách in p522-541 |
| **sample (Number of formations, Table 34.2)** | **173** (bull/UA) + **107** (bear/UA) + **84** (bull/UD) + **98** (bear/UD) = **462 pennants** | Continuation only |
| **Reversal/Continuation** | 100% Continuation (173C / 107C / 84C / 98C) | Chỉ đo continuation |
| **failure break-even BE% (Table 34.3, mốc 5%)** | **2%** (bull/UA, 3/173) · **2%** (bear/UA, 2/107) · **4%** (bull/UD, 3/84) · **0%** (bear/UD, 0/98) | Best = bear/UD (0%); worst = bull/UD (4%) |
| **Failure rate đầy đủ (Table 34.3)** | 5%=2/2/4/0% · 10%=12/9/14/9% · 15%=31/23/37/22% · 20%=47/47/52/44% · 25%=60/65/67/57% · 30%=72/77/85/70% · 35%=78/82/88/84% · 50%=90/94/94/93% · 75%=95/99/100/100% | Thứ tự cột: bull/UA · bear/UA · bull/UD · bear/UD |
| **% meeting price target (Results Snapshot)** | Upward: **60%** (bull) / **63%** (bear) · Downward: **51%** (bull) / **50%** (bear) | Bulkowski: "works between half (bear) and two-thirds (bull) of the time" |
| **Average rise/decline (Table 34.2)** | +25% (bull/UA) · +21% (bear/UA) · −19% (bull/UD) · −25% (bear/UD) | Đo breakout → trend end, KHÔNG phải ultimate |
| **Days to trend high/low (Table 34.2)** | **22** (bull/UA) · **18** (bear/UA) · **16** (bull/UD) · **16** (bear/UD) | Lưu ý: KHÔNG phải ultimate; UA lâu hơn UD |
| **Measure rule (Table 34.8)** | Tính trend height = (pennant top B) − (trend start low A). Add giá trị này vào intraday low tại pennant end (C) → target. **Ví dụ sách (Fig 34.5):** A=7.50, B=10.69, diff=3.19; C=11.44 → target=14.63 | **KHÁC digitized** (cùng method flags) |
| **Throwbacks/pullbacks (Table 34.4)** | 47% / 54% / 31% / 54% | Bull/UA · bear/UA · bull/UD · bear/UD |
| **Avg time to throwback/pullback ends** | 15d / 14d / 12d / 14d | |
| **Avg formation length** | 10d / 9d / 11d / 10d | Median 8/8/9/7 days; giống Flags |
| **Busted pattern performance** | +27% (bull/UA) · N/A (bear/UA, tất cả đều >5%) · −12% (bull/UD) · −24% (bear/UD) | Bulkowski: "look elsewhere" cho busted |

### Đối chiếu với `pennants_digitized.json` — `ghi_chu_lenh_digitized`

| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `failure_rate.bull_pennants_bull_market` | 6% | **2%** (BE bull/UA) | 🔴 LỆCH LỚN (6% vs 2%) |
| `failure_rate.bear_pennants_bear_market` | 7% | **0%** (BE bear/UD) | 🔴 LỆCH LỚN (7% vs 0%) |
| `failure_rate.overall_average` | 6.5% | Trung bình 4 tổ hợp = (2+2+4+0)/4 = **2.0%** | 🔴 LỆCH (6.5% vs 2.0%) |
| `failure_definition.threshold_pct` | 5% | 5% (breakeven) | 🟢 KHỚP |
| `post_breakout_measurement.average_move.bull_pennant_rise_pct` | 19% | **+25%** (bull/UA) | 🟡 LỆCH (19 vs 25) |
| `post_breakout_measurement.average_move.bear_pennant_decline_pct` | 16% | **−25%** (bear/UD) | 🔴 LỆCH (16 vs 25) |
| `performance_statistics.bull_pennants.bull_market_rise_pct` | 19% | **25%** (bull/UA) | 🟡 LỆCH (19 vs 25) |
| `performance_statistics.bear_pennants.bear_market_decline_pct` | 16% | **25%** (bear/UD) | 🔴 LỆCH (16 vs 25) |
| `performance_statistics.bull_pennants.failure_rate_pct` | 6% | 2% (BE bull/UA) | 🔴 LỆCH (6 vs 2) |
| `performance_statistics.bear_pennants.failure_rate_pct` | 7% | 0% (BE bear/UD) | 🔴 LỆCH (7 vs 0) |
| `performance_statistics.bull_pennants.average_days_to_target` | 20 | **22** (days to trend high bull/UA) | 🟡 GẦN KHỚP (20 vs 22) |
| `performance_statistics.bear_pennants.average_days_to_target` | 18 | **16** (days to trend low bear/UD) | 🟡 GẦN KHỚP (18 vs 16) |
| `post_breakout_measurement.ultimate_high_method.average_days` | 20 | **22** (days to trend high, KHÔNG phải ultimate) | 🟡 LỆCH + SAI SEMANTIC |
| `post_breakout_measurement.ultimate_low_method.average_days` | 18 | **16** (days to trend low) | 🟡 LỆCH nhẹ + SAI SEMANTIC |
| `post_breakout_measurement.lookahead_bars` | 63 | 16-22 ngày (PDF) — ~16-22 bars daily | 🟡 LỆCH (63 vs 16-22) |
| `post_breakout_measurement.target_calculation.method` | `flagpole_addition` (target = breakout + flagpole height) | **KHÁC**: target = pennant_end_low + (pennant_top − trend_start_low) | 🔴 **SAI METHOD** |
| `post_breakout_measurement.target_calculation.formula` | `target_price = breakout_price + (flagpole_top - flagpole_bottom)` | `target = formation_end_low + (pennant_top − trend_start_low)` cho UA; đối xứng cho UD | 🔴 **SAI CÔNG THỨC** |
| `breakout_confirmation.throwback_pullback_rate_pct` | 35% | 47% / 54% / 31% / 54% | 🟡 LỆCH (35 vs 31-54) |
| `geometry_constraints.width_optimal_bars` | 10 | Median 7-9d, avg 9-11d | 🟡 GẦN KHỚP |
| `duration_constraints.optimal_bars` | 10 | Avg 9-11d | 🟢 KHỚP |
| Sample size | KHÔNG ghi | **462** pennants | 🔴 THIẾU hoàn toàn |
| `% meeting price target` | KHÔNG có trường | **60/63/51/50%** | 🔴 THIẾU hoàn toàn — đây là metric quan trọng nhất để đánh giá measure rule |

**Tóm tắt lệch Pennants:** 🔴 **LỆCH NGHIÊM TRỌNG** về (1) measure rule method (digitized sai công thức), (2) sample (không ghi), (3) failure rate (6.5% vs 2.0% thực — digitized cao gấp 3 lần), (4) average rise/decline (16-19% vs 21-25%), (5) THIẾU metric % meeting price target.

---

## Bằng chứng verbatim (số liệu thô, không copy câu dài — bản quyền)

### Flags — Table 21.2 (PDF p342 / sách p341)
```
Number of formations   149   133   103   138      (bull/UA, bear/UA, bull/UD, bear/UD)
Reversal (R), cont.    149 C 133 C 103 C 138 C
Average rise/decline   23%   17%   -16%  -25%
Days to trend high/low 17    17    17    17
```

### Flags — Table 21.3 Failure Rates (PDF p343 / sách p342)
```
Max rise/decline   bull/UA   bear/UA   bull/UD   bear/UD
5% (breakeven)     6 or 4%   4 or 3%   2 or 2%   0 or 0%
10                 30 or 20% 28 or 21% 25 or 24% 14 or 10%
15                 53 or 36% 63 or 47% 50 or 49% 35 or 25%
20                 82 or 55% 86 or 65% 74 or 72% 58 or 42%
25                 98 or 66% 101 or 76% 81 or 79% 77 or 56%
```

### Pennants — Table 34.2 (PDF p528 / sách p527)
```
Number of formations   173   107   84    98
Reversal (R), cont.    173 C 107 C 84 C  98 C
Average rise/decline   25%   21%   -19%  -25%
Days to trend high/low 22    18    16    16
```

### Pennants — Table 34.3 Failure Rates (PDF p529 / sách p528)
```
Max rise/decline   bull/UA   bear/UA   bull/UD   bear/UD
5% (breakeven)     3 or 2%   2 or 2%   3 or 4%   0 or 0%
10                 21 or 12% 10 or 9%  12 or 14% 9 or 9%
15                 54 or 31% 25 or 23% 31 or 37% 22 or 22%
20                 81 or 47% 50 or 47% 44 or 52% 43 or 44%
25                 103 or 60% 70 or 65% 56 or 67% 56 or 57%
```

### Pennants — Results Snapshot (PDF p522 / sách p521-522)
```
Upward breakouts:   BE 2%/2%, rise 25%/21%, throwback 47%/54%, %target 60%/63%
Downward breakouts: BE 4%/0%, decline 19%/25%, pullback 31%/54%, %target 51%/50%
```

---

## So sánh Flags ↔ Pennants (cùng method đo → so sánh được)

| Metric | Flags | Pennants | Nhận xét |
|---|---|---|---|
| **Sample tổng** | 523 | 462 | Gần bằng nhau |
| **BE failure rate (trung bình 4 tổ hợp)** | (4+3+2+0)/4 = **2.25%** | (2+2+4+0)/4 = **2.0%** | Pennants hơi thấp hơn |
| **Average rise bull/UA** | +23% | +25% | Pennants cao hơn chút |
| **Average decline bear/UD** | −25% | −25% | Bằng nhau |
| **Days to trend end (trung bình)** | 17d (cả 4) | 18d (avg 22+18+16+16) | Flags hơi nhanh hơn |
| **% meeting target** | không có bảng; ~47% bull/UD | 60/63/51/50% | Pennants công bố số |
| **Throwback/Pullback rate** | 43-53% | 31-54% | Pennants biến thiên lớn hơn |
| **Avg formation length** | 9-11 days | 9-11 days | Bằng nhau |
| **Measure rule method** | trend_height = trend_start − formation_start_low | trend_height = pennant_top − trend_start_low | Cùng pattern: dựa trên trend-start, KHÔNG phải flagpole addition như digitized ghi |

**Kết luận so sánh:** Flags và Pennants có **performance gần như tương đương** (cùng sample, cùng BE%, cùng average move, cùng method đo). Điểm khác biệt chính: Pennants công bố % meeting target (50-63%) còn Flags thì không có bảng riêng.

---

## Reproducer

```bash
PDF="/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet/Nghiên cứu mô hình nến/references/encyclopedia-of-chart-patterns-2nbsped-9786468600-3175723993-9780471668268-0471668265_compress.pdf"

# Flags (chương 21, PDF p362-374)
pdftotext -layout -f 362 -l 374 "$PDF" - | grep -A 25 "Table 21.2"
pdftotext -layout -f 362 -l 374 "$PDF" - | grep -A 20 "Table 21.3"

# Pennants (chương 34, PDF p545-564)
pdftotext -layout -f 545 -l 564 "$PDF" - | grep -A 25 "Table 34.2"
pdftotext -layout -f 545 -l 564 "$PDF" - | grep -A 20 "Table 34.3"
pdftotext -layout -f 545 -l 564 "$PDF" - | grep -A 15 "RESULTS SNAPSHOT"
```

---

## Khuyến nghị (không thực hiện trong task này)

1. **Sửa measure rule cả 2 spec**: Digitized đang dùng `flagpole_addition` (target = breakout + flagpole height) — **SAI HOÀN TOÀN** so với PDF. Cần đổi sang: `target = formation_end_high/low ± (trend_start − formation_start_low/high)`. Đây là lỗi critical vì target sai → toàn bộ risk/reward calculation sai.
2. **Bổ sung sample**: Flags 523, Pennants 462 — digitized đang NOT-RECORDED.
3. **Sửa failure rate**: Flags overall 2.25% (không phải 5.5%), Pennants 2.0% (không phải 6.5%). Digitized cao gấp 2-3 lần thực.
4. **Sửa average rise/decline**: Flags +23%/−25%, Pennants +25%/−25%. Digitized dùng "Similar to flagpole" hoặc 16-19% — không chính xác.
5. **Đổi tên trường `ultimate_high/low_method`**: Đây là **SAI SEMANTIC** — PDF đo trend-end (minor high/low gần nhất), không phải ultimate high/low (đợi 20% reversal). Đề xuất đổi thành `trend_end_method` và ghi rõ "different from other patterns — không so sánh được với H&S/Cup/Pipe".
6. **Sửa lookahead_bars**: 63 → ~17-22 (theo days to trend end PDF).
7. **Bổ sung Pennants % meeting target** (60/63/51/50%): digitized đang thiếu metric quan trọng nhất.
8. **HTF (variant của Flags)**: đã trích riêng trong PDF_REVIEW_20260812.md (chương 22, p373-397) — sample 307, BE 0%/0%, rise 69%/42%, %target 90%/91%. Cần tách thành spec riêng hoặc cập nhật số liệu HTF trong flags_digitized (hiện ghi rise 47%, failure 5% — lệch nghiêm trọng).

---

**Hết file.**
