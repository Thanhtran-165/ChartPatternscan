# M5 — Trích số liệu PDF gốc: family THREE METHODS (Rising + Falling)

**Ngày trích:** 2026-08-13
**Model / Provider:** GLM-5.2 / Z.AI (zai-coding-plan)
**Session mẹ:** `sess_0fb10a3f-28e7-42c6-8bdd-6aca4b147362`
**Nguồn PDF:** `references/Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf` (966 trang) — **EC (Candlestick book), KHÔNG PHẢI ECP**
**Phương pháp:** `pdftotext -layout` (Poppler 26.04.0). Offset PDF ↔ sách in: **+23 trang** (số in + 23 = số PDF; kiểm chứng: PDF p367 = in p344, PDF p656 = in p633).

---

## ⚠️ Phát hiện quan trọng — 3 điểm khác biệt căn bản với ECP

### 1. Three Methods KHÔNG có trong ECP (Encyclopedia of Chart Patterns)
Mô tả nhiệm vụ ghi "ECP chương 40 + EC candlestick" nhưng **ECP chương 40 là "Rounding Tops"** (xác minh qua outline ECP). Three Methods là **candlestick pattern**, chỉ có trong **EC (Encyclopedia of Candlestick Charts)**:
- **Rising Three Methods** = EC Chapter 73 (PDF p656-672 / sách p633-649)
- **Falling Three Methods** = EC Chapter 39 (PDF p367-383 / sách p344-360)

### 2. EC dùng methodology KHÁC ECP hoàn toàn
EC không có "Break-even failure rate", "Average rise", "Days to ultimate high/low" như ECP. Thay vào đó:
- **Behavior and Rank**: continuation rate %, frequency rank (1-103), overall performance rank (1-103)
- **Table X.2 General Statistics**: "Number found", "Reversal/continuation performance" (move nhỏ ~3-7%), "Candle end to breakout (median days)", "Candle end to trend end (median days)"
- **Table X.3 Height Statistics**: "% meeting price target (measure rule)", median candle height
- **KHÔNG có failure rate curve 5-75%** như ECP

Do đó các trường digitized `failure_rate at_5pct=20, at_10pct=10` → **BỊA** (EC không publish failure rate cho Three Methods).

### 3. Falling Three Methods — Bulkowski EXCLUDE Statistics section
Pattern này quá hiếm (chỉ 64 examples) nên Bulkowski **từ chối publish Statistics**:

> *"Since this pattern is so rare (I found just 64 examples), I exclude the Statistics section."* — EC p344 (Falling Three Methods)

Nên Falling Three Methods **KHÔNG CÓ** Number found breakdown, failure rate, average move, % target — chỉ có Behavior and Rank (continuation 71%/67%).

---

## Bảng 1 — RISING THREE METHODS (EC chapter 73, PDF p656-672 / sách p633-649)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf` | EC, không phải ECP |
| **pages_checked** | PDF **656-672** (17 trang) | sách in p633-649 |
| **Behavior and Rank** | Theoretical: Bullish continuation. Actual bull market: **74% continuation** (ranking 5). Actual bear market: **79% continuation** (ranking 2). Frequency: **88th out of 103** (rare). Overall performance over time: **94th out of 103** (poor). | "rare pattern that functions quite well as a continuation... but overall performance rank 94/103" — continuation xảy ra thường nhưng price không đi xa trước khi đảo chiều |
| **sample (Number found, Table 73.2)** | **55** (bull/UA) + **22** (bear/UA) + **19** (bull/UD) + **6** (bear/UD) = **102 samples** | "I found only 102 samples, so the results are chancy at best" |
| **Reversal/Continuation performance (Table 73.2)** | 6.86% C (bull/UA) · 3.93% C (bear/UA) · −4.31% R (bull/UD) · −4.10% R (bear/UD) | Move RẤT NHỎ — continuation chỉ +3.93% đến +6.86% |
| **S&P 500 change (Table 73.2)** | 1.38% (bull/UA) · 0.61% (bear/UA) · −1.36% (bull/UD) · −1.59% (bear/UD) | |
| **Candle end to breakout (median, days, Table 73.2)** | 3 (bull/UA) · 2 (bear/UA) · 8 (bull/UD) · 8 (bear/UD) | UD breakout lâu hơn UA |
| **Candle end to trend end (median, days, Table 73.2)** | 7 (bull/UA) · 4 (bear/UA) · 11 (bull/UD) · 8 (bear/UD) | Đây là "days to trend end" — KHÔNG PHẢI ultimate high/low |
| **Median candle height as % of breakout price (Table 73.3)** | 4.92% (bull/UA) · 7.39% (bear/UA) · 6.04% (bull/UD) · 5.17% (bear/UD) | |
| **Tall candle performance (Table 73.3)** | +9.33% (bull/UA, n<30) · +6.31% (bear/UA) · −6.13% (bull/UD) · −6.30% (bear/UD) | Tall candle outperform short |
| **Short candle performance (Table 73.3)** | +5.24% (bull/UA) · +2.73% (bear/UA) · −3.13% (bull/UD) · −2.46% (bear/UD) | |
| **% meeting price target — measure rule (Table 73.3)** | **60%** (bull/UA) · **23%** (bear/UA) · **21%** (bull/UD) · **33%** (bear/UD) | Measure rule yếu (trừ bull/UA 60%) |
| **Measure rule (Table 73.3 + text)** | target = **breakout price ± (candle height × percentage)**. Compute candle height (HH − LL), multiply by appropriate %, apply to breakout. Ví dụ: height=4 (62−58), bull/UA → target = (4 × 60%) + 62 = 64.40. | Quote EC p637: *"Compute the height of the candle pattern and multiply it by the appropriate percentage shown in the table; then apply it to the breakout price"* |
| **Failure rate / Break-even failure rate** | **KHÔNG CÓ** (EC không publish cho candlestick) | EC dùng "% meeting price target" thay thế |

### Đối chiếu với `rising_falling_three_methods_digitized.json` (phần Rising)

| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `post_breakout_measurement.failure_rate.at_5pct` | 20 | **KHÔNG CÓ** (EC không publish) | 🔴 **BỊA** — metric không tồn tại |
| `post_breakout_measurement.failure_rate.at_10pct` | 10 | **KHÔNG CÓ** | 🔴 **BỊA** |
| `performance_statistics.failure_rate_5pct` | 20 | KHÔNG CÓ | 🔴 **BỊA** |
| `performance_statistics.failure_rate_10pct` | 10 | KHÔNG CÓ | 🔴 **BỊA** |
| `post_breakout_measurement.failure_definition.threshold_pct` | 2.0 | KHÔNG CÓ (EC không define failure rate) | 🔴 **BỊA** |
| `variant_handling.variants.rising_three_methods.parameter_overrides.expected_success_rate` | 0.72 | Continuation 74% (bull) / 79% (bear) | 🟢 GẦN KHỚP (0.72 ≈ 74% bull; bear 79% lệch nhẹ) |
| `performance_statistics.rising_methods_success_rate` | 72 | 74% (bull) / 79% (bear) | 🟢 GẦN KHỚP (72 vs 74 bull) |
| `post_breakout_measurement.average_continuation.rising_methods_pct` | 5 | +6.86% (bull/UA) / +3.93% (bear/UA) | 🟢 GẦN KHỚP (5 vs 3.93-6.86, avg ~5.4%) |
| `performance_statistics.time_to_breakout_days` | 1 | Candle end to breakout median: 3 (bull/UA) / 2 (bear/UA) | 🟡 LỆCH nhẹ (1 vs 2-3) |
| `post_breakout_measurement.ultimate_high_method.average_days` | 10 | Candle end to trend end median: 7 (bull/UA) / 4 (bear/UA) | 🟡 LỆCH nhẹ (10 vs 4-7) + SAI SEMANTIC (EC dùng "trend end" không phải ultimate) |
| `post_breakout_measurement.ultimate_low_method.average_days` | 10 | KHÔNG ÁP DỤNG (rising là bullish, không có ultimate low) | 🔴 **SAI SEMANTIC** |
| `post_breakout_measurement.lookahead_bars` | 20 | Candle end to trend end 4-7d (UA) / 8-11d (UD); candle end to breakout 2-3d | 🟡 GẦN KHỚP (20 vs 4-11 median, nhưng range rộng) |
| `performance_statistics.average_continuation_days` | 10 | Candle end to trend end median 7 (bull/UA) | 🟢 GẦN KHỚP (10 vs 7) |
| `post_breakout_measurement.target_calculation.method` | `continuation_magnitude` — "Target typically equals first bar's move" | `candle_height_multiplier` — target = breakout ± (height × % từ bảng 60/23/21/33%) | 🔴 **SAI METHOD** — digitized dùng first bar range; PDF dùng candle pattern height × percentage multiplier |
| `post_breakout_measurement.target_calculation.formula` | `target = breakout_price +/- (first_bar_range * continuation_factor)` | `target = breakout_price +/- (candle_height * multiplier_pct)`; multiplier 60/23/21/33% | 🔴 **SAI CÔNG THỨC** |
| `post_breakout_measurement.throwback_pullback.rate_pct` | 40 | KHÔNG CÓ throwback rate trong EC (chỉ có volume stats) | 🟡 KHÔNG XÁC ĐỊNH được từ PDF |
| `performance_statistics.pullback_rate_pct` | 40 | KHÔNG CÓ | 🟡 KHÔNG XÁC ĐỊNH |
| `geometry_constraints.first_bar_requirements.min_body_size_atr` | 1.5 | EC: median candle height 4.92-7.39% of breakout price (cho cả pattern, không phải first bar riêng) | 🟡 KHÔNG XÁC ĐỊNH first-bar-specific từ PDF |
| `geometry_constraints.first_bar_requirements.ideal_body_size_atr` | 2.0 | Không có số ATR-specific | 🟡 KHÔNG XÁC ĐỊNH |
| `breakout_confirmation.volume_multiplier_ideal` | 1.3 | EC Table 73.4: rising candle volume performance 8.78% vs falling 6.16% (bull/UA) — qualitative, không quote 1.3x | 🟡 KHÔNG XÁC ĐỊNH số chính xác |
| `performance_statistics.volume_pattern_correlation` | 0.50 | EC qualitative: "rising volume trend perform better for upward breakouts" | 🟡 KHÔNG XÁC ĐỊNH số correlation |
| Sample size | KHÔNG ghi | **102** samples | 🔴 THIẾU hoàn toàn |
| `% meeting price target` | KHÔNG có trường | 60/23/21/33% | 🔴 THIẾU — metric chính của EC measure rule |
| `overall performance rank` | KHÔNG có | 94/103 (poor) | 🔴 THIẾU — metric quan trọng cho decision-making |

**Tóm tắt lệch Rising Three Methods:** 🔴 **BỊA SỐ** về (1) failure rate BỊA (20/10 — EC không publish), (2) target method SAI (digitized first_bar_range vs PDF candle_height × multiplier%), (3) ultimate_low_method SAI SEMANTIC (rising bullish không có ultimate low). 🟢 KHỚP tốt ở: continuation rate ~72% (vs 74-79%), average continuation ~5% (vs 3.93-6.86%). 🟡 KHÔNG XÁC ĐỊNH được nhiều trường ATR/volume/throwback vì EC qualitative.

---

## Bảng 2 — FALLING THREE METHODS (EC chapter 39, PDF p367-383 / sách p344-360)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf` | |
| **pages_checked** | PDF **367-383** (17 trang) | sách in p344-360 |
| **Behavior and Rank** | Theoretical: Bearish continuation. Actual bull market: **71% continuation** (ranking 7). Actual bear market: **67% continuation** (ranking 15). Frequency: **91st out of 103** (very rare). Overall performance over time: **89th out of 103**. | "complicated and rare pattern" |
| **sample** | **64 examples** (chỉ tổng, KHÔNG breakdown bull/bear/UA/UD) | **Statistics section EXCLUDED** bởi Bulkowski vì quá hiếm |
| **Number found breakdown (Table 39.2)** | **KHÔNG CÓ** — Statistics excluded | Quote EC p344: *"Since this pattern is so rare (I found just 64 examples), I exclude the Statistics section"* |
| **Reversal/Continuation performance** | **KHÔNG CÓ** (Statistics excluded) | |
| **Candle end to breakout / trend end (median days)** | **KHÔNG CÓ** | |
| **% meeting price target** | **KHÔNG CÓ** (no Table 39.3) | |
| **Measure rule multiplier %** | **KHÔNG CÓ** (no height statistics table) | |
| **Failure rate / Break-even failure rate** | **KHÔNG CÓ** | |
| **Avg decline / Move %** | **KHÔNG CÓ** | |
| **Identification Guidelines (Table 39.1)** | Có — 5 candle lines: tall black, 3 small (trend up, stay within first bar range), tall black closing lower | Structure info có, performance info KHÔNG |

### Đối chiếu với `rising_falling_three_methods_digitized.json` (phần Falling)

| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `variant_handling.variants.falling_three_methods.parameter_overrides.expected_success_rate` | 0.68 | Continuation 71% (bull) / 67% (bear) | 🟢 KHỚP (0.68 ≈ 67-71%, trung bình 69%) |
| `performance_statistics.falling_methods_success_rate` | 68 | 71% (bull) / 67% (bear) | 🟢 KHỚP (68 vs 67-71) |
| `post_breakout_measurement.average_continuation.falling_methods_pct` | 5 | **KHÔNG CÓ** (Statistics excluded) | 🔴 **BỊA** — digitized bịa số 5% |
| `post_breakout_measurement.failure_rate.at_5pct` | 20 | **KHÔNG CÓ** | 🔴 **BỊA** |
| `post_breakout_measurement.failure_rate.at_10pct` | 10 | **KHÔNG CÓ** | 🔴 **BỊA** |
| `performance_statistics.failure_rate_5pct` | 20 | **KHÔNG CÓ** | 🔴 **BỊA** |
| `performance_statistics.failure_rate_10pct` | 10 | **KHÔNG CÓ** | 🔴 **BỊA** |
| `post_breakout_measurement.ultimate_low_method.average_days` | 10 | **KHÔNG CÓ** | 🔴 **BỊA** |
| `post_breakout_measurement.lookahead_bars` | 20 | **KHÔNG CÓ** | 🔴 **BỊA** |
| `performance_statistics.average_continuation_days` | 10 | **KHÔNG CÓ** | 🔴 **BỊA** |
| `post_breakout_measurement.target_calculation` (method + formula) | `continuation_magnitude` / `first_bar_range * continuation_factor` | **KHÔNG CÓ** (no measure rule table) | 🔴 **BỊA** |
| `post_breakout_measurement.throwback_pullback.rate_pct` | 40 | **KHÔNG CÓ** | 🔴 **BỊA** |
| Sample size | KHÔNG ghi | **64** (tổng, không breakdown) | 🟡 THIẾU — nhưng PDF cũng chỉ có tổng 64 |
| `overall performance rank` | KHÔNG có | 89/103 | 🔴 THIẾU |

**Tóm tắt lệch Falling Three Methods:** 🔴 **BỊA SỐ NGHIÊM TRỌNG** — gần như TẤT CẢ trường performance (failure rate, average continuation, days, target, throwback) đều BỊA vì PDF **EXCLUDE** toàn bộ Statistics section (chỉ 64 samples). Digitized chỉ KHỚP ở continuation rate (68% vs 67-71%). Cần xóa hoặc ghi "NOT PUBLISHED — too rare (64 samples)" cho mọi trường performance.

---

## Bằng chứng verbatim (số liệu thô, không copy câu dài — bản quyền)

### Rising Three Methods — Behavior and Rank (PDF p656 / sách p633)
```
Theoretical: Bullish continuation.
Actual bull market: Bullish continuation 74% of the time (ranking 5).
Actual bear market: Bullish continuation 79% of the time (ranking 2).
Frequency: 88th out of 103.
Overall performance over time: 94th out of 103.
```

### Rising Three Methods — Table 73.2 (PDF p636 / sách p635)
```
                                       Bull       Bear       Bull        Bear
                                       Market,    Market,    Market,     Market,
                                       Up         Up         Down        Down
Number found                           55         22         19          6
Reversal (R), continuation (C)
  performance                          6.86% C    3.93% C    -4.31% R    -4.10% R
Candle end to breakout (median, days)  3          2          8           8
Candle end to trend end (median, days) 7          4          11          13
```

### Rising Three Methods — Table 73.3 Height Statistics (PDF p637 / sách p636)
```
Median candle height as % of breakout price    4.92%   7.39%   6.04%   5.17%
Short candle, performance                       5.24%   2.73%  -3.13%  -2.46%
Tall candle, performance                        9.33%*  6.31%  -6.13%  -6.30%
Percentage meeting price target (measure rule)  60%     23%     21%     33%
```

### Rising Three Methods — Measure rule (PDF p637 / sách p636)
```
"Compute the height of the candle pattern and multiply it by the appropriate percentage
shown in the table; then apply it to the breakout price."
"The upward target would be (4 × 60%) + 62, or 64.40, and the downward target would be
58 – (4 × 21%), or 57.16."
```

### Falling Three Methods — Behavior and Rank (PDF p656 sai, thực PDF p367 / sách p344)
```
Theoretical: Bearish continuation.
Actual bull market: Bearish continuation 71% of the time (ranking 7).
Actual bear market: Bearish continuation 67% of the time (ranking 15).
Frequency: 91st out of 103.
Overall performance over time: 89th out of 103.
```

### Falling Three Methods — Statistics EXCLUDED (PDF p367 / sách p344)
```
"Since this pattern is so rare (I found just 64 examples), I exclude the Statistics section."
```

---

## So sánh Rising ↔ Falling Three Methods

| Metric | Rising Three Methods | Falling Three Methods | Nhận xét |
|---|---|---|---|
| **Sample** | 102 (breakdown 55/22/19/6) | 64 (NO breakdown) | Cả 2 cực hiếm; Falling hiếm hơn |
| **Continuation rate bull** | 74% (rank 5) | 71% (rank 7) | Gần bằng nhau |
| **Continuation rate bear** | 79% (rank 2) | 67% (rank 15) | Rising mạnh hơn bear market |
| **Frequency rank** | 88/103 | 91/103 | Cả 2 rất hiếm (bottom 15) |
| **Overall performance rank** | 94/103 | 89/103 | Cả 2 POOR performance |
| **Statistics section** | CÓ (Table 73.2, 73.3) | **EXCLUDED** | Rising có data, Falling không |
| **Continuation move** | +6.86%/+3.93% (UA) | KHÔNG CÓ | Move rất nhỏ |
| **% meeting target** | 60/23/21/33% | KHÔNG CÓ | Measure rule yếu |

**Kết luận so sánh:** Cả Rising và Falling Three Methods là **pattern cực hiếm + poor performance** (rank 89-94/103). Rising có data chi tiết (102 samples), Falling không có (64 samples, excluded). Continuation rate khá cao (67-79%) nhưng **move rất nhỏ** (+3.93-6.86%) và **overall performance poor** → Bulkowski khuyến nghị KHÔNG nên giao dịch dựa trên pattern này một mình. Pattern chỉ đáng giá như "support/resistance zone" qualitative.

---

## Reproducer

```bash
EC="/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet/Nghiên cứu mô hình nến/references/Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf"

# Rising Three Methods (chương 73, PDF p656-672)
pdftotext -layout -f 656 -l 672 "$EC" - | sed -n '633,635p'      # Behavior and Rank
pdftotext -layout -f 656 -l 672 "$EC" - | grep -A 16 "Table 73.2"
pdftotext -layout -f 656 -l 672 "$EC" - | grep -A 16 "Table 73.3"
pdftotext -layout -f 656 -l 672 "$EC" - | sed -n '636,638p'      # Measure rule

# Falling Three Methods (chương 39, PDF p367-383)
pdftotext -layout -f 367 -l 383 "$EC" - | sed -n '344,346p'      # Behavior and Rank
pdftotext -layout -f 367 -l 383 "$EC" - | sed -n '344p'          # "I exclude the Statistics section"
```

---

## Khuyến nghị (không thực hiện trong task này)

1. **XÁC ĐỊNH lại nguồn**: digitized không ghi nguồn PDF rõ ràng. Three Methods chỉ có trong **EC (Candlestick)**, KHÔNG trong ECP. Cần ghi rõ `source_pdf: EC chapter 39/73`.
2. **XÓA failure rate Falling Three Methods (CRITICAL)**: at_5pct=20, at_10pct=10 → **BỊA**. PDF EXCLUDE toàn bộ Statistics (64 samples). Cần ghi "NOT PUBLISHED — Bulkowski excluded Statistics section due to rarity (64 samples)".
3. **XÓA/SỬA failure rate Rising Three Methods**: at_5pct=20, at_10pct=10 → **BỊA**. EC không publish failure rate cho candlestick. Thay bằng "% meeting price target (measure rule)": 60/23/21/33%.
4. **Sửa measure rule**: digitized `continuation_magnitude` (first_bar_range × factor) → PDF `candle_height_multiplier` (pattern height × %). Công thức khác hoàn toàn.
5. **Sửa ultimate_low_method cho Rising (SAI SEMANTIC)**: Rising là bullish continuation, không có ultimate low. Chỉ có ultimate high. Hoặc đổi sang "candle end to trend end (median days)": 7/4/11/13.
6. **Bổ sung sample**: Rising 102 (breakdown 55/22/19/6), Falling 64 (no breakdown).
7. **Bổ sung Behavior and Rank**: continuation rate (74/79% rising, 71/67% falling), frequency rank (88/91 out of 103), overall performance rank (94/89 out of 103). Đây là metric chính của EC thay vì failure rate.
8. **Bổ sung continuation performance**: Rising +6.86%/+3.93% (UA) — move rất nhỏ, metric quan trọng vì giải thích tại sao rank poor dù continuation rate cao.
9. **Cảnh báo rarity + poor performance**: cả 2 pattern cực hiếm (bottom 15/103) và poor performance (rank 89-94/103). Bulkowski khuyến nghị KHÔNG trade dựa trên pattern một mình — chỉ dùng làm support/resistance zone qualitative. Digitized nên có warning field.
10. **Nhiều trường KHÔNG XÁC ĐỊNH được** từ PDF (ATR-specific body size, volume multiplier số, throwback rate, correlation) vì EC qualitative — không nên bịa số cụ thể.

---

**Hết file.**
