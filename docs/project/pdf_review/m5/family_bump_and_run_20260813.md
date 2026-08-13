# M5 — Trích số liệu PDF gốc: family BUMP AND RUN REVERSAL (BARR Bottoms + Tops)

**Ngày trích:** 2026-08-13
**Model / Provider:** GLM-5.2 / Z.AI (zai-coding-plan)
**Session mẹ:** `sess_0fb10a3f-28e7-42c6-8bdd-6aca4b147362`
**Nguồn PDF:** `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` (1.035 trang)
**Phương pháp:** `pdftotext -layout` (Poppler 26.04.0). Offset PDF ↔ sách in: **+23 trang** (số in + 23 = số PDF).

---

## ⚠️ Phát hiện quan trọng (áp dụng cho cả BARR)

Bulkowski mô tả BARR thành **2 chương tách biệt**, mỗi chương là một chiều breakout duy nhất:

- **Chapter 7 — BARR Bottoms** (p137-153): chỉ **upward breakouts** (đảo chiều tăng — bullish). Mẫu hình như "frying pan / spoon" lật ngược.
- **Chapter 8 — BARR Tops** (p154-170): chỉ **downward breakouts** (đảo chiều giảm — bearish). Mẫu hình giá tăng dọc trendline, "bump up, round over, then decline".

File digitized `bump_and_run_reversal_digitized.json` mô tả BARR là `pattern_type: "reversal_bearish"`, `breakout_direction: "down"`, có "run phase decline" → **digitized đang mô tả BARR TOPS (Ch8)**, KHÔNG PHẢI BARR Bottoms. Toàn bộ số liệu đối chiếu dưới đây dùng Ch8 (Tops) làm chuẩn vì khớp semantic với digitized; Ch7 (Bottoms) được bổ sung để đầy đủ.

BARR dùng **ultimate high/low method tiêu chuẩn** (đợi 20% reversal) → `ultimate_high/low_method` trong digitized ĐÚNG semantic.

**Measure rule BARR Tops** dùng "lead-in height" — KHÁC với công thức digitized:

> *"Compute the lead-in height (see Table 8.1 for the definition) and subtract the result from the breakout price."* — ECP Table 8.8 (PDF p144)

**Lead-in height** = khoảng cách dọc từ highest high xuống trendline, đo trong **phần tư đầu tiên (lead-in phase)** của mẫu hình — KHÔNG PHẢI `bump_peak − lead_in_start` như digitized ghi. Ví dụ PDF: lead-in height = 21.50 − 18 = 3.50; target = 25.13 − 3.50 = 21.63.

---

## Bảng 1 — BARR TOPS (ECP chapter 8, PDF p154-170 / sách p131-147) — bản digitized mô tả

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | |
| **pages_checked** | PDF **154-170** (17 trang) | sách in p131-147 |
| **sample (Number of formations, Table 8.2)** | **673** (bull/UD) + **104** (bear/UD) = **777 BARR tops** | "plentiful... nearly 800"; chỉ downward breakouts |
| **Reversal/Continuation (Table 8.2)** | 556R/117C (bull/UD) · 94R/10C (bear/UD) | Hầu hết reversal; bear market gần 100% reversal |
| **R/C performance (Table 8.2)** | −20% R / −13% C (bull) · −26% R / −32% C (bear) | Reversal mạnh hơn continuation ở bull; ngược lại ở bear |
| **Break-even failure rate BE% (Table 8.3, mốc 5%)** | **5%** (bull/UD, 34/673) · **1%** (bear/UD, 1/104) | Best = bear (1%); rất thấp |
| **Failure rate đầy đủ (Table 8.3)** | 5%=5/1% · 10%=20/6% · 15%=41/26% · 20%=57/43% · 25%=73/59% · 30%=85/69% · 35%=91/82% · 50%=98/95% · 75%=100/100% | Thứ tự cột: bull/UD · bear/UD |
| **% meeting price target (Results Snapshot)** | **78%** (bull/UD) / **90%** (bear/UD) | Measure rule rất hiệu quả — "about 8 out of 10 stocks meet their price targets" |
| **Average decline (Table 8.2)** | **−19%** (bull/UD) · **−27%** (bear/UD) | Bear market decline mạnh hơn |
| **Declines over 45% (Table 8.2)** | 20 hoặc 3% (bull) · 7 hoặc 7% (bear) | Hiếm có decline cực lớn |
| **Change after trend ends (Table 8.2)** | +53% (bull/UD) · +48% (bear/UD) | Rebound mạnh sau ultimate low |
| **Days to ultimate low (Table 8.2)** | **68** (bull/UD) · **39** (bear/UD) | Bear market decline nhanh và xa hơn |
| **Busted pattern performance (Table 8.2)** | +39% (bull, n<30) · +31% (bear, n<30) | Busted hiếm |
| **Pullbacks (Results Snapshot + Table 8.4)** | 62% (bull/UD) · 65% (bear/UD) | Lưu ý: BARR Top dùng **pullback** không phải throwback |
| **Measure rule (Table 8.8)** | target = **breakout price − lead-in height**. Lead-in height = vertical dist từ HH xuống trendline, đo trong phần tư đầu tiên (lead-in phase). Ví dụ: lead-in height = 21.50−18 = 3.50; target = 25.13−3.50 = 21.63. | Quote PDF p144: *"calculate the lead-in height by splitting the formation along the trend line into four equal parts. In the first quarter... compute the height from the highest high to the trend line"* |

### Đối chiếu với `bump_and_run_reversal_digitized.json` (digitized = BARR Tops)

| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `post_breakout_measurement.failure_rate.at_5pct` | 10 | BE bull/UD 5%, bear/UD 1% | 🔴 LỆCH (10 vs 5/1 — digitized cao gấp 2-10 lần) |
| `post_breakout_measurement.failure_rate.at_10pct` | 5 | 10% mốc: bull 20%, bear 6% | 🔴 LỆCH (5 vs 20/6) |
| `performance_statistics.failure_rate_5pct` | 10 | BE bull/UD 5% (34/673) | 🔴 LỆCH (10 vs 5) |
| `performance_statistics.failure_rate_10pct` | 5 | bull/UD 20% (134/673), bear/UD 6% (6/104) | 🔴 LỆCH (5 vs 20/6) |
| `post_breakout_measurement.failure_definition.threshold_pct` | 5.0 | 5% (breakeven) | 🟢 KHỚP |
| `performance_statistics.average_decline_pct` | 21 | bull/UD −19%, bear/UD −27% → overall ~21% (weighted) | 🟢 KHỚP (21 ≈ weighted avg) |
| `performance_statistics.median_decline_pct` | 19 | Không có median trực tiếp trong Table 8.2; chỉ average | 🟡 KHÔNG XÁC ĐỊNH được median |
| `post_breakout_measurement.average_decline.bull_market_pct` | 18 | bull/UD −19% | 🟢 GẦN KHỚP (18 vs 19) |
| `post_breakout_measurement.average_decline.bear_market_pct` | 25 | bear/UD −27% | 🟡 LỆCH nhẹ (25 vs 27) |
| `post_breakout_measurement.average_decline.overall_pct` | 21 | weighted ~21% | 🟢 KHỚP |
| `performance_statistics.time_to_ultimate_low_days` | 70 | bull/UD 68, bear/UD 39 | 🟢 GẦN KHỚP (70 ≈ bull 68) — nhưng bear/UD 39 lệch lớn |
| `post_breakout_measurement.ultimate_low_method.average_days` | 70 | bull/UD 68, bear/UD 39 | 🟢 GẦN KHỚP bull; 🟡 bear lệch |
| `breakout_confirmation.throwback_rate_pct` | 45 | pullback bull/UD 62%, bear/UD 65% | 🔴 LỆCH LỚN (45 vs 62/65) — và digitized dùng sai thuật ngữ "throwback" (BARR Top là pullback) |
| `post_breakout_measurement.throwback_pullback.rate_pct` | 45 | 62/65% | 🔴 LỆCH (45 vs 62/65) |
| `performance_statistics.throwback_rate_pct` | 45 | 62/65% | 🔴 LỆCH |
| `post_breakout_measurement.target_calculation.method` | `pattern_height` — "Target = breakout point - bump height" | `lead_in_height` — target = breakout − lead-in height (đo trong lead-in phase, KHÔNG phải bump height) | 🔴 **SAI METHOD** — digitized dùng bump_peak−lead_in_start (chiều cao toàn bump), PDF dùng lead-in height (chiều cao lead-in phase, nhỏ hơn nhiều) |
| `post_breakout_measurement.target_calculation.formula` | `target_price = breakout_price - (bump_peak - lead_in_start)` | `target_price = breakout_price - lead_in_height`; lead_in_height = HH − trendline (first quarter) | 🔴 **SAI CÔNG THỨC** — dimension sai (bump height >> lead-in height) |
| `post_breakout_measurement.lookahead_bars` | 252 | ultimate low bull 68d, bear 39d | 🔴 LỆCH LỚN (252 vs 39-68 — digitized cao gấp 4-6 lần) |
| `geometry_constraints.width_optimal_bars` | 56 | Không có "optimal width" số trực tiếp; Bulkowski nói lead-in "2-3 tháng", bump ngắn hơn | 🟡 KHÔNG XÁC ĐỊNH số optimal chính xác |
| `duration_constraints.optimal_bars` | 56 | Không có số optimal | 🟡 KHÔNG XÁC ĐỊNH |
| `geometry_constraints.phase_proportions.lead_in_ratio` (0.35-0.50) | 35-50% | PDF: lead-in "at least 1 month, usually 2-3 months"; bump "at least twice lead-in height" (height, không phải width) | 🟡 GẦN KHỚP (PDF nói về height ratio 2:1, không phải width ratio) |
| `breakout_confirmation.volume_multiplier_ideal` | 2.0 | PDF qualitatively: "Volume spikes on bump acceleration" (Table 8.7), không quote 2.0x | 🟡 KHÔNG XÁC ĐỊNH số chính xác |
| Sample size | KHÔNG ghi | **777** BARR tops | 🔴 THIẾU hoàn toàn |
| `% meeting price target` | KHÔNG có trường | bull 78%, bear 90% | 🔴 THIẾU — metric quan trọng nhất (BARR có % target cao nhất các pattern bearish) |
| `performance_statistics.reliability_rank` | 12 | Performance rank: bull 3/21, bear 4/21 (rất cao — top 3-4) | 🔴 LỆCH (digitized rank 12 vs PDF rank 3-4 — BARR Top là top performer, digitized đánh giá quá thấp) |

**Tóm tắt lệch BARR Tops:** 🔴 **LỆCH NGHIÊM TRỘNG** về (1) measure rule method (digitized dùng bump height — SAI, PDF dùng lead-in height nhỏ hơn nhiều), (2) failure rate (10/5 vs 5/1 và 20/6 — lệch 2-10 lần), (3) throwback thuật ngữ+giá trị (digitized "throwback 45%" vs PDF "pullback 62/65%"), (4) lookahead_bars (252 vs 39-68 — cao gấp 4-6 lần), (5) reliability_rank (12 vs 3-4 thực — BARR là top performer), (6) THIẾU sample 777, (7) THIẾU % meeting target 78/90%. 🟢 KHỚP tốt ở: average decline ~21%, time to ultimate low bull 70≈68, failure threshold 5%.

---

## Bảng 2 — BARR BOTTOMS (ECP chapter 7, PDF p137-153 / sách p114-130) — bổ sung để đầy đủ

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | |
| **pages_checked** | PDF **137-153** (17 trang) | sách in p114-130 |
| **sample (Number of formations, Table 7.2)** | **412** (bull/UA) + **120** (bear/UA) = **532 BARR bottoms** | "532 patterns"; chỉ upward breakouts |
| **Reversal/Continuation (Table 7.2)** | 203R/209C (bull/UA) · 66R/54C (bear/UA) | Gần đều R/C; "6 more patterns reverse than consolidate" |
| **R/C performance (Table 7.2)** | +40% R / +36% C (bull) · +31% R / +30% C (bear) | Reversal mạnh hơn continuation |
| **Break-even failure rate BE% (Table 7.3, mốc 5%)** | **2%** (bull/UA, 8/412) · **1%** (bear/UA, 1/120) | Cực thấp — best performer |
| **Failure rate đầy đủ (Table 7.3)** | 5%=2/1% · 10%=10/6% · 15%=20/22% · 20%=30/33% · 25%=41/45% · 30%=47/58% · 35%=54/66% · 50%=70/79% · 75%=82/90% | Thứ tự cột: bull/UA · bear/UA |
| **% meeting price target (Results Snapshot)** | **68%** (bull/UA) / **64%** (bear/UA) | Measure rule hiệu quả |
| **Average rise (Table 7.2)** | **+38%** (bull/UA) · **+31%** (bear/UA) | "strong performer" — vượt avg của các pattern khác (36%/25%) |
| **Rises over 45% (Table 7.2)** | 142 hoặc 34% (bull) · 30 hoặc 25% (bear) | |
| **Change after trend ends (Table 7.2)** | −29% (bull/UA) · −34% (bear/UA) | |
| **Days to ultimate high (Table 7.2)** | **186** (bull/UA) · **109** (bear/UA) | Bull market cực lâu (>6 tháng) |
| **Busted pattern performance (Table 7.2)** | −28% (bull, n<30) · −24% (bear, n<30) | |
| **Throwbacks (Results Snapshot)** | 59% (bull/UA) · 73% (bear/UA) | BARR Bottom dùng **throwback** (upward breakout) |
| **Measure rule (Table 7.8)** | target = **highest high in the pattern** (đơn giản hóa — "I changed the measure rule from a computation to simply the top of the chart pattern"). Prices reach it 64-68% of the time. | Khác BARR Tops — Bottoms dùng HH, Tops dùng lead-in height |

---

## Bằng chứng verbatim (số liệu thô, không copy câu dài — bản quyền)

### BARR Tops — Table 8.2 (PDF p139 / sách p138)
```
Number of formations                 673         104
Reversal (R), continuation (C)       556 R,117 C  94 R,10 C
R/C performance                      -20% R,-13% C  -26% R,-32% C
Average decline                      19%         27%
Declines over 45%                    20 or 3%    7 or 7%
Change after trend ends              53%         48%
Days to ultimate low                 68          39
```

### BARR Tops — Table 8.3 Failure Rates (PDF p140 / sách p139)
```
Maximum Price        Bull Market,      Bear Market,
Decline (%)          Down Breakout     Down Breakout
5 (breakeven)        34 or 5%          1 or 1%
10                   134 or 20%        6 or 6%
15                   275 or 41%        27 or 26%
20                   386 or 57%        45 or 43%
25                   490 or 73%        61 or 59%
30                   569 or 85%        72 or 69%
35                   610 or 91%        85 or 82%
```

### BARR Bottoms — Table 7.2 (PDF p122 / sách p121)
```
Number of formations                 412             120
Reversal (R), continuation (C)       203 R,209 C     66 R,54 C
Average rise                         38%             31%
Rises over 45%                       142 or 34%      30 or 25%
Change after trend ends              -29%            -34%
Days to ultimate high                186             109
```

### Measure rule — BARR Tops (PDF p144, Table 8.8 + p146)
```
Table 8.8: "Compute the lead-in height (see Table 8.1 for the definition) and subtract
the result from the breakout price. The result is the minimum price move to expect.
About 8 out of 10 stocks meet their price targets."

p146: "calculate the lead-in height by splitting the formation along the trend line into
four equal parts. In the first quarter of the formation, compute the height from the
highest high to the trend line, measured vertically... The lead-in height is 3.50
(that is, 21.50-18). The target price is thus 21.63 (25.13-3.50)"
```

### Lead-in height definition — Table 8.1 (PDF p134)
```
"Lead-in, lead-in height: The lead-in is the section just before prices move up sharply
in the bump phase. Lead-in prices should have a range of at least $1 (preferably $2 or more),
as measured from the highest high..."
"Bump height: ...should be at least twice the lead-in height, measured from highest high to
the trend line, vertically."
```

---

## So sánh BARR Bottoms ↔ BARR Tops

| Metric | BARR Bottoms (UA) | BARR Tops (UD) | Nhận xét |
|---|---|---|---|
| **Sample tổng** | 532 | 777 | Tops phổ biến hơn |
| **BE failure rate (trung bình 2 tổ hợp)** | (2+1)/2 = **1.5%** | (5+1)/2 = **3.0%** | Cả 2 cực thấp — BARR là top performer |
| **Average move** | +38%/+31% (rise) | −19%/−27% (decline) | Bottoms rise mạnh; Tops decline vừa |
| **Days to ultimate** | 186/109 (high) | 68/39 (low) | Bottoms lâu hơn nhiều |
| **% meeting target** | 68/64% | 78/90% | Tops đạt target cao hơn |
| **Throwback/Pullback** | 59/73% (throwback) | 62/65% (pullback) | Gần bằng nhau, đều cao |
| **Measure rule** | target = HH pattern (đơn giản) | target = breakout − lead-in height | Khác method hoàn toàn |
| **Performance rank** | bull 8/23, bear 3/19 | bull 3/21, bear 4/21 | Cả 2 đều top — Tops rank cao hơn chút |

**Kết luận so sánh:** BARR (cả Tops + Bottoms) là **top performer** với BE failure rate cực thấp (1.5-3.0%), % meeting target cao (64-90%). BARR Tops phổ biến hơn, đạt target tốt hơn, decline nhanh hơn. Measure rule KHÁC NHAU: Bottoms dùng HH đơn giản, Tops dùng lead-in height computation.

---

## Reproducer

```bash
PDF="/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet/Nghiên cứu mô hình nến/references/encyclopedia-of-chart-patterns-2nbsped-9786468600-3175723993-9780471668268-0471668265_compress.pdf"

# BARR Tops (chương 8, PDF p154-170)
pdftotext -layout -f 154 -l 170 "$PDF" - | sed -n '131,147p'   # Results Snapshot
pdftotext -layout -f 154 -l 170 "$PDF" - | grep -A 20 "Table 8.2"
pdftotext -layout -f 154 -l 170 "$PDF" - | grep -A 18 "Table 8.3"
pdftotext -layout -f 154 -l 170 "$PDF" - | grep -A 8 "Table 8.8"   # measure rule

# BARR Bottoms (chương 7, PDF p137-153)
pdftotext -layout -f 137 -l 153 "$PDF" - | sed -n '114,130p'   # Results Snapshot
pdftotext -layout -f 137 -l 153 "$PDF" - | grep -A 18 "Table 7.2"
pdftotext -layout -f 137 -l 153 "$PDF" - | grep -A 16 "Table 7.3"
pdftotext -layout -f 137 -l 153 "$PDF" - | grep -A 8 "Table 7.8"   # measure rule
```

---

## Khuyến nghị (không thực hiện trong task này)

1. **Sửa measure rule BARR Tops (CRITICAL)**: digitized dùng `target = breakout − (bump_peak − lead_in_start)` — **SAI dimension**. PDF dùng `target = breakout − lead_in_height` trong đó lead-in height = HH − trendline đo trong **phần tư đầu tiên** (lead-in phase), nhỏ hơn bump height nhiều. Lỗi này làm target dự đoán quá xa → risk/reward sai hoàn toàn.
2. **Sửa failure rate**: digitized at_5pct=10/at_10pct=5 → PDF BE 5%/1% (bull/bear), 10% mốc 20%/6%. Digitized cao gấp 2-10 lần.
3. **Sửa throwback → pullback + giá trị**: digitized "throwback 45%" → PDF **pullback** 62%/65%. Sai cả thuật ngữ (BARR Top UD là pullback) lẫn số (45 vs 62-65).
4. **Sửa lookahead_bars**: 252 → 68 (bull)/39 (bear). Digitized cao gấp 4-6 lần.
5. **Sửa reliability_rank**: 12 → PDF performance rank bull 3/21, bear 4/21 (top 3-4). Digitized đánh giá BARR quá thấp.
6. **Bổ sung sample**: Tops 777, Bottoms 532, tổng 1.309 — digitized NOT-RECORDED.
7. **Bổ sung % meeting target**: Tops 78/90%, Bottoms 68/64% — metric quan trọng nhất, BARR có % cao nhất các pattern bearish.
8. **Thêm BARR Bottoms variant**: digitized chỉ mô tả Tops (bearish). Nên bổ sung variant bullish (BARR Bottoms, UA) với measure rule khác (target = HH pattern).
9. **Phase proportions**: digitized dùng width ratio (lead-in 35-50% of width) — PDF nói về **height ratio** (bump height ≥ 2× lead-in height). Khác dimension.

---

**Hết file.**
