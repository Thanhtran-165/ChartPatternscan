# M5 — Trích số liệu PDF gốc: family ROUNDING (Rounding Bottoms + Rounding Tops)

**Ngày trích:** 2026-08-13
**Model / Provider:** GLM-5.2 / builtin:zai-coding-plan (Z.AI Coding Plan)
**Session mẹ:** `sess_0fb10a3f-28e7-42c6-8bdd-6aca4b147362`
**Nguồn PDF:** `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` (ECP 2nd ed., 1.035 trang)
**Phương pháp:** `pdftotext -layout` (Poppler 26.04.0). Offset PDF ↔ sách in: **+23 trang**. Tự kiểm: chương 39 "Rounding Bottoms" sách p595 → PDF p618; chương 40 "Rounding Tops" sách p608 → PDF p631 (khớp offset +23).

---

## ⚠️ Phát hiện quan trọng (áp dụng cho cả Rounding)

Bulkowski mô tả Rounding thành **2 chương tách biệt** trong ECP:

- **Chapter 39 — Rounding Bottoms (RdB)** (p595-607 / PDF p618-630): chủ yếu **upward breakouts** ( UA = Up breakout). Synonyms: *"rounding turns, saucers"*. Được xếp top performer (rank 5/23 bull, 6/19 bear).
- **Chapter 40 — Rounding Tops (RdT)** (p608-623 / PDF p631-646): **cả 2 chiều breakout** (Up + Down). Synonyms: *"domes, rounding turn"*. Tách thành 4 tổ hợp (bull/UA, bear/UA, bull/UD, bear/UD) trong khi RdB chỉ 2 tổ hợp (bull/UA, bear/UA).

**KHÁC VỚI DIGITIZED:**
1. File `rounding_bottoms_tops_digitized.json` mô tả RdB như **reversal_both** (đảo chiều 2 hướng). Thực tế PDF: RdB **chỉ có upward breakouts** — Results Snapshot chỉ liệt kê 1 chiều (Upward Breakouts). RdB "thực chất" là **continuation** không phải reversal (*"This 'bottom' pattern acts as a continuation of the prevailing trend. If they were bottoms, they would act as reversals. Occasionally, price does reverse course and the RdB acts as a true bottom. More likely, however, is that RdBs appear in a [trend]"* — ECP p596). Reversal/Continuation ratio: bull 122R/139C, bear 91R/101C → continuation nhiều hơn.
2. RdT digitized gộp chung 2 chiều (avg_rise 25, avg_decline 18). Thực tế PDF: RdT có 4 tổ hợp với performance KHÁC NHAU rất nhiều:
   - bull/UA: avg rise +37% (best)
   - bear/UA: avg rise +19% (countertrend)
   - bull/UD: avg decline −19%
   - bear/UD: avg decline −23% (best bearish)
3. **Measure rule KHÁC** giữa RdB và RdT:
   - RdB: `target = right_saucer_lip + (right_saucer_lip − lowest_low)` (Table 39.9, dùng **right saucer lip**)
   - RdT: `target = formation_high + (formation_high − right_rim_low)` cho UA / `target = right_rim_low − (formation_high − right_rim_low)` cho UD (Table 40.9, dùng **formation high**)
4. **% meeting price target KHÁC** rất lớn: RdB 53-57% (moderate), RdT-UA 35-61% (moderate), RdT-UD chỉ **15-24%** (RẤT THẤP — measure rule fails 76-85% cho RdT downward!).

---

## Bảng 1 — ROUNDING BOTTOMS (ECP chapter 39, PDF p618-630 / sách p595-607)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | |
| **pages_checked** | PDF **618-630** (13 trang) | sách p595-607 |
| **sample (Number of formations, Table 39.2)** | **261** (bull) + **192** (bear) = **453 RdBs** | "I found 453 RdBs in 500 stocks from mid-1991 to mid-1996 and from 2000 to 2003" |
| **Reversal/Continuation (Table 39.2)** | bull: 122R/139C · bear: 91R/101C | Continuation nhiều hơn reversal ở cả 2 market |
| **R/C performance (Table 39.2)** | bull: 38%R/47%C · bear: 31%R/30%C | Continuation perform tốt hơn, đặc biệt bull (47% vs 38%) |
| **Break-even failure rate BE% (Table 39.3, mốc 5%)** | **5%** (bull, 14/261) · **5%** (bear, 9/192) | Cực thấp — top performer |
| **Failure rate đầy đủ (Table 39.3)** | 5%=5/5% · 10%=12/17% · 15%=21/27% · 20%=28/36% · 25%=39/48% · 30%=45/59% · 35%=50/65% · 50%=64/78% · 75%=81/92% | Thứ tự cột: bull · bear |
| **% meeting price target (Results Snapshot)** | **57%** (bull) / **53%** (bear) | Measure rule chỉ work ~half thời gian |
| **Average rise (Table 39.2)** | **+43%** (bull) · **+31%** (bear) | Bull market rise rất mạnh |
| **Rises over 45% (Table 39.2)** | 108 hoặc 41% (bull) · 48 hoặc 25% (bear) | |
| **Change after trend ends (Table 39.2)** | −31% (bull) · −33% (bear) | |
| **Days to ultimate high (Table 39.2)** | **189** (bull) · **105** (bear) | Bull market cực lâu (>6 tháng) |
| **Busted pattern performance (Table 39.2)** | −33% (bull, n<30) · −37% (bear, n<30) | |
| **Throwbacks (Table 39.4)** | 40% (bull) · 43% (bear) | |
| **Avg time to throwback ends** | 12 days (bull) · 9 days (bear) | |
| **Avg rise WITHOUT throwback** | 50% (bull) · 33% (bear) | |
| **Avg rise WITH throwback** | 33% (bull) · 28% (bear) | |
| **Volume trend (Results Snapshot)** | Upward (cả 2 market) | |
| **Performance rank (Results Snapshot)** | bull **5 out of 23** · bear **6 out of 19** | Top performer |
| **Measure rule (Table 39.9)** | `target = right_saucer_lip + (right_saucer_lip − lowest_low)`. Ví dụ (Fig 39.4): low=25, right_lip=31.44 → height=6.44; target = 31.44+6.44 = **37.88**. | Dùng **right saucer lip** (KHÔNG phải formation high như digitized ghi) |
| **Rims and performance (Table 39.8)** | Higher left rim: 48%/30% · Higher right rim: 39%/31% · Equal rims: 45%ᵃ/30%ᵃ | Bull market: left rim cao hơn perform tốt hơn |

---

## Bảng 2 — ROUNDING TOPS — UPWARD BREAKOUTS (ECP chapter 40, PDF p631-646 / sách p608-623)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | như trên | |
| **pages_checked** | PDF **631-646** (16 trang) | sách p608-623 |
| **sample (Number of formations, Table 40.2)** | **238** (bull/UA) · **173** (bear/UA) | Subset của tổng 776 RdTs |
| **Reversal/Continuation (Table 40.2)** | bull/UA: 84R/154C · bear/UA: 90R/83C | Bull/UA: continuation áp đảo (154 vs 84) |
| **R/C performance (Table 40.2)** | bull/UA: 36%R/38%C · bear/UA: 19%R/19%C | C perform ngang hoặc hơn R |
| **Break-even failure rate BE% (Table 40.3, mốc 5%)** | **9%** (bull/UA, 22/238) · **16%** (bear/UA, 28/173) | Bear/UA cao nhất |
| **% meeting price target (Results Snapshot)** | **61%** (bull/UA) · **35%** (bear/UA) | Bull/UA moderate; bear/UA thấp |
| **Average rise (Table 40.2)** | **+37%** (bull/UA) · **+19%** (bear/UA) | Bear/UA rise yếu (countertrend) |
| **Change after trend ends (Table 40.2)** | −31% (bull/UA) · −35% (bear/UA) | |
| **Days to ultimate high (Table 40.2)** | **161** (bull/UA) · **77** (bear/UA) | |
| **Throwbacks (Results Snapshot)** | 53% (bull/UA) · 52% (bear/UA) | |
| **Volume trend (Results Snapshot)** | Downward (cả 2) | |
| **Performance rank (Results Snapshot)** | bull/UA **13 out of 23** · bear/UA **16 out of 19** | Mid-tier |
| **Measure rule (Table 40.9)** | `target = formation_high + (formation_high − right_rim_low)` cho UA. Works 61% cho bull/UA. | Dùng **formation high** (KHÁC RdB dùng right saucer lip) |

---

## Bảng 3 — ROUNDING TOPS — DOWNWARD BREAKOUTS (ECP chapter 40, PDF p631-646)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **sample (Number of formations, Table 40.2)** | **157** (bull/UD) · **208** (bear/UD) | Subset của tổng 776 RdTs; tổng 4 tổ hợp: 238+173+157+208 = **776 RdTs** |
| **Reversal/Continuation (Table 40.2)** | bull/UD: 89R/68C · bear/UD: 90R/118C | Bear/UD continuation nhiều hơn |
| **R/C performance (Table 40.2)** | bull/UD: −19%R/−20%C · bear/UD: −21%R/−25%C | Bear/UD continuation decline mạnh nhất (−25%) |
| **Break-even failure rate BE% (Table 40.3, mốc 5%)** | **12%** (bull/UD, 19/157) · **9%** (bear/UD, 18/208) | |
| **Failure rate đầy đủ (Table 40.3)** | 5%=12/9% · 10%=27/22% · 15%=39/39% · 20%=57/51% · 25%=69/62% · 30%=79/72% · 35%=83/80% · 50%=98/96% · 75%=100/100% | Thứ tự cột: bull/UD · bear/UD |
| **% meeting price target (Results Snapshot)** | **24%** (bull/UD) · **15%** (bear/UD) | 🔴 **RẤT THẤP** — measure rule fails 76-85% cho RdT-UD |
| **Average decline (Table 40.2)** | **−19%** (bull/UD) · **−23%** (bear/UD) | |
| **Rises or declines over 45%** | 7 hoặc 4% (bull/UD) · 14 hoặc 7% (bear/UD) | Hiếm decline cực lớn |
| **Change after trend ends (Table 40.2)** | +57% (bull/UD) · +53% (bear/UD) | Rebound mạnh |
| **Days to ultimate low (Table 40.2)** | **45** (bull/UD) · **25** (bear/UD) | Bear market decline nhanh |
| **Busted pattern performance (Table 40.2)** | −28%ᵃ (bull/UD) · −38%ᵃ (bear/UD) | Busted RdT-UD perform tốt ("busted patterns perform well") |
| **Pullbacks (Results Snapshot)** | 48% (bull/UD) · 57% (bear/UD) | |
| **Performance rank (Results Snapshot)** | bull/UD **5 out of 21** · bear/UD **10 out of 21** | Bull/UD top 5 — best rank trong RdT |
| **Measure rule (Table 40.9)** | `target = right_rim_low − (formation_high − right_rim_low)` cho UD. Works chỉ 24%/15% — rất kém. | |

---

## Đối chiếu với `rounding_bottoms_tops_digitized.json`

| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `pattern_type` | `reversal_both` | RdB: continuation-heavy (122R/139C bull, 91R/101C bear) — KHÔNG phải reversal cả 2. RdT: 4 tổ hợp với R/C khác nhau (bull/UA 84R/154C, bear/UA 90R/83C, bull/UD 89R/68C, bear/UD 90R/118C) | 🟡 KHỚP tên nhưng sai semantic — Bulkowski nói RdB/RdT "acts as continuation of prevailing trend" nhiều hơn |
| `variant_handling.rounding_bottom.parameter_overrides.average_rise_pct` | 25 | bull 43% · bear 31% (Table 39.2) | 🔴 LỆCH LỚN (25 vs 31-43 — digitized thấp 6-18%) |
| `variant_handling.rounding_bottom.parameter_overrides.failure_rate_pct` | 5 | BE bull 5% · bear 5% (Table 39.3) | 🟢 KHỚP |
| `variant_handling.rounding_top.parameter_overrides.average_decline_pct` | 18 | bull/UD −19% · bear/UD −23% (Table 40.2) | 🟡 GẦN KHỚP (18 vs 19-23) — nhưng digitized bỏ qua RdT-UA (rise +37%/+19%) |
| `variant_handling.rounding_top.parameter_overrides.failure_rate_pct` | 6 | bull/UA 9% · bear/UA 16% · bull/UD 12% · bear/UD 9% | 🔴 LỆCH (6 vs 9-16) — digitized thấp |
| `performance_statistics.rounding_bottom.average_rise_pct` | 25 | bull 43% · bear 31% | 🔴 LỆCH LỚN (25 vs 31-43) |
| `performance_statistics.rounding_bottom.failure_rate_pct` | 5 | bull 5% · bear 5% | 🟢 KHỚP |
| `performance_statistics.rounding_bottom.time_to_target_days` | 84 | days to ultimate high: bull **189** · bear **105** | 🔴 LỆCH LỚN (84 vs 105-189) |
| `performance_statistics.rounding_bottom.pullback_rate_pct` | 45 | throwbacks: bull 40% · bear 43% | 🟢 GẦN KHỚP (45 vs 40-43) |
| `performance_statistics.rounding_top.average_decline_pct` | 18 | bull/UD −19% · bear/UD −23% · UA bull rise +37% · UA bear rise +19% | 🟡 GẦN KHỚP UD (18 vs 19-23); sai hoàn toàn cho UA |
| `performance_statistics.rounding_top.failure_rate_pct` | 6 | bull/UA 9% · bear/UA 16% · bull/UD 12% · bear/UD 9% | 🔴 LỆCH (6 vs 9-16) |
| `performance_statistics.rounding_top.time_to_target_days` | 63 | bull/UA 161 · bear/UA 77 · bull/UD 45 · bear/UD 25 | 🔴 LỆCH LỚN (63 vs 25-161) |
| `performance_statistics.rounding_top.throwback_rate_pct` | 42 | UA throwbacks: 53/52% · UD pullbacks: 48/57% | 🟡 GẦN KHỚP (42 vs 48-57) |
| `post_breakout_measurement.target_calculation.method` | `pattern_height` | RdB: `right_saucer_lip − lowest_low` (Table 39.9) — KHÔNG phải pattern_height tổng quát. RdT-UA: `formation_high − right_rim_low`. RdT-UD: symmetric `right_rim_low − (formation_high − right_rim_low)` | 🟡 KHỚP chung "height" nhưng dimension khác nhau (RdB dùng right lip, RdT dùng formation high) |
| `post_breakout_measurement.target_calculation.formula` | `target_price = breakout_price +/- (formation_height)` | RdB: `target = right_saucer_lip + (right_saucer_lip − lowest_low)`; RdT-UA: `target = formation_high + (formation_high − right_rim_low)`; RdT-UD: `target = right_rim_low − (formation_high − right_rim_low)` | 🔴 **SAI CÔNG THỨC** — digitized generic "breakout_price ± formation_height" không khớp dimension cụ thể của RdB (right saucer lip) hay RdT (formation high + right rim low) |
| `post_breakout_measurement.ultimate_high_method.average_days_bottom` | 84 | days to ultimate high: bull **189** · bear **105** | 🔴 LỆCH LỚN (84 vs 105-189) |
| `post_breakout_measurement.ultimate_low_method.average_days_top` | 63 | days to ultimate low: bull/UD **45** · bear/UD **25** · UA bull 161 · bear 77 | 🔴 LỆCH (63 vs 25-161) |
| `post_breakout_measurement.failure_definition.threshold_pct` | 5.0 | 5% (breakeven trong Table 39.3/40.3) | 🟢 KHỚP |
| `post_breakout_measurement.throwback_pullback.rate_pct` | 45 | RdB throwbacks 40-43% · RdT-UA throwbacks 53/52% · RdT-UD pullbacks 48/57% | 🟡 GẦN KHỚP RdB; lệch cho RdT |
| `post_breakout_measurement.average_move.bullish_bottom_rise_pct` | 25 | bull/UA **+43%** | 🔴 LỆCH LỚN (25 vs 43) |
| `post_breakout_measurement.average_move.bearish_top_decline_pct` | 18 | bull/UD −19% · bear/UD −23% | 🟡 GẦN KHỚP (18 vs 19-23) |
| `post_breakout_measurement.lookahead_bars` | 252 | RdB: 105-189 days · RdT-UA: 77-161 days · RdT-UD: 25-45 days | 🔴 LỆCH LỚN (252 vs 25-189) — digitized cao gấp 1.3-10 lần |
| `geometry_constraints.height_ratio_min` | 8.0 | PDF không có height ratio số trực tiếp; chỉ có throwback/pullback statistics theo height | 🟡 KHÔNG XÁC ĐỊNH |
| `geometry_constraints.width_min_bars` / `width_optimal_bars` | 21 / 63 | PDF qualitative: *"Tall or wide patterns perform better"* (Results Snapshot RdB+RdT) — không có số 21/63 | 🟡 KHÔNG XÁC ĐỊNH |
| `duration_constraints.optimal_bars` | 63 | Không có số optimal trong PDF | 🟡 KHÔNG XÁC ĐỊNH |
| `breakout_confirmation.volume_multiplier_min` / `volume_multiplier_ideal` | 1.5 / 2.0 | PDF: *"RdBs with heavy breakout volume tended to outperform"* (bull market) — qualitative, không có số 1.5/2.0 | 🟡 KHÔNG XÁC ĐỊNH |
| `breakout_confirmation.pullback_rate_pct` | 45 | RdB throwbacks 40/43% (KHỚP); RdT khác | 🟡 KHỚP RdB |
| Sample size | KHÔNG ghi | **RdB 453** + **RdT 776** = tổng **1.229** | 🔴 THIẾU hoàn toàn |
| `% meeting price target` | KHÔNG có trường variant-specific | RdB 53-57% · RdT-UA 35-61% · RdT-UD **15-24%** (RẤT THẤP) | 🔴 THIẾU — metric quan trọng nhất cho RdT-UD |

**Tóm tắt lệch Rounding:** 🔴 **LỆCH NGHIÊM TRỌNG** về (1) average rise RdB (25 vs 31-43% — thấp 6-18%), (2) days to ultimate (84/63 vs 25-189 — lệch 1.3-10 lần), (3) lookahead_bars (252 vs 25-189), (4) **measure rule formula** (digitized generic "breakout ± formation_height" vs PDF chi tiết 3 công thức khác nhau cho RdB/RdT-UA/RdT-UD với dimension khác nhau), (5) failure_rate_pct RdT (6 vs 9-16%), (6) **THIẾU sample 1.229**, (7) **THIẾU % meeting target** đặc biệt RdT-UD chỉ 15-24% (rất thấp). 🟢 **KHỚP** ở: BE failure rate RdB (5%), throwback rate RdB (45 vs 40-43), failure threshold (5%).

---

## Bằng chứng verbatim (số liệu thô, ≤3 dòng, bản quyền)

### Rounding Bottoms — Results Snapshot (PDF p618 / sách p595)
```
Performance rank              5 out of 23           6 out of 19
Break-even failure rate       5%                    5%
Average rise                  43%                   31%
Throwbacks                    40%                   43%
Percentage meeting price target   57%               53%
Volume trend                  Upward                Upward
```

### Rounding Bottoms — Table 39.2 (PDF p620 / sách p597)
```
Number of formations          261            192
Reversal (R), continuation (C) 122 R,139 C   91 R,101 C
R/C performance               38% R,47% C    31% R,30% C
Average rise                  43%            31%
Rises over 45%                108 or 41%     48 or 25%
Days to ultimate high         189            105
```

### Rounding Bottoms — Table 39.3 Failure Rates (PDF p621 / sách p598)
```
Maximum Price Rise (%)    Bull Market       Bear Market
5 (breakeven)             14 or 5%          9 or 5%
10                        31 or 12%         32 or 17%
20                        74 or 28%         70 or 36%
```

### Rounding Bottoms — Measure rule (PDF p625, Table 39.9)
```
"Subtract the lowest low from the right saucer lip. Add the difference to the
value of the right saucer lip to get the target price. This is the minimum price
move to expect. The measure rule only works about half of the time."

(Fig 39.4 example): low = 25, right saucer lip = 31.44, formation height = 6.44
target = 31.44 + 6.44 = 37.88
```

### Rounding Tops — Results Snapshot (PDF p631 / sách p608)
```
Upward Breakouts:    BE 9%/16%, rise 37%/19%, throwback 53%/52%, %target 61%/35%
Downward Breakouts:  BE 12%/9%, decline 19%/23%, pullback 48%/57%, %target 24%/15%
```

### Rounding Tops — Table 40.2 (PDF p632 / sách p609)
```
Number of formations         238    173    157    208
Reversal (R), continuation   84 R,  90 R,  89 R,  90 R,
                              154 C  83 C   68 C   118 C
Average rise or decline      37%    19%    -19%   -23%
Days to ultimate high/low    161    77     45     25
```

### Rounding Tops — Table 40.3 Failure Rates (PDF p633 / sách p610)
```
5 (breakeven)      22 or 9%    28 or 16%   19 or 12%   18 or 9%
10                 47 or 20%   62 or 36%   42 or 27%   46 or 22%
20                 82 or 34%   111 or 64%  90 or 57%   107 or 51%
```

### Rounding Tops — Measure rule (PDF p642, Table 40.9)
```
"Compute the formation height by subtracting the right rim low from the
formation high. Add the difference to the high for upward breakouts or subtract
the difference from the right rim low for downward breakouts to get the target
price."

"The rule works reasonably well, 61% of the time, for RdTs in bull markets
with upward breakouts."
```

---

## So sánh RdB ↔ RdT

| Metric | Rounding Bottoms (RdB) | Rounding Tops — UA | Rounding Tops — UD | Nhận xét |
|---|---|---|---|---|
| **Sample** | 453 | 411 (238+173) | 365 (157+208) | RdT tổng 776 nhiều hơn RdB 453 |
| **BE failure rate (trung bình)** | (5+5)/2 = **5.0%** | (9+16)/2 = **12.5%** | (12+9)/2 = **10.5%** | RdB an toàn nhất |
| **Average move** | rise +43%/+31% | rise +37%/+19% | decline −19%/−23% | RdB rise mạnh nhất bull; RdT-UD decline vừa |
| **Days to ultimate** | 189/105 | 161/77 | 45/25 | RdB + RdT-UA lâu; RdT-UD nhanh |
| **% meeting target** | 57/53% | 61/35% | **24/15%** | RdT-UD measure rule gần như KHÔNG work |
| **Throwback/Pullback** | 40/43% (throwback) | 53/52% (throwback) | 48/57% (pullback) | Đều cao |
| **Performance rank** | bull 5/23 · bear 6/19 | bull 13/23 · bear 16/19 | bull 5/21 · bear 10/21 | RdB + RdT-UD top performer; RdT-UA mid-tier |
| **Measure rule** | right saucer lip + height | formation high + height | right rim low − height | **3 công thức KHÁC** — không generic |

**Kết luận so sánh:** RdB là top performer (BE 5%, rank 5/23, rise 43%); RdT-UD cũng top performer (rank 5/21 bull, decline nhanh 25-45d) nhưng **% meeting target cực thấp (15-24%)** → measure rule gần như vô dụng cho RdT-UD. RdB dùng **right saucer lip** làm base cho measure, RdT dùng **formation high**. Cả 2 đều continuation-heavy (RdB 139C>122R, RdT-UA 154C>84R).

---

## Reproducer

```bash
PDF="/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet/Nghiên cứu mô hình nến/references/encyclopedia-of-chart-patterns-2nbsped-9786468600-3175723993-9780471668268-0471668265_compress.pdf"

# Rounding Bottoms (ch.39, PDF p618-630)
pdftotext -layout -f 618 -l 620 "$PDF" - | sed -n '/RESULTS SNAPSHOT/,/Tour/p'
pdftotext -layout -f 620 -l 622 "$PDF" - | grep -A 15 "Table 39.2"
pdftotext -layout -f 621 -l 622 "$PDF" - | grep -A 13 "Table 39.3"
pdftotext -layout -f 625 -l 626 "$PDF" - | grep -A 8 "Table 39.9"

# Rounding Tops (ch.40, PDF p631-646)
pdftotext -layout -f 631 -l 633 "$PDF" - | sed -n '/RESULTS SNAPSHOT/,/Tour/p'
pdftotext -layout -f 632 -l 634 "$PDF" - | grep -A 16 "Table 40.2"
pdftotext -layout -f 633 -l 634 "$PDF" - | grep -A 13 "Table 40.3"
pdftotext -layout -f 642 -l 643 "$PDF" - | grep -A 8 "Table 40.9"
```

---

## Khuyến nghị (không thực hiện trong task này)

1. **Sửa measure rule (CRITICAL)**: digitized dùng generic `target = breakout_price ± formation_height` — **SAI**. PDF có **3 công thức khác nhau**:
   - RdB: `target = right_saucer_lip + (right_saucer_lip − lowest_low)`
   - RdT-UA: `target = formation_high + (formation_high − right_rim_low)`
   - RdT-UD: `target = right_rim_low − (formation_high − right_rim_low)`
2. **Sửa average rise RdB**: digitized 25% → PDF bull **43%** / bear **31%** (lệch 6-18%, thấp hơn thực tế).
3. **Sửa days to ultimate**: digitized 84/63 → PDF 25-189 days (4 tổ hợp khác nhau).
4. **Sửa lookahead_bars**: 252 → 25-189 days theo tổ hợp.
5. **Sửa failure_rate_pct RdT**: digitized 6% → PDF 9-16% (4 tổ hợp).
6. **Bổ sung sample**: RdB 453 + RdT 776 = tổng 1.229.
7. **Bổ sung % meeting target**: RdB 53-57% · RdT-UA 35-61% · RdT-UD **15-24%** (rất thấp — cảnh báo người dùng).
8. **Tách RdT thành UA/UD riêng**: digitized gộp chung RdT nhưng UA (rise) và UD (decline) có performance khác nhau. Cần 2 variant riêng.
9. **Sửa pattern_type**: RdB thực chất **continuation_both** (RdB 122R/139C — continuation nhiều hơn), không phải `reversal_both`. Bulkowski nói rõ *"This 'bottom' pattern acts as a continuation of the prevailing trend"*.
10. **Volume pattern**: digitized ghi `dome_shaped` — PDF nói RdB có 51% dome-shaped volume, 49% khác; RdT có volume trend "Downward". Cần sửa cho chính xác.

---

**Hết file.**
