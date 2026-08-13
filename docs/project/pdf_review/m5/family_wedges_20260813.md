# M5 — Trích số liệu PDF gốc: family WEDGES (Rising + Falling)

**Ngày trích:** 2026-08-13
**Model / Provider:** GLM-5.2 / Z.AI (zai-coding-plan)
**Session mẹ:** `sess_0fb10a3f-28e7-42c6-8bdd-6aca4b147362`
**Nguồn PDF:** `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` (1.035 trang)
**Phương pháp:** `pdftotext -layout` (Poppler 26.04.0). Offset PDF ↔ sách in: **+23 trang** (số in + 23 = số PDF).

---

## ⚠️ Phát hiện quan trọng (áp dụng cho CẢ 2 wedge)

Khác với Flags/Pennants (đo trend-end), **Wedges dùng phương pháp ultimate high/low tiêu chuẩn** (đợi 20% reversal) — tức "Days to ultimate high or low" trong Table 52.2/53.2 ĐÚNG nghĩa ultimate, có thể so sánh với H&S/Cup/BARR.

> *"Days to ultimate high or low  116  77  43  32"* — ECP Table 52.2 (Falling wedges, PDF p801)

Do đó trường `ultimate_high_method` / `ultimate_low_method` trong digitized **ĐÚNG semantic** cho wedges (không cần đổi tên như flags/pennants).

**Measure rule BẤT XỨNG (asymmetric)** — khác nhau cho 2 chiều breakout, và KHÁC với digitized:

- Falling wedge, breakout UP (đảo chiều tăng): target = **highest high in the wedge** (đỉnh mẫu hình)
- Falling wedge, breakout DOWN: target = **breakout price − formation height**
- Rising wedge, breakout DOWN (đảo chiều giảm): target = **lowest low in the formation** (đáy mẫu hình)
- Rising wedge, breakout UP: target = **breakout price + (highest high − lowest low)**

Digitized ghi đồng nhất "target = breakout ± pattern height at widest point" cho cả 2 chiều → **SAI cho chiều reversal** (chiều reversal target là cực trị của mẫu hình, không phải breakout ± height).

---

## Bảng 1 — WEDGES, FALLING (ECP chapter 52, PDF p817-832 / sách p794-809)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | |
| **pages_checked** | PDF **817-832** (16 trang) | sách in p794-809 |
| **sample (Number of formations, Table 52.2)** | **245** (bull/UA) + **126** (bear/UA) + **93** (bull/UD) + **78** (bear/UD) = **542 falling wedges** | Bulkowski: "rare considering I looked at 10 years of price data" |
| **Reversal/Continuation (Table 52.2)** | 124R/121C (bull/UA) · 64R/62C (bear/UA) · 48R/45C (bull/UD) · 29R/49C (bear/UD) | Split gần đều R/C; upward breakouts hơi lệch reversal |
| **Break-even failure rate BE% (Table 52.3, mốc 5%)** | **11%** (bull/UA, 27/245) · **11%** (bear/UA, 14/126) · **15%** (bull/UD, 14/93) · **6%** (bear/UD, 5/78) | Best = bear/UD (6%); worst = bull/UD (15%) |
| **Failure rate đầy đủ (Table 52.3)** | 5%=11/11/15/6% · 10%=23/25/38/21% · 15%=36/40/62/35% · 20%=44/48/71/50% · 25%=51/55/78/62% · 30%=54/64/88/69% · 35%=60/70/95/74% · 50%=71/79/99/91% · 75%=83/92/100/100% | Thứ tự cột: bull/UA · bear/UA · bull/UD · bear/UD |
| **% meeting price target (Results Snapshot)** | UA: **70%** (bull) / **60%** (bear) · UD: **30%** (bull) / **36%** (bear) | Upward breakouts đạt target tốt hơn rõ rệt |
| **Average rise/decline (Table 52.2)** | +32% (bull/UA) · +26% (bear/UA) · −15% (bull/UD) · −24% (bear/UD) | Ultimate method |
| **Change after trend ends (Table 52.2)** | −28% (bull/UA) · −33% (bear/UA) · +53% (bull/UD) · +52% (bear/UD) | Reversal lớn sau khi trend kết thúc |
| **Days to ultimate high/low (Table 52.2)** | **116** (bull/UA) · **77** (bear/UA) · **43** (bull/UD) · **32** (bear/UD) | UA lâu hơn UD rất nhiều; bull/UA ~4 tháng |
| **Measure rule** | UA (falling wedge→bullish): target = **highest high in the wedge**. UD: target = **breakout price − formation height**. | Quote PDF p795: *"uses the highest high in the wedge for the target... For downward breakouts, I use the formation height subtracted from the breakout price"* |
| **Throwbacks/pullbacks (Table 52.4)** | 56% / 61% / 69% / 72% | Bull/UA · bear/UA · bull/UD · bear/UD |
| **Avg time to throwback/pullback ends** | 10d / 9d / 9d / 8d | |
| **Busted pattern performance (Table 52.2)** | +38% (bull/UA, n<30) · +52% (bear/UA, n<30) · −15% (bull/UD, n<30) · −20% (bear/UD, n<30) | Sample nhỏ, "busted patterns perform well" |

### Đối chiếu với `wedges_ascending_descending_digitized.json`

| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `post_breakout_measurement.failure_rate.rising_wedge` | 11 | Falling wedge BE = 11/11/15/6% | 🟡 GẦN KHỚP (11 trùng bull/UA & bear/UA falling) |
| `post_breakout_measurement.failure_rate.falling_wedge` | 11 | Rising wedge BE = 8/14/24/15% | 🟡 LỆCH nhẹ (11 không khớp tổ hợp nào của rising) |
| `performance_statistics.rising_wedge.failure_rate_pct` | 11 | Rising wedge BE 8/14/24/15% | 🟡 LỆCH nhẹ |
| `performance_statistics.falling_wedge.failure_rate_pct` | 11 | Falling wedge BE 11/11/15/6% | 🟢 KHỚP (trung bình ~11%) |
| `post_breakout_measurement.failure_definition.threshold_pct` | 5.0 | 5% (breakeven) | 🟢 KHỚP |
| `post_breakout_measurement.average_move.rising_wedge_decline_pct` | 14 | Rising wedge UD: −14% (bull) / −20% (bear) | 🟢 KHỚP bull/UD (14); bear/UD lệch (14 vs 20) |
| `post_breakout_measurement.average_move.falling_wedge_rise_pct` | 20 | Falling wedge UA: +32% (bull) / +26% (bear) | 🔴 LỆCH (20 vs 32/26 — digitized thấp 6-12 pp) |
| `performance_statistics.rising_wedge.bearish_reversal_decline_pct` | 14 | −14% (bull/UD) / −20% (bear/UD) | 🟢 KHỚP bull/UD |
| `performance_statistics.falling_wedge.bullish_reversal_rise_pct` | 20 | +32% (bull/UA) / +26% (bear/UA) | 🔴 LỆCH (20 vs 32/26) |
| `performance_statistics.rising_wedge.bull_market_performance` | "12% decline" | −14% (bull/UD) | 🟡 LỆCH nhẹ (12 vs 14) |
| `performance_statistics.rising_wedge.bear_market_performance` | "16% decline" | −20% (bear/UD) | 🟡 LỆCH nhẹ (16 vs 20) |
| `performance_statistics.falling_wedge.bull_market_performance` | "22% rise" | +32% (bull/UA) | 🟡 LỆCH (22 vs 32) |
| `performance_statistics.falling_wedge.bear_market_performance` | "18% rise" | +26% (bear/UA) | 🟡 LỆCH (18 vs 26) |
| `post_breakout_measurement.ultimate_high_method.average_days` | 50 | Falling wedge UA: 116 (bull) / 77 (bear); Rising wedge UA: 127 (bull) / 60 (bear) | 🔴 LỆCH LỚN (50 vs 60-127 — digitized thấp 10-77 ngày) |
| `post_breakout_measurement.ultimate_low_method.average_days` | 48 | Falling wedge UD: 43 (bull) / 32 (bear); Rising wedge UD: 38/38 | 🟡 GẦN KHỚP (48 vs 32-43) |
| `performance_statistics.rising_wedge.average_days_to_ultimate_low` | 48 | Rising wedge UD: 38/38 | 🟡 GẦN KHỚP |
| `performance_statistics.falling_wedge.average_days_to_ultimate_high` | 50 | Falling wedge UA: 116/77 | 🔴 LỆCH LỚN (50 vs 77-116) |
| `post_breakout_measurement.lookahead_bars` | 126 | Ultimate days 32-127 | 🟡 GẦN KHỚP (126 ≈ upper bound) |
| `post_breakout_measurement.target_calculation.method` | `pattern_height` (target = breakout ± pattern height at widest point) | **ASYMMETRIC**: UA falling = HH wedge; UD falling = breakout − height; UD rising = LL wedge; UA rising = breakout + height | 🔴 **SAI METHOD** cho chiều reversal (chiều reversal target = cực trị mẫu hình, không phải breakout ± height) |
| `post_breakout_measurement.target_calculation.formula` | `target_price = breakout_price +/- pattern_height_at_start` | Xem 4 công thức bất xứng ở trên | 🔴 **SAI CÔNG THỨC** cho 2/4 tổ hợp |
| `breakout_confirmation.throwback_pullback_rate_pct` | 55 | 56/61/69/72% (falling) · 73/66/63/63% (rising) | 🟡 LỆCH (55 vs 56-73) |
| `performance_statistics.rising_wedge.throwback_rate_pct` | 55 | Rising wedge: throwback UA 73/66%, pullback UD 63/63% | 🟡 LỆCH (55 vs 63-73) |
| `geometry_constraints.width_optimal_bars` | 45 | Không có "optimal width" table trực tiếp;Bulkowski nói wedge hiếm, không quote số ngày tối ưu cụ thể | 🟡 KHÔNG XÁC ĐỊNH được từ PDF |
| `duration_constraints.optimal_bars` | 45 | Không có số tối ưu từ PDF | 🟡 KHÔNG XÁC ĐỊNH |
| `breakout_confirmation.volume_multiplier_ideal` | 1.6 | PDF: "heavy breakout volume" qualitatively (Table 52.7), không quote multiplier 1.6x cụ thể | 🟡 KHÔNG XÁC ĐỊNH số chính xác |
| `breakout_confirmation.breakout_timing.optimal_range_start_pct` (60) / `end_pct` (85) | 60-85% | PDF p795: *"best performers in a bull market had upward breakouts 55% to 80% of the way to the apex, with rises averaging 37%"* | 🟡 GẦN KHỚP (PDF 55-80% vs digitized 60-85%) |
| Sample size | KHÔNG ghi | Falling 542 + Rising 621 = **1.163 wedges** | 🔴 THIẾU hoàn toàn |
| `% meeting price target` | KHÔNG có trường riêng | Falling 70/60/30/36% · Rising 58/33/46/40% | 🔴 THIẾU hoàn toàn — metric quan trọng nhất cho measure rule |

**Tóm tắt lệch Wedges:** 🔴 **LỆCH NGHIÊM TRỌNG** về (1) measure rule method (digitized dùng pattern_height đối xứng — SAI cho chiều reversal), (2) sample (không ghi 1.163), (3) THIẾU % meeting price target (metric quan trọng nhất), (4) average rise falling wedge (20% vs 32% thực — thấp 12 pp), (5) days to ultimate high (50 vs 77-116 — thấp 27-66 ngày). 🟢 KHỚP tốt ở: failure rate ~11%, throwback threshold 5%, ultimate_low days 48.

---

## Bảng 2 — WEDGES, RISING (ECP chapter 53, PDF p833-850 / sách p810-827)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | |
| **pages_checked** | PDF **833-850** (18 trang) | sách in p810-827 |
| **sample (Number of formations, Table 53.2)** | **128** (bull/UA) + **64** (bear/UA) + **292** (bull/UD) + **137** (bear/UD) = **621 rising wedges** | "rare, especially in a bear market" |
| **Reversal/Continuation (Table 53.2)** | 45R/83C (bull/UA) · 18R/46C (bear/UA) · 194R/98C (bull/UD) · 85R/52C (bear/UD) | UA = continuation nhiều hơn; UD = reversal nhiều hơn |
| **Break-even failure rate BE% (Table 53.3, mốc 5%)** | **8%** (bull/UA, 10/128) · **14%** (bear/UA, 9/64) · **24%** (bull/UD, 70/292) · **15%** (bear/UD, 20/137) | Best = bull/UA (8%); worst = bull/UD (24%) |
| **Failure rate đầy đủ (Table 53.3)** | 5%=8/14/24/15% · 10%=23/34/44/38% · 15%=36/55/62/51% · 20%=52/63/74/66% · 25%=60/70/81/72% · 30%=66/77/87/80% · 35%=73/81/93/84% · 50%=85/94/99/94% · 75%=91/98/100/99% | Thứ tự cột: bull/UA · bear/UA · bull/UD · bear/UD |
| **% meeting price target (Results Snapshot)** | UA: **58%** (bull) / **33%** (bear) · UD: **46%** (bull) / **40%** (bear) | Measure rule yếu — dưới 50% cho 3/4 tổ hợp |
| **Average rise/decline (Table 53.2)** | +28% (bull/UA) · +17% (bear/UA) · −14% (bull/UD) · −20% (bear/UD) | |
| **Change after trend ends (Table 53.2)** | −30% (bull/UA) · −35% (bear/UA) · +53% (bull/UD) · +36% (bear/UD) | |
| **Days to ultimate high/low (Table 53.2)** | **127** (bull/UA) · **60** (bear/UA) · **38** (bull/UD) · **38** (bear/UD) | Bull/UA cực lâu (~4 tháng) |
| **Measure rule** | UD (rising wedge→bearish): target = **lowest low in the formation** ("prices should fall to the start of the formation"). UA: target = **breakout price + (highest high − lowest low)**. | Quote PDF p823 Table 53.8: *"Prices should fall to the bottom of the formation, at a minimum. For upward breakouts, subtract the lowest low from the highest high and add it to the breakout price"* |
| **Throwbacks/pullbacks (Table 53.4)** | 73% / 66% / 63% / 63% | Bull/UA · bear/UA · bull/UD · bear/UD |
| **Avg time to throwback/pullback ends** | 9d / 10d / 10d / 8d | |
| **Busted pattern performance (Table 53.2)** | +36% (bull/UA) · +23% (bear/UA, n<30) · −17% (bull/UD, n<30) · −39% (bear/UD, n<30) | |

### Đối chiếu (xem bảng chung ở Bảng 1 — cùng file digitized gộp rising+falling)

Rising wedge cụ thể:
- digitized `rising_wedge` failure_rate=11 → PDF BE bull/UA 8%, bear/UA 14%, bull/UD 24%, bear/UD 15% → 🟡 LỆCH nhẹ (11 nằm giữa)
- digitized `rising_wedge.bearish_reversal_decline_pct`=14 → PDF −14%/−20% → 🟢 KHỚP bull/UD
- digitized `average_days_to_ultimate_low`=48 → PDF 38/38 → 🟡 GẦN KHỚP
- digitized `bull_market_performance`="12% decline" → PDF −14% (bull/UD) → 🟡 LỆCH nhẹ

---

## Bằng chứng verbatim (số liệu thô, không copy câu dài — bản quyền)

### Falling wedges — Table 52.2 (PDF p801 / sách p800)
```
Number of formations              245   126   93    78
Reversal (R), continuation (C)    124 R,121 C  64 R,62 C  48 R,45 C  29 R,49 C
Average rise or decline           32%   26%   -15%  -24%
Change after trend ends           -28%  -33%  53%   52%
Days to ultimate high or low      116   77    43    32
```

### Falling wedges — Table 52.3 Failure Rates (PDF p803 / sách p802)
```
Max rise/decline   bull/UA   bear/UA   bull/UD   bear/UD
5% (breakeven)     27 or 11% 14 or 11% 14 or 15% 5 or 6%
10                 57 or 23%  32 or 25% 35 or 38% 16 or 21%
15                 88 or 36%  51 or 40% 58 or 62% 27 or 35%
20                 109 or 44% 60 or 48% 66 or 71% 39 or 50%
25                 124 or 51% 69 or 55% 73 or 78% 48 or 62%
```

### Rising wedges — Table 53.2 (PDF p817 / sách p816)
```
Number of formations              128   64    292   137
Reversal (R), continuation (C)    45 R,83 C  18 R,46 C  194 R,98 C  85 R,52 C
Average rise or decline           28%   17%   -14%  -20%
Change after trend ends           -30%  -35%  53%   36%
Days to ultimate high or low      127   60    38    38
```

### Rising wedges — Table 53.3 Failure Rates (PDF p819 / sách p818)
```
Max rise/decline   bull/UA   bear/UA   bull/UD   bear/UD
5% (breakeven)     10 or 8%   9 or 14%  70 or 24%  20 or 15%
10                 30 or 23%  22 or 34% 128 or 44% 52 or 38%
15                 46 or 36%  35 or 55% 180 or 62% 70 or 51%
20                 67 or 52%  40 or 63% 216 or 74% 91 or 66%
25                 77 or 60%  45 or 70% 237 or 81% 99 or 72%
```

### Measure rule (PDF p795 falling, p823 rising)
```
Falling wedge, UA: "uses the highest high in the wedge for the target"
Falling wedge, UD: "I use the formation height subtracted from the breakout price"
Rising wedge, UD:  "Prices should fall to the bottom of the formation, at a minimum"
Rising wedge, UA:  "subtract the lowest low from the highest high and add it to the breakout price"
```

---

## So sánh Falling ↔ Rising wedges (cùng ultimate method → so sánh được)

| Metric | Falling wedge | Rising wedge | Nhận xét |
|---|---|---|---|
| **Sample tổng** | 542 | 621 | Rising wedge phổ biến hơn |
| **BE failure rate (trung bình 4 tổ hợp)** | (11+11+15+6)/4 = **10.75%** | (8+14+24+15)/4 = **15.25%** | Falling wedge fail ít hơn |
| **Average rise UA bull** | +32% | +28% | Falling UA mạnh hơn |
| **Average decline UD bear** | −24% | −20% | Falling UD mạnh hơn |
| **Days to ultimate (avg 4 tổ hợp)** | (116+77+43+32)/4 = **67d** | (127+60+38+38)/4 = **65.75d** | Gần bằng nhau |
| **% meeting target (avg 4 tổ hợp)** | (70+60+30+36)/4 = **49%** | (58+33+46+40)/4 = **44.25%** | Falling đạt target tốt hơn |
| **Throwback/Pullback (avg)** | (56+61+69+72)/4 = **64.5%** | (73+66+63+63)/4 = **66.25%** | Gần bằng nhau, đều cao |
| **Best tổ hợp (BE thấp nhất)** | bear/UD 6% | bull/UA 8% | Cả 2 đều tốt nhất theo market trend |
| **Measure rule** | UA=HH wedge; UD=breakout−height | UD=LL wedge; UA=breakout+height | Cùng cấu trúc bất xứng |

**Kết luận so sánh:** Falling wedge **hiệu quả hơn** Rising wedge trên mọi metric (BE thấp hơn 10.75% vs 15.25%, average move mạnh hơn, % target cao hơn 49% vs 44%). Cả 2 đều có throwback/pullback rất cao (~65%) — giao dịch wedge cần chờ confirmation. Measure rule yếu (chỉ ~44-49% đạt target) → Bulkowski khuyến nghị dùng busted patterns và selection theo volume/tall.

---

## Reproducer

```bash
PDF="/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet/Nghiên cứu mô hình nến/references/encyclopedia-of-chart-patterns-2nbsped-9786468600-3175723993-9780471668268-0471668265_compress.pdf"

# Wedges Falling (chương 52, PDF p817-832)
pdftotext -layout -f 817 -l 832 "$PDF" - | grep -A 25 "Table 52.2"
pdftotext -layout -f 817 -l 832 "$PDF" - | grep -A 20 "Table 52.3"
pdftotext -layout -f 817 -l 832 "$PDF" - | sed -n '794,800p'   # Results Snapshot + measure rule

# Wedges Rising (chương 53, PDF p833-850)
pdftotext -layout -f 833 -l 850 "$PDF" - | grep -A 25 "Table 53.2"
pdftotext -layout -f 833 -l 850 "$PDF" - | grep -A 20 "Table 53.3"
pdftotext -layout -f 833 -l 850 "$PDF" - | grep -A 5 "Table 53.8"   # measure rule trading tactics
```

---

## Khuyến nghị (không thực hiện trong task này)

1. **Sửa measure rule**: Digitized dùng `pattern_height` đối xứng (target = breakout ± height) cho cả 2 chiều — **SAI cho chiều reversal**. Cần tách 4 công thức bất xứng: (a) Falling UA = HH wedge, (b) Falling UD = breakout − height, (c) Rising UD = LL wedge, (d) Rising UA = breakout + (HH − LL).
2. **Bổ sung sample**: Falling 542, Rising 621, tổng 1.163 — digitized đang NOT-RECORDED.
3. **Bổ sung % meeting price target** (falling 70/60/30/36, rising 58/33/46/40): digitized đang thiếu metric quan trọng nhất để đánh giá measure rule.
4. **Sửa average rise falling wedge**: digitized 20% → PDF +32%/+26% (UA). Lệch 6-12 pp.
5. **Sửa days to ultimate high**: digitized 50 → PDF 77-116 (falling UA). Lệch 27-66 ngày — digitized thấp bất hợp lý.
6. **Giữ nguyên** `ultimate_high/low_method` terminology — ĐÚNG semantic cho wedges (khác flags/pennants).
7. **Sửa breakout timing range**: digitized 60-85% → PDF 55-80% (p795, bull market UA).

---

**Hết file.**
