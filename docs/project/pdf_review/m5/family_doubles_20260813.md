# M5 — Trích số liệu PDF gốc: family DOUBLES (Double Bottoms + Double Tops)

**Ngày trích:** 2026-08-13
**Model / Provider:** GLM-5.2 / Z.AI (builtin:zai-coding-plan/GLM-5.2)
**Session mẹ:** `sess_0fb10a3f-28e7-42c6-8bdd-6aca4b147362`
**Nguồn PDF:** `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` (1.035 trang) — ECP 2nd edition
**Phương pháp:** `pdftotext -layout` (Poppler 26.04.0). Offset PDF ↔ sách in: **+23 trang** (số in + 23 = số PDF). Đã xác minh tại 3 điểm: ch1 BB (book p11→PDF p34), ch4 BT (p63→p86), ch13 DB-AA (p213→p236).

---

## ⚠️ Phát hiện quan trọng — Bulkowski tách 4 biến thể thành 4 chương riêng

ECP 2nd edition có **8 chương** cho family doubles (không phải 2 chương như giả định ban đầu):

| Ch | Pattern | Book p | PDF p |
|----|---------|--------|-------|
| 13 | Double Bottoms, Adam & Adam | 213 | 236-251 |
| 14 | Double Bottoms, Adam & Eve | 229 | 252-266 |
| 15 | Double Bottoms, Eve & Adam | 244 | 267-281 |
| 16 | Double Bottoms, Eve & Eve | 259 | 282-297 |
| 17 | Double Tops, Adam & Adam | 275 | 298-313 |
| 18 | Double Tops, Adam & Eve | 291 | 314-329 |
| 19 | Double Tops, Eve & Adam | 307 | 330-343 |
| 20 | Double Tops, Eve & Eve | 321 | 344-357 |

Tất cả 8 chương đều đo theo **ultimate high/low method** (đợi 20% reversal — giống H&S, Cup, không giống Flags/Pennants).

---

## Bảng 1 — DOUBLE BOTTOMS (4 biến thể, tất cả upward breakout, bullish reversal)

### Số liệu tổng hợp từ Results Snapshot + Table X.2

| Biến thể | Sample (bull+bear = tổng) | BE% bull/bear | Avg rise bull/bear | Throwback bull/bear | %target bull/bear | Days to UH bull/bear |
|----------|---------------------------|---------------|--------------------|---------------------|-------------------|---------------------|
| **AA** (ch13) | 206+75 = **281** | 5% / 7% | 35% / 24% | 64% / 61% | 66% / 48% | 136 / 105 |
| **AE** (ch14) | 319+70 = **389** | 5% / 4% | 37% / 33% | 59% / 54% | 66% / 56% | 160 / 99 |
| **EA** (ch15) | 161+66 = **227** | 4% / 8% | 35% / 23% | 57% / 56% | 66% / 47% | 160 / 101 |
| **EE** (ch16) | 412+74 = **486** | 4% / 7% | 40% / 24% | 55% / 46% | 67% / 54% | 170 / 77 |
| **Tổng** | **1383 DB** | — | — | — | — | — |

### Đường cong failure rate đầy đủ (DB-AA, Table 13.3)

Thứ tự cột: Bull-Up · Bear-Up

| Max rise | Bull Mkt | Bear Mkt |
|----------|----------|----------|
| 5% (BE) | 5% | 7% |
| 10% | 17% | 15% |
| 15% | 26% | 35% |
| 20% | 37% | 49% |
| 25% | 45% | 56% |
| 30% | 53% | 68% |
| 35% | 62% | 77% |
| 50% | 75% | 88% |
| 75% | 87% | 91% |

### Measure rule (DB, Table 13.8)

> *"Subtract the lowest low in whichever bottom is lower from the highest high between the two bottoms. Add the difference to the highest high."*

Công thức: `target = highest_high_between_bottoms + (highest_high_between_bottoms − lowest_low_in_lower_bottom)`.
%meeting: **66% bull, 48% bear**.

---

## Bảng 2 — DOUBLE TOPS (4 biến thể, tất cả downward breakout, bearish reversal)

### Số liệu tổng hợp từ Results Snapshot + Table X.2

| Biến thể | Sample (bull+bear = tổng) | BE% bull/bear | Avg decline bull/bear | Pullback bull/bear | %target bull/bear | Days to UL bull/bear |
|----------|---------------------------|---------------|-----------------------|---------------------|-------------------|----------------------|
| **AA** (ch17) | 188+108 = **296** | 8% / 11% | 19% / 19% | 61% / 48% | 72% / 68% | 51 / 32 |
| **AE** (ch18) | 238+102 = **340** | 14% / 7% | 18% / 22% | 59% / 58% | 69% / 69% | 49 / 45 |
| **EA** (ch19) | 212+105 = **317** | 13% / 5% | 15% / 24% | 64% / 54% | 72% / 79% | 43 / 40 |
| **EE** (ch20) | 264+196 = **460** | 11% / 2% | 18% / 25% | 59% / 51% | 73% / 76% | 44 / 39 |
| **Tổng** | **1413 DT** | — | — | — | — | — |

### Đường cong failure rate đầy đủ (DT-EE, Table 20.3)

Thứ tự cột: Bull-Down · Bear-Down

| Max decline | Bull Mkt | Bear Mkt |
|-------------|----------|----------|
| 5% (BE) | 11% | 2% |
| 10% | 31% | 11% |
| 15% | 50% | 25% |
| 20% | 67% | 44% |
| 25% | 77% | 59% |
| 30% | 84% | 69% |
| 35% | 88% | 78% |
| 50% | 98% | 93% |
| 75% | 100% | 99% |

### Measure rule (DT, Table 17.8)

> *"Compute the pattern height from the lowest low between the two tops to the highest peak and then divide in half. Subtract the result from the lowest low."*

Công thức: `height = (highest_peak − lowest_low_between_tops) / 2`; `target = lowest_low − height`.
%meeting: **72% bull, 68% bear**. Lưu ý: DT **chia đôi** height, khác DB (không chia).

---

## Đối chiếu với `double_bottoms_digitized.json`

| Trường digitized | Giá trị digitized | Giá trị PDF gốc | Status |
|---|---|---|---|
| `failure_rate.at_5pct` | 7 | BE bull: 4-5% · BE bear: 4-8% (4 biến thể) | 🟡 LỆCH nhẹ — số "7" gần với AA-bear/EE-bear nhưng không rõ nguồn |
| `failure_rate.at_10pct` | 5 | Failure@10% bull: 17-26% · bear: 15-35% | 🔴 LỆCH LỚN — "5" không khớp bất kỳ cột nào |
| `average_rise.bull_market_pct` | 19% | **35-40%** (bull, 4 biến thể) | 🔴 LỆCH LỚN — digitized chỉ bằng ~½ PDF |
| `average_rise.bear_market_pct` | 16% | **23-33%** (bear, 4 biến thể) | 🔴 LỆCH LỚN |
| `average_rise.overall_pct` | 18% | Trung bình 8 giá trị ≈ **31%** | 🔴 LỆCH LỚN |
| `average_rise.aa_variant_pct` | 21% | **35%** (bull) / 24% (bear) | 🔴 LỆCH LỚN |
| `average_rise.ae_variant_pct` | 17% | **37%** (bull) / 33% (bear) | 🔴 LỆCH LỚN |
| `average_rise.ea_variant_pct` | 19% | **35%** (bull) / 23% (bear) | 🔴 LỆCH LỚN |
| `average_rise.ee_variant_pct` | 15% | **40%** (bull) / 24% (bear) | 🔴 LỆCH LỚN |
| `ultimate_high_method.average_days` | 76 | **105-170** (bull) / 77-101 (bear) | 🔴 LỆCH LỚN |
| `ultimate_high_method.bull_market_days` | 83 | 136-170 (4 biến thể) | 🔴 LỆCH LỚN |
| `ultimate_high_method.bear_market_days` | 58 | 77-105 (4 biến thể) | 🔴 LỆCH LỚN |
| `throwback_pullback.rate_pct` | 56% | 46-64% (8 giá trị) | 🟢 GẦN KHỚP (56 nằm trong khoảng) |
| `target_calculation.method` | `pattern_height` (target = Peak + (Peak − Bottom)) | `highest_high + (highest_high − lowest_low)` | 🟢 KHỚP — cùng concept pattern_height |
| `lookahead_bars` | 252 | Days to UH 77-170 ≈ 77-170 bars daily | 🟡 LỆCH nhẹ (252 vs thực tế 77-170) |
| Sample size | KHÔNG ghi | **1383** total (281+389+227+486) | 🔴 THIẾU hoàn toàn |
| `performance_statistics.best_variant` | "AA (21% rise, 5% failure)" | Best rise bull = **EE (40%)**, best BE = EA/EE (4%) | 🔴 SAI — best variant sai hoàn toàn |
| `performance_statistics.worst_variant` | "EE (15% rise, 9% failure)" | EE thực ra **best** bull (40%); worst rise = EA bear (23%) | 🔴 SAI ngược |
| `variant_handling` failure rates (AA 5, AE 7, EA 6, EE 9) | AA 5, AE 7, EA 6, EE 9 | AA 5/7, AE 5/4, EA 4/8, EE 4/7 (bull/bear) | 🔴 LỆCH — chỉ AA-bull khớp; phần còn lại không khớp |
| % meeting price target | KHÔNG ghi | 47-67% (8 giá trị bull/bear) | 🔴 THIẾU metric quan trọng |

### Bằng chứng verbatim (DB-AA avg rise)

> *"Average rise: The 35% rise is about what you would expect from a bullish pattern in a bull market. The 24% bear market result..."* — ECP p214 (PDF p237)

---

## Đối chiếu với `double_tops_digitized.json`

| Trường digitized | Giá trị digitized | Giá trị PDF gốc | Status |
|---|---|---|---|
| `failure_rate.at_5pct` | 8 | BE bull: 8-14% · BE bear: 2-11% | 🟡 LỆCH nhẹ — "8" khớp AA-bull nhưng không rõ nguồn tổng hợp |
| `failure_rate.at_10pct` | 6 | Failure@10% bull: 31-50% · bear: 11-25% | 🔴 LỆCH LỚN — "6" không khớp |
| `average_decline.bull_market_pct` | 15% | **15-19%** (bull, 4 biến thể) | 🟢 GẦN KHỚP |
| `average_decline.bear_market_pct` | 19% | **19-25%** (bear, 4 biến thể) | 🟡 LỆCH nhẹ |
| `average_decline.overall_pct` | 17% | Trung bình 8 giá trị ≈ **20%** | 🟡 LỆCH nhẹ |
| `average_decline.aa_variant_pct` | 19% | 19% (bull) / 19% (bear) | 🟢 KHỚP |
| `average_decline.ae_variant_pct` | 21% | 18% (bull) / 22% (bear) | 🟢 GẦN KHỚP |
| `average_decline.ea_variant_pct` | 18% | 15% (bull) / 24% (bear) | 🟡 LỆCH nhẹ |
| `average_decline.ee_variant_pct` | 15% | 18% (bull) / 25% (bear) | 🔴 LỆCH — EE thực ra có decline CAO nhất bear (25%) |
| `ultimate_low_method.average_days` | 71 | **32-51** (bull) / 32-45 (bear) | 🔴 LỆCH LỚN — digitized cao gấp đôi |
| `ultimate_low_method.bull_market_days` | 79 | 43-51 (4 biến thể) | 🔴 LỆCH LỚN |
| `ultimate_low_method.bear_market_days` | 55 | 32-45 (4 biến thể) | 🟡 LỆCH nhẹ |
| `throwback_rate_pct` | 57% | 48-64% (8 giá trị) | 🟢 GẦN KHỚP |
| `target_calculation.method` | `pattern_height` (target = Trough − (Top − Trough)) | DT PDF: **height / 2**, target = lowest_low − height/2 | 🔴 **SAI CÔNG THỨC** — DT chia đôi height, digitized không chia |
| `target_calculation.formula` | `trough − pattern_height` | `lowest_low − (highest_peak − lowest_low)/2` | 🔴 SAI — thiếu bước /2 |
| `lookahead_bars` | 252 | Days to UL 32-51 ≈ 32-51 bars | 🔴 LỆCH (252 vs 32-51) |
| Sample size | KHÔNG ghi | **1413** total (296+340+317+460) | 🔴 THIẦU hoàn toàn |
| `performance_statistics.best_variant` | "AE (21% decline, 5% failure)" | Best decline bear = **EE (25%)**, best BE = EE (2% bear) | 🔴 SAI — EE mới là best |
| `performance_statistics.worst_variant` | "EE (15% decline, 10% failure)" | EE thực ra **best** (25% bear decline, 2% BE) | 🔴 SAI ngược |
| `variant_handling` failure rates (AA 6, AE 5, EA 8, EE 10) | AA 6, AE 5, EA 8, EE 10 | AA 8/11, AE 14/7, EA 13/5, EE 11/2 | 🔴 LỆCH — gần như toàn bộ sai |
| % meeting price target | KHÔNG ghi | 68-79% (8 giá trị) | 🔴 THIẾU metric quan trọng |

### Bằng chứng verbatim (DT measure rule)

> *"Compute the pattern height from the lowest low between the two tops to the highest peak and then divide in half. Subtract the result from the lowest low."* — ECP p287 (PDF p310)

---

## Tóm tắt lệch

| Pattern | 🟢 KHỚP | 🟡 LỆCH nhẹ | 🔴 LỆCH LỚN / THIẾU |
|---------|---------|-------------|----------------------|
| **Double Bottoms** | throwback rate · target method | failure@5% | avg rise (½ thực tế) · days to UH · failure@10% · sample · %target · best/worst variant · variant failure rates |
| **Double Tops** | avg decline (bull) · AA decline · throwback | avg decline (bear) · failure@5% | **target formula (thiếu /2)** · days to UL · failure@10% · sample · %target · best/worst variant (đảo ngược) |

**Lệch nghiêm trọng nhất:**
1. 🔴 **DT target formula SAI** — digitized không chia đôi height; PDF rõ ràng "divide in half". Hệ quả: target tính ra gấp đôi thực tế → risk/reward sai hoàn toàn.
2. 🔴 **DB average rise chỉ bằng ~½ PDF** — digitized 15-21% vs PDF 23-40%. Có thể digitized nhầm nguồn hoặc đo sai metric.
3. 🔴 **Best/worst variant ĐẢO NGƯỢC** — digitized ghi EE là worst, nhưng PDF cho thấy EE thực ra là **best performer** (40% bull rise cho DB, 25% bear decline + 2% BE cho DT).
4. 🔴 **Days to ultimate** lệch lớn — DB digitized 76 vs PDF 77-170; DT digitized 71 vs PDF 32-51.
5. 🔴 **Sample + %target THIẾU** hoàn toàn trong cả 2 file.

---

## Reproducer

```bash
PDF="references/encyclopedia-of-chart-patterns-2nbsped-9786468600-3175723993-9780471668268-0471668265_compress.pdf"

# Double Bottoms (ch13-16, PDF p236-297)
for ch in 236 252 267 282; do
  end=$((ch+15))
  pdftotext -layout -f $ch -l $end "$PDF" - | grep -A 20 "RESULTS SNAPSHOT"
done
pdftotext -layout -f 236 -l 251 "$PDF" - | grep -A 20 "Table 13.2"   # sample + days
pdftotext -layout -f 236 -l 251 "$PDF" - | sed -n '/Table 13.3/,/Table 13.4/p'  # failure curve

# Double Tops (ch17-20, PDF p298-357)
for ch in 298 314 330 344; do
  end=$((ch+15))
  pdftotext -layout -f $ch -l $end "$PDF" - | grep -A 20 "RESULTS SNAPSHOT"
done
pdftotext -layout -f 298 -l 313 "$PDF" - | grep -A 12 "Table 17.8"   # measure rule (divide in half)
```

---

## Khuyến nghị (không thực hiện trong task này)

1. **Sửa DT target formula**: thêm bước `/2` — `target = lowest_low − (highest_peak − lowest_low)/2`. Lỗi critical.
2. **Sửa DB average rise**: tăng từ 15-21% lên 23-40% (theo từng biến thể bull/bear). Có thể cần xác minh nguồn digitized gốc.
3. **Đảo best/worst variant**: EE là BEST (không phải worst) cho cả DB và DT.
4. **Sửa days to ultimate**: DB 77-170 (không phải 76), DT 32-51 (không phải 71).
5. **Bổ sung sample**: DB 1383, DT 1413.
6. **Bổ sung % meeting target**: DB 47-67%, DT 68-79%.
7. **Sửa variant failure rates** theo từng biến thể bull/bear (hiện chỉ 1 số/variant, không phân bull/bear).

---

**Hết file.**
