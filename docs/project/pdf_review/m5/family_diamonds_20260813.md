# M5 — Trích số liệu PDF gốc: family DIAMONDS (Diamond Bottom + Diamond Top)

**Ngày trích:** 2026-08-13
**Model / Provider:** GLM-5.2 / Z.AI (builtin:zai-coding-plan/GLM-5.2)
**Session mẹ:** `sess_0fb10a3f-28e7-42c6-8bdd-6aca4b147362`
**Nguồn PDF:** `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` (1.035 trang) — ECP 2nd edition
**Phương pháp:** `pdftotext -layout` (Poppler 26.04.0). Offset PDF ↔ sách in: **+23 trang**. Đã xác minh tại ch11 (book p179→PDF p202) và ch12 (p196→p219).

---

## ⚠️ Phát hiện quan trọng — Diamonds đo CẢ upward lẫn downward breakouts

Khác với digitized (chỉ ghi 1 hướng breakout), PDF đo **cả 2 hướng** cho mỗi diamond:

- **Diamond Bottom**: upward = bullish reversal (chính) · downward = bearish continuation (phụ)
- **Diamond Top**: upward = bullish continuation (phụ) · downward = bearish reversal (chính)

Bulkowski cũng cảnh báo sample nhỏ: *"I found only 45 bottoms [in 1st edition], so I am pleased to report that I have now located 295. Unfortunately, that is not enough for a good statistical analysis when you split it into four columns."* — ECP p186 (PDF p209)

---

## Bảng 1 — DIAMOND BOTTOM (ECP ch11, PDF p202-218 / sách p179-195)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | |
| **pages_checked** | PDF **202-218** (17 trang) | sách in p179-195 |
| **sample (Table 11.2)** | **140** (bull/Up) + **63** (bear/Up) + **72** (bull/Down) + **20** (bear/Down) = **295** | Bear/Down chỉ 20 — quá nhỏ, Bulkowski bảo "remove from consideration" |
| **Reversal/Continuation** | Up: 140R + 63R (100% reversal) · Down: 72C + 20C (100% continuation) | |
| **BE failure rate (Up)** | **4%** (bull) · **3%** (bear) | Rất tốt |
| **BE failure rate (Down)** | **10%** (bull) · **0%** (bear, n=20) | Bear/Down sample quá nhỏ |
| **Failure rate đầy đủ (Up, Table 11.3)** | 5%=4%/3% · 10%=12%/16% · 15%=24%/22% · 20%=31%/27% · 25%=38%/41% | Thứ tự: bull-Up · bear-Up |
| **% meeting price target (Up)** | **81%** (bull) · **60%** (bear) | Rất cao — diamond bottom là top performer |
| **Average rise (Up)** | **36%** (bull) · **36%** (bear) | Bằng nhau 2 thị trường |
| **Days to ultimate high (Up)** | **119** (bull) · **72** (bear) | Bull lâu hơn đáng kể |
| **Throwbacks (Up)** | **53%** (bull) · **60%** (bear) | |
| **Measure rule (Table 11.8)** | height = highest high − lowest low. Target = highest high + height (upward). | Cùng method các pattern khác |
| **Performance rank** | Up: 8/23 (bull) · 2/19 (bear) — xuất sắc | Down: 1/21 (bull) · 2/21 (bear) |

### Dữ liệu downward breakout (Diamond Bottom)

| BE% | Avg decline | Pullback | %target | Days to UL |
|-----|-------------|----------|---------|------------|
| 10%/0% | 21%/44%* | 71%/40% | 63%/80% | 35/28 |

*44% decline bear/Down — Bulkowski bảo *"Ignore the 44% decline. No bearish pattern that I know of has such a [high]..."* (sample quá nhỏ, n=20)

---

## Bảng 2 — DIAMOND TOP (ECP ch12, PDF p219-235 / sách p196-212)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | |
| **pages_checked** | PDF **219-235** (17 trang) | sách in p196-212 |
| **sample (Table 12.2)** | **88** (bull/Up) + **28** (bear/Up) + **203** (bull/Down) + **56** (bear/Down) = **375** | Down breakout phổ biến hơn Up |
| **Reversal/Continuation** | Up: continuation · Down: reversal | Ngược Diamond Bottom |
| **BE failure rate (Down — chính)** | **6%** (bull) · **4%** (bear) | Tốt |
| **BE failure rate (Up — phụ)** | **10%** (bull) · **0%** (bear, n=28) | |
| **% meeting price target (Down)** | **76%** (bull) · **59%** (bear) | |
| **Average decline (Down)** | **21%** (bull) · **24%** (bear) | Bear decline mạnh hơn |
| **Days to ultimate low (Down)** | **52** (bull) · **43** (bear) | |
| **Pullbacks (Down)** | **57%** (bull) · **57%** (bear) | |
| **Measure rule (Table 12.8)** | height = highest high − lowest low. Down: target = lowest low − height. | Cùng method |
| **Performance rank** | Down: 7/21 (bull) · 10/21 (bear) | Trung bình khá |

### Dữ liệu upward breakout (Diamond Top)

| BE% | Avg rise | Throwback | %target | Days to UH |
|-----|----------|-----------|---------|------------|
| 10%/0% | 27%/33% | 59%/54% | 69%/79% | 81/66 |

---

## Đối chiếu với `diamond_bottom_digitized.json`

| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `failure_rate.at_5pct` | 5 | BE Up: **4%** (bull) / **3%** (bear) | 🟡 LỆCH nhẹ (5 vs 4/3) |
| `failure_rate.at_10pct` | 2 | Failure@10% Up: **12%** (bull) / **16%** (bear) | 🔴 LỆCH LỚN (2 vs 12/16) |
| `average_rise.bull_market_pct` | 32% | **36%** (bull/Up) | 🟡 LỆCH nhẹ |
| `average_rise.bear_market_pct` | 25% | **36%** (bear/Up) | 🔴 LỆCH (25 vs 36) |
| `average_rise.overall_pct` | 28% | Trung bình Up = **36%** | 🟡 LỆCH nhẹ |
| `ultimate_high_method.average_days` | 77 | **119** (bull) / **72** (bear) | 🔴 LỆCH (77 gần bear nhưng sai bull) |
| `throwback_pullback.rate_pct` | 53% | **53%** (bull) / **60%** (bear) | 🟢 KHỚP (bull) |
| `target_calculation.method` | `pattern_height` (breakout + height) | highest high + (highest high − lowest low) | 🟢 KHỚP |
| `lookahead_bars` | 252 | Days to UH 72-119 | 🟡 LỆCH (252 vs 72-119) |
| Sample size | KHÔNG ghi | **295** total | 🔴 THIẾU |
| % meeting price target | KHÔNG ghi | **81%/60%** (Up) | 🔴 THIẾU metric quan trọng |
| Downward breakout data | KHÔNG ghi | BE 10%/0%, decline 21%/44%, %target 63%/80% | 🔴 THIẾU toàn bộ downward |
| `performance_statistics.reliability_rank` | 5 | Performance rank Up: 8/23 (bull) · 2/19 (bear) | 🟡 Khác thang đo — khó so |
| `performance_statistics.pattern_rarity` | "very_rare" | *"I found only 45 bottoms [1st ed]... now located 295"* | 🟢 KHỚP — đúng là hiếm |

### Bằng chứng verbatim (DBot failure rate)

> *"4% of the diamonds in a bull market with an upward breakout fail to rise more than 5%. A total of 12% fail to rise at least 10%."* — ECP p186 (PDF p209)

---

## Đối chiếu với `diamond_top_digitized.json`

| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `failure_rate.at_5pct` | 7 | BE Down: **6%** (bull) / **4%** (bear) | 🟡 LỆCH nhẹ (7 vs 6/4) |
| `failure_rate.at_10pct` | 3 | (không trích failure curve DT, nhưng DBot cho thấy 10% failure cao hơn 5% nhiều) | 🔴 KHÔNG THỂ VERIFY — cần trích Table 12.3 |
| `average_decline.bull_market_pct` | 15% | **21%** (bull/Down) | 🔴 LỆCH (15 vs 21) |
| `average_decline.bear_market_pct` | 21% | **24%** (bear/Down) | 🟡 LỆCH nhẹ |
| `average_decline.overall_pct` | 17% | Trung bình Down = **22.5%** | 🟡 LỆCH nhẹ |
| `ultimate_low_method.average_days` | 63 | **52** (bull) / **43** (bear) | 🟡 LỆCH nhẹ (63 cao hơn thực) |
| `throwback_rate_pct` | 57% | **57%** (bull) / **57%** (bear) | 🟢 KHỚP |
| `target_calculation.method` | `pattern_height` (breakout − height) | lowest low − (highest high − lowest low) | 🟢 KHỚP |
| `lookahead_bars` | 252 | Days to UL 43-52 | 🟡 LỆCH (252 vs 43-52) |
| Sample size | KHÔNG ghi | **375** total | 🔴 THIẾU |
| % meeting price target | KHÔNG ghi | **76%/59%** (Down) | 🔴 THIẾU metric quan trọng |
| Upward breakout data | KHÔNG ghi | BE 10%/0%, rise 27%/33%, %target 69%/79% | 🔴 THIẾU toàn bộ upward |
| `performance_statistics.pattern_rarity` | "rare" | Sample 375 — ít hơn DT-EE (460) nhưng nhiều hơn DBot (295) | 🟢 KHỚP |

---

## Tóm tắt lệch

| Pattern | 🟢 KHỚP | 🟡 LỆCH nhẹ | 🔴 LỆCH LỚN / THIẾU |
|---------|---------|-------------|----------------------|
| **Diamond Bottom** | throwback · target method · rarity | failure@5% · avg rise bull | failure@10% · avg rise bear · days to UH · sample · %target · downward breakout |
| **Diamond Top** | throwback · target method · rarity | failure@5% · avg decline bear · days to UL | avg decline bull · failure@10% · sample · %target · upward breakout |

**Lệch nghiêm trọng nhất:**
1. 🔴 **failure_rate@10% SAI lớn** — cả 2 file ghi 2-3% nhưng PDF cho thấy 10% failure thực tế là 12-16% (DBot). Digitized có thể đã nhầm "break-even failure rate" (5% mốc) với "failure at 10%".
2. 🔴 **Sample + %target THIẾU** hoàn toàn — %target 60-81% (DBot), 59-79% (DTop) là metric quan trọng nhất.
3. 🔴 **Thiếu toàn bộ breakout direction phụ** — DBot downward, DTop upward không được digitized.
4. 🟡 **avg rise/decline lệch nhẹ** — DBot bear 25% vs 36%, DTop bull 15% vs 21%.
5. 🟡 **days lệch nhẹ** — DBot 77 vs 119/72, DTop 63 vs 52/43.

---

## Reproducer

```bash
PDF="references/encyclopedia-of-chart-patterns-2nbsped-9786468600-3175723993-9780471668268-0471668265_compress.pdf"

# Diamond Bottom (ch11, PDF p202-218)
pdftotext -layout -f 202 -l 218 "$PDF" - | grep -A 40 "RESULTS SNAPSHOT"
pdftotext -layout -f 202 -l 218 "$PDF" - | grep -A 25 "Table 11.2"   # sample + days
pdftotext -layout -f 202 -l 218 "$PDF" - | sed -n '/Table 11.3/,/Table 11.4/p'  # failure curve

# Diamond Top (ch12, PDF p219-235)
pdftotext -layout -f 219 -l 235 "$PDF" - | grep -A 40 "RESULTS SNAPSHOT"
pdftotext -layout -f 219 -l 235 "$PDF" - | grep -A 25 "Table 12.2"
```

---

## Khuyến nghị (không thực hiện trong task này)

1. **Sửa failure_rate@10%**: DBot 12%/16% (không phải 2%), DTop cần trích Table 12.3.
2. **Bổ sung sample**: DBot 295, DTop 375.
3. **Bổ sung % meeting target**: DBot 81%/60% (Up), DTop 76%/59% (Down).
4. **Bổ sung breakout direction phụ**: DBot downward (BE 10%/0%, decline 21%/44%), DTop upward (BE 10%/0%, rise 27%/33%).
5. **Sửa avg rise DBot bear**: 25% → 36%.
6. **Sửa avg decline DTop bull**: 15% → 21%.
7. **Sửa days**: DBot 77 → 119/72, DTop 63 → 52/43.

---

**Hết file.**
