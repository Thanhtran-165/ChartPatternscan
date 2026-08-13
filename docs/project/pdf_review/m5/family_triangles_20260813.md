# M5 — Family TRIANGLES (Ascending / Descending / Symmetrical)

**Ngày trích:** 2026-08-13
**Model / Provider:** GLM-5.2 / Z.AI (zai-coding-plan)
**Session mẹ:** (theo task — worker subagent)
**Nguồn PDF:** `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` (Bulkowski, ECP 2nd ed, 1.035 trang)
**Digitized spec đối chiếu:** `extraction_phase_1/digitization/patterns_digitized/triangles_digitized.json`
**Phương pháp:** `pdftotext -layout` (Poppler 26.04.0 tại `/opt/homebrew/bin/pdftotext`) + scan TOC để map số trang PDF. Offset sách→PDF = **+23 trang** (đã kiểm chứng: book p711 = PDF p734).

## Quy ước viết tắt
- **UA / UD** = Upward / Downward Breakout
- **BE failure** = Break-even failure rate = tỷ lệ pattern không vượt quá mốc +5% (mốc breakeven) sau breakout
- **Bull/Bear** = Bull market / Bear market
- Thứ tự 4 cột số liệu thống nhất: **Bull/UA | Bear/UA | Bull/UD | Bear/UD**

## Cấu trúc chương PDF
| Biến thể | Chương sách | Trang sách | Trang PDF |
|---|---|---|---|
| Ascending Triangle | 47 | 711–729 | 734–752 |
| Descending Triangle | 48 | 730–747 | 753–770 |
| Symmetrical Triangle | 49 | 748–766 | 771–787 |

---

## Bảng 1 — Ascending Triangle (ch47, PDF p734–752)

| Trường | Giá trị (PDF gốc) | Nguồn |
|---|---|---|
| **pdf_path** | `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | — |
| **pages_checked** | PDF 734–752 (19 trang) | — |
| **Results Snapshot** | PDF p734–735 | — |
| **Sample (Number of formations)** | **1,092** tổng = 663 (Bull/UA) + 103 (Bear/UA) + 237 (Bull/UD) + 89 (Bear/UD) | Table 47.2, p739 |
| **Reversal / Continuation** | 372R/291C (Bull/UA) · 58R/45C (Bear/UA) · 106R/131C (Bull/UD) · 42R/47C (Bear/UD) | Table 47.2 |
| **Performance rank** | 17/23 (Bull/UA) · 11/19 (Bear/UA) · 9/21 (Bull/UD) · 9/21 (Bear/UD) | Snapshot |
| **BE failure (5% mốc)** | **13% / 12% / 11% / 3%** | Snapshot + Table 47.3 row "5 (breakeven)" |
| **Average rise or decline** | **+35% / +30% / –19% / –24%** | Table 47.2 |
| **Rises/declines over 45%** | 30% / 24% / 5% / 9% | Table 47.2 |
| **Days to ultimate high/low** | **185 / 97 / 64 / 39** | Table 47.2 |
| **Throwbacks (UA)** | 57% (Bull) / 54% (Bear) | Snapshot |
| **Pullbacks (UD)** | 49% (Bull) / 45% (Bear) | Snapshot |
| **% meeting price target** | **75% / 63% / 68% / 66%** | Snapshot |
| **Busted pattern performance** | +41% / +20% / –22% / –28% (2 cột cuối <30 samples) | Table 47.2 |
| **Change after trend ends** | –29% / –32% / +52% / +47% | Table 47.2 |
| **S&P 500 change** | +13% / –3% / +3% / –11% | Table 47.2 |

### Table 47.3 — Failure Rates chi tiết (p744)
| Mốc giá (%) | Bull/UA | Bear/UA | Bull/UD | Bear/UD |
|---|---|---|---|---|
| 5 (breakeven) | 83 or 13% | 12 or 12% | 25 or 11% | 3 or 3% |
| 10 | 159 or 24% | 30 or 29% | 60 or 25% | 9 or 10% |
| 15 | 212 or 32% | 37 or 36% | 99 or 42% | 26 or 29% |
| 20 | 278 or 42% | 50 or 49% | 133 or 56% | 38 or 43% |
| 25 | 329 or 50% | 54 or 52% | 169 or 71% | 46 or 52% |
| 30 | 368 or 56% | 59 or 57% | 189 or 80% | 63 or 71% |
| 35 | 411 or 62% | 65 or 63% | 209 or 88% | 70 or 79% |
| 50 | 484 or 73% | 79 or 77% | 229 or 97% | 85 or 96% |
| 75 | 542 or 82% | 90 or 87% | 237 or 100% | 89 or 100% |

### Measure rule (Table 47.8, p748)
> "Compute the height of the formation at the start of the triangle. Add the result to the price of the horizontal trend line (upward breakout) or subtract it from the break price (downward breakout). The result is the minimum price target."

Tóm tắt: **Target = (highest high − lowest low ở đầu pattern) ± giá trendline horizontal**. UA: cộng vào trendline trên. UD: trừ từ breakout price.

### Ghi chú đối chiếu digitized (`triangles_digitized.json`)
| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `performance_statistics.ascending.bull_market_rise_pct` | 32 | 35 (UA Bull) | LỆCH nhẹ (digitized thấp hơn 3%) |
| `performance_statistics.ascending.failure_rate_pct` | 8 | 13 (UA Bull) / 11 (UD Bull) | LỆCH — digitized thấp hơn |
| `performance_statistics.ascending.throwback_rate_pct` | 37 | 57 (UA Bull) / 54 (UA Bear) | LỆCH — digitized thấp hơn đáng kể |
| `performance_statistics.ascending.average_days_to_ultimate_high` | 62 | 185 (UA Bull) / 97 (UA Bear) | **LỆCH LỚN** (digitized thấp 3×) |
| Sample size | NOT-RECORDED | 1,092 | THIẾU trong digitized |

---

## Bảng 2 — Descending Triangle (ch48, PDF p753–770)

| Trường | Giá trị (PDF gốc) | Nguồn |
|---|---|---|
| **pdf_path** | (như Bảng 1) | — |
| **pages_checked** | PDF 753–770 (18 trang) | — |
| **Results Snapshot** | PDF p753–754 | — |
| **Sample (Number of formations)** | **1,166** tổng = 312 (Bull/UA) + 113 (Bear/UA) + 561 (Bull/UD) + 180 (Bear/UD) | Table 48.2, p760 |
| **Reversal / Continuation** | 84R/228C (Bull/UA) · 37R/76C (Bear/UA) · 377R/184C (Bull/UD) · 94R/86C (Bear/UD) | Table 48.2 |
| **Performance rank** | 5/23 (Bull/UA) · 7/19 (Bear/UA) · 10/21 (Bull/UD) · 12/21 (Bear/UD) | Snapshot |
| **BE failure (5% mốc)** | **7% / 9% / 16% / 11%** | Snapshot + Table 48.3 row "5 (breakeven)" |
| **Average rise or decline** | **+47% / +27% / –16% / –25%** | Table 48.2 |
| **Rises/declines over 45%** | 40% / 24% / 3% / 9% | Table 48.2 |
| **Days to ultimate high/low** | **178 / 86 / 50 / 32** | Table 48.2 |
| **Throwbacks (UA)** | 37% (Bull) / 52% (Bear) | Snapshot |
| **Pullbacks (UD)** | 54% (Bull) / 59% (Bear) | Snapshot |
| **% meeting price target** | **84% / 61% / 54% / 50%** | Snapshot |
| **Busted pattern performance** | +52% / +43% / –21% / –25% (3 cột cuối <30 samples) | Table 48.2 |
| **Change after trend ends** | –30% / –34% / +60% / +50% | Table 48.2 |
| **S&P 500 change** | +13% / –4% / +1% / –9% | Table 48.2 |

### Table 48.3 — Failure Rates chi tiết (p762)
| Mốc giá (%) | Bull/UA | Bear/UA | Bull/UD | Bear/UD |
|---|---|---|---|---|
| 5 (breakeven) | 21 or 7% | 10 or 9% | 91 or 16% | 19 or 11% |
| 10 | 35 or 11% | 25 or 23% | 211 or 38% | 42 or 23% |
| 15 | 62 or 20% | 34 or 30% | 300 or 53% | 70 or 39% |
| 20 | 81 or 26% | 45 or 40% | 394 or 70% | 82 or 46% |
| 25 | 111 or 36% | 53 or 47% | 449 or 80% | 105 or 58% |
| 30 | 131 or 42% | 65 or 58% | 487 or 87% | 126 or 70% |
| 35 | 152 or 49% | 75 or 66% | 513 or 91% | 145 or 81% |
| 50 | 198 or 63% | 89 or 79% | 552 or 98% | 169 or 94% |
| 75 | 238 or 76% | 102 or 90% | 561 or 100% | 180 or 100% |

### Measure rule (Table 48.8, p766)
> "Calculate the height of the formation by subtracting the highest high from the lowest low. Subtract the height from the value of the lower trend line to get the predicted minimum price decline. Alternatively, draw a line parallel to the down-sloping trend line starting at the lower left corner of the formation. The value of this line where prices break out of the formation becomes the target price. For upward breakouts, add the height to the price where it pierces the top trend line."

Tóm tắt: **Height = lowest low − highest high** (lấy trị tuyệt đối). UD: subtract từ horizontal trendline dưới. UA: add vào breakout price.

### Ghi chú đối chiếu digitized
| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `performance_statistics.descending.bear_market_decline_pct` | 16 | 25 (UD Bear) / 16 (UD Bull) | **LỆCH** — digitized khớp với UD Bull nhưng lệch UD Bear |
| `performance_statistics.descending.failure_rate_pct` | 9 | 7 (UA Bull) / 9 (UA Bear) / 16 (UD Bull) / 11 (UD Bear) | LỆCH — digitized chỉ khớp UA Bear |
| `performance_statistics.descending.pullback_rate_pct` | 54 | 54 (UD Bull) / 59 (UD Bear) | KHỚP (UD Bull) |
| `performance_statistics.descending.average_days_to_ultimate_low` | 55 | 50 (UD Bull) / 32 (UD Bear) | LỆCH nhẹ |
| Sample size | NOT-RECORDED | 1,166 | THIẾU trong digitized |

---

## Bảng 3 — Symmetrical Triangle (ch49, PDF p771–787)

| Trường | Giá trị (PDF gốc) | Nguồn |
|---|---|---|
| **pdf_path** | (như Bảng 1) | — |
| **pages_checked** | PDF 771–787 (17 trang) | — |
| **Results Snapshot** | PDF p771–772 | — |
| **Synonym** | "Coils" | Snapshot |
| **Sample (Number of formations)** | **1,347** tổng = 476 (Bull/UA) + 246 (Bear/UA) + 361 (Bull/UD) + 264 (Bear/UD) | Table 49.2, p779 |
| **Reversal / Continuation** | (cần xem Table 49.2 chi tiết — text không trích thêm) | Table 49.2 |
| **Performance rank** | 16/23 (Bull/UA) · 7/19 (Bear/UA) · 15/21 (Bull/UD) · 18/21 (Bear/UD) | Snapshot |
| **BE failure (5% mốc)** | **9% / 7% / 13% / 9%** | Snapshot + Table 49.3 row "5 (breakeven)" |
| **Average rise or decline** | **+31% / +26% / –17% / –19%** | Table 49.2 |
| **Rises/declines over 45%** | 30% / 19% / 3% / 7% | Table 49.2 |
| **Days to ultimate high/low** | **124 / 77 / 45 / 30** | Table 49.2 |
| **Throwbacks (UA)** | 37% (Bull) / 55% (Bear) | Snapshot |
| **Pullbacks (UD)** | 59% (Bull) / 62% (Bear) | Snapshot |
| **% meeting price target** | **66% / 57% / 48% / 42%** | Snapshot |
| **Change after trend ends** | –31% / –33% / +50% / +45% | Snapshot |

### Table 49.3 — Failure Rates chi tiết (p780)
| Mốc giá (%) | Bull/UA | Bear/UA | Bull/UD | Bear/UD |
|---|---|---|---|---|
| 5 (breakeven) | 44 or 9% | 18 or 7% | 48 or 13% | 25 or 9% |
| 10 | 104 or 22% | 44 or 18% | 121 or 34% | 81 or 31% |
| 15 | 162 or 34% | 78 or 32% | 179 or 50% | 126 or 48% |
| 20 | 206 or 43% | 115 or 47% | 232 or 64% | 153 or 58% |
| 25 | 233 or 49% | 139 or 57% | 279 or 77% | 193 or 73% |
| 30 | 262 or 55% | 162 or 66% | 309 or 86% | 211 or 80% |
| 35 | 284 or 60% | 178 or 72% | 330 or 91% | 226 or 86% |
| 50 | 350 or 74% | 210 or 85% | 355 or 98% | 251 or 95% |
| 75 | 411 or 86% | 229 or 93% | 361 or 100% | 263 or 100% |

### Measure rule (Table 49.8, p784)
> "Compute the formation height by subtracting the lowest low from the highest high. For upward breakouts, add the difference to the highest high or for downward breakouts, subtract the difference. Alternatively, symmetrical triangles can be halfway points in a move, so project accordingly."

Tóm tắt: **Height = highest high − lowest low**. UA: add vào highest high (hoặc breakout). UD: subtract difference. Variant: "halfway point" — project từ prior trend.

### Ghi chú đối chiếu digitized
| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `performance_statistics.symmetrical.upward_breakout_rise_pct` | 28 | 31 (UA Bull) / 26 (UA Bear) | LỆCH nhẹ |
| `performance_statistics.symmetrical.downward_breakout_decline_pct` | 15 | 17 (UD Bull) / 19 (UD Bear) | LỆCH nhẹ |
| `performance_statistics.symmetrical.failure_rate_pct` | 11 | 9 (UA Bull) / 7 (UA Bear) / 13 (UD Bull) / 9 (UD Bear) — trung bình ~9.5% | LỆCH nhẹ (digitized cao hơn) |
| `performance_statistics.symmetrical.throwback_pullback_rate_pct` | 40 | 37–62% (tùy 4 tổ hợp) | LỆCH — digitized là trung bình ước lượng |
| Sample size | NOT-RECORDED | 1,347 | THIẾU trong digitized |

---

## Tóm tắt đối chiếu digitized vs PDF (cả 3 biến thể)

Quy ước: 🟢 KHỚP | 🟡 LỆCH | 🔴 THIẾU trong digitized

| Biến thể | Sample PDF | BE failure PDF (Bull/UA · Bear/UA · Bull/UD · Bear/UD) | BE digitized | Lookahead PDF (days) | Lookahead digitized | Status tổng |
|---|---|---|---|---|---|---|
| Ascending | 1,092 | 13% · 12% · 11% · 3% | 8% (chỉ 1 giá trị) | 185 · 97 · 64 · 39 | 62d, 126 bars | 🟡 LỆCH |
| Descending | 1,166 | 7% · 9% · 16% · 11% | 9% (chỉ 1 giá trị) | 178 · 86 · 50 · 32 | 55d, 126 bars | 🟡 LỆCH |
| Symmetrical | 1,347 | 9% · 7% · 13% · 9% | 11% (chỉ 1 giá trị) | 124 · 77 · 45 · 30 | 126 bars | 🟡 LỆCH |

### Phát hiện lệch đáng chú ý

1. **Digitized chỉ có 1 giá trị failure rate mỗi biến thể** (8% / 9% / 11%) — PDF cho thấy failure rate phải tách 4 tổ hợp (Bull/Bear × UA/UD), dao động 3–16%. Trung bình đơn giản có thể gây sai lệch định hướng giao dịch.

2. **Lookahead digitized = 126 bars** cho cả 3 biến thể là quá thấp so với thực tế Bull/UA (Ascending 185d ≈ 185 bars, Symmetrical 124d). Với Bear/UD thì lại gần đúng (32–39d). Nên tách lookahead theo market+breakout thay vì 1 giá trị chung.

3. **Throwback/Pullback digitized không phân biệt**: Ascending digitized `throwback_rate_pct: 37` — PDF chỉ khớp Symmetrical Bull/UA (37%), không khớp Ascending (57%/54%).

4. **Sample size THIẾU hoàn toàn** trong digitized (NOT-RECORDED) — cần bổ sung 3 giá trị: 1,092 / 1,166 / 1,347.

5. **Average move digitized thấp hơn PDF 5-15%** ở Ascending (32 vs 35) và Symmetrical (28 vs 31) — có thể do digitized dùng giá trị trung bình qua 4 tổ hợp, trong khi PDF báo riêng Bull/UA (thường cao nhất).

---

## Reproducer (verify lại)

```bash
PDF="/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet/Nghiên cứu mô hình nến/references/encyclopedia-of-chart-patterns-2nbsped-9786468600-3175723993-9780471668268-0471668265_compress.pdf"

# Ascending Triangle — Results Snapshot + Table 47.2/47.3
pdftotext -layout -f 734 -l 735 "$PDF" - | head -50
pdftotext -layout -f 739 -l 741 "$PDF" - | grep -A 18 "Table 47.2"
pdftotext -layout -f 744 -l 744 "$PDF" - | grep -A 12 "Table 47.3"

# Descending Triangle — Results Snapshot + Table 48.2/48.3
pdftotext -layout -f 753 -l 754 "$PDF" - | head -50
pdftotext -layout -f 760 -l 761 "$PDF" - | grep -A 18 "Table 48.2"
pdftotext -layout -f 762 -l 762 "$PDF" - | grep -A 12 "Table 48.3"

# Symmetrical Triangle — Results Snapshot + Table 49.2/49.3
pdftotext -layout -f 771 -l 772 "$PDF" - | head -55
pdftotext -layout -f 779 -l 779 "$PDF" - | grep -A 18 "Table 49.2"
pdftotext -layout -f 780 -l 780 "$PDF" - | grep -A 12 "Table 49.3"
```

**Phương pháp dùng:** `pdftotext -layout` (Poppler 26.04.0). Offset sách→PDF = +23 trang. Đã kiểm chứng 3 chương (47/48/49) đều khớp TOC.

---

**Hết file.**
