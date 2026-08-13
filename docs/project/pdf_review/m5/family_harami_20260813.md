# M5 — Trích số liệu PDF gốc: family HARAMI (Bearish + Bullish + Harami Cross Bear/Bull)

**Ngày trích:** 2026-08-13
**Model / Provider:** GLM-5.2 / Z.AI (zai-coding-plan)
**Session mẹ:** `sess_0fb10a3f-28e7-42c6-8bdd-6aca4b147362`
**Nguồn PDF:** `references/Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf` (966 trang) — **EC (Candlestick book), KHÔNG PHẢI ECP**
**Phương pháp:** `pdftotext -layout` (Poppler 26.04.0). Offset PDF ↔ sách in: **+24 trang** (số in + 24 = số PDF).

> ⚠️ **Sửa offset từ đầu vào nhiệm vụ:** task mô tả ghi "offset +23 (in + 23 = PDF)" nhưng **thực tế là +24**. Kiểm chứng: PDF p398 = in p374 (mở chương 43, dòng đầu "43 / Harami, Bearish" ở PDF p398, footer "374" hiện ở cuối trang). Lệch 1 trang so với mô tả đầu vào. Nếu dùng +23 sẽ lùi sang trang cuối chương 42 (Hanging Man) → sai chương.

---

## ⚠️ Phát hiện quan trọng — 4 điểm khác biệt căn bản

### 1. Harami THƯỜNG (Ch.43/44) là BODY-based — KHÔNG PHẢI RANGE-based như inside_day
Định nghĩa Bulkowski (EC Table 43.1 & 44.1): **thân nến ngày 2 (open-close) nằm trong thân nến ngày 1**, bỏ qua shadow (bóng):

> *"The small black candle on the second day must have a body that ﬁts inside the body of the white candle."* — EC Table 43.1 (Harami Bearish, in p375 / PDF p399)

> *"Second day: A small-bodied white candle. The body must be within the prior candle's body."* — EC Table 44.1 (Harami Bullish, in p384 / PDF p408)

**Đối chiếu với inside_day**: inside_day đo theo **high/low (RANGE)**: `today_high < yesterday_high AND today_low > yesterday_low` (strict, không cho bằng). Harami đo theo **body** (`max(open,close)` của ngày 2 nằm giữa `max(open,close)` và `min(open,close)` của ngày 1; tương tự với `min(open,close)` ngày 2).

→ **Scanner hiện tại (inside_day) KHÔNG phát hiện được harami** vì đo sai metric. Cần detector riêng `harami_body` như task mô tả.

### 2. Harami CROSS (Ch.45/46) lại là RANGE-based — KHÔNG phải BODY-based
Vì doji (nến ngày 2) có body gần bằng 0 (open ≈ close), không thể dùng body để so. Bulkowski chuyển sang đo theo **high-low range**:

> *"Second day: A doji (open and close are equal or nearly so) with a trading range inside the price range of the prior day."* — EC Table 45.1 (Harami Cross Bearish, in p393 / PDF p417)

> *"Second day: A doji...with a high-low price range that fits inside the range of the black candle."* — EC Table 46.1 (Harami Cross Bullish, in p401 / PDF p425)

→ **Harami Cross có thể tái dụng nhiều logic của inside_day** (range-based) + thêm điều kiện "ngày 2 là doji" (body_pct < threshold nhỏ).

### 3. Quy tắc "tops hoặc bottoms bằng nhau nhưng không cả hai"
Cả 4 chương đều nêu: **hoặc đỉnh 2 thân trùng đỉnh thân 1, HOẶC đáy 2 thân trùng đáy thân 1, nhưng KHÔNG đồng thời cả hai** (nếu cả hai trùng → không còn là harami vì thân 2 = thân 1).

> *"Either the tops or the bottoms of the bodies can be equal but not both."* — EC Table 43.1, in p375

> *"The tops or bottoms of the two bodies can be the same price but not both."* — EC Table 44.1, in p384

→ Detector harami cần rule: `body2_high <= body1_high AND body2_low >= body1_low AND NOT(body2_high == body1_high AND body2_low == body1_low)`. Cho phép "top bằng" hoặc "bottom bằng" (dấu ≤/≥ thay vì </>).

### 4. EC dùng methodology KHÁC ECP — KHÔNG có failure rate
Giống như 13 file M5 trước: EC không có "Break-even failure rate", "Average rise", "Days to ultimate high/low" như ECP. Metric chính:
- **Behavior and Rank**: reversal rate % (theoretical vs actual), frequency rank (1-103), overall performance rank (1-103)
- **Table X.2 General Statistics**: Number found, Reversal/continuation performance, Candle end to breakout (median days), Candle end to trend end (median days)
- **Table X.3 Height Statistics**: % meeting price target (measure rule), median candle height, tall/short candle performance
- **KHÔNG có failure rate curve 5-75%** như ECP

---

## Bảng 1 — HARAMI, BEARISH (EC Ch.43, PDF p398-406 / sách in p374-382)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf` | EC, không phải ECP |
| **pages_checked** | PDF **398-406** (9 trang) | sách in p374-382 |
| **Behavior and Rank** | Theoretical: **Bearish reversal**. Actual bull market: **Bullish continuation 53%** (ranking 36). Actual bear market: **Bearish reversal 50%** (ranking 46). Frequency: **26th out of 103**. Overall performance over time: **72nd out of 103** (poor). | "with a reversal rate of just 47% (bull market), that's too close to random to be useful" — in p374. Reversal rate 47% = 100% − 53% continuation |
| **sample (Number found, Table 43.2)** | **8,122** (bull/Up) + **2,342** (bear/Up) + **7,189** (bull/Down) + **2,347** (bear/Down) = **20,000** (capped) | "I limited the number of patterns found to 20,000" — in p376 |
| **Reversal/Continuation performance (Table 43.2)** | 6.03% C (bull/Up) · 6.83% C (bear/Up) · −5.97% R (bull/Down) · −9.32% R (bear/Down) | Bull = continuation (trend up → breakout up); Bear = reversal (trend up → breakout down). Move nhỏ ~6-9% |
| **S&P 500 change (Table 43.2)** | 1.41% · 0.70% · −0.98% · −2.68% | Harami beats S&P trong mọi category |
| **Candle end to breakout (median, days, Table 43.2)** | 4 · 4 · 4 · 4 | Đều 4 ngày — "It takes four days for price to close either above the top of the harami or below the bottom" — in p377 |
| **Candle end to trend end (median, days, Table 43.2)** | 7 · 7 · 8 · 9 | Down breakout lâu hơn 1-2 ngày vì uptrend đang tiến triển |
| **Median candle height as % of breakout price (Table 43.3)** | 3.08% · 4.26% · 3.03% · 4.37% | |
| **Short candle performance (Table 43.3)** | 4.73% · 5.35% · −4.80% · −7.98% | |
| **Tall candle performance (Table 43.3)** | 7.73% · 8.42% · −7.52% · −10.88% | "Tall candles perform substantially better than short ones, so trade only tall ones" — in p377 |
| **% meeting price target — measure rule (Table 43.3)** | **63%** (bull/Up) · **58%** (bear/Up) · **64%** (bull/Down) · **64%** (bear/Down) | |
| **Measure rule (Table 43.3 + text)** | target = **breakout ± (height × %)**. Height = (HH − LL) của cả pattern. Ví dụ Schmidt: height = 63 − 61 = 2, target up = 63 + (2 × 63%) = 64.26, target down = 61 − (2 × 64%) = 59.72. | Quote EC in p378: *"Compute the height of the candle pattern and multiply it by the appropriate percentage shown in the table; then apply it to the breakout price"* |
| **Median upper/lower shadow as % breakout (Table 43.3)** | Upper: 0.45% · 0.65% · 0.49% · 0.76%. Lower: 0.63% · 0.92% · 0.62% · 0.91% | |
| **Closing price confirmation reversal rate (Table 43.5)** | 72% (bull) · 72% (bear) | Wait for close lower next day |
| **Candle color confirmation reversal rate (Table 43.5)** | 69% · 71% | |
| **Opening gap confirmation reversal rate (Table 43.5)** | 56% · 58% | Lowest |
| **Reversal rate trend up/breakout down (Table 43.5)** | 47% (bull) · 50% (bear) | Gần random |
| **Continuation rate trend up/breakout up (Table 43.5)** | 53% (bull) · 50% (bear) | |
| **Failure rate / Break-even failure rate** | **KHÔNG CÓ** (EC không publish cho candlestick) | EC dùng "% meeting price target" thay thế |
| **Identification (Table 43.1)** | 2 dòng. Trend: UP. Day 1: tall **white** candle. Day 2: small **black** candle, **body nằm trong body ngày 1** (ignore shadows). Tops hoặc bottoms của 2 body có thể bằng, nhưng không cả hai. **Day 2 KHÔNG được là doji** (nếu doji → chuyển sang Harami Cross). | Bulkowski **KHÔNG cho phép body color khác** (chỉ white/black): "Some ignore the candle color, but I don't" — in p374 |

---

## Bảng 2 — HARAMI, BULLISH (EC Ch.44, PDF p407-415 / sách in p383-391)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf` | EC |
| **pages_checked** | PDF **407-415** (9 trang) | sách in p383-391 |
| **Behavior and Rank** | Theoretical: **Bullish reversal**. Actual bull market: **Bullish reversal 53%** (ranking 43). Actual bear market: **Bullish reversal 51%** (ranking 41). Frequency: **25th out of 103**. Overall performance over time: **38th out of 103** (respectable). | "acts as a reversal just three percentage points above random. That's not encouraging" — in p383. Bearish harami (Ch.43) tệ hơn (overall rank 72) |
| **sample (Number found, Table 44.2)** | **8,163** (bull/Up) + **2,315** (bear/Up) + **7,318** (bull/Down) + **2,204** (bear/Down) = **20,000** (capped) | "I limited my surveillance to 20,000 patterns" — in p385 |
| **Reversal/Continuation performance (Table 44.2)** | 7.11% R (bull/Up) · 9.71% R (bear/Up) · −5.72% C (bull/Down) · −10.40% C (bear/Down) | Bull = reversal (trend down → breakout up); Down = continuation |
| **S&P 500 change (Table 44.2)** | 1.84% · 1.51% · −0.93% · −2.64% | Harami beats S&P |
| **Candle end to breakout (median, days, Table 44.2)** | 3 · 3 · 4 · 3 | "It takes three or four days" — in p385 |
| **Candle end to trend end (median, days, Table 44.2)** | 9 · 9 · 6 · 7 | Down breakout ngắn hơn vì downtrend đã tiến triển |
| **Median candle height as % of breakout price (Table 44.3)** | 3.25% · 5.04% · 3.45% · 5.13% | |
| **Short candle performance (Table 44.3)** | 5.54% · 7.32% · −4.70% · −8.21% | |
| **Tall candle performance (Table 44.3)** | 9.16% · 12.28% · −7.06% · −12.99% | |
| **% meeting price target — measure rule (Table 44.3)** | **69%** (bull/Up) · **66%** (bear/Up) · **59%** (bull/Down) · **61%** (bear/Down) | Cao hơn bearish harami |
| **Measure rule (Table 44.3 + text)** | target = **breakout ± (height × %)**. Ví dụ Clint: height = 39 − 37 = 2, target up = 39 + (2 × 69%) = 40.38, target down = 37 − (2 × 61%) = 35.78. | Quote EC in p386: *"Compute the height of the candle pattern and multiply it by the appropriate percentage shown in the table; then apply it to the breakout price"* |
| **Median upper/lower shadow as % breakout (Table 44.3)** | Upper: 0.68% · 1.12% · 0.74% · 1.14%. Lower: 0.58% · 0.89% · 0.54% · 0.87% | |
| **Closing price confirmation reversal rate (Table 44.5)** | 76% (bull) · 75% (bear) | Wait for close higher next day |
| **Candle color confirmation reversal rate (Table 44.5)** | 75% · 74% | |
| **Opening gap confirmation reversal rate (Table 44.5)** | 61% · 59% | Lowest, nhưng Performance Indicators (Table 44.6) cho opening gap performance tốt nhất |
| **Reversal rate trend down/breakout up (Table 44.5)** | 53% (bull) · 51% (bear) | |
| **Continuation rate trend down/breakout down (Table 44.5)** | 47% (bull) · 49% (bear) | |
| **Failure rate / Break-even failure rate** | **KHÔNG CÓ** | EC dùng "% meeting price target" thay thế |
| **Identification (Table 44.1)** | 2 dòng. Trend: DOWN. Day 1: tall **black** candle. Day 2: small **white** candle, **body nằm trong body ngày 1**. Tops hoặc bottoms có thể bằng, không cả hai. | Bulkowski chỉ cho phép black/white combo (không cho body color khác) |

---

## Bảng 3 — HARAMI CROSS, BEARISH (EC Ch.45, PDF p416-423 / sách in p392-399)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf` | EC |
| **pages_checked** | PDF **416-423** (8 trang) | sách in p392-399 |
| **Behavior and Rank** | Theoretical: **Bearish reversal**. Actual bull market: **Bullish continuation 57%** (ranking 25). Actual bear market: **Bullish continuation 56%** (ranking 33). Frequency: **45th out of 103**. Overall performance over time: **80th out of 103** (poor). | "acts as a continuation pattern most often, despite claims that it's a reversal... performance is close to random (50%)" — in p392. Ngược lý thuyết: bearish nhưng lại continuation |
| **sample (Number found, Table 45.2)** | **10,693** (bull/Up) + **756** (bear/Up) + **7,945** (bull/Down) + **606** (bear/Down) = **20,000** (capped) | Bear market ít samples (756/606) — pattern hiếm trong bear market |
| **Reversal/Continuation performance (Table 45.2)** | 5.73% C (bull/Up) · 8.82% C (bear/Up) · −5.52% R (bull/Down) · −7.32% R (bear/Down) | Bull = continuation; Down = reversal |
| **S&P 500 change (Table 45.2)** | 1.01% · 0.15% · −0.47% · −1.79% | Harami cross beats S&P substantially |
| **Candle end to breakout (median, days, Table 45.2)** | 4 · 3 · 4 · 4 | "It takes three or four days" — in p394 |
| **Candle end to trend end (median, days, Table 45.2)** | 6 · 7 · 7 · 8 | "median time to the trend end is about a week" — in p394 |
| **Median candle height as % of breakout price (Table 45.3)** | 2.75% · 3.92% · 2.78% · 3.84% | Nhỏ hơn harami thường (3-4% vs 3-5%) |
| **Short candle performance (Table 45.3)** | 4.76% · 8.77% · −4.77% · −6.78% | |
| **Tall candle performance (Table 45.3)** | 7.18% · 8.89% · −6.63% · −7.99% | |
| **% meeting price target — measure rule (Table 45.3)** | **69%** (bull/Up) · **67%** (bear/Up) · **68%** (bull/Down) · **66%** (bear/Down) | Rất đồng đều ~66-69% |
| **Measure rule (Table 45.3 + text)** | target = **breakout ± (height × %)**. Ví dụ Morgan: height = 83 − 80 = 3, target up = 83 + (3 × 69%) = 85.07, target down = 80 − (3 × 68%) = 77.96. | Quote EC in p395: *"Compute the height of the candle pattern and multiply it by the appropriate percentage shown in the table; then apply it to the breakout price"* |
| **Median upper/lower shadow as % breakout (Table 45.3)** | Upper: **0.00%** · 0.29% · **0.00%** · 0.27%. Lower: 0.58% · 0.82% · 0.52% · 0.52% | Upper shadow median = 0.00% — doji thường không có upper shadow |
| **Closing price confirmation reversal rate (Table 45.5)** | 66% (bull) · 69% (bear) | |
| **Candle color confirmation reversal rate (Table 45.5)** | 65% · 66% | |
| **Opening gap confirmation reversal rate (Table 45.5)** | 54% · 53% | |
| **Reversal rate trend up/breakout down (Table 45.5)** | 43% (bull) · 44% (bear) | Rất thấp — chủ yếu continuation |
| **Continuation rate trend up/breakout up (Table 45.5)** | 57% (bull) · 56% (bear) | |
| **Failure rate / Break-even failure rate** | **KHÔNG CÓ** | |
| **Identification (Table 45.1)** | 2 dòng. Trend: UP. Day 1: tall **white** candle. Day 2: **doji** (open ≈ close). **High-low range của doji nằm TRONG high-low range của nến trắng**. | ⚠️ **RANGE-based, không phải body-based** — khác harami thường. Quote EC Table 45.1, in p393: *"A doji...with a trading range inside the price range of the prior day"* |

---

## Bảng 4 — HARAMI CROSS, BULLISH (EC Ch.46, PDF p424-432 / sách in p400-408)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf` | EC |
| **pages_checked** | PDF **424-432** (9 trang) | sách in p400-408 |
| **Behavior and Rank** | Theoretical: **Bullish reversal**. Actual bull market: **Bearish continuation 55%** (ranking 32). Actual bear market: **Bearish continuation 56%** (ranking 32). Frequency: **47th out of 103**. Overall performance over time: **50th out of 103** (middle). | "acts as a bearish continuation pattern... Performs less well (overall rank of 50) than the regular bullish harami (overall rank of 38)" — in p400. Ngược lý thuyết |
| **sample (Number found, Table 46.2)** | **8,381** (bull/Up) + **623** (bear/Up) + **10,212** (bull/Down) + **784** (bear/Down) = **20,000** (capped) | Bear market rất ít (623/784) — "harami cross is indeed rare in a bear market" — in p403 |
| **Reversal/Continuation performance (Table 46.2)** | 7.01% R (bull/Up) · 9.70% R (bear/Up) · −5.99% C (bull/Down) · −9.45% C (bear/Down) | Up = reversal; Down = continuation |
| **S&P 500 change (Table 46.2)** | 1.48% · 0.46% · −0.43% · −1.48% | Harami cross beats S&P |
| **Candle end to breakout (median, days, Table 46.2)** | 4 · 4 · 4 · 3 | "It takes about four days" — in p403 |
| **Candle end to trend end (median, days, Table 46.2)** | 8 · 8 · 6 · 6 | Down breakout ngắn hơn |
| **Median candle height as % of breakout price (Table 46.3)** | 2.82% · 4.28% · 2.91% · 4.33% | |
| **Short candle performance (Table 46.3)** | 5.69% · 8.00% · −5.04% · −7.95% | |
| **Tall candle performance (Table 46.3)** | 8.84% · 12.19% · −7.28% · −11.41% | |
| **% meeting price target — measure rule (Table 46.3)** | **74%** (bull/Up) · **73%** (bear/Up) · **68%** (bull/Down) · **70%** (bear/Down) | Cao nhất trong 4 chương |
| **Measure rule (Table 46.3 + text)** | target = **breakout ± (height × %)**. Ví dụ Randy: height = 13 − 12 = 1, target up = (1 × 74%) + 13 = 13.74, target down = 12 − (1 × 68%) = 11.32. | Quote EC in p404: *"Compute the height of the candle pattern and multiply it by the appropriate percentage shown in the table; then apply it to the breakout price"* |
| **Median upper/lower shadow as % breakout (Table 46.3)** | Upper: 0.55% · 0.62% · 0.63% · 0.75%. Lower: 0.17% · 0.35% · **0.00%** · 0.28% | Lower shadow median = 0.00% cho bull/Down |
| **Closing price confirmation reversal rate (Table 46.5)** | 71% (bull) · 69% (bear) | "self-fulfilling prophecy" — in p406 |
| **Candle color confirmation reversal rate (Table 46.5)** | 70% · 66% | |
| **Opening gap confirmation reversal rate (Table 46.5)** | 57% · 53% | |
| **Reversal rate trend down/breakout up (Table 46.5)** | 45% (bull) · 44% (bear) | Thấp |
| **Continuation rate trend down/breakout down (Table 46.5)** | 55% (bull) · 56% (bear) | |
| **Failure rate / Break-even failure rate** | **KHÔNG CÓ** | |
| **Identification (Table 46.1)** | 2 dòng. Trend: DOWN. Day 1: tall **black** candle. Day 2: **doji** (open ≈ close). **High-low range của doji nằm TRONG range của nến đen**. | ⚠️ **RANGE-based** — Quote EC Table 46.1, in p401: *"A doji...with a high-low price range that fits inside the range of the black candle"* |

---

## Đối chiếu với `inside_day_digitized.json`

`inside_day_digitized.json` = detector RANGE-based strict (today_high < yesterday_high AND today_low > yesterday_low, **không cho bằng**). Đối chiếu từng nhóm trường:

| Trường digitized inside_day | Giá trị digitized | Đối với Harami THƯỜNG (body-based) | Đối với Harami CROSS (range-based) |
|---|---|---|---|
| `geometry_constraints.range_constraints.today_high_below_yesterday_high.strict` | true (strict <) | 🔴 **KHÔNG DÙNG ĐƯỢC** — harami đo theo BODY (open-close), không phải high-low. Logic hoàn toàn khác | 🟡 DÙNG ĐƯỢC một phần — harami cross đo high-low range, nhưng cho phép equal (tops hoặc bottoms có thể bằng). inside_day strict < sẽ loại một số harami cross hợp lệ |
| `geometry_constraints.range_constraints.equal_values_allowed` | high_equal=false, low_equal=false | 🔴 KHÔNG ÁP DỤNG (harami dùng body) | 🟡 **SAI** — harami cross CHO PHÉP tops bằng HOẶC bottoms bằng (chỉ cấm cả hai cùng bằng). inside_day cấm cả hai → sẽ bỏ sót harami cross hợp lệ |
| `geometry_constraints.range_ratio` (min 0.1, max 0.99) | range ngày 2 = 10-99% ngày 1 | 🔴 KHÔNG ÁP DỤNG (body-based) | 🟡 HỢP LÝ về mặt ý tưởng, nhưng PDF không publish range_ratio cho harami cross (chỉ publish candle height median 2.75-4.33% of breakout price cho cả pattern) |
| `geometry_constraints.height_definition` | "Today's range relative to yesterday's range" | 🔴 **SAI định nghĩa** — harami dùng body ratio | 🟢 ĐÚNG định nghĩa cho harami cross |
| `post_breakout_measurement.target_calculation.method` | `breakout_magnitude` — "target = breakout_price +/- (prior_trend_strength * breakout_magnitude)" | 🔴 **SAI METHOD** — PDF dùng `target = breakout ± (candle_height × % multiplier)`, % multiplier = 58-74% (per chương) | 🔴 **SAI METHOD** — tương tự harami thường |
| `post_breakout_measurement.target_calculation.formula` | "prior_trend_strength * breakout_magnitude" | 🔴 **SAI CÔNG THỨC** — PDF: `(HH − LL) × multiplier_pct` | 🔴 **SAI CÔNG THỨC** |
| `post_breakout_measurement.failure_definition.threshold_pct` | 1.0 | 🔴 **BỊA** — EC không publish failure rate cho candlestick | 🔴 **BỊA** |
| `post_breakout_measurement.failure_rate.at_3pct` / `at_5pct` | 25 / 15 | 🔴 **BỊA** — EC không publish | 🔴 **BỊA** |
| `post_breakout_measurement.ultimate_high_method.average_days` | 5 | 🟡 GẦN — PDF: candle end to trend end median 6-9 ngày (harami thường). Nhưng "ultimate high" ≠ "trend end" (semantic khác) | 🟡 GẦN — PDF: 6-8 ngày (harami cross) |
| `post_breakout_measurement.ultimate_low_method.average_days` | 5 | 🟡 GẦN — tương tự ultimate high | 🟡 GẦN |
| `post_breakout_measurement.throwback_pullback.rate_pct` | 35 | 🔴 **KHÔNG CÓ** trong EC — EC không publish throwback rate cho harami | 🔴 **KHÔNG CÓ** |
| `post_breakout_measurement.average_move.bullish_continuation_pct` | 3 | 🟡 GẦN — PDF: +6.03% (bearish harami bull/Up) / +7.11% (bullish harami bull/Up). Trung bình ~6-7%, lớn hơn 3% | 🟡 GẦN — PDF: +5.73% (harami cross bear bull/Up) / +7.01% (harami cross bull bull/Up) |
| `post_breakout_measurement.lookahead_bars` | 10 | 🟢 GẦN KHỚP — PDF: candle end to trend end median 6-9 (harami thường) / 6-8 (harami cross). 10 bars bao phủ được trend end | 🟢 GẦN KHỚP |
| `performance_statistics.success_rate_following_breakout` | 65 | 🟡 KHÔNG TƯƠNG ỨNG — PDF dùng "% meeting price target" 58-74% (khác semantic: success rate ≠ target hit). Tuy nhiên con số 65 nằm trong range 58-74 → có vẻ hợp lý nhưng may mắn | 🟡 Tương tự — 65 trong range 66-74% |
| `performance_statistics.time_to_breakout_days` | 1 | 🔴 **LỆCH** — PDF: 3-4 ngày (median). Inside_day quá乐观 | 🔴 **LỆCH** — PDF: 3-4 ngày |
| `performance_statistics.failure_rate_3pct` / `failure_rate_5pct` | 25 / 15 | 🔴 **BỊA** — EC không publish | 🔴 **BỊA** |
| `performance_statistics.false_breakout_rate_pct` | 30 | 🔴 **KHÔNG CÓ** trong EC | 🔴 **KHÔNG CÓ** |
| `breakout_confirmation.volume_multiplier_ideal` | 1.3 | 🟡 KHÔNG XÁC ĐỊNH — PDF chỉ publish qualitative "heavy breakout volume suggests better performance" (Table 43.4/44.4/45.4/46.4), không có số 1.3x | 🟡 Tương tự |
| `prior_trend_requirements.direction` | "any" | 🟡 HỢP LÝ cho harami tổng quát, nhưng PDF phân rõ: bearish harami cần UP trend, bullish harami cần DOWN trend. "any" sẽ trộn 2 chiều | 🟡 Tương tự |

### Tóm tắt đối chiếu
- **Inside_day và harami THƯỜNG: metric đo hoàn toàn khác** (range vs body) → **KHÔNG thể tái dụng detection logic** của inside_day cho harami thường. Cần detector `harami_body` mới.
- **Inside_day và harami CROSS: cùng RANGE-based**, nhưng (1) inside_day **strict** (không cho bằng), harami cross **cho phép** tops hoặc bottoms bằng; (2) harami cross cần thêm điều kiện "ngày 2 là doji" (body_pct < threshold nhỏ). Có thể tái dụng ~70% logic detection của inside_day + thêm 2 rule.
- **Performance metrics (failure rate, target calc, throwback) của inside_day**: phần lớn **BỊA hoặc SAI semantic** so với PDF harami — vì EC không publish những metric này cho candlestick. Cần re-digitize riêng cho harami từ 4 chương này.

---

## Bằng chứng verbatim (quote ≤3 dòng mỗi chỗ, kèm trang)

### Ch.43 Harami Bearish
> *"The small black candle on the second day must have a body that ﬁts inside the body of the white candle."* — EC Table 43.1, in p375 / PDF p399

> *"Either the tops or the bottoms of the bodies can be equal but not both."* — EC Table 43.1, in p375 / PDF p399

> *"with a reversal rate of just 47% (bull market), that's too close to random to be useful."* — EC in p374 / PDF p398

### Ch.44 Harami Bullish
> *"Second day: A small-bodied white candle. The body must be within the prior candle's body."* — EC Table 44.1, in p384 / PDF p408

> *"acts as a reversal just three percentage points above random."* — EC in p383 / PDF p407

> *"Compute the height of the candle pattern and multiply it by the appropriate percentage shown in the table; then apply it to the breakout price."* — EC in p386 / PDF p410

### Ch.45 Harami Cross Bearish
> *"Second day: A doji (open and close are equal or nearly so) with a trading range inside the price range of the prior day."* — EC Table 45.1, in p393 / PDF p417

> *"acts as a continuation pattern most often, despite claims that it's a reversal."* — EC in p392 / PDF p416

> *"The high and low of the doji must be within the high-low range of the prior white candle."* — EC in p392 / PDF p416

### Ch.46 Harami Cross Bullish
> *"Second day: A doji...with a high-low price range that fits inside the range of the black candle."* — EC Table 46.1, in p401 / PDF p425

> *"acts as a bearish continuation pattern. This candle pattern performs less well (overall rank of 50) than the regular bullish harami (overall rank of 38)."* — EC in p400 / PDF p424

> *"the harami cross is indeed rare in a bear market."* — EC in p403 / PDF p427

---

## Reproducer (lệnh pdftotext chính xác)

```bash
# Workspace root: /Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet/Nghiên cứu mô hình nến
PDF="references/Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf"

# Offset check: PDF p398 = in p374 (mở Ch.43)
pdftotext -layout -f 398 -l 398 "$PDF" - | head -10   # → "43 / Harami, Bearish" + footer "374"

# Ch.43 Harami, Bearish — PDF p398-406 (in p374-382)
pdftotext -layout -f 398 -l 406 "$PDF" -

# Ch.44 Harami, Bullish — PDF p407-415 (in p383-391)
pdftotext -layout -f 407 -l 415 "$PDF" -

# Ch.45 Harami Cross, Bearish — PDF p416-423 (in p392-399)
pdftotext -layout -f 416 -l 423 "$PDF" -

# Ch.46 Harami Cross, Bullish — PDF p424-432 (in p400-408)
pdftotext -layout -f 424 -l 432 "$PDF" -
```

---

## Khuyến nghị cho detector harami (KHÔNG thực hiện trong task này — chỉ đề xuất)

### 1. Tách 2 detector riêng
- **`harami_body`** (cho Ch.43/44 Harami thường): đo BODY-based. Logic:
  - `body1_high = max(open1, close1)`, `body1_low = min(open1, close1)`
  - `body2_high = max(open2, close2)`, `body2_low = min(open2, close2)`
  - Rule: `body2_high <= body1_high AND body2_low >= body1_low` (cho phép ≤/≥ vì tops hoặc bottoms có thể bằng)
  - Rule loại trừ: `NOT(body2_high == body1_high AND body2_low == body1_low)` (cả hai bằng → không phải harami)
  - Body color: day1 ≠ day2 (white/black cho bearish; black/white cho bullish) — Bulkowski yêu cầu strict
  - Day 2 **KHÔNG** là doji: `body2_pct = (body2_high - body2_low) / close2 > doji_threshold` (ngưỡng doji ~5-10% của giá, hoặc < 0.1% nếu muốn chặt)

- **`harami_cross`** (cho Ch.45/46 Harami Cross): đo RANGE-based + doji. Logic:
  - `day2_high < day1_high AND day2_low > day1_low` (range-based, tương tự inside_day nhưng **cho phép equal một đầu**)
  - `body2_pct` rất nhỏ (doji): `body2_pct < doji_threshold` (ngưỡng ~5-10% hoặc nhỏ hơn)
  - Day 1 là tall candle: `body1_pct > tall_threshold` (PDF: median height 2.75-4.33% of breakout price → ngưỡng ~2-3%)

### 2. Measure rule
Dùng công thức EC thống nhất cho cả 4 chương:
```
target_up = breakout_price + (pattern_height × multiplier_pct_up)
target_down = breakout_price - (pattern_height × multiplier_pct_down)
```
trong đó `pattern_height = highest_high - lowest_low` của cả 2 nến, và `multiplier_pct` theo bảng:

| Detector | bull/Up | bear/Up | bull/Down | bear/Down |
|---|---|---|---|---|
| harami_body_bearish | 63% | 58% | 64% | 64% |
| harami_body_bullish | 69% | 66% | 59% | 61% |
| harami_cross_bearish | 69% | 67% | 68% | 66% |
| harami_cross_bullish | 74% | 73% | 68% | 70% |

→ Khi ghi registry: dùng multiplier trung bình hoặc chọn theo context (bull/bear market + breakout direction). Nếu chỉ 1 con số chung: **~65%** là giá trị trung bình hợp lý.

### 3. % target để ghi registry
- Harami thường: 58-69% meeting target (median ~63%)
- Harami cross: 66-74% meeting target (median ~69%)
- Harami cross **cao hơn** harami thường (~6 điểm) — ngược với kỳ vọng (doji đáng tin hơn?)

### 4. Lookahead (candle end to trend end median)
- Harami thường: 6-9 ngày (bearish 7-9, bullish 6-9)
- Harami cross: 6-8 ngày
- **Đề xuất `lookahead_bars = 10`** cho cả 4 (bao phủ trend end median + buffer) — khớp với inside_day digitized hiện tại.

### 5. Lưu ý quan trọng — reversal rate THẤP
Cả 4 chương đều gần random (47-57%):
- Bearish harami: continuation 53% (bull) / reversal 50% (bear)
- Bullish harami: reversal 53% (bull) / 51% (bear)
- Bearish harami cross: continuation 57% (bull) / 56% (bear) — **ngược lý thuyết** (bearish nhưng continuation)
- Bullish harami cross: continuation 55% (bull) / 56% (bear) — **ngược lý thuyết** (bullish nhưng continuation)

→ Detector nên **KHÔNG** ghi `expected_direction` cứng. Pattern này yếu về dự báo chiều — chủ yếu dùng làm **signal cảnh giác** (kết hợp với indicator khác). Bulkowski khuyến nghị: "trade only as reversal of retracement in primary trend" + dùng opening gap confirmation (performance tốt nhất theo Table 43.6/44.6/45.6/46.6).

### 6. Prior trend requirement
- Bearish harami / harami cross: trend **UP** leading to pattern
- Bullish harami / harami cross: trend **DOWN** leading to pattern
- Khác inside_day ("any") — cần rule riêng.
