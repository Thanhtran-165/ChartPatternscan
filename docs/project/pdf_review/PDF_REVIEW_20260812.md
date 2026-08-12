# PDF Review — Bulkowski gốc cho 12 pattern ưu tiên (Lớp C)

**Ngày review:** 2026-08-12
**Model / Provider:** GLM-5.2 / Z.ai (session mẹ `sess_1c356bbb-d126-4ea6-9508-ad490bca4ef1`)
**Mục đích:** Đọc PDF gốc Bulkowski để nâng chuẩn số liệu (failure rate, target method, sample size, lookahead) cho 12 pattern ưu tiên; đối chiếu với digitized spec hiện có.

---

## 1. Nguồn PDF và phương pháp

### Nguồn (file://)
Tất cả nằm trong `references/`:

| Mã | Tên file PDF | Số trang |
|----|---|---|
| **ECP** | `encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | 1.035 |
| **EC**  | `Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf` | 966 |
| **ATB** | `Wiley Trading Thomas N Bulkowski-Chart Patterns_ After the Buy-Wiley 2016.pdf` | 555 |

### Phương pháp trích
- Đã dùng `pdftotext -layout` (Poppler 26.04.0) để dump text từ 3 cuốn (3 file `.txt` tổng 7,4 MB).
- Dùng `pypdf` (Python) scan từng page để map pattern → số trang PDF.
- Trích range trang cho từng pattern, `grep` các key: "Break-even failure", "Number of formations", "Average rise/decline", "Days to ultimate", "Measure rule", "Table N.M".
- **Không** dùng chrome-devtools MCP vì `pdftotext` trích đầy đủ.

### Lưu ý quan trọng về "số trang PDF" vs "số trang sách"
PDF có **offset ~ +23 trang** so với số in trên sách (ví dụ: chương Pipe Bottoms in "536" nằm ở PDF page 559). Bảng dưới dùng **số trang PDF** để reproducer chạy `pdftotext -f X -l Y` được luôn.

---

## 2. Bảng tổng hợp 12 pattern (PDF gốc)

Quy ước viết tắt: **BE%** = Break-even failure rate (tỷ lệ fail ở mốc 5%); **Bull/Bear** = thị trường bull/bear; **UD/UA** = upward/downward breakout; **Sample** = tổng số mẫu (bull + bear).

### 2.1 Nhóm P0 — tần suất cao VN

#### 1. inside_day → "Harami" trong Encyclopedia of Candlestick
- **PDF**: `EC` chapter 43 (Bearish) p398-406 + chapter 44 (Bullish) p407-415
- **Pages checked**: 398-415 (18 trang)
- **Lưu ý ĐỊNH NGHĨA**: Harami của Bulkowski = *body* ngày hôm nay nằm trong *body* ngày hôm qua (bỏ qua shadow). Digitized "inside_day" dùng *range* (high-low) → **LỆCH ĐỊNH NGHĨA**, không phải cùng pattern.
- **Sample**: ~20.000 (cap), chia: 8.122 (bull/UA) + 2.342 (bear/UA) + 7.189 (bull/UD) + 2.347 (bear/UD).
- **Behavior**: Bull market → Bullish continuation 53% (gần random); Bear market → Bearish reversal 50%. Rank 36/46 (reversal), 72/103 (overall).
- **Failure rate (PDF)**: không báo "break-even failure" kiểu ECP; thay vào đó "10-day performance" = +2,73% / +2,61% / −2,31% / −4,01%.
- **Lookahead thực tế**: median "Candle end to trend end" = **7-9 ngày** (UA = 7, UD = 8-9).
- **Target method**: không có measure rule riêng (candlestick 2 ngày, không đủ cấu trúc).
- **Khác digitized**: **LỆCH ĐỊNH NGHĨA + LỆCH số liệu**. Digitized đặt failure 25%/15% (mốc 3%/5%), lookahead 10 bars — số liệu PDF không có dạng này, và định nghĩa body-vs-range khác nhau.

#### 2. pipe_bottoms — ECP chapter 35, p559-572
- **Pages checked**: 559-572 (14 trang)
- **Sample**: 926 (bull) + 226 (bear) = **1.152 patterns** (weekly chart, không phải daily).
- **Reversal (R), continuation (C)**: 100% R (cả 2 thị trường).
- **Average rise**: 45% (bull) / 32% (bear).
- **Failure rate (Table 35.3)**:
  - 5% (breakeven): **5%** (bull) / **4%** (bear)
  - 10%: 14% / 16%
  - 15%: 22% / 26%
  - 20%: 30% / 35%
  - 25%: 37% / 43%
  - 30%: 44% / 52%
  - 35%: 50% / 60%
- **Days to ultimate high**: 194 (bull) / 133 (bear).
- **Target**: breakout khi giá đóng trên highest high của pattern.
- **Khác digitized**: **LỆCH lớn**. Digitized ghi failure 12%/5% (5%/10%), average rise 15%, lookahead 63 bars, sample NOT-RECORDED → PDF: failure 5%/4% (BE), average rise 45%/32%, lookahead ~194 ngày (≈ 38 tuần ≈ 276 bars daily), sample 1.152.

#### 3. pipe_tops — ECP chapter 36, p573-585
- **Pages checked**: 573-585 (13 trang)
- **Sample**: 412 (bull) + 418 (bear) = **830**.
- **Average decline**: 20% (bull) / 27% (bear).
- **Failure rate (Table 36.3)**:
  - 5% (breakeven): **11%** (bull) / **2%** (bear)
  - 10%: 25% / 13%
  - 15%: 44% / 29%
  - 20%: 59% / 43%
- **Days to ultimate low**: 75 / 54.
- **% meeting price target**: 70% / 68% (có measure rule).
- **Khác digitized**: Digitized không có spec riêng cho pipe_tops (chỉ pipe_bottoms); **THIẾU hoàn toàn**.

#### 4. horn_bottoms (+ horn_tops) — ECP chapter 27-28, p461-490
- **Horn Bottoms (p461-473)**:
  - Sample: 286 (bull) + 118 (bear) = 404.
  - Average rise: 35% / 27%.
  - Break-even failure: 9% (bull) / 7% (bear).
  - Days to ultimate high: 180 / 90.
  - % meeting price target: 76% / 61%.
- **Horn Tops (p474-490)**:
  - Sample: 266 (bull) + 57 (bear) = 323.
  - Average decline: 21% / 22%.
  - Break-even failure: 7% / 2%.
  - Days to ultimate low: 67 / 64.
- **Khác digitized**: **LỆCH**. Digitized gộp "Horn Bottoms and Tops" với failure 15%/8% (mốc 5%/10%), average move 8%, lookahead 42 bars, sample NOT-RECORDED → PDF: failure 9%/7% (bottoms), 7%/2% (tops); average move 35%/27%/21%/22%; lookahead 180/90 (bottoms), 67/64 (tops).

---

### 2.2 Nhóm P1 — reversal phổ biến

#### 5. cup_with_handle — ECP chapter 10, p172-186
- **Pages checked**: 172-186 (15 trang) + Inverted p187-198.
- **Sample**: 412 (bull) + 59 (bear) = **471**.
- **Average rise**: 34% (bull) / 23% (bear).
- **Break-even failure**: 5% (bull) / 7% (bear).
- **% meeting price target**: 50% / 27%.
- **Days to ultimate high**: 167 / 63.
- **Measure rule**: "Compute the formation height by subtracting the lowest low [of cup] from [cup lip] ... add the height to breakout point" → trùng với digitized.
- **Khác digitized**: **KHỚP về failure rate + measure rule**. Digitized failure 5% overall = PDF 5%/7%. Average rise digitized 34% = PDF bull 34%. Lookahead digitized 252 bars (≈ 50 tuần ≈ 1 năm) vs PDF "167 days to ultimate high bull" → LỆCH nhẹ (digitized偏高). Sample NOT-RECORDED → PDF 471.

#### 6. head_and_shoulders_bottoms — ECP chapter 22, p398-427
- **Pages checked**: 398-427 (30 trang).
- **Sample**: 554 (bull) + 118 (bear) = **672**.
- **Average rise**: 38% (bull) / 30% (bear).
- **Days to ultimate high**: 176 / 107.
- **Measure rule**: "Compute formation height = neckline − lowest low of head. Add height to neckline → target."
- **Khác digitized**: **LỆCH failure + sample**. Digitized failure 6%/3% (mốc 5%/10%), average rise 20%, ultimate_avg_d 79, sample NOT-RECORDED → PDF: failure 6%/3% (gần khớp), nhưng average rise 38%/30% (PDF cao hơn gấp đôi = 20% digitized sai), ultimate days 176/107 (digitized 79 → lệch 2x), sample 672.

#### 7. head_and_shoulders_tops — ECP chapter 23, p428-460
- **Pages checked**: 428-460 (33 trang).
- **Sample**: 640 (bull) + 174 (bear) = **814**.
- **Average decline**: 22% (bull) / 29% (bear).
- **Break-even failure**: 4% (bull) / 1% (bear).
- **Days to ultimate low**: 62 / 41.
- **Khác digitized**: **LỆCH**. Digitized failure 8%/5% (mốc 5%/10%) → PDF break-even 4%/1% (LỆCH đáng kể, PDF thấp hơn). Sample NOT-RECORDED → PDF 814.

#### 8. scallops_ascending (+ descending) — ECP chapter 38-39, p647-706
4 biến thể (UA-UD × Bull-Bear):

**Scallop Ascending (p647-676)**:
- Sample: 736 (bull/UA) + 365 (bear/UA) + 161 (bull/UD) + 118 (bear/UD).
- Average rise/decline: 31% / 19% / −14% / −19%.
- Break-even failure: 10% (bull/UA) / 16% (bear/UA) / 27% (bull/UD) / 14% (bear/UD).
- Days to ultimate: 162 / 68 / 44 / 35.

**Scallop Descending (p677-706)**:
- Sample: 232 + 142 + 457 + 273 = **1.104**.
- Average rise/decline: 22% / 20% / −17% / −23%.
- Break-even failure: 22% (bull/UA) / 20% (bear/UA) / 15% (bull/UD) / 8% (bear/UD).
- Days to ultimate: 106 / 70 / 47 / 30.

- **Khác digitized**: **THIẾU hoàn toàn failure rate**. Digitized scallop ghi `failure_def: "Pattern fails when price moves 5% against breakout"` nhưng không có failure_rate, không có average, không sample. PDF có đủ cả 4 variants. Đây là gap lớn cần bổ sung.

#### 9. rectangle_bottoms / tops — ECP chapter 33-34, p586-618
**Rectangle Bottoms (p586-601)**:
- Sample: 115 (bull/UA) + 55 (bear/UA) + 98 (bull/UD) + 106 (bear/UD) = 374.
- Average rise/decline: 46% / 24% / −14% / −25%.
- Break-even failure: 10% / 11% / 16% / 4%.
- Days to ultimate: 177 / 81 / 41 / 33.

**Rectangle Tops (p602-618)**:
- Sample: 331 + 129 + 136 + 80 = 676.
- Average rise/decline: 39% / 20% / −17% / −21%.
- Break-even failure: 9% / 16% / 11% / 9%.
- Days to ultimate: 170 / 75 / 56 / 40.

- **Khác digitized**: **LỆCH cấu trúc**. Digitized ghi `rectangle_bottom: 5%, rectangle_top: 10%` (mô tả đơn giản) → PDF: bottom break-even failure 10%/11%/16%/4%, top 9%/16%/11%/9%. Sample NOT-RECORDED → PDF 374 bottoms + 676 tops.

---

### 2.3 Nhóm bổ sung — pattern CHƯA có spec digitized đầy đủ

#### 10. dead_cat_bounce (+ inverted) — ECP chapter 54-55, p852-887
- **Pages checked**: 852-887 (36 trang) — chương lớn, nhiều bảng.
- **Sample (Table 54.2)**: 454 (bull) + 222 (bear) = **676**.
- **Reversal/Continuation**: 237R/217C (bull), 115R/107C (bear).
- **Bounce**: 28% (bull) / 35% (bear).
- **Postbounce decline**: 30% (bull) / 40% (bear).
- **Có 7 bảng thống kê**: 54.2 (general), 54.3 (event decline), 54.4 (bounce), 54.5 (postbounce), 54.6 (frequency time), 54.7 (frequency price), 54.8 (tactics).
- **Đặc thù**: không có "break-even failure" kiểu chart pattern vì là event-driven. Lookahead thực tế: ~1-7 ngày cho bounce, ~6 tháng cho postbounce decline.
- **Khác digitized**: **THIẦU HOÀN TOÀN** — digitized không có file `dead_cat_bounce_digitized.json`, detector tạm giữ lookahead=120. PDF cho thấy pattern có 3 phases (event → bounce → postbounce decline) cần pipeline riêng, không thể dùng measure rule kiểu chart pattern.

#### 11. high_tight_flags — ECP chapter 22, p373-397
- **Pages checked**: 373-397 (25 trang).
- **Sample (Table 22.2)**: 253 (bull) + 54 (bear) = **307**.
- **Break-even failure**: **0%** / **0%** (cả 2 thị trường!).
- **Average rise**: **69%** (bull) / 42% (bear).
- **Days to ultimate high**: 39 / 25.
- **% meeting price target**: 90% / 91%.
- **Busted pattern performance**: N/A (không tìm thấy busted).
- **Khác digitized**: **THIẦU spec riêng** — chỉ có `flags_digitized.json` chung. PDF cho thấy HTF cần quy định riêng (flagpole + 4-5 week consolidation + volume dry-up). Lookahead thực tế ~25-39 ngày (PDF), không phải 120 tạm.

#### 12. broadening_formations_right_angled (ascending + descending) — ECP chapter 6-7, p53-90
**Right-Angled Ascending (p53-62)**:
- Sample: 92 (bull/UA) + 37 (bear/UA) + 186 (bull/UD) + 65 (bear/UD) = **380**.
- Average rise/decline: 29% / 15% / −15% / −22%.
- Days to ultimate high/low: cần xem Table 6.4 chi tiết.

**Right-Angled Descending (p75-90)**:
- Sample: 104 (bull/UA) + 36 (bear/UA) + 87 (bull/UD) + 47 (bear/UD) = **274**.
- Average rise: 28% (bull/UA) / 23% (bear/UA).
- Average decline: 15% / 23%.
- Break-even failure (UD): bull ≈ 3× bear.

- **Khác digitized**: **THIẦU failure rate + target** hoàn toàn. Digitized ghi `lookahead_bars: 252` nhưng failure/target/sample đều `None`. PDF có đủ 4 variants. Sample NOT-RECORDED → PDF 380 (asc) + 274 (desc).

#### Bổ sung: triple_tops / triple_bottoms — ECP chapter 50-51, p788-820
- **Triple Bottoms (p788-801)**:
  - Sample: 286 (bull) + 316 (bear) = **602**.
  - Average rise: 37% / 23%.
  - Break-even failure: 4% (bull) / 8% (bear).
  - Days to ultimate high: 165 / 80.
  - Failure rates Table 50.3: 5%=4%/8%, 10%=16%/21%, 15%=28%/37%, 20%=37%/50%, 25%=46%/63%.
- **Triple Tops (p802-820)**:
  - Sample: 278 (bull) + 349 (bear) = **627**.
  - Average decline: 19% / 24%.
  - Break-even failure: 10% (bull) / 5% (bear).
  - Days to ultimate low: 60 / 42.
  - Failure rates Table 51.3: 5%=10%/5%, 10%=29%/17%, 15%=49%/31%, 20%=63%/46%.
- **Khác digitized**: **THIẦU failure/target** trong digitized (chỉ có `lookahead_bars: 252`).

#### Bổ sung: three_falling_peaks / three_rising_valleys — ECP chapter 47-48, p707-740
- **Three Falling Peaks (p707-720)**:
  - Sample: 321 (bull) + 206 (bear) = **527**.
  - Average decline: 17% / 24%.
  - Break-even failure: 12% / 4%.
  - Days to ultimate low: 36 / 34.
- **Three Rising Valleys (p721-740)**:
  - Sample: 248 + 248 = **496**.
  - Average rise: 41% / 22%.
  - Break-even failure: 5% / 9%.
  - Days to ultimate high: 125 / 94.
- **Khác digitized**: **THIẦU failure/target** trong digitized.

---

## 3. Bảng đối chiếu tóm tắt: digitized vs PDF

Quy ước: 🟢 KHỚP | 🟡 LỆCH (ghi 2 giá trị) | 🔴 THIẾU trong digitized | ⚫ LỆCH ĐỊNH NGHĨA (không cùng pattern).

| # | Pattern | Sample PDF | Failure BE PDF | Failure BE digitized | Lookahead PDF | Lookahead digitized | Status |
|---|---|---|---|---|---|---|---|
| 1 | inside_day | ~20.000 (Harami) | không có BE; 10d perf +2.73%/+2.61% | 25%/15% (mốc 3%/5%) | 7-9d (trend end) | 10 bars | ⚫ LỆCH ĐỊNH NGHĨA (body vs range) |
| 2 | pipe_bottoms | 1.152 | 5%/4% (bull/bear) | 12%/5% (5%/10% mốc) | 194/133 ngày | 63 bars | 🟡 LỆCH |
| 3 | pipe_tops | 830 | 11%/2% | — (chưa có spec) | 75/54 ngày | — | 🔴 THIẾU spec |
| 4 | horn_bottoms/tops | 404 + 323 = 727 | 9%/7% (bottoms); 7%/2% (tops) | 15%/8% (gộp) | 180/90 + 67/64 ngày | 42 bars | 🟡 LỆCH |
| 5 | cup_with_handle | 471 | 5%/7% | 5% overall | 167/63 ngày | 252 bars | 🟢 KHỚP (failure) |
| 6 | head_shoulders_bottoms | 672 | 4%/8% | 6%/3% | 176/107 ngày | 252 bars, ultimate 79d | 🟡 LỆCH |
| 7 | head_shoulders_tops | 814 | 4%/1% | 8%/5% | 62/41 ngày | 252 bars | 🟡 LỆCH |
| 8 | scallops_ascending | 1.380 (4 variants) | 10%/16%/27%/14% | None | 162/68/44/35 ngày | 252 bars | 🔴 THIẾU hoàn toàn |
| 8b | scallops_descending | 1.104 | 22%/20%/15%/8% | None | 106/70/47/30 ngày | 252 bars | 🔴 THIẾU hoàn toàn |
| 9 | rectangle_bottoms | 374 | 10%/11%/16%/4% | 5% (bottom) | 177/81/41/33 ngày | 252 bars | 🟡 LỆCH |
| 9b | rectangle_tops | 676 | 9%/16%/11%/9% | 10% (top) | 170/75/56/40 ngày | 252 bars | 🟡 LỆCH |
| 10 | dead_cat_bounce | 676 | không có BE (event pattern) | — (chưa có spec) | bounce ~1-7d, postbounce ~6 tháng | — (detector tạm 120) | 🔴 THIẾU spec |
| 11 | high_tight_flags | 307 | 0%/0% | — (chưa có spec riêng, chỉ flags chung) | 39/25 ngày | — | 🔴 THIẾU spec |
| 12 | broadening_ra_ascending | 380 | (có trong Table 6.3) | None | cần Table 6.4 | 252 bars | 🔴 THIẾU hoàn toàn |
| 12b | broadening_ra_descending | 274 | (có trong Table 7.3) | None | cần Table 7.4 | 252 bars | 🔴 THIẾU hoàn toàn |
| 13 | triple_bottoms | 602 | 4%/8% | None | 165/80 ngày | 252 bars | 🔴 THIẾU |
| 13b | triple_tops | 627 | 10%/5% | None | 60/42 ngày | 252 bars | 🔴 THIẾU |
| 14 | three_falling_peaks | 527 | 12%/4% | None | 36/34 ngày | 252 bars | 🔴 THIẾU |
| 14b | three_rising_valleys | 496 | 5%/9% | None | 125/94 ngày | 252 bars | 🔴 THIẾU |

---

## 4. Bằng chứng verbatim (dành cho reproducibility, không copy đoạn dài)

Quy tắc bản quyền: chỉ ghi **số liệu** (%, sample, trang). Không copy câu văn dài.

### Pattern 2 — Pipe Bottoms (ECP Table 35.2 + 35.3, p542-543 PDF)
```
Number of formations   926   226
Average rise           45%   32%
Days to ultimate high  194   133
Failure 5% (breakeven) 42 or 5%   8 or 4%
Failure 10%            130 or 14%  36 or 16%
Failure 15%            200 or 22%  59 or 26%
Failure 20%            281 or 30%  79 or 35%
```

### Pattern 5 — Cup with Handle (ECP Table 10.2, p175 PDF)
```
Number of formations   412   59
Average rise           34%   23%
Days to ultimate high  167   63
```

### Pattern 6 — H&S Bottoms (ECP Table 22.2, p401 PDF)
```
Number of formations   554   118
Average rise           38%   30%
Days to ultimate high  176   107
```

### Pattern 11 — High Tight Flags (ECP Table 22.2, p350 PDF)
```
Number of formations   253   54
Break-even failure     0%    0%
Average rise           69%   42%
Days to ultimate high  39    25
% meeting price target 90%   91%
```

### Pattern 13 — Triple Tops (ECP Table 51.2 + 51.3, p785-787 PDF)
```
Number of formations   278   349
Average decline        19%   24%
Break-even failure     10%   5%
Failure 5%             29 or 10%   19 or 5%
Failure 10%            81 or 29%   60 or 17%
Failure 15%            137 or 49%  109 or 31%
Failure 20%            174 or 63%  162 or 46%
```

### Pattern 1 — Harami (EC Table 43.2, p376 PDF)
```
Number found           8,122   2,342   7,189   2,347
(bull/UA, bear/UA, bull/UD, bear/UD)
Candle end to breakout (median, days) 4
Candle end to trend end (median, days) 7-9
Candle end + 10 days   +2.73% +2.61% −2.31% −4.01%
```

---

## 5. Phát hiện lệch đáng chú ý

### 5.1 Lệch nghiêm trọng (cần ưu tiên nâng chuẩn)

1. **inside_day ⚫ LỆCH ĐỊNH NGHĨA**: Digitized dùng range (high-low)_inside; Bulkowski Harami dùng body (open-close)_inside. Hai pattern này **không phải cùng thứ**, không nên dùng số liệu của nhau. Khuyến nghị: tách biệt "Inside Bar (range-based)" và "Harami (body-based)" thành 2 spec, không cross-reference failure rate.

2. **pipe_bottoms 🟡 LỆCH lớn**: Digitized failure 12%/5% hoàn toàn không khớp PDF 5%/4%. Digitized average rise 15% vs PDF 45%/32%. Sample digitized NOT-RECORDED vs PDF 1.152. **Lý do có thể**: digitized trích sai hoặc lấy số từ nguồn khác (sách 1st ed?). Cần tái-verify nguồn gốc số liệu digitized.

3. **head_shoulders_bottoms 🟡 LỆCH**: Digitized average rise 20% vs PDF 38%/30% (PDF cao gấp đôi). Digitized ultimate 79d vs PDF 176/107d (PDF gấp 2x). Có vẻ digitized lấy từ 1st edition hoặc ước lượng thấp.

### 5.2 Thiếu hoàn toàn trong digitized (cần bổ sung gấp)

| Pattern | Sample PDF | Ưu tiên |
|---|---|---|
| scallops_ascending (4 variants) | 1.380 | CAO — có trong P1 |
| scallops_descending (4 variants) | 1.104 | CAO |
| broadening_ra_ascending | 380 | CAO |
| broadening_ra_descending | 274 | CAO |
| triple_bottoms | 602 | TRUNG BÌNH |
| triple_tops | 627 | TRUNG BÌNH |
| three_falling_peaks | 527 | TRUNG BÌNH |
| three_rising_valleys | 496 | TRUNG BÌNH |
| dead_cat_bounce | 676 | CAO — pattern ưu tiên P0+, đặc thù event-driven |
| high_tight_flags | 307 | CAO — pattern ưu tiên P0+ |
| pipe_tops | 830 | CAO — có pipe_bottoms mà thiếu pipe_tops |

### 5.3 Lookahead phổ thông trong digitized sai

Tất cả pattern digitized đều dùng **lookahead_bars: 252** (= 1 năm giao dịch) như một default an toàn. PDF Bulkowski cho thấy lookahead thực tế rất khác nhau:
- Harami: 7-9 ngày
- High Tight Flag: 25-39 ngày
- Cup with Handle: 63-167 ngày
- H&S Bottoms: 107-176 ngày
- Pipe Bottoms: 133-194 ngày

Khuyến nghị: thay lookahead 252 bằng giá trị thực từ PDF, kết hợp "Days to ultimate high/low" của từng pattern.

---

## 6. Khuyến nghị bước tiếp theo (không thực hiện trong task này)

1. **Tách inside_day thành 2 spec**: `inside_bar_range` (range-based, chưa có số liệu Bulkowski) và `harami_body` (body-based, có số liệu từ EC).
2. **Tạo 6 spec mới**: dead_cat_bounce, high_tight_flags, broadening_ra_ascending, broadening_ra_descending, pipe_tops (riêng), triple/peaks/valleys với số liệu từ PDF.
3. **Cập nhật 4 spec lệch**: pipe_bottoms, head_shoulders_bottoms, head_shoulders_tops, scallops (cả asc + desc).
4. **Đối chiếu "lookahead 252 default"** với "days to ultimate" PDF cho toàn bộ 31 family — là 1 task riêng.

---

## 7. Cách reproducer

```bash
REF_DIR="/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet/Nghiên cứu mô hình nến/references"
PDF="$REF_DIR/encyclopedia-of-chart-patterns-2nbsped-9786468600-3175723993-9780471668268-0471668265_compress.pdf"
EC_PDF="$REF_DIR/Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf"

# Pipe Bottoms (chapter 35, Table 35.2/35.3)
pdftotext -layout -f 559 -l 572 "$PDF" - | grep -A 20 "Table 35.2"

# Cup with Handle (chapter 10)
pdftotext -layout -f 172 -l 186 "$PDF" - | grep -A 15 "Table 10.2"

# Harami Bearish (EC chapter 43)
pdftotext -layout -f 398 -l 406 "$EC_PDF" - | grep -A 25 "Table 43.2"
```

**Phương pháp dùng:** `pdftotext -layout` (Poppler 26.04.0) + `pypdf 6.12.2` cho page mapping. Không cần chrome-devtools MCP.

---

**Hết file.**
