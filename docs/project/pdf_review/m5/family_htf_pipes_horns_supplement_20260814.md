# M5 supplement — Trích số liệu PDF gốc: HTF + HORNS + PIPES (5 family thiếu nguồn)

**Ngày trích:** 2026-08-14
**3 lớp bắt buộc:** Session mẹ `sess_0fb10a3f-28e7-42c6-8bdd-6aca4b147362` · Model `glm-5.2` · Provider `OpenCode Go`
**Nguồn PDF:** `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` (1.035 trang)
**Phương pháp:** `pdftotext -layout` (Poppler). Offset PDF ↔ sách in: **+23** (đã xác thực 5 điểm: header "352 Flags, High and Tight" ở PDF P0375; "440 Horn Bottoms" ở P0463; "452 Horn Tops" ở P0475; "538 Pipe Bottoms" ở P0561; "552 Pipe Tops" ở P0575 — tất cả đều +23).
**Lưu ý kỹ thuật dò trang:** đếm trang bằng awk đếm *dòng chứa \f* sẽ SAI (có 9 dòng chứa 2 \f → lệch 7 trang, cho ảo giác offset +16). Phải đếm theo *ký tự \f* (tổng 1.035) — bài học khi reproduce.
**Phạm vi:** chỉ trích số + ghi file này. KHÔNG sửa code, không rescan, không commit.

---

## Mục 1 — Bảng tổng hợp 5 family (7 mục mỗi family)

| Mục | 1. high_tight_flags | 2. horn_bottoms | 3. horn_tops | 4. pipe_bottoms | 5. pipe_tops |
|---|---|---|---|---|---|
| **Chương sách (số in / TOC)** | Ch. 22 "Flags, High and Tight" (sách p350) | **Ch. 28** "Horn Bottoms" (sách p438) — TOC ghi 28, KHÔNG phải 27 như danh mục giao việc | Ch. 29 "Horn Tops" (sách p451) | Ch. 35 "Pipe Bottoms" (sách p536) | Ch. 36 "Pipe Tops" (sách p550) |
| **Trang PDF đã đọc** | **373–398** (chương chiếm 373–386; phần sau là ch. 23 để xác định ranh giới) | **461–473** | **474–486** | **559–572** | **573–585** |
| **1. Measure rule** | target = **flag_low + (HTF_high − trend_start_low) / 2** — **CÓ CHIA /2**, cộng vào **HTF low price** (đáy flag), KHÔNG phải breakout | target = **highest_high + (highest_high − lowest_low)** — không chia; anchor = highest high (= breakout price) | target = **lowest_low − (highest_high − lowest_low)** — không chia; anchor = lowest low (= breakout price) | ⚠️ **CHƯƠNG KHÔNG MÔ TẢ CÔNG THỨC** (Table 35.9 không có dòng Measure rule); % số 83%/72% chỉ ở Results Snapshot | target = **lowest_low − (highest_high − lowest_low)** — không chia; sách ghi rõ breakout price = lowest low |
| **2. % meeting price target** (bull / bear) | **90% / 91%** (Results Snapshot, PDF p373) | **76% / 61%** (Results Snapshot p461; khẳng định lại ở measure rule p470) | **70% / 60%** (Results Snapshot p474 + measure rule p483) | **83% / 72%** (Results Snapshot p559, không kèm công thức) | **70% / 68%** (Results Snapshot p573 + measure rule p582) |
| **3. BE failure rate** (mốc 5%) | **0% / 0%** (Table 22.3, PDF p379) | **9% (26/286) / 7% (7/118)** (Table 28.3, p467) | **7% (19/266) / 2% (1/57)** (Table 29.3, p480) | **5% (42/926) / 4% (8/226)** (Table 35.3, p566) | **11% (44/412) / 2% (9/418)** (Table 36.3, p579) |
| **4. Days to ultimate high/low** | Trung bình **39 / 25** (Table 22.2, p378); median **~21d bull / ~14d bear** (Table 22.5, p380) | Trung bình **180 / 90** (Table 28.2, p466); median **>70d** cả 2 (Table 28.5, p468: bull cum ≤70d = 43%, bear = 69%) | Trung bình **67 / 64** (Table 29.2, p479); median bear ≈ **70d** (cum = 70%), bull **>70d** (cum = 65%) (Table 29.5, p481) | Trung bình **194 / 133** (Table 35.2, p565); median bull **>70d** (cum ≤70d = 45%), bear ≈ **70d+** (52%) (Table 35.5, p567) | Trung bình **75 / 54** (Table 36.2, p578); median bear ≈ **28d** (cum 51%), bull ≈ **49d** (cum 53%) (Table 36.5, p580) |
| **5. Sample size** | **253 bull + 54 bear = 307** (Table 22.2) | **286 + 118 = 404** (Table 28.2) | **266 + 57 = 323** (Table 29.2) | **926 + 226 = 1.152** (Table 35.2) | **412 + 418 = 830** (Table 36.2) |
| **5b. R/C + Timeframe** | 100% continuation (253C/54C); **DAILY** (toàn bộ đo bằng days, không mention weekly) | 100% reversal (286R/118R); **WEEKLY** | 100% reversal (266R/57R); **WEEKLY** | 100% reversal (926R/226R); **WEEKLY — XÁC NHẬN nghi ngờ đợt trước** | 100% reversal (412R/418R); **WEEKLY** |
| **6. Trang PDF then chốt** | Snapshot p373 · "half the move" p374 · T22.2 p378 · T22.3 p379 · T22.5 p380 · measure rule p382 | Snapshot p461 · T28.2 p466 · T28.3 p467 · T28.5 p468 · measure rule p470–471 | Snapshot p474 · T29.2 p479 · T29.3 p480 · T29.5 p481 · measure rule p483 | Snapshot p559 · daily-vs-weekly p559–560 · T35.2 p565 · T35.3 p566 · T35.5 p567 · T35.9 p570 | Snapshot p573 · T36.2 p578 · T36.3 p579 · T36.5 p580 · measure rule p582–583 |

### Quote verbatim measure rule (bản quyền — chỉ trích đoạn ngắn)

**HTF (ch. 22, PDF p382 / sách p359):**
> "To calculate the price target, find where the trend starts and measure the price change from the low at the start to the HTF high (the highest high in the pattern). **Divide the result in half and add it to the HTF's low price.** The result is the target that price reaches 90% of the time."

Table 22.8 (cùng trang): "Measure the rise leading to the flag and project half of it upward, using the flag low price."
Results Snapshot mở rộng (PDF p374): "The percentage meeting the price target, called the measure rule, works nearly all the time but is based on **half the move from the trend start to the flag, projected upward**."

**Horn Bottoms (ch. 28, PDF p470 / sách p447):**
> "Compute the formation height by subtracting the lowest low from the highest high. **Add the difference to the highest high** to get the target price. In a bull market, price exceeds the target 76% of the time, and in a bear market, the method works 61% of the time."

Ghi chú anchor: Table 28.4 (PDF p468) xác nhận "the breakout price, which is the highest high in the pattern" → anchor = breakout price.

**Horn Tops (ch. 29, PDF p483 / sách p460):**
> "Compute the horn height by subtracting the lowest low from the highest high in the pattern. **Subtract the result from the lowest low** to get the target price. In a bull market, price reaches the target 70% of the time. In a bear market, price hits or exceeds the target 60% of the time."

**Pipe Bottoms (ch. 35):** KHÔNG có đoạn measure rule nào trong chương. Table 35.9 (PDF p570) chỉ gồm 4 dòng: Downward trend / Buy / Stop loss / Wait for confirmation — không có "Measure rule". Xem mục 3 (bất thường).

**Pipe Tops (ch. 36, PDF p582 / sách p559):**
> "Compute the formation height by subtracting the lowest low from the highest high in the pipe. **Subtract the result from the breakout price (the lowest low)** to get a target. In a bull market, this method works 70% of the time and in a bear market, it works 68% of the time."

**Pipe Bottoms — weekly vs daily (PDF p559–560 / sách p537, quan trọng nhất cho scanner):**
> "I conducted an in-depth study of pipe bottoms on **daily** price charts and came up disappointed. The statistics show that **daily pipes have a failure rate of 18% with an average gain of 33%**. Almost half the formations (45%) have gains less than 20%. [...] so I **discarded the research and looked at the weekly chart**."

→ Toàn bộ số liệu pipe_bottoms công bố (BE 5%/4%, rise 45%/32%, %target 83%/72%) là **WEEKLY**. Nếu scanner chạy daily thì benchmark hợp lệ là BE 18% / gain 33% (số daily mà Bulkowski loại bỏ).

**HTF — daily:** chương 22 toàn bộ đo bằng days (median length 14 days, prior trend 47 days, days to ultimate high 39/25); không có mention weekly → HTF là **daily**.

---

## Mục 2 — Flags/Pennants anchor (VIỆC 2)

### 2a. Measure rule sách cho bull flag/pennant — chiều cao đo từ đâu, cộng vào giá nào

**Pennants, bull (ch. 34, PDF p556 / sách p533, Table 34.8 + prose):**
> "The trend starts at point A and climbs to the pennant. Take the difference between the **top (intraday high) of the pennant at its start (B at 10.69)** and the **trend start low (point A, at 7.50)** to get the trend height of 3.19. Add this value to the **intraday low at the pennant end (point C at 11.44, the day before the breakout)** to get the price target (14.63)."

Table 34.8: "Calculate the price difference between the start of the trend and the pennant. Prices should move at slightly less than this amount above (for uptrends) or below (for downtrends) **the end of the pennant**."

**Flags (ch. 21, PDF p370 / sách p347, Table 21.8 + prose — ví dụ sách là BEAR flag nên đối xứng cho bull):**
> "First, determine where the trend begins, which is usually the minor high (for downtrends) or low (for uptrends) preceding the formation. [...] Subtract the low at the formation start (point B at 42.75) from point A (47.50), giving a difference of 4.75. Subtract the difference from the high at the formation end (point C at 43) to give the target price of 38.25."

Table 21.8: "Calculate the price difference between the start of the trend and the formation. Prices should move at least this amount above (for uptrends) or below (for downtrends) **the end of the formation**."

**Trả lời câu a (tổng hợp bull case):**
- **Chiều cao trend** đo từ **trend start** (bull: minor low ngay trước formation — đáy trend) tới **thanh ĐẦU của consolidation** (bull pennant: intraday high của thanh bắt đầu pennant — đỉnh cột cờ/tam giác; bull flag: high at formation start). Tức là chiều cao **pole đầy đủ** (đáy trend → đỉnh cột cờ). KHÔNG đo tới breakout, và KHÔNG phải "đỉnh cao nhất của pattern" nói chung (mốc kết thúc là thanh đầu formation).
- **Chiều cao này cộng vào giá ở CUỐI formation**: bull = **intraday low tại pennant/formation end** (ngày ngay trước breakout — ví dụ C = 11.44, "the day before the breakout"); bear = high at formation end (ví dụ C = 43). **KHÔNG phải breakout price**, càng không phải highest high. (Bear flag ví dụ: target = 43 − 4.75 = 38.25.)

### 2b. Code hiện tại đúng hay sai công thức sách?

Code đọc (chỉ đọc, không sửa):
- `scanner/v2/flags_experiment.py:295-296`:
  ```python
  pole_height_abs = abs(anchor_price - float(pole["pole_price"]))
  target_price = float(breakout_price) + pole_height_abs if direction == "up" else float(breakout_price) - pole_height_abs
  ```
- `scanner/v2/pennants.py:229-230`: hai dòng **giống hệt** (cùng công thức).

Biến liên quan (flags_experiment.py, pennants.py tương tự):
- `anchor_price` (flags_experiment.py:237 / pennants.py:170): bull = `upper_points[0][1]` = HIGH của pivot **đầu tiên** của consolidation = **đỉnh thanh đầu formation (đỉnh cột cờ)**.
- `pole["pole_price"]` (hàm `_prior_pole`, flags_experiment.py:168-199): bull = **min low** trong cửa sổ `pole_lookback_bars` trước formation = **đáy trend**.
- → `pole_height_abs` = đỉnh thanh đầu formation − đáy trend = **chiều cao pole**.
- `breakout_price` (hàm `_breakout`, flags_experiment.py:201-219): là **close** vượt trendline boundary (không phải giá tại cuối formation).

**Kết luận: GẦN ĐÚNG — khác anchor.**
- **Chiều cao: ĐÚNG sách.** Code dùng pole_height (trend-start low → đỉnh thanh đầu formation) — chính là "trend height" Bulkowski mô tả (ví dụ Pennants: 10.69 − 7.50 = 3.19). Không chia hệ số — sách flags/pennants cũng không chia (chỉ HTF chia /2).
- **Anchor: KHÁC sách.** Sách cộng/trừ vào **giá tại cuối formation** (bull: intraday low ngày trước breakout; bear: high at formation end). Code cộng/trừ vào **breakout close**.
- **Ảnh hưởng định tính:** bull — breakout close > low at formation end → **target code CAO HƠN sách** (bảo thủ hơn; % đạt target thực sẽ thấp hơn benchmark sách 50–63%). Bear — breakout close < high at formation end → **target code THẤP HƠN sách** (lạc quan hơn; % đạt target thực sẽ cao hơn benchmark sách). Độ lệch bằng khoảng cách từ cực trị cuối formation tới breakout close (thường nhỏ, cỡ vài % giá — bằng quãng breakout phải vượt qua thân pattern).
- Chi tiết phụ: sách dừng trend start ở "nearest minor low/high preceding the formation" (định nghĩa Glossary); code dừng ở min/max trong `pole_lookback_bars` — nếu trend start thật nằm ngoài lookback window thì height lệch. Cùng khái niệm, khác biên xác định.

**Liên quan HTF (main agent hỏi "/2"):** code `scanner/v2/high_tight_flags.py:184-185` đang là:
```python
pole_height_abs = peak_price - pole_price
target_price = float(breakout_price) + pole_height_abs
```
→ KHÔNG chia /2 và anchor là breakout close. Sách (đã trích PDF p382): **flag_low + height/2**. Comment trong code (`high_tight_flags.py:338`: "The source target is half the prior move; full-pole target rows are retained as stress diagnostics") là **ĐÚNG** về sách — nhưng `target_price` chính vẫn là full-pole + breakout anchor, lệch sách 2 lớp cùng chiều (target code CAO HƠN sách đáng kể).

---

## Mục 3 — Bất thường / mâu thuẫn khi đọc

1. **Danh mục giao việc ghi sai số chương Horns:** prompt ghi horn_bottoms = chương 27 "Horns, Horn Bottoms" / horn_tops = chương 28. TOC PDF (P0019) ghi thực tế: **ch. 28 Horn Bottoms (sách p438), ch. 29 Horn Tops (sách p451)**. Đã trích theo số chương thật.
2. **Pipe Bottoms là family DUY NHẤT trong 5 family không có công thức measure rule trong chương** — Results Snapshot công bố "% meeting price target 83%/72%" nhưng Trading Tactics (Table 35.9) không có dòng "Measure rule" và toàn chương không mô tả cách tính target. Nếu spec tiếng Việt cần công thức, chỉ có thể suy diễn từ Pipe Tops (đối xứng: height cộng vào highest high) hoặc từ Horn Bottoms (mục "See also") — nhưng **sách không xác nhận verbatim**. Cần ghi rõ nguồn là suy diễn nếu dùng.
3. **Horns + Pipes đều WEEKLY:** mọi con số performance (BE, rise, %target, days) chỉ hợp lệ trên weekly chart. Riêng pipe_bottoms có số đối chứng daily rõ ràng (BE 18%, gain 33%) mà Bulkowski **từ chối công bố** vì kém — scanner daily không được dùng BE 5%/4% làm benchmark.
4. **Bẫy dò trang:** awk đếm dòng chứa \f cho offset ảo +16 (do 9 dòng chứa 2 ký tự \f). Offset đúng +23 phải xác thực qua header số trang sách in. Đã ghi reproducer dưới.
5. **HTF thời gian ngắn nhưng books ghi theo ultimate (20% reversal):** HTF dùng đúng ultimate high method như đa số pattern (khác flags/pennants đo trend-end như m5 đợt trước phát hiện) → days 39/25 SO ĐƯỢC với các pattern khác (trừ flags/pennants).
6. **Đối chiếu chéo với đợt 12/08 (PDF_REVIEW_20260812.md) và m5 flags/pennants:** số HTF trích lần này (307 sample, BE 0%/0%, rise 69%/42%, %target 90%/91%) khớp hoàn toàn với ghi chép đợt 12/08 — lần trích độc lập thứ 2 cho cùng kết quả.
7. **Break-even 0% của HTF có điều kiện:** Bulkowski ghi nhận (PDF p378-379) đổi tiêu chí breakout từ close trên pattern high (ed.1) sang close trên **trend-line boundary** làm BE giảm mạnh; bear sample chỉ 54 nên BE 2% mốc 10% có thể nhảy (17% mốc 25%).

## Reproducer

```bash
PDF="references/encyclopedia-of-chart-patterns-2nbsped-9786468600-3175723993-9780471668268-0471668265_compress.pdf"
# Offset +23 (sách in + 23 = PDF). Xác thực: grep header "440" trong p463.
pdftotext -layout -f 373 -l 398 "$PDF" -   # ch.22 HTF (Snapshot p373, T22.2 p378, T22.3 p379, T22.5 p380, measure p382)
pdftotext -layout -f 461 -l 486 "$PDF" -   # ch.28 Horn Bottoms (p461-472) + ch.29 Horn Tops (p474-486)
pdftotext -layout -f 559 -l 585 "$PDF" -   # ch.35 Pipe Bottoms (p559-571) + ch.36 Pipe Tops (p573-585)
pdftotext -layout -f 362 -l 375 "$PDF" -   # ch.21 Flags — measure rule Table 21.8 ở p370-371
pdftotext -layout -f 545 -l 564 "$PDF" -   # ch.34 Pennants — measure rule Table 34.8 ở p556
# Đánh số trang chính xác (đếm KÝ TỰ \f, không đếm dòng):
pdftotext -layout "$PDF" /tmp/ecp_full.txt
perl -pe 's/\f/\n<<<PAGEBREAK>>>\n/g' /tmp/ecp_full.txt | perl -ne 'BEGIN{$p=1} if (/<<<PAGEBREAK>>>/){$p++; next} print sprintf("P%04d|",$p), $_'
```

---

**Hết file.**
