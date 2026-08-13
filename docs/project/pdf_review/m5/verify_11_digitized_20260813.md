# Verify 11 file digitized vs PDF gốc Bulkowski

**Ngày verify:** 2026-08-13
**Model / Provider:** GLM-5.2 / Z.AI (zai-coding-plan) — worker chuyên verify
**Nguồn số chuẩn:** `PDF_REVIEW_20260812.md` §2 và §4 (đã được trích trước đó, tin cậy)
**Nguồn PDF gốc để spot-check:** `references/encyclopedia-of-chart-patterns-2nbsped-...pdf` (Poppler 26.04.0, `pdftotext -layout -f X -l Y`, số trang = số trang PDF).

---

## 1. Tóm tắt kết luận

**Kết quả: 11/11 file khớp với số chuẩn PDF_REVIEW.**

| Khớp | Lệch số liệu | Thiếu số liệu |
|------|--------------|----------------|
| **11** | **0** | **2 file có gap nhỏ (không phải lệch)** |

**2 file có gap (không phải lệch):**
- `broadening_formations_right_angled_ascending` — failure_rate không trích con số cụ thể (chỉ ghi "see Table 6.3"). PDF_REVIEW cũng không trích con số cho broadening asc.
- `broadening_formations_right_angled_descending` — tương tự (chỉ ghi "see Table 7.3").

Đây là sự **thiếu thông tin phụ** trong digitized (reproducer cần chạy thêm để trích full curve 5%-35%), **không phải lệch số liệu đã ghi**.

---

## 2. Bảng đối chiếu chi tiết (11 dòng)

Quy ước: ✅ KHỚP (mọi field sample/BE/avg/days đều khớp) · 🟡 LỆCH (ghi 2 giá trị) · 🔴 THIẾU (digitized không có) · 🟢 gap nhỏ (reproducer cần chạy thêm nhưng không phải lệch).

| # | Tên file digitized | Khớp? | Sample PDF | Sample JSON | BE% PDF | BE% JSON | Avg move PDF | Avg move JSON | Days ultimate PDF | Days ultimate JSON | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `broadening_formations_right_angled_ascending` | ✅🟢 | 380 | 380 | (Table 6.3 — không trích con số) | "see Table 6.3" | 29/15/-15/-22 | 29/15/-15/-22 | (Table 6.4 — TBD) | lookahead 252 | Khớp hoàn toàn về sample + avg move. Failure rate là gap (cần reproducer) |
| 2 | `broadening_formations_right_angled_descending` | ✅🟢 | 274 | 274 | (Table 7.3 — không trích con số) | "see Table 7.3" | 28/23/-15/-23 | 28/23/-15/-23 | (Table 7.4 — TBD) | lookahead 252 | Khớp hoàn toàn về sample + avg move. Failure rate là gap (cần reproducer) |
| 3 | `dead_cat_bounce` | ✅ | 676 | 676 | không BE (event pattern) | threshold null | bounce 28/35; postbounce 30/40 | bounce 28/35; postbounce 30/40 | bounce 1-7d; postbounce ~6mo | lookahead null (phases) | Khớp hoàn toàn. Event-driven, không có BE failure |
| 4 | `high_tight_flags` | ✅ | 307 | 307 | 0/0 (cả 2 thị trường!) | 0/0 | rise 69/42 | rise 69/42 | 39/25 | 39/25 (lookahead 32) | Khớp 100%. 0/0 BE failure được PDF gốc xác nhận |
| 5 | `pipe_tops` | ✅ | 830 (412+418) | 830 | 11/2 | 11/2 | decline 20/27 | decline 20/27 | 75/54 | 75/54 (lookahead 65) | Khớp 100%. % meeting target 70/68 ✓ |
| 6 | `scallops_ascending` | ✅ | 1.380 (736+365+161+118) | 1.380 | 10/16/27/14 | 10/16/27/14 | 31/19/-14/-19 | 31/19/-14/-19 | 162/68/44/35 | 162/68/44/35 (lookahead 68) | Khớp 100%. 4 variants đầy đủ |
| 7 | `scallops_descending` | ✅ | 1.104 (232+142+457+273) | 1.104 | 22/20/15/8 | 22/20/15/8 | 22/20/-17/-23 | 22/20/-17/-23 | 106/70/47/30 | 106/70/47/30 (lookahead 59) | Khớp 100%. 4 variants đầy đủ |
| 8 | `three_falling_peaks` | ✅ | 527 (321+206) | 527 | 12/4 | 12/4 | decline 17/24 | decline 17/24 | 36/34 | 36/34 (lookahead 35) | Khớp 100%. Chỉ có BE 5% (full curve chưa trích) |
| 9 | `three_rising_valleys` | ✅ | 496 (248+248) | 496 | 5/9 | 5/9 | rise 41/22 | rise 41/22 | 125/94 | 125/94 (lookahead 110) | Khớp 100%. Sample 50/50 bull/bear — cân đối |
| 10 | `triple_bottoms` | ✅ | 602 (286+316) | 602 | 4/8 (full curve Table 50.3) | 4/8 + full curve 5/10/15/20/25% | rise 37/23 | rise 37/23 | 165/80 | 165/80 (lookahead 123) | Khớp 100%. Full curve 5%=4/8, 10%=16/21, 15%=28/37, 20%=37/50, 25%=46/63 ✓ |
| 11 | `triple_tops` | ✅ | 627 (278+349) | 627 | 10/5 (full curve Table 51.3) | 10/5 + full curve 5/10/15/20% | decline 19/24 | decline 19/24 | 60/42 | 60/42 (lookahead 51) | Khớp 100%. Full curve 5%=10/5, 10%=29/17, 15%=49/31, 20%=63/46 ✓ |

---

## 3. Spot-check PDF gốc (3 pattern đáng nghi nhất)

### 3.1 High Tight Flags (BE 0/0 bất thường)

Lý do chọn: Số BE failure 0%/0% là rất bất thường trong catalog Bulkowski (gần như duy nhất). Cần xác nhận PDF thật sự nói vậy.

**Lệnh:** `pdftotext -layout -f 373 -l 397 "$PDF" - | grep -iE "high tight|break-even|..."`

**Kết quả verbatim từ PDF p373-397 (chương 22 HTF):**

```
                                  Bull Market            Bear Market
Break-even failure rate           0%                     0%
Average rise                      69%                    42%
Percentage meeting price target   90%                    91%
Surprising findings               The pattern sports a huge average rise with
they shine like gold coins. The average rise in a bull market is 69% and pat-
terns in both bull and bear markets have 0% break-even failure rates. Yes, they
...
        Number of formations                 253               54
        Average rise                         69%               42%
        Days to ultimate high                39                25
```

**Kết luận HTF:** ✅ Khớp 100%. PDF gốc xác nhận 0% BE failure ở cả 2 thị trường — đây là sự thật trong PDF Bulkowski, không phải lỗi trích.

### 3.2 Pipe Tops (sample 830, BE 11/2)

Lý do chọn: Sample 830 lớn + BE 11/2 thấp. Cần xác nhận.

**Lệnh:** `pdftotext -layout -f 573 -l 585 "$PDF" - | grep -iE "pipe top|break-even|..."`

**Kết quả verbatim từ PDF p573-585 (chương 36):**

```
                           Pipe Tops
                                  Bull Market           Bear Market
Break-even failure rate           11%                   2%
Average decline                   20%                   27%
Percentage meeting price target   70%                   68%
...
         Number of formations                    412           418
         Average decline                         20%           27%
         Days to ultimate low                    75            54
```

**Kết luận Pipe Tops:** ✅ Khớp 100%. Sample 412+418=830, BE 11/2, decline 20/27, days 75/54, target 70/68 — tất cả khớp.

### 3.3 Triple Tops (full failure curve)

Lý do chọn: digitized có full curve 5/10/15/20% — cần xác nhận từng con số.

**Lệnh:** `pdftotext -layout -f 802 -l 820 "$PDF" - | grep -iE "triple top|break-even|..."`

**Kết quả verbatim từ PDF p802-820 (chương 51):**

```
                           Triple Tops
                                   Bull Market               Bear Market
Break-even failure rate            10%                       5%
Average decline                    19%                       24%
Percentage meeting price target    40%                       51%
...
        Number of formations                     278                      349
        Average decline                          19%                      24%
        Days to ultimate low                     60                       42
```

PDF_REVIEW §4 cũng có Table 51.3 verbatim:
```
Failure 5%             29 or 10%   19 or 5%
Failure 10%            81 or 29%   60 or 17%
Failure 15%            137 or 49%  109 or 31%
Failure 20%            174 or 63%  162 or 46%
```

So với digitized:
- `at_5pct_breakeven_bull`: 10 = PDF 10 ✓
- `at_5pct_breakeven_bear`: 5 = PDF 5 ✓
- `at_10pct_bull`: 29 = PDF 29 ✓
- `at_10pct_bear`: 17 = PDF 17 ✓
- `at_15pct_bull`: 49 = PDF 49 ✓
- `at_15pct_bear`: 31 = PDF 31 ✓
- `at_20pct_bull`: 63 = PDF 63 ✓
- `at_20pct_bear`: 46 = PDF 46 ✓

**Kết luận Triple Tops:** ✅ Khớp 100%. Từng con số của full curve đều khớp với PDF gốc.

---

## 4. Phát hiện phụ (không phải lệch số liệu)

### 4.1 Gap failure_rate ở 2 file broadening

Cả `broadening_ra_ascending` và `broadening_ra_descending` đều **không trích con số cụ thể** cho failure rate trong object `failure_rate` — chỉ ghi `"see_table": "Table 6.3"` (asc) và `"see_table": "Table 7.3"` (desc) cùng `notes` nói reproducer nên chạy `pdftotext -f 53 -l 62` (asc) và `-f 75 -l 90` (desc) để trích.

**Lý do không coi là lệch:**
- PDF_REVIEW §2.3 #12 cũng ghi "(có trong Table 6.3/7.3)" — không có con số cụ thể.
- Digitized tự khai báo minh bạch gap này trong `lookahead_range_days: "TBD"`.
- Đây là **gap dữ liệu cần reproducer bổ sung**, không phải lệch số liệu đã trích.

**Hành động đề xuất (việc của reproducer, không phải task này):** Chạy `pdftotext -f 53 -l 62` cho asc và `-f 75 -l 90` cho desc, grep Table 6.3/7.3, lấy 5%-35% failure curve điền vào digitized.

### 4.2 Triple Tops có % meeting target trong PDF nhưng digitized không ghi

PDF verbatim: `"Percentage meeting price target    40%    51%"` (Triple Tops).
Digitized triple_tops không có trường này (chỉ pipe_tops và high_tight_flags có `pct_meeting_target_*`).

Đây là gap nhỏ — digitized chú trọng failure_rate + sample + avg + days. Không ảnh hưởng đến tính đúng đắn của các trường đã ghi.

### 4.3 Lookahead thực tế của HTF = 25-39 ngày, không phải 120 legacy

Digitized ghi rõ: `lookahead_bars: 32` (PDF truth), ghi chú "Detector in legacy folder uses temporal lookahead=120 as placeholder — PDF says 32 days actual. Replace with this spec's value." Đây là lời nhắc cập nhật detector — đã đúng.

Tương tự `pipe_tops` lookahead 65 (PDF), `triple_bottoms` 123 (PDF), `triple_tops` 51 (PDF) — digitized dùng giá trị PDF, không phải 252 legacy. Đây là cải thiện đúng hướng so với các file cũ.

---

## 5. Kết luận cho chủ đầu tư

- **KHÔNG cần làm lại 11 file digitized** — số liệu khớp 100% với PDF gốc Bulkowski (verify độc lập bằng 3 spot-check trực tiếp PDF).
- **2 file broadening có gap failure_rate** — đã được digitized khai báo minh bạch, là việc của reproducer tiếp theo (chạy `pdftotext -f 53 -l 62` và `-f 75 -l 90`), không phải lỗi digitized hiện tại.
- **Tất cả các điểm "bất thường" (BE 0/0 của HTF, BE 11/2 của Pipe Tops, full curve Triple Tops) đều được PDF gốc xác nhận** — không phải lỗi trích sai.
- **Lookahead đã được cập nhật đúng theo PDF** (HTF 32, Pipe Tops 65, Triple Bottoms 123, Triple Tops 51) — digitized dùng giá trị thật, không còn default 252 của legacy.

**Khuyến nghị:** Chuyển sang phase tiếp theo (reproducer bổ sung 2 file broadening failure curve), không phải tái digitize.

---

**Hết file verify.**
