# M5 — Trích số liệu PDF gốc: family GAPS

**Ngày trích:** 2026-08-13
**Model / Provider:** GLM-5.2 / builtin:zai-coding-plan (Z.AI Coding Plan)
**Session mẹ:** `sess_0fb10a3f-28e7-42c6-8bdd-6aca4b147362`
**Nguồn PDF:** `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` (ECP 2nd ed., 1.035 trang)
**Phương pháp:** `pdftotext -layout` (Poppler 26.04.0). Offset PDF ↔ sách in: **+23 trang** (số in + 23 = số PDF). Đã tự kiểm: chương 23 "Gaps" sách p362 → PDF p385 (khớp offset +23).

---

## ⚠️ Phát hiện quan trọng (đặc thù family GAPS)

Bulkowski xếp Gaps vào nhóm **event pattern** và đo performance **KHÁC HOÀN TOÀN** với chart patterns:

> *"Performance rank: **Not applicable**"* — ECP p362 (Results Snapshot Gaps, PDF p385)

Hệ quả (xác minh trực tiếp từ PDF):

1. **KHÔNG có "Break-even failure rate"** cho gaps. Digitized ghi `breakaway_at_5pct: 15, continuation_at_5pct: 20, exhaustion_at_5pct: 10` → **KHÔNG có trong PDF** (bịa hoặc lấy nguồn khác).
2. **KHÔNG có "Average rise/decline"** tổng hợp. Chỉ có "Percentage rise/decline for each 12-month lookback period" theo L/C/H (trong Table 23.3 breakaway) — đây là performance theo vị trí năm, không phải average move.
3. **KHÔNG có "Days to ultimate high/low"**. Chỉ có "Average time to close the gap" (số ngày để giá quay lại fill gap).
4. **KHÔNG có "% meeting price target"** và **KHÔNG có "Measure rule"** cho gaps.
5. Metric duy nhất Bulkowski đo cho gaps là **"Close Within a Week"** (tỷ lệ gap đóng lại trong 1 tuần) + thời gian đóng gap.

→ **Không thể so sánh performance của gaps với chart patterns khác** (Bulkowski tự xếp "Performance rank: Not applicable").

---

## Cấu trúc chương Gaps (ECP chapter 23, PDF p385-396 / sách p362-373)

Bulkowski gộp **4 subtype gap vào 1 chương duy nhất** (chapter 23), mỗi subtype có 1 bảng statistics riêng:

| Subtype | Bảng PDF | Trang PDF | Trang sách |
|---|---|---|---|
| Area gaps | Table 23.2 | p389 | p366 |
| Breakaway gaps | Table 23.3 | p391 | p368 |
| Continuation gaps | Table 23.4 | p392 | p369 |
| Exhaustion gaps | Table 23.5 | p393 | p370 |

**Note:** ECP có 5 loại gap (area/common, breakaway, continuation/measured/runaway, exhaustion, ex-dividend) nhưng Bulkowski chỉ đo 4 loại (ex-dividend bị loại vì "rarely happens and has no investment significance").

---

## Bảng số chính per subtype

### AREA GAPS (Table 23.2, PDF p389 / sách p366)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | |
| **pages_checked** | PDF **389** (sách p366) | |
| **Number of formations** | **127** (bull/Up) + **154** (bear/Up) + **127** (bull/Down) + **76** (bear/Down) = **484 area gaps** | Bulkowski: "484 area gaps in 97 stocks from mid-1991 to early 2004" |
| **Average time to close the gap** | **3 days** (cả 4 tổ hợp) | Nhanh nhất trong 4 subtype |
| **Closed in 1 week** | **89% / 93% / 92% / 89%** | Thứ tự: bull/Up · bear/Up · bull/Down · bear/Down |
| **Closed in 2 weeks** | 99% / 100% / 100% / 100% | |
| **Closed in 3 weeks** | 100% (all) | Hầu như tất cả đóng trong 3 tuần |
| **Average gap size** | $0.31 / $0.55 / $0.28 / $0.59 | Bear market gap lớn hơn bull |
| **Break-even failure rate** | **KHÔNG TÌM THẤY** | Gaps không có failure rate table |
| **Average rise/decline** | **KHÔNG TÌM THẤY** | Chỉ có "Percentage closed" |
| **% meeting price target** | **KHÔNG TÌM THẤY** | Gaps không có measure rule |

### BREAKAWAY GAPS (Table 23.3, PDF p391 / sách p368)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | như trên | |
| **pages_checked** | PDF **391** (sách p368) | |
| **Number of formations** | **345** (bull/Up) + **92** (bear/Up) + **226** (bull/Down) + **74** (bear/Down) = **737 breakaway gaps** | Bulkowski: "737 breakaway gaps" in 132 stocks |
| **Average time to close the gap** | **136 days** (bull/Up) · **61 days** (bear/Up) · **168 days** (bull/Down) · **111 days** (bear/Down) | Chậm nhất trong 4 subtype — breakaway rarely closes |
| **Closed in 1 week** | **2% / 9% / 1% / 1%** | Rất hiếm khi đóng nhanh |
| **Closed in 2 weeks** | 9% / 22% / 6% / 15% | |
| **Closed in 3 weeks** | 18% / 38% / 11% / 35% | |
| **Closed in 1 month** | 23% / 45% / 17% / 38% | |
| **Closed in 3 months** | 46% / 73% / 46% / 55% | |
| **Closed in 1 year** | 66% / 95% / 75% / 70% | ~1/3 breakaway gaps vẫn mở sau 1 năm |
| **Average gap size** | $0.43 / $0.85 / $0.68 / $1.38 | Bear/Down lớn nhất |
| **Percentage rise/decline by 12-month position** | bull/Up: L38% / C35% / H41% · bear/Up: L36% / C30% / H29%ᵃ · bull/Down: L19% / C18% / H20% · bear/Down: L29%ᵃ / C29% / H29%ᵃ | L=yearly low, C=center, H=yearly high; ᵃ = n<30 |
| **Break-even failure rate** | **KHÔNG TÌM THẤY** | |
| **% meeting price target** | **KHÔNG TÌM THẤY** | |

### CONTINUATION GAPS (Table 23.4, PDF p392 / sách p369)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | như trên | |
| **pages_checked** | PDF **392** (sách p369) | |
| **Number of formations** | **168** (bull/Up) + **83** (bear/Up) + **122** (bull/Down) + **122** (bear/Down) = **495 continuation gaps** | Bulkowski: "495 continuation gaps" in 173 stocks |
| **Average time to close the gap** | **98 days** (bull/Up) · **43 days** (bear/Up) · **77 days** (bull/Down) · **91 days** (bear/Down) | Giữa breakaway và area |
| **Closed in 1 week** | **4% / 20% / 9% / 13%** | |
| **Closed in 2 weeks** | 20% / 39% / 23% / 27% | |
| **Closed in 3 months** | 61% / 86% / 75% / 66% | |
| **Closed in 1 year** | 80% / 95% / 93% / 81% | |
| **Average gap size** | $0.47 / $0.86 / $0.48 / $1.24 | |
| **Position in time trend (trend start → gap)** | 50% / 55% / 69% / 72% | Continuation xuất hiện ở giữa trend |
| **Break-even failure rate** | **KHÔNG TÌM THẤY** | |

### EXHAUSTION GAPS (Table 23.5, PDF p393 / sách p370)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | như trên | |
| **pages_checked** | PDF **393** (sách p370) | |
| **Number of formations** | **120** (bull/Up) + **111** (bear/Up) + **129** (bull/Down) + **111** (bear/Down) = **471 exhaustion gaps** | Bulkowski: "471 exhaustion gaps" in 173 stocks |
| **Average time to close the gap** | **9 days** (bull/Up) · **7 days** (bear/Up) · **14 days** (bull/Down) · **10 days** (bear/Down) | Đóng nhanh (1-2 tuần) — tại end of trend |
| **Closed in 1 week** | **61% / 78% / 64% / 63%** | ~2/3 đóng trong 1 tuần |
| **Closed in 2 weeks** | 91% / 90% / 78% / 85% | |
| **Closed in 1 year** | 100% / 100% / 100% / 99% | Hầu hết đóng trong 1 năm |
| **Average gap size** | $0.49 / $0.79 / $0.63 / $0.94 | Bear/Down lớn nhất ($0.94) |
| **Break-even failure rate** | **KHÔNG TÌM THẤY** | |

### RESULTS SNAPSHOT tổng hợp (PDF p385-386 / sách p362-363)

| Metric | Upward Breakouts (bull/bear) | Downward Breakouts (bull/bear) |
|---|---|---|
| **Close Within a Week — Area gap** | 89% / 93% | 92% / 89% |
| **Close Within a Week — Breakaway** | 2% / 9% | 1% / 1% |
| **Close Within a Week — Continuation** | 4% / 20% | 9% / 13% |
| **Close Within a Week — Exhaustion** | 61% / 78% | 64% / 63% |

**Performance rank:** Not applicable (cho cả 4 subtype).

---

## Đối chiếu với `gaps_digitized.json`

| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `failure_rate.breakaway_at_5pct` | 15 | **KHÔNG TÌM THẤY** (gaps không có failure rate) | 🔴 **BỊA** — trường này không tồn tại trong PDF |
| `failure_rate.continuation_at_5pct` | 20 | **KHÔNG TÌM THẤY** | 🔴 **BỊA** |
| `failure_rate.exhaustion_at_5pct` | 10 | **KHÔNG TÌM THẤY** | 🔴 **BỊA** |
| `failure_rate.common_fills` | 80 | Area gap Closed in 1 week = **89-93%** | 🟡 LỆCH nhẹ (80 vs 89-93) |
| `failure_definition.threshold_pct` | "varies_by_type" | **KHÔNG TÌM THẤY** (no failure definition for gaps) | 🔴 **BỊA** — gaps không có failure definition |
| `post_breakout_measurement.ultimate_high_method.breakaway_average_days` | 42 | **KHÔNG TÌM THẤY** (gaps không có ultimate method) | 🔴 **BỊA** |
| `post_breakout_measurement.ultimate_high_method.continuation_average_days` | 21 | **KHÔNG TÌM THẤY** | 🔴 **BỊA** |
| `post_breakout_measurement.ultimate_high_method.exhaustion_average_days` | 5 | **KHÔNG TÌM THẤY** (PDF: exhaustion "average time to close" = 7-14 days, không phải ultimate) | 🔴 **BỊA + nhầm khái niệm** |
| `post_breakout_measurement.average_move.breakaway_rise_pct` | 15 | **KHÔNG TÌM THẤY** (PDF chỉ có % rise theo yearly position, không có average rise) | 🔴 **BỊA** |
| `post_breakout_measurement.average_move.continuation_rise_pct` | 8 | **KHÔNG TÌM THẤY** | 🔴 **BỊA** |
| `post_breakout_measurement.average_move.exhaustion_reversal_pct` | -5 | **KHÔNG TÌM THẤY** | 🔴 **BỊA** |
| `post_breakout_measurement.lookahead_bars` | 63 | KHÔNG có lookahead; "Average time to close" = 3-168 days tùy subtype | 🔴 LỆCH LỚN |
| `post_breakout_measurement.target_calculation.method` | varies_by_type | **KHÔNG CÓ measure rule cho gaps** | 🔴 **BỊA** |
| `performance_statistics.gap_fill_rates.common_gap_fill_rate` | 80 | Area gap Closed in 1 week = **89-93%** | 🟡 LỆCH nhẹ |
| `performance_statistics.gap_fill_rates.breakaway_fill_rate` | 15 | Breakaway Closed in 1 week = **1-9%**, Closed in 1 year = **66-95%** | 🔴 LỆCH LỚN (15% "fill rate" mơ hồ; PDF đo theo nhiều mốc thời gian) |
| `performance_statistics.gap_fill_rates.continuation_fill_rate` | 25 | Continuation Closed in 1 week = **4-20%**, 1 year = **80-95%** | 🔴 LỆCH LỚN |
| `performance_statistics.gap_fill_rates.exhaustion_fill_rate` | 60 | Exhaustion Closed in 1 week = **61-78%** | 🟢 GẦN KHỚP (60 vs 61-78) |
| `performance_statistics.time_to_fill_days.common_gap` | 2 | Area gap = **3 days** | 🟢 GẦN KHỚP |
| `performance_statistics.time_to_fill_days.breakaway_gap` | "never_or_long" | Breakaway = **61-168 days** | 🟢 KHỚP (semantic đúng) |
| `performance_statistics.time_to_fill_days.continuation_gap` | 14 | Continuation = **43-98 days** | 🔴 LỆCH LỚN (14 vs 43-98) |
| `performance_statistics.time_to_fill_days.exhaustion_gap` | 5 | Exhaustion = **7-14 days** | 🟡 LỆCH nhẹ (5 vs 7-14) |
| `variant_handling.variants[].volume_multiplier_min` | breakaway 1.5, common 0.8, continuation 1.3, exhaustion 2.0 | PDF: chỉ mô tả định tính ("high volume", "usually high") — **KHÔNG có số multiplier chính xác** | 🔴 THIẾU nguồn số |
| Sample size (Number of formations) | KHÔNG ghi | Area **484** · Breakaway **737** · Continuation **495** · Exhaustion **471** | 🔴 THIẾU hoàn toàn |

**Tóm tắt lệch Gaps:** 🔴 **LỆCH NGHIÊM TRỌNG và BỊA NHIỀU TRƯỜNG.** File digitized chứa nhiều trường KHÔNG TỒN TẠI trong PDF Bulkowski (failure_rate, ultimate_high/low_method, average_move, target_calculation cho gaps). Các trường "fill rate" lệch lớn với PDF. Chỉ trường `time_to_fill_days` và `gap_size_effect` là có cơ sở (gần đúng). **Sample size (2.187 gaps tổng) hoàn toàn thiếu.**

---

## Bằng chứng verbatim (số liệu thô, ≤3 dòng, bản quyền)

### Results Snapshot Gaps (PDF p385 / sách p362)
```
Performance rank              Not applicable
Close Within a Week           Bull Market    Bear Market
  Area gap                    89%            93%
  Breakaway                   2%             9%
  Continuation                4%             20%
  Exhaustion                  61%            78%
```

### Table 23.2 Area gaps (PDF p389 / sách p366)
```
Number of formations          127    154    127    76
Average time to close         3 days 3 days 3 days 3 days
Closed in 1 week              89%    93%    92%    89%
```

### Table 23.3 Breakaway gaps (PDF p391 / sách p368)
```
Number of formations          345    92     226    74
Average time to close         136 d  61 d   168 d  111 d
Closed in 1 year              66%    95%    75%    70%
```

### Table 23.5 Exhaustion gaps (PDF p393 / sách p370)
```
Number of formations          120    111    129    111
Average time to close         9 d    7 d    14 d   10 d
Closed in 1 week              61%    78%    64%    63%
```

---

## Reproducer

```bash
PDF="/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet/Nghiên cứu mô hình nến/references/encyclopedia-of-chart-patterns-2nbsped-9786468600-3175723993-9780471668268-0471668265_compress.pdf"

# Results Snapshot (PDF p385-386)
pdftotext -layout -f 385 -l 386 "$PDF" - | sed -n '/RESULTS SNAPSHOT/,/Tour/p'

# 4 subtype tables (PDF p389, 391, 392, 393)
for p in 389 391 392 393; do
  echo "=== Page $p ==="
  pdftotext -layout -f $p -l $p "$PDF" - | sed -n '/Number of formations/,/Average gap size/p'
done
```

---

## Khuyến nghị (không thực hiện trong task này)

1. **Xóa các trường BỊA trong digitized**: `failure_rate.*_at_5pct`, `ultimate_high/low_method.*_average_days`, `average_move.*_rise_pct`, `target_calculation` — gaps KHÔNG có các metric này trong Bulkowski.
2. **Bổ sung sample size** cho 4 subtype: Area 484, Breakaway 737, Continuation 495, Exhaustion 471 (tổng 2.187).
3. **Thay metric chính bằng "Close Within a Week"** (fill rate trong 1 tuần) + "Average time to close the gap" — đây là 2 metric duy nhất Bulkowski đo cho gaps.
4. **Sửa gap_fill_rates**: breakaway 1-9% (1 week) / 66-95% (1 year); continuation 4-20% / 80-95%; exhaustion 61-78% / 99-100%; area 89-93% / 100%. Digitized hiện ghi 1 con số mơ hồ per subtype — không đúng vì fill rate phụ thuộc thời gian.
5. **Ghi rõ "Performance rank: Not applicable"** trong digitized — gaps KHÔNG thể so sánh performance với chart patterns khác.
6. **Đánh dấu gaps là "event pattern"** (không phải chart pattern) — cấu trúc detection khác hẳn.

---

**Hết file.**
