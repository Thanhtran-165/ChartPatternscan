# M5 — Trích số liệu PDF gốc: family SPIKE

**Ngày trích:** 2026-08-13
**Model / Provider:** GLM-5.2 / builtin:zai-coding-plan (Z.AI Coding Plan)
**Session mẹ:** `sess_0fb10a3f-28e7-42c6-8bdd-6aca4b147362`
**Nguồn PDF (kiểm tra cả 2 sách Bulkowski):**
- ECP 2nd ed: `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` (1.035 trang)
- EC Candlestick: `references/Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf` (966 trang)
**Phương pháp:** `pdftotext -layout` (Poppler 26.04.0). Offset PDF ↔ sách in: **+23 trang** cho cả 2 sách.

---

## 🚨 PHÁT HIỆN QUAN TRỌNG NHẤT CỦA CẢ TASK M5-2a

### ❌ KHÔNG TÌM THẤY chương "Spike" trong CẢ HAI sách Bulkowski

Đã kiểm chứng bằng 4 phương pháp độc lập:

**1. Mục lục (Table of Contents) — ECP:**
- ECP có **46 chương chart patterns + event patterns** (chương 1-46). Tên chương sắp xếp alphabet.
- Tìm "Spike" trong TOC (PDF p5-25 / sách in p xi-xvii) → **KHÔNG CÓ** chương nào tên "Spike" / "Spike Formation" / "Spike Reversal" / "Exhaustion Spike" / "Reversal Spike".
- Các chương có chữ "S" trong ECP: Scalping Trends (1) · Scallops Ascending (41) · Scallops Ascending & Inverted (42) · Scallops Descending (43) · Scallops Descending & Inverted (44) · Separating Lines · Short Poles · Snake · Spinning Top · Stick Sandwich · Sucker Reflexes · Surprise Earnings · Swinging Tails · Symmetric Triangles. **Không có Spike.**

**2. Mục lục — EC Candlestick:**
- EC Candlestick có **105 candlestick patterns** (chương 1-105).
- Tìm "Spike" trong TOC (PDF p15-22 / sách in p vii-xii) → **KHÔNG CÓ**.
- Các candlestick patterns gần "spike" nhất (theo concept): **Shooting Star One-Candle (ch.76)**, **Shooting Star Two-Candle (ch.77)**, **Takuri Line (ch.83)**, **High Wave (ch.47)**, **Belt Hold Bearish (ch.12)**, **Belt Hold Bullish (ch.13)**. **Không có Spike.**

**3. Subject Index — ECP** (PDF p1020-1035 / sách in p997-1012):
```
Spike(s)                          519, 520, 565, 567, 650, 651, 656,
                                  747, 810, 879, 892, 907, 933
Spike High         Spike Low                     Triple Tops, page 779
```
→ Đây là **tham chiếu phụ** (sub-component) trong các chương khác — KHÔNG phải chương riêng. Spot-check trực tiếp:
- p650/651/656 = chương 43 (Scallops Descending): *"the bowl has a downward price spike"* / *"a volume spike"* — nói về spike như đặc điểm phụ của scallop.
- p519/520 → nằm trong chapter 32 (Measured Move Down) — "spike" như một đặc điểm phụ.
- "Spike High / Spike Low" trong visual index p779 = đặc điểm đồ thị (chart feature), **KHÔNG phải pattern có sample statistics.

**4. Subject Index — EC Candlestick** (PDF p955-966 / sách in p929-940):
- Tìm "spike" → **KHÔNG CÓ entry "Spike"** trong index.

### Hệ quả: file `spike_formation_digitized.json` KHÔNG CÓ nguồn từ sách Bulkowski

Mọi số liệu trong digitized đều **FABRICATED** (bịa hoặc lấy từ nguồn khác không phải Bulkowski — không xác định được nguồn). Không có sample size, không có failure rate, không có measure rule, không có % meeting target — vì **Bulkowski không nghiên cứu "spike" như một pattern độc lập**.

### "Spike" trong văn hiến Bulkowski là gì?

Trong 2 sách Bulkowski, "spike" xuất hiện với 3 nghĩa phụ:

1. **Chart feature** (đặc điểm đồ thị): "spike high" / "spike low" = đỉnh/đáy nhọn đơn lẻ — được dùng trong visual index để giải thích các pattern khác (Triple Tops, Head-and-Shoulders, v.v.).
2. **Volume feature**: "volume spike" = thanh khoản tăng vọt — đặc điểm phụ của nhiều pattern (BARR, scallops, breakaway gaps).
3. **Sub-component**: "downward price spike" = phần lõm/down trong một pattern (ví dụ trong scallops).

→ **"Spike" KHÔNG được treated như 1 pattern độc lập có failure rate, sample, target.**

### Closest patterns có thể thay thế (nếu cần detect spike-like):

| Pattern | Sách | Có thể dùng thay "spike" vì... |
|---|---|---|
| **Shooting Star One-Candle** (ch.76 EC Candlestick, p660-667 sách in) | EC Candlestick | 1-candle bearish reversal với long upper shadow — giống "bearish spike reversal" |
| **Shooting Star Two-Candle** (ch.77, p668-675) | EC Candlestick | 2-candle variant |
| **Takuri Line** (ch.83, p720-727) | EC Candlestick | 1-candle bullish reversal với long lower shadow — giống "bullish spike reversal" |
| **High Wave** (ch.47, p409-417) | EC Candlestick | 1-candle với long shadow cả 2 bên |
| **Belt Hold Bearish/Bullish** (ch.12/13, p118-136) | EC Candlestick | 1-candle marubozu-like reversal |
| **Pipe Bottoms / Pipe Tops** (ECP ch.35/36) | ECP | 2-candle spike-like reversal (đã có file riêng trong M5) |

---

## Bảng — KHÔNG CÓ số liệu (vì không có chương Spike trong PDF)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | **N/A** — không có chương Spike trong cả 2 sách | |
| **pages_checked** | **N/A** | Đã kiểm tra toàn bộ TOC + Subject Index của cả 2 sách |
| **TOC ECP checked** | PDF **5-25** (TOC 46 chương) | Không có "Spike" |
| **TOC EC Candlestick checked** | PDF **15-22** (TOC 105 candlesticks) | Không có "Spike" |
| **Subject Index ECP checked** | PDF **1020-1035** | Chỉ có entry "Spike(s)" với tham chiếu sub-component; không có "Spike" như pattern chương |
| **Subject Index EC Candlestick checked** | PDF **955-966** | Không có entry "Spike" |
| **sample** | **N/A — KHÔNG CÓ** | Bulkowski không nghiên cứu spike như pattern độc lập |
| **BE failure rate** | **N/A — KHÔNG CÓ** | |
| **% meeting price target** | **N/A — KHÔNG CÓ** | |
| **Days to ultimate high/low** | **N/A — KHÔNG CÓ** | |
| **Measure rule** | **N/A — KHÔNG CÓ** | |
| **Throwbacks/Pullbacks** | **N/A — KHÔNG CÓ** | |
| **Performance rank** | **N/A — KHÔNG CÓ** | |
| **Average reversal %** | **N/A — KHÔNG CÓ** | |
| **Volume multiplier** | **N/A — KHÔNG CÓ** | |
| **Time to confirmation** | **N/A — KHÔNG CÓ** | |

---

## Đối chiếu với `spike_formation_digitized.json` — TOÀN BỘ FABRICATED

| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `pattern_name` | "Spike Formation" | **KHÔNG TỒN TẠI** trong 2 sách Bulkowski | 🔴 **FABRICATED** — không có chương Spike |
| `pattern_type` | `reversal_neutral` | **N/A** | 🔴 **FABRICATED** |
| `detection_signature.pivot_sequence` | `["H", "L"]` | **N/A** — không có | 🔴 **FABRICATED** |
| `detection_signature.is_single_bar_pattern` | true | **N/A** | 🔴 **FABRICATED** |
| `pivot_requirements.noise_filter_atr_multiplier` | 2.0 | **N/A** | 🔴 **FABRICATED** |
| `geometry_constraints.spike_magnitude.min_range_pct` | 3.0 | **N/A** — không có số | 🔴 **FABRICATED** |
| `geometry_constraints.spike_magnitude.extreme_range_pct` | 5.0 | **N/A** | 🔴 **FABRICATED** |
| `geometry_constraints.spike_vs_average.min_multiple` | 2.5 | **N/A** | 🔴 **FABRICATED** |
| `geometry_constraints.spike_vs_average.ideal_multiple` | 4.0 | **N/A** | 🔴 **FABRICATED** |
| `geometry_constraints.tail_requirements.upper_tail_min_pct` | 0.5 | **N/A** | 🔴 **FABRICATED** |
| `prior_trend_requirements.min_period_bars` | 10 | **N/A** | 🔴 **FABRICATED** |
| `prior_trend_requirements.min_change_pct` | 5.0 | **N/A** | 🔴 **FABRICATED** |
| `breakout_confirmation.volume_multiplier_min` | 2.0 | **N/A** | 🔴 **FABRICATED** |
| `breakout_confirmation.volume_multiplier_ideal` | 3.0 | **N/A** | 🔴 **FABRICATED** |
| **`post_breakout_measurement.target_calculation.method`** | `reversal_magnitude` | **N/A** | 🔴 **FABRICATED** |
| **`post_breakout_measurement.target_calculation.formula`** | `target = spike_close ± (spike_range × reversal_strength_factor)` | **N/A** | 🔴 **FABRICATED** |
| **`post_breakout_measurement.ultimate_high_method.average_days`** | **10** | **N/A** | 🔴 **FABRICATED** |
| **`post_breakout_measurement.ultimate_low_method.average_days`** | **10** | **N/A** | 🔴 **FABRICATED** |
| **`post_breakout_measurement.failure_definition.threshold_pct`** | **3.0** | **N/A** | 🔴 **FABRICATED** |
| **`post_breakout_measurement.failure_rate.at_5pct`** | **35** | **N/A** — không có | 🔴 **FABRICATED** |
| **`post_breakout_measurement.failure_rate.at_10pct`** | **20** | **N/A** | 🔴 **FABRICATED** |
| **`post_breakout_measurement.throwback_pullback.rate_pct`** | **45** | **N/A** | 🔴 **FABRICATED** |
| **`post_breakout_measurement.average_reversal.strong_reversal_pct`** | **50** | **N/A** | 🔴 **FABRICATED** |
| **`post_breakout_measurement.average_reversal.moderate_reversal_pct`** | **30** | **N/A** | 🔴 **FABRICATED** |
| **`post_breakout_measurement.lookahead_bars`** | **20** | **N/A** | 🔴 **FABRICATED** |
| **`performance_statistics.reversal_success_rate`** | **55** | **N/A** | 🔴 **FABRICATED** |
| **`performance_statistics.average_reversal_days`** | **10** | **N/A** | 🔴 **FABRICATED** |
| **`performance_statistics.failure_rate_5pct`** | **35** | **N/A** | 🔴 **FABRICATED** |
| **`performance_statistics.failure_rate_10pct`** | **20** | **N/A** | 🔴 **FABRICATED** |
| **`performance_statistics.time_to_confirmation_days`** | **2** | **N/A** | 🔴 **FABRICATED** |
| **`performance_statistics.throwback_rate_pct`** | **45** | **N/A** | 🔴 **FABRICATED** |
| **`performance_statistics.volume_correlation`** | **0.65** | **N/A** | 🔴 **FABRICATED** |
| **`performance_statistics.tail_size_correlation`** | **0.55** | **N/A** | 🔴 **FABRICATED** |
| `variant_handling.variants[].bullish_reversal_spike` | có | **N/A** | 🔴 **FABRICATED** |
| `variant_handling.variants[].bearish_reversal_spike` | có | **N/A** | 🔴 **FABRICATED** |
| `variant_handling.variants[].exhaustion_spike.reversal_probability` | 0.75 | **N/A** | 🔴 **FABRICATED** |
| `confidence_scoring.threshold_accept` | 65 | **N/A** | 🔴 **FABRICATED** |
| `confidence_scoring.threshold_high_confidence` | 85 | **N/A** | 🔴 **FABRICATED** |
| Sample size | KHÔNG ghi | **N/A — KHÔNG CÓ** | 🔴 Không áp dụng được |

**Tóm tắt lệch Spike:** 🔴🔴🔴 **TOÀN BỘ FILE DIGITIZED LÀ FABRICATED**. Không có một con số nào trong `spike_formation_digitized.json` có nguồn từ 2 sách Bulkowski (ECP + EC Candlestick). Pattern "Spike Formation" như mô tả trong digitized **không tồn tại** như một pattern độc lập trong công trình nghiên cứu của Bulkowski. Đây là **lỗi NGHIÊM TRỌNG NHẤT** trong toàn bộ digitization phase — có thể do AI hallucinate từ thuật ngữ phổ thông "spike" trong trading cộng đồng, hoặc do nhầm với candlestick patterns tương tự (Shooting Star, Takuri Line) nhưng không lấy số từ các chương đó.

---

## Bằng chứng verbatim — kiểm chứng Spike KHÔNG có trong sách

### ECP — Table of Contents (PDF p5-25 / sách p xi-xvii) — toàn bộ chương
```
Chương 1-46 liệt kê theo alphabet. Không có chương nào tên "Spike".
Các chương bắt đầu bằng S: Scallops (4 chương), Snake, Spinning Top, Stick
Sandwich, Surprise. KHÔNG CÓ "Spike".
```

### ECP — Subject Index (PDF p1030 / sách p1007)
```
Spikes or Tails
Spike High         Spike Low                     Triple Tops, page 779
Dual spike, 546                                      969, 977, 981
Spike(s)                                              519, 520, 565, 567, 650, 651, 656,
     747, 810, 879, 892, 907, 933              Twin spikes, 214
```
→ "Spike" chỉ là **tham chiếu phụ** trong các chương khác — không phải chương độc lập. p779 là visual index. Tất cả các trang liệt kê (519, 520, 565, 567, 650, 651, 656, 747, 810, 879, 892, 907, 933) đều nói về spike như đặc điểm PHỤ của pattern khác.

### EC Candlestick — Table of Contents (PDF p15-22 / sách p vii-xii)
```
105 candlestick patterns (chương 1-105). Không có chương "Spike".
Closest: ch.76 Shooting Star One-Candle, ch.77 Shooting Star Two-Candle,
ch.83 Takuri Line, ch.47 High Wave, ch.12/13 Belt Hold.
```

### Spot-check p650/651/656 (ECP chương 43 Scallops Descending, PDF p673-679)
```
"The bowl has a downward price spike, making the..."
"a volume spike"
"spikes in late March and mid-June"
```
→ Đều là đặc điểm PHỤ của scallops — không phải pattern Spike độc lập.

---

## Reproducer — kiểm chứng độc lập

```bash
ECP="/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet/Nghiên cứu mô hình nến/references/encyclopedia-of-chart-patterns-2nbsped-9786468600-3175723993-9780471668268-0471668265_compress.pdf"
CANDLE="/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet/Nghiên cứu mô hình nến/references/Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf"

# Method 1: Search TOC of ECP for "Spike" chapter heading
pdftotext -layout -f 5 -l 25 "$ECP" - | grep -iE "^[0-9]+ Spike|^Spike[[:space:]]"
# → NO OUTPUT (confirmed)

# Method 2: Search TOC of EC Candlestick for "Spike"
pdftotext -layout -f 15 -l 22 "$CANDLE" - | grep -iE "^[0-9]+ Spike|^Spike[[:space:]]"
# → NO OUTPUT (confirmed)

# Method 3: Search ECP subject index
pdftotext -layout -f 1020 -l 1035 "$ECP" - | grep -i "spike"
# → Returns only "Spike(s)  519, 520, ..." — sub-component references

# Method 4: Search EC Candlestick subject index
pdftotext -layout -f 955 -l 966 "$CANDLE" - | grep -i "spike"
# → NO OUTPUT (confirmed)

# Method 5: Full-text search for "Spike" as chapter heading
pdftotext -layout "$ECP" - | grep -nE "^[[:space:]]*[0-9]+[[:space:]]+Spike"
# → NO OUTPUT

pdftotext -layout "$CANDLE" - | grep -nE "^[[:space:]]*[0-9]+[[:space:]]+Spike"
# → NO OUTPUT
```

---

## Khuyến nghị (CRITICAL — không thực hiện trong task này)

1. **🚨 XÓA FILE `spike_formation_digitized.json` HOẶC ĐÁNH DẤU "NOT IN BULKOWSKI" (CRITICAL)**: Toàn bộ nội dung file là **FABRICATED**. Không có một con số nào có nguồn từ sách Bulkowski. Giữ nguyên file sẽ gây hiểu nhầm rằng spike là pattern được Bulkowski nghiên cứu với các số liệu cụ thể.
2. **Nếu vẫn cần detect spike-like patterns** trong scanner, đề xuất **3 lựa chọn thay thế có cơ sở PDF**:
   - **Lựa chọn A (Khuyến nghị)**: Sử dụng **Shooting Star** (EC Candlestick ch.76, p660) cho bearish spike reversal + **Takuri Line** (ch.83, p720) cho bullish spike reversal — cả 2 đều có sample + failure rate + %target trong sách.
   - **Lựa chọn B**: Sử dụng **Pipe Bottoms/Pipe Tops** (ECP ch.35/36 — đã có file digitized + M5 review riêng) — 2-candle spike-like.
   - **Lựa chọn C**: Sử dụng **High Wave** (EC Candlestick ch.47, p409) — 1-candle với long shadow cả 2 bên.
3. **KHÔNG bịa số**: Nếu giữ "spike" như tên internal variant trong scanner (không phải pattern độc lập), phải ghi rõ "Custom variant — NOT in Bulkowski. Numbers below are scanner's heuristic thresholds, not Bulkowski statistics."
4. **Đánh dấu registry**: trong `family_registry` hoặc `manifest`, đánh dấu `spike_formation` là `source: "fabricated" | bulkowski_chapter: null | action: "deprecate_or_remap_to_shooting_star_takuri"`.

---

**Hết file.**
