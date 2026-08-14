# Deep-dive: BARR BOTTOMS — measure rule hit rate giảm 48,6% → 35,9%

**Ngày phân tích:** 2026-08-14
**3 lớp bắt buộc:** Session mẹ `sess_0fb10a3f-28e7-42c6-8bdd-6aca4b147362` · Model `glm-5.2` · Provider `OpenCode Go`
**Vai trò:** Verify độc lập — CHỈ đọc/phân tích/đề xuất. KHÔNG sửa code, KHÔNG rescan, KHÔNG commit.
**Nguồn PDF:** `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` (ECP)
**Phương pháp:** `pdftotext -layout` (Poppler). Offset PDF ↔ sách in: **+23** (đã xác thực 5 điểm ở file supplement).

---

## TÓM TẮT KẾT LUẬN (đọc trước)

| # | Hỏi | Trả lời |
|---|---|---|
| 1 | "top of the chart pattern" nghĩa gì? | **= đỉnh cao nhất TOÀN mẫu hình = "old high" = ĐẦU formation** (không phải chỉ lead-in). Sách verbatim rõ ràng (xem Mục 1). |
| 2 | Code bắt đúng "top of pattern" chưa? | **VỀ ĐỊNH NGHĨA là ĐÚNG** (`max(high)` trên cửa sổ `[lead_start-2, bump_idx]` ≈ đỉnh đầu lead-in). Công thức sửa 14/08 theo đúng ý sách. Cửa sổ [-2..] hẹp nhưng sai số (nếu có) chỉ làm target THẤP HƠN thực → đẩy hit rate LÊN, KHÔNG giải thích việc giảm. |
| 3 | Lookahead pipeline vs sách? | **LỆCH NGHIÊM TRỌNG — đây là NGUYÊN NHÂN chính.** Registry cấp cho bottoms lookahead = **68 bars (số TOPS, ch.8 ultimate-low)**, nhưng sách bottoms days-to-ultimate-high = **186 (bull) / 109 (bear)**. Bottoms bị đo trong ~1/3 thời gian cần thiết. |
| 4 | Kết luận (a/b/c)? | **(b) lookahead lệch là nguyên nhân chính (CHẮC CHẮN).** Công thức (a) ĐÚNG sách rồi. (c) thị trường VN chưa kết luận được vì lookahead sai làm số đo không đáng tin. |
| 5 | Đề xuất | Thêm 2 entry `_VARIANT_LOOKAHEAD` cho `bump_and_run_reversal_bottoms` (186/109) và `bump_and_run_reversal_tops` (68/39), rồi rescan. KHÔNG động đến công thức target nữa. |

**Bằng chứng bất đối xứng (smoking gun):** Tops dùng lookahead 68 ≈ sách ch.8 ultimate-low 68 → Tops đo ổn (1 trong 5 mẫu "đúng hướng sách"). Cùng con số 68 đó áp cho Bottoms lại SAI vì Bottoms cần 186. Đúng 1 con số, đúng cho Tops, sai cho Bottoms → chỉ Bottoms lệch.

---

## MỤC 1 — Sách ECP chương 7 (BARR Bottoms): số verbatim + trang

> Lưu ý sửa nhãn trang: file `family_bump_and_run_20260813.md` (đợt 13/08, cùng model) ghi số liệu ĐÚNG nhưng dán nhãn "PDF p122 / PDF p139" — đó thực ra là **số trang SÁCH IN**, không phải PDF. Offset thật +23 (xác thực ở file supplement). Bảng dưới dùng **số sách in (→ PDF = +23)** cho chuẩn.

### 1.1 Measure rule — Table 7.8 (sách p127–128 / PDF p150–151)

**Prose (verbatim ngắn, bản quyền):**
> "Measure rule. After properly identifying a BARR bottom, you will want to determine how profitable is a trade likely to be. You do that using the measure rule. **I changed the measure rule from a computation to simply the top of the chart pattern. The highest high is the target, and prices reach the high 64% to 68% of the time.**"

**Table 7.8 — dòng Measure rule (verbatim):**
> "**The highest high in the pattern is the target.**"

**Tactic kèm theo — "Sell at old high" (xác nhận ý "top of pattern"):**
> "I have discussed how often a stock showing a BARR bottom stops near **the old high (which is the start of the formation)**. Place a sell order near the price level of the old high."

### 1.2 "top of the chart pattern" nghĩa chính xác là gì?

- Sách dùng 3 cụm đồng nghĩa: **"the top of the chart pattern"** = **"the highest high [in the pattern]"** = **"the old high (which is the start of the formation)"**.
- → Đó là **đỉnh cao nhất của TOÀN mẫu hình**, và sách khẳng định nó nằm ở **ĐẦU formation** (nơi trendline lead-in bắt đầu dốc xuống). KHÔNG phải "chỉ đỉnh của lead-in phase" tách rời — mà chính là điểm bắt đầu mà lead-in xuống từ đó.
- **KHÔNG có ví dụ số cụ thể** cho bottoms (khác Tops có ví dụ 21.50−18=3.50). Bulkowski nói thẳng ông **bỏ hẳn phép tính** ("changed … from a computation to simply the top") → target = 1 con số (đỉnh cao nhất), không cần công thức trừ/cộng.
- % meeting target: **64% (bear/UA) – 68% (bull/UA)**.

### 1.3 % meeting price target — Results Snapshot (sách p115 / PDF p138)

| | Bull Market, Up Breakout | Bear Market, Up Breakout |
|---|---|---|
| **Percentage meeting price target** | **68%** | **64%** |
| Break-even failure rate | 2% | 1% |
| Average rise | 38% | 31% |
| Throwbacks | 59% | 73% |
| Performance rank | 8 out of 23 | 3 out of 19 |

### 1.4 Days to ultimate high + Sample — Table 7.2 (sách p121–122 / PDF p144–145)

```
Number of formations          412             120
Reversal (R), continuation    203 R,209 C     66 R,54 C
Average rise                  38%             31%
Rises over 45%                142 or 34%      30 or 25%
Change after trend ends       –29%            –34%
Days to ultimate high         186             109
```

- **Sample:** 412 (bull/UA) + 120 (bear/UA) = **532 BARR bottoms**.
- **Days to ultimate high: 186 (bull) / 109 (bear)** — đây là số chuẩn để so lookahead pipeline.
- **Timeframe:** sách = 500 cổ phiếu Mỹ, mid-1991→mid-1996 + đợt bear 2000–2002; bảng đo bằng **NGÀY GIAO DỊCH** (daily), dùng **ultimate-high method** (đợi giá đảo chiều 20%).
- Ghi chú: "ultimate high" (186d) thường NẮM TRÊN target (old high) — tức giá vượt old high rồi tiếp tục lên đỉnh cuối rồi mới quay xuống 20%. Nên "đạt target (old high)" xáy ra TRƯỚC ultimate high, nhưng vẫn cần nhiều tuần tháng.

---

## MỤC 2 — Đọc code đã sửa

### 2.1 Công thức target bottoms — `scanner/v2/bump_and_run.py:208-212`

```python
if self.direction == 1:
    # Sửa 14/08/2026 theo ... sách ECP Table 7.8:
    # "I changed the measure rule ... to simply the top of the chart pattern"
    # → target = đỉnh cao nhất toàn pattern (mở nhẹ 2 nến về trước để bắt đỉnh thật đầu lead-in).
    target = float(df.iloc[max(0, lead_start - 2) : bump_idx + 1]["high"].max())
```

**Truy ngược `lead_start` (`scan_candidate`, dòng 152-155):**
```python
bump_idx = int(bump.idx)                                   # pivot LOW = đáy bump
lead_end  = bump_idx - bump_min_bars_after_lead            # - 8
lead_start = lead_end - lead_bars + 1                      # lead_bars ∈ {35, 50, 70, 95}
```
- Lead-in được fit regression trên close `[lead_start, lead_end]` (dòng 162-165); lead-in là **downtrend** (direction==1 yêu cầu `lead_change_pct < -5%`, dòng 173-175) → điểm CAO NHẤT của lead-in nằm ở `lead_start`.
- Cửa sổ target `[lead_start-2, bump_idx]`: mở về trước 2 nến rồi kéo dài tới đáy bump. Vì lead-in dốc xuống từ `lead_start`, `max(high)` trong cửa sổ này ≈ **đỉnh đầu lead-in**.

**Đánh giá: công thức ĐÚNG tinh thần sách** ("top of the chart pattern" ≈ đỉnh đầu lead-in = old high). Đây chính là điều Bulkowski nói ("changed from a computation to simply the top").

### 2.2 Cửa sổ [-2..] có bắt ĐÚNG "top of pattern" không?

| Câu hỏi | Trả lời |
|---|---|
| `lead_start` có = "start of formation" / "old high" theo sách? | **Gần đúng.** Code đặt `formation_start_idx = lead_start` (dòng 239). Sách: "old high (start of formation)". Vì lead-in dốc xuống, `lead_start` là điểm cao nhất của lead-in ≈ old high. |
| Có thể ĐỈNH thật nằm cách `lead_start` NHIỀU nến? | **Có thể** (vd 1 nhô cao rồi lead-in mới bắt đầu từ đỉnh thấp hơn — đúng kịch bản Fig 7.6: "stock peaked … then formed a second minor high"). Khi đó cửa sổ `[-2..]` có thể **BỎ SÓT** đỉnh thật cao hơn. |
| Hệ quả nếu bỏ sót đỉnh cao hơn? | Target code THẤP HƠN thực → **hit rate ĐẨY LÊN** (dễ chạm hơn). |
| → Có thể là nguyên nhân hit rate GIẢM (35,9%)? | **KHÔNG.** Sai số cửa sổ (nếu có) chỉ kéo hit rate LÊN, ngược chiều với hiện tượng giảm. |

**Kết luận 2.2:** Cửa sổ `[-2..]` hơi hẹp về mặt lý tưởng (nên mở rộng hoặc neo vào pivot high thực sự trước lead-in), nhưng KHÔNG PHẢI nguyên nhân hit rate giảm. Công thức sửa 14/08 ĐÚNG sách — không cần sửa lại công thức.

### 2.3 So sánh công thức CŨ vs MỚI (git diff commit f7ff7c6)

```diff
- target = confirmation_price + bump_height_abs if self.direction == 1 else confirmation_price - bump_height_abs
+ if self.direction == 1:
+     target = float(df.iloc[max(0, lead_start - 2) : bump_idx + 1]["high"].max())   # = highest high
```
- **CŨ (bottoms):** `target = breakout + bump_height` — project chiều cao bump lên từ breakout. Đây CHÍNH LÀ loại "computation" mà Bulkowski nói ông **BỎ** ("changed the measure rule from a computation to simply the top"). CŨ = công thức bị sách loại bỏ.
- **MỚI (bottoms):** `target = highest high in pattern` — ĐÚNG Table 7.8.
- Sửa đúng hướng. Vấn đề KHÔNG nằm ở công thức.

### 2.4 Lookahead pipeline vs sách — `scanner/v2/measurement_registry.py`

Trace `lookahead_bars("bump_and_run_reversal_bottoms")`:
1. `_VARIANT_LOOKAHEAD.get("bump_and_run_reversal_bottoms")` → **KHÔNG CÓ** (dict chỉ có horn/rectangle/rounding/gaps/triangles — không có bump, đã `grep` xác nhận).
2. `family_of(...)` = `"bump_and_run_reversal"` (dòng 537).
3. `_PDF_OVERRIDES["bump_and_run_reversal"]` (dòng 51-52):
   ```python
   "bump_and_run_reversal": {"lookahead_bull": 68, "lookahead_bear": 39, "sample": 1309,
       "note": "M5 PDF ch.8 Tops: ultimate low bull 68 / bear 39 (digitized 252 SAI 4-6 lần)"}
   ```
4. → `lookahead_bars = 68` (giá trị bull).

**Đã chạy xác nhận runtime:**
```
bump_and_run_reversal_bottoms: lookahead_bars=68  bull=68 bear=39  note="M5 PDF ch.8 Tops: ultimate low bull 68 / bear 39"
bump_and_run_reversal_tops:    lookahead_bars=68  bull=68 bear=39  (cùng note Tops)
```

**Commit fix f7ff7c6 KHÔNG sửa registry** (`git show --stat`: 5 file, không có measurement_registry.py) → registry vẫn cấp 68 cho bottoms.

### 2.5 Bảng so lookahead pipeline vs sách

| | Sách (days to ultimate) | Pipeline (lookahead_bars) | Đủ? |
|---|---|---|---|
| **BARR Bottoms (ch.7)** | **186 bull / 109 bear** | **68** (lấy nhầm số Tops) | 🔴 **THIẾU 118 ngày (bull) / 41 ngày (bear)** — chỉ ~37% thời gian cần |
| **BARR Tops (ch.8)** | 68 bull / 39 bear | 68 | 🟢 ĐÚNG (trùng hợp: số Tops đúng cho Tops) |

**Cơ chế gây giảm hit rate:** target mới = highest high = giá phải **truy ngược toàn bộ đoạn giảm** từ đỉnh mẫu hình xuống breakout. Sách tự khẳng định việc này mất **186 ngày** (bull) trung bình + throwback 59–73% làm chậm thêm. Pipeline chỉ cho **68 bars** → phần lớn pattern **chưa kịp** chạm target → hit rate bị ép xuống giả tạo.

**Tại sao CŨ (công thức bị bỏ) lại "đạt" 64% ở 68 bars?** Bản backup `artifacts/scanner_v2_backup_lookahead_fix_20260812/.../bump_and_run_reversal_bottoms/statistics.json` (công thức cũ + lookahead 68) ghi `target_hit_rate = 64,43%`, `median_target_dist_pct = 14,3%`. Target CŨ = breakout + bump_height (gần, ~14%) → dễ chạm trong 68 bars. CŨ "khớp sách 64-68%" **do ngẫu nhiên** (target thấp + cửa sổ ngắn triệt tiêu nhau), chứ không phải đo đúng. MỚI target đúng nhưng XA hơn + vẫn bị kẹp trong 68 bars → lộ ra chỗ yếu thật: lookahead.

### 2.6 Đọc nhanh `family_bump_and_run_20260813.md` phần bottoms

File đợt 13/08 đã trích ĐỦ dữ liệu sách cho bottoms (sample 532, %target 68/64, days 186/109, measure rule = highest high). **NHƯNG** khi nạp vào registry, người nạp **chỉ dùng số Tops** (68/39) cho entry family `bump_and_run_reversal`, và **quên tách 2 entry variant** bottoms/tops như đã làm cho horn/rectangle/rounding. Đây là lỗi "rò rỉ" giữa bước trích-số và bước nạp-registry — số bottoms có sẵn trong file review nhưng không vào được registry.

---

## MỤC 3 — KẾT LUẬN + ĐỀ XUẤT (chỉ đề xuất, không tự sửa)

### Nguyên nhân (chọn a/b/c)

**(b) LOOKAHEAD LỆCH — nguyên nhân chính. Độ tin cậy: CHẮC CHẮN.**
- Bằng chứng cứng (registry + runtime): bottoms nhận lookahead 68 (số Tops) thay vì 186/109 (sách).
- Bằng chứng bất đối xứng: cùng con số 68 đúng cho Tops (1 trong 5 mẫu OK) nhưng sai cho Bottoms → đúng pattern duy nhất lệch.
- Commit fix không đụng registry → lỗi còn nguyên.

**(a) sai cách áp công thức — KHÔNG PHẢI nguyên nhân. Độ tin cậy: CHẮC CHẮN KHÔNG PHẢI.**
- Công thức mới `highest high in pattern` ĐÚNG verbatim Table 7.8. Sai số cửa sổ `[-2..]` (nếu có) đẩy hit rate LÊN, ngược chiều giảm.

**(c) thị trường VN khác — CHƯA KẾT LUẬN ĐƯỢC. Độ tin cậy: CẦN XÁC NHẬN sau khi sửa lookahead.**
- Vì lookahead sai, số đo VN hiện tại (35,9%) KHÔNG đáng tin để so với sách. Phải sửa lookahead trước, rescan, rồi mới phán (c).

### Đề xuất cụ thể (chỉ đề xuất)

**1. (CRITICAL) Sửa registry — thêm 2 entry variant trong `_VARIANT_LOOKAHEAD` (`measurement_registry.py:85`):**
```python
"bump_and_run_reversal_bottoms": {"lookahead_bull": 186, "lookahead_bear": 109, "source": "pdf",
    "note": "M5 PDF ch.7 Bottoms: ultimate high bull 186 / bear 109 (family-level đang dùng Tops 68 — SAI cho Bottoms)"},
"bump_and_run_reversal_tops":    {"lookahead_bull": 68,  "lookahead_bear": 39,  "source": "pdf",
    "note": "M5 PDF ch.8 Tops: ultimate low bull 68 / bear 39"},
```
Rồi rescan bottoms, đo lại hit rate. Dự kiến nhích về phía 64-68% sách (có thể vẫn lệch nhẹ do VN).

**2. (TÙY CHỌN, độ ưu tiên thấp) Mở rộng cửa sổ "top of pattern":** thay `lead_start-2` bằng việc tìm pivot HIGH thực sự trước/near lead_start (neo vào đỉnh thật đầu formation) — khớp chặt hơn "old high = start of formation". Lưu ý: chỉ làm target CHÍNH XÁC HƠN (thường CAO hơn → hit rate ĐI XUỐNG chút), nên làm SAU bước 1 để không pha tạp 2 nguyên nhân.

**3. KHÔNG sửa lại công thức target bottoms** — nó đã đúng sách.

### Caveats / giới hạn phân tích

- Không tìm được nguồn file chính xác của cặp số "48,6% → 35,9%" (grep quét scan_results/logs/docs chưa khớp; artifacts statistics.json hiện có đều mang timestamp 12/08 = trước commit fix 14/08, tức scanner chưa rescan chính thức sau fix). Con số do main agent báo từ lần đo lại riêng. Chẩn đoán dựa trên cấu trúc registry+code, độc lập với con số chính xác → vẫn đứng vững.
- Số "hit rate cũ 64,43%" lấy từ bản backup lookahead_fix_20260812 (công thức CŨ + lookahead 68) — dùng làm đối chứng cơ chế, không phải kết quả hậu-fix.

---

## Reproducer

```bash
PDF="references/encyclopedia-of-chart-patterns-2nbsped-9786468600-3175723993-9780471668268-0471668265_compress.pdf"
# Measure rule Bottoms — Table 7.8 (sách p127-128 = PDF p150-151)
pdftotext -layout "$PDF" - | sed -n '6210,6260p'      # prose "changed the measure rule ... top of the chart pattern"
# Results Snapshot Bottoms (sách p115 = PDF p138)
pdftotext -layout "$PDF" - | sed -n '5598,5618p'      # % meeting target 68/64
# Table 7.2 (sách p121-122 = PDF p144-145)
pdftotext -layout "$PDF" - | sed -n '5916,5950p'      # sample 412/120, days to ultimate high 186/109

# Registry lookahead (runtime)
.venv/bin/python -c "from scanner.v2.measurement_registry import lookahead_bars as L; print('bottoms',L('bump_and_run_reversal_bottoms'),'tops',L('bump_and_run_reversal_tops'))"
# Git: commit fix không sửa registry
git show f7ff7c6 --stat | grep measurement_registry   # → không có (confirm)
```

---

**Hết file.**
