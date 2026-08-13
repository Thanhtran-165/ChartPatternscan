# M5 — Trích số liệu PDF gốc: family MEASURED MOVES (Down + Up)

**Ngày trích:** 2026-08-13
**Model / Provider:** GLM-5.2 / Z.AI (zai-coding-plan)
**Session mẹ:** `sess_0fb10a3f-28e7-42c6-8bdd-6aca4b147362`
**Nguồn PDF:** `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` (1.035 trang)
**Phương pháp:** `pdftotext -layout` (Poppler 26.04.0). Offset PDF ↔ sách in: **+23 trang** (số in + 23 = số PDF).

---

## ⚠️ Phát hiện quan trọng (áp dụng cho cả MMD + MMU) — KHÁC HOÀN TOÀN với pattern khác

Measured Move là mẫu hình **ĐẶC BIỆT** trong ECP — Bulkowski **từ chối dùng** các metric tiêu chuẩn:

> *"I do not show the average decline or the failure rate. The average decline measures the move from the breakout to the ultimate low, and neither applies to this pattern. The failure rate is a measure of how far price moves after the breakout. That also does not applied."* — ECP p496 (MMD), p510 (MMU)

Lý do: MMD/MMU **không có breakout đơn** (3 phase liên tục) và **không có ultimate low theo nghĩa 20% reversal**. Thay vào đó Bulkowski đo **3 leg riêng biệt**:
- **First leg** (impulse đầu)
- **Corrective phase** (retrace first leg)
- **Last leg** (continuation ≈ first leg)

Do đó:
- **KHÔNG CÓ failure rate** cho MMD/MMU trong PDF
- **KHÔNG CÓ break-even failure rate** trong Results Snapshot (Performance rank = "Not Applicable")
- **KHÔNG CÓ "average rise/decline"** đơn lẻ (chỉ có per-leg)
- Metric chính là **% meeting price target** (measure rule) — rất thấp: MMD 35-39%, MMU 45-56%

File digitized `measured_move_down_up_digitized.json` ghi `failure_rate at_5pct=15, at_10pct=8` → **BỊA** (PDF không có). digitized ghi `success_rate_reaching_target=72` → **SAI LỆCH LỚN** (PDF MMD 35-39%, MMU 45-56%).

**Measure rule** = project first leg từ đỉnh/đáy corrective phase. Dùng "half of first leg" thì đạt target 83-93% (thay vì full first leg chỉ 35-56%).

---

## Bảng 1 — MEASURED MOVE DOWN (ECP chapter 32, PDF p518-531 / sách p495-508)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | |
| **pages_checked** | PDF **518-531** (14 trang) | sách in p495-508 |
| **sample (Number of formations, Table 32.2)** | **647** (bull) + **264** (bear) = **911 MMDs** | "I found 911 MMDs using 5 years of daily price data beginning from mid-1991 on 500 stocks. Another 220 or so stocks created additional bull/bear market data" |
| **Reversal/Continuation (Table 32.2)** | 566R/81C (bull, 87% R) · 214R/50C (bear, 81% R) | Phần lớn reversal; "intermediate-term bearish reversal" |
| **Change after trend ends (Table 32.2)** | +46% (bull) · +49% (bear) | Rebound sau khi MMD kết thúc |
| **Most frequent corrective phase retrace (Table 32.2)** | 40% to 60% (cả bull & bear) | Fibonacci range 38-62% |
| **Average MMD length (Table 32.2)** | **153 days** (bull, ~5 tháng) · **113 days** (bear, ~4 tháng) | Toàn bộ ABCD pattern |
| **Average first leg decline (Results Snapshot + Table 32.2)** | **27% in 61 days** (bull) · **36% in 45 days** (bear) | Bear decline xa hơn, nhanh hơn |
| **Average corrective phase retrace (Results Snapshot)** | **48% in 30 days** (bull) · **44% in 22 days** (bear) | Retrace gần nửa first leg |
| **Average last leg decline (Results Snapshot)** | **25% in 62 days** (bull) · **36% in 46 days** (bear) | Last leg ≈ first leg (dollars), nhưng % thấp hơn vì base cao hơn |
| **% meeting price target (Results Snapshot)** | **35%** (bull) · **39%** (bear) | Measure rule yếu — dưới 40% |
| **% meeting time target (Results Snapshot)** | 53% (bull) · 49% (bear) | |
| **Failure rate / Average decline / BE failure rate** | **NOT APPLICABLE** (PDF từ chối) | "neither applies to this pattern" |
| **Measure rule** | target = **corrective_phase_top − first_leg_decline** (project first leg xuống từ đỉnh corrective phase). Dùng "half of first leg" → đạt target 83-93%. | Quote PDF p497: *"If you measure the first leg price decline and project it downward from the top of the corrective phase, you will hit your target between 35% and 39% of the time. If you take half of the first leg move and project downward, you will hit your target between 83% and 93% of the time"* |
| **Corrective phase retrace frequency (Table 32.2 text)** | <38%: 22% có last leg dài hơn · 38-62%: 31% · >62%: 58% | Retrace lớn → last leg dài hơn |

### Đối chiếu với `measured_move_down_up_digitized.json` (phần MMD)

| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `post_breakout_measurement.failure_rate.at_5pct` | 15 | **NOT APPLICABLE** (PDF không có) | 🔴 **BỊA** — metric không tồn tại cho MMD |
| `post_breakout_measurement.failure_rate.at_10pct` | 8 | **NOT APPLICABLE** | 🔴 **BỊA** |
| `performance_statistics.failure_rate_5pct` | 15 | NOT APPLICABLE | 🔴 **BỊA** |
| `performance_statistics.failure_rate_10pct` | 8 | NOT APPLICABLE | 🔴 **BỊA** |
| `post_breakout_measurement.failure_definition.threshold_pct` | 5.0 | NOT APPLICABLE (no failure rate) | 🔴 **BỊA** — failure definition không áp dụng |
| `performance_statistics.success_rate_reaching_target` | 72 | MMD: bull 35%, bear 39% | 🔴 **LỆCH LỚN** (72 vs 35-39 — digitized cao gấp ~2 lần) |
| `post_breakout_measurement.average_move.phase3_to_target_pct` | 100 | % meeting target 35-39%; last leg ≈ 80% of first leg (dollars) | 🔴 **SAI SEMANTIC** — "100%" không phải % reaching target, mà là khái niệm lý thuyết (phase 3 = phase 1) |
| `performance_statistics.phase2_ideal_retracement_pct` | 50 | most frequent 40-60%, avg 48% (bull)/44% (bear) | 🟢 KHỚP (50 nằm trong 40-60%, gần avg) |
| `geometry_constraints.phase2_retracement_ratio` (min 0.33, max 0.67) | 33-67% | PDF: <38% → shorter last leg; >62% → longer last leg; no hard invalidation | 🟡 GẦN KHỚP (PDF dùng 38/62 làm ngưỡng thống kê, không phải hard rule) |
| `performance_statistics.average_completion_days` | 42 | MMD length: bull 153d, bear 113d (toàn pattern); last leg 62/46d | 🔴 LỆCH (42 vs 113-153 whole / 46-62 last leg) |
| `performance_statistics.time_to_completion_days` | 21 | last leg: bull 62d, bear 46d | 🔴 LỆCH (21 vs 46-62) |
| `post_breakout_measurement.ultimate_high_method.average_days` | 21 | MMD không dùng ultimate method | 🔴 **SAI SEMANTIC** — không có ultimate high cho MMD |
| `post_breakout_measurement.ultimate_low_method.average_days` | 21 | NOT APPLICABLE | 🔴 **SAI SEMANTIC** |
| `post_breakout_measurement.lookahead_bars` | 63 | MMD length 113-153 days; last leg 46-62 days | 🔴 LỆCH (63 vs 113-153 whole) |
| `post_breakout_measurement.target_calculation.method` | `phase1_distance` — "Target = Phase 2 end +/- Phase 1 distance" | `first_leg_projection` — target = corrective_phase_top/bottom −/+ first_leg_decline | 🟢 KHỚP method (cùng ý: project first leg từ corrective phase) |
| `post_breakout_measurement.target_calculation.formula` | `bullish_target = phase2_low + (phase1_high - phase1_low)` | `target = corrective_phase_top − first_leg_decline` (MMD) | 🟢 GẦN KHỚP (cùng cấu trúc) |
| `post_breakout_measurement.throwback_pullback.rate_pct` | 30 | "minor pullbacks during Phase 3 are common" — không có số % cụ thể trong PDF | 🟡 KHÔNG XÁC ĐỊNH số chính xác |
| `geometry_constraints.phase1_phase3_ratio` (min 0.85, max 1.15) | 85-115% | PDF: dollar basis last leg DÀI HƠN first leg 35-39% of time; avg last leg 19-20% SHORTER than first | 🔴 **SAI** — digitized bắt phase3 ≈ phase1 ±15%, nhưng PDF: avg last leg SHORTER 19-20% (dollars); ratio thực 0.80-0.81, ngoài khoảng 0.85-1.15 |
| `confidence_scoring.scoring_rules.phase1_phase3_distance_similarity.within_5pct=100` | within 5% = best | PDF: chỉ 35-39% meeting target (full first leg) | 🔴 **SAI** — within 5% cực hiếm (35-39% đạt target); majority last leg 19-20% shorter |
| `geometry_constraints.width_optimal_bars` | 42 | MMD length bull 153d, bear 113d | 🔴 LỆCH (42 vs 113-153) |
| `duration_constraints.optimal_bars` | 42 | 113-153d whole MMD | 🔴 LỆCH |
| `breakout_confirmation.volume_multiplier_min` | 1.2 | PDF: "Volume should increase on Phase 1 and Phase 3, decrease during Phase 2" qualitatively | 🟡 KHÔNG XÁC ĐỊNH số chính xác |
| Sample size | KHÔNG ghi | MMD **911**, MMU **810** | 🔴 THIẾU hoàn toàn |
| `% meeting time target` | KHÔNG có | MMD 53%/49% | 🔴 THIẾU |

**Tóm tắt lệch MMD:** 🔴 **LỆCH NGHIÊM TRỌNG + BỊA SỐ** về (1) failure rate BỊA (15/8 — PDF từ chối metric này), (2) success_rate_reaching_target (72 vs 35-39 — gấp đôi), (3) ultimate_high/low_method SAI SEMANTIC (MMD không dùng ultimate), (4) phase1_phase3_ratio (0.85-1.15 — nhưng PDF last leg SHORTER 19-20%, ratio thực 0.80-0.81 ngoài khoảng), (5) lookahead/completion days (21-63 vs 113-153). 🟢 KHỚP tốt ở: target method (phase1_distance), phase2 retracement 50% (40-60%).

---

## Bảng 2 — MEASURED MOVE UP (ECP chapter 33, PDF p532-543 / sách p509-520)

| Trường | Giá trị PDF gốc | Ghi chú |
|---|---|---|
| **pdf_path** | `references/encyclopedia-of-chart-patterns-2nbsped-...0471668265_compress.pdf` | |
| **pages_checked** | PDF **532-543** (12 trang) | sách in p509-520 |
| **sample (Number of formations, Table 33.2)** | **577** (bull) + **233** (bear) = **810 MMUs** | "reverse of MMD but with worse performance" |
| **Reversal/Continuation (Table 33.2)** | 393R/184C (bull, 68% R) · 179R/54C (bear, 77% R) | "long-term bullish reversal" |
| **Change after trend ends (Table 33.2)** | −26% (bull) · −27% (bear) | Decline sau khi MMU kết thúc |
| **Most frequent corrective phase retrace (Table 33.2)** | 40% to 60% (cả bull & bear) | |
| **Average MMU length (Table 33.2)** | **180 days** (bull, ~6 tháng) · **85 days** (bear, ~3 tháng) | Bull market MMU dài gấp đôi bear |
| **Average first leg rise (Results Snapshot)** | **46% in 87 days** (bull) · **39% in 30 days** (bear) | Bull first leg mạnh hơn nhưng chậm hơn |
| **Average corrective phase retrace (Results Snapshot)** | **47% in 32 days** (bull) · **50% in 22 days** (bear) | |
| **Average last leg rise (Results Snapshot)** | **32% in 60 days** (bull) · **35% in 33 days** (bear) | Last leg SHORTER first leg: bull 32% vs 46% (first leg ~50% longer) |
| **% meeting price target (Results Snapshot)** | **45%** (bull) · **56%** (bear) | Measure rule yếu — dưới 60% |
| **% meeting time target (Results Snapshot)** | 38% (bull) · 56% (bear) | |
| **Failure rate / Average rise / BE failure rate** | **NOT APPLICABLE** (PDF từ chối) | |
| **Measure rule** | target = **corrective_phase_bottom + first_leg_rise** (project first leg lên từ đáy corrective phase). | Quote PDF p511: *"Using the first leg height, in dollars, projected upward from the bottom of the corrective phase to get a target price only works 45% to 56% of the time"* |
| **First vs Last leg (Table 33.2 text)** | Bull: first leg 46%/87d vs last leg 32%/60d (first leg ~50% longer cả % lẫn time) | First leg ≠ last leg cho MMU |

### Đối chiếu với `measured_move_down_up_digitized.json` (phần MMU)

| Trường digitized | Giá trị digitized | Giá trị PDF | Status |
|---|---|---|---|
| `success_rate_reaching_target` (áp dụng chung) | 72 | MMU: bull 45%, bear 56% | 🔴 **LỆCH LỚN** (72 vs 45-56) |
| `failure_rate.at_5pct` / `at_10pct` | 15 / 8 | NOT APPLICABLE | 🔴 **BỊA** |
| `phase1_phase3_ratio` | 0.85-1.15 | MMU bull: first 46% vs last 32% → last leg = 70% of first (ngoài 0.85-1.15) | 🔴 **SAI** — first leg ~50% LONGER than last; ratio thực ~0.70 |
| `average_completion_days` | 42 | MMU length bull 180d, bear 85d | 🔴 LỆCH (42 vs 85-180) |
| `target_calculation.method` | `phase1_distance` | `first_leg_projection` | 🟢 KHỚP method |
| `phase2_ideal_retracement_pct` | 50 | MMU avg 47% (bull)/50% (bear); most freq 40-60% | 🟢 KHỚP |

---

## Bằng chứng verbatim (số liệu thô, không copy câu dài — bản quyền)

### MMD — Results Snapshot (PDF p495 / sách p495)
```
Average first leg decline           27% in 61 days    36% in 45 days
Average corrective phase retrace    48% in 30 days    44% in 22 days
Average last leg decline            25% in 62 days    36% in 46 days
Percentage meeting price target     35%               39%
Percentage meeting time target      53%               49%
```

### MMD — Table 32.2 (PDF p504 / sách p503)
```
Number of formations                            647               264
Reversal (R), continuation (C)                  566 R,81 C        214 R,50 C
Change after trend ends                         46%               49%
Most frequent corrective phase retrace          40% to 60%        40% to 60%
Average MMD length                              153 days          113 days
Average first leg price decline                 27% in 61 days    36% in 45 days
Average corrective phase retrace                48% in 30 days    44% in 22 days
Average last leg price decline                  25% in 62 days    36% in 46 days
```

### MMD — KHÔNG có failure rate (PDF p496)
```
"I do not show the average decline or the failure rate. The average decline measures the
move from the breakout to the ultimate low, and neither applies to this pattern. The failure
rate is a measure of how far price moves after the breakout. That also does not apply."
```

### MMD — Measure rule (PDF p497)
```
"If you measure the first leg price decline and project it downward from the top of the
corrective phase, you will hit your target between 35% and 39% of the time. If you take
half of the first leg move and project downward, you will hit your target between 83% and
93% of the time."
```

### MMU — Results Snapshot (PDF p510 / sách p510)
```
Average first leg rise               46% in 87 days    39% in 30 days
Average corrective phase retrace     47% in 32 days    50% in 22 days
Average last leg rise                32% in 60 days    35% in 33 days
Percentage meeting price target      45%               56%
Percentage meeting time target       38%               56%
```

### MMU — Table 33.2 (PDF p515 / sách p514)
```
Number of formations                            577               233
Reversal (R), continuation (C)                  393 R,184 C       179 R,54 C
Change after trend ends                         -26%              -27%
Most frequent corrective phase retrace          40% to 60%        40% to 60%
Average MMU length                              180 days          85 days
Average first leg price rise                    46% in 87 days    39% in 30 days
Average corrective phase retrace                47% in 32 days    50% in 22 days
Average last leg price rise                     32% in 60 days    35% in 33 days
```

---

## So sánh MMD ↔ MMU

| Metric | MMD (Down) | MMU (Up) | Nhận xét |
|---|---|---|---|
| **Sample tổng** | 911 | 810 | MMD phổ biến hơn |
| **Reversal rate** | bull 87%, bear 81% | bull 68%, bear 77% | MMD reversal nhiều hơn |
| **Average first leg** | 27%/36% decline | 46%/39% rise | MMU first leg mạnh hơn (bull) |
| **Corrective retrace** | 48%/44% | 47%/50% | Gần bằng nhau (~nửa first leg) |
| **Average last leg** | 25%/36% decline | 32%/35% rise | MMU last leg ~70% first leg (bull); MMD last leg ~93% first leg |
| **% meeting target** | 35/39% | 45/56% | MMU đạt target tốt hơn nhưng đều yếu (<60%) |
| **Pattern length** | bull 153d, bear 113d | bull 180d, bear 85d | MMU bull dài nhất (6 tháng) |
| **Last leg vs first leg** | avg 19-20% SHORTER (dollars) | bull first ~50% longer than last | Cả 2: last leg ≠ first leg (lý thuyết = thực tế khác) |

**Kết luận so sánh:** Cả MMD và MMU đều có **measure rule yếu** (% target 35-56%) — Bulkowski khuyến nghị dùng "half of first leg" để đạt 83-93%. MMU có first leg mạnh hơn nhưng last leg yếu hơn rõ (không bằng first leg). Corrective phase retrace ổn định ~40-60% (Fibonacci). Pattern length dài (3-6 tháng) — không phải pattern ngắn hạn.

---

## Reproducer

```bash
PDF="/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet/Nghiên cứu mô hình nến/references/encyclopedia-of-chart-patterns-2nbsped-9786468600-3175723993-9780471668268-0471668265_compress.pdf"

# Measured Move Down (chương 32, PDF p518-531)
pdftotext -layout -f 518 -l 531 "$PDF" - | sed -n '495,500p'      # Results Snapshot
pdftotext -layout -f 518 -l 531 "$PDF" - | grep -A 18 "Table 32.2"
pdftotext -layout -f 518 -l 531 "$PDF" - | sed -n '496,498p'      # "I do not show failure rate"

# Measured Move Up (chương 33, PDF p532-543)
pdftotext -layout -f 532 -l 543 "$PDF" - | sed -n '510,516p'      # Results Snapshot
pdftotext -layout -f 532 -l 543 "$PDF" - | grep -A 16 "Table 33.2"
```

---

## Khuyến nghị (không thực hiện trong task này)

1. **XÓA failure rate (CRITICAL)**: digitized at_5pct=15/at_10pct=8 → **BỊA**. PDF từ chối metric này cho MMD/MMU. Cần xóa hoặc ghi rõ "NOT APPLICABLE — MMD/MMU không có breakout đơn, không dùng failure rate".
2. **Sửa success_rate_reaching_target (CRITICAL)**: 72 → MMD 35/39%, MMU 45/56%. Digitized cao gấp ~2 lần. Hoặc tách thành 2 metric: "full first leg projection" (35-56%) vs "half first leg projection" (83-93%).
3. **Sửa ultimate_high/low_method (SAI SEMANTIC)**: MMD/MMU không dùng ultimate method. Đổi thành "pattern_length_method" hoặc xóa.
4. **Sửa phase1_phase3_ratio**: digitized 0.85-1.15 (phase3 ≈ phase1 ±15%) → **SAI**. PDF: MMD last leg SHORTER 19-20% (dollars), ratio thực ~0.80-0.81; MMU bull last leg = 70% first leg (ratio 0.70). Cả 2 ngoài khoảng 0.85-1.15.
5. **Sửa lookahead/completion days**: 21-63 → MMD 113-153d, MMU 85-180d (whole pattern). Digitized thấp 2-4 lần.
6. **Bổ sung sample**: MMD 911, MMU 810, tổng 1.721.
7. **Bổ sung 3-leg breakdown**: first leg / corrective retrace / last leg (cả % lẫn days) — đây là metric chính của MMD/MMU thay vì average rise/decline đơn lẻ.
8. **Bổ sung % meeting time target** (MMD 53/49%, MMU 38/56%).
9. **Thêm khuyến nghị "half first leg"**: để đạt target 83-93% thay vì full first leg 35-56%.
10. **Giữ nguyên** target method (`phase1_distance`) và phase2 retracement (50%) — 2 trường này ĐÚNG.

---

**Hết file.**
