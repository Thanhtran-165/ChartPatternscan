# 03 — Chuẩn đo lường V3

> File kỹ thuật quan trọng nhất. Định nghĩa **chuẩn đo lường đúng** cho từng pattern: lookahead, failure, target, cỡ mẫu, chất lượng dữ liệu. Mọi detector + build profile + dashboard phải tuân thủ file này.

> Nguồn chuẩn: `extraction_phase_1/digitization/patterns_digitized/*_digitized.json` (31 family, đã trích từ Encyclopedia Bulkowski).

> ⚠️ **VÁ 12/08 (K3-1 phán quyết — mục này THAY THẾ toàn bộ bảng §1.1–§1.2 bên dưới):**
> **Nguồn chuẩn lookahead DUY NHẤT từ M1 = `scanner/v2/measurement_registry.py`** (K3-1 PASS).
> Thứ tự ưu tiên: PDF_REVIEW (12 family, source=pdf) > digitized `average_days` (days to ultimate — source=digitized) > `lookahead_bars` spec > detector_legacy (dead_cat=63).
> Các bảng §1.1/§1.2 dưới đây là **bản dự thảo cũ** (ghi lookahead_bars/default 252 — bị kế hoạch 08 §2 "bỏ default 252" thay thế). Mọi con số ở đây CHỈ để tham chiếu lịch sử; số thực thi lấy từ registry. Family chưa có PDF giữ `source=digitized`, M5 nâng thành `source=pdf`.
> `spec_la` trong §1.3 = **giá trị registry** (không phải bảng §1.2).

---

## 1. Bảng lookahead chuẩn (31 family digitized)

### 1.1. Bảng chính

| Family digitized | `lookahead_bars` (spec) | Detector V2 hiện tại | Chênh | Hướng sửa |
|---|---:|---:|---|---|
| inside_day | **10** | 60 | 6.0× | giảm về 10 |
| rising_falling_three_methods | **20** | ? | — | xác nhận = 20 |
| spike_formation | **20** | (không detector) | — | pattern chưa scan |
| horn_bottoms_tops | **42** | 120 (qua pipes) | 2.9× | giảm về 42 |
| islands | **42** | ? | — | xác nhận = 42 |
| flags | **63** | ? (bull_flags source-grounded) | — | xác nhận = 63 |
| pennants | **63** | ? | — | xác nhận = 63 |
| gaps | **63** | ? | — | xác nhận = 63 |
| measured_move_down_up | **63** | ? | — | xác nhận = 63 |
| pipe_bottoms | **63** | 120 (pipes mặc định) | 1.9× | giảm về 63 |
| triangles | **126** | ? | — | xác nhận = 126 |
| wedges_ascending_descending | **126** | ? | — | xác nhận = 126 |
| broadening_bottoms | **252** | ? | — | xác nhận = 252 |
| broadening_tops | **252** | ? | — | xác nhận = 252 |
| broadening_formations_right_angled_* (asc/desc) | **252** | ? | — | xác nhận = 252 |
| broadening_wedges | **252** | ? | — | xác nhận = 252 |
| bump_and_run_reversal | **252** | ? | — | xác nhận = 252 |
| cup_with_handle | **252** | ? | — | xác nhận = 252 |
| diamond_bottom / diamond_top | **252** | ? | — | xác nhận = 252 |
| double_bottoms / double_tops | **252** | ? | — | xác nhận = 252 |
| head_and_shoulders_bottom / top | **252** | ? | — | xác nhận = 252 |
| rectangle_bottoms_tops | **252** | ? | — | xác nhận = 252 |
| rounding_bottoms_tops | **252** | ? | — | xác nhận = 252 |
| scallop_ascending_descending | **252** | 120 (qua pipes) | 2.1× | tăng lên 252 |
| three_falling_peaks / three_rising_valleys | **252** | ? | — | xác nhận = 252 |
| triple_bottoms_tops | **252** | ? | — | xác nhận = 252 |

**Kết luận lookahead:** 3 nhóm rõ rệt —
- **Nhóm ngắn** (10–20): inside_day, three_methods, spike — pattern nến ngắn hạn.
- **Nhóm trung** (42–63): horn, island, flag, pennant, gap, measured_move, pipe — pattern tuần/trung hạn.
- **Nhóm dài** (126–252): triangle, wedge, broadening, bump, cup, diamond, double, HSB/HST, rectangle, rounding, scallop, three_peaks/valleys, triple — pattern đảo chiều cấu trúc lớn.

### 1.2. Mapping 55 pattern_key (EVENT_SOURCES) → 31 family → lookahead

Pattern_key trên artifact web đa số là biến thể của 31 family. Mapping lookahead:

| Pattern_key (artifact) | Family digitized | Lookahead |
|---|---|---:|
| inside_day | inside_day | 10 |
| rising_three_methods, falling_three_methods | rising_falling_three_methods | 20 |
| horn_bottoms, horn_tops | horn_bottoms_tops | 42 |
| island_reversals, islands_long | islands | 42 |
| bull_flags, bear_flags | flags | 63 |
| bull_pennants, bear_pennants | pennants | 63 |
| area_gaps, breakaway_gaps, continuation_gaps, exhaustion_gaps | gaps | 63 |
| measured_move_up, measured_move_down | measured_move_down_up | 63 |
| pipe_bottoms, pipe_tops | pipe_bottoms | 63 |
| triangles_ascending, triangles_descending, triangles_symmetrical | triangles | 126 |
| wedges_falling, wedges_rising | wedges_ascending_descending | 126 |
| broadening_bottoms, broadening_tops | broadening_bottoms / broadening_tops | 252 |
| broadening_formations_right_angled_ascending/descending | broadening_formations_right_angled | 252 |
| broadening_wedges_ascending/descending | broadening_wedges | 252 |
| bump_and_run_reversal_bottoms/tops | bump_and_run_reversal | 252 |
| cup_with_handle, cup_with_handle_inverted | cup_with_handle | 252 |
| diamond_bottoms, diamond_tops | diamond_bottom / diamond_top | 252 |
| double_bottoms_{AA,AE,EA,EE}, double_tops_{AA,AE,EA,EE} | double_bottoms / double_tops | 252 |
| head_and_shoulders_bottoms(+_complex) | head_and_shoulders_bottom | 252 |
| head_and_shoulders_tops(+_complex) | head_and_shoulders_top | 252 |
| rectangle_bottoms, rectangle_tops | rectangle_bottoms_tops | 252 |
| rounding_bottoms, rounding_tops | rounding_bottoms_tops | 252 |
| scallops_ascending(_inverted), scallops_descending(_inverted) | scallop_ascending_descending | 252 |
| three_falling_peaks, three_rising_valleys | three_falling_peaks / three_rising_valleys | 252 |
| triple_tops, triple_bottoms | triple_bottoms_tops | 252 |
| dead_cat_bounce(_inverted) | (không có spec riêng) | **cần M5 đọc PDF** |
| high_tight_flags | flags (gần) hoặc riêng | **xác nhận M5** |

### 1.3. Nghiệm thu lookahead (M1)

Sau khi sửa, kiểm chứng:
```
cho mỗi pattern_key:
    spec_la  = lookahead từ bảng §1.2
    actual   = median(events.csv[evaluated_bars])
    assert abs(actual - spec_la) / spec_la <= 0.05   # chênh ≤5%
```

---

## 2. Định nghĩa failure ĐÚNG chuẩn Bulkowski

### 2.1. Failure hiện tại (SAI)

`scanner/v2/pipes.py:392`:
```python
failure_5pct: bool(float(mfe) < 5.0)
```
→ "MFE dưới 5%" = move không đạt 5% theo hướng có lợi. **Đây KHÔNG phải failure chuẩn Bulkowski.**

### 2.2. Failure chuẩn Bulkowski (busted)

Bulkowski "busted" = **giá vượt lại đáy/đỉnh pattern (hoặc một ngưỡng gần đó) TRƯỚC khi chạm target**. Cần path rows (chuỗi giá sau breakout), không phải chỉ MFE cuối.

Công thức chung (pseudo):
```
failure_busted = tồn tại bar sau breakout mà:
    (direction == up  AND  low  <= breakout_level_failure)   # giá quay lại đáy pattern
    (direction == down AND  high >= breakout_level_failure)  # giá quay lại đỉnh pattern
    VÀ điều đó xảy ra TRƯỚC khi chạm target
```

`breakout_level_failure` per-pattern = đường đáy/đỉnh pattern × (1 ± threshold).

### 2.3. Bảng failure per-pattern (từ digitized spec)

| Family | `failure_threshold_pct` | `failure_definition` (spec) | failure_rate (spec) |
|---|---:|---|---|
| inside_day | **1.0** | giá quay lại trong range bar mẹ sau breakout | at_3pct=25, at_5pct=15 |
| rising_falling_three_methods | **2.0** | giá quay lại trong range bar đầu sau breakout | at_5pct=20 |
| horn_bottoms_tops | **3.0** | giá vượt qua đỉnh horn đối diện | at_5pct=15 |
| islands | **2.0** | giá quay lại qua island level | at_5pct=18 |
| pipe_bottoms | **3.0** | giá đóng dưới pipe bottoms sau breakout lên | at_5pct=12 |
| spike_formation | **3.0** | giá tiếp tục qua spike extreme thêm 3% | at_5pct=35 |
| flags | **5.0** | giá đi ngược flagpole 5% sau breakout | bull=5, bear=6, TB=5.5 |
| pennants | **5.0** | giá đi ngược flagpole 5% | bull=6, bear=7, TB=6.5 |
| gaps | **varies** | breakaway fill = failure; common fill = không | breakaway=15, continuation=20, exhaustion=10 |
| measured_move_down_up | **5.0** | Phase 3 không chạm target hoặc đảo sớm | at_5pct=15 |
| triangles | **5.0** | giá đi ngược breakout 5% | asc=8, desc=9, sym=11 |
| wedges | **5.0** | giá đi ngược breakout 5% | rising=11, falling=11 |
| broadening_bottoms | **5.0** | giá đóng dưới đáy thấp nhất sau breakout lên | overall=10 |
| broadening_tops | **5.0** | giá đóng trên đỉnh cao nhất sau breakout xuống | overall=9 |
| bump_and_run | **5.0** | giá đóng trên bump peak sau khi run phase bắt đầu | at_5pct=10 |
| cup_with_handle | **5.0** | giá giảm dưới handle low sau breakout lên | overall=5 |
| diamond_bottom | **5.0** | giá đóng dưới diamond apex sau breakout lên | at_5pct=5 |
| diamond_top | **5.0** | giá đóng trên diamond apex sau breakout xuống | at_5pct=7 |
| double_bottoms | **5.0** | giá đóng dưới peak (neckline) sau breakout lên | at_5pct=7 |
| double_tops | **5.0** | giá đóng trên trough (neckline) sau breakout xuống | at_5pct=8 |
| HSB | **5.0** | giá đóng dưới vai phải sau breakout lên | at_5pct=6 |
| HST | **5.0** | giá đóng trên vai phải sau breakout xuống | at_5pct=8 |
| rectangle | **5.0** | giá quay lại trong rectangle sau breakout | bottom=5, top=10 |
| rounding | **5.0** | giá đi ngược breakout 5% | — |
| scallop | **5.0** | giá đi ngược breakout 5% | — |
| **broadening_right_angled_*, broadening_wedges, three_peaks/valleys, triple** | **chưa có trong digitized** | **cần M5 đọc PDF** | — |
| dead_cat_bounce, high_tight_flags | **chưa có spec riêng** | **cần M5** | — |

### 2.4. Đề xuất triển khai

1. Detector thêm hàm `_compute_failure_busted(path_rows, breakout_level, direction, threshold_pct, target_price)`:
   - duyệt path_rows theo thứ tự;
   - tại mỗi bar, check `low <= fail_level` (up) hoặc `high >= fail_level` (down);
   - nếu failure xảy ra **trước** bar đầu tiên đạt target → `failure_busted = True`;
   - nếu target chạm trước → `failure_busted = False`.
2. `threshold_pct` + `breakout_level` từ registry per-pattern (§2.3).
3. Giữ `failure_5pct` (MFE<5%) nhưng **đổi tên `weak_move_5pct`** — còn ý nghĩa "move yếu" không phải "thất bại".
4. events.csv thêm: `failure_busted`, `failure_threshold_pct`, `weak_move_5pct`.
5. Build profile + dashboard dùng `failure_busted_rate` làm failure chính.

### 2.5. Nghiệm thu failure (M2) — CHUẨN LẠI THEO K3-2 (12/08/2026)

> ⚠️ Bản gốc của mục này đặt mốc tuyệt đối (bull_flags ≈5,5%, inside_day ≈15%, horn ≈15%, cup ≈5%)
> lấy từ `failure_rate` spec — nhưng số đó là **Break-even failure** (giá không đạt +5% theo hướng
> breakout / quay về breakout), KHÔNG phải `failure_busted` (quay lại đáy/đỉnh pattern). → So SAI ĐƠN VỊ.
> K3-2 (agent_f1f52d0b, 12/08) phán quyết: bỏ mốc tuyệt đối; đối chiếu đúng đơn vị; đóng băng baseline VN.

**Đối chiếu theo đúng đơn vị:**

1. `weak_move_5pct` (MFE<5%) ↔ **Break-even failure spec** (số duy nhất trong sách có bảng %).
   Dung sai mềm (VN khác US): chênh ≤3× coi là đặc thù thị trường; chênh >5× → audit.
   Ghi chú riêng: flags/pennants VN yếu hơn US nhiều (weak 43% vs BE 5%) — đặc thù T+ và biên độ VN.
2. `failure_busted` ↔ **baseline VN v1** (bảng đóng băng 12/08/2026, 1599 mã, nguồn 09-m2 §3)
   + nhận định định tính: busted hiếm ở US (mô tả định tính, không có bảng %); VN cao hơn
   do cấu trúc thị trường (penny thanh khoản kém, khung đo dài 167d với threshold sát breakout).
3. `inside_day`: **bỏ hẳn mọi mốc %** — spec lệch định nghĩa (Harami body vs range, PDF_REVIEW §5.1),
   không có căn cứ so hợp lệ; chỉ dùng baseline VN.
4. CẤM chỉnh threshold/khung đo để ép số về US (curve-fitting — phá mục đích khám phá số thật VN).

---

## 3. Target — phải kèm độ lớn mục tiêu

### 3.1. Vấn đề

Dashboard hiện show `target_hit_rate_pct` đơn lẻ. Nhưng `target_dist_pct` (độ lớn mục tiêu) chênh lớn giữa pattern: inside_day median 2.32%, bull_flags 16.3%, pipe 15.5%. → "88% hit" với target 2.3% ≠ "88% hit" với target 16%.

### 3.2. Bảng target method per-pattern (từ spec)

| Family | `target_method` | Ý nghĩa |
|---|---|---|
| inside_day | breakout_magnitude | target = breakout + prior_trend_strength × breakout_magnitude |
| flags, pennants | flagpole_addition | target = breakout + flagpole height |
| pipe_bottoms | prior_drop_height | target = breakout + chiều cao nhịp giảm trước pipe |
| gaps | varies_by_type | mỗi loại gap target khác nhau |
| measured_move | phase1_distance | target = khoảng cách Phase 1 |
| islands | gap_height | target = chiều cao gap |
| rising_three_methods | continuation_magnitude | target = biên độ tiếp diễn |
| spike | reversal_magnitude | target = biên độ đảo chiều |
| **phần lớn reversal** (triangle, wedge, broadening, bump, cup, diamond, double, HSB/HST, rectangle, rounding, scallop, horn) | **pattern_height** | target = breakout + chiều cao pattern (full-height) |
| cup_with_handle | cup_height | target = breakout + chiều cao cup (không tính handle) |

### 3.3. Đề xuất

1. Build profile thêm `median_target_dist_pct`, `p25_target_dist_pct`, `p75_target_dist_pct`.
2. Dashboard hiển thị: "đạt mục tiêu 88% (mục tiêu TB 2.3%, khoảng 1.5–3.8%)".
3. **Cảnh báo nổi** nếu `median_target_dist_pct < 3%`: "mục tiêu quá nhỏ — hit rate chỉ tham khảo, không có ý nghĩa thực chiến".
4. Khi so sánh pattern, **luôn chuẩn hoá** bằng cách show cả target_dist + hit_rate, không so hit_rate không.

### 3.4. Nghiệm thu target (M1/M3)

- Mỗi event trong events.csv có `target_dist_pct` không null.
- Mỗi profile có `median_target_dist_pct`.
- Dashboard có dòng cảnh báo cho pattern target quá nhỏ.

---

## 4. Cỡ mẫu tối thiểu đáng tin

### 4.1. Ngưỡng (triết lý #11 — ngưỡng số dưới bảng)

| Cỡ mẫu (n events/mã/pattern) | Nhãn | Hành vi |
|---:|---|---|
| n ≥ 30 | "đủ mẫu" | hiển thị đầy đủ |
| 10 ≤ n < 30 | "cỡ mẫu vừa" | hiển thị + nhãn vàng |
| 5 ≤ n < 10 | "mẫu mỏng" | hiển thị + nhãn cam + không đưa vào "best_historical" |
| n < 5 | "quá mỏng" | **ẩn khỏi profile mặc định**, chỉ show khi toggle |

> Lưu ý: ngưỡng 30 cho **hồ sơ 1 mã**. Ngưỡng cho **thống kê toàn thị trường** (publication chapter) cao hơn — xem file 04 §2.

### 4.2. Xử lý cap `max_events_per_symbol`

- Thêm `hit_cap: bool` (= n ≥ cap) vào profile.
- Khi `hit_cap=true`: `frequency_score` đánh dấu "≈cap" không cho điểm tuyệt đối.
- Dashboard chú thích: "n bị giới hạn ở {cap}, tần suất thật có thể cao hơn".
- Đề xuất cap mới cho pattern hiếm (bump_and_run 10→20) nếu performance cho phép — task riêng, không chặn V3.

---

## 5. Chất lượng dữ liệu

### 5.1. Filter dữ liệu bẩn MAE>80% (split chưa điều chỉnh)

**Quy tắc:** event có `mae_pct > 80` → gần như chắc chắn do corporate action (split/share dividend) chưa adjust trong DB → **loại khỏi stats** + đánh dấu.

**Triển khai ở build profile:**
```python
# pseudo
SUSPECT_MAE_PCT = 80
clean = events[events["mae"].fillna(0) <= SUSPECT_MAE_PCT]
suspect = events[events["mae"].fillna(0) > SUSPECT_MAE_PCT]
metadata["events_dropped_split_suspect"] = len(suspect)
metadata["drop_rate_pct"] = round(len(suspect) / max(len(events),1) * 100, 2)
# profile tính trên clean
```

**Nghiệm thu:** report `events_dropped_split_suspect` cho mỗi pattern; inside_day kỳ vọng ~277/9847 ≈ 2.8%.

### 5.2. Survivorship / delisting

- **Không xóa** mã delisted (dữ liệu thật).
- Metadata ghi `delisted_symbols` (last_date < 2024), `delisted_rate_pct`.
- Dashboard: toggle "ẩn mã delisted" (mặc định hiện) + cảnh báo non-advice: "thống kê bao gồm N mã đã ngừng giao dịch, thiên lệch nhẹ về phía mã sống sót".
- Lớp sâu hơn (ngoài V3): rà corporate action DB + đối chiếu — đề xuất task riêng.

### 5.3. Các flag chất lượng khác (đã có, giữ nguyên)

- `path_quality_bucket`: clean / usable / loose / short_path / zero_and_stale / zero_volume / mixed_flag.
- `tradability_quality_bucket`: clean / usable / impaired.
- `missing_bar_rate_60d`, `zero_volume_rate_60d`, `price_limit_proxy_rate_60d` — đã đo trong detector.
- Build profile đã có `data_limited` tier — giữ.

### 5.4. `data_quality_flag` tổng hợp mới (đề xuất)

Mỗi event có 1 flag tổng hợp:
| Flag | Điều kiện | Ý nghĩa |
|---|---|---|
| `ok` | mae ≤ 80 + path_quality ∈ {clean,usable} + not delisted | sạch, dùng cho stats |
| `suspect_split` | mae > 80 | likely split chưa adjust → loại khỏi stats |
| `delisted` | symbol trong delisted list | giữ nhưng minh bạch |
| `data_limited` | path_quality ∈ data_limited set | giữ nhưng hạ trọng số |

---

## 6. Tóm tắt — checklist chuẩn đo lường V3

Một pattern "đo đúng chuẩn" khi:

- [ ] Lookahead detector = lookahead spec (§1, chênh ≤5%)
- [ ] `failure_busted` tính từ path rows theo định nghĩa spec (§2)
- [ ] `weak_move_5pct` (đổi tên từ failure_5pct) giữ làm phụ, không làm failure chính
- [ ] Mọi event có `target_dist_pct`, mọi profile có `median_target_dist_pct` (§3)
- [ ] Cỡ mẫu có nhãn (≥30 đủ / 10–30 vừa / 5–10 mỏng / <5 ẩn) (§4)
- [ ] `hit_cap` flag khi n chạm max_events_per_symbol (§4.2)
- [ ] Event MAE>80% bị loại + report (§5.1)
- [ ] `data_quality_flag` tổng hợp trên mỗi event (§5.4)
- [ ] Metadata ghi delisted + drop_rate (§5.1, §5.2)

Khi checklist trên PASS cho 1 pattern → pattern đó đủ điều kiện kỹ thuật (K1–K6 ở file 01 §4.1) để xét nấc độ tin cậy.
