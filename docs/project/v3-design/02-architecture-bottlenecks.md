# 02 — Điểm nghẽn hiện tại + kiến trúc pipeline V3

> Đọc kèm `00-overview.md` và `03-measurement-standards.md`. File này liệt kê 9 điểm nghẽn (mỗi điểm có: triệu chứng, nguyên nhân, bằng chứng file:line, giải pháp thiết kế) rồi vẽ kiến trúc pipeline mới + cơ chế tự động hoá.

---

## 1. Pipeline hiện tại (đã truy ngược — xác minh 12/08/2026)

```
[PDF gốc Bulkowski]                                    ← chưa đọc trực tiếp (bản quyền)
        ↓ (trích thủ công)
[digitized spec: extraction_phase_1/digitization/.../*_digitized.json]   (31 family)
        ↓ (người triển khai hardcode 1 vài thông số)
[detector: scanner/v2/<family>.py]
   ├─ inside_days.py: _evaluate_detection(df,row,lookahead=60)   ← HARDCODE 60
   ├─ pipes.py:       _evaluate_detection(..., lookahead=120)     ← MẶC ĐỊNH 120
   ├─ horns.py, scallops.py: import _evaluate_detection từ pipes  ← kế thừa 120
   └─ ... (24 detector)
        ↓
[events.csv: artifacts/scanner_v2/<family>/<pattern>/db_active/events.csv]
        ↓ (EVENT_SOURCES trong rebuild_source_guided_final_chapters.py:73-129 — 55 key)
[build_stock_pattern_profiles.py: scripts/]   ← tổng hợp read-only, công thức score ở :434-475
        ↓
[web/stock_pattern_profiles.json]
        ↓ (CHẠY TAY: split_stock_history_artifacts.mjs)
[web/profiles/<symbol>.json, web/personality/<symbol>.json, web/setups/<symbol>.json]
        ↓
[dashboard: market_stats/web/stock_history_pattern_module.js]
```

**Vấn đề cấu trúc:** thông số chuẩn (lookahead, failure threshold, target method) bị **mất/không truyền** ở 2 nút: (a) detector không đọc spec, (b) build profile không biết lookahead của từng event → không thể cảnh báo.

---

## 2. 9 điểm nghẽn + giải pháp thiết kế

### Điểm nghẽn 1 — Lookahead lệch chuẩn (HỆ THỐNG, nghiêm trọng nhất)

**Triệu chứng:** inside_day median MFE = 15.2% trên dashboard, trong khi Encyclopedia Bulkowski nói inside_day move chỉ ~3%. Scallop/cup_with_handle đo 120 phiên (chưa hết move, chuẩn 252).

**Nguyên nhân:**
- `scanner/v2/inside_days.py:231` → `_evaluate_detection(df, row, lookahead=60)`
- `scanner/v2/pipes.py:344` → `def _evaluate_detection(df, detection, *, lookahead=120)` (mặc định)
- `scanner/v2/horns.py`, `scallops.py` → import `_evaluate_detection` từ pipes → kế thừa 120
- Digitized spec có `post_breakout_measurement.lookahead_bars` đa dạng: inside_day=10, scallop=252, cup=252, horn=42, pipe=63, flag=63, pennant=63, gap=63, triangle=126, wedge=126, three_methods=20, island=42, spike=20, phần lớn reversal=252 (xem bảng đầy đủ file 03 §1).

**Hậu quả:**
- inside_day lệch 6× (60 vs 10) → MFE phồng bởi drift 3 tháng.
- scallop/cup/HSB lệch ~2× (120 vs 252) → đo chưa hết move dài hạn.
- horn lệch ~3× (120 vs 42) → ngược lại, đo thỡ, pha tạp nhiễu sau khi pattern đã失效.
- pipe/gap/flag lệch ~2×.

**Giải pháp thiết kế (Lớp A đang thi công):**
1. Detector đọc `lookahead_bars` từ digitized spec tại lúc khởi tạo (không hardcode).
2. Thêm bảng mapping `pattern_key → spec_path → lookahead_bars` ở 1 nơi trung tâm (ví dụ `scanner/v2/lookahead_registry.py`) để mọi detector cùng dùng.
3. `_evaluate_detection` nhận `lookahead` từ detector config, không từ default.
4. Ghi `expected_lookahead` + `actual_evaluated_bars` vào events.csv để audit.

**Nghiệm thu điểm này:** bảng đối chiếu 24 detector: `pattern | spec_lookahead | detector_lookahead_before | detector_lookahead_after | chênh %`. inside_day phải về 10, scallop về 252.

---

### Điểm nghẽn 2 — Định nghĩa failure SAI chuẩn Bulkowski

**Triệu chứng:** bull_flags `failure_5_rate = 25.4%` trên dashboard, trong khi Bulkowski báo ~5.5% (gấp 4.6×).

**Nguyên nhân:**
- `scanner/v2/pipes.py:392` → `failure_5pct: bool(float(mfe) < 5.0)` — tức **"MFE dưới 5%"** (move không đạt 5% theo hướng có lợi).
- Chuẩn Bulkowski (từ digitized spec):
  - **inside_day:** "price returns inside yesterday's range after breakout", threshold 1.0%
  - **cup_with_handle:** "price drops below handle low after upward breakout", threshold 5.0%
  - **scallop:** "price moves 5% against breakout", threshold 5.0%
  - **horn:** "price returns past opposite horn peak", threshold 3.0%
  - **flags:** "price moves 5% against flagpole direction after breakout", threshold 5.0%
  - **pipe:** "price closes below pipe bottoms after upward breakout", threshold 3.0%
- Bulkowski "busted" thật sự = **giá vượt lại đáy/đỉnh pattern TRƯỚC khi chạm target** — cần path rows (chuỗi giá sau breakout), không phải chỉ MFE.

**Hậu quả:** failure rate trên dashboard không có ý nghĩa so sánh với literature. Người đọc hiểu "25% thất bại" theo nghĩa "25% move dưới 5%", nhưng Bulkowski nói "5.5% busted" theo nghĩa hoàn toàn khác.

**Giải pháp thiết kế:**
1. Thêm trường `failure_busted` (theo chuẩn Bulkowski) tính từ path rows: giá có chạm lại đường đáy/đỉnh pattern (± threshold) trước khi chạm target không.
2. Giữ `failure_5pct` (MFE<5%) nhưng **đổi tên** thành `weak_move_5pct` để tránh nhầm.
3. Dashboard hiển thị `failure_busted` làm failure chính, `weak_move_5pct` làm phụ.
4. Threshold per-pattern từ spec (1/3/5%) — không dùng 5% chung.

**Nghiệm thu:** audit code detector không còn dòng `failure = mfe < 5` cho pattern có spec failure định nghĩa khác.

---

### Điểm nghẽn 3 — target_hit không kèm độ lớn mục tiêu

**Triệu chứng:** inside_day `target_hit_rate = 88%` nghe rất ấn tượng, nhưng `target_dist_pct` median chỉ 2.32% → "đạt mục tiêu" ở đây = đạt 1 mục tiêu rất nhỏ. bull_flags target_dist 16.3%, pipe 15.5% → cùng "88% hit" nhưng ý nghĩa hoàn toàn khác.

**Nguyên nhân:**
- `build_stock_pattern_profiles.py:468` → chỉ xuất `target_hit_rate_pct`, không kèm `target_dist_pct`.
- `stock_history_pattern_module.js:44` → hiển thị "đạt mục tiêu X%" không có độ lớn.

**Giải pháp thiết kế:**
1. Build profile thêm `median_target_dist_pct` (đã có trong detector, chỉ cần truyền lên).
2. Dashboard hiển thị: "đạt mục tiêu 88% (mục tiêu TB 2.3%)" thay vì chỉ "88%".
3. Cảnh báo nếu `median_target_dist_pct < 3%`: "mục tiêu quá nhỏ, hit rate không có ý nghĩa thực chiến".

---

### Điểm nghẽn 4 — Cap `max_events_per_symbol` bão hoà

**Triệu chứng:** CTD inside_day n=12 = **đúng bằng cap** (`max_events_per_symbol=12`, `inside_days.py:69`). Tương tự bump_and_run=10. → frequency_score đạt 100 không vì mã "phát sinh nhiều pattern" mà vì **chạm trần**.

**Nguyên nhân:**
- inside_days.py:69 → `max_events_per_symbol: int = 12`
- scallops.py:83 → 14, pipes.py:68 → 18, horns.py:81 → 18, bump → 10.
- Detector `break` khi `len(rows) >= max_events` (inside_days.py:211-212).

**Hậu quả:**
- Tần suất thật bị che — 2 mã đều n=12 nhưng 1 mã có thể phát sinh 50 event, mã kia 12 event.
- `frequency_score = clamp(n/10*100)` (build.py:441) → bão hoà ở n=10+ → mọi mã "tần suất 100%".
- "Mẫu thường gặp" trên dashboard sai lệch.

**Giải pháp thiết kế:**
1. **Không bỏ cap** (cap cần thiết để tránh 1 mã độc chiếm artifact), nhưng **minh bạch**: thêm cột `hit_cap: bool` (= n ≥ cap).
2. `frequency_score` mới: khi `hit_cap=true`, đánh dấu "≈cap" không cho điểm tuyệt đối; hoặc tính frequency theo **tỉ lệ event/năm** (dựa trên历史 data range) thay vì raw count.
3. Dashboard: cạnh "Mẫu thường gặp" thêm chú thích "n bị giới hạn ở {cap}, tần suất thật có thể cao hơn".
4. Xét tăng cap cho pattern hiếm (bump_and_run=10 → có thể 20) khi performance cho phép.

---

### Điểm nghẽn 5 — Dữ liệu bẩn MAE>80% (split chưa điều chỉnh)

**Triệu chứng:** 277/9847 inside_day event có `mae_pct > 80%` — giá giảm >80% sau breakout, gần như bất khả thi về mặt kinh tế (trừ khi công ty phá sản). Nguyên nhân khả nghi: **corporate action (split/share dividend) chưa được điều chỉnh trong OHLCV DB** → giá "nhảy" tạo MAE ảo.

**Nguyên nhân:**
- `_enrich_events_from_series` trong `scanner/run_bear_flag_db_source_parity_audit.py` đọc series từ DB; nếu DB có giá chưa split-adjusted → path rows có giá "rơi vỡ".
- `build_stock_pattern_profiles.py` **không filter** event có MAE>80%.

**Hậu quả:** median MAE bị kéo lên, outcome_score sai, "caution_patterns" sai lệch.

**Giải pháp thiết kế:**
1. **Filter ở build profile** (không sửa DB): drop event `mae_pct > 80` + đánh dấu `data_quality_flag=suspect_split`.
2. **Report minh bạch:** metadata ghi `events_dropped_split_suspect: N`, `drop_rate_pct`.
3. **Cảnh báo theo mã:** nếu 1 mã có >3 event suspect → flag mã đó "dữ liệu cần rà corporate action".
4. Lớp sâu hơn (ngoài V3): rà DB OHLCV + đối chiếu với bảng corporate action (chia cổ tức, chia tách) — đề xuất task riêng.

---

### Điểm nghẽn 6 — Stale split artifact (chạy tay)

**Triệu chứng:** sau khi scan lại detector + build profile, phải **chạy tay** `node split_stock_history_artifacts.mjs` mới cập nhật `web/profiles/<symbol>.json`. Quên chạy → dashboard hiển thị dữ liệu cũ (stale).

**Nguyên nhân:**
- `split_stock_history_artifacts.mjs` là script độc lập, không hook vào pipeline.
- Không có timestamp check (dashboard không biết profile cũ bao nhiêu ngày).

**Giải pháp thiết kế:**
1. Tự động hoá (xem §4 dưới).
2. Thêm `profile_generated_at` + `scan_generated_at` vào metadata; dashboard cảnh báo "dữ liệu >7 ngày, chạy refresh" nếu stale.

---

### Điểm nghẽn 7 — Publication trộn draft/final

**Triệu chứng:** ~41/55 pattern trên artifact là DRAFT (không có publication status trong manifest) nhưng dashboard hiển thị như đã kiểm định.

**Nguyên nhân:**
- `build_stock_pattern_profiles.py` không đọc `pattern_family_manifest.json`.
- `stock_history_pattern_module.js` không có nhãn publication.

**Giải pháp thiết kế:** xem file `04-governance-roadmap.md` §1 (gắn publication_status vào profile + UI).

---

### Điểm nghẽn 8 — Survivorship (mã delisted)

**Triệu chứng:** 62/1715 mã trong DB không hoạt động (last_date < 2024). README nói chưa xử lý delisting tape.

**Hậu quả:** bias nhẹ về phía mã sống sót — các mẫu hình trên mã đã delisted (thường là mã yếu) bị thiếu → thống kê thiên về mã "còn sống".

**Giải pháp thiết kế:**
1. **Không xóa mã delisted** (chúng là dữ liệu thật), nhưng **minh bạch**: metadata ghi `delisted_symbols: N`, `delisted_rate_pct`.
2. Thêm toggle dashboard "ẩn mã delisted" (mặc định hiện).
3. Cảnh báo trong non-advice boundary: "thống kê bao gồm N mã đã ngừng giao dịch, có thể thiên lệch về phía mã sống sót".

---

### Điểm nghẽn 9 — Pipeline thủ công, không tái lập

**Triệu chứng:** để refresh dashboard, phải chạy tuần tự: detector → build profile → split. Quên 1 bước = dữ liệu lỗi thời hoặc mâu thuẫn.

**Giải pháp thiết kế:** xem §4 dưới (1 lệnh `make pattern-refresh` hoặc `scripts/refresh_pattern_pipeline.sh`).

---

## 3. Kiến trúc pipeline V3 (đề xuất)

```
┌─────────────────────────────────────────────────────────────────────┐
│  TẦNG NGUỒN (không đổi)                                             │
│  references/*.pdf (Bulkowski) — chưa đọc trực tiếp                  │
│         ↓                                                           │
│  extraction_phase_1/digitization/.../*_digitized.json (31 family)   │
│  + scanner/v2/pattern_family_manifest.json (publication status)     │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TẦNG ĐIỀU CHỈNH V3 (MỚI)                                           │
│  scanner/v2/lookahead_registry.py                                   │
│    - map pattern_key → {spec_path, lookahead_bars,                  │
│       failure_threshold, failure_definition, target_method}         │
│    - 1 nguồn duy nhất, mọi detector + build cùng đọc                │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TẦNG DETECTOR V3 (sửa, không viết lại)                             │
│  scanner/v2/<family>.py                                             │
│    - đọc lookahead/failure từ registry (KHÔNG hardcode)             │
│    - thêm failure_busted (từ path rows, chuẩn Bulkowski)            │
│    - giữ failure_5pct → đổi tên weak_move_5pct                      │
│    - events.csv thêm: expected_lookahead, actual_evaluated_bars,    │
│      target_dist_pct, failure_busted, data_quality_flag             │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TẦNG BUILD PROFILE V3 (sửa build_stock_pattern_profiles.py)        │
│    - filter MAE>80% (drop + report)                                 │
│    - gắn publication_status từ manifest                             │
│    - thêm median_target_dist_pct, failure_busted_rate               │
│    - thêm hit_cap flag, events_dropped_split_suspect                │
│    - thêm nấc độ tin cậy (Nấc 1/2/3 từ §4.2 file 01)                │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TẦNG AUTO-SPLIT V3 (MỚI — tự động, không chạy tay)                 │
│  scripts/refresh_pattern_pipeline.sh                                │
│    1. chạy build profile                                            │
│    2. chạy split_stock_history_artifacts.mjs                        │
│    3. ghi timestamp                                                │
│    4. (tuỳ chọn) reload dashboard                                   │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TẦNG HIỂN THỊ V3 (sửa stock_history_pattern_module.js)             │
│    - nhãn Nấc 1/2/3 + màu (đỏ/vàng/xanh)                            │
│    - target_hit kèm target_dist_pct                                 │
│    - cảnh báo cỡ mẫu, lookahead, dữ liệu bẩn                        │
│    - chú thích định nghĩa dưới mỗi bảng                             │
│    - ẩn Nấc 1 mặc định, toggle "hiện bản nháp"                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.1. Schema events.csv mới (đề xuất thêm cột)

| Cột mới | Kiểu | Nguồn | Ý nghĩa |
|---|---|---|---|
| `expected_lookahead` | int | registry | lookahead chuẩn từ spec |
| `actual_evaluated_bars` | int | detector (đã có `evaluated_bars`) | lookahead thực tế đo được |
| `target_dist_pct` | float | detector (đã có) | độ lớn mục tiêu — truyền lên profile |
| `failure_busted` | bool | detector mới | failure chuẩn Bulkowski |
| `weak_move_5pct` | bool | = failure_5pct cũ | MFE<5% (đổi tên cho rõ) |
| `failure_threshold_pct` | float | registry | threshold failure per-pattern (1/3/5%) |
| `data_quality_flag` | str | build | `ok` / `suspect_split` (MAE>80) / `delisted` |
| `publication_status` | str | manifest | draft/candidate/final |
| `hit_cap` | bool | detector | n có chạm max_events_per_symbol không |

---

## 4. Cơ chế tự động hoá (giải quyết điểm nghẽn 6 + 9)

### 4.1. 1 lệnh refresh toàn pipeline

Đề xuất `scripts/refresh_pattern_pipeline.sh` (hoặc Makefile target `pattern-refresh`):

```bash
# pseudo-code (KHÔNG viết file trong V3 design — chỉ đặc tả)
1. chạy detector cho 24 family (theo EVENT_SOURCES)
   → artifacts/scanner_v2/.../events.csv
2. chạy build_stock_pattern_profiles.py
   → web/stock_pattern_profiles.json (+ metadata timestamp)
3. chạy split_stock_history_artifacts.mjs
   → web/profiles/*.json, web/personality/*.json, web/setups/*.json
4. ghi web/.pipeline_last_refresh (timestamp)
5. (tuỳ chọn) curl dashboard reload
```

**Đầu ra kiểm chứng:** sau 1 lệnh, `web/.pipeline_last_refresh` cập nhật; dashboard không còn cảnh báo stale.

### 4.2. Timestamp check ở dashboard

`stock_history_pattern_module.js` đọc `metadata.generated_at` của profile → so với now → cảnh báo "dữ liệu >7 ngày" nếu stale. Đơn giản, không cần cron.

### 4.3. (Tùy chọn) Cron / launchd

Nếu chủ đầu tư muốn refresh định kỳ (ví dụ cuối tuần sau khi có data OHLCV mới): đề xuất `launchd` agent chạy `refresh_pattern_pipeline.sh` mỗi thứ 7. **Chỉ làm nếu chủ đầu tư yêu cầu** — V3 mặc định chạy thủ công 1 lệnh là đủ.

---

## 5. Hiển thị dashboard V3 (giải quyết điểm nghẽn 3, 4, 7)

### 5.1. Nhãn publication (mỗi pattern)

```
[pattern_name]  🟢 Đã kiểm định     ← Nấc 3
[pattern_name]  🟡 Ứng viên          ← Nấc 2
[pattern_name]  🔴 Bản nháp          ← Nấc 1 (ẩn mặc định)
```

### 5.2. Cảnh báo nổi (triết lý #11: ngưỡng số dưới bảng)

Mỗi bảng pattern có dòng chú thích:
```
Lookahead: 10 phiên · Failure: giá quay lại trong range mẹ (1%) · Mục tiêu TB: 2.3%
Cỡ mẫu: 12 (bị giới hạn ở cap 12) · Độ tin cậy: 🔴 Bản nháp
```

### 5.3. target_hit kèm độ lớn

```
❌ Hiện tại: "đạt mục tiêu 88%"
✅ V3:        "đạt mục tiêu 88% (mục tiêu TB 2.3% — quá nhỏ, hit rate chỉ tham khảo)"
```

### 5.4. Ẩn Nấc 1 mặc định

- Hồ sơ mã chỉ show Nấc 2/3.
- Có nút "Hiện bản nháp (chưa kiểm định)" → click mới show Nấc 1 + cảnh báo đỏ.

---

## 6. Tóm tắt giải pháp theo điểm nghẽn

| # | Điểm nghẽn | Giải pháp | Mốc |
|---|---|---|---|
| 1 | Lookahead lệch | Detector đọc từ registry | M1 |
| 2 | Failure sai định nghĩa | Thêm failure_busted, đổi tên failure_5pct | M2 |
| 3 | target_hit thiếu độ lớn | Thêm median_target_dist_pct + UI | M1/M3 |
| 4 | Cap bão hoà | hit_cap flag + frequency_score mới | M3 |
| 5 | MAE>80% bẩn | Filter + report ở build | M4 |
| 6 | Stale split | Auto-pipeline | M4 |
| 7 | Publication trộn | Nhãn Nấc 1/2/3 | M3 |
| 8 | Survivorship | Minh bạch metadata + toggle | M4 |
| 9 | Pipeline thủ công | 1 lệnh refresh | M4 |

Chi tiết chuẩn đo lường (lookahead/failure/target từng pattern) → file `03-measurement-standards.md`.
