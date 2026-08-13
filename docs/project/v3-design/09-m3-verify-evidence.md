# 09 — Bằng chứng nghiệm thu M3: UI tab Mẫu hình + mail 3 rào (chuẩn V3)

- **Ngày**: 12/08/2026 (verify UI xong ~23:55)
- **Mốc**: M3 trong kế hoạch K3 (08-k3-final-plan.md)
- **Tài liệu chuẩn đối chiếu**: 03-measurement-standards.md §2 (K3-1 duyệt) · 09-m2-verify-evidence.md (failure_busted K3-2 ký) · 01 §4.2 (tier Nấc 1/2/3)
- **Chờ**: K3-3 ký duyệt + phán quyết §5.2 (personality V1 vs V3) → chủ đầu tư ký tab Mẫu hình

---

## 1. Việc đã làm (tổng hợp M3)

1. **Re-scan 38 pattern toàn DB** (1599 mã, tới 2026-08-11) → `artifacts/scanner_v2_v3/<pattern>/db_active/events.csv` — **137.925 events**, có cột V3: `weak_move_5pct`, `failure_busted`, `days_to_bust`, `target_dist_pct`.
2. **Build profile V3** (`market_stats/scripts/build_stock_pattern_profiles.py` → `web_v3/stock_pattern_profiles.json`):
   - 1.583 mã · 38 pattern · bucket VN100 36/38 (2 pattern không có mã VN100 — hồ sơ chỉ ngoài VN100)
   - Mỗi pattern row: `n`, `n_busted`, `failure_busted_rate_pct`, `weak_move_5pct_rate_pct`, `target_hit_rate_pct`, `median_days_to_bust`, `median_target_dist_pct`, `median_mfe/mae`, `events_per_year`, `tier` (Nấc 1/2/3), `hit_cap`, `frequency_score`, `last_seen`
   - failure_busted tách **bucket VN100 vs ngoài** (`failure_busted_rate_vn100_pct` / `_outside_vn100_pct`)
   - metadata ghi `baseline_note` chuẩn Bulkowski + `source_events_dir` scanner_v2_v3
3. **Mail tín hiệu 3 rào** (`summarize_watchlist.py`): (a) chỉ xếp top Nấc ≥2; Nấc 1 → section riêng cuối mail nhãn đỏ "NHÁP"; (b) sắp theo `failure_busted_rate ≤ spec+2×` + `median_target_dist_pct ≥ 5%` + `n ≥ 30`; (c) cảnh báo rủi ro cố định đầu mail. **8/8 test pass**.
4. **UI web_v3** (staging — dashboard chính `web/` đóng băng tới khi M3 PASS):
   - Badge Nấc: 🟢 Đã kiểm định (n≥30) · 🟡 Ứng viên (5–29) · 🔴 NHÁP (n<5/không kiểm định — **ẩn mặc định**, toggle riêng "Mẫu hình bản nháp Nấc 1")
   - `failure_busted` hiển thị kèm (n_busted/n mẫu) · `median_days_to_bust` · `median_target_dist_pct` · `median_mfe/mae`
   - Chú thích "Cách đo kết quả mẫu hình" + "Chú thích cách đo vỡ mẫu (chuẩn V3)" — Bulkowski, cảnh báo VN cao hơn US (penny/T+/thanh khoản), kèm cỡ mẫu + nhóm vốn hóa
   - Banner **CẢNH BÁO non-advice** đầu section: "Đây là quét hình học từ dữ liệu lịch sử, KHÔNG phải khuyến nghị mua bán…"
   - `generated_at` bump (cache token) + split artifacts mới cho 3 mã verify + ABW (test Nấc 1)

## 2. Verify UI thật — chrome-devtools (server 8770, thư mục web_v3)

### 2.1 CTD (badge 16 🟡, không có Nấc 1 — đúng thiết kế)
- Banner CẢNH BÁO hiển thị đầu section ✅
- Bảng lịch sử V3: Inside Day `đạt mục tiêu 50% (6/12)` · `vỡ mẫu 16,7% (2/12)`; các pattern khác: 40% (4/10) · 25% (2/8) · 50% (4/8) · **75% (6/8)** — failure_busted kèm n_busted/n ✅ (re-verify UI thật 13/08 00:05: khớp profile V3 100%)
- Không còn text kỹ thuật ("Nguồn kỳ vọng…") ✅

### 2.2 ACB (badge 13 🟡)
- Banner CẢNH BÁO ✅ · `vỡ mẫu 8,3% (1/12)` · `20% (2/10)` · `25% (2/8)` · `50% (4/8)` · `37,5% (3/8)` ✅
- Empty state "Mẫu đã xác nhận gần đây" → dòng thân thiện "Hiện chưa có mẫu hình mới đủ điều kiện trong khung 20/60/120 phiên." ✅ (fix m3d — xem §4)

### 2.3 VNM (badge 12 🟡)
- Banner CẢNH BÁO ✅ · `vỡ mẫu 0% (0/12)` · `25% (2/8)` · `37,5% (3/8)` · `14,3% (1/7)` · `57,1% (4/7)` ✅
- `đạt mục tiêu 66,7% (8/12)` · chú thích V3 cuối trang đầy đủ (Bulkowski + US/VN + cỡ mẫu + vốn hóa) ✅
- Không có toggle Nấc 1 (không có pattern Nấc 1 — đúng thiết kế) ✅

### 2.4 ABW (test Nấc 1, trước đó)
- Toggle "Mẫu hình bản nháp Nấc 1 (3 — ẩn mặc định)" mở ra 3 rows 🔴 NHÁP (n=3), chú thích vỡ mẫu đầy đủ ✅

## 3. Tổng hợp verify

| Mã | Banner | Badge 🟡 | failure_busted kèm n | Chú thích V3 | Text kỹ thuật | Nấc 1 |
|---|---|---|---|---|---|---|
| CTD | ✅ | 16 | ✅ | ✅ | hết | không có (đúng) |
| ACB | ✅ | 13 | ✅ | ✅ | hết | không có |
| VNM | ✅ | 12 | ✅ | ✅ | hết | không có |
| ABW | ✅ | — | ✅ | ✅ | — | 3 rows 🔴 toggle ✅ |

## 4. Fix trong lúc verify (commit chưa tạo — chờ M3 PASS)

**m3d — empty state lộ text kỹ thuật** (`web_v3/stock_history_pattern_module.js` `currentPatternTable`):
- Trước: khi profile không có `current_patterns` (script build V3 KHÔNG sinh field này — events V3 chỉ ghi mẫu đã đóng lịch sử, không có "mẫu đang mở"), UI gọi `stockHistoryArtifactEmptyState` → hiển thị text hướng dẫn kỹ thuật: *"Nguồn kỳ vọng: profiles/VNM.json. Nếu file nhỏ chưa có, loader chỉ được fallback sang file tổng…"* — vi phạm triết lý #11 (không đưa thông tin phụ kỹ thuật dạng text hiển thị).
- Sau: bảng "Mẫu đã xác nhận gần đây" hiển thị dòng thân thiện: *"Hiện chưa có mẫu hình mới đủ điều kiện trong khung 20/60/120 phiên."*
- Version query bump `stockhistory-20260812-m3d` + cache token `generated_at` (23:10).

## 5. Ghi chú cho K3-3 — 2 vấn đề còn mở

### 5.1 ✅ Đã xử lý (fix m3d, §4)
Text kỹ thuật trong empty state — đã thay bằng text thân thiện.

### 5.2 ⚠️ Chưa xử lý — personality (tính cách mẫu hình) vẫn dữ liệu V1 cũ
- **Bằng chứng**: ACB "Tóm tắt tính cách mẫu hình" → Inside Day: *"đạt mục tiêu 100% (12/12 mẫu) · đi ngược 5% 0%"* trong khi bảng "Mẫu hình thường gặp" (V3) cùng tab: *"đạt mục tiêu 75% (9/12) · vỡ mẫu 8,3% (1/12)"*. **2 con số mâu thuẫn nhau trên cùng 1 tab.**
- **Nguồn**: `web_v3/personality/<SYM>.json` (split từ `web_v3/stock_pattern_personality_profiles.json`, metadata `workflow_id: stock_pattern_personality_profiles_v1`, `generated_at: 2026-08-06`) — script `scripts/build_stock_pattern_personality_profiles.py` vẫn đọc: (a) nguồn profile **V2 cũ** (`market_stats/web/stock_pattern_profiles.json` — default dòng 24), (b) thước **failure_5** (không phải failure_busted chuẩn V3), (c) events từ `_discover_research_dir`/`load_events` bản cũ (scanner_v2).
- **Vì sao không tự sửa**: rebuild personality theo V3 = đổi thước đo (failure_5 → failure_busted) + đổi ngưỡng nhãn hành vi (behavior_label/failure_style/bear_trap dùng ngưỡng 55%/50% của failure_5 — failure_busted thấp hơn nhiều, inside_day 8,3% vs failure_5 66,7%) → **quyết định thiết kế thuộc K3**, không tự ý chốt trong M3.
- **Khả thi để sửa**: script có sẵn, profile V3 đã có đủ field (`failure_busted_rate_pct`, `n_busted`, `median_days_to_bust`…). Cần: (a) trỏ nguồn profile V3, (b) đổi thước + ngưỡng theo chuẩn V3, (c) load_events đọc scanner_v2_v3, (d) rebuild + split lại personality.
- **Đề xuất 2 phương án cho K3-3**:
  - **A (khuyến nghị)**: rebuild personality theo V3 trong đợt M3 PASS (phụ thêm ~1–2 giờ) — tab đồng bộ 1 con số, tránh mâu thuẫn người đọc.
  - **B**: giữ nguyên personality V1, tab Mẫu hình ký phần lịch sử V3; thêm ghi chú nhỏ "phần tính cách đang chuẩn hóa" — sửa ở mốc sau (M4/M5).

## 6. Việc còn lại sau M3 PASS (theo 08-k3-final-plan.md)

1. Chuyển `web_v3/` → `web/` đúng 1 lần (rsync) + restart server 8766.
2. Chạy `summarize_watchlist` mail mẫu thật — kiểm: cảnh báo đầu mail + section Nấc 1 + số top theo 3 rào.
3. M4: `refresh_pattern_pipeline.sh` 1 lệnh end-to-end + filter MAE>80% + delisted metadata + stale check >7 ngày.
4. M5 (song song): GLM đọc PDF 19 family + nâng Nấc 3 P0/P1.
5. Verify tổng 5 tiêu chí + commit + nghiệm thu cuối.

## 7. K3-3 PASS CÓ ĐIỀU KIỆN — 3 điều kiện ĐÃ HOÀN TẤT (13/08 00:20)

### 7.1 ĐK1 — re-split toàn bộ + verify chéo HPG/VIC ✅
- `SPLIT_WEB_ROOT=web_v3 node split_stock_history_artifacts.mjs` (không symbol) → profiles 1599 (16 source_missing), personality 1601→1599, setups 1579.
- HPG/VIC file split giờ có `failure_busted_rate_pct` V3 (generated 2026-08-12T22:59:16) — hết split stale 2026-08-04.

### 7.2 ĐK2 — rebuild personality V3 (ngưỡng K3-3) ✅
- `scripts/build_stock_pattern_personality_profiles.py` đã sửa theo phán quyết K3-3: nguồn profile → `web_v3/`; events → `load_events_v3()` đọc thẳng `scanner_v2_v3` (38 pattern, PATTERN_LABELS đồng bộ); thước `failure_busted` thay `failure_5`; ngưỡng: behavior_label "hay vỡ mẫu" busted≥30%, "đi tiếp khá sạch" busted≤15% + hit≥60% + mfe≥1,4×mae; failure_style headline busted≥30% ("hay vỡ mẫu sau xác nhận"); bear_trap reclaim busted≥30%; metadata `workflow_id v3` + `thresholds` ghi rõ ngưỡng + `decided_by: K3-3`.
- Rebuild 1.599 mã PASS → split lại personality 1599 (0 source_missing).
- **Verify UI thật (Chrome web_v3)**:
  - ACB: "KIỂU HÀNH VI CHÍNH: dễ có phá vỡ gây đọc sai" · "SAU XÁC NHẬN: đi tiếp khá sạch" · Inside Day "đạt mục tiêu 75% (9/12) · vỡ mẫu 8,33% (1/12)" — **khớp 100% bảng V3, hết mâu thuẫn 100% vs 75%**.
  - VIC: Scallop tăng "hay vỡ mẫu 60% (6/10)", Bump-and-Run đỉnh "hay vỡ mẫu 40% (4/10)", bear_trap "có dấu hiệu bẫy giảm" (38,46%≥30) — đúng ngưỡng.
  - CTD/VNM/HPG file split khớp bảng V3 (CTD Inside Day 50% (6/12)/16,7% (2/12); HPG 75%/8,3%; VNM 66,7%/0%).

### 7.3 ĐK3 — mail mẫu thật (build preview, KHÔNG gửi) ✅
- `python -m scanner.send_realtime_scan_email` (không `--send`) với env `REALTIME_SCAN_V3_PROFILE` trỏ web_v3 (đã thêm env override vào module — 1 chỗ, dòng `_load_v3_pattern_stats`).
- **Cảnh báo đầu mail**: đúng văn bản bắt buộc + dòng "⚠️ LƯU Ý: hệ thống đang chuẩn hóa lại số liệu mẫu hình (nâng cấp V3)" (R10).
- **3 rào dùng số V3 thật**: watchlist FRT (broadening_bottoms) `failure_busted_rate_pct: 22,5` + `median_target_dist_pct: 26,1` (≥5%) — từ web_v3 profile; risk_context 20 mã (pipe_bottoms 27,5%, horn_bottoms 31,3%...); buy_candidates=0 hôm nay (không có tín hiệu mới đạt chuẩn — hợp lệ).
- **Section Nấc 1 riêng cuối mail**: "3. Không đạt chuẩn tín hiệu (4 mã, chỉ tham khảo)" — ACL double_bottoms_adam_* (variant ngoài registry 38) → `failure_busted_rate_pct: None`, nhãn rõ "bản nháp Nấc 1 chưa kiểm định / mẫu nhỏ <30 / mục tiêu <5% — không dùng làm tín hiệu".

### 7.4 Footnote VN100 — ĐIỀU CHỈNH SO VỚI K3-3 (bằng chứng mới)
- K3-3 yêu cầu ghi "diamond_bottoms/tops **không có quan sát** trong VN100" — dựa trên `bucket_vn100: null`.
- **Kiểm chứng thật**: cả 2 pattern đều CÓ quan sát VN100 — diamond_bottoms 4 (DBC/HDB/KDC/MSN), diamond_tops 4 (DPM/LPB/VCG/VND); null vì `bucket_vn100_stats` chỉ công bố khi bucket ≥20 quan sát (chính sách K3-2).
- **Đã sửa cho đúng sự thật**: (a) `bucket_vn100_stats` luôn trả `n_vn100`/`n_outside` (kể cả <20) — rebuild profile 1.583 mã (gen 2026-08-13T00:09:09) + re-split; (b) footnote UI: "Diamond Bottoms và Diamond Tops đạt Nấc 3 theo n toàn thị trường, nhưng mới có vài quan sát trong nhóm VN100 (chưa đủ 20 để công bố chỉ số riêng cho nhóm vốn hóa lớn) — đọc như hành vi trung bình cả thị trường". Version bump `stockhistory-20260813-m3e`; verify Chrome: footnote hiển thị + ACB bảng khớp + 0 lỗi JS.

### 7.5 Vấn đề vận hành phát hiện khi chạy mail mẫu (báo chủ đầu tư)
1. **Job 17:00 đang FAIL** (log 08-12 17:00): `Fatal Python error ... Resource deadlock avoided` — `.venv` của repo nghiên cứu nằm trong thư mục iCloud → deadlock khi launchd chạy; kèm `refresh_symbol_args: unbound variable`.
2. **Provider vnstock_data = free tier** (60 requests/phút, community): refresh 100 mã VN100 bị chặn "Rate Limit Exceeded" → mail mẫu đã chạy KHÔNG refresh (DB dev clone mới tới 08-12, đủ dùng).
3. Giải pháp đề xuất (M4): script ưu tiên python3.14 hệ thống (có vnstock_data) thay `.venv`; xem xét provider trả phí nếu cần refresh hằng ngày.
