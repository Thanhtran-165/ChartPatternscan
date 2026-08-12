# 07 — Bản đồ codebase cho Kimi K3 (đọc nhanh, không cần dò lại)

> File này do main agent khảo sát thực tế 12/08/2026, để K3 không phải dò codebase từ đầu. Mọi đường dẫn đã kiểm chứng tồn tại. Bổ sung đọc sâu khi cần thiết: `03-measurement-standards.md` (chuẩn đo), `02-architecture-bottlenecks.md` (9 điểm nghẽn + file:line), `PDF_REVIEW_20260812.md` (số liệu PDF gốc).

---

## 1. Sơ đồ tổng thể (4 repo/thư mục liên quan)

```
┌─────────────────────────────────────────────────────────────────┐
│ A. REPO NGHIÊN CỨU (bản chính — iCloud, git riêng):            │
│    "/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/   │
│     main sonet/Nghiên cứu mô hình nến/"                         │
│    → scanner/v2 (detector) · artifacts (kết quả) · docs         │
│    → extraction_phase_1/digitization (spec) · references (PDF)  │
└─────────────────────────────────────────────────────────────────┘
                              ↓ events.csv
┌─────────────────────────────────────────────────────────────────┐
│ B. REPO MARKET STATS (iCloud — bản gốc):                        │
│    ".../main sonet/scripts/" (build scripts)                    │
│    ".../main sonet/market_stats/" (web artifacts + dashboard)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓ (đồng bộ qua git)
┌─────────────────────────────────────────────────────────────────┐
│ C. REPO MARKET STATS (dev — bản CHẠY dashboard port 8766):      │
│    ~/dev/market_stats_v2/                                       │
│    → scripts/ (bản scripts đang chạy)                           │
│    → market_stats/web/ (web artifacts + dashboard module JS)    │
└─────────────────────────────────────────────────────────────────┘
```

**Quy ước:** code nghiên cứu sửa ở A. Build + web artifacts + dashboard sửa ở B (gốc) hoặc C (chạy). Hiện tại Lớp A/B/C đã build vào **C (dev)** để dashboard chạy xem được ngay; B (iCloud) chưa sync — chờ chủ đầu tư duyệt commit.

**⚠️ Cảnh báo 3 thư mục trùng tên** (chưa dọn, chờ duyệt):
- `Nghiên cứu mô hình nến/` = bản chính (git riêng, đang làm)
- `Nghiên cứu mô hình nền/` = bản TRÙNG (cũng có .git, cấu trúc giống hệt)
- `Nghiên cứu mô hình nén/` = bản cũ nhất (scanner cũ + vietnam_stocks.db)

---

## 2. Bộ scanner (LÕI — lõi chất lượng của cả 3 dự án)

### 2.1. Vị trí + cấu trúc
`scanner/v2/` — **41 file .py** (gồm detector chính + monograph/tradable_setup/audit phụ). Detector chính theo family:

| Nhóm | File |
|---|---|
| Ngắn hạn | `inside_days.py` (lookahead=10 ✅ đã sửa), `three_methods.py` (20 ✅) |
| Trung hạn | `pipes.py` (63 ✅ — **detector trung tâm**: horns.py, scallops.py, gaps.py, islands.py, measured_moves.py, pennants.py import `_evaluate_detection` từ đây), `horns.py` (42 ✅), `flags_experiment.py` (63 ✅), `high_tight_flags.py` |
| Dài hạn | `cup_with_handle.py` (252 ✅), `head_shoulders.py` (252 ✅), `double_patterns.py` (252 ✅), `diamonds.py` (252 ✅), `rectangles.py` (252 ✅), `rounding.py` (252 ✅), `broadening_patterns.py` (252 ✅), `bump_and_run.py` (252 ✅), `scallops.py` (252 ✅), `triple_patterns.py` (252 ✅), `three_peaks_valleys.py` (252 ✅) |
| Tam giác/nêm | `ascending_triangles.py` (126 ✅), `descending_triangles.py` (126 ✅), `symmetrical_triangles.py` (126 ✅), `rising_wedges.py` (126 ✅), `falling_wedges.py` (126 ✅) |
| Đặc biệt | `dead_cat_bounce.py` (120 tạm — chờ spec PDF), `bull_flags.py`, `bear_flags.py`, `pennants.py` |

> ✅ = đã sửa lookahead theo bảng chuẩn (Lớp A, 12/08). Bảng chuẩn đầy đủ: `03-measurement-standards.md` §1.

### 2.2. Hàm trung tâm (điểm chung mọi detector)
- `scanner/v2/pipes.py:344` — `_evaluate_detection(df, detection, *, lookahead=63)` — **hàm đo lường dùng chung** (MFE/MAE/target/failure/evaluated_bars). Sau Lớp A: default 63 (trước 120).
- `scanner/v2/pipes.py:56-68` — `PipeConfig` (max_events_per_symbol=18, pattern_key, từ_mapping)
- `scanner/v2/pipes.py:400` — `scan_symbol(...)` entrypoint mỗi mã
- `scanner/v2/inside_days.py:231` — `_evaluate_detection(df, row, lookahead=10)` ✅

### 2.3. Các điểm cần K3 phán quyết (bottleneck còn mở — xem file 02 chi tiết)
- `pipes.py:392` — `failure_5pct = MFE < 5%` → **SAI chuẩn Bulkowski**; cần `failure_busted` (từ path rows: giá vượt lại đáy/đỉnh pattern trước khi chạm target) + đổi tên `weak_move_5pct`
- `inside_days.py:69` — `max_events_per_symbol=12`; scallops.py:83 → 14; pipes.py:68 → 18; bump → 10 → **cap bão hoà** (cần `hit_cap` flag)
- **Lookahead 252 default phổ thông SAI** (Lớp C): PDF thực tế Harami 7-9d, Pipe 133-194d, Cup 63-167d, H&S 107-176d → cần audit toàn 31 family, quyết định lấy chuẩn nào làm gốc (digitized hiện tại vs PDF gốc)
- **inside_day = Harami lệch ĐỊNH NGHĨA** (digitized dùng range high-low; Bulkowski EC dùng body open-close) → quyết định tách 2 spec
- **pipe_bottoms nghiên cứu trên WEEKLY chart** trong sách, digitized không ghi chú → nguồn lệch

### 2.4. Đầu ra detector (per pattern, mỗi family 1 thư mục)
`artifacts/scanner_v2/<family>/<pattern>/db_active/` chứa 4 file:
- `events.csv` — mỗi dòng 1 event (55-60 cột: mfe_pct, mae_pct, target_dist_pct, target_hit, failure_5pct, evaluated_bars, path_quality_bucket, missing_bar_rate_60d...)
- `post_breakout_path.csv` — **chuỗi giá sau breakout (path rows)** — cần cho failure_busted (M2)
- `detections.json`, `statistics.json`

**59 events.csv** được tái sinh hôm nay (12/08, Lớp A). Backup trước khi sửa: `artifacts/scanner_v2_backup_lookahead_fix_20260812/` (cũ hơn): `scanner_v2_backup_pre_lookahead_fix_20260812/`.

---

## 3. Chuẩn đo (spec) — Tầng 1 của lõi

### 3.1. Digitized spec (31 family — đang dùng)
`extraction_phase_1/digitization/patterns_digitized/*_digitized.json` (31 file). Schema: family_name, post_breakout_measurement (lookahead_bars, failure, target), aliases...

### 3.2. Manifest publication (55 pattern → nấc)
`scanner/v2/pattern_family_manifest.json` — cấu trúc `families.<family>.patterns.<pattern>.status`. Chỉ cover **14/55** pattern; 41 còn lại không có entry (mặc định Nấc 1 draft).

### 3.3. PDF gốc (mới đọc — Lớp C)
`references/`:
- `encyclopedia-of-chart-patterns-2nbsped-....pdf` (ECP, 1035 trang — cuốn chính)
- `Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf` (EC, 966 trang — inside_day/Harami)
- `Wiley Trading ... Chart Patterns_ After the Buy-Wiley 2016.pdf` (ATB, 555 trang)

Kết quả đọc: `docs/project/pdf_review/PDF_REVIEW_20260812.md` (20KB — số liệu failure/target/sample/lookahead của 12 pattern + số trang). Cách đọc: `pdftotext -layout` + `pypdf` map trang (offset +23 PDF vs sách).

### 3.4. Spec MỚI cho pattern thiếu (đang tạo, Lớp Bổ sung)
`extraction_phase_1/digitization/patterns_digitized_pdfreview/` — 11 file JSON mới (scallops 4, broadening_ra 2, triple 2, peaks/valleys 2, dead_cat, high_tight_flags, pipe_tops) theo schema digitized cũ + key `pdf_review_source`.

---

## 4. Pipeline build → dashboard

```
[events.csv từ detector]
   ↓ EVENT_SOURCES — scripts/build_stock_pattern_profiles.py:141 (import từ
     scanner.rebuild_source_guided_final_chapters — 55 key pattern_id → path + filters)
[build_stock_pattern_profiles.py]
   - _pattern_profile (line 434) — score: 0.25*freq + 0.35*outcome + 0.25*clean + 0.15*fresh
   - build_profiles (478) → metadata.generated_at, publication_narrative_tier_counts (mới Lớp B)
   - Lớp B đã thêm: publication_status / publication_narrative_tier / publication_note
     (PUBLICATION_MAPPING_VERSION v3-lopB-20260812, dòng 32-100)
   → web/stock_pattern_profiles.json (~31MB, 1585 profiles, 169.870 events)
[build_stock_pattern_personality_profiles.py] → web/stock_pattern_personality_profiles.json
[build_current_pattern_setups.py] → web/current_pattern_setups.json
   ↓ CHẠY TAY (điểm nghẽn 6): scripts/split_stock_history_artifacts.mjs
[web/profiles/<symbol>.json + web/personality/ + web/setups/] (~1600 file mỗi loại)
   ↓
[dashboard: market_stats/web/stock_history_pattern_module.js] (26KB, đã sửa Lớp B:
   publicationBadge (98), publicationFootnote (109), splitByTier (114), rowsTable ẩn Nấc 1 + toggle)
```

**Lệnh build đã dùng hôm nay (dev):** `cd ~/dev/market_stats_v2 && python scripts/build_stock_pattern_profiles.py` (~58s) → 3 file web (16:57-16:58) → `node market_stats/split_stock_history_artifacts.mjs` (split script nằm tại `~/dev/market_stats_v2/market_stats/split_stock_history_artifacts.mjs` — iCloud: `market_stats/split_stock_history_artifacts.mjs`).

---

## 5. 3 dự án tiêu thụ lõi (K3 cố vấn phần này)

### 5.1. Dự án 1 — SÁCH (mảng publication)
- `docs/publication/` — architecture.md, chapter-contract.md, data-contracts.md, sample-selection-rules.md, commentary-style-guide.md
- `docs/publication/book-v2/` (mới) + `book-v1-legacy` (cũ)
- `docs/project/bulkowski-vietnam-*` (framework, methodology-contract, statistics-contract, release-gate, chapter-framework)
- Scanner build chương: `scanner/build_*_public_chapter.py`, `scanner/*_family_publication_specs.py`
- Publication gate: `docs/project/source-grounded-publication-gate.md` (11 nguyên tắc) + `publication-semantic-gate.md`

### 5.2. Dự án 2 — MAIL TÍN HIỆU = "BUY Candidate Scan - VN100 Watchlist"
- LaunchAgent: `com.bobo.pattern-buy-scanner` (17:00 hằng ngày) — `scripts/pattern_buy_scanner_launchd.sh` + `scripts/run_pattern_buy_scanner_daily.sh`
- Scanner thật (trong repo nghiên cứu):
  - `scanner/run_realtime_scan_watchlist.py` — quét BUY_PULLBACK từ event artifacts
  - `scanner/run_buy_setup_scan_watchlist.py` — quét BUY_SETUP trước phá vỡ (VN100)
  - `scanner/send_realtime_scan_email.py` — gửi mail (REALTIME_SCAN_EMAIL_TO=stevetransg@gmail.com)
  - `scanner/refresh_realtime_market_data.py`, `scanner/realtime_scan_history.py`, `scanner/build_realtime_scan_pdf_report.py`
- Tests: `tests/test_realtime_scan_*.py`, `tests/test_buy_setup_scan_watchlist.py`
- Runbook: `docs/project/pattern-buy-scanner-daily-runbook.md`
- ⚠️ ĐỘC LẬP: không gắn R2/TApro/Market flow (R2 Signal là logic khác — `R2_85/`, `email_templates/r2_signals/`)

### 5.3. Dự án 3 — MARKET STATS (dashboard)
- Dashboard module: `~/dev/market_stats_v2/market_stats/web/stock_history_pattern_module.js` (tab "Lớp mẫu hình phụ trợ"; đã có nhãn 🟢🟡🔴 + ẩn Nấc 1 + toggle "Hiện bản nháp" từ Lớp B)
- Web artifacts: `~/dev/market_stats_v2/market_stats/web/stock_pattern*.json` + profiles/personality/setups split
- Dashboard chạy: port 8766 (dev), server: `scripts/run_vietnam_stock_webapp.sh`
- Chủ đầu tư đã ký nghiệm thu 6/7 tab; tab "Lớp mẫu hình" chờ v3 xong

---

## 6. Trạng thái hiện tại (12/08, trước khi K3 chốt)

| Hạng mục | Trạng thái |
|---|---|
| Lớp A (lookahead 25 detector) | ✅ XONG — verify: inside_day MFE 15.0→5.1%, cup 23.6→38.5% |
| Lớp B (nhãn publication 55 pattern) | ✅ XONG — 5🟢/3🟡/47🔴, UI test thật CTD/ACB |
| Lớp C (đọc PDF 12 pattern) | ✅ XONG — PDF_REVIEW_20260812.md |
| Spec mới 11 pattern thiếu | 🔄 GLM đang tạo (patterns_digitized_pdfreview/) |
| Commit | ❌ CHƯA — chờ chủ đầu tư duyệt |
| Backup | `artifacts/scanner_v2_backup_lookahead_fix_20260812/` · `~/dev/market_stats_v2/backups/lopB_20260812-1706/` |
| Đặc tả v3 | `docs/project/v3-design/` 00-overview · 01-vision-scope · 02-architecture-bottlenecks · 03-measurement-standards · 04-governance-roadmap · 05-main-refinement (chốt main agent) · 06-glm-plan-reference (bản GLM tham khảo) |

**Điều K3 cần phán quyết (tóm tắt):** ① chuẩn gốc nào (digitized vs PDF) + thứ tự mốc M1-M5; ② failure_busted định nghĩa + ngưỡng per-pattern; ③ xử lý inside_day (tách spec); ④ cap/MAE>80/survivorship chính sách; ⑤ UI freeze web_v3 staging; ⑥ lộ trình cho 3 dự án tiêu thụ (sách · mail · dashboard).
