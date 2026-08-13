# THƯ GIAO VIỆC — DỰ ÁN M4 + M5 (Nghiên cứu mẫu hình nến Bulkowski V3)

> **Loại**: Handoff cho session RIÊNG (tách khỏi session nghiệm thu tab Mẫu hình).
> **Lead**: **deepseek-v4-pro** (provider OpenCode Go) — chỉ đạo, phán quyết kỹ thuật, nghiệm thu nội bộ.
> **Subagent được phép gọi**: **GLM-5.2** (Z.AI — đọc PDF gốc/ảnh qua MCP image; websearch) · **deepseek-v4-flash** (OpenCode Go — thi công lệnh "bảo gì làm đó").
> **Ngày**: 13/08/2026 · **Chủ đầu tư**: duyệt tách dự án (quyết định 13/08/2026).

---

## 0. Bối cảnh (đọc kỹ trước khi làm)

Dự án nghiên cứu mẫu hình nến Bulkowski (ChartPatternscan) đã qua V0→V3. **Tab "Mẫu hình" (pattern-profile) trên dashboard market_stats ĐÃ ĐẠT NGHIỆM THU VÒNG 2** (V4 Pro black-box: 0 critical/0 high/0 medium, 7 mức low không chặn — chi tiết `docs/project/v3-design/09-m3-verify-evidence.md` §8). Việc còn lại của V3 là **M4** (tự động hóa + lọc dữ liệu nghi ngờ) và **M5** (đối chiếu sách PDF cho 19 family) — được tách sang session này.

**BẮT BUỘC đọc trước khi làm bất cứ việc gì:**
1. `docs/project/v3-design/08-k3-final-plan.md` — kế hoạch tổng K3 (M0–M5, quyết định, rủi ro, mốc nghiệm thu).
2. `docs/project/v3-design/09-m3-verify-evidence.md` — bằng chứng đợt M3 + review 2 model + 3 phán quyết chủ đầu tư (H1/H2/H3).
3. `docs/project/pdf_review/PDF_REVIEW_20260812.md` — đối chiếu sách Bulkowski (ECP 2nd ed) vs digitized: sample/failure/lookahead từng pattern.
4. `docs/project/v3-design/03-measurement-standards.md` + `04-governance-roadmap.md` — chuẩn đo lường + roadmap.
5. `scanner/v2/measurement_registry.py` + `scanner/v2/pattern_family_manifest.json` — registry cap + manifest status (publication_final/active/candidate).

**Trạng thái code hiện tại (đã commit + push, ngày 13/08):**
- `market_stats/scripts/build_stock_pattern_profiles.py` = bản V3 (41 pattern, `patterns_stats` gom theo pattern_key chuẩn hóa — 4 gaps tách riêng, three_methods → falling/rising). Nhãn Nấc H1: "Đã đối chiếu PDF" (publication_final/active + n≥30) / "Đã đo chuẩn V3" (registry n≥30) / "Ứng viên" (5–29) / "Bản nháp" (<5).
- `market_stats/scripts/` (root) = GIỐNG HỆT bản V3 (C1 đã vá — pipeline refresh hằng ngày không còn ghi đè bản cũ).
- `scanner/send_realtime_scan_email.py`: rào mail H2 `failure_busted_rate_pct ≤ 2×spec` (PATTERN_FAILURE_SPECS_PCT 16 pattern, có spec VN thấp nhất từ PDF_REVIEW) + fallback manifest (M2) cho pattern sách chưa đo V3.

**Những thứ ĐÃ CÓ SẴN (không làm lại):**
- `artifacts/scanner_v2_v3/<pattern>/db_active/events.csv` — 37 thư mục, events chuẩn V3 (cột: mfe_pct, failure_busted, weak_move_5pct, target_dist_pct, days_to_bust, breakout_date...).
- `extraction_phase_1/digitization/patterns_digitized_pdfreview/` — 11 pattern đã digitize từ PDF review (broadening RA asc/desc, dead_cat_bounce, high_tight_flags, pipe_tops, scallops asc/desc, three_falling_peaks, three_rising_valleys, triple_bottoms, triple_tops).
- `docs/project/v3-design/07-codebase-map.md` — bản đồ code cho model review.
- `market_cache/stock_ohlcv/latest.sqlite` — symlink → `~/dev/market_stats_v2/market_cache/stock_ohlcv/latest.sqlite` (DB thật 4,17M rows; đã gitignore).
- Server dev 8766: `~/dev/market_stats_v2/market_stats/local_server.py --host` (WEB_DIR = market_stats/web).

**Quy trình pipeline hiện tại (đã vận hành tay, cần tự động hóa ở M4):**
events V3 → `python3 market_stats/scripts/build_stock_pattern_profiles.py` (out web_v3) → `SPLIT_WEB_ROOT=web_v3 node split_stock_history_artifacts.mjs` → `rsync web_v3/ → web/` → `rsync web/ → ~/dev/market_stats_v2/market_stats/web/` → restart server 8766 → mail preview.

---

## 1. NHIỆM VỤ M4 — Tự động hóa + lọc dữ liệu nghi ngờ (ưu tiên cao nhất)

Nguồn chuẩn: `08-k3-final-plan.md` §90 (M4) + `09-m3-verify-evidence.md` §6.3 + ghi chú job 17:00.

### 1a. `refresh_pattern_pipeline.sh` — 1 lệnh end-to-end, exit 0
Gom toàn bộ chuỗi trên thành **1 script** (đặt cạnh build script), có log rõ từng bước, `set -euo pipefail`, exit 0 khi thành công. Nhận diện được khi nào cần chạy (dữ liệu mới?) và cho phép `--skip-*` từng bước. **KHÔNG tự cài launchd/job 17:00** — chỉ tạo script + test chạy được; việc cài job là quyết định chủ đầu tư ở mốc sau.

### 1b. Fix job 17:00 đang FAIL (bằng chứng `09-m3` §6.3)
- Lỗi 1: `Fatal Python error ... Resource deadlock avoided` — `.venv` của repo nghiên cứu nằm TRONG iCloud → deadlock khi launchd chạy. Giải pháp đề xuất trong tài liệu: script ưu tiên `python3.14` hệ thống (có vnstock_data) thay `.venv`.
- Lỗi 2: `refresh_symbol_args: unbound variable`.
- Việc này cần điều tra THẬT (đọc script job hiện tại: `pattern_buy_scanner_launchd.sh` + script refresh liên quan), không đoán.

### 1c. Filter MAE > 80% (suspect split — quyết định K3 §50 `08-k3-final-plan.md`)
- Trong build profile V3: events có `mae_pct > 80` → KHÔNG vào stats, đánh `data_quality_flag=suspect_split`, ghi report per pattern: `events_dropped_split_suspect` / `drop_rate_pct` (kỳ vọng inside_bar ≈ 2,8%).
- Mã có >3 event suspect → `data_review_needed` trong hồ sơ mã.
- KHÔNG thu phân phối MAE để cân nhắc ngưỡng 60–70% (đó là V3.1 — chỉ ghi nhận).
- Ngưỡng 80% là BẢO THỦ — chỉ loại outlier gần như chắc chắn split chưa adjust.

### 1d. Delisted metadata + stale check
- Delisted: xử lý MINH BẠCH theo K3 §53 (survivorship = minh bạch, không xử lý) — nếu có cơ chế delisted metadata sẵn (grep codebase), gắn vào; nếu chưa có, ghi nhận và đề xuất.
- Stale check >7 ngày: build cảnh báo khi events `last_event_date` > 7 ngày (server refresh hằng ngày ~40 phút/1355 mã — xem `~/dev/market_stats_v2` `local_server.py` endpoint `/api/refresh-data`).

### 1e. Nghiệm thu M4 (tự kiểm trước khi báo):
- Script chạy end-to-end exit 0 từ máy sạch.
- Report `drop_rate_pct` per pattern có (inside_bar ≈ 2,8%).
- metadata có `pipeline_version` + `data_quality_flag`.
- **SAU KHI XONG: giao K3-4 phản biện** (gọi qua subagent kimi-k3-worker — xem `provider-opencode-go-cac-model.md` memory; nếu K3 không gọi được, báo chủ đầu tư quyết).

---

## 2. NHIỆM VỤ M5 — GLM đọc PDF 19 family + nâng Nấc 3 (song song)

Nguồn chuẩn: `08-k3-final-plan.md` §91 + §17-19 + §95 (thứ tự P0→P3).

### 2a. GLM-5.2 đọc PDF gốc Bulkowski (ECP 2nd ed) cho 19 family còn thiếu
- Gọi **GLM-5.2 (Z.AI)** — model duyệt đọc được ảnh/PDF qua MCP image (V4 Pro/Flash mù ảnh — KHÔNG cố đọc PDF bằng V4 Pro).
- Giao việc THEO FAMILY, không 1 prompt khổng lồ: mỗi lần gọi 3–5 family, prompt phải ghi rõ đường dẫn PDF + mục cần trích (sample, break-even failure các hướng, % meet target, days to ultimate, measure rule).
- Đầu ra chuẩn hóa **1 bảng per family**: `pdf_path / pages đã kiểm / sample / failure BE / target / days to ultimate / ghi chú lệch digitized`.
- Tham chiếu: 11 file đã digitize trong `extraction_phase_1/digitization/patterns_digitized_pdfreview/` (GLM xác nhận lại 11 file đó có đúng PDF gốc không, tránh làm 2 lần), + `PDF_REVIEW_20260812.md` (đã có 14 pattern — 5 khớp/3 lệch/6 thiếu).
- Cập nhật spec vào `measurement_registry.py` + manifest khi có số PDF xác nhận (detector KHÔNG re-code — chỉ nạp lại số, gắn cờ `lookahead_source`).
- Pattern bị lệch lớn (pipe/H&S/cup lookahead 63–252 vs PDF 167–194 ngày) → ưu tiên audit sớm theo thứ tự P0: inside_bar, pipe_bottoms/tops, horn, flags/pennants/HTF, cup. P1: HSB/HST, scallops, rectangle, triangles. P2: double variants, broadening, diamond, wedges, harami detector. P3: gaps, islands, measured_move, three_methods, peaks/valleys, triple, bump, dead_cat, rounding.

### 2b. Nâng Nấc 3 dần
- Family nào có PDF xác nhận + n≥30 → chuyển manifest status → nhãn "Đã đối chiếu PDF" (build lại profile). Cập nhật registry NGAY khi có số, không đợi hết 19 family.

### 2c. Detector harami body-based (P2 — chỉ khi M5 phần chính xong)
- inside_day hiện range-based (định nghĩa lệch Harami — PDF_REVIEW §1). Tách 2 spec: inside_bar (range, giữ detector) + harami (body, detector mới P2).

### 2d. Nghiệm thu M5 (tự kiểm trước khi báo):
- Bảng per family đầy đủ 19 family (pdf_path/pages/failure/target/sample).
- 5 chương sách P0 đủ điều kiện viết (theo `canonical-publication-flow.md`).
- **SAU KHI XONG: giao K3-5 phản biện** (như M4).

---

## 3. QUY TẮC LÀM VIỆC (bắt buộc — AGENTS.md chủ đầu tư)

1. **3 lớp mọi lúc**: khi gọi subagent / ghi note → ghi đủ session mẹ · model · provider. Bộ nhớ subagent: `.zcode/subagent-memory/sess_<session-mẹ-id>/` (mỗi session 1 thư mục, không đọc session khác).
2. **Không tự quyết định model**: chủ đầu tư đã chỉ định — V4 Pro lead, GLM (Z.AI) đọc PDF/ảnh, Flash thi công. Không tạo profile agent mới.
3. **Git kỷ luật**: 1 việc = 1 branch riêng từ main; commit thường xuyên `type(scope): mô tả tiếng Việt`; push sau commit quan trọng; KHÔNG `git add -A` mù khi >100 files; xóa branch sau merge; giữ ≤5 branch local.
4. **Xác nhận trước khi phá/xóa/ghi đè** file quan trọng (đặc biệt build script, manifest, registry) — hỏi chủ đầu tư, kèm recommended + trade-off.
5. **Prompt verify K3**: KHÔNG đưa số "cần đạt" (gây anchoring); chỉ đúng nguồn runtime; đối chiếu TỪNG THÀNH PHẦN công thức.
6. **Môi trường**: DB thật qua symlink `market_cache/stock_ohlcv/latest.sqlite` (đã gitignore). Không tạo clone thư mục mới.

---

## 4. BÁO CÁO CHO CHỦ ĐẦU TƯ

- Kết thúc mỗi mốc (M4 xong / M5 xong): báo cáo 1 trang — làm gì, kết quả (số cụ thể), phát hiện, cần chủ đầu tư quyết gì (nếu có), kèm đường dẫn bằng chứng.
- Trước khi giao K3-4/K3-5: chuẩn bị context pack đầy đủ (file tham chiếu + thay đổi) — tránh K3 phải đoán.
- KHÔNG tự ý chuyển `web_v3/ → web/` lần cuối hay ký nghiệm thu tổng — việc đó ở session chính (session nghiệm thu tab Mẫu hình), sau khi chủ đầu tư duyệt.
