# 06 — Kế hoạch chốt của GLM-5.2 (BẢN THAM KHẢO — chưa phải chốt cuối)

> **Trạng thái:** ⚠️ BẢN THAM KHẢO — chưa phải kế hoạch final.
> **Lý do:** bản này do GLM-5.2 (Z.ai) viết — CÙNG model đã viết đặc tả 00-04 → KHÔNG độc lập (người viết tự chốt cho mình). Vi phạm quy trình "2 model độc lập" của chủ đầu tư.
> **Kế hoạch:** K3 thật (OpenCode Go) sẽ duyệt/điều chỉnh bản này thành kế hoạch FINAL sau khi khởi động lại ZCode (override hiện chỉ có hiệu lực sau restart — kiểm chứng 11/08).
> **Ngày:** 12/08/2026 · **Session mẹ:** sess_1c356bbb-d126-4ea6-9508-ad490bca4ef1
> **Ghi nhận theo chỉ đạo chủ đầu tư 12/08:** "ghi nhận kết quả GLM vừa làm xong như 1 bản tham khảo".

---

## 1. Phán quyết tổng thể (GLM)

Đặc tả v3 (00-05) **ĐỦ để triển khai M1-M4**, cần bổ sung 3 chỗ trước khi thi công:
1. Bảng lookahead 03 §1.1 còn nhiều "?" (chỉ 5/24 detector có giá trị) → cần **audit 24 detector trước M1** (baseline before/after chính xác)
2. `failure_threshold_pct` gaps = "varies" → đọc lại `gaps_digitized.json` lấy threshold per gap type trước M2
3. File 05 §5 để ngỏ registry ("nếu rẻ") → **CHỐT: registry bắt buộc làm trong M1** (M2/M3 cần 1 nguồn chuẩn duy nhất)

## 2. Kế hoạch mốc (GLM đề xuất: M0 mới + M1-M5)

| Mốc | Mục tiêu | Đầu ra kiểm chứng | Thời gian |
|---|---|---|---|
| **M0** (MỚI) | Audit 24 detector + snapshot baseline + tạo `web_v3/` staging | Bảng audit (file:line, lookahead hiện tại, cap); snapshot `backups/v3_benchmark/m0/`; `web_v3/` tạo | 1 ngày |
| **M1** | Sửa lookahead 24 detector theo spec + **registry tập trung** | inside_day 60→10, scallop 120→252, horn 120→42; MFE inside_day ~15%→~3% (±2%); events.csv có `expected_lookahead`, `actual_evaluated_bars`; `lookahead_registry.py` tồn tại | 2-3 ngày |
| **M2** | failure_busted chuẩn + đổi tên failure_5pct→weak_move_5pct + target kèm dist_pct | bull_flags failure_busted ≈5.5% (±3%); inside_day ≈15%; cup ≈5%; events.csv có `failure_busted`, `failure_threshold_pct`, `weak_move_5pct`, `target_dist_pct` | 2-3 ngày |
| **M3** | Publication status → UI + nhãn Nấc 1/2/3 + ẩn Nấc 1 + target kèm dist_pct | Build profile gắn `publication_narrative_tier` 55 pattern; dashboard nhãn Nấc + màu + chú thích; toggle "hiện bản nháp"; cảnh báo target<3%; **chuyển web_v3→web (bật V3 1 lần)** | 2-3 ngày |
| **M4** | 1 lệnh refresh pipeline + filter MAE>80% + survivorship minh bạch | `scripts/refresh_pattern_pipeline.sh` end-to-end; report `events_total|events_dropped_split_suspect|drop_rate_pct|delisted_symbols`; inside_day drop ≈2.8%; cảnh báo stale >7 ngày | 1-2 ngày |
| **M5** | Đọc PDF gốc Bulkowski P0+P1 | Bảng `pattern|pdf_path|book_pages_checked|failure_rate_pdf|target_pdf|sample_pdf`; P0+P1 `direct_pdf_review.status=PASS` → lên Nấc 3 | 5-7 ngày |

Thứ tự: M0 → M1 (kèm registry) → M2 → M3 (bật V3) → M4 → M5 (M5 song song từ sau M2). Tổng ~13-19 ngày làm việc.

## 3. 6 quyết định GLM chốt (để K3 duyệt lại)

| # | Câu hỏi | Chốt của GLM |
|---|---|---|
| 1 | Cap max_events_per_symbol | **GIỮ cap (12/14/18) + hit_cap flag ở M3. KHÔNG tăng cap trong V3** (bump 10→20 = task riêng) |
| 2 | Registry tập trung | **BẮT BUỘC làm trong M1** (~2-4h) — 1 nguồn duy nhất |
| 3 | Đóng băng UI | **web_v3 staging, chuyển 1 lần khi M3 xong** + banner "Đang nâng cấp V3" trên dashboard hiện tại |
| 4 | failure_busted vs weak_move_5pct | **Hiển thị CẢ 2**: failure_busted làm CHÍNH (headline), weak_move_5pct làm phụ (expandable) + tooltip |
| 5 | Ngưỡng cỡ mẫu | **GIỮ n≥30** (hồ sơ 1 mã); publication chapter = n≥100 — 2 ngưỡng khác nhau |
| 6 | MAE>80% | **DÙNG 80% cho V3** + thu phân phối MAE trong M4, đề xuất lại ở V3.1 nếu cần |

## 4. Thứ tự rủi ro (GLM)

1. **R4.1** Đảo thứ hạng pattern → xử TRƯỚC M1 (M0): snapshot bắt buộc
2. **R4.4** Stale tạm thời → xử TRƯỚC M1 (M0): đóng băng UI web_v3
3. **R4.6** Agent song song sửa detector → đặc tả read-only + đối chiếu sau mỗi mốc
4. **R4.3** failure không khớp literature → tại M2+M5: chênh ≤2× chấp nhận "khác thị trường", >2× audit 10 event
5. **R4.2** Số events giảm → tại M4: report n_before/n_after/drop_reason, chỉ ẩn n<5
6. **R4.5** Nhầm "kiểm định" = "mua được" → tại M3: nhãn "kiểm định như tài liệu tham khảo" + non-advice boundary

## 5. Truyền thông với chủ đầu tư (GLM đề xuất — 6 phần cố định sau mỗi mốc)

1. Mục tiêu mốc (1 câu đời thường) · 2. Kết quả chính (3-5 dòng, số cụ thể) · 3. Thay đổi so với trước (bảng before→after) · 4. Ảnh hưởng tới chủ đầu tư (1-2 câu) · 5. Rủi ro phát sinh (1 câu) · 6. Câu hỏi quyết định (nếu cần). KHÔNG show code; mọi số kèm "nghĩa là gì". Chủ đầu tư ký 1 lần cuối sau M3.

## 6. 12 điều kiện nghiệm thu (GLM đề xuất — để K3 duyệt)

| # | Điều kiện | Đạt khi |
|---|---|---|
| 1 | K1-K6 PASS cho ≥5 pattern Nấc 3 | 5/5 PASS |
| 2 | Benchmark before/after M1-M4 | 4 report trong `backups/v3_benchmark/` |
| 3 | 1 lệnh refresh pipeline | Exit 0, dashboard refresh |
| 4 | web_v3 → web | Hết banner staging |
| 5 | Cross-check GLM độc lập M1-M4 | 4/4 PASS (không đọc kết quả thi công trước) |
| 6 | Non-advice boundary mọi nơi | Header + footer + tooltip Nấc |
| 7 | Nấc 1 ẩn mặc định + toggle | Click thử hoạt động |
| 8 | Mã delisted minh bạch | `delisted_symbols` + toggle "ẩn mã delisted" |
| 9 | Báo cáo MAE drop_rate từng pattern | Có dòng `events_dropped_split_suspect` |
| 10 | failure_busted headline + weak_move_5pct phụ | Click → expand + giải thích |
| 11 | target_hit kèm dist_pct + cảnh báo <3% | Mọi pattern có "(mục tiêu TB X%)" |
| 12 | Chủ đầu tư duyệt UI thực tế (chrome-devtools) | Chủ đầu tư xác nhận "được" |

**Ghi chú cuối:** M5 không chặn nghiệm thu dashboard — chỉ nâng thêm pattern lên Nấc 3 sau này.

---

## ⏳ Chờ K3 thật duyệt (sau restart ZCode)

Bản này dùng làm INPUT cho K3 (provider OpenCode Go, 1 lần gọi duy nhất). K3 sẽ: phán quyết 3 điểm bổ sung, duyệt/điều chỉnh mốc + 6 quyết định + 12 điều kiện → ra kế hoạch FINAL → lưu thành `07-k3-final-plan.md`.
