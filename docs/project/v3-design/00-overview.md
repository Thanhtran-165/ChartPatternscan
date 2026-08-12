# Nghiên cứu mẫu hình giá V3 — Đặc tả thiết kế

> **Phiên bản đặc tả:** v3.0-draft
> **Ngày:** 12/08/2026
> **Tác giả:** GLM-5.2 (provider Z.ai) — vai trò verify/thiết kế
> **Session mẹ:** `sess_1c356bbb-d126-4ea6-9508-ad490bca4ef1`
> **Phạm vi ghi:** chỉ thư mục `docs/project/v3-design/` (read-only với phần còn lại của repo)

---

## Mục lục

| File | Nội dung |
|---|---|
| `00-overview.md` | File này — mục lục + tóm tắt 1 trang cho chủ đầu tư |
| `01-vision-scope.md` | Tầm nhìn V3, phạm vi, khác gì V2, tiêu chí "đủ chuẩn phát hành" từng pattern |
| `02-architecture-bottlenecks.md` | 9 điểm nghẽn hiện tại + kiến trúc pipeline mới + tự động hóa rebuild/split |
| `03-measurement-standards.md` | Chuẩn đo lường V3: bảng lookahead chuẩn 31 family, định nghĩa failure/target ĐÚNG Bulkowski, cỡ mẫu, chất lượng dữ liệu |
| `04-governance-roadmap.md` | Publication governance mở rộng 55 pattern (draft→candidate→final), 5 mốc triển khai, rủi ro + giảm thiểu |

---

## Tóm tắt 1 trang (dành cho chủ đầu tư)

### Vấn đề cốt lõi
Dashboard `market_stats_v2` (tab "Lớp mẫu hình phụ trợ") **chưa nghiệm thu được** vì dữ liệu mẫu hình giá (`stock_pattern_profiles.json`) **không đáng tin ở 3 lớp**:

1. **Đo sai cửa sổ thời gian** (lookahead) — detector hardcode đo 60–120 phiên cho mọi pattern, trong khi chuẩn Bulkowski yêu cầu 10–252 phiên tuỳ pattern. Ví dụ inside_day đo 60 phiên (gấp 6 lần chuẩn 10 phiên) → số MFE bị "phồng" bởi drift 3 tháng (median 15% thay vì ~3% chuẩn).
2. **Định nghĩa "thất bại" sai** — hiện tính `failure = MFE < 5%` (move không đạt 5%), còn chuẩn Bulkowski là "giá quay lại vượt đáy/đỉnh mẫu hình trước khi chạm mục tiêu". Hai định nghĩa này cho 2 con số hoàn toàn khác nhau (bull_flags 25% vs ~5.5%).
3. **Trộn lẫn pattern đã kiểm định và bản nháp** — chỉ ~14/55 pattern qua publication gate, ~41 pattern còn là DRAFT nhưng dashboard hiển thị tất cả như nhau.

### 9 điểm nghẽn (chi tiết ở file 02)
Lookahead lệch chuẩn hệ thống · Định nghĩa failure sai · target_hit không kèm độ lớn mục tiêu · Cap `max_events_per_symbol` bão hoà · Dữ liệu bẩn MAE>80% (split chưa điều chỉnh) · Stale split artifact (chạy tay) · Publication trộn draft/final · Survivorship (mã delisted) · Pipeline thủ công không tự động hoá.

### Kiến trúc đề xuất (chi tiết ở file 02)
```
[source PDF Bulkowski] → [digitized spec: lookahead/failure/target CHUẨN]
        ↓
[detector v3: đọc lookahead từ spec, KHÔNG hardcode]
        ↓
[events.csv: kèm evaluated_bars, target_dist_pct, publication_status, data_quality_flag]
        ↓
[build_profiles v3: filter MAE>80%, label cỡ mẫu, gắn nhãn publication]
        ↓
[auto-split: chạy tự động sau mỗi scan — không còn chạy tay]
        ↓
[dashboard: nhãn "đã kiểm định/bản nháp", cảnh báo cỡ mẫu, cảnh báo lookahead, chú thích định nghĩa]
```

### 5 mốc triển khai (chi tiết ở file 04)
| Mốc | Tên | Đầu ra kiểm chứng được | Phụ thuộc |
|---|---|---|---|
| **M1** | Sửa lookahead 24 detector theo spec digitized | Bảng đối chiếu before/after: median MFE inside_day giảm từ ~15% → ~3% | Lớp A đang làm |
| **M2** | Định nghĩa lại failure/target chuẩn Bulkowski | Report: failure_rate inside_day/cup/horn/flag so sánh với spec | M1 |
| **M3** | Publication status từ manifest → artifact web | Mỗi pattern trên dashboard có nhãn rõ draft/candidate/final | Lớp B |
| **M4** | Tự động hoá rebuild + split + filter dữ liệu bẩn | 1 lệnh chạy toàn pipeline; báo cáo MAE>80% bị loại | M1-M3 |
| **M5** | Đọc PDF gốc nâng chuẩn pattern ưu tiên | Bảng failure/target/sample từ PDF cho top-10 pattern | Lớp C, M2 |

### Tiêu chí nghiệm thu V3 ("đủ chuẩn")
Một pattern được hiển thị như **"đã kiểm định"** trên dashboard khi và chỉ khi:
1. Lookahead detector = lookahead spec digitized (chênh ≤ 5%)
2. Failure/target tính bằng định nghĩa Bulkowski chuẩn
3. publication_status ∈ {`publication_final`, `active`}
4. Cỡ mẫu n ≥ 30 (VN cash equities) HOẶC có cảnh báo "mẫu mỏng" hiện rõ
5. Không còn event có MAE>80% (dữ liệu bẩn đã filter)
6. target_hit luôn kèm `target_dist_pct` (độ lớn mục tiêu)

Pattern nào không đủ → hiển thị nhãn **"bản nháp — chưa kiểm định"** + cảnh báo cụ thể.

### Rủi ro chính (chi tiết ở file 04)
- Đổi lookahead làm **đảo thứ hạng pattern** (inside_day, scallop sẽ tụt hạng mạnh)
- **Số events giảm** khi filter dữ liệu bẩn + giảm cap bão hoà
- Cần benchmark so sánh trước/sau để tránh "sửa một chỗ hỏng chỗ khác"

### Giới hạn của đặc tả này
- Đặc tả dừng ở mức **thiết kế/spec** — không sửa code (một agent khác đang sửa `scanner/v2/*.py` song song).
- Số liệu đối chiếu lấy từ digitized JSON (31 family) + note thẩm định session trước. Chưa đọc lại PDF gốc (bản quyền, chỉ liệt kê).
- "55 pattern" = số lượng pattern key trong artifact web; thực tế 31 family digitized + biến thể (double AA/AE/EA/EE, HSB complex, 4 scallop, 4 broadening...) = ~55 key.
