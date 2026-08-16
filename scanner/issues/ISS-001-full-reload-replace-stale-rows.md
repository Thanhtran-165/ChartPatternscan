# ISS-001 — Full-reload không thay thế hàng cũ (stale rows) trong latest.sqlite

- **Trạng thái:** RESOLVED 16/08/2026 — vá `--replace` + guard mất dữ liệu, 5/5 test đơn vị PASS (chi tiết cuối file)
- **Phát hiện:** 16/08/2026 (đợt B, theo điều kiện Sol HIGH-3 / db_manifest)
- **Mức độ:** HIGH (toàn vẹn dữ liệu nguồn cho mọi bộ scan xuất bản)
- **Liên quan:** `artifacts/scanner_v2/db_manifest.json` (mục `open_issues` — bản ghi giữ nguyên trạng thái mở TẠI THỜI ĐIỂM xuất bản; đóng dấu tại đây)

## Hiện tượng

Script refresh giá `market_stats/update_latest_stock_ohlcv.py`
(repo `~/dev/market_stats_v2`) cập nhật DB bằng UPSERT:

```sql
INSERT INTO stock_price_history(symbol, time, open, high, low, close, volume, source)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(symbol, time) DO UPDATE SET open=excluded.open, …
```

Cơ chế này **chỉ thêm/cập nhật**, không bao giờ **xóa**:

1. **Hàng sai bị nguồn thu hồi** — nếu provider (VCI/VND/KBS/MAS) sửa lại hoặc
   rút bỏ một phiên đã phát hành, hàng cũ sai vẫn nằm trong DB.
2. **Mã bị hủy niêm yết / đổi nguồn cấp** — refresh universe gần nhất chỉ chạm
   1.348/1.599 mã; lịch sử của ~250 mã còn lại không được xác nhận lại.
3. **Điều chỉnh lại toàn tuyến** — updater đã có cơ chế overlap-check + full
   reload từng mã khi phát hiện lệch điều chỉnh (`VERIFY_OVERLAP_DAYS_DEFAULT=10`,
   `OVERLAP_CLOSE_TOLERANCE=0.005`), nhưng cơ chế này chỉ chạy cho các mã
   TRONG refresh universe của lần đó; mã không được refresh thì bậc thang giá
   cũ (nếu có) vẫn giữ nguyên.

## Tác động

- Mọi thống kê sách Edition 2 quét từ `latest.sqlite` có thể chứa hàng không
  còn được nguồn xác nhận.
- Đã kiểm 16/08/2026, đo trực tiếp trên snapshot đóng băng `latest.sqlite.dotb_20260815`
  (SHA-256 `9c0164b01ac9d7bf48d31284a2a57ce78799e84590734ce6a678483470ca535d`,
  1.599 mã / 4.255.894 rows): DB có **4.556 hàng `close<=0` trên 223 mã** (đánh dấu
  delisted/halted của nguồn) — các detector loại tại chỗ khi đọc, không nhiễm
  events xuất bản; nhưng đây là bằng chứng các hàng "không còn giá trị giao
  dịch" vẫn tồn tại trong DB. (Số cũ 5.026/242 là bản đo trộn NULL trên DB sống
  trước khi chốt snapshot — xem `db_manifest.json` mục `close_zero_handling`.)

## Ràng buộc đặt ra

- **CHẶN refresh kế tiếp** của `latest.sqlite` (không chạy
  `update_latest_stock_ohlcv.py`) cho tới khi issue này xử lí xong, để bộ
  bằng chứng đợt B (SHA-256 `37bde6e186e68381…`) không bị trôi giữa chừng.

## Đề xuất xử lí (chưa thi công)

1. Thêm chế độ `--replace`: nạp lịch sử mới của từng mã vào bảng tạm rồi
   `DELETE`+chèn lại (hoặc `DROP PARTITION` theo symbol) thay vì UPSERT dồn.
2. Audit sau refresh: số hàng mỗi mã trước/sau; hàng nào biến mất thì phải
   khớp danh sách mã được nguồn thu hồi.
3. Ghi nhận quyết định cho hàng close<=0: giữ (dấu halted) hay lọc tại nguồn —
   hiện đang lọc tại detector (xem `db_manifest.json` mục close_zero_handling).

## Lịch sử

- 16/08/2026: mở issue (đợt B — GLM-5.3), ghi vào `db_manifest.json`.
- 16/08/2026 chiều: **RESOLVED** — anh duyệt xử lý trước đợt nạp dữ liệu mới.

## Cách đã xử lý (16/08/2026)

Sửa `market_stats/update_latest_stock_ohlcv.py` (repo `~/dev/market_stats_v2`, branch v3/main):

1. **Hàm `replace_symbol_rows`** — DELETE toàn bộ hàng cũ của từng mã rồi INSERT lại, trong MỘT transaction (đứt giữa chừng rollback, không mất dữ liệu cũ). Áp dụng cho **đường nạp lại toàn bộ (ADJUST)** — nơi stale sinh ra: UPSERT dồn khiến hàng bị nguồn thu hồi nằm lại mãi.
2. **Guard mất dữ liệu `REPLACE_MIN_RATIO = 0.5`** — nguồn trả <50% số hàng hiện có của một mã → nghi provider lỗi/thiếu dữ liệu → giữ nguyên UPSERT cho mã đó + báo `REPLACE-SKIP` qua stderr (chống kịch bản xóa sạch vì nguồn hụt).
3. **Cờ `--replace` opt-in, mặc định TẮT** — hành vi refresh hằng ngày qua server 8766 KHÔNG đổi. Đợt nạp dữ liệu chủ đích kế tiếp chạy trực tiếp với `--replace` để dọn stale.
4. **Audit**: stats JSON ghi thêm `replace_symbols` / `replace_deleted_rows` / `replace_skipped` (số mã đã thay, tổng hàng cũ bị dọn, số mã bị guard chặn) — meta `market_stats_latest_refresh` trong DB đọc được từ UI refresh-status.

**Test đơn vị 5/5 PASS** (DB giả /tmp, không đụng DB sống): T1 replace dọn 2 hàng stale + mã khác nguyên vẹn · T2 guard từ chối khi nguồn trả 30/100, UPSERT giữ 130 · T3 rollback khi dữ liệu có ngày trùng · T4 rows rỗng · T5 mã mới (old=0) replace bình thường.

**Quyết định hàng close<=0 (đề xuất 3):** GIỮ NGUYÊN trong DB (đóng vai dấu delisted/halted của nguồn), detector lọc tại chỗ khi đọc — duy trì hiện trạng như `db_manifest.json` mục `close_zero_handling` đã kê khai với Sol. Backup trước vá: `update_latest_stock_ohlcv.py.bak_pre_iss001_20260816`.

**Lệnh chạy đợt nạp kế tiếp (chủ đích):**
```bash
cd ~/dev/market_stats_v2 && .venv-vnstock-sponsor311/bin/python -m market_stats.update_latest_stock_ohlcv --replace
# hoặc qua server: thêm --replace khi gọi trực tiếp updater; job daily-refresh (qua API, không cờ) giữ hành vi cũ
```
