# ISS-001 — Full-reload không thay thế hàng cũ (stale rows) trong latest.sqlite

- **Trạng thái:** OPEN — chặn refresh kế tiếp của `latest.sqlite`
- **Phát hiện:** 16/08/2026 (đợt B, theo điều kiện Sol HIGH-3 / db_manifest)
- **Mức độ:** HIGH (toàn vẹn dữ liệu nguồn cho mọi bộ scan xuất bản)
- **Liên quan:** `artifacts/scanner_v2/db_manifest.json` (mục `open_issues`)

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
