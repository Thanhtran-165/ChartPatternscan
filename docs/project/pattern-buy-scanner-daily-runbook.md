# Pattern BUY Scanner Daily Runbook

Mục tiêu của job này là chạy radar mẫu hình BUY độc lập với daily pipeline lớn.
Nó không thay daily pipeline hiện tại và không được gắn vào R2/TApro/Market flow.

## Lịch chạy

Mặc định LaunchAgent chạy lúc 17:00 mỗi ngày theo giờ máy.

```bash
bash scripts/pattern_buy_scanner_launchd.sh install
```

Kiểm tra:

```bash
bash scripts/pattern_buy_scanner_launchd.sh status
bash scripts/pattern_buy_scanner_launchd.sh logs
```

Chạy thử thủ công:

```bash
PATTERN_BUY_SEND=0 bash scripts/pattern_buy_scanner_launchd.sh run
```

Gửi thật:

```bash
PATTERN_BUY_SEND=1 REALTIME_SCAN_EMAIL_TO=stevetransg@gmail.com bash scripts/pattern_buy_scanner_launchd.sh run
```

## Flow vận hành

```text
Acquire lock
→ refresh/audit OHLCV cache riêng của scanner
→ regenerate market_stats artifact nếu cần
→ quét BUY_PULLBACK từ event artifacts
→ quét BUY_SETUP trước phá vỡ trong VN100
→ tạo email summary
→ tạo PDF chi tiết
→ cập nhật history ledger
→ gửi email nếu PATTERN_BUY_SEND=1
```

## Các biến cấu hình chính

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `PATTERN_BUY_RUN_HOUR` | `17` | Giờ chạy LaunchAgent |
| `PATTERN_BUY_RUN_MINUTE` | `00` | Phút chạy LaunchAgent |
| `PATTERN_BUY_SEND` | `1` | `1` gửi mail, `0` chỉ tạo artifact |
| `REALTIME_SCAN_EMAIL_TO` | `stevetransg@gmail.com` | Người nhận email |
| `PATTERN_BUY_LOOKBACK_DAYS` | `7` | Cửa sổ đọc BUY_PULLBACK |
| `PATTERN_BUY_LIMIT_EACH` | `20` | Số dòng tối đa mỗi nhóm email |
| `PATTERN_BUY_SETUP_LIMIT_PER_PATTERN` | `8` | Số BUY_SETUP tối đa mỗi pattern |
| `PATTERN_BUY_REFRESH_STALENESS_DAYS` | `0` | Ngưỡng stale cho data refresh; job 17:00 phải thử làm mới nếu DB chưa có dữ liệu ngày hiện tại |

## Artifact chính

| Artifact | Đường dẫn |
|---|---|
| Log mới nhất | `logs/pattern_buy_scanner/latest.log` |
| Email text/html | `artifacts/realtime_scan/latest/email/` |
| PDF chi tiết | `artifacts/realtime_scan/latest/email/realtime_scan_detail.pdf` |
| History ledger | `artifacts/realtime_scan/history/` |
| Data refresh report | `artifacts/realtime_scan/latest/data_refresh/data_refresh_report.json` |

## Nguyên tắc an toàn

- Job này có lock riêng, không chạy chồng phiên.
- Nếu provider refresh xong nhưng DB vẫn cũ hơn ngưỡng freshness, job strict sẽ dừng với trạng thái `REFRESHED_STALE` thay vì gửi mail như một phiên sạch.
- Chỉ tập trung BUY setup và BUY pullback; không gửi tín hiệu bán/short.
- Nếu không có ứng viên VN100 phù hợp thì để trống, không backfill từ ngoài VN100.
- Email/PDF là radar ứng viên, không phải khuyến nghị giao dịch.
