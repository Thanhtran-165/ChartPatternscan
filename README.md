# ChartPatternscan

Nghiên cứu và quét (scan) mô hình giá trên dữ liệu **OHLCV** quy mô lớn (SQLite), kèm lớp **đánh giá hậu breakout** (look-ahead) để tổng hợp thống kê.

## Tuyên bố học thuật, phạm vi và trích dẫn

- Dự án phục vụ **nghiên cứu/giáo dục**, không phải tư vấn đầu tư.
- Lớp **Post-Breakout Analyzer** sử dụng **dữ liệu tương lai** (look-ahead) để đo *failure/ultimate/throwback/target/MFE/MAE*; **không** dùng để ra quyết định giao dịch thời gian thực.
- Nhiều khái niệm/thuật ngữ và cách đo lường trong lĩnh vực chart patterns thường được chuẩn hoá theo tài liệu kinh điển. Nếu bạn dùng dự án này trong báo cáo học thuật, hãy **trích dẫn nguồn** bên dưới.

**Tài liệu tham khảo chính (primary reference):**
- Thomas N. Bulkowski, *Encyclopedia of Chart Patterns*, 2nd Edition, Wiley. ISBN: 978-0-471-66826-8.

**Lưu ý bản quyền:**
- Repo này **không** phân phối sách/tài liệu có bản quyền hoặc các bản trích xuất dung lượng lớn từ sách.
- Nếu bạn có bộ “digitized specs”/pattern definitions được trích xuất từ sách hoặc nguồn có bản quyền: hãy đảm bảo **quyền sử dụng/phân phối** trước khi public.

## Kiến trúc

- `scanner/ohlcv_normalizer.py`: làm sạch OHLCV (NULL, high/low đảo, clamp open/close, loại bỏ giá <=0), tạo cột dẫn xuất (ATR, volume_ma, volume_ratio…)
- `scanner/pivot_detector.py`: phát hiện pivot highs/lows + lọc spacing để giảm nhiễu, đảm bảo chuỗi pivot alternating
- `scanner/pattern_scanner.py`: scanner MVP cho một số pattern (hiện có: `double_tops`, `head_and_shoulders_top`)
- `scanner/post_breakout_analyzer.py`: đo thống kê hậu breakout (look-ahead 252 bars) + variant `AA/AE/EA/EE` cho Double Tops
- `scanner/results_db.py`: persist kết quả ra **DB riêng** (không ghi vào DB giá nguồn) + `run_id` + index
- `scanner/run_full_scan.py`: chạy scan full DB và lưu kết quả + thống kê tổng hợp

## Yêu cầu dữ liệu

Mặc định đọc SQLite table `stock_price_history` với các cột:
- `symbol`, `time` (hoặc `date`), `open`, `high`, `low`, `close`, `volume`

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy scan (khuyến nghị)

Chạy full scan và lưu sang DB kết quả riêng:

```bash
python3 scanner/run_full_scan.py \
  --db /path/to/your_price.db \
  --results-db scan_results/pattern_scans.sqlite
```

Test nhanh (giới hạn số mã):

```bash
python3 scanner/run_full_scan.py --db /path/to/your_price.db --limit 50
```

Chỉ scan một số pattern:

```bash
python3 scanner/run_full_scan.py --patterns double_tops,head_and_shoulders_top
```

## Query kết quả (SQLite)

Ví dụ truy vấn nhanh:

```sql
-- Tổng detections theo pattern
SELECT pattern_name, COUNT(*) AS n
FROM pattern_detections
WHERE run_id = '...'
GROUP BY pattern_name
ORDER BY n DESC;

-- Thống kê hậu breakout theo variant (Double Tops)
SELECT variant, COUNT(*) AS n, AVG(max_favorable_excursion_pct) AS avg_mfe
FROM post_breakout_results
WHERE run_id = '...' AND pattern_name = 'double_tops' AND variant IS NOT NULL
GROUP BY variant
ORDER BY n DESC;
```

## Tái lập (reproducibility)

Mỗi lần chạy tạo `run_id` và lưu:
- config scan/eval
- detections
- post-breakout results (look-ahead)
- run_statistics (JSON)
- scan_errors (stacktrace)

## Giới hạn hiện tại

- Scanner MVP hiện mới cover một phần pattern (ưu tiên chất lượng pipeline + persist + thống kê). Có thể mở rộng dần theo bộ digitized specs.

