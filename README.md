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
- `scanner/pivot_detector.py`: phát hiện pivot highs/lows + lọc spacing để giảm nhiễu
- `scanner/digitized_pattern_engine.py`: scanners theo các bộ `--pattern-set`:
  - `digitized`: **spec-driven** đọc từ `extraction_phase_1/digitization/patterns_digitized/*_digitized.json` (nếu có) để cover **toàn bộ digitized specs** (hiện có **31** keys)
  - `bulkowski_53`: Bulkowski Part One (53 chapters) (tách biến thể theo chapter; nếu có digitized specs thì dùng spec-driven, nếu không sẽ fallback một số built-in proxies)
  - `bulkowski_53_strict`: phiên bản “spec-anchored” của `bulkowski_53` (khi đã digitize đủ 53/53 chapters thì **trùng với** `bulkowski_53`)
  - `bulkowski_strict_ohlcv`: `bulkowski_53_strict` + `event_ohlcv` → **55 patterns** (53 chart + 2 event-OHLCV)
  - `event_ohlcv`: Event patterns “ngoại lệ” có thể định nghĩa chỉ từ OHLCV (**Dead‑Cat Bounce**, **Dead‑Cat Bounce (Inverted)**)
  - `bulkowski_55_ohlcv`: `bulkowski_53` + `event_ohlcv`
- `scanner/pattern_scanner.py`: orchestrator (normalize → pivots → scan). Nếu thiếu digitized specs (repo public), fallback về legacy MVP scanners
- `scanner/post_breakout_analyzer.py`: đo thống kê hậu breakout (look-ahead **theo từng pattern** nếu có digitized specs; mặc định 252 bars), thời gian đo theo **calendar days**, và variant `AA/AE/EA/EE` cho Double Tops
- `scanner/results_db.py`: persist kết quả ra **DB riêng** (không ghi vào DB giá nguồn) + `run_id` + index
- `scanner/run_full_scan.py`: chạy scan full DB và lưu kết quả + thống kê tổng hợp
- `scanner/audit_kpi.py`: audit KPI + compliance theo digitized specs
- `scanner/bulkowski_report.py`: tạo báo cáo thống kê kiểu Bulkowski (median + bull/bear regime 18 tháng)

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

Chạy theo “universe” (ví dụ VN100):

```bash
python3 scanner/run_full_scan.py --db /path/to/your_price.db --universe-index VN100
```

Scan theo giai đoạn (phục vụ calibration/validation):

```bash
python3 scanner/run_full_scan.py \
  --pattern-set bulkowski_53_strict \
  --db /path/to/your_price.db \
  --results-db scan_results/calib.sqlite \
  --universe-index VN30 \
  --date-from 2018-01-01 \
  --date-to 2021-12-31 \
  --date-anchor breakout_or_end \
  --warmup-bars 300
```

Bulkowski-style (khuyến nghị cho nghiên cứu):

```bash
python3 scanner/run_full_scan.py --db /path/to/your_price.db --results-db scan_results/full_bulkowski.sqlite
python3 scanner/bulkowski_report.py --results-db scan_results/full_bulkowski.sqlite --price-db /path/to/your_price.db --index-symbol VN30 --out-md scan_results/bulkowski_report.md
```

Bulkowski Part One (53 chart patterns theo chapter của sách):

```bash
python3 scanner/run_full_scan.py --pattern-set bulkowski_53 --db /path/to/your_price.db --results-db scan_results/bulkowski_53.sqlite
python3 scanner/bulkowski_report.py --results-db scan_results/bulkowski_53.sqlite --price-db /path/to/your_price.db --index-symbol VN30 --out-md scan_results/bulkowski_53_report.md
```

Bulkowski Part One (strict: chỉ những pattern có digitized spec anchor):

```bash
python3 scanner/run_full_scan.py --pattern-set bulkowski_53_strict --db /path/to/your_price.db --results-db scan_results/bulkowski_53_strict.sqlite
python3 scanner/bulkowski_report.py --results-db scan_results/bulkowski_53_strict.sqlite --price-db /path/to/your_price.db --index-symbol VN30 --out-md scan_results/bulkowski_53_strict_report.md
```

Event patterns chỉ từ OHLCV (không cần “event DB”):

```bash
python3 scanner/run_full_scan.py --pattern-set event_ohlcv --db /path/to/your_price.db --results-db scan_results/event_ohlcv.sqlite
```

Strict chart patterns + Event OHLCV (tổng 55):

```bash
python3 scanner/run_full_scan.py --pattern-set bulkowski_strict_ohlcv --db /path/to/your_price.db --results-db scan_results/bulkowski_strict_ohlcv.sqlite
```

Bulkowski 53 + Event OHLCV (tổng 55):

```bash
python3 scanner/run_full_scan.py --pattern-set bulkowski_55_ohlcv --db /path/to/your_price.db --results-db scan_results/bulkowski_55_ohlcv.sqlite
```

Group variants theo canonical/spec key (để tổng hợp biến thể theo “cùng một pattern”):

```bash
python3 scanner/bulkowski_report.py --results-db scan_results/bulkowski_53.sqlite --price-db /path/to/your_price.db --index-symbol VN30 --group-by canonical_key --out-md scan_results/bulkowski_53_report_canonical.md
```

Góc nhìn theo **cổ phiếu** (mã XXX): pattern nào xuất hiện (bull/bear) và hiệu suất:

```bash
python3 scanner/symbol_report.py --results-db scan_results/bulkowski_53.sqlite --price-db /path/to/your_price.db --index-symbol VN30 --symbol FPT --out-md scan_results/FPT_symbol_report.md
```

Sinh “book” tiếng Việt (Markdown + tùy chọn PDF) theo cấu trúc Bulkowski-style:

```bash
# (tuỳ chọn) dùng DeepSeek để viết lời bình / narrative
cp .env.example .env
# rồi điền DEEPSEEK_API_KEY trong `.env` (hoặc export trực tiếp như bên dưới)
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com"   # mặc định
export DEEPSEEK_MODEL="deepseek-chat"                 # mặc định

python3 scanner/build_book_vi.py \
  --results-db-valid scan_results/valid_2022_2025_vn30_eval.sqlite \
  --results-db-calib scan_results/calib_2018_2021_vn30_eval.sqlite \
  --price-db vietnam_stocks.db \
  --index-symbol VN30 \
  --out-dir scan_results/book_vi
```

Ghi chú:
- `scanner/build_book_vi.py` sẽ tự load `.env` (repo root) nếu có.
- Nếu thiếu `DEEPSEEK_API_KEY` và chưa có cache thì chương sẽ giữ placeholder narrative.
- `--skip-ai` sẽ không gọi DeepSeek (nhưng vẫn reuse cache narrative nếu đã sinh trước đó).
- Nếu máy chưa có `pandoc`/LaTeX thì script vẫn sinh `book.md` (PDF chỉ build khi toolchain sẵn có).
- Có thể test 1 pattern: `--patterns broadening_wedges_ascending`

Review nhanh chất lượng mapping/metadata của các `--pattern-set` (không scan dữ liệu):

```bash
python3 scanner/review_pattern_sets.py --out-md scan_results/pattern_set_review.md
```

Ghi chú:
- `--overlap-policy bulkowski` (mặc định) ưu tiên pattern timeframe lớn hơn khi overlap.
- `--min-breakout-price` cho phép loại các breakout ở mức giá quá thấp (tuỳ dữ liệu/thị trường).

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

- Repo public **không** đi kèm `extraction_phase_1/` (digitized specs + dữ liệu trích xuất) vì lý do bản quyền.
- Nếu bạn có bộ digitized specs ở local: có thể scan **31/31** (mặc định) hoặc scan **Bulkowski 53** bằng `--pattern-set bulkowski_53`.
- `event_ohlcv` là bản “OHLCV‑proxy” cho event patterns (không thay thế hoàn toàn nghiên cứu dựa trên cơ sở dữ liệu sự kiện).
- Bull/bear regime phụ thuộc `--index-symbol` có mặt trong `stock_price_history` (ví dụ VN30/VN100). Với DB hiện tại, các index series bắt đầu từ **2017-08-24** → các pattern trước mốc này sẽ có regime `unknown`.
