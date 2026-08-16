# ISS-001 — Nhật ký xử lý (resolution log)

> File `ISS-001-full-reload-replace-stale-rows.md` nằm trong danh sách evidence đã
> bind của `edition2_release_dossier.json` (SHA-256 `e996ec9b…`, GO cuối của Sol
> 16/08) nên **giữ nguyên văn, không sửa nữa** — mọi cập nhật trạng thái ghi tại đây.

- **16/08/2026 chiều: ISS-001 RESOLVED** (anh duyệt xử lý trước đợt nạp dữ liệu mới).
- Vá tại repo `~/dev/market_stats_v2`, branch `v3/main`, commit `f362794f4`:
  1. `market_stats/update_latest_stock_ohlcv.py` thêm hàm `replace_symbol_rows` —
     DELETE toàn bộ hàng cũ của từng mã rồi INSERT lại trong MỘT transaction
     (đứt giữa chừng rollback). Áp dụng cho **đường nạp lại toàn bộ (ADJUST)** —
     nơi stale sinh ra; đường incremental giữ nguyên.
  2. Guard mất dữ liệu `REPLACE_MIN_RATIO = 0.5`: nguồn trả <50% số hàng hiện có
     của một mã → nghi provider thiếu → giữ UPSERT cho mã đó, báo `REPLACE-SKIP`.
  3. Cờ `--replace` opt-in (mặc định TẮT) — hành vi refresh hằng ngày qua server
     8766 KHÔNG đổi. Đợt nạp chủ đích kế tiếp chạy trực tiếp với `--replace`.
  4. Audit vào stats meta: `replace_symbols` / `replace_deleted_rows` / `replace_skipped`.
- **Unit test 5/5 PASS** (DB giả /tmp): dọn stale + mã khác nguyên vẹn · guard từ
  chối 30/100 giữ UPSERT · rollback khi trùng PK · rows rỗng · mã mới old=0 OK.
- **Quyết định hàng close<=0:** GIỮ NGUYÊN trong DB (dấu delisted/halted), detector
  lọc tại chỗ — đúng hiện trạng `db_manifest.json` mục `close_zero_handling` đã kê khai.
- Backup trước vá: `update_latest_stock_ohlcv.py.bak_pre_iss001_20260816` (repo dev).

## Lệnh chạy đợt nạp dữ liệu kế tiếp (chủ đích)

```bash
cd ~/dev/market_stats_v2 && .venv-vnstock-sponsor311/bin/python -m market_stats.update_latest_stock_ohlcv --replace
```

Job daily-refresh (qua API server, không cờ) giữ hành vi cũ — không đổi.

## Lịch sử sơ suất (minh bạch)

- 16/08: sửa nhầm file gốc ISS-001 (commit `eccd4a5`) → tự bắt ra rằng file thuộc
  evidence bind → restore nguyên văn về SHA `e996ec9b…` (khớp dossier), chuyển
  toàn bộ ghi nhận sang file này. Sau restore, mọi artifact bind khớp lại GO.
