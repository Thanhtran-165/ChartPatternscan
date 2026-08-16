# Đợt B — Bảng TRƯỚC/SAU 3 chương (rescan toàn thị trường 16/08/2026)

TRƯỚC = `events.csv.bak_pre_dotb` (code cũ). SAU = rescan bằng `target_hit_core` full precision.

Nguồn TRƯỚC từng chương:
- double_tops: events.csv.bak_vintage_pre_rerun (đời code cũ, double không thuộc EVENT_SOURCES nên backup đợt B không phủ — cùng tập event_id với bản SAU, khác mỗi cột đánh giá)
- bump_and_run_reversal_bottoms: events.csv.bak_pre_dotb (backup ngay trước rescan đợt B)
- inside_day: events.csv.bak_pre_dotb (backup ngay trước rescan đợt B)

| Chương | Phạm vi | N TRƯỚC | N SAU | Hit TRƯỚC | Hit SAU | Δ pp | Fail5 TRƯỚC | Fail5 SAU | Median dist TRƯỚC | SAU |
|---|---|---|---|---|---|---|---|---|---|---|
| double_tops | toàn bộ | 782 | 783 | 50.77% | 69.48% | +18.71 | 40.92% | 40.74% | 6.43% | 3.77% |
| double_tops | premium+standard | 188 | 208 | 70.21% | 84.62% | +14.41 | 20.74% | 22.6% | 7.04% | 4.42% |
| bump_and_run_reversal_bottoms | toàn bộ | 2116 | 1447 | 57.37% | 41.11% | -16.26 | 12.29% | 13.49% | 18.59% | 28.69% |
| bump_and_run_reversal_bottoms | premium+standard | 1189 | 860 | 56.01% | 41.98% | -14.03 | 11.27% | 11.16% | 18.92% | 28.02% |
| inside_day | toàn bộ | 14282 | 14282 | 59.0% | 58.98% | -0.02 | 61.74% | 61.73% | 2.38% | 2.38% |
| inside_day | premium+standard | 4442 | 4446 | 69.18% | 69.14% | -0.04 | 56.33% | 56.34% | 2.09% | 2.09% |

## Nhóm BARR Bottom dist > 110% bị cổng loại (sau rescan)

- Tổng events BARR bottoms: **1447**
- Bị loại dist > 110%: **0** (0.0%)
- Phân bố tier nhóm bị loại: {}
- Khoảng dist nhóm bị loại: None% – None%
- Hit rate toàn bộ: 41.11% · sau loại: 41.11% (tác động +0.00 pp)
- Ghi chú: Nhóm dist>110 bị cổng publication loại theo quyết định đợt A2 (target quá xa so khoảng cách đo sách). Ngoài ra events không vào được chuỗi neo pivot lead-in đã bị detector loại TRƯỚC khi ghi events.csv — xem scanner/audits/barr_old_high_audit.json.
