# Đợt B — Recompute độc lập 5 chương từ DB giá full precision

Recompute độc lập từ snapshot DB giá raw (latest.sqlite.dotb_20260815, SHA trong db_manifest) — không dùng post_breakout_path.csv hay cột target_hit; cửa sổ forward theo evaluated_bars của từng event (trần measurement_registry); chương tuần (horn) resample W-FRI đúng detector; failure_5pct = MFE full < 5.0.

| Chương | Events | So sánh | Lệch hit | Lệch fail5 | Parity | Hit 1.0x recompute |
|---|---|---|---|---|---|---|
| double_tops | 783 | 783 | 0 | 0 | PASS | 69.48% |
| bump_and_run_reversal_bottoms | 1447 | 1445 | 0 | 0 | PASS | 41.11% |
| inside_day | 14282 | 14281 | 0 | 0 | PASS | 58.98% |
| area_gaps | 7902 | 7902 | 0 | 0 | PASS | 27.73% |
| horn_bottoms | 2226 | 2223 | 0 | 0 | PASS | 76.34% |

## So khớp payload (multiple base + 1.0x; KHỚP TUYỆT ĐỐI — lệch 0.00pp, sai số làm tròn hiển thị tối đa 0.01pp; dotC Sol round-2 bỏ ngưỡng 1.0pp)

| Chương | Hàng multiple | n payload | n recompute | Payload % | Recompute % | Lệch | Kết quả |
|---|---|---|---|---|---|---|---|
| double_tops | (bỏ qua) | - | - | - | - | - | payload double_tops chung là bản TRƯỚC đợt B (sách dùng 8 variant riêng có payload fresh) — không so hit-rate ở đây. |
| bump_and_run_reversal_bottoms | multiple_0.5x_conservative_half_bump_height | 860 | 860 | 64.19% | 64.19% | 0.0 pp | KHỚP |
| bump_and_run_reversal_bottoms | multiple_1.0x_source_full_bump_height | 860 | 860 | 41.98% | 41.98% | 0.0 pp | KHỚP |
| inside_day | multiple_0.5x_conservative_half_inside_range | 4446 | 4446 | 78.7% | 78.7% | 0.0 pp | KHỚP |
| inside_day | multiple_1.0x_full_inside_range | 4446 | 4446 | 69.14% | 69.14% | 0.0 pp | KHỚP |
| area_gaps | multiple_0.5x_conservative_half_gap | 2709 | 2709 | 61.68% | 61.68% | 0.0 pp | KHỚP |
| area_gaps | multiple_1.0x_source_full_gap | 2709 | 2709 | 50.72% | 50.72% | 0.0 pp | KHỚP |
| horn_bottoms | multiple_0.5x_conservative_half_horn | 809 | 809 | 89.37% | 89.37% | 0.0 pp | KHỚP |
| horn_bottoms | multiple_1.0x_source_full_horn | 809 | 809 | 80.84% | 80.84% | 0.0 pp | KHỚP |
