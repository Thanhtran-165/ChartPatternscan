"""Thước đo SỐ (deterministic) cho độ chính xác hình học của bộ quét — Nấc 1 "khám mắt".

Với mỗi family, kiểm tra các tiêu chí hình học ĐỊNH NGHĨA mẫu hình (theo sách Bulkowski)
trên TOÀN TẬP events — không lấy mẫu, không dùng vision. Đầu ra: bảng % đạt từng tiêu chí
+ % đạt TẤT CẢ = "precision hình học" của detector.

v2 (13/08/2026): nối đủ 38 family khớp bảng tiêu chí của subagent Flash
(artifacts/eye_exam/geometry_criteria.md — 34 mẫu hình; harami chưa lên pipeline nên chưa đo).
Quy ước ngưỡng:
- KHÔNG dấu (*) = con số có trong sách (ECP/EC) hoặc đúng định nghĩa cứng.
- Dấu (*) = sách không ghi số → tự chọn: ưu tiên lấy đúng ngưỡng cấu hình detector hiện tại
  (đã xác minh bằng code) hoặc chọn thận trọng gần mức "rất tốt" detector ghi trong code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EVENTS_ROOT = Path("artifacts/scanner_v2_v3")
OUT_ROOT = Path("artifacts/eye_exam/precision")


def _c(col: str) -> str:
    return col


# Mỗi tiêu chí: (tên, biểu thức, kind) — kind: "shape" = định nghĩa bắt buộc của mẫu hình;
# "quality" = điểm chất lượng (không bắt buộc để là mẫu hình đúng, dùng xếp tier).
CRITERIA: dict[str, list[tuple[str, str, str]]] = {
    # --- Nến đơn/đôi (inside day) ---
    "inside_day": [
        ("con nằm trọn trong mẹ", "(inside_day_high <= mother_bar_high) & (inside_day_low >= mother_bar_low)", "shape"),
        ("mẹ không phải doji", "mother_body_pct >= 0.1", "shape"),
        ("volume co lại", "volume_contracts == True", "quality"),
    ],
    # --- Tam giác ---
    "triangles_ascending": [
        ("đỉnh gần phẳng (|upper|<=5°)", "upper_slope_deg.abs() <= 5", "shape"),
        ("đáy dốc lên (lower>=10°)", "lower_slope_deg >= 10", "shape"),
        ("đủ chạm: >=2 đỉnh & >=2 đáy", "(upper_touch_count >= 2) & (lower_touch_count >= 2)", "shape"),
        ("volume khô dần trong pattern", "volume_trend_direction == 'down'", "quality"),
        ("volume breakout xác nhận", "volume_confirmed == True", "quality"),
    ],
    "triangles_descending": [
        ("đáy gần phẳng (|lower|<=5°)", "lower_slope_deg.abs() <= 5", "shape"),
        ("đỉnh dốc xuống (upper<=-10°)", "upper_slope_deg <= -10", "shape"),
        ("đủ chạm: >=2 đỉnh & >=2 đáy", "(upper_touch_count >= 2) & (lower_touch_count >= 2)", "shape"),
        ("volume khô dần trong pattern", "volume_trend_direction == 'down'", "quality"),
        ("volume breakout xác nhận", "volume_confirmed == True", "quality"),
    ],
    "triangles_symmetrical": [
        ("đỉnh dốc xuống (upper<=-10°)", "upper_slope_deg <= -10", "shape"),
        ("đáy dốc lên (lower>=10°)", "lower_slope_deg >= 10", "shape"),
        ("đủ chạm: >=2 đỉnh & >=2 đáy", "(upper_touch_count >= 2) & (lower_touch_count >= 2)", "shape"),
        ("volume khô dần trong pattern", "volume_trend_direction == 'down'", "quality"),
        ("volume breakout xác nhận", "volume_confirmed == True", "quality"),
    ],
    # --- Cờ & cờ đuôi nheo ---
    "bull_flags": [
        ("cột cờ đủ mạnh (|pole_move_pct|>=10%)", "pole_move_pct.abs() >= 10", "shape"),
        ("hướng cột cờ khớp hướng breakout", "((breakout_direction == 'up') & (pole_price < (flag_upper_price0 + flag_lower_price0) / 2)) | ((breakout_direction == 'down') & (pole_price > (flag_upper_price0 + flag_lower_price0) / 2))", "shape"),
        ("cờ hẹp (height<=20%) (*)", "pattern_height_pct <= 20", "shape"),
        ("cờ gọn (|slope_gap|<=10°) (*)", "slope_gap_deg.abs() <= 10", "shape"),
        ("cờ nhỏ so với cột cờ (<=35%)", "flag_to_pole_pct <= 35", "quality"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    "pennants": [
        ("2 đường hội tụ ngược chiều (upper<=-2°, lower>=2°) (*)", "(upper_slope_deg <= -2) & (lower_slope_deg >= 2)", "shape"),
        ("có cột cờ trước (|pole_move_pct|>=10%) (*)", "pole_move_pct.abs() >= 10", "shape"),
        ("độ dài 9-11 ngày theo sách (<=11) (*)", "pattern_width_bars <= 11", "shape"),
        ("pennant nhỏ so với cột cờ (<=35%) (*)", "pennant_to_pole_pct <= 35", "quality"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    "high_tight_flags": [
        ("cột cờ rất mạnh (>=40%) (*)", "pole_move_pct >= 40", "shape"),
        ("consolidation 4-5 tuần theo sách (18-35 nến) (*)", "(pattern_width_bars >= 18) & (pattern_width_bars <= 35)", "shape"),
        ("nén hẹp (compression<=0.35) (*)", "compression_ratio <= 0.35", "shape"),
        ("volume khô trong nén", "volume_contracts == True", "quality"),
    ],
    # --- Đỉnh/đáy đôi ---
    "double_bottoms": [
        ("2 đáy xấp xỉ (chênh<=5%) (*)", "extreme_spread_pct <= 5", "shape"),
        ("đỉnh trung gian cao hơn 2 đáy", "(middle_extreme_price > first_extreme_price) & (middle_extreme_price > second_extreme_price)", "shape"),
        ("đủ khoảng cách 2 bên (>=3 nến) (*)", "(left_spacing_bars >= 3) & (right_spacing_bars >= 3)", "shape"),
        ("2 bên cân đối (balance>=0.3) (*)", "balance_ratio >= 0.3", "quality"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    "double_tops": [
        ("2 đỉnh xấp xỉ (chênh<=5%) (*)", "extreme_spread_pct <= 5", "shape"),
        ("đáy trung gian thấp hơn 2 đỉnh", "(middle_extreme_price < first_extreme_price) & (middle_extreme_price < second_extreme_price)", "shape"),
        ("đủ khoảng cách 2 bên (>=3 nến) (*)", "(left_spacing_bars >= 3) & (right_spacing_bars >= 3)", "shape"),
        ("2 bên cân đối (balance>=0.3) (*)", "balance_ratio >= 0.3", "quality"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    # --- Hình chữ nhật ---
    "rectangle_bottoms": [
        ("đỉnh phẳng (spread<=3%) (*)", "high_spread_pct <= 3", "shape"),
        ("đáy phẳng (spread<=3%) (*)", "low_spread_pct <= 3", "shape"),
        ("đủ chạm: >=2 đỉnh & >=2 đáy (*)", "(upper_touch_count >= 2) & (lower_touch_count >= 2)", "shape"),
        ("giá nằm trong kênh (>=80%) (*)", "rectangle_containment_pct >= 80", "quality"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    "rectangle_tops": [
        ("đỉnh phẳng (spread<=3%) (*)", "high_spread_pct <= 3", "shape"),
        ("đáy phẳng (spread<=3%) (*)", "low_spread_pct <= 3", "shape"),
        ("đủ chạm: >=2 đỉnh & >=2 đáy (*)", "(upper_touch_count >= 2) & (lower_touch_count >= 2)", "shape"),
        ("giá nằm trong kênh (>=80%) (*)", "rectangle_containment_pct >= 80", "quality"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    # --- Kim cương ---
    "diamond_bottoms": [
        ("nửa trái mở rộng (expansion>=1.10)", "expansion_ratio >= 1.10", "shape"),
        ("nửa phải thu hẹp (contraction<=0.88)", "contraction_ratio <= 0.88", "shape"),
        ("đỉnh trái cao dần (>=2%)", "left_high_rise_pct >= 2", "shape"),
        ("đáy trái thấp dần (>=2%)", "left_low_drop_pct >= 2", "shape"),
    ],
    "diamond_tops": [
        ("nửa trái mở rộng (expansion>=1.10)", "expansion_ratio >= 1.10", "shape"),
        ("nửa phải thu hẹp (contraction<=0.88)", "contraction_ratio <= 0.88", "shape"),
        ("đỉnh trái cao dần (>=2%)", "left_high_rise_pct >= 2", "shape"),
        ("đáy trái thấp dần (>=2%)", "left_low_drop_pct >= 2", "shape"),
    ],
    # --- Mở rộng (broadening) ---
    "broadening_bottoms": [
        ("đỉnh cao dần (>=2%) (*)", "high_rise_pct >= 2", "shape"),
        ("đáy thấp dần (>=2%) (*)", "low_fall_pct >= 2", "shape"),
        ("biên độ mở rộng (expansion>=1.10) (*)", "expansion_ratio >= 1.10", "shape"),
        ("đủ chạm: >=2 đỉnh & >=2 đáy (*)", "(touch_count_high >= 2) & (touch_count_low >= 2)", "shape"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    "broadening_tops": [
        ("đỉnh cao dần (>=2%) (*)", "high_rise_pct >= 2", "shape"),
        ("đáy thấp dần (>=2%) (*)", "low_fall_pct >= 2", "shape"),
        ("biên độ mở rộng (expansion>=1.10) (*)", "expansion_ratio >= 1.10", "shape"),
        ("đủ chạm: >=2 đỉnh & >=2 đáy (*)", "(touch_count_high >= 2) & (touch_count_low >= 2)", "shape"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    # --- Bump-and-run reversal ---
    "bump_and_run_reversal_bottoms": [
        ("lead-in đi ngang (|change|<=10%) (*)", "lead_in_change_pct.abs() <= 10", "shape"),
        ("bump >= 2× lead-in (sách)", "bump_slope_ratio >= 2", "shape"),
        ("bump đủ cao (>=9%)", "bump_height_pct >= 9", "shape"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    "bump_and_run_reversal_tops": [
        ("lead-in đi ngang (|change|<=10%) (*)", "lead_in_change_pct.abs() <= 10", "shape"),
        ("bump >= 2× lead-in (sách)", "bump_slope_ratio >= 2", "shape"),
        ("bump đủ cao (>=9%)", "bump_height_pct >= 9", "shape"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    # --- Measured moves ---
    "measured_move_up": [
        ("retrace 40-60% của first leg (sách) (*)", "(corrective_retrace_pct >= 40) & (corrective_retrace_pct <= 60)", "shape"),
        ("first leg tuyến tính (r2>=0.6) (*)", "first_leg_linearity_r2 >= 0.6", "shape"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    "measured_move_down": [
        ("retrace 40-60% của first leg (sách) (*)", "(corrective_retrace_pct >= 40) & (corrective_retrace_pct <= 60)", "shape"),
        ("first leg tuyến tính (r2>=0.6) (*)", "first_leg_linearity_r2 >= 0.6", "shape"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    # --- Ba phương pháp (three methods) ---
    "three_methods_rising": [
        ("nến 1 thân dài (>=1 ATR) (*)", "first_body_atr >= 1", "shape"),
        ("nến 1 trắng", "first_bar_close > first_bar_open", "shape"),
        ("3 nến giữa nằm trong range nến 1", "middle_inside_count == 3", "shape"),
        ("nến 5 đóng cao hơn nến 1", "last_bar_close > first_bar_close", "shape"),
        ("volume theo mẫu kinh điển", "volume_contracts == True", "quality"),
    ],
    "three_methods_falling": [
        ("nến 1 thân dài (>=1 ATR) (*)", "first_body_atr >= 1", "shape"),
        ("nến 1 đen", "first_bar_close < first_bar_open", "shape"),
        ("3 nến giữa nằm trong range nến 1", "middle_inside_count == 3", "shape"),
        ("nến 5 đóng thấp hơn nến 1", "last_bar_close < first_bar_close", "shape"),
        ("volume theo mẫu kinh điển", "volume_contracts == True", "quality"),
    ],
    # --- Ba đỉnh giảm / ba đáy tăng ---
    "three_falling_peaks": [
        ("3 đỉnh thấp dần", "(pivot_3_price < pivot_1_price) & (pivot_5_price < pivot_3_price)", "shape"),
        ("đáy trung gian hỗ trợ (<=+2%) (*)", "pivot_4_price <= pivot_2_price * 1.02", "shape"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    "three_rising_valleys": [
        ("3 đáy cao dần", "(pivot_3_price > pivot_1_price) & (pivot_5_price > pivot_3_price)", "shape"),
        ("đỉnh trung gian kháng cự (>=-2%) (*)", "pivot_4_price >= pivot_2_price * 0.98", "shape"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    # --- Cốc tay cầm ---
    "cup_with_handle": [
        ("2 môi xấp xỉ (chênh<=5%) (*)", "rim_diff_pct <= 5", "shape"),
        ("đáy giữa cup (vị trí 30-70%) (*)", "(bottom_pos_pct >= 30) & (bottom_pos_pct <= 70)", "shape"),
        ("đáy tròn (>=2 nến gần đáy) (*)", "near_bottom_bars >= 2", "shape"),
        ("handle nằm nửa trên cup (>=50%)", "handle_pos_pct >= 50", "shape"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    "cup_with_handle_inverted": [
        ("2 môi xấp xỉ (chênh<=5%) (*)", "rim_diff_pct <= 5", "shape"),
        ("đỉnh giữa cup (vị trí 30-70%) (*)", "(bottom_pos_pct >= 30) & (bottom_pos_pct <= 70)", "shape"),
        ("đỉnh tròn (>=2 nến gần đỉnh) (*)", "near_bottom_bars >= 2", "shape"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    # --- Dead cat bounce ---
    "dead_cat_bounce": [
        ("sụt giảm mạnh (>=20%) (*)", "event_decline_pct >= 20", "shape"),
        ("bounce ngắn 1-7 ngày (sách)", "(bounce_bars >= 1) & (bounce_bars <= 7)", "shape"),
        ("bounce không vượt quá đợt giảm (*)", "bounce_pct <= event_decline_pct", "shape"),
    ],
    "dead_cat_bounce_inverted": [
        ("tăng mạnh ngày 1 (>=5%) (*)", "event_rise_pct >= 5", "shape"),
        ("ngày 2 đẩy tiếp (day2_push)", "day2_push == True", "shape"),
    ],
    # --- Gap ---
    "gaps": [
        ("gap đáng kể (>=0.5%) (*)", "gap_size_pct >= 0.5", "shape"),
    ],
    # --- Sò (scallops) ---
    "scallops_ascending": [
        ("2 môi xấp xỉ (|shift|<=5%) (*)", "lip_shift_pct.abs() <= 5", "shape"),
        ("môi phải cao hơn môi trái (ascending)", "end_anchor_price > start_anchor_price", "shape"),
        ("đáy lõm đủ sâu (excursion>=35%)", "arc_excursion_pct >= 35", "shape"),
        ("đường đi mượt (<=8 lần đảo chiều) (*)", "smooth_turn_count <= 8", "shape"),
    ],
    "scallops_descending": [
        ("2 môi xấp xỉ (|shift|<=5%) (*)", "lip_shift_pct.abs() <= 5", "shape"),
        ("môi phải thấp hơn môi trái (descending)", "end_anchor_price < start_anchor_price", "shape"),
        ("đáy lõm đủ sâu (excursion>=35%)", "arc_excursion_pct >= 35", "shape"),
        ("đường đi mượt (<=8 lần đảo chiều) (*)", "smooth_turn_count <= 8", "shape"),
    ],
    # --- Tròn (rounding) ---
    "rounding_bottoms": [
        ("2 môi xấp xỉ (chênh<=5%) (*)", "lip_mismatch_pct <= 5", "shape"),
        ("cực trị giữa hình (vị trí 0.30-0.72)", "(center_position >= 0.30) & (center_position <= 0.72)", "shape"),
        ("vùng đáy đủ dài (>=10% nến)", "bottom_zone_fraction >= 0.10", "shape"),
        ("đường tròn (corr>=0.18)", "roundness_corr >= 0.18", "shape"),
    ],
    "rounding_tops": [
        ("2 môi xấp xỉ (chênh<=5%) (*)", "lip_mismatch_pct <= 5", "shape"),
        ("cực trị giữa hình (vị trí 0.30-0.72)", "(center_position >= 0.30) & (center_position <= 0.72)", "shape"),
        ("vùng đỉnh đủ dài (>=10% nến)", "bottom_zone_fraction >= 0.10", "shape"),
        ("đường tròn (corr>=0.18)", "roundness_corr >= 0.18", "shape"),
    ],
    # --- Ống (pipe) ---
    "pipe_bottoms": [
        ("2 đáy xấp xỉ (chênh<=5%) (*)", "spike_similarity_pct <= 5", "shape"),
        ("2 nến chồng giá nhau (overlap>=55%)", "spike_overlap_pct >= 55", "shape"),
        ("biên độ nến đủ (>=2% mỗi nến) (*)", "(left_spike_pct >= 2) & (right_spike_pct >= 2)", "shape"),
        ("volume 2 nến spike cao", "(left_volume_ratio_20 > 1) & (right_volume_ratio_20 > 1)", "quality"),
    ],
    "pipe_tops": [
        ("2 đỉnh xấp xỉ (chênh<=5%) (*)", "spike_similarity_pct <= 5", "shape"),
        ("2 nến chồng giá nhau (overlap>=55%)", "spike_overlap_pct >= 55", "shape"),
        ("biên độ nến đủ (>=2% mỗi nến) (*)", "(left_spike_pct >= 2) & (right_spike_pct >= 2)", "shape"),
        ("volume 2 nến spike cao", "(left_volume_ratio_20 > 1) & (right_volume_ratio_20 > 1)", "quality"),
    ],
    # --- Sừng (horn) ---
    "horn_bottoms": [
        ("2 spike xấp xỉ (chênh<=5%) (*)", "spike_similarity_pct <= 5", "shape"),
        ("center cách xa spike (>=2.4%)", "center_clearance_pct >= 2.4", "shape"),
        ("spike nổi bật (percentile>=70) (*)", "(left_spike_visibility_percentile >= 70) | (right_spike_visibility_percentile >= 70)", "shape"),
        ("volume spike cao", "(left_volume_ratio_20 > 1) | (right_volume_ratio_20 > 1)", "quality"),
    ],
    "horn_tops": [
        ("2 spike xấp xỉ (chênh<=5%) (*)", "spike_similarity_pct <= 5", "shape"),
        ("center cách xa spike (>=2.4%)", "center_clearance_pct >= 2.4", "shape"),
        ("spike nổi bật (percentile>=70) (*)", "(left_spike_visibility_percentile >= 70) | (right_spike_visibility_percentile >= 70)", "shape"),
        ("volume spike cao", "(left_volume_ratio_20 > 1) | (right_volume_ratio_20 > 1)", "quality"),
    ],
    # --- Nêm (wedges) ---
    "wedges_falling": [
        ("cả 2 đường dốc xuống", "(upper_slope_deg < 0) & (lower_slope_deg < 0)", "shape"),
        ("hội tụ (compression<1) (*)", "compression_ratio < 1", "shape"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
    "wedges_rising": [
        ("cả 2 đường dốc lên", "(upper_slope_deg > 0) & (lower_slope_deg > 0)", "shape"),
        ("hội tụ (compression<1) (*)", "compression_ratio < 1", "shape"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
}


def build_report(families: list[str]) -> dict:
    report = {"generated_by": "build_precision_report_v2", "families": {}}
    for fam in families:
        csv = EVENTS_ROOT / fam / "db_active" / "events.csv"
        if not csv.exists():
            report["families"][fam] = {"error": "no events"}
            continue
        df = pd.read_csv(csv)
        n = len(df)
        crits = []
        shape_pass = pd.Series(True, index=df.index)
        for name, expr, kind in CRITERIA[fam]:
            mask = df.eval(expr).fillna(False)
            crits.append({"criteria": name, "kind": kind, "pass": int(mask.sum()), "pct": round(100.0 * mask.mean(), 1)})
            if kind == "shape":
                shape_pass &= mask
        report["families"][fam] = {
            "n": n,
            "precision_shape_pct": round(100.0 * shape_pass.mean(), 1),
            "criteria": crits,
        }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append", help="chỉ chạy family này (lặp lại được)")
    args = ap.parse_args()
    families = args.family or list(CRITERIA.keys())
    report = build_report(families)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "precision_report_latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'family':26s} {'n':>7s} {'hình học':>9s}")
    for fam, r in report["families"].items():
        if "error" in r:
            print(f"{fam:26s} {'-':>7s} {r['error']}")
            continue
        print(f"{fam:26s} {r['n']:7d} {r['precision_shape_pct']:8.1f}%")
        for c in r["criteria"]:
            tag = "hình" if c["kind"] == "shape" else "chất"
            print(f"    [{tag}] {c['criteria']:52s} {c['pass']:7d} ({c['pct']:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
