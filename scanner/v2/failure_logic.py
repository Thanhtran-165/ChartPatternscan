"""Failure logic chuẩn Bulkowski (M2 — K3-1 hướng dẫn 12/08).

Thay `failure_5pct` (MFE < 5% — SAI chuẩn) bằng 2 thước đo:
- `weak_move_5pct`: move sau breakout yếu (MFE < 5%) — giữ ý nghĩa cũ, đổi tên rõ nghĩa.
- `failure_busted`: giá quay lại ĐÁY pattern (up) / ĐỈNH pattern (down) TRƯỚC khi chạm target —
  đúng định nghĩa "busted" của Bulkowski (03-measurement-standards §2.2):
      up   → busted khi low  <= đáy pattern × (1 − threshold/100) TRƯỚC target
      down → busted khi high >= đỉnh pattern × (1 + threshold/100) TRƯỚC target
  Ngưỡng (threshold_pct) lấy từ measurement_registry: inside_day 1%, islands/three_methods 2%,
  pipes/horn/spike 3%, gaps 2% (fill), còn lại 5%. Gaps: breakaway fill = failure.

Mức đáy/đỉnh pattern lấy từ các key THỰC của detector (khảo sát 12/08):
handle_extreme_price (cup), flag_lower/upper_breakout_value (flags/pennants), triangle_support/
resistance (triangles), rectangle_support/resistance, pattern_low/high (broadening/diamond),
extreme_price (rounding), support_resistance_price (horn/pipe), first_leg_start_price
(measured move), bounce_high_price (dead_cat), first_bar_low/high (three methods)...
Nếu family không có key khả dụng → ước lượng đáy/đỉnh = breakout ± pattern_height_pct.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import pandas as pd

from .measurement_registry import failure_threshold_pct, family_of

# Mức giá failure reference per family — theo direction breakout:
#   "up"   → các key mức ĐÁY pattern (ưu tiên theo thứ tự)
#   "down" → các key mức ĐỈNH pattern (ưu tiên theo thứ tự)
# Tên key khảo sát từ detection thực của từng detector (scripts/survey_failure_keys.py).
_FAMILY_LEVEL_KEYS: Dict[str, Dict[str, tuple[str, ...]]] = {
    "inside_day": {"up": ("inside_day_low", "mother_bar_low"), "down": ("inside_day_high", "mother_bar_high")},
    "pipe_bottoms": {"up": ("low_boundary_price", "support_resistance_price"), "down": ()},
    "pipe_tops": {"up": (), "down": ("high_boundary_price", "support_resistance_price")},
    "horn_bottoms_tops": {"up": ("support_resistance_price", "low_boundary_price"), "down": ("support_resistance_price", "high_boundary_price")},
    "flags": {"up": ("flag_lower_breakout_value", "flag_lower_price0"), "down": ("flag_upper_breakout_value", "flag_upper_price0")},
    "pennants": {"up": ("flag_lower_breakout_value", "flag_lower_price0"), "down": ("flag_upper_breakout_value", "flag_upper_price0")},
    "high_tight_flags": {"up": ("flag_lower_breakout_value", "flag_lower_price0"), "down": ("flag_upper_breakout_value", "flag_upper_price0")},
    "cup_with_handle": {"up": ("handle_extreme_price", "stop_loss_price"), "down": ()},
    "head_and_shoulders_bottom": {"up": ("neckline_price",), "down": ()},
    "head_and_shoulders_top": {"up": (), "down": ("neckline_price",)},
    "triangles": {"up": ("triangle_support", "triangle_lower_price0"), "down": ("triangle_resistance", "triangle_upper_price0")},
    "wedges_ascending_descending": {
        "up": ("wedge_support", "triangle_support", "triangle_lower_price0"),
        "down": ("wedge_resistance", "triangle_resistance", "triangle_upper_price0"),
    },
    "broadening_bottoms": {"up": ("pattern_low", "broadening_support"), "down": ()},
    "broadening_tops": {"up": (), "down": ("pattern_high", "broadening_resistance")},
    "broadening_wedges": {"up": ("pattern_low", "broadening_support"), "down": ("pattern_high", "broadening_resistance")},
    "broadening_formations_right_angled": {"up": ("pattern_low", "broadening_support"), "down": ("pattern_high", "broadening_resistance")},
    "bump_and_run_reversal": {"up": ("bump_low", "pattern_low"), "down": ("bump_high", "pattern_high")},
    "diamond_bottom": {"up": ("pattern_low",), "down": ()},
    "diamond_top": {"up": (), "down": ("pattern_high",)},
    "double_bottoms": {"up": ("first_extreme_price", "second_extreme_price"), "down": ()},
    "double_tops": {"up": (), "down": ("first_extreme_price", "second_extreme_price")},
    "rectangle_bottoms_tops": {"up": ("rectangle_support", "pattern_low"), "down": ("rectangle_resistance", "pattern_high")},
    "rounding_bottoms_tops": {"up": ("extreme_price", "pattern_low"), "down": ("extreme_price", "pattern_high")},
    "scallops_ascending": {"up": ("low_boundary_price", "middle_anchor_price"), "down": ()},
    "scallops_descending": {"up": (), "down": ("high_boundary_price",)},
    "three_falling_peaks": {"up": (), "down": ("peak_high",)},
    "three_rising_valleys": {"up": ("valley_low",), "down": ()},
    "triple_bottoms": {"up": ("pivot_1_price", "pivot_3_price", "pivot_5_price"), "down": ()},
    "triple_tops": {"up": (), "down": ("pivot_2_price", "pivot_4_price")},
    "measured_move_down_up": {"up": ("first_leg_start_price", "correction_end_price"), "down": ("first_leg_start_price",)},
    "gaps": {"up": ("gap_rim_close_price", "gap_bottom_price"), "down": ("gap_top_price", "gap_rim_close_price")},
    "islands": {"up": ("breakout_price",), "down": ("breakout_price",)},
    "dead_cat_bounce": {"up": (), "down": ("bounce_high_price", "event_low_price")},
    "rising_falling_three_methods": {"up": ("first_bar_low",), "down": ("first_bar_high",)},  # K3-1: first_bar_range
    "spike_formation": {"up": ("spike_extreme",), "down": ("spike_extreme",)},
}

_FALLBACK_KEYS: tuple[str, ...] = ("pattern_low", "pattern_high", "breakout_price")


def _pick_failure_level(row: Mapping[str, Any], family: str, up: bool) -> Optional[float]:
    """Lấy mức đáy/đỉnh pattern từ row (key đầu tiên có giá trị số dương theo direction).

    Không có key khả dụng → ước lượng breakout ± pattern_height_pct (đáy/đỉnh pattern).
    """
    keys = _FAMILY_LEVEL_KEYS.get(family, {}).get("up" if up else "down", ())
    for key in keys:
        val = row.get(key)
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if fval > 0:
            return fval
    # Fallback: đáy/đỉnh pattern ≈ breakout ± chiều cao pattern
    try:
        h = float(row.get("pattern_height_pct") or 0.0)
        bk = float(row.get("breakout_price") or 0.0)
        if h > 0 and bk > 0:
            return bk * (1.0 - h / 100.0) if up else bk * (1.0 + h / 100.0)
    except (TypeError, ValueError):
        pass
    for key in _FALLBACK_KEYS:
        val = row.get(key)
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if fval > 0:
            return fval
    return None


def _is_up(direction: Any) -> bool:
    if isinstance(direction, str):
        return direction.strip().lower() == "up"
    return bool(direction) and float(direction) >= 0


def failure_busted_flag(
    detection: Mapping[str, Any],
    future: pd.DataFrame,
    *,
    breakout_price: float,
    target_price: float,
    mfe_pct: Optional[float],
) -> bool:
    """failure_busted (low/high-based, trước target) cho 1 event — call site chung mọi detector."""
    return _evaluate(detection, future, breakout_price=breakout_price, target_price=target_price, mfe_pct=mfe_pct)["failure_busted"]


def failure_busted_days(
    detection: Mapping[str, Any],
    future: pd.DataFrame,
    *,
    breakout_price: float,
    target_price: float,
    mfe_pct: Optional[float],
) -> Optional[int]:
    """Số phiên từ breakout tới lúc busted (None nếu không busted)."""
    return _evaluate(detection, future, breakout_price=breakout_price, target_price=target_price, mfe_pct=mfe_pct)["days_to_bust"]


def _evaluate(
    detection: Mapping[str, Any],
    future: pd.DataFrame,
    *,
    breakout_price: float,
    target_price: float,
    mfe_pct: Optional[float],
) -> Dict[str, Any]:
    pk = detection.get("pattern_key") or ""
    fam = family_of(pk) if pk else ""
    up = _is_up(detection.get("breakout_direction", "up"))
    level = _pick_failure_level(detection, fam, up)
    threshold = float(failure_threshold_pct(pk) if pk else 5.0)
    if future is None or future.empty:
        return {"weak_move_5pct": None, "failure_busted": None, "days_to_bust": None}
    if level is None or level <= 0:
        level = float(breakout_price)
    if level <= 0:
        return {"weak_move_5pct": None, "failure_busted": None, "days_to_bust": None}
    weak_move = bool(float(mfe_pct) < 5.0) if mfe_pct is not None else None
    busted: Optional[bool] = None
    days_to_bust: Optional[int] = None
    edge = level * (1.0 - threshold / 100.0) if up else level * (1.0 + threshold / 100.0)
    for offset, (_, row) in enumerate(future.iterrows(), start=1):
        high = float(row["high"])
        low = float(row["low"])
        hit_target = (high >= target_price) if up else (low <= target_price)
        # 03 §2.2: busted khi giá quay lại ĐÁY (up) / ĐỈNH (down) pattern — dùng low/high
        crossed = (low <= edge) if up else (high >= edge)
        if hit_target:
            busted = False if busted is None else busted
            break
        if crossed:
            busted = True
            days_to_bust = offset
            break
    if busted is None:
        busted = False
    return {"weak_move_5pct": weak_move, "failure_busted": busted, "days_to_bust": days_to_bust}
