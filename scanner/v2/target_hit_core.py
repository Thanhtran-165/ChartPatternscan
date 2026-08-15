"""target_hit_core — NGUỒN CHUẨN DUY NHẤT tính target_hit / target_first / days_to_target.

Đợt A round-2 (16/08/2026, Sol NO-GO MEDIUM-1): trước đây detector tính
target_hit so giá full precision, nhưng builder tái lập từ post_breakout_path.csv
bằng `target_dist_pct` LÀM TRÒN 2 chữ số → lệch hệ thống (CSV rounding luôn làm
khoảng cách lớn hơn → 111/4.442 event Inside Day lật hit→miss; payload 66,68%
so với raw 69,18%).

Đợt A2 (16/08/2026, Sol BLOCKER 3): core giờ là nguồn chuẩn duy nhất KỂ CẢ
DETECTOR — mọi `_evaluate_detection` trong scanner/v2 gọi `evaluate_target_hit`
thay vì vòng lặp/so sánh riêng. Release gate bắt buộc: `raw events target_hit ==
canonical core target_hit, mismatch = 0 trên toàn bộ events được xuất bản`
(script `scanner/audit_target_hit_core_parity.py`, chạy trong
`tests/test_target_hit_core_parity.py`).

PRECISION CHUẨN (tuyên bố chính thức, phương án Sol (ii)):
  - `target_price` và `breakout_price` được LÀM TRÒN 4 CHỮ SỐ THẬP PHÂN tại
    detection (giá DB VND có tối đa 4dp; round 4dp = không mất thông tin giá)
    và mọi tính toán sau đó dùng ĐÚNG giá trị đã round này. KHÔNG có đường nào
    tính trên target chưa round.
  - Đường giá so sánh (high/low trong post_breakout_path.csv) là FULL
    PRECISION — độ chính xác giá thực tế của DB.
  - `target_dist_pct` (2dp) CHỈ là cột hiển thị/phân tích khoảng cách, KHÔNG
    được dùng làm mốc so target_hit.

Module này là hàm chuẩn dùng CHUNG cho detector và mọi builder:
  - mốc target nội suy từ `breakout_price` + `target_price` (độ chính xác 4dp
    lưu trong events.csv — không phụ thuộc target_dist_pct làm tròn);
  - đường giá so sánh là high/low FULL PRECISION trong post_breakout_path.csv;
  - adverse 5% tính từ breakout_price full precision (không qua excursion rounded).

events.csv, payload JSON và PDF phải ra CÙNG kết quả khi dữ liệu không đổi.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


def effective_target_price(breakout_price: float, target_price: float, multiple: float = 1.0) -> float:
    """Mốc target hiệu dụng theo hệ số khoảng cách `multiple`.

    multiple=1.0 → đúng target gốc. Nội suy tuyến tính dọc trục khoảng cách
    (breakout → target), giữ nguyên hướng pattern cho mọi multiple > 0.
    """
    return float(breakout_price) + float(multiple) * (float(target_price) - float(breakout_price))


def evaluate_target_hit(
    highs,
    lows,
    breakout_price: float,
    target_price: float,
    direction: int,
    *,
    multiple: float = 1.0,
    adverse_pct: float = 5.0,
) -> Dict[str, Any]:
    """Tính target_hit / days_to_target / target_first từ chuỗi giá full precision.

    highs/lows: giá high/low các nến SAU breakout, đúng thứ tự thời gian
    (bar 1..N). direction: 1 = break lên (target phía trên), -1 = break xuống.
    adverse_pct: ngưỡng kéo ngược bất lợi (%) tính từ breakout_price.

    Trả dict: target_hit, days_to_target, days_to_adverse,
    target_first_before_adverse (hit xảy ra TRƯỚC adverse, cùng nến không tính),
    target_price_effective.
    """
    highs = np.asarray(list(highs), dtype=float)
    lows = np.asarray(list(lows), dtype=float)
    breakout_price = float(breakout_price)
    target_price = float(target_price)
    if breakout_price <= 0 or target_price <= 0 or highs.size == 0:
        return {
            "target_hit": False,
            "days_to_target": None,
            "days_to_adverse": None,
            "target_first_before_adverse": False,
            "target_price_effective": target_price,
        }
    target_eff = effective_target_price(breakout_price, target_price, multiple)
    if int(direction) == 1:
        hit_mask = highs >= target_eff
        adverse_mask = lows <= breakout_price * (1.0 - float(adverse_pct) / 100.0)
    else:
        hit_mask = lows <= target_eff
        adverse_mask = highs >= breakout_price * (1.0 + float(adverse_pct) / 100.0)
    hit_idx = int(np.argmax(hit_mask)) if bool(hit_mask.any()) else None
    adv_idx = int(np.argmax(adverse_mask)) if bool(adverse_mask.any()) else None
    days_hit = hit_idx + 1 if hit_idx is not None else None
    days_adv = adv_idx + 1 if adv_idx is not None else None
    first = False if days_hit is None else (True if days_adv is None else days_hit < days_adv)
    return {
        "target_hit": bool(days_hit is not None),
        "days_to_target": days_hit,
        "days_to_adverse": days_adv,
        "target_first_before_adverse": bool(first),
        "target_price_effective": target_eff,
    }


def _direction_of(event: Mapping[str, Any]) -> int:
    raw = str(event.get("breakout_direction") or "").strip().lower()
    if raw in ("up", "1", "bull", "bottom"):
        return 1
    if raw in ("down", "-1", "bear", "top"):
        return -1
    try:  # fallback suy hướng từ target so breakout
        return 1 if float(event["target_price"]) >= float(event["breakout_price"]) else -1
    except (KeyError, TypeError, ValueError):
        return 1


def target_hit_stats(
    events: pd.DataFrame,
    path_df: pd.DataFrame,
    multiple: float = 1.0,
    *,
    adverse_pct: float = 5.0,
) -> Tuple[List[bool], List[bool], List[Optional[float]]]:
    """Tính (hits, firsts, days) cho TỪNG event theo thứ tự `events`.

    events cần: event_id (hoặc detection_id), breakout_price, target_price,
    breakout_direction. path_df cần: event_id, bar_after_breakout, high, low
    (giá full precision). Event thiếu path hoặc thiếu giá → (False, False, NaN)
    — giữ hành vi các builder trước đây.

    Đợt B (15/08/2026, gate parity Sol): nếu events có cột `evaluated_bars`
    (số bars detector thực sự đánh giá — ghi tại detection), path được CẮT
    theo đúng ngưỡng đó cho từng event trước khi so target. Trước đây hàm dùng
    TOÀN path: family có path dài hơn cửa sổ detector (gap path 136 bars,
    area_gaps đánh giá 3 bars) → payload lệch events.csv — đúng loại lệch mà
    gate parity cấm.
    """
    grouped: Dict[str, pd.DataFrame] = {}
    if "event_id" in getattr(path_df, "columns", []):
        for event_id, group in path_df.groupby("event_id"):
            grouped[str(event_id)] = group.sort_values("bar_after_breakout")
    has_evaluated_bars = "evaluated_bars" in getattr(events, "columns", [])
    hits: List[bool] = []
    firsts: List[bool] = []
    days: List[Optional[float]] = []
    has_event_id = "event_id" in events.columns or "detection_id" in events.columns
    for _, event in events.iterrows():
        eid = str(event.get("event_id") if "event_id" in events.columns else event.get("detection_id")) if has_event_id else ""
        group = grouped.get(eid)
        bp = event.get("breakout_price")
        tp = event.get("target_price")
        if group is None or group.empty or pd.isna(bp) or pd.isna(tp):
            hits.append(False)
            firsts.append(False)
            days.append(float("nan"))
            continue
        if has_evaluated_bars:
            eb = event.get("evaluated_bars")
            if eb is not None and not pd.isna(eb):
                try:
                    eb_int = int(eb)
                    bars = pd.to_numeric(group["bar_after_breakout"], errors="coerce")
                    clipped = group[bars <= eb_int]
                    if not clipped.empty:
                        group = clipped
                except (TypeError, ValueError):
                    pass
        res = evaluate_target_hit(
            group["high"].to_numpy(),
            group["low"].to_numpy(),
            float(bp),
            float(tp),
            _direction_of(event),
            multiple=multiple,
            adverse_pct=adverse_pct,
        )
        hits.append(bool(res["target_hit"]))
        firsts.append(bool(res["target_first_before_adverse"]))
        days.append(float(res["days_to_target"]) if res["days_to_target"] is not None else float("nan"))
    return hits, firsts, days


def enrich_events_with_target_hit(
    events: pd.DataFrame,
    path_df: pd.DataFrame,
    multiple: float = 1.0,
    *,
    adverse_pct: float = 5.0,
) -> pd.DataFrame:
    """Đè 3 cột target_hit / target_first_before_adverse_5pct / days_to_target
    bằng hàm chuẩn full precision — thay thế trực tiếp thân
    `_enrich_events_for_target` ở các builder (cùng chữ ký và cùng tên cột).
    """
    events = events.copy()
    if "event_id" not in events.columns:
        events["event_id"] = events["detection_id"]
    hits, firsts, days = target_hit_stats(events, path_df, multiple, adverse_pct=adverse_pct)
    events["target_hit"] = hits
    events["target_first_before_adverse_5pct"] = firsts
    events["days_to_target"] = days
    return events
