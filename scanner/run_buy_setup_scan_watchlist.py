"""Pre-breakout BUY setup radar for the buy-eligible pattern set.

This is intentionally separate from ``run_realtime_scan_watchlist``.  The
existing realtime watchlist reads confirmed pattern events after breakout; this
module looks for incomplete structures that are close to a confirmation level.
Rows produced here are therefore setup candidates, not confirmed signals.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.run_bear_flag_db_source_parity_audit import DEFAULT_DB, _load_symbol_from_db, _symbols_in_db  # noqa: E402
from scanner.run_realtime_scan_watchlist import DEFAULT_AFTER_BUY_CONFIG, _family_for_pattern  # noqa: E402
from scanner.v2.source_data import DEFAULT_MEMBERSHIP_DB, load_current_members, market_group  # noqa: E402
from scanner.volume_features import compute_latest_volume_features  # noqa: E402


WORKFLOW_ID = "buy_setup_scan_watchlist_v1"
DETECTOR_VERSION = "setup_proxy_v1"
DEFAULT_OUT_DIR = Path("artifacts/realtime_scan/latest/buy_setup")
VN100_MARKET_GROUPS = {"VN30", "VN100 ex VN30"}


@dataclass(frozen=True)
class BuySetupSpec:
    pattern_id: str
    local_role: str
    buy_scope: str
    family: str
    detector_family: str


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a / b - 1.0) * 100.0


def _load_buy_setup_specs(config_path: Path = DEFAULT_AFTER_BUY_CONFIG) -> list[BuySetupSpec]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    rows = data.get("patterns") if isinstance(data.get("patterns"), list) else []
    specs: list[BuySetupSpec] = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("buy_layer_allowed"):
            continue
        pattern_id = str(row.get("pattern_id"))
        specs.append(
            BuySetupSpec(
                pattern_id=pattern_id,
                local_role=str(row.get("local_role") or "buy_watchlist"),
                buy_scope=str(row.get("buy_scope") or "full_pattern_or_family_scope"),
                family=_family_label_for_pattern(pattern_id),
                detector_family=_detector_family_for_pattern(pattern_id),
            )
        )
    return sorted(specs, key=lambda spec: (spec.family, spec.pattern_id))


def _detector_family_for_pattern(pattern_id: str) -> str:
    if pattern_id in {"bull_flags", "bull_pennants", "high_tight_flags"}:
        return "flaglike_continuation"
    if pattern_id.startswith("triangles_"):
        return "triangle_near_breakout"
    if pattern_id == "rectangle_bottoms":
        return "rectangle_bottom_near_breakout"
    if pattern_id.startswith("double_bottoms_"):
        return "double_bottom_neckline"
    if pattern_id.startswith("head_and_shoulders_bottoms"):
        return "head_shoulders_bottom_neckline"
    if pattern_id == "broadening_bottoms":
        return "broadening_bottom_upper_boundary"
    if pattern_id == "measured_move_up":
        return "measured_move_up_continuation"
    return "generic_buy_near_resistance"


def _family_label_for_pattern(pattern_id: str) -> str:
    if pattern_id == "measured_move_up":
        return "measured_move_family"
    return _family_for_pattern(pattern_id)


def _load_current_vn100_symbols(
    *,
    db_path: Path = DEFAULT_DB,
    membership_db: Path = DEFAULT_MEMBERSHIP_DB,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    vn30 = load_current_members("VN30", membership_db)
    vn100 = load_current_members("VN100", membership_db)
    allowed = sorted(vn30 | vn100)
    symbols = _symbols_in_db(db_path, allowed)
    groups = {symbol: market_group(symbol, vn30, vn100) for symbol in symbols}
    meta = {
        "method": "current VN30/VN100 membership snapshot",
        "point_in_time": False,
        "membership_db": str(membership_db),
        "db_path": str(db_path),
        "vn30_members": len(vn30),
        "vn100_members": len(vn100),
        "symbols_in_db": len(symbols),
    }
    return symbols, groups, meta


def _last(df: pd.DataFrame, n: int, *, offset: int = 0) -> pd.DataFrame:
    end = len(df) - int(offset)
    start = max(0, end - int(n))
    return df.iloc[start:end]


def _distance_to_trigger(close: float, trigger: float) -> float:
    return max(0.0, _pct(trigger, close))


def _common_liquidity_fields(df: pd.DataFrame) -> dict[str, Any]:
    value_20 = pd.to_numeric(_last(df, 20)["value"], errors="coerce").median()
    zero_volume_rate_20 = float((pd.to_numeric(_last(df, 20)["volume"], errors="coerce").fillna(0) <= 0).mean() * 100.0)
    if pd.isna(value_20):
        value_20 = None
    return {
        "median_value_20": round(float(value_20), 0) if value_20 is not None else None,
        "zero_volume_rate_20": round(zero_volume_rate_20, 2),
    }


def _score_candidate(
    *,
    role: str,
    distance_to_trigger_pct: float,
    setup_strength: float,
    potential_profit_pct: float,
    zero_volume_rate_20: float,
) -> tuple[float, str]:
    score = 45.0
    score += 10.0 if role == "buy_core" else 4.0
    score += max(0.0, min(22.0, 22.0 - distance_to_trigger_pct * 4.0))
    score += max(0.0, min(18.0, setup_strength))
    score += max(0.0, min(10.0, potential_profit_pct / 2.0))
    if zero_volume_rate_20 <= 5:
        score += 5.0
    elif zero_volume_rate_20 >= 25:
        score -= 10.0
    score = round(max(0.0, min(100.0, score)), 2)
    if score >= 82:
        tier = "strong_setup"
    elif score >= 70:
        tier = "watchlist_setup"
    elif score >= 58:
        tier = "early_setup"
    else:
        tier = "weak_setup"
    return score, tier


def _finalize_candidate(
    *,
    spec: BuySetupSpec,
    symbol: str,
    group: str,
    df: pd.DataFrame,
    trigger_price: float,
    invalidation_price: float,
    setup_strength: float,
    setup_reason: str,
    trigger_basis: str,
    target_basis: str,
    target_price: float | None = None,
    setup_start_date: Any | None = None,
) -> dict[str, Any] | None:
    latest = df.iloc[-1]
    close = float(latest["close"])
    latest_high = float(latest["high"])
    if close <= 0 or trigger_price <= 0 or invalidation_price <= 0:
        return None
    if close >= trigger_price:
        return None
    distance = _distance_to_trigger(close, trigger_price)
    if distance > 8.0:
        return None
    if target_price is None:
        target_price = trigger_price + max(trigger_price - invalidation_price, trigger_price * 0.025) * 0.75
    potential_profit = _pct(float(target_price), close)
    if potential_profit <= 1.0:
        return None
    liquidity = _common_liquidity_fields(df)
    volume_features = compute_latest_volume_features(df, setup_start_date=setup_start_date)
    score, tier = _score_candidate(
        role=spec.local_role,
        distance_to_trigger_pct=distance,
        setup_strength=setup_strength,
        potential_profit_pct=potential_profit,
        zero_volume_rate_20=float(liquidity.get("zero_volume_rate_20") or 0.0),
    )
    return {
        "workflow_id": WORKFLOW_ID,
        "detector_version": DETECTOR_VERSION,
        "buy_stage": "BUY_SETUP",
        "setup_status": "pre_breakout_candidate",
        "pattern_id": spec.pattern_id,
        "family": spec.family,
        "detector_family": spec.detector_family,
        "after_buy_role": spec.local_role,
        "buy_scope": spec.buy_scope,
        "symbol": symbol,
        "market_group": group,
        "latest_date": pd.to_datetime(latest["date"]).date().isoformat(),
        "setup_start_date": pd.to_datetime(setup_start_date).date().isoformat() if setup_start_date is not None else None,
        "last_close": round(close, 4),
        "latest_high": round(latest_high, 4),
        "trigger_price": round(float(trigger_price), 4),
        "distance_to_trigger_pct": round(distance, 2),
        "invalidation_price": round(float(invalidation_price), 4),
        "target_price": round(float(target_price), 4),
        "potential_profit_pct": round(potential_profit, 2),
        "setup_quality_score": score,
        "setup_quality_tier": tier,
        "setup_strength": round(float(setup_strength), 2),
        "setup_reason": setup_reason,
        "trigger_basis": trigger_basis,
        "target_basis": target_basis,
        "is_confirmed_breakout": False,
        **liquidity,
        **volume_features,
    }


def _scan_flaglike(symbol: str, group: str, df: pd.DataFrame, spec: BuySetupSpec) -> dict[str, Any] | None:
    if len(df) < 90:
        return None
    bars = 8 if spec.pattern_id == "bull_pennants" else 12
    if spec.pattern_id == "high_tight_flags":
        bars = 15
    cons = _last(df, bars)
    prior = df.iloc[max(0, len(df) - 90) : max(1, len(df) - bars)]
    if cons.empty or prior.empty:
        return None
    prior_low = float(prior["low"].min())
    cons_high = float(cons["high"].max())
    cons_low = float(cons["low"].min())
    prior_move_pct = _pct(cons_high, prior_low)
    min_move = 25.0 if spec.pattern_id == "high_tight_flags" else 10.0
    if prior_move_pct < min_move:
        return None
    cons_range_pct = _pct(cons_high, cons_low)
    if cons_range_pct > (18.0 if spec.pattern_id == "high_tight_flags" else 12.0):
        return None
    close = float(df.iloc[-1]["close"])
    if close < cons_low or _distance_to_trigger(close, cons_high) > 6.0:
        return None
    setup_strength = min(18.0, prior_move_pct / 2.5) + max(0.0, 6.0 - cons_range_pct / 2.0)
    target = cons_high + (cons_high - prior_low) * (0.46 if spec.pattern_id != "high_tight_flags" else 0.50)
    return _finalize_candidate(
        spec=spec,
        symbol=symbol,
        group=group,
        df=df,
        trigger_price=cons_high,
        invalidation_price=cons_low,
        setup_strength=setup_strength,
        setup_reason=f"Nhịp tăng trước đủ lực ({prior_move_pct:.1f}%) và thân mẫu đang nén trong {bars} phiên.",
        trigger_basis="Đóng cửa vượt biên trên của thân cờ/pennant hiện tại.",
        target_basis="Mục tiêu tham khảo theo phần thận trọng của nhịp dẫn trước.",
        target_price=target,
        setup_start_date=cons.iloc[0]["date"],
    )


def _scan_triangle(symbol: str, group: str, df: pd.DataFrame, spec: BuySetupSpec) -> dict[str, Any] | None:
    if len(df) < 80:
        return None
    base = _last(df, 45)
    if len(base) < 35:
        return None
    first = base.iloc[: len(base) // 2]
    second = base.iloc[len(base) // 2 :]
    resistance = float(base["high"].max())
    support = float(base["low"].min())
    close = float(df.iloc[-1]["close"])
    range_pct = _pct(resistance, support)
    if range_pct > 22.0 or _distance_to_trigger(close, resistance) > 6.0:
        return None
    low_lift_pct = _pct(float(second["low"].min()), float(first["low"].min()))
    high_change_pct = _pct(float(second["high"].max()), float(first["high"].max()))
    if spec.pattern_id == "triangles_ascending":
        if low_lift_pct < 1.0 or abs(high_change_pct) > 8.0:
            return None
        reason = "Đáy sau cao dần trong khi vùng kháng cự phía trên tương đối rõ."
    elif spec.pattern_id == "triangles_symmetrical":
        if low_lift_pct < -1.0 or high_change_pct > 2.5:
            return None
        reason = "Biên dao động đang hẹp lại và giá tiến sát vùng xác nhận phá lên."
    else:
        if close < base["close"].median() or high_change_pct > 4.0:
            return None
        reason = "Nhánh chỉ xét phá lên: giá hồi lên gần vùng cản sau pha tam giác."
    setup_strength = max(0.0, 18.0 - range_pct / 1.8) + min(8.0, max(0.0, low_lift_pct))
    return _finalize_candidate(
        spec=spec,
        symbol=symbol,
        group=group,
        df=df,
        trigger_price=resistance,
        invalidation_price=support,
        setup_strength=setup_strength,
        setup_reason=reason,
        trigger_basis="Đóng cửa vượt vùng kháng cự/biên trên của tam giác.",
        target_basis="Mục tiêu tham khảo bằng phần thận trọng của chiều cao tam giác.",
        target_price=resistance + (resistance - support) * 0.75,
        setup_start_date=base.iloc[0]["date"],
    )


def _scan_rectangle_bottom(symbol: str, group: str, df: pd.DataFrame, spec: BuySetupSpec) -> dict[str, Any] | None:
    if len(df) < 90:
        return None
    base = _last(df, 50)
    prior = df.iloc[max(0, len(df) - 120) : max(1, len(df) - 50)]
    resistance = float(base["high"].max())
    support = float(base["low"].min())
    close = float(df.iloc[-1]["close"])
    range_pct = _pct(resistance, support)
    prior_decline = _pct(float(prior["high"].max()), support) if not prior.empty else 0.0
    if range_pct > 20.0 or prior_decline < 8.0 or _distance_to_trigger(close, resistance) > 6.0:
        return None
    setup_strength = max(0.0, 18.0 - range_pct / 2.0) + min(8.0, prior_decline / 3.0)
    return _finalize_candidate(
        spec=spec,
        symbol=symbol,
        group=group,
        df=df,
        trigger_price=resistance,
        invalidation_price=support,
        setup_strength=setup_strength,
        setup_reason="Giá tạo vùng đi ngang sau nhịp giảm trước đó và đang tiến gần cạnh trên.",
        trigger_basis="Đóng cửa vượt cạnh trên của vùng rectangle bottom.",
        target_basis="Mục tiêu tham khảo bằng phần thận trọng của chiều cao hộp giá.",
        target_price=resistance + (resistance - support) * 0.75,
        setup_start_date=base.iloc[0]["date"],
    )


def _find_two_lows_neckline(df: pd.DataFrame, window: int = 90) -> tuple[float, float, float, Any] | None:
    base = _last(df, window)
    if len(base) < 50:
        return None
    lows = base["low"].astype(float).rolling(5, center=True).min()
    low_points = base.loc[base["low"].astype(float) <= lows.fillna(float("inf"))]
    if len(low_points) < 2:
        return None
    candidates = low_points.tail(8).reset_index()
    best: tuple[float, float, float, Any] | None = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            left = candidates.iloc[i]
            right = candidates.iloc[j]
            gap = int(right["index"] - left["index"])
            if gap < 10:
                continue
            low1 = float(left["low"])
            low2 = float(right["low"])
            if abs(_pct(low2, low1)) > 6.0:
                continue
            between = df.iloc[int(left["index"]) : int(right["index"]) + 1]
            neckline = float(between["high"].max())
            if neckline <= max(low1, low2) * 1.04:
                continue
            best = (min(low1, low2), max(low1, low2), neckline, left["date"])
    return best


def _scan_double_bottom(symbol: str, group: str, df: pd.DataFrame, spec: BuySetupSpec) -> dict[str, Any] | None:
    found = _find_two_lows_neckline(df)
    if not found:
        return None
    low_floor, _, neckline, start_date = found
    close = float(df.iloc[-1]["close"])
    if close < low_floor * 1.04 or _distance_to_trigger(close, neckline) > 8.0:
        return None
    prior = df.iloc[max(0, len(df) - 150) : max(1, len(df) - 90)]
    prior_decline = _pct(float(prior["high"].max()), low_floor) if not prior.empty else 0.0
    if prior_decline < 10.0:
        return None
    setup_strength = min(14.0, prior_decline / 2.5) + min(8.0, _pct(neckline, low_floor) / 2.0)
    return _finalize_candidate(
        spec=spec,
        symbol=symbol,
        group=group,
        df=df,
        trigger_price=neckline,
        invalidation_price=low_floor,
        setup_strength=setup_strength,
        setup_reason="Hai vùng đáy gần nhau đã hình thành; giá đang tiến lại vùng neckline.",
        trigger_basis="Đóng cửa vượt neckline của mẫu hai đáy.",
        target_basis="Mục tiêu tham khảo bằng phần thận trọng của chiều cao từ đáy tới neckline.",
        target_price=neckline + (neckline - low_floor) * 0.75,
        setup_start_date=start_date,
    )


def _scan_hs_bottom(symbol: str, group: str, df: pd.DataFrame, spec: BuySetupSpec) -> dict[str, Any] | None:
    found = _find_two_lows_neckline(df, window=120)
    if not found:
        return None
    low_floor, _, neckline, start_date = found
    close = float(df.iloc[-1]["close"])
    if _distance_to_trigger(close, neckline) > (9.0 if spec.pattern_id.endswith("complex") else 7.0):
        return None
    if _pct(neckline, low_floor) < 8.0:
        return None
    setup_strength = min(22.0, _pct(neckline, low_floor) / 1.5)
    return _finalize_candidate(
        spec=spec,
        symbol=symbol,
        group=group,
        df=df,
        trigger_price=neckline,
        invalidation_price=low_floor,
        setup_strength=setup_strength,
        setup_reason="Cấu trúc đáy rộng đang hồi lên gần neckline; cần xác nhận bằng đóng cửa.",
        trigger_basis="Đóng cửa vượt neckline của vai đầu vai đáy.",
        target_basis="Mục tiêu tham khảo bằng phần thận trọng của chiều cao từ đáy sâu tới neckline.",
        target_price=neckline + (neckline - low_floor) * 0.75,
        setup_start_date=start_date,
    )


def _scan_broadening_bottom(symbol: str, group: str, df: pd.DataFrame, spec: BuySetupSpec) -> dict[str, Any] | None:
    if len(df) < 100:
        return None
    base = _last(df, 70)
    first = base.iloc[: len(base) // 2]
    second = base.iloc[len(base) // 2 :]
    upper = float(second["high"].max())
    lower = float(base["low"].min())
    if float(second["high"].max()) <= float(first["high"].max()) * 1.01:
        return None
    if float(second["low"].min()) >= float(first["low"].min()) * 0.99:
        return None
    close = float(df.iloc[-1]["close"])
    if _distance_to_trigger(close, upper) > 8.0:
        return None
    height_pct = _pct(upper, lower)
    if height_pct < 10.0:
        return None
    setup_strength = min(22.0, height_pct / 1.4)
    return _finalize_candidate(
        spec=spec,
        symbol=symbol,
        group=group,
        df=df,
        trigger_price=upper,
        invalidation_price=lower,
        setup_strength=setup_strength,
        setup_reason="Biên dao động mở rộng ở vùng đáy và giá đang áp sát biên trên.",
        trigger_basis="Đóng cửa vượt biên trên của broadening bottom.",
        target_basis="Mục tiêu tham khảo bằng phần thận trọng của chiều cao mẫu.",
        target_price=upper + (upper - lower) * 0.65,
        setup_start_date=base.iloc[0]["date"],
    )


def _scan_measured_move_up(symbol: str, group: str, df: pd.DataFrame, spec: BuySetupSpec) -> dict[str, Any] | None:
    if len(df) < 120:
        return None
    base = _last(df, 100)
    low_idx = int(base["low"].astype(float).idxmin())
    after_low = df.loc[low_idx:].tail(90)
    if len(after_low) < 30:
        return None
    high_idx = int(after_low["high"].astype(float).idxmax())
    if high_idx <= low_idx + 8:
        return None
    first_low = float(df.loc[low_idx, "low"])
    first_high = float(df.loc[high_idx, "high"])
    first_leg_pct = _pct(first_high, first_low)
    if first_leg_pct < 12.0:
        return None
    correction = df.loc[high_idx:].tail(45)
    if len(correction) < 10:
        return None
    correction_low = float(correction["low"].min())
    retrace = (first_high - correction_low) / max(first_high - first_low, 1e-9)
    if retrace < 0.25 or retrace > 0.72:
        return None
    trigger = float(correction["high"].max())
    close = float(df.iloc[-1]["close"])
    if close < correction_low * 1.03 or _distance_to_trigger(close, trigger) > 7.0:
        return None
    setup_strength = min(16.0, first_leg_pct / 2.0) + max(0.0, 8.0 - abs(retrace - 0.50) * 20.0)
    return _finalize_candidate(
        spec=spec,
        symbol=symbol,
        group=group,
        df=df,
        trigger_price=trigger,
        invalidation_price=correction_low,
        setup_strength=setup_strength,
        setup_reason=f"Nhịp đầu tăng {first_leg_pct:.1f}%, sau đó điều chỉnh khoảng {retrace:.0%} và đang hồi lại.",
        trigger_basis="Đóng cửa vượt đỉnh của pha điều chỉnh để xác nhận nhịp tăng thứ hai.",
        target_basis="Mục tiêu tham khảo theo phần thận trọng của nhịp tăng đầu.",
        target_price=trigger + (first_high - first_low) * 0.50,
        setup_start_date=df.loc[low_idx, "date"],
    )


DETECTORS: dict[str, Callable[[str, str, pd.DataFrame, BuySetupSpec], dict[str, Any] | None]] = {
    "flaglike_continuation": _scan_flaglike,
    "triangle_near_breakout": _scan_triangle,
    "rectangle_bottom_near_breakout": _scan_rectangle_bottom,
    "double_bottom_neckline": _scan_double_bottom,
    "head_shoulders_bottom_neckline": _scan_hs_bottom,
    "broadening_bottom_upper_boundary": _scan_broadening_bottom,
    "measured_move_up_continuation": _scan_measured_move_up,
}


def scan_buy_setups(
    *,
    db_path: Path = DEFAULT_DB,
    membership_db: Path = DEFAULT_MEMBERSHIP_DB,
    after_buy_config_path: Path = DEFAULT_AFTER_BUY_CONFIG,
    patterns: Sequence[str] | None = None,
    limit_per_pattern: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    specs = _load_buy_setup_specs(after_buy_config_path)
    wanted = set(patterns or [])
    if wanted:
        specs = [spec for spec in specs if spec.pattern_id in wanted]
    symbols, groups, membership_meta = _load_current_vn100_symbols(db_path=db_path, membership_db=membership_db)
    rows: list[dict[str, Any]] = []
    symbol_failures = 0
    conn = sqlite3.connect(str(db_path))
    try:
        for symbol in symbols:
            df = _load_symbol_from_db(conn, symbol)
            if df.empty or len(df) < 80:
                symbol_failures += 1
                continue
            group = groups.get(symbol, "Outside VN100")
            if group not in VN100_MARKET_GROUPS:
                continue
            for spec in specs:
                detector = DETECTORS.get(spec.detector_family)
                if not detector:
                    continue
                row = detector(symbol, group, df, spec)
                if row:
                    rows.append(row)
    finally:
        conn.close()
    if rows:
        out = pd.DataFrame(rows)
        out = out.sort_values(
            ["setup_quality_score", "distance_to_trigger_pct", "potential_profit_pct", "symbol"],
            ascending=[False, True, False, True],
        )
        if limit_per_pattern > 0:
            out = out.groupby("pattern_id", group_keys=False).head(int(limit_per_pattern)).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=_empty_columns())
    meta = {
        "workflow_id": WORKFLOW_ID,
        "detector_version": DETECTOR_VERSION,
        "scope": "current VN100/VN30 only",
        "patterns_scanned": [spec.pattern_id for spec in specs],
        "pattern_count": len(specs),
        "symbols_scanned": len(symbols),
        "symbol_failures": symbol_failures,
        "candidate_count": int(len(out)),
        "membership": membership_meta,
        "non_advice_boundary": "Setup candidates only; confirmation and risk controls remain required.",
    }
    return out, meta


def dedupe_buy_setups(setups: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate operational views while keeping raw rows auditable."""
    if setups.empty:
        return setups.copy()
    work = setups.copy()
    for column, default in {
        "trigger_price": 0.0,
        "detector_family": "unknown",
        "setup_quality_score": 0.0,
        "after_buy_role": "buy_watchlist",
        "distance_to_trigger_pct": 999.0,
        "potential_profit_pct": 0.0,
        "pattern_id": "",
        "symbol": "",
    }.items():
        if column not in work.columns:
            work[column] = default
    work["trigger_bucket"] = work["trigger_price"].round(2)
    work["dedupe_key"] = (
        work["symbol"].astype(str)
        + "|"
        + work["detector_family"].astype(str)
        + "|"
        + work["trigger_bucket"].astype(str)
    )
    work = work.sort_values(
        ["setup_quality_score", "after_buy_role", "distance_to_trigger_pct", "potential_profit_pct", "pattern_id"],
        ascending=[False, True, True, False, True],
    )
    out = work.drop_duplicates("dedupe_key", keep="first").drop(columns=["trigger_bucket", "dedupe_key"])
    return out.sort_values(
        ["setup_quality_score", "distance_to_trigger_pct", "potential_profit_pct", "symbol"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)


def _empty_columns() -> list[str]:
    return [
        "workflow_id",
        "detector_version",
        "buy_stage",
        "setup_status",
        "pattern_id",
        "family",
        "detector_family",
        "after_buy_role",
        "buy_scope",
        "symbol",
        "market_group",
        "latest_date",
        "setup_start_date",
        "last_close",
        "latest_high",
        "trigger_price",
        "distance_to_trigger_pct",
        "invalidation_price",
        "target_price",
        "potential_profit_pct",
        "setup_quality_score",
        "setup_quality_tier",
        "setup_strength",
        "setup_reason",
        "trigger_basis",
        "target_basis",
        "is_confirmed_breakout",
        "median_value_20",
        "zero_volume_rate_20",
        "volume_ratio_20",
        "value_ratio_20",
        "volume_trend_slope_20",
        "price_volume_phase",
        "volume_quality_label",
        "volume_warning_label",
        "pattern_volume_contraction_ratio",
        "obv_slope_20",
        "vpt_slope_20",
        "mfi_14",
        "vwma_fast_minus_slow",
        "vwma_trend_confirmed",
    ]


def write_buy_setup_outputs(setups: pd.DataFrame, meta: Mapping[str, Any], out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    deduped = dedupe_buy_setups(setups)
    csv_path = out_dir / "buy_setup_watchlist.csv"
    deduped_csv_path = out_dir / "buy_setup_watchlist_deduped.csv"
    json_path = out_dir / "buy_setup_watchlist.json"
    deduped_json_path = out_dir / "buy_setup_watchlist_deduped.json"
    meta_path = out_dir / "buy_setup_meta.json"
    md_path = out_dir / "buy_setup_watchlist.md"
    setups.to_csv(csv_path, index=False)
    deduped.to_csv(deduped_csv_path, index=False)
    json_path.write_text(setups.to_json(orient="records", force_ascii=False, indent=2) + "\n", encoding="utf-8")
    deduped_json_path.write_text(deduped.to_json(orient="records", force_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta_payload = {**dict(meta), "deduped_candidate_count": int(len(deduped))}
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "# BUY_SETUP Watchlist",
        "",
        f"Workflow: `{WORKFLOW_ID}`",
        f"Detector: `{DETECTOR_VERSION}`",
        f"Scope: `{meta.get('scope')}`",
        f"Raw candidate count: `{len(setups)}`",
        f"Operational deduped count: `{len(deduped)}`",
        "",
        "| Ngày | Mã | Mẫu | Nhóm | Còn cách xác nhận | Giá xác nhận | Mục tiêu tham khảo | Điểm setup | Khối lượng | Lý do |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in deduped.head(80).to_dict("records"):
        lines.append(
            f"| {row.get('latest_date')} | {row.get('symbol')} | {row.get('pattern_id')} | {row.get('market_group')} | "
            f"{row.get('distance_to_trigger_pct')}% | {row.get('trigger_price')} | {row.get('target_price')} | "
            f"{row.get('setup_quality_score')} | {row.get('volume_quality_label')} / {row.get('volume_warning_label')} | {row.get('setup_reason')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "csv": str(csv_path),
        "deduped_csv": str(deduped_csv_path),
        "json": str(json_path),
        "deduped_json": str(deduped_json_path),
        "meta": str(meta_path),
        "report_md": str(md_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan current VN100/VN30 for pre-breakout BUY setup candidates.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--membership-db", default=str(DEFAULT_MEMBERSHIP_DB))
    parser.add_argument("--after-buy-config", default=str(DEFAULT_AFTER_BUY_CONFIG))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--limit-per-pattern", type=int, default=8)
    args = parser.parse_args()
    setups, meta = scan_buy_setups(
        db_path=Path(args.db),
        membership_db=Path(args.membership_db),
        after_buy_config_path=Path(args.after_buy_config),
        patterns=list(args.pattern) or None,
        limit_per_pattern=int(args.limit_per_pattern),
    )
    paths = write_buy_setup_outputs(setups, meta, Path(args.out_dir))
    print(
        json.dumps(
            {
                "workflow_id": WORKFLOW_ID,
                "status": "PASS",
                "counts": {
                    "patterns": meta["pattern_count"],
                    "symbols": meta["symbols_scanned"],
                    "candidates": len(setups),
                },
                "paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
