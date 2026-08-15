"""Harami scanner (M5-2c, 13/08/2026).

Harami là mẫu hình 2 nến, 2 CÁCH ĐO khác nhau tùy biến thể (số sách đã kiểm
13/08 — docs/project/pdf_review/m5/family_harami_20260813.md):

- Harami THƯỜNG (ch.43 Bearish / ch.44 Bullish): đo bằng THÂN NẾN (body-based)
  — thân nến con (open→close) nằm TRONG thân nến mẹ, bỏ qua bóng nến.
- Harami CROSS (ch.45/46): nến con là DOJI (thân gần 0) nên đo bằng BIÊN ĐỘ
  (range-based) — high/low con nằm trong high/low mẹ.

Khác inside_days.py (luôn range-based strict, không cho bằng):
- Harami cho phép đỉnh HOẶC đáy bằng nhau (không cả hai cùng bằng).
- Measure rule EC: target = breakout ± (HH − LL của cả 2 nến × multiplier
  58-74% theo chương) — KHÔNG phải nguyên chiều cao pattern.

Lưu ý sách: cả 4 chương có reversal rate 47-57% (gần ngẫu nhiên) — pattern
yếu về dự báo chiều; EC KHÔNG publish failure rate cho candlestick.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scanner.v2.measurement_registry import lookahead_bars as _registry_lookahead

from scanner.ohlcv_normalizer import OHLCVNormalizer  # noqa: E402
from scanner.research_support_analysis import PatternArtifacts, build_target_calibration_decisions, target_sensitivity  # noqa: E402
from scanner.run_bear_flag_db_source_parity_audit import (  # noqa: E402
    DEFAULT_DB,
    _db_meta,
    _enrich_events_from_series,
    _load_symbol_from_db,
    _path_rows_from_series,
    _symbols_in_db,
)
from scanner.v2.bull_flags_monograph import DEFAULT_MARKET_STATS_JSON, _load_active_symbols  # noqa: E402
from scanner.v2.flags_experiment import DEFAULT_INDEX_DB, DEFAULT_INDEX_SYMBOL, _write_csv, _write_json  # noqa: E402
from scanner.v2.pipes import (  # noqa: E402
    _evaluate_detection,
    _mean,
    _median,
    _prior_trend_pct,
    _quantiles,
    _rate,
    _rolling_volume_ratio,
    _safe_float,
    _score_band,
    _truthy,
)
from scanner.v2.source_data import attach_current_market_groups, classify_market_regimes  # noqa: E402


HARAMI = "harami"
DEFAULT_OUT_DIR = Path("artifacts/scanner_v2/harami_family")

# Measure rule EC ch.43-46 (family_harami_20260813.md, bảng "% meeting price target"):
# target = breakout ± ((HH − LL) của cả 2 nến × multiplier). Dùng cột BULL MARKET
# (mẫu lớn nhất mỗi chương) cho từng hướng breakout; neutral không có chương riêng
# trong sách → trung bình 4 chương (65%).
_MULTIPLIER_PCT: Dict[str, Tuple[float, float]] = {
    # variant: (multiplier_up, multiplier_down)
    "bearish_harami": (0.63, 0.64),        # EC ch.43: 63/58/64/64 → bull: 63 up / 64 down
    "bullish_harami": (0.69, 0.59),        # EC ch.44: 69/66/59/61 → bull: 69 up / 59 down
    "harami_cross_bearish": (0.69, 0.68),  # EC ch.45: 69/67/68/66 → bull: 69 up / 68 down
    "harami_cross_bullish": (0.74, 0.68),  # EC ch.46: 74/73/68/70 → bull: 74 up / 68 down
    "neutral_harami": (0.65, 0.65),        # không có chương riêng — trung bình 4 chương
}


@dataclass(frozen=True)
class HaramiConfig:
    confirmation_search_bars: int = 2
    breakout_threshold: float = 0.0
    cross_body_pct_max: float = 0.10  # thân con <0.1% giá → harami cross (doji)
    mother_body_pct_min: float = 0.1  # M4-scanner (13/08): loại nến mẹ doji thuần
    prior_trend_lookback_bars: int = 10
    prior_trend_min_abs_pct: float = 2.0
    max_events_per_symbol: int = 12
    breakout_cooldown_bars: int = 18

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]] = None) -> "HaramiConfig":
        if value is None:
            return cls()
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in allowed})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _body_bounds(open_: float, close: float) -> Tuple[float, float]:
    """Thân nến = khoảng open→close. Trả (body_top, body_bottom)."""
    return (max(open_, close), min(open_, close))


def _variant_for(mother: pd.Series, child: pd.Series, config: HaramiConfig) -> str:
    """Phân loại theo sách EC: bearish (mẹ tăng), bullish (mẹ giảm), cross (con doji)."""
    child_open = _safe_float(child.get("open"))
    child_close = _safe_float(child.get("close"))
    child_body_pct = abs(float(child_close) - float(child_open)) / max(float(child_close), 1e-9) * 100.0
    mother_up = float(mother["close"]) > float(mother["open"])
    child_up = float(child["close"]) > float(child["open"])
    if child_body_pct <= config.cross_body_pct_max:
        # Harami Cross: ch.45 (mẹ tăng) vs ch.46 (mẹ giảm) — measure rule khác nhau
        return "harami_cross_bearish" if mother_up else "harami_cross_bullish"
    if mother_up != child_up:
        return "bearish_harami" if mother_up else "bullish_harami"
    # Cùng chiều — sách chỉ công nhận harami đảo chiều màu nến; gọi là neutral để hậu kiểm
    return "neutral_harami"


class HaramiDetector:
    def __init__(self, config: Optional[HaramiConfig | Mapping[str, Any]] = None) -> None:
        self.config = config if isinstance(config, HaramiConfig) else HaramiConfig.from_mapping(config)

    def _breakout_candidate(self, df: pd.DataFrame, idx: int, pattern_high: float, pattern_low: float) -> Optional[tuple[int, str, float]]:
        # EC: "price to close either above the top of the harami or below the bottom"
        # — top/bottom = high/low của CẢ pattern (2 nến), không chỉ nến con.
        upper = pattern_high * (1.0 + self.config.breakout_threshold)
        lower = pattern_low * (1.0 - self.config.breakout_threshold)
        for j in range(idx + 1, min(len(df), idx + 1 + self.config.confirmation_search_bars)):
            close = _safe_float(df.iloc[j].get("close"))
            if close is None:
                continue
            if close > upper:
                return j, "up", float(close)
            if close < lower:
                return j, "down", float(close)
        return None

    def scan(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        used: list[int] = []
        for idx in range(1, len(df) - 1):
            mother = df.iloc[idx - 1]
            child = df.iloc[idx]
            m_open = _safe_float(mother.get("open"))
            m_close = _safe_float(mother.get("close"))
            c_open = _safe_float(child.get("open"))
            c_close = _safe_float(child.get("close"))
            if None in (m_open, m_close, c_open, c_close) or float(m_close) == 0:
                continue
            m_body_top, m_body_bottom = _body_bounds(float(m_open), float(m_close))
            c_body_top, c_body_bottom = _body_bounds(float(c_open), float(c_close))
            # M4-scanner (13/08): loại nến mẹ doji thuần (thân gần 0 tạo "thân bao" ảo)
            mother_body_pct = abs(float(m_close) - float(m_open)) / float(m_close) * 100.0
            if mother_body_pct < self.config.mother_body_pct_min:
                continue
            child_body_pct = abs(float(c_close) - float(c_open)) / max(float(c_close), 1e-9) * 100.0
            child_is_doji = child_body_pct <= self.config.cross_body_pct_max
            # Containment theo sách EC (family_harami_20260813.md):
            # - Harami thường: BODY nằm trong BODY (bỏ bóng nến), cho phép đỉnh HOẶC đáy
            #   bằng nhau nhưng KHÔNG cả hai (nếu cả hai bằng → thân con = thân mẹ).
            # - Harami Cross (doji): RANGE nằm trong RANGE mẹ (đo high-low như inside_day).
            if child_is_doji:
                c_high = float(child["high"])
                c_low = float(child["low"])
                m_high = float(mother["high"])
                m_low = float(mother["low"])
                containment_mode = "range"
                if not (c_high <= m_high and c_low >= m_low):
                    continue
                if c_high == m_high and c_low == m_low:
                    continue
            else:
                containment_mode = "body"
                if not (c_body_top <= m_body_top and c_body_bottom >= m_body_bottom):
                    continue
                if c_body_top == m_body_top and c_body_bottom == m_body_bottom:
                    continue
            body_ratio = child_body_pct / max(mother_body_pct, 1e-9)
            pattern_high = max(float(mother["high"]), float(child["high"]))
            pattern_low = min(float(mother["low"]), float(child["low"]))
            breakout = self._breakout_candidate(df, idx, pattern_high, pattern_low)
            if breakout is None:
                continue
            breakout_idx, direction, breakout_price = breakout
            if any(abs(breakout_idx - used_idx) <= self.config.breakout_cooldown_bars for used_idx in used):
                continue
            variant = _variant_for(mother, child, self.config)
            if variant == "neutral_harami":
                continue  # siết 14/08/2026: sách chỉ công nhận harami đảo chiều màu nến — loại cùng chiều
            # Measure rule EC ch.43-46: target = breakout ± ((HH − LL) của cả 2 nến ×
            # multiplier theo chương). (Trước đây tạm dùng nguyên chiều cao nến mẹ — SAI
            # theo sách, đã hiệu chỉnh 13/08 theo family_harami_20260813.md.)
            mult_up, mult_down = _MULTIPLIER_PCT.get(variant, (0.65, 0.65))
            pattern_height_abs = pattern_high - pattern_low
            multiplier = mult_up if direction == "up" else mult_down
            target_price = breakout_price + pattern_height_abs * multiplier if direction == "up" else breakout_price - pattern_height_abs * multiplier
            if target_price <= 0:
                continue
            target_dist_pct = abs(target_price - breakout_price) / breakout_price * 100.0
            prior_trend = _prior_trend_pct(df, idx - 1, self.config.prior_trend_lookback_bars)
            volume_ratio = _rolling_volume_ratio(df, idx, lookback=20)
            volume_contracts = volume_ratio is not None and volume_ratio < 0.85
            pattern_height_pct = pattern_height_abs / max(float(c_close), 1e-9) * 100.0
            mother_breakout = (
                (direction == "up" and breakout_price > float(mother["high"]))
                or (direction == "down" and breakout_price < float(mother["low"]))
            )
            score = 34.0
            score += _score_band(float(body_ratio), good=0.30, weak=0.99, reverse=True, weight=0.22)
            score += _score_band(abs(float(prior_trend or 0.0)), good=6.0, weak=self.config.prior_trend_min_abs_pct, weight=0.12)
            score += 8.0 if volume_contracts else 2.0
            score += 8.0 if mother_breakout else 0.0
            score += _score_band(float(target_dist_pct), good=1.2, weak=8.0, reverse=True, weight=0.10)
            score = int(max(0, min(100, round(score))))
            used.append(breakout_idx)
            rows.append(
                {
                    "symbol": str(df.iloc[0]["symbol"]),
                    "pattern_key": HARAMI,
                    "variant": variant,
                    "formation_start_idx": int(idx - 1),
                    "formation_end_idx": int(idx),
                    "formation_start_date": str(pd.Timestamp(df.iloc[idx - 1]["date"]).date()),
                    "formation_end_date": str(pd.Timestamp(df.iloc[idx]["date"]).date()),
                    "breakout_idx": int(breakout_idx),
                    "breakout_date": str(pd.Timestamp(df.iloc[breakout_idx]["date"]).date()),
                    "breakout_direction": direction,
                    "breakout_price": round(float(breakout_price), 4),
                    "target_price": round(float(target_price), 4),
                    "target_dist_pct": round(float(target_dist_pct), 2),
                    "mother_body_top": round(float(m_body_top), 4),
                    "mother_body_bottom": round(float(m_body_bottom), 4),
                    "child_body_top": round(float(c_body_top), 4),
                    "child_body_bottom": round(float(c_body_bottom), 4),
                    "mother_bar_high": round(float(mother["high"]), 4),
                    "mother_bar_low": round(float(mother["low"]), 4),
                    # 14/08: thêm high/low nến con để thước đo tự kiểm containment RANGE
                    # của harami cross (doji) — không phụ thuộc cờ containment_mode của detector
                    "child_bar_high": round(float(child["high"]), 4),
                    "child_bar_low": round(float(child["low"]), 4),
                    "pattern_width_bars": 2,
                    "pattern_height_pct": round(float(pattern_height_pct), 2),
                    "target_multiplier_pct": round(float(multiplier * 100.0), 1),
                    "containment_mode": containment_mode,
                    "mother_body_pct": round(float(mother_body_pct), 4),
                    "child_body_pct": round(float(child_body_pct), 4),
                    "body_ratio": round(float(body_ratio), 4),
                    "prior_trend_pct": round(float(prior_trend), 2) if prior_trend is not None else None,
                    "volume_ratio_20": volume_ratio,
                    "volume_contracts": bool(volume_contracts),
                    "mother_bar_breakout": bool(mother_breakout),
                    "breakout_lag_bars": int(breakout_idx - idx),
                    "pattern_quality_score": score,
                    "pattern_quality_tier": "clean" if score >= 76 else ("usable" if score >= 60 else "loose"),
                }
            )
            if len(rows) >= self.config.max_events_per_symbol:
                break
        return rows


def scan_symbol(
    df_raw: pd.DataFrame,
    *,
    detector_config: Optional[HaramiConfig | Mapping[str, Any]] = None,
    max_events_per_symbol: Optional[int] = None,
) -> Tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(df_raw) < 80:
        return [], {"rows": int(len(df_raw)), "skipped": "too_few_rows"}
    config = detector_config if isinstance(detector_config, HaramiConfig) else HaramiConfig.from_mapping(detector_config)
    if max_events_per_symbol is not None:
        config = HaramiConfig.from_mapping({**config.to_dict(), "max_events_per_symbol": int(max_events_per_symbol)})
    df, norm_stats = OHLCVNormalizer().normalize(df_raw)
    rows = HaramiDetector(config).scan(df)
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({**row, **_evaluate_detection(df, row, lookahead=_registry_lookahead(row.get("pattern_key") or HARAMI))})
    return out, {"rows": int(len(df)), "normalizer": norm_stats, "detector_config": config.to_dict()}


def _assign_publication_quality_tiers(rows: list[dict[str, Any]]) -> None:
    data_limited = {"short_path", "zero_and_stale", "zero_volume", "mixed_flag"}
    for row in rows:
        path_bucket = str(row.get("path_quality_bucket") or "unknown")
        tradability_bucket = str(row.get("tradability_quality_bucket") or "unknown")
        if path_bucket in data_limited or tradability_bucket == "impaired":
            row["publication_quality_score"] = 0.0
            row["publication_quality_tier"] = "data_limited"
            row["publication_quality_reasons"] = f"path:{path_bucket},tradability:{tradability_bucket}"
            continue
        score = 0.0
        score += _score_band(_safe_float(row.get("body_ratio")), good=0.30, weak=0.99, reverse=True, weight=0.28)
        score += _score_band(abs(float(row.get("prior_trend_pct") or 0.0)), good=6.0, weak=2.0, weight=0.12)
        if _truthy(row.get("volume_contracts")):
            score += 10.0
        if _truthy(row.get("mother_bar_breakout")):
            score += 10.0
        if path_bucket == "clean":
            score += 10.0
        elif path_bucket == "usable":
            score += 5.0
        if tradability_bucket == "clean":
            score += 6.0
        elif tradability_bucket == "usable":
            score += 3.0
        reasons: list[str] = []
        if not _truthy(row.get("mother_bar_breakout")):
            reasons.append("not_mother_bar_breakout")
        if not _truthy(row.get("volume_contracts")):
            reasons.append("no_volume_contraction")
        score = round(float(max(0.0, min(100.0, score))), 2)
        row["publication_quality_score"] = score
        row["publication_quality_tier"] = "premium" if score >= 70 and path_bucket == "clean" else ("standard" if score >= 52 else "loose")
        row["publication_quality_reasons"] = ",".join(sorted(set(reasons)))


def _group_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    evals = [row for row in rows if row.get("mfe_pct") is not None]
    median_mfe = _median([row.get("mfe_pct") for row in evals])
    median_mae = _median([row.get("mae_pct") for row in evals])
    return {
        "detection_count": len(rows),
        "evaluated_count": len(evals),
        "n": len(rows),
        "median_mfe_pct": median_mfe,
        "median_mae_pct": median_mae,
        "mfe_mae_median_ratio": round(float(median_mfe) / max(float(median_mae), 1.0), 2) if median_mfe is not None and median_mae is not None else None,
        "average_mfe_pct": _mean([row.get("mfe_pct") for row in evals]),
        "average_mae_pct": _mean([row.get("mae_pct") for row in evals]),
        "target_hit_rate": _rate(evals, "target_hit"),
        "failure_5pct_rate": _rate(evals, "failure_5pct"),
        "target_first_before_adverse_5pct_rate": _rate(evals, "target_first_before_adverse_5pct"),
        "throwback_pullback_30d_rate": _rate(evals, "throwback_pullback_30d"),
        "median_target_dist_pct": _median([row.get("target_dist_pct") for row in evals]),
        "median_quality_score": _median([row.get("pattern_quality_score") for row in rows]),
        "median_publication_quality_score": _median([row.get("publication_quality_score") for row in rows]),
        "median_body_ratio": _median([row.get("body_ratio") for row in rows]),
        "median_child_body_pct": _median([row.get("child_body_pct") for row in rows]),
    }


def _group_table(rows: Sequence[Mapping[str, Any]], column: str, labels: Sequence[str]) -> Dict[str, Any]:
    return {label: _group_stats([row for row in rows if str(row.get(column) or "unknown") == label]) for label in labels}


def summarize(scan: Mapping[str, Any]) -> Dict[str, Any]:
    rows = list(scan.get("detections") or [])
    evals = [row for row in rows if row.get("mfe_pct") is not None]
    return {
        "generated_at": _utc_now(),
        "pattern_key": HARAMI,
        "symbols_scanned": int(scan.get("symbols_scanned") or 0),
        "detection_count": len(rows),
        "evaluated_count": len(evals),
        **_group_stats(rows),
        "variant_table": _group_table(rows, "variant", ("bullish_harami", "bearish_harami", "harami_cross_bearish", "harami_cross_bullish", "neutral_harami")),
        "direction_table": _group_table(rows, "breakout_direction", ("up", "down")),
        "quality_table": {tier: _group_stats([row for row in rows if row.get("pattern_quality_tier") == tier]) for tier in ("clean", "usable", "loose")},
        "publication_quality_table": {tier: _group_stats([row for row in rows if row.get("publication_quality_tier") == tier]) for tier in ("premium", "standard", "loose", "data_limited")},
        "regime_groups": _group_table(rows, "market_regime", ("bull", "bear", "unknown")),
        "market_group_table": _group_table(rows, "market_group", ("VN30", "VN100 ex VN30", "Outside VN100")),
        "liquidity_proxy_table": _group_table(rows, "liquidity_bucket", ("high", "mid", "low", "unknown")),
        "path_quality_audit": {
            "bucket_counts": dict(pd.Series([str(row.get("path_quality_bucket") or "unknown") for row in rows]).value_counts().sort_index()),
            "median_coverage_60d": _median([row.get("evaluated_bars") for row in rows]),
        },
        "symbol_concentration": {
            "symbols_with_events": len({str(row.get("symbol")) for row in rows if row.get("symbol")}),
            "top10_symbol_share_pct": round(float(pd.Series([str(row.get("symbol")) for row in rows if row.get("symbol")]).value_counts().head(10).sum()) / max(len(rows), 1) * 100.0, 2),
        },
        "quantile_metrics": {
            "fav_exc_pct": _quantiles([row.get("mfe_pct") for row in evals]),
            "adv_exc_pct": _quantiles([row.get("mae_pct") for row in evals]),
            "target_dist_pct": _quantiles([row.get("target_dist_pct") for row in evals]),
            "pattern_height_pct": _quantiles([row.get("pattern_height_pct") for row in rows]),
            "body_ratio": _quantiles([row.get("body_ratio") for row in rows]),
            "target_days": _quantiles([row.get("days_to_target") for row in evals]),
        },
        "experiment_note": (
            "Harami Family (EC ch.43-46): harami thường đo BODY-containment, harami cross đo "
            "RANGE-containment (doji). Measure rule EC: breakout ± ((HH−LL) × multiplier 58-74% "
            "theo chương). Sách KHÔNG publish failure rate; reversal rate 47-57% (gần ngẫu nhiên)."
        ),
    }


def _add_target_calibration(stats: Dict[str, Any], scan: Mapping[str, Any], path_rows: Sequence[Mapping[str, Any]]) -> None:
    events = pd.DataFrame(list(scan.get("detections") or []))
    path = pd.DataFrame(list(path_rows))
    if events.empty:
        stats["target_family_sensitivity"] = []
        stats["target_calibration_decision"] = None
        return
    if "event_id" not in events.columns and "detection_id" in events.columns:
        events["event_id"] = events["detection_id"]
    sensitivity = target_sensitivity(PatternArtifacts(HARAMI, events, path), HARAMI, horizon_days=10)
    stats["target_family_sensitivity"] = sensitivity
    stats["target_calibration_decision"] = (build_target_calibration_decisions(sensitivity, family_labels=(HARAMI,)) or [None])[0]
    stats["target_family"] = {"half_mother_range": 0.5, "full_mother_range": 1.0, "two_x_mother_range": 2.0}


EVENT_FIELDS = [
    "detection_id",
    "symbol",
    "variant",
    "market_group",
    "market_regime",
    "formation_start_date",
    "formation_end_date",
    "breakout_date",
    "breakout_direction",
    "breakout_price",
    "b_exec_price",
    "target_price",
    "target_dist_pct",
    "target_multiplier_pct",
    "containment_mode",
    "mfe_pct",
    "mae_pct",
    "mfe_pct_full",
    "mae_pct_full",
    "target_hit",
    "failure_5pct", "weak_move_5pct", "failure_busted", "days_to_bust",
    "target_first_before_adverse_5pct",
    "days_to_target",
    "throwback_pullback_30d",
    "pattern_quality_score",
    "pattern_quality_tier",
    "publication_quality_score",
    "publication_quality_tier",
    "publication_quality_reasons",
    "pattern_width_bars",
    "pattern_height_pct",
    "body_ratio",
    "mother_body_pct",
    "child_body_pct",
    "mother_body_top",
    "mother_body_bottom",
    "child_body_top",
    "child_body_bottom",
    "mother_bar_high",
    "mother_bar_low",
    "prior_trend_pct",
    "volume_ratio_20",
    "volume_contracts",
    "mother_bar_breakout",
    "breakout_lag_bars",
    "evaluated_bars",
    "is_primary_event_60d",
    "liquidity_bucket",
    "path_quality_bucket",
    "tradability_quality_bucket",
    "tradability_quality_score",
    "missing_bar_rate_60d",
    "zero_volume_rate_60d",
    "price_limit_proxy_rate_60d",
]


def scan_harami_db(
    *,
    db_path: Path,
    out_dir: Path,
    allowed_symbols: Optional[Sequence[str]] = None,
    detector_config: Optional[Mapping[str, Any]] = None,
    limit_symbols: Optional[int] = None,
    index_db: Path = DEFAULT_INDEX_DB,
    index_symbol: str = DEFAULT_INDEX_SYMBOL,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = HaramiConfig.from_mapping(detector_config)
    symbols = _symbols_in_db(db_path, allowed_symbols)
    if limit_symbols is not None:
        symbols = symbols[: int(limit_symbols)]
    detections: list[dict[str, Any]] = []
    symbol_stats: list[dict[str, Any]] = []
    series_by_symbol: dict[str, pd.DataFrame] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        for symbol in symbols:
            try:
                frame = _load_symbol_from_db(conn, symbol)
                rows, stats = scan_symbol(frame, detector_config=config)
                if rows:
                    series_by_symbol[symbol] = OHLCVNormalizer().normalize(frame)[0]
                detections.extend(rows)
                symbol_stats.append({"symbol": symbol, "detections": len(rows), **stats})
            except Exception as exc:
                symbol_stats.append({"symbol": symbol, "detections": 0, "error": str(exc)})
    finally:
        conn.close()
    for i, row in enumerate(detections):
        row["detection_id"] = f"{HARAMI}:{i + 1:06d}"
    detections, regime_meta = classify_market_regimes(detections, index_db=index_db, index_symbol=index_symbol)
    market_group_meta = attach_current_market_groups(detections)
    scan: dict[str, Any] = {
        "generated_at": _utc_now(),
        "source": "Market Cache latest.sqlite stock_price_history",
        "db_path": str(db_path),
        "pattern_key": HARAMI,
        "symbols_scanned": len(symbols),
        "detections": detections,
        "symbol_stats": symbol_stats,
        "regime": regime_meta,
        "market_group": market_group_meta,
        "detector_config": config.to_dict(),
    }
    _enrich_events_from_series(scan, series_by_symbol, corporate_db=index_db)
    _assign_publication_quality_tiers(scan["detections"])
    path_rows = _path_rows_from_series(scan, series_by_symbol, horizon_bars=_registry_lookahead(scan["pattern_key"]))
    stats = summarize(scan)
    stats["source"] = scan["source"]
    stats["db_source_meta"] = _db_meta(db_path)
    stats["detector_config"] = config.to_dict()
    _add_target_calibration(stats, scan, path_rows)
    paths = {
        "detections": out_dir / "detections.json",
        "statistics": out_dir / "statistics.json",
        "events_csv": out_dir / "events.csv",
        "post_breakout_path_csv": out_dir / "post_breakout_path.csv",
    }
    _write_json(paths["detections"], scan)
    _write_json(paths["statistics"], stats)
    _write_csv(paths["events_csv"], scan.get("detections") or [], EVENT_FIELDS)
    _write_csv(
        paths["post_breakout_path_csv"],
        path_rows,
        ["event_id", "symbol", "trade_date", "bar_after_breakout", "open", "high", "low", "close", "volume", "signed_close_return_pct", "signed_high_excursion_pct", "signed_low_excursion_pct"],
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Harami Family scanner against Market Cache latest.sqlite.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--market-stats-json", default=str(DEFAULT_MARKET_STATS_JSON))
    parser.add_argument("--limit-symbols", type=int, default=None)
    args = parser.parse_args()
    active_meta = _load_active_symbols(Path(args.market_stats_json) if args.market_stats_json else None)
    active_symbols = active_meta.get("active_symbols") if active_meta.get("enabled") else None
    paths = scan_harami_db(
        db_path=Path(args.db),
        out_dir=Path(args.out_dir) / HARAMI / "db_active",
        allowed_symbols=active_symbols,
        limit_symbols=args.limit_symbols,
    )
    print(json.dumps({"status": "PASS", "outputs": {key: str(value) for key, value in paths.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
