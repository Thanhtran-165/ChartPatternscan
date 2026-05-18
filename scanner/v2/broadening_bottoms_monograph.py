"""End-to-end Broadening Bottoms V2 monograph pipeline."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import textwrap
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from ..ohlcv_normalizer import OHLCVNormalizer
from ..pivot_detector import PivotDetector, Pivot, PivotType
from .broadening_bottoms import run_broadening_bottoms_fixture
from .contracts import ScannerV2Engine, canonical_spec_hash, load_core_registry
from .release_gate import enrich_payload_with_p1_p5_status
from .source_alignment import verify_pattern_source_alignment


PATTERN_KEY = "broadening_bottoms"
DEFAULT_SOURCE_DIR = Path("../market_stats/web/stock_series")
DEFAULT_OUT_DIR = Path("artifacts/scanner_v2/broadening_bottoms")
DEFAULT_MEMBERSHIP_DB = Path("../market_stats/cache/membership_history.sqlite")
DEFAULT_INDEX_DB = Path("vietnam_stocks.db")
DEFAULT_INDEX_SYMBOL = "VNINDEX"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _symbol_from_path(path: Path) -> str:
    return path.stem.split(" ", 1)[0].strip().upper()


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load_market_stats_symbol(path: Path) -> pd.DataFrame:
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list of OHLCV rows")
    symbol = _symbol_from_path(path)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"])
    cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    return df[cols].copy()


def _load_index_series(index_db: Path, index_symbol: str) -> pd.DataFrame:
    if not index_db.exists():
        return pd.DataFrame(columns=["date", "close"])
    conn = sqlite3.connect(str(index_db))
    try:
        df = pd.read_sql_query(
            "SELECT time AS date, close FROM stock_price_history WHERE symbol = ? ORDER BY time",
            conn,
            params=[index_symbol],
        )
    except Exception:
        return pd.DataFrame(columns=["date", "close"])
    finally:
        conn.close()
    if df.empty:
        return pd.DataFrame(columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date")[["date", "close"]].copy()


def _classify_market_regimes(
    detections: Sequence[Mapping[str, Any]],
    *,
    index_db: Path = DEFAULT_INDEX_DB,
    index_symbol: str = DEFAULT_INDEX_SYMBOL,
    anchor_field: str = "formation_start_date",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = [dict(d) for d in detections]
    if not rows:
        return rows, {
            "enabled": False,
            "reason": "no_detections",
            "index_symbol": index_symbol,
            "anchor_field": anchor_field,
        }
    index_df = _load_index_series(index_db, index_symbol)
    if index_df.empty:
        for row in rows:
            row["market_regime"] = "unknown"
        return rows, {
            "enabled": False,
            "reason": "missing_index_series",
            "index_db": str(index_db),
            "index_symbol": index_symbol,
            "anchor_field": anchor_field,
        }

    anchors = pd.DataFrame(
        {
            "row_id": list(range(len(rows))),
            "anchor_date": [d.get(anchor_field) for d in rows],
        }
    )
    anchors["anchor_date"] = pd.to_datetime(anchors["anchor_date"], errors="coerce")
    anchors = anchors.dropna(subset=["anchor_date"]).sort_values("anchor_date")
    idx = index_df.rename(columns={"close": "index_close"}).sort_values("date")

    at_anchor = pd.merge_asof(anchors, idx, left_on="anchor_date", right_on="date", direction="backward").rename(
        columns={"index_close": "close_anchor"}
    )
    lookback = anchors[["row_id", "anchor_date"]].copy()
    lookback["lookback_date"] = lookback["anchor_date"] - pd.DateOffset(months=18)
    at_lookback = pd.merge_asof(
        lookback[["row_id", "lookback_date"]].sort_values("lookback_date"),
        idx,
        left_on="lookback_date",
        right_on="date",
        direction="backward",
    ).rename(columns={"index_close": "close_lookback"})

    merged = at_anchor.merge(at_lookback[["row_id", "close_lookback"]], on="row_id", how="left")
    close_anchor = pd.to_numeric(merged["close_anchor"], errors="coerce")
    close_lookback = pd.to_numeric(merged["close_lookback"], errors="coerce")
    regimes: Dict[int, str] = {}
    for row_id, ca, cl in zip(merged["row_id"], close_anchor, close_lookback):
        if pd.isna(ca) or pd.isna(cl):
            regimes[int(row_id)] = "unknown"
        elif float(ca) > float(cl):
            regimes[int(row_id)] = "bull"
        else:
            regimes[int(row_id)] = "bear"
    for i, row in enumerate(rows):
        row["market_regime"] = regimes.get(i, "unknown")
    return rows, {
        "enabled": True,
        "method": "VNINDEX 18-month close change at formation_start_date",
        "index_db": str(index_db),
        "index_symbol": index_symbol,
        "anchor_field": anchor_field,
        "index_rows": int(len(index_df)),
        "unknown_count": sum(1 for row in rows if row.get("market_regime") == "unknown"),
    }


def _pivot_to_fixture_dict(pivot: Pivot) -> Dict[str, Any]:
    return {
        "idx": int(pivot.idx),
        "type": "H" if pivot.type == PivotType.HIGH else "L",
        "price": float(pivot.price),
    }


def _prior_trend_direction(df: pd.DataFrame, start_idx: int, *, lookback: int = 42, min_change_pct: float = 5.0) -> Tuple[str, float]:
    if start_idx < lookback:
        return "unknown", 0.0
    p0 = float(df.iloc[start_idx - lookback]["close"])
    p1 = float(df.iloc[start_idx]["close"])
    if p0 <= 0:
        return "unknown", 0.0
    change = (p1 - p0) / p0 * 100.0
    if change <= -min_change_pct:
        return "down", change
    if change >= min_change_pct:
        return "up", change
    return "sideways", change


def _post_closes(df: pd.DataFrame, formation_end_idx: int, *, bars: int = 40) -> List[Dict[str, Any]]:
    end = min(len(df), formation_end_idx + 1 + bars)
    return [
        {"idx": int(i), "close": float(df.iloc[i]["close"])}
        for i in range(formation_end_idx + 1, end)
        if pd.notna(df.iloc[i]["close"])
    ]


def _window_width(window: Sequence[Pivot]) -> int:
    return int(window[-1].idx - window[0].idx) if window else 0


def _formation_bounds(window: Sequence[Pivot]) -> Tuple[float, float, float]:
    highs = [float(p.price) for p in window if p.type == PivotType.HIGH]
    lows = [float(p.price) for p in window if p.type == PivotType.LOW]
    high = max(highs)
    low = min(lows)
    return high, low, high - low


def _alternation_ratio(window: Sequence[Pivot]) -> float:
    if len(window) < 2:
        return 0.0
    transitions = sum(1 for a, b in zip(window, window[1:]) if a.type != b.type)
    return transitions / (len(window) - 1)


def _pivot_slope(points: Sequence[Pivot]) -> Optional[float]:
    if len(points) < 2:
        return None
    first = points[0]
    last = points[-1]
    dx = int(last.idx) - int(first.idx)
    if dx <= 0:
        return None
    return (float(last.price) - float(first.price)) / dx


def _quality_tier(score: int) -> str:
    if score >= 75:
        return "clean"
    if score >= 55:
        return "usable"
    return "loose"


def _quality_assessment(
    *,
    window: Sequence[Pivot],
    formation_high: float,
    formation_low: float,
    height: float,
    width: int,
    prior_change_pct: float,
    breakout_price: float,
    breakout_direction: str,
) -> Dict[str, Any]:
    """Score morphology quality without changing the official detection gate.

    The V2 detector stays source-rule driven. This score is a research layer for
    answering whether weak report quality comes from broad rule acceptance.
    """

    score = 0
    reasons: List[str] = []
    highs = [p for p in window if p.type == PivotType.HIGH]
    lows = [p for p in window if p.type == PivotType.LOW]
    height_pct = height / formation_low * 100.0 if formation_low > 0 else 0.0
    alternation = _alternation_ratio(window)

    if prior_change_pct <= -12.0:
        score += 20
    elif prior_change_pct <= -8.0:
        score += 15
    elif prior_change_pct <= -5.0:
        score += 10
    else:
        reasons.append("prior_trend_too_shallow")

    if len(window) >= 6:
        score += 15
    elif len(window) == 5:
        score += 10
    elif len(window) == 4:
        score += 5
        reasons.append("minimum_touch_count_only")

    if 42 <= width <= 168:
        score += 15
    elif 21 <= width <= 252:
        score += 8
        reasons.append("width_outside_preferred_band")
    else:
        reasons.append("width_outside_allowed_band")

    if 8.0 <= height_pct <= 40.0:
        score += 15
    elif 5.0 <= height_pct <= 60.0:
        score += 8
        reasons.append("height_outside_preferred_band")
    else:
        reasons.append("height_extreme_or_too_small")

    if alternation >= 0.8:
        score += 10
    elif alternation >= 0.6:
        score += 5
        reasons.append("pivot_sequence_not_cleanly_alternating")
    else:
        reasons.append("pivot_sequence_choppy")

    high_slope = _pivot_slope(highs)
    low_slope = _pivot_slope(lows)
    if high_slope is not None and low_slope is not None and low_slope != 0:
        slope_ratio = abs(high_slope / low_slope)
        if 0.33 <= slope_ratio <= 3.0:
            score += 10
        elif 0.2 <= slope_ratio <= 5.0:
            score += 5
            reasons.append("trendline_slope_imbalance")
        else:
            reasons.append("trendline_slope_extreme_imbalance")
    else:
        slope_ratio = None
        reasons.append("slope_not_measurable")

    if height > 0 and breakout_price > 0:
        if breakout_direction == "up":
            clearance = (breakout_price - formation_high) / height
        else:
            clearance = (formation_low - breakout_price) / height
        if 0.02 <= clearance <= 0.35:
            score += 15
        elif 0.0 < clearance <= 0.5:
            score += 8
            reasons.append("breakout_clearance_outside_preferred_band")
        else:
            reasons.append("breakout_clearance_extreme_or_missing")
    else:
        clearance = None
        reasons.append("breakout_clearance_not_measurable")

    score = min(100, int(score))
    return {
        "pattern_quality_score": score,
        "pattern_quality_tier": _quality_tier(score),
        "pattern_quality_reasons": reasons,
        "alternation_ratio": round(float(alternation), 3),
        "slope_ratio": round(float(slope_ratio), 3) if slope_ratio is not None else None,
        "breakout_clearance_ratio": round(float(clearance), 3) if clearance is not None else None,
    }


def _measure_rule_target(detection: Mapping[str, Any]) -> Tuple[Optional[float], str]:
    pivots = list(detection.get("pivots") or [])
    highs = [p for p in pivots if str(p.get("type") or "").upper() == "H"]
    lows = [p for p in pivots if str(p.get("type") or "").upper() == "L"]
    if not highs or not lows:
        return None, "missing_minor_extremes"
    height = float(detection["pattern_height"])
    direction = str(detection["breakout_direction"])
    if direction == "up":
        recent_high = max(highs, key=lambda p: int(p.get("idx") or -1))
        return float(recent_high["price"]) + height, "recent_minor_high_plus_height"
    if direction == "down":
        recent_low = max(lows, key=lambda p: int(p.get("idx") or -1))
        return float(recent_low["price"]) - height, "recent_minor_low_minus_height"
    return None, "unknown_breakout_direction"


def _evaluate_detection(df: pd.DataFrame, detection: Mapping[str, Any], *, lookahead: int = 60) -> Dict[str, Any]:
    breakout_idx = int(detection["breakout_idx"])
    breakout_price = float(detection["breakout_price"])
    direction = str(detection["breakout_direction"])
    target, target_method = _measure_rule_target(detection)
    end = min(len(df), breakout_idx + 1 + lookahead)
    future = df.iloc[breakout_idx + 1 : end]
    b_exec_price = None
    if not future.empty and pd.notna(future.iloc[0].get("open")):
        b_exec_price = float(future.iloc[0]["open"])
    if future.empty or breakout_price <= 0 or target is None:
        return {
            "lookahead_bars": lookahead,
            "evaluated_bars": 0,
            "b_ref_price": round(float(breakout_price), 4),
            "b_exec_price": round(float(b_exec_price), 4) if b_exec_price is not None else None,
            "mfe_pct": None,
            "mae_pct": None,
            "failure_5pct": None,
            "failure_10pct": None,
            "failure_20pct": None,
            "failure_40pct": None,
            "target_method": target_method,
            "target_rule_id": "bb.measure.height_from_recent_extreme",
            "target_price": round(float(target), 4) if target is not None else None,
            "target_dist_pct": None,
            "target_hit": None,
            "days_to_target": None,
            "target_first_before_adverse_5pct": None,
            "tbpb_30": None,
            "days_to_tbpb": None,
        }
    target_dist_pct = abs(float(target) - breakout_price) / breakout_price * 100.0
    if direction == "up":
        mfe = (float(future["high"].max()) - breakout_price) / breakout_price * 100.0
        mae = (breakout_price - float(future["low"].min())) / breakout_price * 100.0
        target_hit = bool(float(future["high"].max()) >= target)
    else:
        mfe = (breakout_price - float(future["low"].min())) / breakout_price * 100.0
        mae = (float(future["high"].max()) - breakout_price) / breakout_price * 100.0
        target_hit = bool(float(future["low"].min()) <= target)
    days_to_target: Optional[int] = None
    days_to_adverse_5: Optional[int] = None
    days_to_tbpb: Optional[int] = None
    tbpb_tolerance = 0.005
    for offset, (_, row) in enumerate(future.iterrows(), start=1):
        high = float(row["high"])
        low = float(row["low"])
        if direction == "up":
            if days_to_target is None and high >= float(target):
                days_to_target = offset
            if days_to_adverse_5 is None and low <= breakout_price * 0.95:
                days_to_adverse_5 = offset
            if offset <= 30 and days_to_tbpb is None and low <= breakout_price * (1 + tbpb_tolerance):
                days_to_tbpb = offset
        else:
            if days_to_target is None and low <= float(target):
                days_to_target = offset
            if days_to_adverse_5 is None and high >= breakout_price * 1.05:
                days_to_adverse_5 = offset
            if offset <= 30 and days_to_tbpb is None and high >= breakout_price * (1 - tbpb_tolerance):
                days_to_tbpb = offset
    if days_to_target is None:
        target_first = False
    elif days_to_adverse_5 is None:
        target_first = True
    else:
        target_first = days_to_target < days_to_adverse_5
    return {
        "lookahead_bars": lookahead,
        "evaluated_bars": int(len(future)),
        "b_ref_price": round(float(breakout_price), 4),
        "b_exec_price": round(float(b_exec_price), 4) if b_exec_price is not None else None,
        "mfe_pct": round(float(mfe), 2),
        "mae_pct": round(float(mae), 2),
        "failure_5pct": bool(float(mfe) < 5.0),
        "failure_10pct": bool(float(mfe) < 10.0),
        "failure_20pct": bool(float(mfe) < 20.0),
        "failure_40pct": bool(float(mfe) < 40.0),
        "target_method": target_method,
        "target_rule_id": "bb.measure.height_from_recent_extreme",
        "target_price": round(float(target), 4),
        "target_dist_pct": round(float(target_dist_pct), 2),
        "target_hit": target_hit,
        "days_to_target": int(days_to_target) if days_to_target is not None else None,
        "target_first_before_adverse_5pct": bool(target_first),
        "tbpb_30": days_to_tbpb is not None,
        "days_to_tbpb": int(days_to_tbpb) if days_to_tbpb is not None else None,
    }


def scan_symbol(df_raw: pd.DataFrame, *, max_windows_per_symbol: int = 8) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(df_raw) < 260:
        return [], {"rows": int(len(df_raw)), "skipped": "too_few_rows"}

    normalizer = OHLCVNormalizer()
    df, norm_stats = normalizer.normalize(df_raw)
    if len(df) < 260:
        return [], {"rows": int(len(df)), "normalizer": norm_stats, "skipped": "too_few_rows_after_normalize"}

    pivots = PivotDetector().detect_pivots(df, pivot_type="intermediate")
    detections: List[Dict[str, Any]] = []
    used_breakouts: List[int] = []

    for end_pos in range(3, len(pivots)):
        for size in range(4, 9):
            start_pos = end_pos - size + 1
            if start_pos < 0:
                continue
            window = pivots[start_pos : end_pos + 1]
            width = _window_width(window)
            if width < 21 or width > 252:
                continue

            prior_dir, prior_change = _prior_trend_direction(df, int(window[0].idx))
            if prior_dir != "down":
                continue

            fixture = {
                "prior_trend": {"direction": prior_dir, "change_pct": prior_change},
                "pivots": [_pivot_to_fixture_dict(p) for p in window],
                "post_formation_closes": _post_closes(df, int(window[-1].idx)),
            }
            result = run_broadening_bottoms_fixture(fixture)
            if not result.matched or result.breakout_idx is None:
                continue
            if any(abs(int(result.breakout_idx) - prev) <= 20 for prev in used_breakouts):
                continue

            formation_high, formation_low, height = _formation_bounds(window)
            symbol = str(df.iloc[0]["symbol"])
            quality = _quality_assessment(
                window=window,
                formation_high=formation_high,
                formation_low=formation_low,
                height=height,
                width=width,
                prior_change_pct=prior_change,
                breakout_price=float(result.breakout_price or 0),
                breakout_direction=str(result.breakout_direction),
            )
            record: Dict[str, Any] = {
                "symbol": symbol,
                "pattern_key": PATTERN_KEY,
                "formation_start_idx": int(window[0].idx),
                "formation_end_idx": int(window[-1].idx),
                "formation_start_date": str(pd.Timestamp(df.iloc[int(window[0].idx)]["date"]).date()),
                "formation_end_date": str(pd.Timestamp(df.iloc[int(window[-1].idx)]["date"]).date()),
                "breakout_idx": int(result.breakout_idx),
                "breakout_date": str(pd.Timestamp(df.iloc[int(result.breakout_idx)]["date"]).date()),
                "breakout_direction": str(result.breakout_direction),
                "breakout_price": round(float(result.breakout_price or 0), 4),
                "pattern_height": round(float(height), 4),
                "pattern_height_pct": round(float(height / formation_low * 100.0), 2) if formation_low > 0 else None,
                "pattern_width_bars": int(width),
                "formation_high": round(float(formation_high), 4),
                "formation_low": round(float(formation_low), 4),
                "prior_trend_change_pct": round(float(prior_change), 2),
                "touch_count": int(len(window)),
                "pivot_indices": [int(p.idx) for p in window],
                "pivots": [_pivot_to_fixture_dict(p) for p in window],
                **quality,
            }
            record.update(_evaluate_detection(df, record))
            detections.append(record)
            used_breakouts.append(int(result.breakout_idx))
            if len(detections) >= max_windows_per_symbol:
                break
        if len(detections) >= max_windows_per_symbol:
            break

    return detections, {"rows": int(len(df)), "pivots": int(len(pivots)), "normalizer": norm_stats}


def scan_market_stats(
    source_dir: Path,
    *,
    limit_symbols: Optional[int] = None,
    index_db: Path = DEFAULT_INDEX_DB,
    index_symbol: str = DEFAULT_INDEX_SYMBOL,
) -> Dict[str, Any]:
    paths = sorted(source_dir.glob("*.json"))
    if limit_symbols is not None:
        paths = paths[: int(limit_symbols)]
    all_detections: List[Dict[str, Any]] = []
    symbol_rows: List[Dict[str, Any]] = []
    for path in paths:
        try:
            df = _load_market_stats_symbol(path)
            detections, stats = scan_symbol(df)
            all_detections.extend(detections)
            symbol_rows.append({"symbol": _symbol_from_path(path), "detections": len(detections), **stats})
        except Exception as exc:
            symbol_rows.append({"symbol": _symbol_from_path(path), "detections": 0, "error": str(exc)})
    registry = load_core_registry()
    compiled = ScannerV2Engine(registry=registry).compile_pattern(PATTERN_KEY, require_official=True)
    for i, row in enumerate(all_detections):
        row["detection_id"] = f"{PATTERN_KEY}:{i + 1:06d}"
        row.update(compiled.result_metadata())
    all_detections, regime_meta = _classify_market_regimes(
        all_detections,
        index_db=index_db,
        index_symbol=index_symbol,
    )
    market_group_meta = _attach_current_market_groups(all_detections)
    return {
        "generated_at": _utc_now(),
        "source": "Market Stats V1 stock_series JSON",
        "source_dir": str(source_dir),
        "pattern_key": PATTERN_KEY,
        "scanner_metadata": compiled.result_metadata(),
        "regime": regime_meta,
        "market_group": market_group_meta,
        "symbols_scanned": len(paths),
        "detections": all_detections,
        "symbol_stats": symbol_rows,
    }


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return round(float(np.median(vals)), 2)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return round(float(np.mean(vals)), 2)


def _quantiles(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    points = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    if not vals:
        return {f"P{p}": None for p in points}
    return {f"P{p}": round(float(np.percentile(vals, p)), 2) for p in points}


def _pct(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(numerator / denominator * 100.0, 2)


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    vals = [row.get(key) for row in rows if row.get(key) is not None]
    if not vals:
        return None
    return _pct(sum(1 for val in vals if val is True), len(vals))


def _breakout_group_stats(detections: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    evals = [d for d in detections if d.get("mfe_pct") is not None]
    failures = [d for d in evals if d.get("failure_5pct") is True]
    target_hits = [d for d in evals if d.get("target_hit") is True]
    return {
        "detection_count": len(detections),
        "evaluated_count": len(evals),
        "median_mfe_pct": _median(d.get("mfe_pct") for d in evals),
        "median_mae_pct": _median(d.get("mae_pct") for d in evals),
        "average_mfe_pct": _mean(d.get("mfe_pct") for d in evals),
        "average_mae_pct": _mean(d.get("mae_pct") for d in evals),
        "failure_5pct_rate": _pct(len(failures), len(evals)),
        "target_hit_rate": _pct(len(target_hits), len(evals)),
        "target_first_before_adverse_5pct_rate": _rate(evals, "target_first_before_adverse_5pct"),
        "tbpb_30_rate": _rate(evals, "tbpb_30"),
        "median_days_to_target": _median(d.get("days_to_target") for d in evals),
        "median_days_to_tbpb": _median(d.get("days_to_tbpb") for d in evals),
        "median_width_bars": _median(d.get("pattern_width_bars") for d in detections),
        "median_height_pct": _median(d.get("pattern_height_pct") for d in detections),
    }


def _failure_ladder(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    evals = [row for row in rows if row.get("mfe_pct") is not None]
    return {
        "fail_5pct_rate": _rate(evals, "failure_5pct"),
        "fail_10pct_rate": _rate(evals, "failure_10pct"),
        "fail_20pct_rate": _rate(evals, "failure_20pct"),
        "fail_40pct_rate": _rate(evals, "failure_40pct"),
    }


def _symbol_concentration(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "UNKNOWN").upper()
        counts[symbol] = counts.get(symbol, 0) + 1
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if total <= 0:
        return {"top10_symbol_share": None, "hhi_symbol": None, "top_symbols": []}
    return {
        "top10_symbol_share": round(sum(n for _, n in ranked[:10]) / total * 100.0, 2),
        "hhi_symbol": round(sum((n / total) ** 2 for _, n in ranked), 4),
        "top_symbols": [{"symbol": sym, "events": n} for sym, n in ranked[:10]],
    }


def _quality_table(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tiers = ("clean", "usable", "loose")
    return {
        tier: _breakout_group_stats([row for row in rows if str(row.get("pattern_quality_tier") or "loose") == tier])
        for tier in tiers
    }


def _market_group(symbol: str, vn30: set[str], vn100: set[str]) -> str:
    sym = str(symbol).upper()
    if sym in vn30:
        return "VN30"
    if sym in vn100:
        return "VN100 ex VN30"
    return "Outside VN100"


def _attach_current_market_groups(detections: Sequence[Dict[str, Any]], membership_db: Path = DEFAULT_MEMBERSHIP_DB) -> Dict[str, Any]:
    vn30 = _load_current_members("VN30", membership_db)
    vn100 = _load_current_members("VN100", membership_db)
    for row in detections:
        row["market_group"] = _market_group(str(row.get("symbol") or ""), vn30, vn100)
    return {
        "method": "current membership snapshot from Market Stats V1 membership DB",
        "point_in_time": False,
        "membership_db": str(membership_db),
        "vn30_members": len(vn30),
        "vn100_members": len(vn100),
    }


def summarize_statistics(scan: Mapping[str, Any]) -> Dict[str, Any]:
    detections = list(scan.get("detections", []))
    up = [d for d in detections if d.get("breakout_direction") == "up"]
    down = [d for d in detections if d.get("breakout_direction") == "down"]
    evals = [d for d in detections if d.get("mfe_pct") is not None]
    failures = [d for d in evals if d.get("failure_5pct") is True]
    target_hits = [d for d in evals if d.get("target_hit") is True]
    breakout_groups = {
        "all": _breakout_group_stats(detections),
        "up": _breakout_group_stats(up),
        "down": _breakout_group_stats(down),
    }
    regime_groups = {
        regime: _breakout_group_stats([d for d in detections if str(d.get("market_regime") or "unknown") == regime])
        for regime in ("bull", "bear", "unknown")
    }
    market_group_table = {
        group: _breakout_group_stats([d for d in detections if str(d.get("market_group") or "Outside VN100") == group])
        for group in ("VN30", "VN100 ex VN30", "Outside VN100")
    }
    quantile_metrics = {
        "width_days": _quantiles(d.get("pattern_width_bars") for d in detections),
        "height_pct": _quantiles(d.get("pattern_height_pct") for d in detections),
        "target_dist_pct": _quantiles(d.get("target_dist_pct") for d in evals),
        "fav_exc_pct": _quantiles(d.get("mfe_pct") for d in evals),
        "adv_exc_pct": _quantiles(d.get("mae_pct") for d in evals),
        "time_to_target": _quantiles(d.get("days_to_target") for d in evals),
    }
    return {
        "generated_at": _utc_now(),
        "pattern_key": PATTERN_KEY,
        "source": scan.get("source"),
        "symbols_scanned": int(scan.get("symbols_scanned") or 0),
        "detection_count": len(detections),
        "evaluated_count": len(evals),
        "up_breakouts": len(up),
        "down_breakouts": len(down),
        "median_mfe_pct": _median(d.get("mfe_pct") for d in evals),
        "median_mae_pct": _median(d.get("mae_pct") for d in evals),
        "average_mfe_pct": _mean(d.get("mfe_pct") for d in evals),
        "average_mae_pct": _mean(d.get("mae_pct") for d in evals),
        "failure_5pct_rate": _pct(len(failures), len(evals)),
        "target_hit_rate": _pct(len(target_hits), len(evals)),
        "median_width_bars": _median(d.get("pattern_width_bars") for d in detections),
        "median_height_pct": _median(d.get("pattern_height_pct") for d in detections),
        "anchor_mode": "B_ref_and_B_exec",
        "failure_ladder": _failure_ladder(detections),
        "target_first_before_adverse_5pct": _rate(evals, "target_first_before_adverse_5pct"),
        "tbpb_30_rate": _rate(evals, "tbpb_30"),
        "median_quality_score": _median(d.get("pattern_quality_score") for d in detections),
        "quality_tier_counts": {
            tier: sum(1 for d in detections if str(d.get("pattern_quality_tier") or "loose") == tier)
            for tier in ("clean", "usable", "loose")
        },
        "time_to_target": {
            "median_days_to_target": _median(d.get("days_to_target") for d in evals),
            "evaluated_count": len(evals),
            "censored_count": sum(1 for d in evals if d.get("days_to_target") is None),
        },
        "quantile_metrics": quantile_metrics,
        "symbol_concentration": _symbol_concentration(detections),
        "breakout_groups": breakout_groups,
        "regime_groups": regime_groups,
        "market_group_table": market_group_table,
        "quality_table": _quality_table(detections),
        "failure_target_table": {
            "overall": {
                **_failure_ladder(detections),
                "target_hit_rate": _pct(len(target_hits), len(evals)),
                "target_first_before_adverse_5pct_rate": _rate(evals, "target_first_before_adverse_5pct"),
                "median_target_dist_pct": _median(d.get("target_dist_pct") for d in evals),
                "median_days_to_target": _median(d.get("days_to_target") for d in evals),
            },
            "by_direction": {
                "up": {
                    **_failure_ladder(up),
                    "target_hit_rate": _rate([d for d in up if d.get("mfe_pct") is not None], "target_hit"),
                    "median_days_to_target": _median(d.get("days_to_target") for d in up),
                },
                "down": {
                    **_failure_ladder(down),
                    "target_hit_rate": _rate([d for d in down if d.get("mfe_pct") is not None], "target_hit"),
                    "median_days_to_target": _median(d.get("days_to_target") for d in down),
                },
            },
        },
        "post_breakout_table": {
            "lookahead_bars": 60,
            "tbpb_30_rate": _rate(evals, "tbpb_30"),
            "median_days_to_tbpb": _median(d.get("days_to_tbpb") for d in evals),
            "median_mfe_pct": _median(d.get("mfe_pct") for d in evals),
            "median_mae_pct": _median(d.get("mae_pct") for d in evals),
            "target_first_before_adverse_5pct_rate": _rate(evals, "target_first_before_adverse_5pct"),
        },
        "regime": scan.get("regime") or {},
        "market_group": scan.get("market_group") or {},
        "calculation_notes": {
            "mfe_pct": "Biên thuận lợi lớn nhất trong 60 phiên sau phá vỡ.",
            "mae_pct": "Biên bất lợi lớn nhất trong 60 phiên sau phá vỡ.",
            "failure_5pct_rate": "Tỷ lệ mẫu có biên thuận lợi lớn nhất thấp hơn 5%.",
            "target_hit_rate": "Tỷ lệ đạt mục tiêu giá theo Bảng 1.8: cộng/trừ chiều cao mẫu hình từ đỉnh/đáy phụ gần nhất.",
            "tbpb_30_rate": "Tỷ lệ quay lại vùng giá phá vỡ trong 30 phiên với dung sai 0,5%.",
        },
    }


def _top_examples(detections: Sequence[Mapping[str, Any]], *, count: int = 3) -> List[Dict[str, Any]]:
    ranked = sorted(detections, key=lambda d: (float(d.get("mfe_pct") or -999), -int(d.get("breakout_idx") or 0)), reverse=True)
    return [dict(row) for row in ranked[:count]]


def _adverse_abs(detection: Mapping[str, Any]) -> float:
    value = detection.get("mae_pct")
    if value is None:
        return -999.0
    return abs(float(value))


def _research_examples(detections: Sequence[Mapping[str, Any]], *, count: int = 4) -> List[Dict[str, Any]]:
    rows = [dict(d) for d in detections if d.get("mfe_pct") is not None]
    selected: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()

    def add(category: str, candidates: Sequence[Mapping[str, Any]]) -> None:
        for item in candidates:
            key = (str(item.get("symbol")), str(item.get("breakout_date")))
            if key in seen:
                continue
            row = dict(item)
            row["example_category"] = category
            selected.append(row)
            seen.add(key)
            return

    winners = sorted(
        [d for d in rows if d.get("target_hit") is True],
        key=lambda d: (float(d.get("mfe_pct") or -999), -int(d.get("breakout_idx") or 0)),
        reverse=True,
    )
    misses = sorted(
        [d for d in rows if d.get("target_hit") is not True],
        key=lambda d: (float(d.get("mfe_pct") or 999), int(d.get("breakout_idx") or 0)),
    )
    failures = sorted(
        [d for d in rows if d.get("failure_5pct") is True],
        key=lambda d: (float(d.get("mfe_pct") or 999), -_adverse_abs(d)),
    )
    adverse = sorted(rows, key=lambda d: (_adverse_abs(d), float(d.get("mfe_pct") or -999)), reverse=True)

    add("đạt mục tiêu mạnh", winners)
    add("đạt mục tiêu mạnh", winners[1:])
    add("không đạt mục tiêu", misses)
    add("biên bất lợi lớn", adverse)
    add("thất bại theo ngưỡng nhỏ", failures)
    if len(selected) < count:
        for row in _top_examples(rows, count=count):
            add("bổ sung", [row])
            if len(selected) >= count:
                break
    return selected[:count]


def _load_current_members(index_code: str, membership_db: Path = DEFAULT_MEMBERSHIP_DB) -> set[str]:
    if not membership_db.exists():
        return set()
    conn = sqlite3.connect(str(membership_db))
    try:
        rows = conn.execute(
            """
            SELECT ticker
            FROM index_membership_history
            WHERE index_code = ? AND effective_to IS NULL
            """,
            (index_code,),
        ).fetchall()
    finally:
        conn.close()
    return {str(row[0]).upper() for row in rows}


def _top_examples_for_universe(
    detections: Sequence[Mapping[str, Any]],
    *,
    universe: str = "VN100",
    count: int = 4,
    membership_db: Path = DEFAULT_MEMBERSHIP_DB,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    members = _load_current_members(universe, membership_db)
    if members:
        filtered = [d for d in detections if str(d.get("symbol") or "").upper() in members]
        if filtered:
            return _research_examples(filtered, count=count), {
                "universe": universe,
                "membership_count": len(members),
                "eligible_detection_count": len(filtered),
                "fallback_used": False,
        }
    return _research_examples(detections, count=count), {
        "universe": universe,
        "membership_count": len(members),
        "eligible_detection_count": 0,
        "fallback_used": True,
    }


EVENT_FIELDNAMES = [
    "event_id",
    "pattern_id",
    "symbol",
    "market_group",
    "formation_start",
    "formation_end",
    "breakout_date",
    "breakout_direction",
    "breakout_price",
    "b_exec_price",
    "target_price",
    "target_dist_pct",
    "market_regime",
    "rule_version",
    "scanner_version",
    "data_version",
    "overlap_group_id",
    "is_primary_event",
    "mfe_pct",
    "mae_pct",
    "pattern_quality_score",
    "pattern_quality_tier",
    "failure_5pct",
    "failure_10pct",
    "failure_20pct",
    "failure_40pct",
    "target_hit",
    "days_to_target",
    "target_first_before_adverse_5pct",
    "tbpb_30",
    "days_to_tbpb",
    "evaluated_bars",
]

PATH_FIELDNAMES = [
    "event_id",
    "symbol",
    "trade_date",
    "bar_after_breakout",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "signed_close_return_pct",
    "signed_high_excursion_pct",
    "signed_low_excursion_pct",
]


def _event_record(detection: Mapping[str, Any]) -> Dict[str, Any]:
    event_id = str(detection.get("detection_id") or "")
    return {
        "event_id": event_id,
        "pattern_id": PATTERN_KEY,
        "symbol": detection.get("symbol"),
        "market_group": detection.get("market_group"),
        "formation_start": detection.get("formation_start_date"),
        "formation_end": detection.get("formation_end_date"),
        "breakout_date": detection.get("breakout_date"),
        "breakout_direction": detection.get("breakout_direction"),
        "breakout_price": detection.get("breakout_price"),
        "b_exec_price": detection.get("b_exec_price"),
        "target_price": detection.get("target_price"),
        "target_dist_pct": detection.get("target_dist_pct"),
        "market_regime": detection.get("market_regime"),
        "rule_version": detection.get("spec_hash"),
        "scanner_version": detection.get("scanner_version"),
        "data_version": "market_stats_v1_stock_series_json",
        "overlap_group_id": f"{detection.get('symbol')}_{detection.get('breakout_date')}",
        "is_primary_event": True,
        "mfe_pct": detection.get("mfe_pct"),
        "mae_pct": detection.get("mae_pct"),
        "pattern_quality_score": detection.get("pattern_quality_score"),
        "pattern_quality_tier": detection.get("pattern_quality_tier"),
        "failure_5pct": detection.get("failure_5pct"),
        "failure_10pct": detection.get("failure_10pct"),
        "failure_20pct": detection.get("failure_20pct"),
        "failure_40pct": detection.get("failure_40pct"),
        "target_hit": detection.get("target_hit"),
        "days_to_target": detection.get("days_to_target"),
        "target_first_before_adverse_5pct": detection.get("target_first_before_adverse_5pct"),
        "tbpb_30": detection.get("tbpb_30"),
        "days_to_tbpb": detection.get("days_to_tbpb"),
        "evaluated_bars": detection.get("evaluated_bars"),
    }


def _path_rows_for_detection(df: pd.DataFrame, detection: Mapping[str, Any], *, horizon_bars: int = 120) -> List[Dict[str, Any]]:
    breakout_idx = int(detection["breakout_idx"])
    breakout_price = float(detection["breakout_price"])
    direction = 1 if str(detection.get("breakout_direction")) == "up" else -1
    end = min(len(df), breakout_idx + 1 + horizon_bars)
    rows: List[Dict[str, Any]] = []
    if breakout_price <= 0:
        return rows
    for offset, (_, row) in enumerate(df.iloc[breakout_idx + 1 : end].iterrows(), start=1):
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        if direction == 1:
            signed_high = (high - breakout_price) / breakout_price * 100.0
            signed_low = (low - breakout_price) / breakout_price * 100.0
            signed_close = (close - breakout_price) / breakout_price * 100.0
        else:
            signed_high = (breakout_price - low) / breakout_price * 100.0
            signed_low = (breakout_price - high) / breakout_price * 100.0
            signed_close = (breakout_price - close) / breakout_price * 100.0
        rows.append(
            {
                "event_id": detection.get("detection_id"),
                "symbol": detection.get("symbol"),
                "trade_date": str(pd.Timestamp(row["date"]).date()),
                "bar_after_breakout": offset,
                "open": round(float(row["open"]), 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(close, 4),
                "volume": int(row["volume"]) if pd.notna(row["volume"]) else None,
                "signed_close_return_pct": round(float(signed_close), 4),
                "signed_high_excursion_pct": round(float(signed_high), 4),
                "signed_low_excursion_pct": round(float(signed_low), 4),
            }
        )
    return rows


def write_event_artifacts(scan: Mapping[str, Any], *, source_dir: Path, out_dir: Path, horizon_bars: int = 120) -> Dict[str, Any]:
    detections = list(scan.get("detections") or [])
    event_rows = [_event_record(d) for d in detections]
    symbol_paths = {_symbol_from_path(path): path for path in sorted(source_dir.glob("*.json"))}
    df_cache: Dict[str, pd.DataFrame] = {}
    path_rows: List[Dict[str, Any]] = []
    missing_symbols: List[str] = []
    for detection in detections:
        symbol = str(detection.get("symbol") or "").upper()
        path = symbol_paths.get(symbol)
        if path is None:
            missing_symbols.append(symbol)
            continue
        if symbol not in df_cache:
            df_cache[symbol] = _load_market_stats_symbol(path).reset_index(drop=True)
        path_rows.extend(_path_rows_for_detection(df_cache[symbol], detection, horizon_bars=horizon_bars))

    event_json = out_dir / "events.json"
    event_csv = out_dir / "events.csv"
    path_json = out_dir / "post_breakout_path.json"
    path_csv = out_dir / "post_breakout_path.csv"
    _write_json(event_json, event_rows)
    _write_csv(event_csv, event_rows, EVENT_FIELDNAMES)
    _write_json(path_json, path_rows)
    _write_csv(path_csv, path_rows, PATH_FIELDNAMES)
    return {
        "event_json": str(event_json),
        "event_csv": str(event_csv),
        "post_breakout_path": str(path_json),
        "post_breakout_path_csv": str(path_csv),
        "path_horizon_bars": horizon_bars,
        "event_count": len(event_rows),
        "path_row_count": len(path_rows),
        "missing_path_symbol_count": len(set(missing_symbols)),
    }


def build_payload(
    scan: Mapping[str, Any],
    stats: Mapping[str, Any],
    *,
    example_universe: str = "VN100",
    event_artifacts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    registry = load_core_registry()
    pattern = registry["patterns"][PATTERN_KEY]
    alignment = verify_pattern_source_alignment(PATTERN_KEY, registry=registry)
    contract = ScannerV2Engine(registry=registry).compile_pattern(PATTERN_KEY, require_official=True)
    detections = list(scan.get("detections", []))
    examples, example_scope = _top_examples_for_universe(detections, universe=example_universe)
    payload = {
        "generated_at": _utc_now(),
        "pattern_key": PATTERN_KEY,
        "display_name": pattern["display_name"],
        "source_document": registry["source_document"],
        "source_alignment": alignment,
        "scanner_contract": contract.result_metadata(),
        "rules": pattern["rules"],
        "golden_fixtures": pattern["golden_fixtures"],
        "market_data": {
            "source": scan.get("source"),
            "source_dir": scan.get("source_dir"),
            "symbols_scanned": scan.get("symbols_scanned"),
            "regime": scan.get("regime") or {},
            "market_group": scan.get("market_group") or {},
        },
        "event_artifacts": dict(event_artifacts or {}),
        "sample_policy": {
            "overlap_policy": "one event per symbol within 20 breakout bars in the V2 scanner loop",
            "censoring_policy": "post-breakout paths are right-censored at available data or configured horizon",
        },
        "statistics": dict(stats),
        "example_scope": {
            **example_scope,
            "selection_mode": "ranked_mixed_research_examples",
            "target_selection_mode": "seeded_stratified_random",
            "seeded_stratified_random_ready": False,
        },
        "example_detections": examples,
        "investment_reference_status": {
            "level": "research_only_draft",
            "bulkowski_for_vietnam_score_estimate": "chưa đạt 85-90 theo chuẩn P1-P5",
            "tradable_setup": False,
            "requirements_met": [
                "quy tắc nhận diện có truy vết nguồn",
                "quy tắc mục tiêu giá có truy vết nguồn",
                "quét toàn thị trường bằng Market Stats V1",
                "thống kê theo hướng phá vỡ",
                "phân nhóm bối cảnh VNINDEX khi có dữ liệu chỉ số",
                "bộ ví dụ hỗn hợp từ VN100",
            ],
            "remaining_limits": [
                "chưa có point-in-time universe gồm mã hủy niêm yết hoặc tạm ngừng",
                "chưa có audit corporate actions point-in-time",
                "chưa có event-level JSON/CSV và OHLC path hậu phá vỡ đầy đủ",
                "chưa có CI, bootstrap, KM và release gate pass",
            ],
        },
        "limitations": [
            "Đây là lượt dựng chuyên khảo V2 đầu tiên.",
            "Bộ phát hiện được giữ nghiêm và có truy vết nguồn trước khi hiệu chỉnh rộng.",
            "Quy tắc mục tiêu giá đã được đối chiếu từ Bảng 1.8, nhưng các chiến thuật phụ như tăng/giảm một phần và quản trị điểm dừng chưa được triển khai.",
        ],
    }
    return enrich_payload_with_p1_p5_status(payload)


def render_core_markdown(payload: Mapping[str, Any]) -> str:
    stats = payload["statistics"]
    rules = payload["rules"]
    examples = payload["example_detections"]
    evidence_vi = {
        "bb.prior_trend.down": "xu hướng giá ngắn hạn nên đi xuống",
        "bb.shape.megaphone": "đỉnh cao dần và đáy thấp dần",
        "bb.trendlines.diverge": "đường trên dốc lên và đường dưới dốc xuống",
        "bb.touches.min_two_each": "tối thiểu hai đỉnh phụ và hai đáy phụ",
        "bb.volume.context": "xu hướng khối lượng thường tăng, đôi khi có dạng chữ U",
        "bb.breakout.close_either_side": "giá đóng cửa vượt lên trên đỉnh mẫu hình",
        "bb.measure.height_from_recent_extreme": "highest high and the lowest low in the formation",
        "bb.invalidation.not_broadening": "dạng hình loa với đỉnh cao dần và đáy thấp dần",
    }
    interpretation_vi = {
        "bb.prior_trend.down": "Yêu cầu xu hướng giảm trước khi mẫu hình bắt đầu.",
        "bb.shape.megaphone": "Yêu cầu biên dao động mở rộng: đỉnh sau cao hơn và đáy sau thấp hơn.",
        "bb.trendlines.diverge": "Dựng đường biên trên dốc lên và đường biên dưới dốc xuống cho mẫu hình.",
        "bb.touches.min_two_each": "Yêu cầu tối thiểu hai lần chạm ở phía đỉnh và hai lần chạm ở phía đáy.",
        "bb.volume.context": "Ghi nhận xu hướng hoặc hình dạng khối lượng như bối cảnh, không dùng làm điều kiện bắt buộc.",
        "bb.breakout.close_either_side": "Xác nhận phá vỡ khi giá đóng cửa vượt lên trên đỉnh mẫu hình hoặc xuống dưới đáy mẫu hình.",
        "bb.measure.height_from_recent_extreme": "Đo chiều cao mẫu hình rồi cộng vào đỉnh phụ gần nhất cho phá vỡ lên hoặc trừ khỏi đáy phụ gần nhất cho phá vỡ xuống.",
        "bb.invalidation.not_broadening": "Loại mẫu nếu đỉnh không cao dần hoặc đáy không thấp dần.",
    }
    lines: List[str] = []
    lines.append("# Đáy mở rộng - chuyên khảo Bộ quét V2 cho thị trường Việt Nam")
    lines.append("")
    lines.append("## Tóm tắt kết quả")
    lines.append("")
    lines.append("| Chỉ tiêu | Giá trị |")
    lines.append("|---|---:|")
    for label, key in [
        ("Số mã đã quét", "symbols_scanned"),
        ("Số mẫu phát hiện", "detection_count"),
        ("Số mẫu có đủ dữ liệu hậu phá vỡ", "evaluated_count"),
        ("Số lần phá vỡ lên", "up_breakouts"),
        ("Số lần phá vỡ xuống", "down_breakouts"),
        ("Trung vị biên thuận lợi lớn nhất (%)", "median_mfe_pct"),
        ("Trung vị biên bất lợi lớn nhất (%)", "median_mae_pct"),
        ("Trung bình biên thuận lợi lớn nhất (%)", "average_mfe_pct"),
        ("Trung bình biên bất lợi lớn nhất (%)", "average_mae_pct"),
        ("Tỷ lệ thất bại theo ngưỡng 5%", "failure_5pct_rate"),
        ("Tỷ lệ đạt mục tiêu giá", "target_hit_rate"),
        ("Trung vị độ rộng mẫu hình (phiên)", "median_width_bars"),
        ("Trung vị chiều cao mẫu hình (%)", "median_height_pct"),
    ]:
        value = stats.get(key)
        lines.append(f"| {label} | {value if value is not None else 'không có'} |")
    lines.append("")
    groups = stats.get("breakout_groups") or {}
    if groups:
        lines.append("## Hiệu suất theo hướng phá vỡ")
        lines.append("")
        lines.append("| Chỉ tiêu | Toàn bộ | Phá vỡ lên | Phá vỡ xuống |")
        lines.append("|---|---:|---:|---:|")
        for label, key in [
            ("Số mẫu phát hiện", "detection_count"),
            ("Số mẫu có đánh giá", "evaluated_count"),
            ("Trung vị biên thuận lợi lớn nhất (%)", "median_mfe_pct"),
            ("Trung vị biên bất lợi lớn nhất (%)", "median_mae_pct"),
            ("Trung bình biên thuận lợi lớn nhất (%)", "average_mfe_pct"),
            ("Trung bình biên bất lợi lớn nhất (%)", "average_mae_pct"),
            ("Tỷ lệ thất bại theo ngưỡng 5%", "failure_5pct_rate"),
            ("Tỷ lệ đạt mục tiêu giá", "target_hit_rate"),
            ("Trung vị độ rộng mẫu hình (phiên)", "median_width_bars"),
            ("Trung vị chiều cao mẫu hình (%)", "median_height_pct"),
        ]:
            values = []
            for group_key in ["all", "up", "down"]:
                group = groups.get(group_key) or {}
                values.append(group.get(key, "không có"))
            lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
        lines.append("")
    regimes = stats.get("regime_groups") or {}
    if regimes:
        lines.append("## Hiệu suất theo bối cảnh VNINDEX")
        lines.append("")
        lines.append("| Chỉ tiêu | Bull | Bear | Không xác định |")
        lines.append("|---|---:|---:|---:|")
        for label, key in [
            ("Số mẫu phát hiện", "detection_count"),
            ("Số mẫu có đánh giá", "evaluated_count"),
            ("Trung vị biên thuận lợi lớn nhất (%)", "median_mfe_pct"),
            ("Tỷ lệ thất bại theo ngưỡng 5%", "failure_5pct_rate"),
            ("Tỷ lệ đạt mục tiêu giá", "target_hit_rate"),
        ]:
            values = []
            for group_key in ["bull", "bear", "unknown"]:
                group = regimes.get(group_key) or {}
                values.append(group.get(key, "không có"))
            lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
        lines.append("")
    lines.append("## Quy tắc nhận diện")
    lines.append("")
    lines.append("- Xu hướng trước mẫu hình phải là xu hướng giảm.")
    lines.append("- Mẫu hình phải có đỉnh sau cao hơn đỉnh trước và đáy sau thấp hơn đáy trước.")
    lines.append("- Đường biên trên dốc lên và đường biên dưới dốc xuống.")
    lines.append("- Cần tối thiểu hai đỉnh phụ và hai đáy phụ.")
    lines.append("- Phá vỡ chỉ được tính khi giá đóng cửa vượt lên trên đỉnh mẫu hình hoặc xuống dưới đáy mẫu hình.")
    lines.append("")
    lines.append("## Truy vết quy tắc quét")
    lines.append("")
    lines.append("| Quy tắc | Trang nguồn | Bằng chứng ngắn | Cách diễn giải trong bộ quét |")
    lines.append("|---|---:|---|---|")
    for rule in rules:
        rule_id = str(rule["rule_id"])
        lines.append(
            f"| `{rule_id}` | {rule['source_page']} | {evidence_vi.get(rule_id, rule['evidence_excerpt'])} | {interpretation_vi.get(rule_id, rule['interpreted_rule'])} |"
        )
    lines.append("")
    lines.append("## Truy vết nguồn")
    lines.append("")
    alignment = payload["source_alignment"]
    lines.append(f"- Đối chiếu nguồn: {'đạt' if alignment.get('aligned') else 'không đạt'}")
    lines.append(f"- Mã bộ quét: `{payload['scanner_contract']['scanner_pattern_key']}`")
    lines.append(f"- Dấu vân tay đặc tả: `{payload['scanner_contract']['spec_hash']}`")
    lines.append("")
    lines.append("## Ví dụ phát hiện")
    lines.append("")
    scope = payload.get("example_scope") or {}
    if scope:
        fallback = "Có" if scope.get("fallback_used") else "Không"
        lines.append(
            f"Phạm vi ví dụ ưu tiên: {scope.get('universe')} "
            f"(số mã thành phần: {scope.get('membership_count')}, "
            f"số mẫu đủ điều kiện: {scope.get('eligible_detection_count')}, "
            f"dùng fallback: {fallback})."
        )
        lines.append("")
    if not examples:
        lines.append("Lượt chạy V2 đầu tiên không tạo ra mẫu phát hiện nào.")
    else:
        lines.append("| Loại ví dụ | Mã | Giai đoạn hình thành | Ngày phá vỡ | Hướng phá vỡ | Biên thuận lợi lớn nhất (%) | Đạt mục tiêu |")
        lines.append("|---|---|---|---|---|---:|---|")
        for ex in examples:
            direction = "lên" if ex.get("breakout_direction") == "up" else "xuống"
            target_hit = "có" if ex.get("target_hit") else "không"
            lines.append(
                f"| {ex.get('example_category', 'ví dụ')} | {ex['symbol']} | {ex['formation_start_date']} đến {ex['formation_end_date']} | {ex['breakout_date']} | {direction} | {ex.get('mfe_pct')} | {target_hit} |"
            )
    lines.append("")
    lines.append("## Trạng thái quản trị")
    lines.append("")
    lines.append("- Trạng thái mẫu hình: ứng viên chính thức của Bộ quét V2.")
    lines.append("- DeepSeek chỉ được dùng để viết nhận xét trên gói dữ kiện đã khóa.")
    lines.append("- Các số liệu ở trên được sinh bằng mã lệnh xác định trước khi có phần nhận xét.")
    lines.append("")
    lines.append("## Lưu ý và giới hạn hiện tại")
    lines.append("")
    for item in payload["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _extract_numbers(text: str) -> set[str]:
    numbers: set[str] = set()
    for raw in re.findall(r"(?<![A-Za-z])-?\d+(?:[.,]\d+)?%?", text or ""):
        is_pct = raw.endswith("%")
        token = raw[:-1] if is_pct else raw
        if "," in token:
            token = token.replace(",", ".")
        elif "." in token:
            left, right = token.split(".", 1)
            if len(right) == 3 and left.lstrip("-").isdigit():
                token = left + right
        try:
            value = float(token)
        except ValueError:
            normalized = token
        else:
            normalized = f"{value:g}"
        numbers.add(normalized)
    return numbers


def validate_commentary(commentary: str, allowed_text: str) -> Tuple[bool, List[str]]:
    allowed = _extract_numbers(allowed_text)
    used = _extract_numbers(commentary)
    unsupported = sorted(x for x in used if x not in allowed)
    return not unsupported, unsupported


COMMENTARY_FORBIDDEN_PATTERNS = [
    r"\bpattern\b",
    r"\bbreakout\b",
    r"\btarget\b",
    r"\bhit\b",
    r"\bmedian\b",
    r"\bMFE\b",
    r"\bMAE\b",
    r"\bpipeline\b",
    r"\bscanner\b",
    r"\bsnapshot\b",
    r"\bregistry\b",
    r"\bprovenance\b",
    r"\bfallback\b",
    r"\bpartial\b",
    r"\bEncyclopedia\b",
    r"\bChart Patterns\b",
    r"bb\.",
    r"pattern_key",
    r"spec_hash",
]


def validate_commentary_style(commentary: str) -> Tuple[bool, List[str]]:
    forbidden = []
    for pattern in COMMENTARY_FORBIDDEN_PATTERNS:
        if re.search(pattern, commentary or "", flags=re.IGNORECASE):
            forbidden.append(pattern)
    return not forbidden, forbidden


def validate_commentary_fact_consistency(commentary: str, payload: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    text = commentary or ""
    symbols_scanned = payload.get("market_data", {}).get("symbols_scanned")
    example_scope = payload.get("example_scope") or {}
    universe = str(example_scope.get("universe") or "")
    if symbols_scanned and universe:
        pattern = rf"\b{re.escape(str(symbols_scanned))}\s+mã\s+{re.escape(universe)}\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            issues.append(f"misstates {symbols_scanned} scanned symbols as {universe} members")
    rule_count = len(payload.get("rules") or [])
    if rule_count >= 8 and re.search(r"\bbảy\s+quy\s+tắc\b", text, flags=re.IGNORECASE):
        issues.append(f"misstates {rule_count} rules as seven")
    return not issues, issues


def _deepseek_chat(*, base_url: str, api_key: str, model: str, messages: List[Dict[str, str]], timeout_s: int) -> str:
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.2}).encode("utf-8")
    req = Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        msg = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"DeepSeek HTTPError {getattr(exc, 'code', '?')}: {msg}")
    except URLError as exc:
        raise RuntimeError(f"DeepSeek URLError: {exc}")
    return str(payload["choices"][0]["message"]["content"])


def _commentary_prompt(payload: Mapping[str, Any], core_md: str, *, forbid_digits: bool, repair_instructions: str = "") -> Tuple[str, List[str]]:
    required_headings = [
        "### Nhận xét chính",
        "### Đối chiếu với quy tắc nhận diện",
        "### Chất lượng mẫu và phép đo",
        "### Cách dùng trong nghiên cứu",
    ]
    number_rule = (
        "Không dùng bất kỳ chữ số nào trong commentary. Diễn giải định tính, không nhắc lại số liệu."
        if forbid_digits
        else "Được dùng số, nhưng chỉ dùng đúng số đã có trong payload/core. Không tự làm tròn, không đổi mốc đo, không thêm tỷ lệ mới."
    )
    examples = payload.get("example_detections") or []
    example_lines = [
        f"- {ex.get('symbol')}: hình thành {ex.get('formation_start_date')} đến {ex.get('formation_end_date')}, "
        f"phá vỡ {ex.get('breakout_date')}, hướng {'lên' if ex.get('breakout_direction') == 'up' else 'xuống'}, "
        f"loại ví dụ {ex.get('example_category')}, bối cảnh {ex.get('market_regime')}, "
        f"biên thuận lợi lớn nhất {ex.get('mfe_pct')}, biên bất lợi lớn nhất {ex.get('mae_pct')}, đạt mục tiêu: {ex.get('target_hit')}"
        for ex in examples[:3]
    ]
    prompt = (
        "Bạn là biên tập viên nghiên cứu kỹ thuật đang viết một chương chuyên khảo theo tinh thần sách tham khảo mẫu hình giá: "
        "mạch văn cô đọng, kiểm chứng được, ưu tiên mô tả quy tắc và kết quả quan sát hơn là lời khuyên giao dịch.\n"
        "Mục tiêu là làm phần diễn giải đọc giống một tài liệu nghiên cứu, không giống báo cáo dashboard hoặc quảng cáo tín hiệu.\n"
        "Không chép văn phong hoặc câu chữ từ sách nguồn. Chỉ diễn giải từ payload/core và các quy tắc đã truy vết.\n"
        "Không dùng các từ tiếng Anh như pattern, breakout, target, hit, median, MFE, MAE, pipeline, scanner, official ready, snapshot, registry, provenance, fallback.\n"
        "Không nêu tên tiếng Anh của sách nguồn trong thân bài; hãy gọi là tài liệu nguồn, ấn bản 2.\n"
        "Không đưa mã kỹ thuật vào bài viết: không viết các khóa như bb.prior_trend.down, bb.breakout.close_either_side, pattern_key, spec_hash.\n"
        "Hãy dùng các từ Việt: mẫu hình, phá vỡ, mục tiêu, đạt mục tiêu, trung vị, biên thuận lợi lớn nhất, biên bất lợi lớn nhất, quy trình, bộ quét, đạt chuẩn chính thức.\n"
        "Registry hiện có tám quy tắc cho mẫu này; không được viết 'bảy quy tắc'.\n"
        "Không đưa khuyến nghị mua/bán. Không viết câu khẳng định quá mức như chắc chắn sinh lợi, vượt trội rủi ro, khớp hoàn toàn, đáng mua, nên bán.\n"
        "Không được gọi quy tắc mục tiêu là tạm thời nếu payload cho biết target_rule_id là bb.measure.height_from_recent_extreme; phần còn thiếu là chiến thuật phụ.\n"
        "Tránh cụm 'lợi nhuận tiềm năng'; hãy dùng 'biên thuận lợi quan sát được' hoặc 'kết quả hậu phá vỡ'.\n"
        "Bắt buộc xử lý đủ bốn việc: "
        "một là diễn giải kết quả thống kê chính, bao gồm bảng phân nhóm toàn bộ/phá vỡ lên/phá vỡ xuống và bối cảnh VNINDEX nếu có; "
        "hai là đối chiếu từng nhóm quy tắc nhận diện với tài liệu nguồn ở mức khái niệm, không nêu mã quy tắc; "
        "ba là nêu rõ phép đo mục tiêu giá đã theo Bảng 1.8, còn phần chưa triển khai là chiến thuật phụ như tăng/giảm một phần và quản trị điểm dừng; "
        "bốn là giải thích vai trò của các ví dụ VN100 gồm cả mẫu đạt mục tiêu và mẫu không đạt/yếu, không phải khuyến nghị.\n"
        "Phân biệt rõ: số mã đã quét là toàn bộ nguồn Market Stats V1; VN100 chỉ là phạm vi ưu tiên chọn ví dụ. Tuyệt đối không viết '812 mã VN100'.\n"
        "Câu mở đầu nên theo nghĩa: 'Trên nguồn Market Stats V1 gồm 812 mã...' rồi sau đó mới nói ví dụ VN100 ở mục cuối.\n"
        "Ưu tiên câu ngắn, cụ thể. Mỗi mục khoảng 80 đến 130 từ. "
        "Chỉ dùng facts trong payload/core. "
        + number_rule
        + ("\nYêu cầu sửa lỗi bắt buộc: " + repair_instructions if repair_instructions else "")
        + "\nBắt buộc dùng đúng bốn tiêu đề sau:\n"
        + "\n".join(required_headings)
        + "\n\nVÍ DỤ VN100 ĐƯỢC PHÉP NHẮC:\n"
        + "\n".join(example_lines)
        + "\n\nPAYLOAD:\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        + "\n\nCORE:\n"
        + core_md
    )
    return prompt, required_headings


def generate_commentary(payload: Mapping[str, Any], core_md: str, *, timeout_s: int = 120) -> Tuple[str, Dict[str, Any]]:
    _load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    _, required_headings = _commentary_prompt(payload, core_md, forbid_digits=True)
    if not api_key:
        fallback = "\n\n".join([h + "\nKhông có khóa DeepSeek trong môi trường, nên phần nhận xét AI được bỏ qua." for h in required_headings])
        return fallback, {"status": "skipped_no_api_key", "model": model}

    attempts: List[Dict[str, Any]] = []
    attempt_configs = [
        (False, ""),
        (
            False,
            "Giữ số liệu thật, nhưng sửa rõ universe: 812 là số mã trong nguồn Market Stats V1, không phải VN100; VN100 chỉ dùng để chọn ví dụ PLX, PDR, VND.",
        ),
        (True, ""),
    ]
    for forbid_digits, repair_instructions in attempt_configs:
        prompt, required_headings = _commentary_prompt(payload, core_md, forbid_digits=forbid_digits, repair_instructions=repair_instructions)
        raw = _deepseek_chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": "Bạn viết nhận xét nghiên cứu bằng tiếng Việt tự nhiên, không bịa dữ kiện hoặc số liệu."},
                {"role": "user", "content": prompt},
            ],
            timeout_s=timeout_s,
        ).strip()
        ok, unsupported = validate_commentary(raw, core_md + "\n" + json.dumps(payload, ensure_ascii=False))
        style_ok, forbidden_terms = validate_commentary_style(raw)
        fact_ok, fact_issues = validate_commentary_fact_consistency(raw, payload)
        missing = [h for h in required_headings if h not in raw]
        attempts.append(
            {
                "forbid_digits": forbid_digits,
                "repair_instructions": repair_instructions,
                "unsupported_numbers": unsupported,
                "forbidden_terms": forbidden_terms,
                "fact_issues": fact_issues,
                "missing_headings": missing,
                "raw": raw,
            }
        )
        if ok and style_ok and fact_ok and not missing:
            return raw, {"status": "generated", "model": model, "forbid_digits": forbid_digits, "attempts": attempts}

    fallback = "\n\n".join([h + "\nNhận xét bị loại vì không đạt kiểm tra số liệu, thuật ngữ hoặc tiêu đề." for h in required_headings])
    return fallback, {"status": "validation_failed", "model": model, "attempts": attempts}


def _plain_md_lines(markdown: str) -> List[str]:
    lines: List[str] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = line.replace("`", "")
        if line.startswith("|"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if parts and not all(set(p) <= {"-", ":"} for p in parts):
                line = " | ".join(parts)
        lines.append(line)
    return lines


def _pdf_text_pages(pdf: PdfPages, title: str, markdown: str) -> None:
    lines = _plain_md_lines(markdown)
    page_lines: List[str] = []
    max_lines = 42
    for line in lines:
        wrapped = textwrap.wrap(line, width=105) if line else [""]
        for item in wrapped:
            page_lines.append(item)
            if len(page_lines) >= max_lines:
                _emit_text_page(pdf, title, page_lines)
                page_lines = []
    if page_lines:
        _emit_text_page(pdf, title, page_lines)


def _emit_text_page(pdf: PdfPages, title: str, lines: Sequence[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.965, title, fontsize=14, weight="bold", ha="left", va="top")
    y = 0.925
    for line in lines:
        fig.text(0.06, y, line, fontsize=8.2, ha="left", va="top", family="DejaVu Sans")
        y -= 0.021
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_wrapped_text(
    fig: Any,
    x: float,
    y: float,
    text: str,
    *,
    width: int = 88,
    fontsize: float = 9.0,
    line_height: float = 0.021,
    weight: str = "normal",
    color: str = "#111827",
) -> float:
    lines = textwrap.wrap(str(text), width=width) or [""]
    for line in lines:
        fig.text(x, y, line, fontsize=fontsize, ha="left", va="top", weight=weight, color=color, family="DejaVu Sans")
        y -= line_height
    return y


BOOK_BG = "#fffdf8"
BOOK_INK = "#171717"
BOOK_MUTED = "#555555"
BOOK_RULE = "#222222"


def _fig_line(fig: Any, x0: float, y0: float, x1: float, y1: float, *, lw: float = 0.7, color: str = BOOK_RULE) -> None:
    fig.lines.append(plt.Line2D([x0, x1], [y0, y1], transform=fig.transFigure, color=color, linewidth=lw))


def _book_page(*, running_title: str = "Đáy mở rộng", page_no: Optional[int] = None) -> Any:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(BOOK_BG)
    if page_no is not None:
        fig.text(0.12, 0.955, str(page_no), fontsize=10, weight="bold", family="DejaVu Serif", color=BOOK_INK)
        fig.text(0.18, 0.955, running_title, fontsize=10, family="DejaVu Serif", color=BOOK_INK)
    return fig


def _book_section_title(fig: Any, y: float, title: str, *, subtitle: Optional[str] = None) -> float:
    fig.text(0.5, y, title.upper(), fontsize=11, weight="bold", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    y -= 0.022
    if subtitle:
        fig.text(0.5, y, subtitle, fontsize=9.5, style="italic", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
        y -= 0.026
    _fig_line(fig, 0.12, y, 0.88, y, lw=0.8)
    return y - 0.018


def _book_table(
    fig: Any,
    *,
    x: float,
    y: float,
    width: float,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    col_fracs: Sequence[float],
    wrap_chars: Sequence[int],
    fontsize: float = 8.7,
    line_height: float = 0.0175,
    min_row_height: float = 0.034,
) -> float:
    pad = 0.006
    _fig_line(fig, x, y, x + width, y, lw=0.8)
    y -= 0.014
    col_x = [x]
    acc = x
    for frac in col_fracs[:-1]:
        acc += width * frac
        col_x.append(acc)
    for i, header in enumerate(headers):
        fig.text(col_x[i] + pad, y, header, fontsize=fontsize, weight="bold", ha="left", va="top", family="DejaVu Sans", color=BOOK_INK)
    y -= 0.024
    _fig_line(fig, x, y, x + width, y, lw=0.45)
    y -= 0.008
    for row in rows:
        wrapped_cols = [textwrap.wrap(str(cell), width=wrap_chars[i]) or [""] for i, cell in enumerate(row)]
        row_height = max(min_row_height, line_height * max(len(lines) for lines in wrapped_cols) + 0.014)
        for i, lines in enumerate(wrapped_cols):
            ty = y
            for line in lines:
                fig.text(col_x[i] + pad, ty, line, fontsize=fontsize, ha="left", va="top", family="DejaVu Sans", color=BOOK_INK)
                ty -= line_height
        y -= row_height
    _fig_line(fig, x, y + 0.006, x + width, y + 0.006, lw=0.8)
    return y - 0.026


def _draw_pattern_sketch(fig: Any) -> None:
    ax = fig.add_axes([0.34, 0.64, 0.32, 0.13])
    ax.set_facecolor(BOOK_BG)
    x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y = np.array([5.7, 4.2, 6.3, 3.4, 7.0, 2.8], dtype=float)
    ax.plot(x, y, color=BOOK_INK, linewidth=1.2)
    ax.scatter(x[[0, 2, 4]], y[[0, 2, 4]], s=16, color=BOOK_INK)
    ax.scatter(x[[1, 3, 5]], y[[1, 3, 5]], s=16, facecolor=BOOK_BG, edgecolor=BOOK_INK, linewidth=1.0)
    ax.plot([0, 4], [5.7, 7.0], color=BOOK_INK, linewidth=0.8)
    ax.plot([1, 5], [4.2, 2.8], color=BOOK_INK, linewidth=0.8)
    ax.annotate("đỉnh cao dần", xy=(3.2, 6.75), xytext=(2.1, 7.35), fontsize=7.2, family="DejaVu Sans", arrowprops={"arrowstyle": "-", "lw": 0.5})
    ax.annotate("đáy thấp dần", xy=(4.2, 3.1), xytext=(2.6, 2.25), fontsize=7.2, family="DejaVu Sans", arrowprops={"arrowstyle": "-", "lw": 0.5})
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _metric_value(stats: Mapping[str, Any], key: str) -> str:
    value = stats.get(key)
    if value is None:
        return "không có"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _metric_cell(row: Mapping[str, Any], key: str, *, pct: bool = False) -> str:
    value = row.get(key)
    if value is None:
        return "không có"
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text + ("%" if pct else "")


def _section_frame(title: str) -> Any:
    fig = _book_page()
    fig.text(0.12, 0.94, title, fontsize=15, weight="bold", ha="left", va="top", family="DejaVu Serif", color=BOOK_INK)
    _fig_line(fig, 0.12, 0.913, 0.88, 0.913, lw=0.8)
    return fig


def _render_snapshot_page(pdf: PdfPages, payload: Mapping[str, Any]) -> None:
    stats = payload["statistics"]
    fig = _book_page()
    fig.text(0.5, 0.89, "1", fontsize=25, weight="bold", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    fig.text(0.5, 0.845, "Đáy mở rộng", fontsize=22, weight="bold", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    fig.text(0.5, 0.812, "Chuyên khảo mẫu hình giá cho thị trường Việt Nam", fontsize=10.5, ha="center", va="top", family="DejaVu Serif", color=BOOK_MUTED)
    _draw_pattern_sketch(fig)

    y = _book_section_title(fig, 0.58, "Kết quả tóm tắt", subtitle="Dữ liệu Market Stats V1")
    rows = [
        ("Số mã đã quét", _metric_value(stats, "symbols_scanned")),
        ("Số mẫu phát hiện", _metric_value(stats, "detection_count")),
        ("Mẫu có đủ dữ liệu sau phá vỡ", _metric_value(stats, "evaluated_count")),
        ("Trung vị điểm chất lượng mẫu", _metric_value(stats, "median_quality_score")),
        ("Phá vỡ lên", _metric_value(stats, "up_breakouts")),
        ("Phá vỡ xuống", _metric_value(stats, "down_breakouts")),
        ("Trung vị biên thuận lợi lớn nhất", _metric_value(stats, "median_mfe_pct") + "%"),
        ("Trung vị biên bất lợi lớn nhất", _metric_value(stats, "median_mae_pct") + "%"),
        ("Tỷ lệ thất bại theo ngưỡng nhỏ", _metric_value(stats, "failure_5pct_rate") + "%"),
        ("Tỷ lệ đạt mục tiêu giá", _metric_value(stats, "target_hit_rate") + "%"),
    ]
    y = _book_table(
        fig,
        x=0.2,
        y=y,
        width=0.6,
        headers=["Chỉ tiêu", "Kết quả"],
        rows=rows,
        col_fracs=[0.68, 0.32],
        wrap_chars=[38, 14],
        fontsize=8.9,
        min_row_height=0.027,
    )
    _add_wrapped_text(
        fig,
        0.2,
        max(0.14, y),
        "Định nghĩa nhận diện và truy vết nguồn được trình bày ở Bảng 1.1; trang này chỉ giữ phần kết quả để đọc nhanh.",
        width=74,
        fontsize=8.2,
        line_height=0.018,
        color=BOOK_MUTED,
    )
    fig.text(0.12, 0.055, f"Nguồn: Market Stats V1. Dấu vân tay đặc tả: {payload['scanner_contract']['spec_hash']}", fontsize=6.8, family="DejaVu Sans", color=BOOK_MUTED)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_statistics_page(pdf: PdfPages, payload: Mapping[str, Any]) -> None:
    stats = payload["statistics"]
    groups = stats.get("breakout_groups") or {
        "all": stats,
        "up": {},
        "down": {},
    }
    fig = _book_page(running_title="Đáy mở rộng", page_no=2)
    fig.text(0.5, 0.90, "KẾT QUẢ THỊ TRƯỜNG VIỆT NAM", fontsize=13, weight="bold", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    fig.text(0.5, 0.875, "Bảng phân nhóm theo hướng phá vỡ", fontsize=9.5, style="italic", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    group_rows = []
    for label, key, is_pct in [
        ("Số mẫu phát hiện", "detection_count", False),
        ("Số mẫu có đánh giá", "evaluated_count", False),
        ("Trung vị biên thuận lợi lớn nhất", "median_mfe_pct", True),
        ("Trung vị biên bất lợi lớn nhất", "median_mae_pct", True),
        ("Trung bình biên thuận lợi lớn nhất", "average_mfe_pct", True),
        ("Trung bình biên bất lợi lớn nhất", "average_mae_pct", True),
        ("Tỷ lệ thất bại theo ngưỡng nhỏ", "failure_5pct_rate", True),
        ("Tỷ lệ đạt mục tiêu giá", "target_hit_rate", True),
        ("Trung vị độ rộng mẫu hình", "median_width_bars", False),
        ("Trung vị chiều cao mẫu hình", "median_height_pct", True),
    ]:
        group_rows.append(
            [
                label,
                _metric_cell(groups.get("all") or {}, key, pct=is_pct),
                _metric_cell(groups.get("up") or {}, key, pct=is_pct),
                _metric_cell(groups.get("down") or {}, key, pct=is_pct),
            ]
        )
    y = _book_table(
        fig,
        x=0.12,
        y=0.835,
        width=0.76,
        headers=["Chỉ tiêu", "Toàn bộ", "Phá vỡ lên", "Phá vỡ xuống"],
        rows=group_rows,
        col_fracs=[0.42, 0.18, 0.2, 0.2],
        wrap_chars=[30, 12, 12, 12],
        fontsize=8.1,
        line_height=0.0165,
        min_row_height=0.029,
    )

    y -= 0.006
    fig.text(0.12, y, "Đọc nhanh", fontsize=10.5, weight="bold", family="DejaVu Serif", color=BOOK_INK)
    y -= 0.026
    read_rows = [
        ("Nguồn dữ liệu", f"{stats.get('symbols_scanned')} mã từ Market Stats V1."),
        ("Phân bố hướng", f"{stats.get('up_breakouts')} phá vỡ lên và {stats.get('down_breakouts')} phá vỡ xuống."),
        ("Khung hậu phá vỡ", "Biên thuận lợi và bất lợi lớn nhất được đo trong 60 phiên sau phá vỡ."),
        ("Mục tiêu giá", "Đo theo Bảng 1.8: cộng/trừ chiều cao mẫu hình từ đỉnh/đáy phụ gần nhất."),
        ("Giới hạn", "Chưa triển khai chiến thuật phụ như tăng/giảm một phần và quản trị điểm dừng."),
    ]
    y = _book_table(
        fig,
        x=0.12,
        y=y,
        width=0.76,
        headers=["Mục", "Ghi chú"],
        rows=read_rows,
        col_fracs=[0.24, 0.76],
        wrap_chars=[18, 70],
        fontsize=7.9,
        line_height=0.0158,
        min_row_height=0.032,
    )
    y -= 0.005
    fig.text(0.12, y, "Cách đọc", fontsize=10.5, weight="bold", family="DejaVu Serif", color=BOOK_INK)
    y -= 0.026
    _add_wrapped_text(
        fig,
        0.14,
        y,
        "Bảng này đưa chương gần hơn với cấu trúc sách tham khảo: mỗi hướng phá vỡ được đọc riêng thay vì trộn vào một kết quả tổng. "
        "Tỷ lệ mục tiêu đã dùng quy tắc đo từ nguồn, nhưng chưa bao gồm các chiến thuật giao dịch phụ.",
        width=92,
        fontsize=8.2,
        line_height=0.0175,
        color=BOOK_INK,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_regime_page(pdf: PdfPages, payload: Mapping[str, Any]) -> None:
    stats = payload["statistics"]
    regimes = stats.get("regime_groups") or {}
    regime_meta = stats.get("regime") or {}
    fig = _book_page(running_title="Đáy mở rộng", page_no=3)
    fig.text(0.5, 0.90, "BỐI CẢNH THỊ TRƯỜNG", fontsize=13, weight="bold", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    fig.text(0.5, 0.875, "Phân nhóm theo VNINDEX", fontsize=9.5, style="italic", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    rows = []
    for label, key, is_pct in [
        ("Số mẫu phát hiện", "detection_count", False),
        ("Số mẫu có đánh giá", "evaluated_count", False),
        ("Trung vị biên thuận lợi lớn nhất", "median_mfe_pct", True),
        ("Tỷ lệ thất bại theo ngưỡng nhỏ", "failure_5pct_rate", True),
        ("Tỷ lệ đạt mục tiêu giá", "target_hit_rate", True),
        ("Trung vị độ rộng mẫu hình", "median_width_bars", False),
    ]:
        rows.append(
            [
                label,
                _metric_cell(regimes.get("bull") or {}, key, pct=is_pct),
                _metric_cell(regimes.get("bear") or {}, key, pct=is_pct),
                _metric_cell(regimes.get("unknown") or {}, key, pct=is_pct),
            ]
        )
    y = _book_table(
        fig,
        x=0.12,
        y=0.835,
        width=0.76,
        headers=["Chỉ tiêu", "Bull", "Bear", "Không rõ"],
        rows=rows,
        col_fracs=[0.42, 0.18, 0.18, 0.22],
        wrap_chars=[30, 12, 12, 14],
        fontsize=8.2,
        line_height=0.0165,
        min_row_height=0.034,
    )
    y -= 0.01
    fig.text(0.12, y, "Phương pháp", fontsize=10.5, weight="bold", family="DejaVu Serif", color=BOOK_INK)
    y -= 0.026
    if regime_meta.get("enabled"):
        text = (
            f"Dùng {regime_meta.get('index_symbol')} làm chỉ số bối cảnh. Một mẫu được xếp Bull nếu điểm chỉ số tại ngày bắt đầu mẫu "
            "cao hơn điểm chỉ số trước đó 18 tháng; ngược lại xếp Bear. Đây là lớp bối cảnh nghiên cứu, không phải tín hiệu giao dịch độc lập."
        )
    else:
        text = "Không đủ dữ liệu chỉ số để phân nhóm bối cảnh thị trường; các mẫu được xếp không rõ."
    _add_wrapped_text(fig, 0.14, y, text, width=92, fontsize=8.6, line_height=0.0185, color=BOOK_INK)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_quality_page(pdf: PdfPages, payload: Mapping[str, Any]) -> None:
    stats = payload["statistics"]
    quality = stats.get("quality_table") or {}
    counts = stats.get("quality_tier_counts") or {}
    fig = _book_page(running_title="Đáy mở rộng", page_no=4)
    fig.text(0.5, 0.90, "CHẤT LƯỢNG HÌNH THÁI", fontsize=13, weight="bold", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    fig.text(0.5, 0.875, "So sánh toàn bộ mẫu với nhóm hình thái sạch", fontsize=9.5, style="italic", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    rows = []
    for label, tier in [("Sạch", "clean"), ("Dùng được", "usable"), ("Lỏng", "loose")]:
        item = quality.get(tier) or {}
        rows.append(
            [
                label,
                str(counts.get(tier, item.get("detection_count", 0))),
                _metric_cell(item, "median_mfe_pct", pct=True),
                _metric_cell(item, "median_mae_pct", pct=True),
                _metric_cell(item, "target_hit_rate", pct=True),
                _metric_cell(item, "failure_5pct_rate", pct=True),
            ]
        )
    y = _book_table(
        fig,
        x=0.12,
        y=0.835,
        width=0.76,
        headers=["Nhóm", "N", "Biên thuận lợi", "Biên bất lợi", "Đạt mục tiêu", "Fail 5%"],
        rows=rows,
        col_fracs=[0.18, 0.12, 0.18, 0.18, 0.18, 0.16],
        wrap_chars=[14, 8, 12, 12, 12, 10],
        fontsize=8.0,
        line_height=0.016,
        min_row_height=0.034,
    )
    fig.text(0.12, y, "Cách đọc", fontsize=10.5, weight="bold", family="DejaVu Serif", color=BOOK_INK)
    y -= 0.026
    _add_wrapped_text(
        fig,
        0.14,
        y,
        "Điểm chất lượng không thay thế quy tắc nguồn. Nó là lớp nghiên cứu để kiểm tra liệu kết quả yếu đến từ mẫu hình lỏng hay từ chính hành vi hậu phá vỡ của thị trường Việt Nam. "
        "Nhóm sạch yêu cầu xu hướng trước mẫu đủ mạnh, số lần chạm tốt hơn mức tối thiểu, độ rộng/chiều cao nằm trong dải ưu tiên, chuỗi điểm xoay rõ và phá vỡ không quá sát biên.",
        width=92,
        fontsize=8.4,
        line_height=0.018,
        color=BOOK_INK,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_release_gate_page(pdf: PdfPages, payload: Mapping[str, Any]) -> None:
    gate = payload.get("release_gate_status") or {}
    stats_status = payload.get("statistics_contract_status") or {}
    framework = payload.get("chapter_framework_status") or {}
    fig = _book_page(running_title="Đáy mở rộng", page_no=5)
    fig.text(0.5, 0.90, "RELEASE GATE P1-P5", fontsize=13, weight="bold", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    fig.text(0.5, 0.875, "Trạng thái theo chuẩn 85-90% Bulkowski cho Việt Nam", fontsize=9.5, style="italic", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)

    summary_rows = [
        ("Phân loại hiện tại", str(gate.get("classification") or "không có")),
        ("Trạng thái công bố", str(gate.get("publish_status") or "không có")),
        ("Điểm chapter", str(gate.get("chapter_score") or "không có")),
        ("P0 hoàn chỉnh", "có" if stats_status.get("p0_complete") else "không"),
        ("Độ phủ hình bắt buộc", str(framework.get("current_required_figures_coverage") or "chưa triển khai")),
    ]
    y = _book_table(
        fig,
        x=0.16,
        y=0.83,
        width=0.68,
        headers=["Mục", "Trạng thái"],
        rows=summary_rows,
        col_fracs=[0.42, 0.58],
        wrap_chars=[24, 44],
        fontsize=8.7,
        line_height=0.017,
        min_row_height=0.031,
    )

    high_failures = list(gate.get("high_severity_failures") or [])
    p0_missing = list(gate.get("p0_missing") or [])
    y -= 0.004
    fig.text(0.16, y, "Các chặn chính", fontsize=10.5, weight="bold", family="DejaVu Serif", color=BOOK_INK)
    y -= 0.026
    if high_failures:
        y = _add_wrapped_text(
            fig,
            0.18,
            y,
            "High/Critical chưa đạt: " + ", ".join(str(x) for x in high_failures[:8]) + ("..." if len(high_failures) > 8 else ""),
            width=86,
            fontsize=8.2,
            line_height=0.018,
            color=BOOK_INK,
        )
    else:
        y = _add_wrapped_text(fig, 0.18, y, "Không có High/Critical failure.", width=86, fontsize=8.2, line_height=0.018, color=BOOK_INK)
    if p0_missing:
        y -= 0.01
        y = _add_wrapped_text(
            fig,
            0.18,
            y,
            "P0 còn thiếu: " + ", ".join(str(x) for x in p0_missing[:10]) + ("..." if len(p0_missing) > 10 else ""),
            width=86,
            fontsize=8.2,
            line_height=0.018,
            color=BOOK_INK,
        )

    y -= 0.018
    fig.text(0.16, y, "Kết luận vận hành", fontsize=10.5, weight="bold", family="DejaVu Serif", color=BOOK_INK)
    y -= 0.026
    _add_wrapped_text(
        fig,
        0.18,
        y,
        str(gate.get("allowed_claim") or "Chưa đủ dữ kiện release gate."),
        width=86,
        fontsize=8.5,
        line_height=0.018,
        color=BOOK_INK,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_investment_reference_page(pdf: PdfPages, payload: Mapping[str, Any]) -> None:
    status = payload.get("investment_reference_status") or {}
    gate = payload.get("release_gate_status") or {}
    fig = _book_page(running_title="Đáy mở rộng", page_no=6)
    fig.text(0.5, 0.90, "ĐỘ TIN CẬY THAM KHẢO ĐẦU TƯ", fontsize=13, weight="bold", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    fig.text(0.5, 0.875, "Không phải khuyến nghị giao dịch", fontsize=9.5, style="italic", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    rows = [
        ("Trạng thái", str(gate.get("classification") or status.get("level") or "không rõ")),
        ("Release gate", str(gate.get("publish_status") or "không có")),
        ("Mức phù hợp KPI", str(status.get("bulkowski_for_vietnam_score_estimate") or "chưa đánh giá")),
        ("Có dùng như thiết lập giao dịch?", "Không"),
        ("Đã có", "; ".join(str(x) for x in status.get("requirements_met", [])[:6])),
        ("Còn thiếu", "; ".join(str(x) for x in status.get("remaining_limits", [])[:4])),
    ]
    y = _book_table(
        fig,
        x=0.12,
        y=0.835,
        width=0.76,
        headers=["Mục", "Đánh giá"],
        rows=rows,
        col_fracs=[0.28, 0.72],
        wrap_chars=[20, 78],
        fontsize=8.2,
        line_height=0.0168,
        min_row_height=0.04,
    )
    _add_wrapped_text(
        fig,
        0.12,
        max(0.10, y),
        "Cách hiểu: chương này là bản nghiên cứu/draft có truy vết rule và số liệu ban đầu. "
        "Để gọi là tài liệu tham khảo đầu tư theo chuẩn P1-P5 cần pass release gate, có dữ liệu point-in-time, path hậu phá vỡ, uncertainty và artifact tái lập.",
        width=100,
        fontsize=8.4,
        line_height=0.018,
        color=BOOK_MUTED,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_rules_page(pdf: PdfPages, payload: Mapping[str, Any]) -> None:
    fig = _book_page(running_title="Đáy mở rộng", page_no=7)
    fig.text(0.5, 0.90, "Bảng 1.1", fontsize=10, weight="bold", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    fig.text(0.5, 0.878, "Đặc điểm nhận diện của đáy mở rộng", fontsize=11, ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)

    identification_rows = [
        ("Xu hướng giá", "Xu hướng ngắn hạn trước khi hình thành mẫu phải đi xuống."),
        ("Hình dạng", "Dạng hình loa: các đỉnh phụ cao dần và các đáy phụ thấp dần."),
        ("Đường xu hướng", "Đường nối các đỉnh dốc lên; đường nối các đáy dốc xuống. Hai đường biên phải phân kỳ."),
        ("Số lần chạm", "Cần ít nhất hai đỉnh phụ và hai đáy phụ. Mẫu không bắt buộc phải luân phiên hoàn hảo."),
        ("Khối lượng", "Khối lượng là bối cảnh: xu hướng thường tăng hoặc có dạng chữ U, nhưng chưa là cổng loại bắt buộc trong lượt này."),
        ("Phá vỡ", "Chỉ công nhận khi giá đóng cửa vượt lên trên đỉnh mẫu hình hoặc xuống dưới đáy mẫu hình."),
        ("Đo mục tiêu", "Chiều cao mẫu hình được cộng vào đỉnh phụ gần nhất cho phá vỡ lên hoặc trừ khỏi đáy phụ gần nhất cho phá vỡ xuống."),
    ]
    y = _book_table(
        fig,
        x=0.12,
        y=0.84,
        width=0.76,
        headers=["Đặc điểm", "Diễn giải"],
        rows=identification_rows,
        col_fracs=[0.23, 0.77],
        wrap_chars=[19, 70],
        fontsize=8.6,
        line_height=0.017,
        min_row_height=0.038,
    )

    fig.text(0.12, max(0.2, y), "Ghi chú", fontsize=10.8, weight="bold", ha="left", va="top", family="DejaVu Serif", color=BOOK_INK)
    _add_wrapped_text(
        fig,
        0.14,
        max(0.165, y - 0.03),
        "Bảng này là lớp diễn giải đọc được. Trang tiếp theo giữ bản truy vết ở cấp quy tắc để kiểm toán nguồn.",
        width=92,
        fontsize=8.4,
        line_height=0.018,
        color=BOOK_MUTED,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_provenance_page(pdf: PdfPages, payload: Mapping[str, Any]) -> None:
    fig = _book_page(running_title="Đáy mở rộng", page_no=8)
    rules = payload["rules"]
    fig.text(0.5, 0.90, "TRUY VẾT NGUỒN", fontsize=13, weight="bold", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    fig.text(0.5, 0.875, "Từ đoạn sách đến quy tắc trong bộ quét", fontsize=9.5, style="italic", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    _fig_line(fig, 0.12, 0.845, 0.88, 0.845, lw=0.8)
    y = 0.81
    provenance_rows = []
    interpretation_vi = {
        "bb.prior_trend.down": "Yêu cầu xu hướng giảm trước mẫu hình.",
        "bb.shape.megaphone": "Yêu cầu biên dao động mở rộng: đỉnh sau cao hơn và đáy sau thấp hơn.",
        "bb.trendlines.diverge": "Dựng đường biên trên dốc lên và đường biên dưới dốc xuống.",
        "bb.touches.min_two_each": "Yêu cầu tối thiểu hai lần chạm ở phía đỉnh và hai lần chạm ở phía đáy.",
        "bb.volume.context": "Ghi nhận khối lượng như bối cảnh, chưa dùng làm điều kiện loại bắt buộc.",
        "bb.breakout.close_either_side": "Xác nhận phá vỡ bằng giá đóng cửa vượt ra ngoài biên mẫu hình.",
        "bb.measure.height_from_recent_extreme": "Đo mục tiêu bằng chiều cao mẫu hình từ đỉnh/đáy phụ gần nhất.",
        "bb.invalidation.not_broadening": "Loại mẫu nếu không còn dạng mở rộng.",
    }
    rule_label_vi = {
        "bb.prior_trend.down": "xu_hướng_giảm",
        "bb.shape.megaphone": "hình_loa",
        "bb.trendlines.diverge": "biên_phân_kỳ",
        "bb.touches.min_two_each": "đủ_lần_chạm",
        "bb.volume.context": "khối_lượng",
        "bb.breakout.close_either_side": "phá_vỡ",
        "bb.measure.height_from_recent_extreme": "đo_mục_tiêu",
        "bb.invalidation.not_broadening": "loại_mẫu",
    }
    for rule in rules:
        rid = str(rule["rule_id"])
        interpreted = interpretation_vi.get(rid, str(rule.get("interpreted_rule") or ""))
        provenance_rows.append([rule_label_vi.get(rid, rid), str(rule["source_page"]), interpreted])
    y = _book_table(
        fig,
        x=0.12,
        y=y,
        width=0.76,
        headers=["Mã quy tắc", "Trang", "Cách đưa vào bộ quét"],
        rows=provenance_rows,
        col_fracs=[0.28, 0.1, 0.62],
        wrap_chars=[24, 7, 56],
        fontsize=8.0,
        line_height=0.016,
        min_row_height=0.038,
    )
    _add_wrapped_text(
        fig,
        0.12,
        max(0.075, y),
        "Ghi chú: sổ quy tắc vẫn giữ đoạn trích gốc và bảng đối chiếu tự động với PDF nguồn. Nếu thiếu truy vết nguồn, quy tắc không được bật vào bộ quét chính thức.",
        width=100,
        fontsize=7.8,
        line_height=0.017,
        color=BOOK_MUTED,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _split_commentary_sections(markdown: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []
    for line in markdown.splitlines():
        if line.startswith("### "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line.replace("### ", "").strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def _render_commentary_page(pdf: PdfPages, payload: Mapping[str, Any], commentary_md: str) -> None:
    fig = _book_page(running_title="Đáy mở rộng", page_no=9)
    fig.text(0.5, 0.90, "NHẬN XÉT BIÊN TẬP", fontsize=13, weight="bold", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    fig.text(0.5, 0.876, "Diễn giải từ gói dữ kiện đã khóa", fontsize=9.5, style="italic", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    _fig_line(fig, 0.12, 0.845, 0.88, 0.845, lw=0.8)
    y = 0.815
    for title, body in _split_commentary_sections(commentary_md):
        fig.text(0.12, y, title, fontsize=10.7, weight="bold", color=BOOK_INK, ha="left", va="top", family="DejaVu Serif")
        y -= 0.026
        y = _add_wrapped_text(fig, 0.14, y, body, width=92, fontsize=8.8, line_height=0.019, color=BOOK_INK)
        y -= 0.024
        if y < 0.12:
            break
    fig.text(0.12, 0.07, "DeepSeek chỉ viết lớp diễn giải; các bảng số liệu phía trước được sinh từ dữ kiện đã khóa.", fontsize=7.5, family="DejaVu Sans", color=BOOK_MUTED)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_example(pdf: PdfPages, source_dir: Path, detection: Mapping[str, Any], *, figure_no: int) -> None:
    path_matches = sorted(source_dir.glob(f"{detection['symbol']}*.json"))
    if not path_matches:
        return
    df = _load_market_stats_symbol(path_matches[0])
    df = df.reset_index(drop=True)
    start = max(0, int(detection["formation_start_idx"]) - 30)
    end = min(len(df), int(detection["breakout_idx"]) + 80)
    sub = df.iloc[start:end].copy()
    fig = _book_page(running_title="Đáy mở rộng", page_no=9 + figure_no)
    direction = "phá vỡ lên" if detection.get("breakout_direction") == "up" else "phá vỡ xuống"
    fig.text(0.5, 0.90, f"VÍ DỤ THỊ TRƯỜNG VIỆT NAM: {detection['symbol']}", fontsize=13, weight="bold", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    fig.text(0.5, 0.876, direction, fontsize=9.5, style="italic", ha="center", va="top", family="DejaVu Serif", color=BOOK_INK)
    ax = fig.add_axes([0.12, 0.26, 0.76, 0.50])
    vol_ax = fig.add_axes([0.12, 0.16, 0.76, 0.08], sharex=ax)
    ax.set_facecolor(BOOK_BG)
    vol_ax.set_facecolor(BOOK_BG)
    ax.plot(sub["date"], sub["close"], color=BOOK_INK, linewidth=1.15, label="Giá đóng cửa")
    f0 = df.iloc[int(detection["formation_start_idx"])]["date"]
    f1 = df.iloc[int(detection["formation_end_idx"])]["date"]
    ax.axvspan(f0, f1, color="#d6d3d1", alpha=0.35, label="Vùng hình thành")
    high_points: List[Tuple[Any, float]] = []
    low_points: List[Tuple[Any, float]] = []
    for p in detection.get("pivots", []):
        idx = int(p["idx"])
        if start <= idx < end:
            date = df.iloc[idx]["date"]
            price = float(p["price"])
            if p["type"] == "H":
                high_points.append((date, price))
                ax.scatter(date, price, s=30, color=BOOK_INK, zorder=3)
            else:
                low_points.append((date, price))
                ax.scatter(date, price, s=30, facecolor=BOOK_BG, edgecolor=BOOK_INK, linewidth=0.9, zorder=3)
    if len(high_points) >= 2:
        ax.plot([high_points[0][0], high_points[-1][0]], [high_points[0][1], high_points[-1][1]], color=BOOK_INK, linewidth=0.8)
    if len(low_points) >= 2:
        ax.plot([low_points[0][0], low_points[-1][0]], [low_points[0][1], low_points[-1][1]], color=BOOK_INK, linewidth=0.8)
    bidx = int(detection["breakout_idx"])
    breakout_date = df.iloc[bidx]["date"]
    ax.axvline(breakout_date, color=BOOK_INK, linestyle="--", linewidth=0.9)
    ax.annotate("Phá vỡ", xy=(breakout_date, float(detection["breakout_price"])), xytext=(18, 28), textcoords="offset points", fontsize=8, family="DejaVu Sans", arrowprops={"arrowstyle": "-", "lw": 0.6, "color": BOOK_INK})
    target_price = detection.get("target_price")
    if target_price is not None:
        ax.axhline(float(target_price), color=BOOK_INK, linestyle=":", linewidth=0.85)
        ax.annotate(
            "Mục tiêu",
            xy=(sub.iloc[-1]["date"], float(target_price)),
            xytext=(-54, 8),
            textcoords="offset points",
            fontsize=8,
            family="DejaVu Sans",
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": BOOK_INK},
        )
    vol_ax.bar(sub["date"], sub["volume"] / 1_000_000, color="#2f2f2f", width=1.8, alpha=0.8)
    vol_ax.set_ylabel("Triệu CP", fontsize=7, family="DejaVu Sans")
    vol_ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    vol_ax.tick_params(axis="y", labelsize=6)
    vol_ax.grid(False)
    for spine in ["top", "right"]:
        vol_ax.spines[spine].set_visible(False)
    ax.set_ylabel("Giá")
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.18, linewidth=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    vol_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(vol_ax.get_xticklabels(), rotation=0, ha="center", fontsize=7)
    caption = (
        f"Hình 1.{figure_no}  {detection['symbol']} tạo mẫu đáy mở rộng từ {detection['formation_start_date']} "
        f"đến {detection['formation_end_date']}; phá vỡ vào {detection['breakout_date']}. "
        "Các điểm đặc đánh dấu đỉnh phụ; các điểm rỗng đánh dấu đáy phụ; đường chấm ngang là mục tiêu giá theo Bảng 1.8."
    )
    _add_wrapped_text(fig, 0.12, 0.105, caption, width=100, fontsize=8.7, line_height=0.018, color=BOOK_INK)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_pdf(final_md: str, payload: Mapping[str, Any], *, source_dir: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    commentary_path = pdf_path.parent / "chapter_commentary.md"
    commentary_md = commentary_path.read_text(encoding="utf-8") if commentary_path.exists() else ""
    with PdfPages(pdf_path) as pdf:
        _render_snapshot_page(pdf, payload)
        _render_statistics_page(pdf, payload)
        _render_regime_page(pdf, payload)
        _render_quality_page(pdf, payload)
        _render_release_gate_page(pdf, payload)
        _render_investment_reference_page(pdf, payload)
        _render_rules_page(pdf, payload)
        _render_provenance_page(pdf, payload)
        if commentary_md:
            _render_commentary_page(pdf, payload, commentary_md)
        for i, detection in enumerate(payload.get("example_detections", [])[:4], start=1):
            _plot_example(pdf, source_dir, detection, figure_no=i)


def run_pipeline(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    limit_symbols: Optional[int] = None,
    skip_ai: bool = False,
    example_universe: str = "VN100",
    index_db: Path = DEFAULT_INDEX_DB,
    index_symbol: str = DEFAULT_INDEX_SYMBOL,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scan = scan_market_stats(source_dir, limit_symbols=limit_symbols, index_db=index_db, index_symbol=index_symbol)
    event_artifacts = write_event_artifacts(scan, source_dir=source_dir, out_dir=out_dir)
    stats = summarize_statistics(scan)
    payload = build_payload(scan, stats, example_universe=example_universe, event_artifacts=event_artifacts)
    core_md = render_core_markdown(payload)
    if skip_ai:
        commentary_md = "### Nhận xét chính\nAI commentary skipped.\n\n### So với mốc tham chiếu\nAI commentary skipped.\n\n### Lưu ý về chất lượng mẫu\nAI commentary skipped.\n\n### Hàm ý sử dụng\nAI commentary skipped.\n"
        commentary_meta = {"status": "skipped_by_flag", "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")}
    else:
        commentary_md, commentary_meta = generate_commentary(payload, core_md)
    final_md = core_md + "\n\n## DeepSeek Commentary\n\n" + commentary_md.strip() + "\n"

    paths = {
        "detections": out_dir / "detections.json",
        "statistics": out_dir / "statistics.json",
        "payload": out_dir / "chapter_payload.json",
        "core": out_dir / "chapter_core.md",
        "commentary": out_dir / "chapter_commentary.md",
        "commentary_meta": out_dir / "chapter_commentary.json",
        "release_gate": out_dir / "release_gate_status.json",
        "events_json": out_dir / "events.json",
        "events_csv": out_dir / "events.csv",
        "post_breakout_path_json": out_dir / "post_breakout_path.json",
        "post_breakout_path_csv": out_dir / "post_breakout_path.csv",
        "final": out_dir / "chapter_final.md",
        "pdf": out_dir / "broadening_bottoms.pdf",
        "render_meta": out_dir / "render_meta.json",
    }
    _write_json(paths["detections"], scan)
    _write_json(paths["statistics"], stats)
    _write_json(paths["payload"], payload)
    _write_json(paths["release_gate"], payload.get("release_gate_status") or {})
    _write_text(paths["core"], core_md)
    _write_text(paths["commentary"], commentary_md)
    _write_json(paths["commentary_meta"], commentary_meta)
    _write_text(paths["final"], final_md)
    render_pdf(final_md, payload, source_dir=source_dir, pdf_path=paths["pdf"])
    _write_json(
        paths["render_meta"],
        {
            "generated_at": _utc_now(),
            "pdf": str(paths["pdf"]),
            "payload_hash": canonical_spec_hash(payload),
            "spec_hash": payload["scanner_contract"]["spec_hash"],
            "deepseek": commentary_meta,
            "source_dir": str(source_dir),
            "layout": "bulkowski_inspired_v4_sourced_measure_rule",
            "classification": payload.get("release_gate_status", {}).get("classification"),
            "publish_status": payload.get("release_gate_status", {}).get("publish_status"),
            "example_universe": example_universe,
            "target_rule_id": "bb.measure.height_from_recent_extreme",
            "index_db": str(index_db),
            "index_symbol": index_symbol,
        },
    )
    return paths
