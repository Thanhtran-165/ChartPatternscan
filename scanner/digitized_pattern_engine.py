"""
Digitized Pattern Engine
-----------------------
Loads pattern definitions from `extraction_phase_1/digitization/patterns_digitized/*_digitized.json`
and provides a set of scanners that can cover all digitized patterns.

Notes:
- The digitized specs directory is intentionally gitignored in the public repo because it may
  be derived from copyrighted sources. This module will gracefully degrade if specs are missing.
- Detection uses NO look-ahead beyond breakout confirmation. Post-breakout evaluation is handled
  separately in `post_breakout_analyzer.py`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    # Package import (preferred)
    from .pivot_detector import Pivot, PivotDetector, PivotType
except ImportError:  # pragma: no cover - support running as a script from scanner/
    from pivot_detector import Pivot, PivotDetector, PivotType


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _pct_diff(a: float, b: float) -> float:
    denom = min(abs(a), abs(b))
    if denom <= 0:
        return float("inf")
    return abs(a - b) / denom * 100.0


def _slope_degrees(idx1: int, price1: float, idx2: int, price2: float) -> float:
    """
    Approximate slope in degrees using % change per bar.
    This isn't a charting angle; it's a stable numeric proxy that matches digitized thresholds reasonably.
    """
    bars = max(1, int(idx2) - int(idx1))
    if price1 == 0:
        return 0.0
    change_pct = (price2 - price1) / price1 * 100.0
    return float(np.degrees(np.arctan(change_pct / bars)))


@dataclass
class Trendline:
    idx0: int
    price0: float
    slope_per_bar: float

    def value_at(self, idx: int) -> float:
        return self.price0 + self.slope_per_bar * (idx - self.idx0)


class DigitizedPatternLibrary:
    def __init__(self, patterns_dir: Optional[str] = None):
        if patterns_dir is None:
            patterns_dir = os.path.join(
                os.path.dirname(__file__),
                "..",
                "extraction_phase_1",
                "digitization",
                "patterns_digitized",
            )
        self.patterns_dir = os.path.abspath(patterns_dir)

    def list_keys(self) -> List[str]:
        if not os.path.isdir(self.patterns_dir):
            return []
        keys: List[str] = []
        for name in os.listdir(self.patterns_dir):
            if not name.endswith("_digitized.json"):
                continue
            keys.append(name.replace("_digitized.json", ""))
        keys.sort()
        return keys

    def load(self, key: str) -> Dict[str, Any]:
        path = os.path.join(self.patterns_dir, f"{key}_digitized.json")
        with open(path, "r") as f:
            return json.load(f)


class BaseDigitizedScanner:
    def __init__(self, key: str, spec: Dict[str, Any]):
        self.key = key
        self.spec = spec
        self.pattern_type = str(spec.get("pattern_type") or "unknown")

        # Stable-ish hash for result traceability
        payload = json.dumps(
            {
                "key": self.key,
                "digitization_version": spec.get("digitization_version"),
                "pattern_type": self.pattern_type,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        self.config_hash = hashlib.md5(payload).hexdigest()[:8]

    def scan(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        pivots_filtered: List[Pivot],
        pivots_raw: List[Pivot],
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class PivotSequenceScanner(BaseDigitizedScanner):
    """
    Generic pivot-sequence scanner:
    - Match required H/L sequence on pivots
    - Apply a pragmatic subset of digitized constraints
    - Find breakout based on computed boundaries
    """

    def __init__(self, key: str, spec: Dict[str, Any]):
        super().__init__(key, spec)
        self.ds = spec.get("detection_signature", {}) or {}
        self.geom = spec.get("geometry_constraints", {}) or {}
        self.prior = spec.get("prior_trend_requirements", {}) or {}
        self.breakout = spec.get("breakout_confirmation", {}) or {}

        seq = self.ds.get("pivot_sequence", []) or []
        self.pivot_tokens = [t for t in seq if t in ("H", "L")]
        self.pivot_order = str(self.ds.get("pivot_order") or "alternating")

    def _pick_pivots(self, pivots_filtered: List[Pivot], pivots_raw: List[Pivot]) -> List[Pivot]:
        if self.pivot_order in ("alternating", "sequential_trend_following", "alternating_required"):
            return pivots_filtered
        return pivots_raw or pivots_filtered

    def _validate_width_height(self, window: List[Pivot]) -> Tuple[bool, float, int]:
        start_idx = window[0].idx
        end_idx = window[-1].idx
        width = end_idx - start_idx

        wmin = self.geom.get("width_min_bars")
        wmax = self.geom.get("width_max_bars")
        if wmin is not None and width < int(wmin):
            return False, 0.0, width
        if wmax is not None and width > int(wmax):
            return False, 0.0, width

        highs = [p.price for p in window if p.type == PivotType.HIGH]
        lows = [p.price for p in window if p.type == PivotType.LOW]
        if not highs or not lows:
            return False, 0.0, width

        upper = max(highs)
        lower = min(lows)
        mid = (upper + lower) / 2.0 if (upper + lower) != 0 else max(upper, 1e-9)
        height_pct = (upper - lower) / mid * 100.0

        hmin = self.geom.get("height_ratio_min")
        hmax = self.geom.get("height_ratio_max")
        if hmin is not None and height_pct < float(hmin):
            return False, height_pct, width
        if hmax is not None and height_pct > float(hmax):
            return False, height_pct, width

        return True, height_pct, width

    def _check_prior_trend(self, df: pd.DataFrame, pattern_start_idx: int) -> Tuple[bool, Optional[str]]:
        direction = str(self.prior.get("direction") or "any").lower()
        min_bars = int(self.prior.get("min_period_bars") or 0)
        min_change = float(self.prior.get("min_change_pct") or 0.0)
        if min_bars <= 0 or min_change <= 0:
            return True, None
        if pattern_start_idx < min_bars:
            return False, None

        start = pattern_start_idx - min_bars
        end = pattern_start_idx
        p0 = _safe_float(df.iloc[start].get("close"))
        p1 = _safe_float(df.iloc[end].get("close"))
        if p0 is None or p1 is None or p0 <= 0:
            return False, None
        change_pct = (p1 - p0) / p0 * 100.0

        if direction == "up":
            return (change_pct >= min_change), "up"
        if direction == "down":
            return (change_pct <= -min_change), "down"
        # any: require at least some move for context
        return (abs(change_pct) >= min_change), ("up" if change_pct >= 0 else "down")

    def _near_equal(self, a: float, b: float, tol_pct: float) -> bool:
        return _pct_diff(a, b) <= tol_pct

    def _constraint_ok(self, constraint: str, pos: int, window: List[Pivot]) -> bool:
        # pos is 1-based
        cur = window[pos - 1]
        c = constraint

        tol = float(self.geom.get("near_equal_tolerance_pct") or 3.0)

        def _prev_same_type() -> Optional[Pivot]:
            for j in range(pos - 2, -1, -1):
                if window[j].type == cur.type:
                    return window[j]
            return None

        if c == "near_equal":
            ref = _prev_same_type() or window[0]
            return self._near_equal(cur.price, ref.price, tol)

        if c.startswith("near_equal_to_position_"):
            try:
                ref_pos = int(c.split("_")[-1])
            except Exception:
                return True
            if 1 <= ref_pos <= len(window):
                return self._near_equal(cur.price, window[ref_pos - 1].price, tol)
            return True

        if c in ("lower_than_previous", "lower_than_previous_low", "lower_than_previous_L"):
            prev = _prev_same_type()
            return True if prev is None else (cur.price < prev.price)

        if c in ("higher_than_previous", "higher_than_previous_high", "higher_than_previous_H"):
            prev = _prev_same_type()
            return True if prev is None else (cur.price > prev.price)

        if c == "lower_than_previous_high":
            prev = None
            for j in range(pos - 2, -1, -1):
                if window[j].type == PivotType.HIGH:
                    prev = window[j]
                    break
            return True if prev is None else (cur.price < prev.price)

        if c == "higher_than_previous_low":
            prev = None
            for j in range(pos - 2, -1, -1):
                if window[j].type == PivotType.LOW:
                    prev = window[j]
                    break
            return True if prev is None else (cur.price > prev.price)

        if c == "must_be_lowest":
            lows = [p.price for p in window if p.type == PivotType.LOW]
            return True if not lows else (cur.price <= min(lows))

        if c == "must_be_highest":
            highs = [p.price for p in window if p.type == PivotType.HIGH]
            return True if not highs else (cur.price >= max(highs))

        if c == "not_lowest":
            lows = [p.price for p in window if p.type == PivotType.LOW]
            return True if not lows else (cur.price > min(lows))

        if c == "not_highest":
            highs = [p.price for p in window if p.type == PivotType.HIGH]
            return True if not highs else (cur.price < max(highs))

        if c == "must_be_between":
            # For lows: must be below adjacent highs, for highs: above adjacent lows
            if pos <= 1 or pos >= len(window):
                return True
            left = window[pos - 2]
            right = window[pos]
            if cur.type == PivotType.LOW:
                return cur.price < left.price and cur.price < right.price
            if cur.type == PivotType.HIGH:
                return cur.price > left.price and cur.price > right.price
            return True

        # Cup-with-handle specific pragmatic constraints
        if c == "in_upper_third":
            # Assume cup bottom at pos2 and lip at pos3 (works for digitized spec order)
            if len(window) >= 4:
                cup_bottom = window[1].price
                cup_lip = window[2].price
                return cur.price >= cup_bottom + (cup_lip - cup_bottom) * (2.0 / 3.0)
            return True

        if c == "near_cup_lip":
            if len(window) >= 3:
                cup_lip = window[2].price
                return self._near_equal(cur.price, cup_lip, tol)
            return True

        if c == "shallow_decline":
            # Require last low not too deep vs handle resistance (position 5)
            if len(window) >= 6:
                handle_res = window[4].price
                if handle_res == 0:
                    return True
                drawdown = (handle_res - cur.price) / handle_res * 100.0
                return drawdown <= float(self.geom.get("depth_ratio_tolerance_pct") or 15.0)
            return True

        # Default: accept unknown constraints (we still want broad coverage)
        return True

    def _validate_mandatory(self, window: List[Pivot]) -> bool:
        mps = (self.ds.get("mandatory_pivots") or []) if isinstance(self.ds.get("mandatory_pivots"), list) else []
        for mp in mps:
            pos = mp.get("position")
            if not isinstance(pos, int):
                continue
            if not (1 <= pos <= len(window)):
                continue
            c = mp.get("constraint")
            if c:
                if not self._constraint_ok(str(c), pos, window):
                    return False
        return True

    def _build_boundaries(self, window: List[Pivot]) -> Tuple[Trendline, Trendline]:
        highs = [p for p in window if p.type == PivotType.HIGH]
        lows = [p for p in window if p.type == PivotType.LOW]

        # Upper boundary
        if len(highs) >= 2:
            p0, p1 = highs[0], highs[-1]
            slope = (p1.price - p0.price) / max(1, p1.idx - p0.idx)
            upper = Trendline(idx0=p0.idx, price0=p0.price, slope_per_bar=slope)
        else:
            p0 = highs[0]
            upper = Trendline(idx0=p0.idx, price0=p0.price, slope_per_bar=0.0)

        # Lower boundary
        if len(lows) >= 2:
            p0, p1 = lows[0], lows[-1]
            slope = (p1.price - p0.price) / max(1, p1.idx - p0.idx)
            lower = Trendline(idx0=p0.idx, price0=p0.price, slope_per_bar=slope)
        else:
            p0 = lows[0]
            lower = Trendline(idx0=p0.idx, price0=p0.price, slope_per_bar=0.0)

        return upper, lower

    def _breakout_directions(self, prior_dir: Optional[str]) -> List[str]:
        bd = str(self.breakout.get("breakout_direction") or "").lower()
        ptype = self.pattern_type.lower()

        # Prefer explicit reversal type
        if ptype == "reversal_bearish":
            return ["down"]
        if ptype == "reversal_bullish":
            return ["up"]

        if bd in ("up", "down"):
            return [bd]
        if bd in ("same_as_flagpole", "depends_on_prior_trend"):
            if prior_dir in ("up", "down"):
                return [prior_dir]
            return ["up", "down"]
        if bd in ("neutral", "both", "continuation_both"):
            return ["up", "down"]
        return ["up", "down"]

    def _find_breakout(
        self,
        df: pd.DataFrame,
        *,
        formation_end_idx: int,
        upper: Trendline,
        lower: Trendline,
        prior_dir: Optional[str],
    ) -> Tuple[Optional[int], Optional[str], Optional[float], bool]:
        thr_pct = float(self.breakout.get("breakout_threshold_pct") or 1.0) / 100.0
        vol_min = float(self.breakout.get("volume_multiplier_min") or 1.3)
        vol_required = bool(self.breakout.get("volume_required") is True)

        # Keep search bounded for performance; longer patterns can override in the future.
        search_bars = int(self.geom.get("breakout_search_bars") or 40)
        end = min(len(df), formation_end_idx + 1 + search_bars)

        for idx in range(formation_end_idx + 1, end):
            close = _safe_float(df.iloc[idx].get("close"))
            if close is None or close <= 0:
                continue

            up_level = upper.value_at(idx)
            dn_level = lower.value_at(idx)

            for d in self._breakout_directions(prior_dir):
                if d == "up":
                    if close > up_level * (1.0 + thr_pct):
                        vr = df.iloc[idx].get("volume_ratio", np.nan)
                        vol_ok = bool(pd.notna(vr) and np.isfinite(vr) and float(vr) >= vol_min)
                        if vol_required and not vol_ok:
                            continue
                        return idx, "up", close, vol_ok
                else:
                    if close < dn_level * (1.0 - thr_pct):
                        vr = df.iloc[idx].get("volume_ratio", np.nan)
                        vol_ok = bool(pd.notna(vr) and np.isfinite(vr) and float(vr) >= vol_min)
                        if vol_required and not vol_ok:
                            continue
                        return idx, "down", close, vol_ok

        return None, None, None, False

    def scan(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        pivots_filtered: List[Pivot],
        pivots_raw: List[Pivot],
    ) -> List[Dict[str, Any]]:
        if not self.pivot_tokens:
            return []

        pivots = self._pick_pivots(pivots_filtered, pivots_raw)
        if len(pivots) < len(self.pivot_tokens):
            return []

        out: List[Dict[str, Any]] = []
        n = len(self.pivot_tokens)

        for i in range(len(pivots) - n + 1):
            window = pivots[i : i + n]

            # Type match (H/L)
            ok = True
            for tok, p in zip(self.pivot_tokens, window):
                if tok == "H" and p.type != PivotType.HIGH:
                    ok = False
                    break
                if tok == "L" and p.type != PivotType.LOW:
                    ok = False
                    break
            if not ok:
                continue

            # Width/height constraints
            ok, height_pct, width = self._validate_width_height(window)
            if not ok:
                continue

            # Mandatory constraints
            if not self._validate_mandatory(window):
                continue

            # Prior trend
            prior_ok, prior_dir = self._check_prior_trend(df, window[0].idx)
            if not prior_ok:
                continue

            upper, lower = self._build_boundaries(window)
            breakout_idx, breakout_dir, breakout_price, vol_ok = self._find_breakout(
                df,
                formation_end_idx=window[-1].idx,
                upper=upper,
                lower=lower,
                prior_dir=prior_dir,
            )

            # For some high-frequency patterns, requiring breakout keeps result size sane.
            if self.key in ("pipe_bottoms",) and breakout_idx is None:
                continue

            # Pattern height for target: use boundary range at pattern start
            height_abs = max(0.0, upper.value_at(window[0].idx) - lower.value_at(window[0].idx))
            if height_abs <= 0:
                continue

            target = None
            stop = None
            if breakout_idx is not None and breakout_dir is not None and breakout_price is not None:
                if breakout_dir == "up":
                    target = breakout_price + height_abs
                    stop = lower.value_at(breakout_idx)
                else:
                    target = breakout_price - height_abs
                    stop = upper.value_at(breakout_idx)

            # Confidence: pragmatic scoring (hard constraints already passed)
            confidence = 70
            if vol_ok:
                confidence += 10
            if breakout_idx is not None:
                confidence += 10
            confidence = int(min(100, confidence))

            start_idx = window[0].idx
            end_idx = window[-1].idx
            pattern_id = f"{symbol}_{self.key}_{start_idx}_{end_idx}"

            out.append(
                {
                    "pattern_id": pattern_id,
                    "symbol": symbol,
                    "pattern_name": self.key,
                    "pattern_type": self.pattern_type,
                    "formation_start": str(df.iloc[start_idx]["date"].date()) if "date" in df.columns else str(start_idx),
                    "formation_end": str(df.iloc[end_idx]["date"].date()) if "date" in df.columns else str(end_idx),
                    "breakout_date": str(df.iloc[breakout_idx]["date"].date()) if breakout_idx is not None and "date" in df.columns else None,
                    "breakout_idx": int(breakout_idx) if breakout_idx is not None else None,
                    "breakout_direction": breakout_dir,
                    "breakout_price": breakout_price,
                    "target_price": target,
                    "stop_loss_price": stop,
                    "confidence_score": confidence,
                    "volume_confirmed": bool(vol_ok),
                    "pattern_height_pct": round(height_pct, 2),
                    "pattern_width_bars": int(width),
                    "touch_count": int(len(window)),
                    "pivot_indices": [int(p.idx) for p in window],
                    "config_hash": self.config_hash,
                    "created_at": datetime.now().isoformat(),
                }
            )

        return out


class InsideDayScanner(BaseDigitizedScanner):
    def scan(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        pivots_filtered: List[Pivot],
        pivots_raw: List[Pivot],
    ) -> List[Dict[str, Any]]:
        # Keep result size sane: only emit inside-day when a breakout happens within max_return_days.
        bo = self.spec.get("breakout_confirmation", {}) or {}
        thr_pct = float(bo.get("breakout_threshold_pct") or 0.5) / 100.0
        max_return = int(bo.get("max_return_days") or 2)

        out: List[Dict[str, Any]] = []
        for i in range(1, len(df)):
            y = df.iloc[i - 1]
            t = df.iloc[i]
            if t["high"] < y["high"] and t["low"] > y["low"]:
                inside_high = float(t["high"])
                inside_low = float(t["low"])

                breakout_idx = None
                breakout_dir = None
                breakout_price = None
                for j in range(i + 1, min(len(df), i + 1 + max_return)):
                    close = _safe_float(df.iloc[j].get("close"))
                    if close is None:
                        continue
                    if close > inside_high * (1.0 + thr_pct):
                        breakout_idx = j
                        breakout_dir = "up"
                        breakout_price = close
                        break
                    if close < inside_low * (1.0 - thr_pct):
                        breakout_idx = j
                        breakout_dir = "down"
                        breakout_price = close
                        break

                if breakout_idx is None:
                    continue

                pattern_id = f"{symbol}_{self.key}_{i}_{i}"
                height_abs = inside_high - inside_low
                target = breakout_price + height_abs if breakout_dir == "up" else breakout_price - height_abs
                stop = inside_low if breakout_dir == "up" else inside_high

                out.append(
                    {
                        "pattern_id": pattern_id,
                        "symbol": symbol,
                        "pattern_name": self.key,
                        "pattern_type": self.pattern_type,
                        "formation_start": str(t["date"].date()) if "date" in df.columns else str(i),
                        "formation_end": str(t["date"].date()) if "date" in df.columns else str(i),
                        "breakout_date": str(df.iloc[breakout_idx]["date"].date()) if "date" in df.columns else None,
                        "breakout_idx": int(breakout_idx),
                        "breakout_direction": breakout_dir,
                        "breakout_price": breakout_price,
                        "target_price": target,
                        "stop_loss_price": stop,
                        "confidence_score": 70,
                        "volume_confirmed": False,
                        "pattern_height_pct": round(height_abs / max(1e-9, (inside_high + inside_low) / 2.0) * 100.0, 2),
                        "pattern_width_bars": 1,
                        "touch_count": 1,
                        "pivot_indices": [int(i - 1), int(i)],
                        "config_hash": self.config_hash,
                        "created_at": datetime.now().isoformat(),
                    }
                )

        return out


class GapScanner(BaseDigitizedScanner):
    def scan(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        pivots_filtered: List[Pivot],
        pivots_raw: List[Pivot],
    ) -> List[Dict[str, Any]]:
        geom = self.spec.get("geometry_constraints", {}) or {}
        min_gap = float(((geom.get("gap_constraints") or {}).get("min_gap_size_pct")) or 0.1) / 100.0

        out: List[Dict[str, Any]] = []
        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            cur = df.iloc[i]
            # Gap up: today's low above yesterday's high
            if cur["low"] > prev["high"] * (1.0 + min_gap):
                pattern_id = f"{symbol}_{self.key}_{i}_{i}"
                height_abs = float(cur["low"] - prev["high"])
                out.append(
                    {
                        "pattern_id": pattern_id,
                        "symbol": symbol,
                        "pattern_name": self.key,
                        "pattern_type": self.pattern_type,
                        "formation_start": str(cur["date"].date()) if "date" in df.columns else str(i),
                        "formation_end": str(cur["date"].date()) if "date" in df.columns else str(i),
                        "breakout_date": str(cur["date"].date()) if "date" in df.columns else None,
                        "breakout_idx": int(i),
                        "breakout_direction": "up",
                        "breakout_price": float(cur["close"]),
                        "target_price": float(cur["close"]) + height_abs,
                        "stop_loss_price": float(prev["high"]),
                        "confidence_score": 65,
                        "volume_confirmed": False,
                        "pattern_height_pct": round(height_abs / max(1e-9, float(prev["high"])) * 100.0, 3),
                        "pattern_width_bars": 1,
                        "touch_count": 1,
                        "pivot_indices": [int(i - 1), int(i)],
                        "config_hash": self.config_hash,
                        "created_at": datetime.now().isoformat(),
                    }
                )
                continue

            # Gap down: today's high below yesterday's low
            if cur["high"] < prev["low"] * (1.0 - min_gap):
                pattern_id = f"{symbol}_{self.key}_{i}_{i}"
                height_abs = float(prev["low"] - cur["high"])
                out.append(
                    {
                        "pattern_id": pattern_id,
                        "symbol": symbol,
                        "pattern_name": self.key,
                        "pattern_type": self.pattern_type,
                        "formation_start": str(cur["date"].date()) if "date" in df.columns else str(i),
                        "formation_end": str(cur["date"].date()) if "date" in df.columns else str(i),
                        "breakout_date": str(cur["date"].date()) if "date" in df.columns else None,
                        "breakout_idx": int(i),
                        "breakout_direction": "down",
                        "breakout_price": float(cur["close"]),
                        "target_price": float(cur["close"]) - height_abs,
                        "stop_loss_price": float(prev["low"]),
                        "confidence_score": 65,
                        "volume_confirmed": False,
                        "pattern_height_pct": round(height_abs / max(1e-9, float(prev["low"])) * 100.0, 3),
                        "pattern_width_bars": 1,
                        "touch_count": 1,
                        "pivot_indices": [int(i - 1), int(i)],
                        "config_hash": self.config_hash,
                        "created_at": datetime.now().isoformat(),
                    }
                )

        return out


class IslandScanner(BaseDigitizedScanner):
    def scan(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        pivots_filtered: List[Pivot],
        pivots_raw: List[Pivot],
    ) -> List[Dict[str, Any]]:
        # Simplified island detection: gap up then gap down (top) OR gap down then gap up (bottom).
        min_gap_pct = 0.001  # 0.1%
        max_island_bars = 20

        out: List[Dict[str, Any]] = []

        def is_gap_up(i: int) -> bool:
            if i <= 0:
                return False
            return df.iloc[i]["low"] > df.iloc[i - 1]["high"] * (1.0 + min_gap_pct)

        def is_gap_down(i: int) -> bool:
            if i <= 0:
                return False
            return df.iloc[i]["high"] < df.iloc[i - 1]["low"] * (1.0 - min_gap_pct)

        i = 1
        while i < len(df):
            if is_gap_up(i):
                # Look for gap down within window
                j_end = min(len(df), i + 1 + max_island_bars)
                j = i + 1
                while j < j_end:
                    if is_gap_down(j):
                        # Island top
                        pattern_id = f"{symbol}_{self.key}_{i}_{j}"
                        breakout_price = float(df.iloc[j]["close"])
                        island_high = float(df.iloc[i:j]["high"].max())
                        island_low = float(df.iloc[i:j]["low"].min())
                        height_abs = island_high - island_low
                        out.append(
                            {
                                "pattern_id": pattern_id,
                                "symbol": symbol,
                                "pattern_name": self.key,
                                "pattern_type": self.pattern_type,
                                "formation_start": str(df.iloc[i]["date"].date()) if "date" in df.columns else str(i),
                                "formation_end": str(df.iloc[j]["date"].date()) if "date" in df.columns else str(j),
                                "breakout_date": str(df.iloc[j]["date"].date()) if "date" in df.columns else None,
                                "breakout_idx": int(j),
                                "breakout_direction": "down",
                                "breakout_price": breakout_price,
                                "target_price": breakout_price - height_abs,
                                "stop_loss_price": island_high,
                                "confidence_score": 75,
                                "volume_confirmed": False,
                                "pattern_height_pct": round(height_abs / max(1e-9, island_high) * 100.0, 2),
                                "pattern_width_bars": int(j - i + 1),
                                "touch_count": 2,
                                "pivot_indices": [int(i), int(j)],
                                "config_hash": self.config_hash,
                                "created_at": datetime.now().isoformat(),
                            }
                        )
                        i = j  # skip ahead
                        break
                    j += 1
                i += 1
                continue

            if is_gap_down(i):
                # Look for gap up within window
                j_end = min(len(df), i + 1 + max_island_bars)
                j = i + 1
                while j < j_end:
                    if is_gap_up(j):
                        # Island bottom
                        pattern_id = f"{symbol}_{self.key}_{i}_{j}"
                        breakout_price = float(df.iloc[j]["close"])
                        island_high = float(df.iloc[i:j]["high"].max())
                        island_low = float(df.iloc[i:j]["low"].min())
                        height_abs = island_high - island_low
                        out.append(
                            {
                                "pattern_id": pattern_id,
                                "symbol": symbol,
                                "pattern_name": self.key,
                                "pattern_type": self.pattern_type,
                                "formation_start": str(df.iloc[i]["date"].date()) if "date" in df.columns else str(i),
                                "formation_end": str(df.iloc[j]["date"].date()) if "date" in df.columns else str(j),
                                "breakout_date": str(df.iloc[j]["date"].date()) if "date" in df.columns else None,
                                "breakout_idx": int(j),
                                "breakout_direction": "up",
                                "breakout_price": breakout_price,
                                "target_price": breakout_price + height_abs,
                                "stop_loss_price": island_low,
                                "confidence_score": 75,
                                "volume_confirmed": False,
                                "pattern_height_pct": round(height_abs / max(1e-9, island_low) * 100.0, 2),
                                "pattern_width_bars": int(j - i + 1),
                                "touch_count": 2,
                                "pivot_indices": [int(i), int(j)],
                                "config_hash": self.config_hash,
                                "created_at": datetime.now().isoformat(),
                            }
                        )
                        i = j
                        break
                    j += 1
                i += 1
                continue

            i += 1

        return out


class ThreeMethodsScanner(BaseDigitizedScanner):
    def scan(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        pivots_filtered: List[Pivot],
        pivots_raw: List[Pivot],
    ) -> List[Dict[str, Any]]:
        geom = self.spec.get("geometry_constraints", {}) or {}
        first_req = geom.get("first_bar_requirements", {}) or {}
        last_req = geom.get("last_bar_requirements", {}) or {}
        middle_req = geom.get("middle_bars_requirements", {}) or {}

        min_body_atr = float(first_req.get("min_body_size_atr") or 1.5)
        min_last_body_atr = float(last_req.get("min_body_size_atr") or 1.5)

        out: List[Dict[str, Any]] = []
        if "atr" not in df.columns:
            return out

        for i in range(4, len(df)):
            w = df.iloc[i - 4 : i + 1].copy()
            if len(w) != 5:
                continue

            atr = _safe_float(w.iloc[0].get("atr")) or _safe_float(w.iloc[-1].get("atr"))
            if atr is None or atr <= 0:
                continue

            o1, c1 = float(w.iloc[0]["open"]), float(w.iloc[0]["close"])
            o5, c5 = float(w.iloc[-1]["open"]), float(w.iloc[-1]["close"])
            body1 = abs(c1 - o1)
            body5 = abs(c5 - o5)

            # Middle 3 bars small and inside first bar's range
            first_high = float(w.iloc[0]["high"])
            first_low = float(w.iloc[0]["low"])
            inside_ok = True
            for k in range(1, 4):
                bk = w.iloc[k]
                if float(bk["high"]) > first_high or float(bk["low"]) < first_low:
                    inside_ok = False
                    break
                if abs(float(bk["close"]) - float(bk["open"])) > body1 * (float(middle_req.get("max_body_size_pct") or 50) / 100.0):
                    inside_ok = False
                    break
            if not inside_ok:
                continue

            # Rising methods: long white, 3 small, long white closing above first close
            is_bull = (c1 > o1) and (c5 > o5) and (body1 >= min_body_atr * atr) and (body5 >= min_last_body_atr * atr) and (c5 > c1)
            # Falling methods: long black, 3 small, long black closing below first close
            is_bear = (c1 < o1) and (c5 < o5) and (body1 >= min_body_atr * atr) and (body5 >= min_last_body_atr * atr) and (c5 < c1)

            if not (is_bull or is_bear):
                continue

            breakout_dir = "up" if is_bull else "down"
            breakout_idx = i
            breakout_price = float(df.iloc[breakout_idx]["close"])

            pattern_id = f"{symbol}_{self.key}_{i-4}_{i}"
            height_abs = first_high - first_low
            target = breakout_price + height_abs if breakout_dir == "up" else breakout_price - height_abs
            stop = first_low if breakout_dir == "up" else first_high

            out.append(
                {
                    "pattern_id": pattern_id,
                    "symbol": symbol,
                    "pattern_name": self.key,
                    "pattern_type": self.pattern_type,
                    "formation_start": str(df.iloc[i - 4]["date"].date()) if "date" in df.columns else str(i - 4),
                    "formation_end": str(df.iloc[i]["date"].date()) if "date" in df.columns else str(i),
                    "breakout_date": str(df.iloc[breakout_idx]["date"].date()) if "date" in df.columns else None,
                    "breakout_idx": int(breakout_idx),
                    "breakout_direction": breakout_dir,
                    "breakout_price": breakout_price,
                    "target_price": target,
                    "stop_loss_price": stop,
                    "confidence_score": 80,
                    "volume_confirmed": False,
                    "pattern_height_pct": round(height_abs / max(1e-9, breakout_price) * 100.0, 2),
                    "pattern_width_bars": 5,
                    "touch_count": 5,
                    "pivot_indices": [int(i - 4), int(i - 3), int(i - 2), int(i - 1), int(i)],
                    "config_hash": self.config_hash,
                    "created_at": datetime.now().isoformat(),
                }
            )

        return out


class TripleBottomsTopsScanner(BaseDigitizedScanner):
    """
    `triple_bottoms_tops` is the only digitized spec that nests 2 sub-specs under
    detection_signature.{triple_bottom,triple_top}. We support both sequences here.
    """

    def __init__(self, key: str, spec: Dict[str, Any]):
        super().__init__(key, spec)
        ds = spec.get("detection_signature", {}) or {}

        self._bottom: Optional[PivotSequenceScanner] = None
        self._top: Optional[PivotSequenceScanner] = None

        tb = ds.get("triple_bottom")
        if isinstance(tb, dict):
            bottom_spec = dict(spec)
            bottom_spec["detection_signature"] = tb
            # Keep top-level pattern_type for now (reversal_both); breakout_direction disambiguates.
            self._bottom = PivotSequenceScanner(key, bottom_spec)

        tt = ds.get("triple_top")
        if isinstance(tt, dict):
            top_spec = dict(spec)
            top_spec["detection_signature"] = tt
            self._top = PivotSequenceScanner(key, top_spec)

    def scan(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        pivots_filtered: List[Pivot],
        pivots_raw: List[Pivot],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if self._bottom is not None:
            out.extend(
                self._bottom.scan(
                    symbol=symbol,
                    df=df,
                    pivots_filtered=pivots_filtered,
                    pivots_raw=pivots_raw,
                )
            )
        if self._top is not None:
            out.extend(
                self._top.scan(
                    symbol=symbol,
                    df=df,
                    pivots_filtered=pivots_filtered,
                    pivots_raw=pivots_raw,
                )
            )
        return out


class PipeBottomScanner(BaseDigitizedScanner):
    """
    Pipe bottoms have pivot_sequence ["L","L"] so the generic pivot-sequence implementation
    (which expects both highs and lows) cannot compute height/boundaries. This scanner:
    - finds 2 near-equal lows separated by a few bars
    - validates "vertical-ish" legs into/out of each low (pragmatic)
    - confirms breakout above the interim high after the 2nd low
    """

    def __init__(self, key: str, spec: Dict[str, Any]):
        super().__init__(key, spec)
        self.ds = spec.get("detection_signature", {}) or {}
        self.geom = spec.get("geometry_constraints", {}) or {}
        self.prior = spec.get("prior_trend_requirements", {}) or {}
        self.bo = spec.get("breakout_confirmation", {}) or {}

        self.width_min = int(self.geom.get("width_min_bars") or 0)
        self.width_max = int(self.geom.get("width_max_bars") or 10_000)

        sep = self.geom.get("parallel_separation_bars", {}) or {}
        self.sep_min = int(sep.get("min") or 0)
        self.sep_max = int(sep.get("max") or self.width_max)

        sim = self.geom.get("pipe_bottom_similarity_pct", {}) or {}
        self.sim_min = float(sim.get("min") or 0.97)
        self.sim_max = float(sim.get("max") or 1.03)

        vdr = self.geom.get("vertical_drop_ratio", {}) or {}
        self.drop_min = float(vdr.get("min") or 0.0)
        self.drop_max = float(vdr.get("max") or 1e9)

        slope = self.geom.get("slope_constraints", {}) or {}
        self.angle_min = float(slope.get("pipe_vertical_angle_min_degrees") or 0.0)
        self.angle_max = float(slope.get("pipe_vertical_angle_max_degrees") or 90.0)

        self.height_min = float(self.geom.get("height_ratio_min") or 0.0)
        self.height_max = float(self.geom.get("height_ratio_max") or 1e9)

        self.breakout_thr = float(self.bo.get("breakout_threshold_pct") or 1.0) / 100.0
        self.confirm_bars = int(self.bo.get("confirmation_bars") or 1)
        self.close_beyond = bool(self.bo.get("close_beyond_required") if self.bo.get("close_beyond_required") is not None else True)
        self.vol_required = bool(self.bo.get("volume_required") or False)
        self.vol_mult_min = float(self.bo.get("volume_multiplier_min") or 1.3)

        # Pragmatic local windows for "vertical" leg checks
        self._leg_window_pre = 3
        self._leg_window_post = 3

    def _check_prior_trend(self, df: pd.DataFrame, pattern_start_idx: int) -> bool:
        direction = str(self.prior.get("direction") or "any").lower()
        min_bars = int(self.prior.get("min_period_bars") or 0)
        min_change = float(self.prior.get("min_change_pct") or 0.0)
        if min_bars <= 0 or min_change <= 0:
            return True
        if pattern_start_idx < min_bars:
            return False

        start = pattern_start_idx - min_bars
        end = pattern_start_idx
        p0 = _safe_float(df.iloc[start].get("close"))
        p1 = _safe_float(df.iloc[end].get("close"))
        if p0 is None or p1 is None or p0 <= 0:
            return False
        change_pct = (p1 - p0) / p0 * 100.0

        if direction == "down":
            return change_pct <= -min_change
        if direction == "up":
            return change_pct >= min_change
        return abs(change_pct) >= min_change

    def _leg_ok(self, df: pd.DataFrame, low_idx: int, low_price: float) -> bool:
        if low_idx <= 0 or low_idx >= len(df):
            return False

        pre0 = max(0, low_idx - self._leg_window_pre)
        pre = df.iloc[pre0:low_idx]
        if len(pre) == 0:
            return False
        pre_high = float(pre["high"].max())
        if pre_high <= 0:
            return False
        drop_pct = (pre_high - low_price) / pre_high * 100.0
        if not (self.drop_min <= drop_pct <= self.drop_max):
            return False

        # Angle proxy (steeper = closer to vertical)
        pre_high_idx = int(pre["high"].idxmax())
        angle = abs(_slope_degrees(pre_high_idx, pre_high, low_idx, low_price))
        if not (self.angle_min <= angle <= self.angle_max):
            return False

        post1 = min(len(df), low_idx + 1 + self._leg_window_post)
        post = df.iloc[low_idx + 1 : post1]
        if len(post) == 0:
            return False
        post_high = float(post["high"].max())
        if low_price <= 0:
            return False
        rise_pct = (post_high - low_price) / low_price * 100.0
        if rise_pct < self.drop_min:
            return False

        return True

    def _find_breakout(
        self,
        df: pd.DataFrame,
        *,
        start_idx: int,
        level: float,
        direction: str = "up",
        max_lookahead: int = 30,
    ) -> Tuple[Optional[int], Optional[float], bool]:
        thr = level * (1.0 + self.breakout_thr) if direction == "up" else level * (1.0 - self.breakout_thr)

        for i in range(start_idx, min(len(df), start_idx + max_lookahead)):
            close = _safe_float(df.iloc[i].get("close"))
            if close is None:
                continue
            ok = close > thr if direction == "up" else close < thr
            if not ok:
                continue

            # Confirmation bars: require consecutive closes beyond threshold.
            if self.confirm_bars > 1:
                j_end = min(len(df), i + self.confirm_bars)
                all_ok = True
                for j in range(i, j_end):
                    c = _safe_float(df.iloc[j].get("close"))
                    if c is None:
                        all_ok = False
                        break
                    if direction == "up" and not (c > thr):
                        all_ok = False
                        break
                    if direction == "down" and not (c < thr):
                        all_ok = False
                        break
                if not all_ok:
                    continue

            vr = df.iloc[i].get("volume_ratio", np.nan)
            vol_ok = bool(pd.notna(vr) and np.isfinite(vr) and float(vr) >= self.vol_mult_min)
            if self.vol_required and not vol_ok:
                continue
            return i, close, vol_ok

        return None, None, False

    def scan(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        pivots_filtered: List[Pivot],
        pivots_raw: List[Pivot],
    ) -> List[Dict[str, Any]]:
        pivots = pivots_raw or pivots_filtered
        lows = [p for p in pivots if p.type == PivotType.LOW]
        if len(lows) < 2:
            return []

        out: List[Dict[str, Any]] = []

        for i in range(len(lows) - 1):
            l1 = lows[i]
            for j in range(i + 1, len(lows)):
                l2 = lows[j]
                sep = int(l2.idx) - int(l1.idx)
                if sep < self.width_min:
                    continue
                if sep > self.width_max:
                    break
                if sep < self.sep_min or sep > self.sep_max:
                    continue

                if l1.price <= 0 or l2.price <= 0:
                    continue
                ratio = float(l2.price) / float(l1.price)
                if not (self.sim_min <= ratio <= self.sim_max):
                    continue

                if not self._check_prior_trend(df, int(l1.idx)):
                    continue

                if not self._leg_ok(df, int(l1.idx), float(l1.price)):
                    continue
                if not self._leg_ok(df, int(l2.idx), float(l2.price)):
                    continue

                # Interim high between the two lows acts as breakout level.
                interim = df.iloc[int(l1.idx) : int(l2.idx) + 1]
                if len(interim) == 0:
                    continue
                interim_high = _safe_float(interim["high"].max())
                if interim_high is None or interim_high <= 0:
                    continue
                interim_high_idx = int(interim["high"].idxmax())

                avg_low = (float(l1.price) + float(l2.price)) / 2.0
                height_abs = interim_high - avg_low
                if height_abs <= 0:
                    continue

                height_drop_pct = height_abs / interim_high * 100.0
                if not (self.height_min <= height_drop_pct <= self.height_max):
                    continue

                breakout_idx, breakout_price, vol_ok = self._find_breakout(
                    df,
                    start_idx=int(l2.idx) + 1,
                    level=float(interim_high),
                    direction="up",
                    max_lookahead=30,
                )
                if breakout_idx is None or breakout_price is None:
                    continue

                target = breakout_price + height_abs
                stop = avg_low

                confidence = 75
                if vol_ok:
                    confidence += 10
                confidence = int(min(100, confidence))

                pattern_id = f"{symbol}_{self.key}_{int(l1.idx)}_{int(l2.idx)}"

                out.append(
                    {
                        "pattern_id": pattern_id,
                        "symbol": symbol,
                        "pattern_name": self.key,
                        "pattern_type": self.pattern_type,
                        "formation_start": str(df.iloc[int(l1.idx)]["date"].date()) if "date" in df.columns else str(int(l1.idx)),
                        "formation_end": str(df.iloc[int(l2.idx)]["date"].date()) if "date" in df.columns else str(int(l2.idx)),
                        "breakout_date": str(df.iloc[int(breakout_idx)]["date"].date()) if "date" in df.columns else None,
                        "breakout_idx": int(breakout_idx),
                        "breakout_direction": "up",
                        "breakout_price": float(breakout_price),
                        "target_price": float(target),
                        "stop_loss_price": float(stop),
                        "confidence_score": confidence,
                        "volume_confirmed": bool(vol_ok),
                        "pattern_height_pct": round(height_drop_pct, 2),
                        "pattern_width_bars": int(sep),
                        "touch_count": 2,
                        "pivot_indices": [int(l1.idx), int(interim_high_idx), int(l2.idx)],
                        "config_hash": self.config_hash,
                        "created_at": datetime.now().isoformat(),
                    }
                )

        return out


class SpikeScanner(BaseDigitizedScanner):
    def scan(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        pivots_filtered: List[Pivot],
        pivots_raw: List[Pivot],
    ) -> List[Dict[str, Any]]:
        geom = self.spec.get("geometry_constraints", {}) or {}
        mag = geom.get("spike_magnitude", {}) or {}
        min_range_pct = float(mag.get("min_range_pct") or 3.0) / 100.0

        out: List[Dict[str, Any]] = []
        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            cur = df.iloc[i]
            if float(prev["close"]) <= 0:
                continue
            rng = float(cur["high"] - cur["low"])
            rng_pct = rng / float(prev["close"])
            if rng_pct < min_range_pct:
                continue

            # Treat spike day as breakout; direction based on close-open
            breakout_dir = "up" if float(cur["close"]) >= float(cur["open"]) else "down"
            breakout_price = float(cur["close"])
            breakout_idx = i

            pattern_id = f"{symbol}_{self.key}_{i}_{i}"
            out.append(
                {
                    "pattern_id": pattern_id,
                    "symbol": symbol,
                    "pattern_name": self.key,
                    "pattern_type": self.pattern_type,
                    "formation_start": str(cur["date"].date()) if "date" in df.columns else str(i),
                    "formation_end": str(cur["date"].date()) if "date" in df.columns else str(i),
                    "breakout_date": str(cur["date"].date()) if "date" in df.columns else None,
                    "breakout_idx": int(breakout_idx),
                    "breakout_direction": breakout_dir,
                    "breakout_price": breakout_price,
                    "target_price": breakout_price + rng if breakout_dir == "up" else breakout_price - rng,
                    "stop_loss_price": float(cur["low"]) if breakout_dir == "up" else float(cur["high"]),
                    "confidence_score": 70,
                    "volume_confirmed": False,
                    "pattern_height_pct": round(rng_pct * 100.0, 2),
                    "pattern_width_bars": 1,
                    "touch_count": 1,
                    "pivot_indices": [int(i)],
                    "config_hash": self.config_hash,
                    "created_at": datetime.now().isoformat(),
                }
            )

        return out


def build_digitized_scanners(
    library: DigitizedPatternLibrary,
) -> Dict[str, BaseDigitizedScanner]:
    scanners: Dict[str, BaseDigitizedScanner] = {}
    for key in library.list_keys():
        spec = library.load(key)
        ds = spec.get("detection_signature", {}) or {}

        if key == "triple_bottoms_tops":
            scanners[key] = TripleBottomsTopsScanner(key, spec)
            continue
        if key == "pipe_bottoms":
            scanners[key] = PipeBottomScanner(key, spec)
            continue

        if key == "inside_day":
            scanners[key] = InsideDayScanner(key, spec)
            continue
        if key == "gaps":
            scanners[key] = GapScanner(key, spec)
            continue
        if key == "islands":
            scanners[key] = IslandScanner(key, spec)
            continue
        if key == "rising_falling_three_methods":
            scanners[key] = ThreeMethodsScanner(key, spec)
            continue
        if key == "spike_formation":
            scanners[key] = SpikeScanner(key, spec)
            continue

        # Default pivot-based implementation
        scanners[key] = PivotSequenceScanner(key, spec)

    return scanners
