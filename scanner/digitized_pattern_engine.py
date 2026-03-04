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

import copy
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
        # KPI + spec use bar counts (inclusive).
        width_bars = (end_idx - start_idx) + 1

        wmin = self.geom.get("width_min_bars")
        wmax = self.geom.get("width_max_bars")
        if wmin is not None and width_bars < int(wmin):
            return False, 0.0, width_bars
        if wmax is not None and width_bars > int(wmax):
            return False, 0.0, width_bars

        highs = [p.price for p in window if p.type == PivotType.HIGH]
        lows = [p.price for p in window if p.type == PivotType.LOW]
        if not highs or not lows:
            return False, 0.0, width_bars

        upper = max(highs)
        lower = min(lows)
        mid = (upper + lower) / 2.0 if (upper + lower) != 0 else max(upper, 1e-9)
        height_pct = (upper - lower) / mid * 100.0

        hmin = self.geom.get("height_ratio_min")
        hmax = self.geom.get("height_ratio_max")
        if hmin is not None and height_pct < float(hmin):
            return False, height_pct, width_bars
        if hmax is not None and height_pct > float(hmax):
            return False, height_pct, width_bars

        return True, height_pct, width_bars

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
            ok, height_pct, width_bars = self._validate_width_height(window)
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
                    "pattern_width_bars": int(width_bars),
                    "touch_count": int(len(window)),
                    "pivot_indices": [int(p.idx) for p in window],
                    "config_hash": self.config_hash,
                    "created_at": datetime.now().isoformat(),
                }
            )

        return out


class RoundingBottomsTopsScanner(BaseDigitizedScanner):
    """
    The digitized spec contains both rounding bottoms and rounding tops in one file.
    Mandatory pivots are annotated with `variant` ("bottom"/"top"), but the generic
    PivotSequenceScanner does not interpret that field. This wrapper instantiates
    two PivotSequenceScanners (bottom + top) using filtered mandatory pivots and
    an inverted pivot sequence for the top variant.
    """

    def __init__(self, key: str, spec: Dict[str, Any]):
        super().__init__(key, spec)

        ds = spec.get("detection_signature", {}) or {}
        base_seq = (ds.get("pivot_sequence") or []) if isinstance(ds.get("pivot_sequence"), list) else []
        base_mps = (ds.get("mandatory_pivots") or []) if isinstance(ds.get("mandatory_pivots"), list) else []

        def _invert_seq(seq: List[Any]) -> List[Any]:
            out: List[Any] = []
            for t in seq:
                if t == "H":
                    out.append("L")
                elif t == "L":
                    out.append("H")
                else:
                    out.append(t)
            return out

        bottom_spec = copy.deepcopy(spec)
        bottom_spec["pattern_type"] = "reversal_bullish"
        bottom_spec.setdefault("detection_signature", {})
        bottom_spec["detection_signature"]["pivot_sequence"] = list(base_seq)
        bottom_spec["detection_signature"]["mandatory_pivots"] = [
            mp for mp in base_mps if not isinstance(mp, dict) or (mp.get("variant") in (None, "", "bottom"))
        ]
        self._bottom = PivotSequenceScanner(key, bottom_spec)

        top_spec = copy.deepcopy(spec)
        top_spec["pattern_type"] = "reversal_bearish"
        top_spec.setdefault("detection_signature", {})
        top_spec["detection_signature"]["pivot_sequence"] = _invert_seq(list(base_seq))
        top_spec["detection_signature"]["mandatory_pivots"] = [
            mp for mp in base_mps if not isinstance(mp, dict) or (mp.get("variant") in (None, "", "top"))
        ]
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

        for row in self._bottom.scan(symbol=symbol, df=df, pivots_filtered=pivots_filtered, pivots_raw=pivots_raw):
            row["pattern_id"] = f"{row['pattern_id']}_bottom"
            row["pattern_type"] = "reversal_bullish"
            out.append(row)

        for row in self._top.scan(symbol=symbol, df=df, pivots_filtered=pivots_filtered, pivots_raw=pivots_raw):
            row["pattern_id"] = f"{row['pattern_id']}_top"
            row["pattern_type"] = "reversal_bearish"
            out.append(row)

        return out


class CupWithHandleScanner(BaseDigitizedScanner):
    """
    Cup-with-handle is fundamentally a curved formation; a strict 6-pivot template is brittle.
    This scanner uses a pragmatic approach:
    - Identify candidate cup rims (two highs) separated by the cup width
    - Find the cup bottom between rims
    - Verify depth + handle constraints from digitized spec
    - Confirm breakout above handle resistance with optional volume requirement

    The goal is coverage on real-world OHLCV while staying anchored to digitized constraints.
    """

    def __init__(self, key: str, spec: Dict[str, Any]):
        super().__init__(key, spec)
        self.geom = spec.get("geometry_constraints", {}) or {}
        self.prior = spec.get("prior_trend_requirements", {}) or {}
        self.bo = spec.get("breakout_confirmation", {}) or {}

        self.width_min = int(self.geom.get("width_min_bars") or 84)
        self.width_max = int(self.geom.get("width_max_bars") or 504)
        self.depth_min = float(self.geom.get("height_ratio_min") or 15.0)
        self.depth_max = float(self.geom.get("height_ratio_max") or 33.0)

        handle = self.geom.get("handle_constraints", {}) or {}
        self.handle_min_bars = 5
        self.handle_max_bars = 20
        self.handle_pos_min_pct = float(handle.get("handle_position_min_pct") or 67.0)
        self.handle_min_decline_pct = float(handle.get("handle_min_decline_pct") or 2.0)
        self.handle_max_decline_pct = float(handle.get("handle_max_decline_pct") or 12.0)

        # Rims "near equal" is very strict in some digitizations; treat the digitized tolerance
        # as "ideal" and apply a looser hard cap for coverage. Confidence scoring rewards tight rims.
        self.rim_tol_ideal = float(self.geom.get("near_equal_tolerance_pct") or 2.0)
        self.rim_tol_hard = max(self.rim_tol_ideal, 10.0)

        self.breakout_thr = float(self.bo.get("breakout_threshold_pct") or 1.0) / 100.0
        self.confirm_bars = int(self.bo.get("confirmation_bars") or 1)
        self.close_beyond = bool(self.bo.get("close_beyond_required") if self.bo.get("close_beyond_required") is not None else True)
        self.vol_required = bool(self.bo.get("volume_required") or False)
        self.vol_mult_min = float(self.bo.get("volume_multiplier_min") or 1.3)

        # Bound breakout search to keep scans fast.
        self.breakout_search_bars = int(self.geom.get("breakout_search_bars") or 60)

    def _prior_trend_ok(self, df: pd.DataFrame, start_idx: int) -> bool:
        direction = str(self.prior.get("direction") or "up").lower()
        min_bars = int(self.prior.get("min_period_bars") or 0)
        min_change = float(self.prior.get("min_change_pct") or 0.0)
        if min_bars <= 0 or min_change <= 0:
            return True
        if start_idx < min_bars:
            return False
        p0 = _safe_float(df.iloc[start_idx - min_bars].get("close"))
        p1 = _safe_float(df.iloc[start_idx].get("close"))
        if p0 is None or p1 is None or p0 <= 0:
            return False
        change_pct = (p1 - p0) / p0 * 100.0
        if direction == "up":
            return change_pct >= min_change
        if direction == "down":
            return change_pct <= -min_change
        return abs(change_pct) >= min_change

    def _breakout_ok(self, df: pd.DataFrame, idx0: int, level: float) -> Tuple[Optional[int], Optional[float], bool]:
        """
        Find breakout above `level` starting at idx0. Returns (breakout_idx, breakout_price, vol_ok).
        """
        end = min(len(df), idx0 + self.breakout_search_bars)
        thr = level * (1.0 + self.breakout_thr)
        for i in range(idx0, end):
            close = _safe_float(df.iloc[i].get("close"))
            if close is None:
                continue
            if close <= thr:
                continue

            # Confirmation bars: require consecutive closes beyond threshold.
            if self.confirm_bars > 1:
                j_end = min(len(df), i + self.confirm_bars)
                all_ok = True
                for j in range(i, j_end):
                    c = _safe_float(df.iloc[j].get("close"))
                    if c is None or c <= thr:
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
        pivots = pivots_filtered
        highs = [p for p in pivots if p.type == PivotType.HIGH]
        if len(highs) < 2:
            return []

        out: List[Dict[str, Any]] = []

        for i in range(len(highs) - 1):
            left = highs[i]
            # width_min is a bar-count (inclusive)
            if int(left.idx) + int(self.width_min) - 1 >= len(df):
                continue
            if not self._prior_trend_ok(df, int(left.idx)):
                continue

            for j in range(i + 1, len(highs)):
                right = highs[j]
                width_bars = int(right.idx) - int(left.idx) + 1
                if width_bars < self.width_min:
                    continue
                if width_bars > self.width_max:
                    break

                # Rim similarity (hard cap for coverage; scoring uses ideal tolerance).
                rim_diff = _pct_diff(float(left.price), float(right.price))
                if rim_diff > self.rim_tol_hard:
                    continue

                seg = df.iloc[int(left.idx) : int(right.idx) + 1]
                if len(seg) < 3:
                    continue
                bottom_low = _safe_float(seg["low"].min())
                if bottom_low is None or bottom_low <= 0:
                    continue

                # Depth % (cup depth vs average rim)
                rim_avg = (float(left.price) + float(right.price)) / 2.0
                if rim_avg <= 0:
                    continue
                depth_pct = (rim_avg - bottom_low) / rim_avg * 100.0
                if depth_pct < self.depth_min or depth_pct > self.depth_max:
                    continue

                # Handle search window
                handle_start = int(right.idx) + 1
                if handle_start + self.handle_min_bars >= len(df):
                    continue
                handle_end_max = min(len(df) - 1, handle_start + self.handle_max_bars - 1)

                for handle_end in range(handle_start + self.handle_min_bars - 1, handle_end_max + 1):
                    # Enforce width constraints on the *full* formation (cup + handle).
                    # We persist `pattern_width_bars` as left-rim -> handle_end inclusive.
                    total_width_bars = int(handle_end) - int(left.idx) + 1
                    if total_width_bars > self.width_max:
                        break
                    hseg = df.iloc[handle_start : handle_end + 1]
                    if len(hseg) == 0:
                        continue
                    handle_low = _safe_float(hseg["low"].min())
                    if handle_low is None:
                        continue
                    handle_decline = (rim_avg - handle_low) / rim_avg * 100.0
                    if handle_decline < self.handle_min_decline_pct or handle_decline > self.handle_max_decline_pct:
                        continue

                    # Handle must be in upper third of cup depth (position % from bottom -> rim).
                    denom = rim_avg - bottom_low
                    if denom <= 0:
                        continue
                    handle_pos_pct = (handle_low - bottom_low) / denom * 100.0
                    if handle_pos_pct < self.handle_pos_min_pct:
                        continue

                    handle_res = _safe_float(hseg["high"].max())
                    if handle_res is None or handle_res <= 0:
                        continue
                    if _pct_diff(handle_res, rim_avg) > self.rim_tol_hard:
                        continue

                    breakout_idx, breakout_price, vol_ok = self._breakout_ok(df, handle_end + 1, handle_res)
                    if breakout_idx is None or breakout_price is None:
                        continue

                    # Target/stop (pragmatic): depth in absolute price units.
                    depth_abs = max(0.0, rim_avg - bottom_low)
                    target = breakout_price + depth_abs
                    stop = float(handle_low)

                    confidence = 60
                    if rim_diff <= self.rim_tol_ideal:
                        confidence += 10
                    elif rim_diff <= 5.0:
                        confidence += 5
                    if vol_ok:
                        confidence += 10
                    confidence += 10  # breakout found
                    confidence = int(min(100, confidence))

                    # Use positional indices (pivot engine works on iloc positions).
                    bottom_idx = int(left.idx) + int(np.argmin(seg["low"].to_numpy()))
                    handle_low_idx = int(handle_start) + int(np.argmin(hseg["low"].to_numpy()))

                    pattern_id = f"{symbol}_{self.key}_{int(left.idx)}_{int(handle_end)}"
                    out.append(
                        {
                            "pattern_id": pattern_id,
                            "symbol": symbol,
                            "pattern_name": self.key,
                            "pattern_type": self.pattern_type,
                            "formation_start": str(df.iloc[int(left.idx)]["date"].date()) if "date" in df.columns else str(int(left.idx)),
                            "formation_end": str(df.iloc[int(handle_end)]["date"].date()) if "date" in df.columns else str(int(handle_end)),
                            "breakout_date": str(df.iloc[int(breakout_idx)]["date"].date()) if "date" in df.columns else None,
                            "breakout_idx": int(breakout_idx),
                            "breakout_direction": "up",
                            "breakout_price": float(breakout_price),
                            "target_price": float(target),
                            "stop_loss_price": float(stop),
                            "confidence_score": confidence,
                            "volume_confirmed": bool(vol_ok),
                            "pattern_height_pct": round(float(depth_pct), 2),
                            "pattern_width_bars": (int(handle_end) - int(left.idx)) + 1,
                            "touch_count": 6,
                            "pivot_indices": [int(left.idx), bottom_idx, int(right.idx), handle_low_idx, int(handle_end)],
                            "config_hash": self.config_hash,
                            "created_at": datetime.now().isoformat(),
                        }
                    )
                    break  # stop at first valid handle+breakout for this rim pair

        return out


def _invert_ohlcv_prices(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Invert OHLC prices around a constant to mirror bullish/bearish formations.

    Returns (inverted_df, a) where inverted_price = a - price.
    """
    if df.empty:
        return df.copy(), 0.0
    hi = pd.to_numeric(df["high"], errors="coerce")
    lo = pd.to_numeric(df["low"], errors="coerce")
    if hi.dropna().empty or lo.dropna().empty:
        return df.copy(), 0.0
    a = float(hi.max()) + float(lo.min())

    inv = df.copy()
    inv_open = a - pd.to_numeric(inv["open"], errors="coerce")
    inv_close = a - pd.to_numeric(inv["close"], errors="coerce")
    inv_high_raw = a - pd.to_numeric(inv["low"], errors="coerce")
    inv_low_raw = a - pd.to_numeric(inv["high"], errors="coerce")
    inv["open"] = inv_open
    inv["close"] = inv_close
    inv["high"] = np.maximum(inv_high_raw, inv_low_raw)
    inv["low"] = np.minimum(inv_high_raw, inv_low_raw)
    return inv, a


def _invert_pivots(pivots: List[Pivot], a: float) -> List[Pivot]:
    out: List[Pivot] = []
    for p in pivots:
        if p.type == PivotType.HIGH:
            t = PivotType.LOW
        elif p.type == PivotType.LOW:
            t = PivotType.HIGH
        else:
            t = p.type
        try:
            price = float(a) - float(p.price)
        except Exception:
            continue
        out.append(
            Pivot(
                idx=int(p.idx),
                date=p.date,
                price=price,
                type=t,
                strength=int(getattr(p, "strength", 0) or 0),
                classification=str(getattr(p, "classification", "") or ""),
            )
        )
    return out


class InvertedCupWithHandleScanner(CupWithHandleScanner):
    """
    Detects Bulkowski's "Cup with Handle, Inverted" by mirroring price series and pivots,
    then re-using the bullish cup-with-handle logic.

    Output is mapped back into the original (non-inverted) price space:
      - breakout_direction is 'down'
      - breakout/target/stop prices use the original df
    """

    def scan(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        pivots_filtered: List[Pivot],
        pivots_raw: List[Pivot],
    ) -> List[Dict[str, Any]]:
        inv_df, a = _invert_ohlcv_prices(df)
        inv_pf = _invert_pivots(pivots_filtered, a)
        inv_pr = _invert_pivots(pivots_raw, a)

        rows = super().scan(symbol=symbol, df=inv_df, pivots_filtered=inv_pf, pivots_raw=inv_pr)
        out: List[Dict[str, Any]] = []

        for r in rows:
            piv = r.get("pivot_indices") or []
            if not isinstance(piv, (list, tuple)) or len(piv) < 5:
                continue
            try:
                left_idx = int(piv[0])
                right_idx = int(piv[2])
                handle_end_idx = int(piv[4])
            except Exception:
                continue
            if not (0 <= left_idx < len(df) and 0 <= right_idx < len(df) and 0 <= handle_end_idx < len(df)):
                continue

            # Inverted cup rims correspond to *lows* in the original df.
            rim1 = _safe_float(df.iloc[left_idx].get("low"))
            rim2 = _safe_float(df.iloc[right_idx].get("low"))
            if rim1 is None or rim2 is None or rim1 <= 0 or rim2 <= 0:
                continue
            rim_avg = (float(rim1) + float(rim2)) / 2.0

            seg = df.iloc[min(left_idx, right_idx) : max(left_idx, right_idx) + 1]
            if len(seg) == 0:
                continue
            cup_top = _safe_float(seg["high"].max())
            if cup_top is None or cup_top <= 0:
                continue

            depth_abs = float(cup_top) - float(rim_avg)
            if depth_abs <= 0:
                continue

            breakout_idx = r.get("breakout_idx")
            try:
                bi = int(breakout_idx) if breakout_idx is not None else None
            except Exception:
                bi = None
            if bi is None or not (0 <= bi < len(df)):
                continue

            breakout_price = _safe_float(df.iloc[bi].get("close"))
            if breakout_price is None or breakout_price <= 0:
                continue

            # Stop = handle high (original). Handle starts after right rim.
            hs = min(len(df), max(right_idx + 1, 0))
            he = min(len(df) - 1, max(handle_end_idx, hs))
            hseg = df.iloc[hs : he + 1] if hs <= he else df.iloc[handle_end_idx : handle_end_idx + 1]
            handle_high = _safe_float(hseg["high"].max()) if len(hseg) else None
            if handle_high is None or handle_high <= 0:
                continue

            row = dict(r)
            row["pattern_type"] = "continuation_bearish"
            row["breakout_direction"] = "down"
            row["breakout_price"] = float(breakout_price)
            row["target_price"] = float(breakout_price) - float(depth_abs)
            row["stop_loss_price"] = float(handle_high)
            out.append(row)

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

        geom = self.spec.get("geometry_constraints", {}) or {}
        hmin = geom.get("height_ratio_min")
        hmax = geom.get("height_ratio_max")

        out: List[Dict[str, Any]] = []
        for i in range(1, len(df)):
            y = df.iloc[i - 1]
            t = df.iloc[i]
            if t["high"] < y["high"] and t["low"] > y["low"]:
                inside_high = float(t["high"])
                inside_low = float(t["low"])
                if inside_high <= inside_low:
                    continue
                height_abs = inside_high - inside_low
                ref = (inside_high + inside_low) / 2.0
                if ref <= 0:
                    continue
                height_pct = height_abs / ref * 100.0
                if hmin is not None and height_pct < float(hmin):
                    continue
                if hmax is not None and height_pct > float(hmax):
                    continue

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

                pattern_id = f"{symbol}_{self.key}_{i-1}_{i}"
                target = breakout_price + height_abs if breakout_dir == "up" else breakout_price - height_abs
                stop = inside_low if breakout_dir == "up" else inside_high

                out.append(
                    {
                        "pattern_id": pattern_id,
                        "symbol": symbol,
                        "pattern_name": self.key,
                        "pattern_type": self.pattern_type,
                        "formation_start": str(y["date"].date()) if "date" in df.columns else str(i - 1),
                        "formation_end": str(t["date"].date()) if "date" in df.columns else str(i),
                        "breakout_date": str(df.iloc[breakout_idx]["date"].date()) if "date" in df.columns else None,
                        "breakout_idx": int(breakout_idx),
                        "breakout_direction": breakout_dir,
                        "breakout_price": breakout_price,
                        "target_price": target,
                        "stop_loss_price": stop,
                        "confidence_score": 70,
                        "volume_confirmed": False,
                        "pattern_height_pct": round(height_pct, 2),
                        "pattern_width_bars": 2,
                        "touch_count": 2,
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
        gap_cfg = (geom.get("gap_constraints") or {}) if isinstance(geom.get("gap_constraints"), dict) else {}
        min_gap = float((gap_cfg.get("min_gap_size_pct")) or 0.1) / 100.0
        max_gap_pct = gap_cfg.get("max_gap_size_pct")
        max_gap = float(max_gap_pct) / 100.0 if max_gap_pct is not None else None
        hmin = geom.get("height_ratio_min")
        hmax = geom.get("height_ratio_max")

        out: List[Dict[str, Any]] = []
        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            cur = df.iloc[i]
            # Gap up: today's low above yesterday's high
            if cur["low"] > prev["high"] * (1.0 + min_gap):
                pattern_id = f"{symbol}_{self.key}_{i}_{i}"
                gap_edge_1 = float(prev["high"])
                gap_edge_2 = float(cur["low"])
                height_abs = float(gap_edge_2 - gap_edge_1)
                if height_abs <= 0:
                    continue
                ref = (gap_edge_1 + gap_edge_2) / 2.0
                if ref <= 0:
                    continue
                height_pct = height_abs / ref
                if max_gap is not None and height_pct > max_gap:
                    continue
                height_pct_100 = height_pct * 100.0
                if hmin is not None and height_pct_100 < float(hmin):
                    continue
                if hmax is not None and height_pct_100 > float(hmax):
                    continue
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
                        "pattern_height_pct": round(height_pct_100, 3),
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
                gap_edge_1 = float(cur["high"])
                gap_edge_2 = float(prev["low"])
                height_abs = float(gap_edge_2 - gap_edge_1)
                if height_abs <= 0:
                    continue
                ref = (gap_edge_1 + gap_edge_2) / 2.0
                if ref <= 0:
                    continue
                height_pct = height_abs / ref
                if max_gap is not None and height_pct > max_gap:
                    continue
                height_pct_100 = height_pct * 100.0
                if hmin is not None and height_pct_100 < float(hmin):
                    continue
                if hmax is not None and height_pct_100 > float(hmax):
                    continue
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
                        "pattern_height_pct": round(height_pct_100, 3),
                        "pattern_width_bars": 1,
                        "touch_count": 1,
                        "pivot_indices": [int(i - 1), int(i)],
                        "config_hash": self.config_hash,
                        "created_at": datetime.now().isoformat(),
                    }
                )

        return out


class DeadCatBounceScanner(BaseDigitizedScanner):
    """
    Bulkowski Part Two (Event Patterns): Dead-Cat Bounce (DCB).

    Detection is OHLCV-only and avoids look-ahead beyond the "breakout" anchor:
      - Identify a sharp event decline (15%+ from pre-event high to event low within <= 8 bars)
      - Identify a bounce of 15% to 35% from the event low, peaking 5 to 25 bars after the event low
      - Breakout/anchor is the bounce peak day, expecting a post-bounce decline (measured by evaluator)
    """

    def __init__(self, key: str, spec: Dict[str, Any]):
        super().__init__(key, spec)
        s = spec.get("event_constraints", {}) or {}
        self.event_decline_min_pct = float(s.get("event_decline_min_pct") or 15.0)
        self.event_decline_max_bars = int(s.get("event_decline_max_bars") or 8)
        self.bounce_min_pct = float(s.get("bounce_min_pct") or 15.0)
        self.bounce_max_pct = float(s.get("bounce_max_pct") or 35.0)
        self.bounce_min_bars = int(s.get("bounce_min_bars") or 5)
        self.bounce_max_bars = int(s.get("bounce_max_bars") or 25)
        self.gap_preferred = bool(s.get("gap_preferred") if s.get("gap_preferred") is not None else True)

        v = spec.get("volume_constraints", {}) or {}
        self.vol_ratio_preferred = float(v.get("event_volume_ratio_preferred") or 2.0)

    @staticmethod
    def _gap_down(df: pd.DataFrame, i: int) -> bool:
        if i <= 0 or i >= len(df):
            return False
        try:
            return float(df.iloc[i]["high"]) < float(df.iloc[i - 1]["low"])
        except Exception:
            return False

    def scan(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        pivots_filtered: List[Pivot],
        pivots_raw: List[Pivot],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if len(df) < 50:
            return out

        seen: set[tuple[int, int]] = set()
        n = len(df)

        for event_start in range(1, n - 2):
            pre_high = _safe_float(df.iloc[event_start - 1].get("high"))
            if pre_high is None or pre_high <= 0:
                continue

            # Event decline window (<= 8 bars, inclusive).
            end = min(n - 1, event_start + max(1, self.event_decline_max_bars))
            seg = df.iloc[event_start : end + 1]
            if seg.empty:
                continue

            event_low = _safe_float(seg["low"].min())
            if event_low is None or event_low <= 0:
                continue
            try:
                low_vals = seg["low"].to_numpy(dtype=float, copy=False)
                if low_vals.size == 0 or not np.isfinite(low_vals).any():
                    continue
                event_low_rel = int(np.nanargmin(low_vals))
                event_low_idx = int(event_start + event_low_rel)
            except Exception:
                continue

            event_decline_pct = (float(pre_high) - float(event_low)) / float(pre_high) * 100.0
            if event_decline_pct < float(self.event_decline_min_pct):
                continue

            # Bounce peak must occur 5..25 bars after the event low.
            b0 = event_low_idx + int(self.bounce_min_bars)
            b1 = event_low_idx + int(self.bounce_max_bars)
            if b0 >= n:
                continue
            b1 = min(n - 1, b1)
            bseg = df.iloc[b0 : b1 + 1]
            if bseg.empty:
                continue
            bounce_high = _safe_float(bseg["high"].max())
            if bounce_high is None or bounce_high <= 0:
                continue
            try:
                high_vals = bseg["high"].to_numpy(dtype=float, copy=False)
                if high_vals.size == 0 or not np.isfinite(high_vals).any():
                    continue
                bounce_high_rel = int(np.nanargmax(high_vals))
                bounce_high_idx = int(b0 + bounce_high_rel)
            except Exception:
                continue

            bounce_pct = (float(bounce_high) - float(event_low)) / float(event_low) * 100.0
            if bounce_pct < float(self.bounce_min_pct) or bounce_pct > float(self.bounce_max_pct):
                continue

            sig = (int(event_low_idx), int(bounce_high_idx))
            if sig in seen:
                continue
            seen.add(sig)

            # Anchor = bounce peak close (expecting subsequent decline).
            breakout_idx = int(bounce_high_idx)
            breakout_price = _safe_float(df.iloc[breakout_idx].get("close"))
            if breakout_price is None or breakout_price <= 0:
                continue

            gap = self._gap_down(df, int(event_start))
            confidence = 65
            if gap:
                confidence += 10
            if float(event_decline_pct) >= 25.0:
                confidence += 5
            if float(bounce_pct) >= 25.0:
                confidence += 5

            vr = df.iloc[int(event_start)].get("volume_ratio", np.nan)
            if pd.notna(vr) and np.isfinite(vr) and float(vr) >= float(self.vol_ratio_preferred):
                confidence += 10
            confidence = int(min(100, confidence))

            pattern_id = f"{symbol}_{self.key}_{int(event_start)}_{int(bounce_high_idx)}"
            width_bars = int(bounce_high_idx - event_start + 1)

            out.append(
                {
                    "pattern_id": pattern_id,
                    "symbol": symbol,
                    "pattern_name": self.key,
                    "pattern_type": self.pattern_type,
                    "formation_start": str(df.iloc[int(event_start)]["date"].date()) if "date" in df.columns else str(int(event_start)),
                    "formation_end": str(df.iloc[int(bounce_high_idx)]["date"].date()) if "date" in df.columns else str(int(bounce_high_idx)),
                    "breakout_date": str(df.iloc[int(breakout_idx)]["date"].date()) if "date" in df.columns else None,
                    "breakout_idx": int(breakout_idx),
                    "breakout_direction": "down",
                    "breakout_price": float(breakout_price),
                    "target_price": None,
                    "stop_loss_price": None,
                    "confidence_score": confidence,
                    "volume_confirmed": bool(gap) if self.gap_preferred else False,
                    # Store event-decline magnitude as pattern height (event KPI).
                    "pattern_height_pct": round(float(event_decline_pct), 2),
                    "pattern_width_bars": int(width_bars),
                    "touch_count": 4,
                    # Indices: pre-event high, event start, event low, bounce peak.
                    "pivot_indices": [int(event_start - 1), int(event_start), int(event_low_idx), int(bounce_high_idx)],
                    "config_hash": self.config_hash,
                    "created_at": datetime.now().isoformat(),
                }
            )

        return out


class DeadCatBounceInvertedScanner(BaseDigitizedScanner):
    """
    Bulkowski Part Two (Event Patterns): Inverted Dead-Cat Bounce (iDCB).

    OHLCV-only, causal detection at day 2 (Bulkowski-style: selling on day 2):
      - Day 1: large 1-day upward move (>= 5% close-to-close by default)
      - Day 2: higher high and higher low than day 1 (a final push)
      - Anchor/breakout is day 2 close, expecting a giveback (measured by evaluator)
    """

    def __init__(self, key: str, spec: Dict[str, Any]):
        super().__init__(key, spec)
        s = spec.get("event_constraints", {}) or {}
        self.up_move_min_pct = float(s.get("up_move_min_pct") or 5.0)
        self.gap_preferred = bool(s.get("gap_preferred") if s.get("gap_preferred") is not None else True)

        v = spec.get("volume_constraints", {}) or {}
        self.vol_ratio_preferred = float(v.get("event_volume_ratio_preferred") or 2.0)

    @staticmethod
    def _gap_up(df: pd.DataFrame, i: int) -> bool:
        if i <= 0 or i >= len(df):
            return False
        try:
            return float(df.iloc[i]["low"]) > float(df.iloc[i - 1]["high"])
        except Exception:
            return False

    def scan(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        pivots_filtered: List[Pivot],
        pivots_raw: List[Pivot],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if len(df) < 10:
            return out

        n = len(df)
        for day1 in range(1, n - 1):
            ref_close = _safe_float(df.iloc[day1 - 1].get("close"))
            c1 = _safe_float(df.iloc[day1].get("close"))
            if ref_close is None or c1 is None or ref_close <= 0:
                continue
            move_pct = (float(c1) - float(ref_close)) / float(ref_close) * 100.0
            if move_pct < float(self.up_move_min_pct):
                continue

            day2 = day1 + 1
            try:
                h1 = float(df.iloc[day1]["high"])
                l1 = float(df.iloc[day1]["low"])
                h2 = float(df.iloc[day2]["high"])
                l2 = float(df.iloc[day2]["low"])
            except Exception:
                continue

            # Bulkowski-style day 2 push: higher high and higher low.
            if not (h2 > h1 and l2 > l1):
                continue

            breakout_idx = int(day2)
            breakout_price = _safe_float(df.iloc[breakout_idx].get("close"))
            if breakout_price is None or breakout_price <= 0:
                continue

            gap = self._gap_up(df, int(day1))
            confidence = 60
            if gap:
                confidence += 10
            if float(move_pct) >= 10.0:
                confidence += 10
            vr = df.iloc[int(day1)].get("volume_ratio", np.nan)
            if pd.notna(vr) and np.isfinite(vr) and float(vr) >= float(self.vol_ratio_preferred):
                confidence += 10
            confidence = int(min(100, confidence))

            pattern_id = f"{symbol}_{self.key}_{int(day1)}_{int(day2)}"
            out.append(
                {
                    "pattern_id": pattern_id,
                    "symbol": symbol,
                    "pattern_name": self.key,
                    "pattern_type": self.pattern_type,
                    "formation_start": str(df.iloc[int(day1 - 1)]["date"].date()) if "date" in df.columns else str(int(day1 - 1)),
                    "formation_end": str(df.iloc[int(day2)]["date"].date()) if "date" in df.columns else str(int(day2)),
                    "breakout_date": str(df.iloc[int(breakout_idx)]["date"].date()) if "date" in df.columns else None,
                    "breakout_idx": int(breakout_idx),
                    "breakout_direction": "down",
                    "breakout_price": float(breakout_price),
                    "target_price": None,
                    "stop_loss_price": None,
                    "confidence_score": confidence,
                    "volume_confirmed": bool(gap) if self.gap_preferred else False,
                    "pattern_height_pct": round(float(move_pct), 2),
                    "pattern_width_bars": 3,
                    "touch_count": 3,
                    "pivot_indices": [int(day1 - 1), int(day1), int(day2)],
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
        # Respect digitized constraints where possible.
        geom = self.spec.get("geometry_constraints", {}) or {}
        gap_cfg = (geom.get("gap_constraints") or {}) if isinstance(geom.get("gap_constraints"), dict) else {}
        min_gap_pct = float((gap_cfg.get("min_gap_size_pct")) or 0.5) / 100.0
        max_gap_pct = gap_cfg.get("max_gap_size_pct")
        max_gap = float(max_gap_pct) / 100.0 if max_gap_pct is not None else None

        dur = self.spec.get("duration_constraints", {}) or {}
        max_island_bars = int((geom.get("island_duration") or {}).get("max_bars") or geom.get("width_max_bars") or dur.get("max_bars") or 20)
        min_island_bars = int(dur.get("min_bars") or 1)

        hmin = geom.get("height_ratio_min")
        hmax = geom.get("height_ratio_max")

        out: List[Dict[str, Any]] = []

        def _gap_pct(prev_high: float, cur_low: float) -> Optional[float]:
            # Return gap size as a fraction of mid-price.
            height_abs = cur_low - prev_high
            if height_abs <= 0:
                return None
            ref = (prev_high + cur_low) / 2.0
            if ref <= 0:
                return None
            return height_abs / ref

        def is_gap_up(i: int) -> bool:
            if i <= 0:
                return False
            prev_high = float(df.iloc[i - 1]["high"])
            cur_low = float(df.iloc[i]["low"])
            if cur_low <= prev_high * (1.0 + min_gap_pct):
                return False
            gp = _gap_pct(prev_high, cur_low)
            if gp is None:
                return False
            if max_gap is not None and gp > max_gap:
                return False
            return True

        def is_gap_down(i: int) -> bool:
            if i <= 0:
                return False
            prev_low = float(df.iloc[i - 1]["low"])
            cur_high = float(df.iloc[i]["high"])
            if cur_high >= prev_low * (1.0 - min_gap_pct):
                return False
            # For gap down, swap to reuse _gap_pct (prev_high < cur_low analogue)
            gp = _gap_pct(cur_high, prev_low)
            if gp is None:
                return False
            if max_gap is not None and gp > max_gap:
                return False
            return True

        i = 1
        while i < len(df):
            if is_gap_up(i):
                # Look for gap down within window
                j_end = min(len(df), i + max_island_bars)
                j = i + 1
                while j < j_end:
                    if is_gap_down(j):
                        # Island top
                        width_bars = int(j - i + 1)
                        if width_bars < min_island_bars or width_bars > max_island_bars:
                            j += 1
                            continue
                        if (geom.get("width_min_bars") is not None and width_bars < int(geom.get("width_min_bars"))):
                            j += 1
                            continue
                        if (geom.get("width_max_bars") is not None and width_bars > int(geom.get("width_max_bars"))):
                            j += 1
                            continue
                        pattern_id = f"{symbol}_{self.key}_{i}_{j}"
                        breakout_price = float(df.iloc[j]["close"])
                        island_high = float(df.iloc[i:j]["high"].max())
                        island_low = float(df.iloc[i:j]["low"].min())
                        height_abs = island_high - island_low
                        if height_abs <= 0:
                            j += 1
                            continue
                        ref = (island_high + island_low) / 2.0
                        if ref <= 0:
                            j += 1
                            continue
                        height_pct = height_abs / ref * 100.0
                        if hmin is not None and height_pct < float(hmin):
                            j += 1
                            continue
                        if hmax is not None and height_pct > float(hmax):
                            j += 1
                            continue
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
                                "pattern_height_pct": round(height_pct, 2),
                                "pattern_width_bars": width_bars,
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
                j_end = min(len(df), i + max_island_bars)
                j = i + 1
                while j < j_end:
                    if is_gap_up(j):
                        # Island bottom
                        width_bars = int(j - i + 1)
                        if width_bars < min_island_bars or width_bars > max_island_bars:
                            j += 1
                            continue
                        if (geom.get("width_min_bars") is not None and width_bars < int(geom.get("width_min_bars"))):
                            j += 1
                            continue
                        if (geom.get("width_max_bars") is not None and width_bars > int(geom.get("width_max_bars"))):
                            j += 1
                            continue
                        pattern_id = f"{symbol}_{self.key}_{i}_{j}"
                        breakout_price = float(df.iloc[j]["close"])
                        island_high = float(df.iloc[i:j]["high"].max())
                        island_low = float(df.iloc[i:j]["low"].min())
                        height_abs = island_high - island_low
                        if height_abs <= 0:
                            j += 1
                            continue
                        ref = (island_high + island_low) / 2.0
                        if ref <= 0:
                            j += 1
                            continue
                        height_pct = height_abs / ref * 100.0
                        if hmin is not None and height_pct < float(hmin):
                            j += 1
                            continue
                        if hmax is not None and height_pct > float(hmax):
                            j += 1
                            continue
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
                                "pattern_height_pct": round(height_pct, 2),
                                "pattern_width_bars": width_bars,
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

        # KPI sanity: enforce digitized height constraints using a symmetric % definition.
        hmin = geom.get("height_ratio_min")
        hmax = geom.get("height_ratio_max")

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
            if height_abs <= 0:
                continue
            ref = (first_high + first_low) / 2.0
            if ref <= 0:
                continue
            height_pct = height_abs / ref * 100.0
            if hmin is not None and height_pct < float(hmin):
                continue
            if hmax is not None and height_pct > float(hmax):
                continue
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
                    "pattern_height_pct": round(height_pct, 2),
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
        if not np.isfinite(pre_high) or pre_high <= 0:
            return False
        drop_pct = (pre_high - low_price) / pre_high * 100.0
        if not (self.drop_min <= drop_pct <= self.drop_max):
            return False

        # Angle proxy (steeper = closer to vertical). Compute indices in *positional* space.
        try:
            pre_high_vals = pre["high"].to_numpy(dtype=float, copy=False)
            if pre_high_vals.size == 0 or not np.isfinite(pre_high_vals).any():
                return False
            pre_high_rel = int(np.nanargmax(pre_high_vals))
            pre_high_idx = int(pre0 + pre_high_rel)
        except Exception:
            return False
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
                width_bars = sep + 1
                if width_bars < self.width_min:
                    continue
                if width_bars > self.width_max:
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
                try:
                    interim_high_vals = interim["high"].to_numpy(dtype=float, copy=False)
                    if interim_high_vals.size == 0 or not np.isfinite(interim_high_vals).any():
                        continue
                    interim_high_rel = int(np.nanargmax(interim_high_vals))
                    interim_high_idx = int(int(l1.idx) + interim_high_rel)
                except Exception:
                    continue

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
                        "pattern_width_bars": int(width_bars),
                        "touch_count": 2,
                        "pivot_indices": [int(l1.idx), int(interim_high_idx), int(l2.idx)],
                        "config_hash": self.config_hash,
                        "created_at": datetime.now().isoformat(),
                    }
                )

        return out


class PipeTopScanner(BaseDigitizedScanner):
    """
    Pipe tops are the bearish mirror of pipe bottoms. This scanner:
    - finds 2 near-equal highs separated by a few bars
    - validates "vertical-ish" legs into/out of each high (pragmatic)
    - confirms breakout below the interim low after the 2nd high
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
        self.rise_min = float(vdr.get("min") or 0.0)
        self.rise_max = float(vdr.get("max") or 1e9)

        slope = self.geom.get("slope_constraints", {}) or {}
        self.angle_min = float(slope.get("pipe_vertical_angle_min_degrees") or 0.0)
        self.angle_max = float(slope.get("pipe_vertical_angle_max_degrees") or 90.0)

        self.height_min = float(self.geom.get("height_ratio_min") or 0.0)
        self.height_max = float(self.geom.get("height_ratio_max") or 1e9)

        self.breakout_thr = float(self.bo.get("breakout_threshold_pct") or 1.0) / 100.0
        self.confirm_bars = int(self.bo.get("confirmation_bars") or 1)
        self.vol_required = bool(self.bo.get("volume_required") or False)
        self.vol_mult_min = float(self.bo.get("volume_multiplier_min") or 1.3)

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

        if direction == "up":
            return change_pct >= min_change
        if direction == "down":
            return change_pct <= -min_change
        return abs(change_pct) >= min_change

    def _leg_ok(self, df: pd.DataFrame, high_idx: int, high_price: float) -> bool:
        if high_idx <= 0 or high_idx >= len(df):
            return False
        if high_price <= 0:
            return False

        pre0 = max(0, high_idx - self._leg_window_pre)
        pre = df.iloc[pre0:high_idx]
        if len(pre) == 0:
            return False
        pre_low = _safe_float(pre["low"].min())
        if pre_low is None or pre_low <= 0:
            return False
        rise_pct = (high_price - pre_low) / pre_low * 100.0
        if not (self.rise_min <= rise_pct <= self.rise_max):
            return False

        try:
            pre_low_vals = pre["low"].to_numpy(dtype=float, copy=False)
            if pre_low_vals.size == 0 or not np.isfinite(pre_low_vals).any():
                return False
            pre_low_rel = int(np.nanargmin(pre_low_vals))
            pre_low_idx = int(pre0 + pre_low_rel)
        except Exception:
            return False
        angle = abs(_slope_degrees(pre_low_idx, float(pre_low), high_idx, float(high_price)))
        if not (self.angle_min <= angle <= self.angle_max):
            return False

        post1 = min(len(df), high_idx + 1 + self._leg_window_post)
        post = df.iloc[high_idx + 1 : post1]
        if len(post) == 0:
            return False
        post_low = _safe_float(post["low"].min())
        if post_low is None or post_low <= 0:
            return False
        drop_pct = (high_price - post_low) / high_price * 100.0
        if drop_pct < self.rise_min:
            return False
        return True

    def _find_breakout(
        self,
        df: pd.DataFrame,
        *,
        start_idx: int,
        level: float,
        direction: str = "down",
        max_lookahead: int = 30,
    ) -> Tuple[Optional[int], Optional[float], bool]:
        thr = level * (1.0 - self.breakout_thr) if direction == "down" else level * (1.0 + self.breakout_thr)

        for i in range(start_idx, min(len(df), start_idx + max_lookahead)):
            close = _safe_float(df.iloc[i].get("close"))
            if close is None:
                continue
            ok = close < thr if direction == "down" else close > thr
            if not ok:
                continue

            if self.confirm_bars > 1:
                j_end = min(len(df), i + self.confirm_bars)
                all_ok = True
                for j in range(i, j_end):
                    c = _safe_float(df.iloc[j].get("close"))
                    if c is None:
                        all_ok = False
                        break
                    if direction == "down" and not (c < thr):
                        all_ok = False
                        break
                    if direction == "up" and not (c > thr):
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
        highs = [p for p in pivots if p.type == PivotType.HIGH]
        if len(highs) < 2:
            return []

        out: List[Dict[str, Any]] = []
        for i in range(len(highs) - 1):
            h1 = highs[i]
            for j in range(i + 1, len(highs)):
                h2 = highs[j]
                sep = int(h2.idx) - int(h1.idx)
                width_bars = sep + 1
                if width_bars < self.width_min:
                    continue
                if width_bars > self.width_max:
                    break
                if sep < self.sep_min or sep > self.sep_max:
                    continue

                if h1.price <= 0 or h2.price <= 0:
                    continue
                ratio = float(h2.price) / float(h1.price)
                if not (self.sim_min <= ratio <= self.sim_max):
                    continue

                if not self._check_prior_trend(df, int(h1.idx)):
                    continue

                if not self._leg_ok(df, int(h1.idx), float(h1.price)):
                    continue
                if not self._leg_ok(df, int(h2.idx), float(h2.price)):
                    continue

                interim = df.iloc[int(h1.idx) : int(h2.idx) + 1]
                if len(interim) == 0:
                    continue
                interim_low = _safe_float(interim["low"].min())
                if interim_low is None or interim_low <= 0:
                    continue
                try:
                    interim_low_vals = interim["low"].to_numpy(dtype=float, copy=False)
                    if interim_low_vals.size == 0 or not np.isfinite(interim_low_vals).any():
                        continue
                    interim_low_rel = int(np.nanargmin(interim_low_vals))
                    interim_low_idx = int(int(h1.idx) + interim_low_rel)
                except Exception:
                    continue

                avg_high = (float(h1.price) + float(h2.price)) / 2.0
                height_abs = avg_high - float(interim_low)
                if height_abs <= 0:
                    continue
                mid = (avg_high + float(interim_low)) / 2.0
                height_pct = (height_abs / mid * 100.0) if mid > 0 else None
                if height_pct is None or not np.isfinite(height_pct) or height_pct <= 0:
                    continue
                if not (self.height_min <= float(height_pct) <= self.height_max):
                    continue

                breakout_idx, breakout_price, vol_ok = self._find_breakout(
                    df,
                    start_idx=int(h2.idx) + 1,
                    level=float(interim_low),
                    direction="down",
                    max_lookahead=30,
                )
                if breakout_idx is None or breakout_price is None:
                    continue

                target = float(breakout_price) - float(height_abs)
                stop = float(avg_high)

                confidence = 75
                if vol_ok:
                    confidence += 10
                confidence = int(min(100, confidence))

                pattern_id = f"{symbol}_{self.key}_{int(h1.idx)}_{int(h2.idx)}"
                out.append(
                    {
                        "pattern_id": pattern_id,
                        "symbol": symbol,
                        "pattern_name": self.key,
                        "pattern_type": self.pattern_type,
                        "formation_start": str(df.iloc[int(h1.idx)]["date"].date()) if "date" in df.columns else str(int(h1.idx)),
                        "formation_end": str(df.iloc[int(h2.idx)]["date"].date()) if "date" in df.columns else str(int(h2.idx)),
                        "breakout_date": str(df.iloc[int(breakout_idx)]["date"].date()) if "date" in df.columns else None,
                        "breakout_idx": int(breakout_idx),
                        "breakout_direction": "down",
                        "breakout_price": float(breakout_price),
                        "target_price": float(target),
                        "stop_loss_price": float(stop),
                        "confidence_score": confidence,
                        "volume_confirmed": bool(vol_ok),
                        "pattern_height_pct": round(float(height_pct), 2),
                        "pattern_width_bars": int(width_bars),
                        "touch_count": 2,
                        "pivot_indices": [int(h1.idx), int(interim_low_idx), int(h2.idx)],
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
        hmin = geom.get("height_ratio_min")
        hmax = geom.get("height_ratio_max")

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
            height_pct = rng_pct * 100.0
            if hmin is not None and height_pct < float(hmin):
                continue
            if hmax is not None and height_pct > float(hmax):
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
                    "pattern_height_pct": round(height_pct, 2),
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

        if key == "rounding_bottoms_tops":
            scanners[key] = RoundingBottomsTopsScanner(key, spec)
            continue
        if key == "cup_with_handle":
            scanners[key] = CupWithHandleScanner(key, spec)
            continue

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


class _CachedScanner:
    def __init__(self, scanner: Any):
        self._scanner = scanner
        self._last_key: Optional[Tuple[str, int]] = None
        self._last_rows: Optional[List[Dict[str, Any]]] = None

    def scan(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        pivots_filtered: List[Pivot],
        pivots_raw: List[Pivot],
    ) -> List[Dict[str, Any]]:
        ck = (str(symbol), int(id(df)))
        if self._last_key == ck and self._last_rows is not None:
            return self._last_rows
        rows = self._scanner.scan(symbol=symbol, df=df, pivots_filtered=pivots_filtered, pivots_raw=pivots_raw)
        # Cache the original list; downstream wrappers must not mutate it.
        self._last_key = ck
        self._last_rows = list(rows)
        return self._last_rows


class _DerivedScanner:
    def __init__(
        self,
        key: str,
        base: Any,
        *,
        keep_if: Any,
        transform: Optional[Any] = None,
    ):
        self.key = key
        self._base = base
        self._keep_if = keep_if
        self._transform = transform

    def scan(
        self,
        *,
        symbol: str,
        df: pd.DataFrame,
        pivots_filtered: List[Pivot],
        pivots_raw: List[Pivot],
    ) -> List[Dict[str, Any]]:
        base_rows = self._base.scan(symbol=symbol, df=df, pivots_filtered=pivots_filtered, pivots_raw=pivots_raw)
        out: List[Dict[str, Any]] = []
        for r in base_rows:
            try:
                ok = bool(self._keep_if(r, df, pivots_filtered, pivots_raw))
            except Exception:
                ok = False
            if not ok:
                continue
            row = dict(r)
            row["pattern_name"] = self.key
            # Ensure uniqueness across derived patterns.
            row["pattern_id"] = f"{symbol}_{self.key}_{row.get('pattern_id')}"
            if self._transform is not None:
                try:
                    row = self._transform(row, df, pivots_filtered, pivots_raw) or row
                except Exception:
                    pass
            out.append(row)
        return out


def build_bulkowski_53_scanners(library: DigitizedPatternLibrary) -> Dict[str, Any]:
    """
    Build a scanner set aligned to Bulkowski Part One (53 chart-pattern chapters).

    Notes:
    - Uses local digitized specs when available, but does not require one spec file per chapter.
    - Some chapters are implemented as derived sub-views of a base detector (cached per symbol).
    - Event patterns (Part Two) are intentionally excluded here.
    """

    digitized = build_digitized_scanners(library)
    out: Dict[str, Any] = {}

    def _get(key: str) -> Optional[Any]:
        return digitized.get(key)

    def _pct(a: float, b: float) -> float:
        if b == 0:
            return float("inf")
        return abs(a - b) / abs(b) * 100.0

    def _peak_width_bars(df: pd.DataFrame, peak_idx: int, peak_price: float, *, tol_pct: float = 2.0, window: int = 15) -> Optional[int]:
        if peak_idx < 0 or peak_idx >= len(df) or peak_price <= 0:
            return None
        thr = float(peak_price) * (1.0 - float(tol_pct) / 100.0)
        left = peak_idx
        for k in range(1, window + 1):
            j = peak_idx - k
            if j < 0:
                break
            if float(df.iloc[j]["high"]) >= thr:
                left = j
            else:
                break
        right = peak_idx
        for k in range(1, window + 1):
            j = peak_idx + k
            if j >= len(df):
                break
            if float(df.iloc[j]["high"]) >= thr:
                right = j
            else:
                break
        return int(right - left + 1)

    def _trough_width_bars(df: pd.DataFrame, trough_idx: int, trough_price: float, *, tol_pct: float = 2.0, window: int = 15) -> Optional[int]:
        if trough_idx < 0 or trough_idx >= len(df) or trough_price <= 0:
            return None
        thr = float(trough_price) * (1.0 + float(tol_pct) / 100.0)
        left = trough_idx
        for k in range(1, window + 1):
            j = trough_idx - k
            if j < 0:
                break
            if float(df.iloc[j]["low"]) <= thr:
                left = j
            else:
                break
        right = trough_idx
        for k in range(1, window + 1):
            j = trough_idx + k
            if j >= len(df):
                break
            if float(df.iloc[j]["low"]) <= thr:
                right = j
            else:
                break
        return int(right - left + 1)

    def _ae(width: Optional[int], *, adam_max: int = 3, eve_min: int = 7) -> Optional[str]:
        if width is None:
            return None
        if int(width) <= int(adam_max):
            return "A"
        if int(width) >= int(eve_min):
            return "E"
        return None

    # Chapter 1 + 4
    if _get("broadening_bottoms"):
        out["broadening_bottoms"] = _get("broadening_bottoms")
    if _get("broadening_tops"):
        out["broadening_tops"] = _get("broadening_tops")

    # Chapter 2 + 3: built-in right-angled broadening formations
    right_asc_spec = library.load("broadening_formations_right_angled_ascending") if "broadening_formations_right_angled_ascending" in library.list_keys() else None
    if not isinstance(right_asc_spec, dict):
        right_asc_spec = {
            "pattern_name": "Broadening Formations, Right-Angled and Ascending",
            "pattern_type": "continuation_both",
            "digitization_version": "builtin_bulkowski_53_v1",
            "detection_signature": {
                "pivot_sequence": ["L", "H", "L", "H", "L", "H"],
                "pivot_order": "alternating",
                "mandatory_pivots": [
                    {"position": 3, "type": "L", "constraint": "near_equal"},
                    {"position": 5, "type": "L", "constraint": "near_equal"},
                    {"position": 4, "type": "H", "constraint": "higher_than_previous"},
                    {"position": 6, "type": "H", "constraint": "higher_than_previous"},
                ],
            },
            "geometry_constraints": {
                "width_min_bars": 21,
                "width_max_bars": 210,
                "height_ratio_min": 6.0,
                "height_ratio_max": 80.0,
                "near_equal_tolerance_pct": 2.0,
                "breakout_search_bars": 60,
            },
            "prior_trend_requirements": {"direction": "any", "min_period_bars": 21, "min_change_pct": 10.0},
            "breakout_confirmation": {"breakout_direction": "both", "breakout_threshold_pct": 1.0, "confirmation_bars": 1, "close_beyond_required": True},
        }

    right_desc_spec = library.load("broadening_formations_right_angled_descending") if "broadening_formations_right_angled_descending" in library.list_keys() else None
    if not isinstance(right_desc_spec, dict):
        right_desc_spec = {
            "pattern_name": "Broadening Formations, Right-Angled and Descending",
            "pattern_type": "continuation_both",
            "digitization_version": "builtin_bulkowski_53_v1",
            "detection_signature": {
                "pivot_sequence": ["H", "L", "H", "L", "H", "L"],
                "pivot_order": "alternating",
                "mandatory_pivots": [
                    {"position": 3, "type": "H", "constraint": "near_equal"},
                    {"position": 5, "type": "H", "constraint": "near_equal"},
                    {"position": 4, "type": "L", "constraint": "lower_than_previous"},
                    {"position": 6, "type": "L", "constraint": "lower_than_previous"},
                ],
            },
            "geometry_constraints": {
                "width_min_bars": 21,
                "width_max_bars": 210,
                "height_ratio_min": 6.0,
                "height_ratio_max": 80.0,
                "near_equal_tolerance_pct": 2.0,
                "breakout_search_bars": 60,
            },
            "prior_trend_requirements": {"direction": "any", "min_period_bars": 21, "min_change_pct": 10.0},
            "breakout_confirmation": {"breakout_direction": "both", "breakout_threshold_pct": 1.0, "confirmation_bars": 1, "close_beyond_required": True},
        }
    out["broadening_formations_right_angled_ascending"] = PivotSequenceScanner("broadening_formations_right_angled_ascending", right_asc_spec)
    out["broadening_formations_right_angled_descending"] = PivotSequenceScanner("broadening_formations_right_angled_descending", right_desc_spec)

    # Chapter 5 + 6: built-in broadening wedges (diverging wedge boundaries)
    bw_spec = library.load("broadening_wedges") if "broadening_wedges" in library.list_keys() else None
    if not isinstance(bw_spec, dict):
        bw_spec = {
            "pattern_name": "Broadening Wedges, Ascending/Descending",
            "pattern_type": "continuation_both",
            "digitization_version": "builtin_bulkowski_53_v1",
            "detection_signature": {
                "pivot_sequence": ["H", "L", "H", "L", "H", "L"],
                "pivot_order": "alternating",
                "mandatory_pivots": [],
            },
            "geometry_constraints": {
                "width_min_bars": 21,
                "width_max_bars": 180,
                "height_ratio_min": 6.0,
                "height_ratio_max": 80.0,
                "near_equal_tolerance_pct": 2.0,
                "breakout_search_bars": 60,
            },
            "prior_trend_requirements": {"direction": "any", "min_period_bars": 21, "min_change_pct": 10.0},
            "breakout_confirmation": {"breakout_direction": "both", "breakout_threshold_pct": 1.0, "confirmation_bars": 1, "close_beyond_required": True},
        }
    bw_base = _CachedScanner(PivotSequenceScanner("__broadening_wedges_base", bw_spec))

    def _bw_slopes(r: Dict[str, Any], df: pd.DataFrame) -> Optional[Tuple[float, float, float, float]]:
        piv = r.get("pivot_indices") or []
        if not isinstance(piv, (list, tuple)) or len(piv) < 6:
            return None
        try:
            idxs = [int(x) for x in piv[:6]]
        except Exception:
            return None
        hs = [idxs[i] for i in (0, 2, 4)]
        ls = [idxs[i] for i in (1, 3, 5)]
        if any(i < 0 or i >= len(df) for i in hs + ls):
            return None
        h0, h1 = hs[0], hs[-1]
        l0, l1 = ls[0], ls[-1]
        upper0 = float(df.iloc[h0]["high"])
        upper1 = float(df.iloc[h1]["high"])
        lower0 = float(df.iloc[l0]["low"])
        lower1 = float(df.iloc[l1]["low"])
        up_deg = _slope_degrees(h0, upper0, h1, upper1)
        lo_deg = _slope_degrees(l0, lower0, l1, lower1)
        d0 = upper0 - lower0
        d1 = upper1 - lower1
        return up_deg, lo_deg, d0, d1

    def _keep_bw_ascending(r: Dict[str, Any], df: pd.DataFrame, *_: Any) -> bool:
        s = _bw_slopes(r, df)
        if s is None:
            return False
        up_deg, lo_deg, d0, d1 = s
        if up_deg <= 0.2 or lo_deg <= 0.2:
            return False
        if d1 <= d0 * 1.05:
            return False
        # Upper boundary rises faster than lower -> broadening.
        return up_deg > lo_deg + 0.1

    def _keep_bw_descending(r: Dict[str, Any], df: pd.DataFrame, *_: Any) -> bool:
        s = _bw_slopes(r, df)
        if s is None:
            return False
        up_deg, lo_deg, d0, d1 = s
        if up_deg >= -0.2 or lo_deg >= -0.2:
            return False
        if d1 <= d0 * 1.05:
            return False
        # Lower boundary falls faster (more negative) -> broadening.
        return lo_deg < up_deg - 0.1

    out["broadening_wedges_ascending"] = _DerivedScanner("broadening_wedges_ascending", bw_base, keep_if=_keep_bw_ascending)
    out["broadening_wedges_descending"] = _DerivedScanner("broadening_wedges_descending", bw_base, keep_if=_keep_bw_descending)

    # Chapter 7 + 8: bump-and-run reversal bottom/top
    barr = library.load("bump_and_run_reversal") if "bump_and_run_reversal" in library.list_keys() else None
    if isinstance(barr, dict):
        out["bump_and_run_reversal_tops"] = PivotSequenceScanner("bump_and_run_reversal_tops", barr)

        barr_b = copy.deepcopy(barr)
        barr_b["pattern_type"] = "reversal_bullish"
        barr_b.setdefault("prior_trend_requirements", {})
        barr_b["prior_trend_requirements"]["direction"] = "down"
        barr_b.setdefault("breakout_confirmation", {})
        barr_b["breakout_confirmation"]["breakout_direction"] = "up"
        ds = barr_b.get("detection_signature", {}) or {}
        seq = (ds.get("pivot_sequence") or []) if isinstance(ds.get("pivot_sequence"), list) else []
        inv = []
        for t in seq:
            if t == "H":
                inv.append("L")
            elif t == "L":
                inv.append("H")
            else:
                inv.append(t)
        barr_b.setdefault("detection_signature", {})
        barr_b["detection_signature"]["pivot_sequence"] = inv
        out["bump_and_run_reversal_bottoms"] = PivotSequenceScanner("bump_and_run_reversal_bottoms", barr_b)

    # Chapter 9 + 10: cup with handle (+ inverted)
    cwh = library.load("cup_with_handle") if "cup_with_handle" in library.list_keys() else None
    if isinstance(cwh, dict):
        out["cup_with_handle"] = CupWithHandleScanner("cup_with_handle", cwh)
        out["cup_with_handle_inverted"] = InvertedCupWithHandleScanner("cup_with_handle_inverted", cwh)

    # Chapter 11 + 12: diamonds
    db = library.load("diamond_bottom") if "diamond_bottom" in library.list_keys() else None
    if isinstance(db, dict):
        out["diamond_bottoms"] = PivotSequenceScanner("diamond_bottoms", db)
    dt = library.load("diamond_top") if "diamond_top" in library.list_keys() else None
    if isinstance(dt, dict):
        out["diamond_tops"] = PivotSequenceScanner("diamond_tops", dt)

    # Chapter 13-20: double bottoms/tops by Adam/Eve variant
    if _get("double_bottoms"):
        base = _CachedScanner(_get("double_bottoms"))

        def _db_variant(r: Dict[str, Any], df: pd.DataFrame) -> Optional[str]:
            piv = r.get("pivot_indices") or []
            if not isinstance(piv, (list, tuple)) or len(piv) < 3:
                return None
            try:
                b1, b2 = int(piv[0]), int(piv[2])
            except Exception:
                return None
            if not (0 <= b1 < len(df) and 0 <= b2 < len(df)):
                return None
            p1 = float(df.iloc[b1]["low"])
            p2 = float(df.iloc[b2]["low"])
            w1 = _trough_width_bars(df, b1, p1)
            w2 = _trough_width_bars(df, b2, p2)
            t1 = _ae(w1)
            t2 = _ae(w2)
            return f"{t1}{t2}" if (t1 and t2) else None

        out["double_bottoms_adam_adam"] = _DerivedScanner("double_bottoms_adam_adam", base, keep_if=lambda r, df, *_: _db_variant(r, df) == "AA")
        out["double_bottoms_adam_eve"] = _DerivedScanner("double_bottoms_adam_eve", base, keep_if=lambda r, df, *_: _db_variant(r, df) == "AE")
        out["double_bottoms_eve_adam"] = _DerivedScanner("double_bottoms_eve_adam", base, keep_if=lambda r, df, *_: _db_variant(r, df) == "EA")
        out["double_bottoms_eve_eve"] = _DerivedScanner("double_bottoms_eve_eve", base, keep_if=lambda r, df, *_: _db_variant(r, df) == "EE")

    if _get("double_tops"):
        base = _CachedScanner(_get("double_tops"))

        def _dt_variant(r: Dict[str, Any], df: pd.DataFrame) -> Optional[str]:
            piv = r.get("pivot_indices") or []
            if not isinstance(piv, (list, tuple)) or len(piv) < 3:
                return None
            try:
                p1i, p2i = int(piv[0]), int(piv[2])
            except Exception:
                return None
            if not (0 <= p1i < len(df) and 0 <= p2i < len(df)):
                return None
            p1 = float(df.iloc[p1i]["high"])
            p2 = float(df.iloc[p2i]["high"])
            w1 = _peak_width_bars(df, p1i, p1)
            w2 = _peak_width_bars(df, p2i, p2)
            t1 = _ae(w1)
            t2 = _ae(w2)
            return f"{t1}{t2}" if (t1 and t2) else None

        out["double_tops_adam_adam"] = _DerivedScanner("double_tops_adam_adam", base, keep_if=lambda r, df, *_: _dt_variant(r, df) == "AA")
        out["double_tops_adam_eve"] = _DerivedScanner("double_tops_adam_eve", base, keep_if=lambda r, df, *_: _dt_variant(r, df) == "AE")
        out["double_tops_eve_adam"] = _DerivedScanner("double_tops_eve_adam", base, keep_if=lambda r, df, *_: _dt_variant(r, df) == "EA")
        out["double_tops_eve_eve"] = _DerivedScanner("double_tops_eve_eve", base, keep_if=lambda r, df, *_: _dt_variant(r, df) == "EE")

    # Chapter 21-22: flags (+ high & tight)
    if _get("flags"):
        base = _CachedScanner(_get("flags"))

        def _is_high_tight(r: Dict[str, Any], df: pd.DataFrame, *_: Any) -> bool:
            if str(r.get("breakout_direction") or "") != "up":
                return False
            piv = r.get("pivot_indices") or []
            if not isinstance(piv, (list, tuple)) or not piv:
                return False
            try:
                start = int(min(int(x) for x in piv if x is not None))
                end = int(max(int(x) for x in piv if x is not None))
            except Exception:
                return False
            if start < 0 or end >= len(df):
                return False

            form = df.iloc[start : end + 1]
            if len(form) == 0:
                return False
            form_high = float(form["high"].max())
            form_low = float(form["low"].min())
            if form_high <= 0:
                return False

            # Look back up to 60 bars for pole start (pragmatic).
            lb = max(0, start - 60)
            pre = df.iloc[lb : start + 1]
            if len(pre) == 0:
                return False
            pole_low = float(pre["low"].min())
            try:
                pole_low_vals = pre["low"].to_numpy(dtype=float, copy=False)
                if pole_low_vals.size == 0 or not np.isfinite(pole_low_vals).any():
                    return False
                pole_low_rel = int(np.nanargmin(pole_low_vals))
                pole_low_idx = int(lb + pole_low_rel)
            except Exception:
                return False
            if pole_low <= 0:
                return False
            pole_gain = (form_high - pole_low) / pole_low * 100.0
            pole_dur = int(start - pole_low_idx + 1)
            if pole_gain < 100.0 or pole_dur > 60:
                return False

            dd = (form_high - form_low) / form_high * 100.0
            return dd <= 20.0

        out["flags_high_tight"] = _DerivedScanner("flags_high_tight", base, keep_if=_is_high_tight)
        out["flags"] = _DerivedScanner("flags", base, keep_if=lambda r, df, *_: not _is_high_tight(r, df))

    # Chapter 23: gaps
    if _get("gaps"):
        out["gaps"] = _get("gaps")

    # Chapter 24-27: head & shoulders (standard vs complex)
    if _get("head_and_shoulders_bottom"):
        base = _CachedScanner(_get("head_and_shoulders_bottom"))
        hs_spec = library.load("head_and_shoulders_bottom") if "head_and_shoulders_bottom" in library.list_keys() else {}
        tol = float((hs_spec.get("geometry_constraints", {}) or {}).get("near_equal_tolerance_pct") or 3.0)

        def _hsb_complex(r: Dict[str, Any], df: pd.DataFrame, _pf: List[Pivot], pr: List[Pivot]) -> bool:
            piv = r.get("pivot_indices") or []
            if not isinstance(piv, (list, tuple)) or len(piv) < 5:
                return False
            try:
                l1, l2, l3 = int(piv[0]), int(piv[2]), int(piv[4])
            except Exception:
                return False
            if not (0 <= l1 < len(df) and 0 <= l2 < len(df) and 0 <= l3 < len(df)):
                return False
            shoulder_level = (float(df.iloc[l1]["low"]) + float(df.iloc[l3]["low"])) / 2.0
            head = float(df.iloc[l2]["low"])
            if shoulder_level <= 0:
                return False

            lows = [p for p in (pr or _pf) if p.type == PivotType.LOW and l1 <= int(p.idx) <= l3]
            count = 0
            for p in lows:
                if int(p.idx) in (l1, l2, l3):
                    continue
                if float(p.price) <= head * 1.02:
                    continue
                if _pct(float(p.price), shoulder_level) <= tol:
                    count += 1
            return count >= 1

        out["head_and_shoulders_bottoms_complex"] = _DerivedScanner("head_and_shoulders_bottoms_complex", base, keep_if=_hsb_complex)
        out["head_and_shoulders_bottoms"] = _DerivedScanner("head_and_shoulders_bottoms", base, keep_if=lambda r, df, pf, pr: not _hsb_complex(r, df, pf, pr))

    if _get("head_and_shoulders_top"):
        base = _CachedScanner(_get("head_and_shoulders_top"))
        hs_spec = library.load("head_and_shoulders_top") if "head_and_shoulders_top" in library.list_keys() else {}
        tol = float((hs_spec.get("geometry_constraints", {}) or {}).get("near_equal_tolerance_pct") or 3.0)

        def _hst_complex(r: Dict[str, Any], df: pd.DataFrame, _pf: List[Pivot], pr: List[Pivot]) -> bool:
            piv = r.get("pivot_indices") or []
            if not isinstance(piv, (list, tuple)) or len(piv) < 5:
                return False
            try:
                h1, h2, h3 = int(piv[0]), int(piv[2]), int(piv[4])
            except Exception:
                return False
            if not (0 <= h1 < len(df) and 0 <= h2 < len(df) and 0 <= h3 < len(df)):
                return False
            shoulder_level = (float(df.iloc[h1]["high"]) + float(df.iloc[h3]["high"])) / 2.0
            head = float(df.iloc[h2]["high"])
            if shoulder_level <= 0:
                return False

            highs = [p for p in (pr or _pf) if p.type == PivotType.HIGH and h1 <= int(p.idx) <= h3]
            count = 0
            for p in highs:
                if int(p.idx) in (h1, h2, h3):
                    continue
                if float(p.price) >= head * 0.98:
                    continue
                if _pct(float(p.price), shoulder_level) <= tol:
                    count += 1
            return count >= 1

        out["head_and_shoulders_tops_complex"] = _DerivedScanner("head_and_shoulders_tops_complex", base, keep_if=_hst_complex)
        out["head_and_shoulders_tops"] = _DerivedScanner("head_and_shoulders_tops", base, keep_if=lambda r, df, pf, pr: not _hst_complex(r, df, pf, pr))

    # Chapter 28-29: horns
    if _get("horn_bottoms_tops"):
        base = _CachedScanner(_get("horn_bottoms_tops"))
        out["horn_bottoms"] = _DerivedScanner("horn_bottoms", base, keep_if=lambda r, *_: str(r.get("breakout_direction") or "") == "up")
        out["horn_tops"] = _DerivedScanner("horn_tops", base, keep_if=lambda r, *_: str(r.get("breakout_direction") or "") == "down")

    # Chapter 30-31: islands (regular vs long)
    islands_spec = library.load("islands") if "islands" in library.list_keys() else None
    if isinstance(islands_spec, dict):
        spec_long = copy.deepcopy(islands_spec)
        spec_long.setdefault("geometry_constraints", {})
        spec_long["geometry_constraints"]["width_max_bars"] = 40
        spec_long.setdefault("duration_constraints", {})
        spec_long["duration_constraints"]["max_bars"] = 42
        isl_base = _CachedScanner(IslandScanner("__islands_base", spec_long))

        def _is_long_island(r: Dict[str, Any], *_: Any) -> bool:
            try:
                return int(r.get("pattern_width_bars") or 0) > 10
            except Exception:
                return False

        out["islands_long"] = _DerivedScanner("islands_long", isl_base, keep_if=_is_long_island)
        out["island_reversals"] = _DerivedScanner("island_reversals", isl_base, keep_if=lambda r, *_: not _is_long_island(r))

    # Chapter 32-33: measured moves
    if _get("measured_move_down_up"):
        base = _CachedScanner(_get("measured_move_down_up"))
        out["measured_move_down"] = _DerivedScanner("measured_move_down", base, keep_if=lambda r, *_: str(r.get("breakout_direction") or "") == "down")
        out["measured_move_up"] = _DerivedScanner("measured_move_up", base, keep_if=lambda r, *_: str(r.get("breakout_direction") or "") == "up")

    # Chapter 34: pennants
    if _get("pennants"):
        out["pennants"] = _get("pennants")

    # Chapter 35-36: pipes
    pb = library.load("pipe_bottoms") if "pipe_bottoms" in library.list_keys() else None
    if isinstance(pb, dict):
        out["pipe_bottoms"] = PipeBottomScanner("pipe_bottoms", pb)
        pt = copy.deepcopy(pb)
        pt["pattern_type"] = "reversal_bearish"
        pt.setdefault("prior_trend_requirements", {})
        pt["prior_trend_requirements"]["direction"] = "up"
        pt.setdefault("breakout_confirmation", {})
        pt["breakout_confirmation"]["breakout_direction"] = "down"
        out["pipe_tops"] = PipeTopScanner("pipe_tops", pt)

    # Chapter 37-38: rectangles
    if _get("rectangle_bottoms_tops"):
        base = _CachedScanner(_get("rectangle_bottoms_tops"))
        out["rectangle_bottoms"] = _DerivedScanner("rectangle_bottoms", base, keep_if=lambda r, *_: str(r.get("breakout_direction") or "") == "up")
        out["rectangle_tops"] = _DerivedScanner("rectangle_tops", base, keep_if=lambda r, *_: str(r.get("breakout_direction") or "") == "down")

    # Chapter 39-40: rounding bottoms/tops
    if _get("rounding_bottoms_tops"):
        base = _CachedScanner(_get("rounding_bottoms_tops"))
        out["rounding_bottoms"] = _DerivedScanner("rounding_bottoms", base, keep_if=lambda r, *_: str(r.get("pattern_type") or "") == "reversal_bullish")
        out["rounding_tops"] = _DerivedScanner("rounding_tops", base, keep_if=lambda r, *_: str(r.get("pattern_type") or "") == "reversal_bearish")

    # Chapter 41-44: scallops (ascending/descending + inverted)
    if _get("scallop_ascending_descending"):
        base = _CachedScanner(_get("scallop_ascending_descending"))

        def _sc_dir(r: Dict[str, Any], df: pd.DataFrame) -> Optional[str]:
            piv = r.get("pivot_indices") or []
            if not isinstance(piv, (list, tuple)) or len(piv) < 3:
                return None
            try:
                s, e = int(piv[0]), int(piv[2])
            except Exception:
                return None
            if not (0 <= s < len(df) and 0 <= e < len(df)):
                return None
            s_low = float(df.iloc[s]["low"])
            e_low = float(df.iloc[e]["low"])
            return "ascending" if e_low > s_low else "descending"

        def _keep_sc(name: str):
            def _f(r: Dict[str, Any], df: pd.DataFrame, *_: Any) -> bool:
                d = _sc_dir(r, df)
                bo = str(r.get("breakout_direction") or "")
                if d == "ascending" and bo == "up":
                    return name == "scallops_ascending"
                if d == "ascending" and bo == "down":
                    return name == "scallops_ascending_inverted"
                if d == "descending" and bo == "down":
                    return name == "scallops_descending"
                if d == "descending" and bo == "up":
                    return name == "scallops_descending_inverted"
                return False

            return _f

        out["scallops_ascending"] = _DerivedScanner("scallops_ascending", base, keep_if=_keep_sc("scallops_ascending"))
        out["scallops_ascending_inverted"] = _DerivedScanner("scallops_ascending_inverted", base, keep_if=_keep_sc("scallops_ascending_inverted"))
        out["scallops_descending"] = _DerivedScanner("scallops_descending", base, keep_if=_keep_sc("scallops_descending"))
        out["scallops_descending_inverted"] = _DerivedScanner("scallops_descending_inverted", base, keep_if=_keep_sc("scallops_descending_inverted"))

    # Chapter 45-46: three falling peaks / three rising valleys
    tfp_spec = library.load("three_falling_peaks") if "three_falling_peaks" in library.list_keys() else None
    if not isinstance(tfp_spec, dict):
        tfp_spec = {
            "pattern_name": "Three Falling Peaks",
            "pattern_type": "reversal_bearish",
            "digitization_version": "builtin_bulkowski_53_v1",
            "detection_signature": {
                "pivot_sequence": ["H", "L", "H", "L", "H"],
                "pivot_order": "alternating",
                "mandatory_pivots": [
                    {"position": 3, "type": "H", "constraint": "lower_than_previous"},
                    {"position": 5, "type": "H", "constraint": "lower_than_previous"},
                ],
            },
            "geometry_constraints": {"width_min_bars": 42, "width_max_bars": 270, "height_ratio_min": 6.0, "height_ratio_max": 80.0, "near_equal_tolerance_pct": 3.0},
            "prior_trend_requirements": {"direction": "up", "min_period_bars": 21, "min_change_pct": 10.0},
            "breakout_confirmation": {"breakout_direction": "down", "breakout_threshold_pct": 1.0, "confirmation_bars": 1, "close_beyond_required": True},
        }

    trv_spec = library.load("three_rising_valleys") if "three_rising_valleys" in library.list_keys() else None
    if not isinstance(trv_spec, dict):
        trv_spec = {
            "pattern_name": "Three Rising Valleys",
            "pattern_type": "reversal_bullish",
            "digitization_version": "builtin_bulkowski_53_v1",
            "detection_signature": {
                "pivot_sequence": ["L", "H", "L", "H", "L"],
                "pivot_order": "alternating",
                "mandatory_pivots": [
                    {"position": 3, "type": "L", "constraint": "higher_than_previous"},
                    {"position": 5, "type": "L", "constraint": "higher_than_previous"},
                ],
            },
            "geometry_constraints": {"width_min_bars": 42, "width_max_bars": 270, "height_ratio_min": 6.0, "height_ratio_max": 80.0, "near_equal_tolerance_pct": 3.0},
            "prior_trend_requirements": {"direction": "down", "min_period_bars": 21, "min_change_pct": 10.0},
            "breakout_confirmation": {"breakout_direction": "up", "breakout_threshold_pct": 1.0, "confirmation_bars": 1, "close_beyond_required": True},
        }
    out["three_falling_peaks"] = PivotSequenceScanner("three_falling_peaks", tfp_spec)
    out["three_rising_valleys"] = PivotSequenceScanner("three_rising_valleys", trv_spec)

    # Chapter 47-49: triangles (ascending/descending/symmetrical)
    tri_spec = library.load("triangles") if "triangles" in library.list_keys() else None
    if isinstance(tri_spec, dict):
        tri_any = copy.deepcopy(tri_spec)
        tri_any.setdefault("detection_signature", {})
        tri_any["detection_signature"]["mandatory_pivots"] = []
        base = _CachedScanner(PivotSequenceScanner("__triangles_base", tri_any))

        tol = float((tri_spec.get("geometry_constraints", {}) or {}).get("near_equal_tolerance_pct") or 2.0)

        def _tri_class(r: Dict[str, Any], df: pd.DataFrame) -> Optional[str]:
            piv = r.get("pivot_indices") or []
            if not isinstance(piv, (list, tuple)) or len(piv) < 5:
                return None
            try:
                idxs = [int(x) for x in piv[:5]]
            except Exception:
                return None
            if any(i < 0 or i >= len(df) for i in idxs):
                return None
            h1 = float(df.iloc[idxs[0]]["high"])
            l1 = float(df.iloc[idxs[1]]["low"])
            h2 = float(df.iloc[idxs[2]]["high"])
            l2 = float(df.iloc[idxs[3]]["low"])
            h3 = float(df.iloc[idxs[4]]["high"])

            highs_near = (_pct(h2, h1) <= tol) and (_pct(h3, h1) <= tol)
            lows_near = _pct(l2, l1) <= tol
            highs_fall = (h2 < h1) and (h3 < h2)
            lows_rise = l2 > l1

            if highs_near and lows_rise:
                return "ascending"
            if lows_near and highs_fall:
                return "descending"
            if highs_fall and lows_rise:
                return "symmetrical"
            return None

        out["triangles_ascending"] = _DerivedScanner("triangles_ascending", base, keep_if=lambda r, df, *_: _tri_class(r, df) == "ascending")
        out["triangles_descending"] = _DerivedScanner("triangles_descending", base, keep_if=lambda r, df, *_: _tri_class(r, df) == "descending")
        out["triangles_symmetrical"] = _DerivedScanner("triangles_symmetrical", base, keep_if=lambda r, df, *_: _tri_class(r, df) == "symmetrical")

    # Chapter 50-51: triple bottoms/tops
    tbt = library.load("triple_bottoms_tops") if "triple_bottoms_tops" in library.list_keys() else None
    if isinstance(tbt, dict):
        ds = tbt.get("detection_signature", {}) or {}
        bo = tbt.get("breakout_confirmation", {}) or {}
        tb = ds.get("triple_bottom")
        tt = ds.get("triple_top")
        if isinstance(tb, dict) and isinstance(bo.get("triple_bottom"), dict):
            spec = copy.deepcopy(tbt)
            spec["pattern_type"] = "reversal_bullish"
            spec["detection_signature"] = tb
            spec["breakout_confirmation"] = bo["triple_bottom"]
            out["triple_bottoms"] = PivotSequenceScanner("triple_bottoms", spec)
        if isinstance(tt, dict) and isinstance(bo.get("triple_top"), dict):
            spec = copy.deepcopy(tbt)
            spec["pattern_type"] = "reversal_bearish"
            spec["detection_signature"] = tt
            spec["breakout_confirmation"] = bo["triple_top"]
            out["triple_tops"] = PivotSequenceScanner("triple_tops", spec)

    # Chapter 52-53: wedges (falling/rising)
    wedge_spec = library.load("wedges_ascending_descending") if "wedges_ascending_descending" in library.list_keys() else None
    if isinstance(wedge_spec, dict):
        w_any = copy.deepcopy(wedge_spec)
        w_any.setdefault("detection_signature", {})
        w_any["detection_signature"]["mandatory_pivots"] = []
        base = _CachedScanner(PivotSequenceScanner("__wedges_base", w_any))

        def _w_slopes(r: Dict[str, Any], df: pd.DataFrame) -> Optional[Tuple[float, float]]:
            piv = r.get("pivot_indices") or []
            if not isinstance(piv, (list, tuple)) or len(piv) < 6:
                return None
            try:
                idxs = [int(x) for x in piv[:6]]
            except Exception:
                return None
            hs = [idxs[i] for i in (0, 2, 4)]
            ls = [idxs[i] for i in (1, 3, 5)]
            if any(i < 0 or i >= len(df) for i in hs + ls):
                return None
            up0, up1 = hs[0], hs[-1]
            lo0, lo1 = ls[0], ls[-1]
            up_deg = _slope_degrees(up0, float(df.iloc[up0]["high"]), up1, float(df.iloc[up1]["high"]))
            lo_deg = _slope_degrees(lo0, float(df.iloc[lo0]["low"]), lo1, float(df.iloc[lo1]["low"]))
            return up_deg, lo_deg

        def _keep_w_falling(r: Dict[str, Any], df: pd.DataFrame, *_: Any) -> bool:
            s = _w_slopes(r, df)
            if s is None:
                return False
            up_deg, lo_deg = s
            return up_deg < -0.2 and lo_deg < -0.2 and str(r.get("breakout_direction") or "") == "up"

        def _keep_w_rising(r: Dict[str, Any], df: pd.DataFrame, *_: Any) -> bool:
            s = _w_slopes(r, df)
            if s is None:
                return False
            up_deg, lo_deg = s
            return up_deg > 0.2 and lo_deg > 0.2 and str(r.get("breakout_direction") or "") == "down"

        def _set_type(ptype: str):
            def _t(row: Dict[str, Any], *_: Any) -> Dict[str, Any]:
                row["pattern_type"] = ptype
                return row

            return _t

        out["wedges_falling"] = _DerivedScanner("wedges_falling", base, keep_if=_keep_w_falling, transform=_set_type("reversal_bullish"))
        out["wedges_rising"] = _DerivedScanner("wedges_rising", base, keep_if=_keep_w_rising, transform=_set_type("reversal_bearish"))

    # Return stable ordering for CLI/help usage.
    ordered_keys = [
        "broadening_bottoms",
        "broadening_formations_right_angled_ascending",
        "broadening_formations_right_angled_descending",
        "broadening_tops",
        "broadening_wedges_ascending",
        "broadening_wedges_descending",
        "bump_and_run_reversal_bottoms",
        "bump_and_run_reversal_tops",
        "cup_with_handle",
        "cup_with_handle_inverted",
        "diamond_bottoms",
        "diamond_tops",
        "double_bottoms_adam_adam",
        "double_bottoms_adam_eve",
        "double_bottoms_eve_adam",
        "double_bottoms_eve_eve",
        "double_tops_adam_adam",
        "double_tops_adam_eve",
        "double_tops_eve_adam",
        "double_tops_eve_eve",
        "flags",
        "flags_high_tight",
        "gaps",
        "head_and_shoulders_bottoms",
        "head_and_shoulders_bottoms_complex",
        "head_and_shoulders_tops",
        "head_and_shoulders_tops_complex",
        "horn_bottoms",
        "horn_tops",
        "island_reversals",
        "islands_long",
        "measured_move_down",
        "measured_move_up",
        "pennants",
        "pipe_bottoms",
        "pipe_tops",
        "rectangle_bottoms",
        "rectangle_tops",
        "rounding_bottoms",
        "rounding_tops",
        "scallops_ascending",
        "scallops_ascending_inverted",
        "scallops_descending",
        "scallops_descending_inverted",
        "three_falling_peaks",
        "three_rising_valleys",
        "triangles_ascending",
        "triangles_descending",
        "triangles_symmetrical",
        "triple_bottoms",
        "triple_tops",
        "wedges_falling",
        "wedges_rising",
    ]
    return {k: out[k] for k in ordered_keys if k in out}


def build_event_ohlcv_scanners() -> Dict[str, Any]:
    """
    Build a minimal OHLCV-only event-pattern scanner set.

    This intentionally includes only the event patterns that can be defined
    from price/volume alone (no external event database required).
    """

    dcb_spec = {
        "pattern_name": "Dead-Cat Bounce",
        "pattern_type": "event_bearish",
        "digitization_version": "builtin_event_ohlcv_v1",
        "event_constraints": {
            # Bulkowski: min event decline used in study (15%), usually higher; up to ~8 sessions.
            "event_decline_min_pct": 15.0,
            "event_decline_max_bars": 8,
            # Bulkowski: bounce recovery typically 15% to 35%, peaking 5 to 25 days.
            "bounce_min_pct": 15.0,
            "bounce_max_pct": 35.0,
            "bounce_min_bars": 5,
            "bounce_max_bars": 25,
            "gap_preferred": True,
        },
        "volume_constraints": {"event_volume_ratio_preferred": 2.0},
    }
    idcb_spec = {
        "pattern_name": "Dead-Cat Bounce, Inverted",
        "pattern_type": "event_bearish",
        "digitization_version": "builtin_event_ohlcv_v1",
        "event_constraints": {
            # Bulkowski: large 1-day upward move (>= 5% in the book's frequency distributions).
            "up_move_min_pct": 5.0,
            "gap_preferred": True,
        },
        "volume_constraints": {"event_volume_ratio_preferred": 2.0},
    }

    return {
        "dead_cat_bounce": DeadCatBounceScanner("dead_cat_bounce", dcb_spec),
        "dead_cat_bounce_inverted": DeadCatBounceInvertedScanner("dead_cat_bounce_inverted", idcb_spec),
    }


def build_bulkowski_55_ohlcv_scanners(library: DigitizedPatternLibrary) -> Dict[str, Any]:
    """
    Bulkowski Part One (53 chart patterns) + OHLCV-only event exceptions:
      - Dead-Cat Bounce
      - Dead-Cat Bounce, Inverted
    """

    out: Dict[str, Any] = {}
    out.update(build_bulkowski_53_scanners(library))
    out.update(build_event_ohlcv_scanners())
    return out


def build_bulkowski_53_strict_scanners(library: DigitizedPatternLibrary) -> Dict[str, Any]:
    """
    Strict-ish subset of Bulkowski Part One (53 chart-pattern chapters).

    Definition:
      - Keep only patterns that map to an existing digitized spec key (via `spec_key` mapping).
      - Exclude built-in proxy patterns that do not have a digitized spec anchor (e.g., some
        broadening sub-types and a few standalone proxy chapters).

    This is useful when you want research runs whose pattern definitions are tightly tied to
    the digitized spec library, while still keeping the Bulkowski chapter/variant keys.
    """

    full = build_bulkowski_53_scanners(library)
    try:
        from .pattern_set_metadata import BULKOWSKI_53_META  # type: ignore
    except Exception:  # pragma: no cover
        from pattern_set_metadata import BULKOWSKI_53_META  # type: ignore

    strict: Dict[str, Any] = {}
    for k, scanner in full.items():
        meta = BULKOWSKI_53_META.get(str(k), {})
        if meta.get("spec_key"):
            strict[str(k)] = scanner
    return strict


def build_bulkowski_strict_ohlcv_scanners(library: DigitizedPatternLibrary) -> Dict[str, Any]:
    """
    Spec-anchored Bulkowski chart patterns + OHLCV-only event exceptions.

    Currently:
      bulkowski_53_strict (53) + event_ohlcv (2) = 55 patterns
    """

    out: Dict[str, Any] = {}
    out.update(build_bulkowski_53_strict_scanners(library))
    out.update(build_event_ohlcv_scanners())
    return out


def build_bulkowski_49_strict_ohlcv_scanners(library: DigitizedPatternLibrary) -> Dict[str, Any]:
    """
    Deprecated alias for `build_bulkowski_strict_ohlcv_scanners`.

    Kept for backward compatibility with older runs/scripts.
    """

    return build_bulkowski_strict_ohlcv_scanners(library)
