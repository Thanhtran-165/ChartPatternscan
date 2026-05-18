"""Broadening Bottoms V2 fixture-level detector.

This detector is intentionally small and deterministic. It exists to make the
first official V2 pattern auditable before we connect V2 to market-wide scans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BROADENING_BOTTOMS_SUPPORTED_RULE_IDS = {
    "bb.prior_trend.down",
    "bb.shape.megaphone",
    "bb.trendlines.diverge",
    "bb.touches.min_two_each",
    "bb.volume.context",
    "bb.breakout.close_either_side",
    "bb.measure.height_from_recent_extreme",
    "bb.invalidation.not_broadening",
}


@dataclass(frozen=True)
class V2Pivot:
    idx: int
    type: str
    price: float


@dataclass(frozen=True)
class V2Close:
    idx: int
    close: float


@dataclass(frozen=True)
class BroadeningBottomsResult:
    matched: bool
    breakout_direction: Optional[str]
    breakout_idx: Optional[int]
    breakout_price: Optional[float]
    reasons: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "breakout_direction": self.breakout_direction,
            "breakout_idx": self.breakout_idx,
            "breakout_price": self.breakout_price,
            "reasons": list(self.reasons),
        }


def _as_pivots(items: Sequence[Mapping[str, Any]]) -> List[V2Pivot]:
    out: List[V2Pivot] = []
    for item in items:
        out.append(
            V2Pivot(
                idx=int(item["idx"]),
                type=str(item["type"]).upper(),
                price=float(item["price"]),
            )
        )
    out.sort(key=lambda p: p.idx)
    return out


def _as_closes(items: Sequence[Mapping[str, Any]]) -> List[V2Close]:
    out: List[V2Close] = []
    for item in items:
        out.append(V2Close(idx=int(item["idx"]), close=float(item["close"])))
    out.sort(key=lambda c: c.idx)
    return out


def _strictly_increasing(values: Sequence[float]) -> bool:
    return all(b > a for a, b in zip(values, values[1:]))


def _strictly_decreasing(values: Sequence[float]) -> bool:
    return all(b < a for a, b in zip(values, values[1:]))


def _slope(points: Sequence[V2Pivot]) -> Optional[float]:
    if len(points) < 2:
        return None
    first = points[0]
    last = points[-1]
    dx = last.idx - first.idx
    if dx <= 0:
        return None
    return (last.price - first.price) / dx


class BroadeningBottomsV2Detector:
    """Evaluate the minimal official broadening-bottom rule set."""

    def scan_fixture(self, fixture: Mapping[str, Any]) -> BroadeningBottomsResult:
        prior = fixture.get("prior_trend")
        prior_direction = str((prior or {}).get("direction") or "").lower() if isinstance(prior, dict) else ""
        pivots = _as_pivots(fixture.get("pivots", []))
        closes = _as_closes(fixture.get("post_formation_closes", []))

        reasons: List[str] = []
        if prior_direction != "down":
            reasons.append("prior_trend_not_down")

        highs = [p for p in pivots if p.type == "H"]
        lows = [p for p in pivots if p.type == "L"]
        if len(highs) < 2 or len(lows) < 2:
            reasons.append("insufficient_high_low_touches")

        if len(highs) >= 2 and not _strictly_increasing([p.price for p in highs]):
            reasons.append("highs_not_rising")
        if len(lows) >= 2 and not _strictly_decreasing([p.price for p in lows]):
            reasons.append("lows_not_falling")

        high_slope = _slope(highs)
        low_slope = _slope(lows)
        if high_slope is None or high_slope <= 0:
            reasons.append("upper_trendline_not_up")
        if low_slope is None or low_slope >= 0:
            reasons.append("lower_trendline_not_down")

        if reasons:
            return BroadeningBottomsResult(False, None, None, None, tuple(reasons))

        formation_high = max(p.price for p in highs)
        formation_low = min(p.price for p in lows)
        for close in closes:
            if close.close > formation_high:
                return BroadeningBottomsResult(True, "up", close.idx, close.close, tuple())
            if close.close < formation_low:
                return BroadeningBottomsResult(True, "down", close.idx, close.close, tuple())

        return BroadeningBottomsResult(False, None, None, None, ("no_close_beyond_boundary",))


def run_broadening_bottoms_fixture(fixture: Mapping[str, Any]) -> BroadeningBottomsResult:
    return BroadeningBottomsV2Detector().scan_fixture(fixture)
