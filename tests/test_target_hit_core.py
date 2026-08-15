from __future__ import annotations

import pandas as pd
import pytest

from scanner.v2.target_hit_core import (
    effective_target_price,
    enrich_events_with_target_hit,
    evaluate_target_hit,
    target_hit_stats,
)


def test_effective_target_price_interpolates_linearly() -> None:
    assert effective_target_price(100.0, 120.0, 1.0) == pytest.approx(120.0)
    assert effective_target_price(100.0, 120.0, 0.5) == pytest.approx(110.0)
    assert effective_target_price(100.0, 80.0, 0.5) == pytest.approx(90.0)


def test_evaluate_up_direction_hit_and_adverse_order() -> None:
    highs = [101.0, 105.0, 112.0, 90.0]
    lows = [99.0, 100.0, 104.0, 88.0]
    res = evaluate_target_hit(highs, lows, 100.0, 110.0, 1)
    assert res["target_hit"] is True
    assert res["days_to_target"] == 3
    assert res["days_to_adverse"] == 4
    assert res["target_first_before_adverse"] is True


def test_evaluate_same_day_hit_and_adverse_is_not_first() -> None:
    highs = [112.0]
    lows = [94.0]  # cùng nến: hit lẫn adverse 5%
    res = evaluate_target_hit(highs, lows, 100.0, 110.0, 1)
    assert res["target_hit"] is True
    assert res["days_to_target"] == 1
    assert res["days_to_adverse"] == 1
    assert res["target_first_before_adverse"] is False


def test_evaluate_down_direction_uses_lows() -> None:
    highs = [99.0, 101.0]
    lows = [95.0, 88.0]
    res = evaluate_target_hit(highs, lows, 100.0, 90.0, -1)
    assert res["target_hit"] is True
    assert res["days_to_target"] == 2
    assert res["days_to_adverse"] is None
    assert res["target_first_before_adverse"] is True


def test_evaluate_multiple_half_target_needs_half_distance() -> None:
    highs = [104.0, 109.0]
    lows = [100.0, 100.0]
    res_10 = evaluate_target_hit(highs, lows, 100.0, 110.0, 1, multiple=1.0)
    res_05 = evaluate_target_hit(highs, lows, 100.0, 110.0, 1, multiple=0.5)
    assert res_10["target_hit"] is False  # max 109 < 110
    assert res_05["target_hit"] is True  # 105 đủ — có nến 109


def test_target_hit_stats_missing_path_keeps_legacy_false_nan() -> None:
    events = pd.DataFrame(
        {
            "detection_id": ["x:1", "x:2"],
            "breakout_price": [100.0, 100.0],
            "target_price": [110.0, 110.0],
            "breakout_direction": ["up", "up"],
        }
    )
    path = pd.DataFrame(
        {
            "event_id": ["x:1", "x:1", "x:2"],
            "bar_after_breakout": [2, 1, 1],
            "high": [111.0, 105.0, 100.5],
            "low": [99.0, 99.5, 99.0],
        }
    )
    hits, firsts, days = target_hit_stats(events, path, 1.0)
    assert hits == [True, False]
    assert firsts == [True, False]
    assert days[0] == 2.0  # sort theo bar_after_breakout — bar 2 là nến hit
    assert pd.isna(days[1])


def test_enrich_events_overwrites_three_columns() -> None:
    events = pd.DataFrame(
        {
            "detection_id": ["x:1"],
            "breakout_price": [100.0],
            "target_price": [110.0],
            "breakout_direction": ["up"],
            "target_hit": [False],
            "target_first_before_adverse_5pct": [False],
            "days_to_target": [float("nan")],
        }
    )
    path = pd.DataFrame(
        {"event_id": ["x:1"], "bar_after_breakout": [1], "high": [112.0], "low": [100.0]}
    )
    out = enrich_events_with_target_hit(events, path, 1.0)
    assert bool(out.iloc[0]["target_hit"]) is True
    assert float(out.iloc[0]["days_to_target"]) == 1.0
    assert "event_id" in out.columns
