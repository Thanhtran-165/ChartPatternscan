from __future__ import annotations

import json

import pandas as pd

from scanner.double_pattern_utils import _classify_extreme_shape
from scanner.digitized_pattern_engine import (
    CupWithHandleScanner,
    HeadShouldersBottomFamilyScanner,
    InvertedCupWithHandleScanner,
    RoundingBottomsTopsScanner,
    ScallopFamilyScanner,
    _HeadShouldersFamilyScanner,
)


def test_double_gap_midpoint_width_stays_unresolved() -> None:
    result = _classify_extreme_shape(width=5, reaction_pct=10.0, adam_max=3, eve_min=7)
    assert result["label"] is None
    assert "gap_not_resolved" in result["evidence"]


def test_double_near_threshold_widths_still_resolve() -> None:
    near_adam = _classify_extreme_shape(width=4, reaction_pct=4.5, adam_max=3, eve_min=7)
    near_eve = _classify_extreme_shape(width=6, reaction_pct=3.0, adam_max=3, eve_min=7)
    assert near_adam["label"] == "A"
    assert near_eve["label"] == "E"


def test_hs_bottom_single_extra_shoulder_is_demoted_to_standard(monkeypatch) -> None:
    scanner = HeadShouldersBottomFamilyScanner("head_and_shoulders_bottom", {})

    def fake_classify(self, **kwargs):  # type: ignore[no-untyped-def]
        return {
            "variant_code": "complex",
            "variant_confidence": 76,
            "evidence": {
                "extra_shoulders_total": 1,
                "width_exceeds_standard_max": False,
            },
        }

    monkeypatch.setattr(_HeadShouldersFamilyScanner, "_classify_variant", fake_classify)
    result = scanner._classify_variant(row={}, metrics={}, pivots_filtered=[], pivots_raw=[])
    assert result["variant_code"] == "standard"
    assert result["variant_confidence"] == 68
    assert result["evidence"]["single_extra_demoted_to_standard"] is True


def test_scallop_descending_requires_stronger_bearish_shift() -> None:
    scanner = ScallopFamilyScanner("scallops", {})
    weak_metrics = {
        "sequence_tag": "HLH",
        "overall_shift_pct": -2.5,
        "left_share_pct": 60.0,
        "left_leg_deg": -28.0,
        "right_leg_deg": 36.0,
        "arc_excursion_pct": 82.0,
        "left_directional_ratio": 0.52,
        "right_directional_ratio": 0.55,
    }
    strong_metrics = dict(weak_metrics, overall_shift_pct=-6.5)

    assert scanner._resolve_variant(weak_metrics, breakout_direction="down") is None
    resolved = scanner._resolve_variant(strong_metrics, breakout_direction="down")
    assert resolved is not None
    assert resolved["variant_code"] == "scallops_descending"


def test_rounding_top_uses_tighter_second_pass_gate() -> None:
    scanner = RoundingBottomsTopsScanner(
        "rounding_bottoms_tops",
        {"detection_signature": {"pivot_sequence": ["L", "H", "L", "H", "L", "H"]}},
    )
    metrics = {
        "variant_tag": "top",
        "width_bars": 100,
        "center_pos_pct": 50.0,
        "span_balance_ratio": 1.3,
        "center_clearance_pct": 8.0,
        "fit_error_pct": 15.0,
        "monotonic_left_ratio": 0.6,
        "monotonic_right_ratio": 0.6,
        "curvature_coeff": -1.0,
        "expected_curvature_sign": -1.0,
    }
    assert scanner._family_metrics_ok(metrics) is False

    metrics["width_bars"] = 90
    assert scanner._family_metrics_ok(metrics) is True


def test_inverted_cup_rejects_handle_that_rebounds_too_high(monkeypatch) -> None:
    scanner = InvertedCupWithHandleScanner("cup_with_handle_inverted", {})
    df = pd.DataFrame(
        {
            "open": [101.0, 108.0, 118.0, 110.0, 101.0, 106.0, 95.0],
            "high": [103.0, 112.0, 120.0, 115.0, 103.0, 110.0, 96.0],
            "low": [100.0, 102.0, 110.0, 105.0, 100.0, 102.0, 93.0],
            "close": [102.0, 110.0, 118.0, 108.0, 101.0, 104.0, 94.0],
        }
    )

    row = {
        "pivot_indices": [0, 2, 4, 5, 5],
        "breakout_idx": 6,
        "family_metrics_json": json.dumps(
            {"handle_slope_pct": 0.0, "breakout_lag_bars": 1, "rim_diff_pct": 5.0},
            sort_keys=True,
        ),
    }

    monkeypatch.setattr(CupWithHandleScanner, "scan", lambda self, **kwargs: [row])
    result = scanner.scan(symbol="XYZ", df=df, pivots_filtered=[], pivots_raw=[])
    assert result == []


def test_inverted_cup_keeps_original_space_metrics_for_survivor(monkeypatch) -> None:
    scanner = InvertedCupWithHandleScanner("cup_with_handle_inverted", {})
    df = pd.DataFrame(
        {
            "open": [101.0, 108.0, 118.0, 110.0, 101.0, 104.0, 95.0],
            "high": [103.0, 112.0, 120.0, 115.0, 103.0, 106.0, 96.0],
            "low": [100.0, 102.0, 110.0, 105.0, 100.0, 102.0, 93.0],
            "close": [102.0, 110.0, 118.0, 108.0, 101.0, 104.0, 94.0],
        }
    )

    row = {
        "pivot_indices": [0, 2, 4, 5, 5],
        "breakout_idx": 6,
        "family_metrics_json": json.dumps(
            {"handle_slope_pct": 0.0, "breakout_lag_bars": 1, "rim_diff_pct": 5.0},
            sort_keys=True,
        ),
    }

    monkeypatch.setattr(CupWithHandleScanner, "scan", lambda self, **kwargs: [dict(row)])
    result = scanner.scan(symbol="XYZ", df=df, pivots_filtered=[], pivots_raw=[])
    assert len(result) == 1
    family_metrics = json.loads(result[0]["family_metrics_json"])
    assert family_metrics["orig_handle_rebound_pct"] == 6.0
    assert family_metrics["orig_handle_ceiling_pct"] == 30.0
    assert result[0]["breakout_direction"] == "down"
