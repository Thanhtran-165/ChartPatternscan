"""Rào failure H2 (phán quyết chủ đầu tư 13/08/2026):
failure_busted_rate_pct ≤ 2×spec chuẩn Bulkowski (PATTERN_FAILURE_SPECS_PCT)
— pattern thực đo VN tệ hơn 2 lần chuẩn sách → hạ xuống draft kèm lý do.
Pattern không có spec / chưa đo failure → không gate (không bịa số)."""

from __future__ import annotations

import pandas as pd

from scanner.send_realtime_scan_email import (
    PATTERN_FAILURE_SPECS_PCT,
    _apply_v3_gates,
)


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _stats(**kwargs) -> dict:
    s = {"tier": 3, "n": 100, "median_target_dist_pct": 15.0, "failure_busted_rate_pct": 5.0}
    s.update(kwargs)
    return s


def test_spec_map_only_chuong_sach_co_so() -> None:
    """Spec map chỉ chứa pattern có số thật trong PDF_REVIEW (12 pattern ưu tiên)."""
    assert set(PATTERN_FAILURE_SPECS_PCT) == {
        "pipe_bottoms",
        "pipe_tops",
        "horn_bottoms",
        "horn_tops",
        "cup_with_handle",
        "scallops_ascending",
        "scallops_descending",
        "rectangle_bottoms",
        "rectangle_tops",
        "three_falling_peaks",
        "three_rising_valleys",
        "high_tight_flags",
    }


def test_failure_vuot_2x_spec_bi_day_xuong_draft() -> None:
    """cup_with_handle: spec 5% → rào 10%; thực đo 43.8% → draft kèm lý do."""
    df = _df([{"pattern_id": "cup_with_handle", "symbol": "AAA"}])
    stats = {"cup_with_handle": _stats(failure_busted_rate_pct=43.8)}
    qualified, draft = _apply_v3_gates(df, stats)
    assert qualified.empty
    assert len(draft) == 1
    assert "2×spec" in draft.iloc[0]["v3_gate_note"]


def test_failure_trong_2x_spec_van_qualified() -> None:
    """three_rising_valleys: spec 5% → rào 10%; thực đo 3.6% → qualified."""
    df = _df([{"pattern_id": "three_rising_valleys", "symbol": "AAA"}])
    stats = {"three_rising_valleys": _stats(failure_busted_rate_pct=3.6)}
    qualified, draft = _apply_v3_gates(df, stats)
    assert len(qualified) == 1
    assert draft.empty


def test_pattern_khong_co_spec_khong_bi_gate() -> None:
    """bull_flags không có spec trong PDF_REVIEW → failure cao không chặn (không bịa số)."""
    df = _df([{"pattern_id": "bull_flags", "symbol": "AAA"}])
    stats = {"bull_flags": _stats(failure_busted_rate_pct=80.0)}
    qualified, draft = _apply_v3_gates(df, stats)
    assert len(qualified) == 1
    assert draft.empty


def test_failure_chua_do_none_khong_gate() -> None:
    """failure_busted_rate_pct=None (chưa đo) → không gate."""
    df = _df([{"pattern_id": "pipe_bottoms", "symbol": "AAA"}])
    stats = {"pipe_bottoms": _stats(failure_busted_rate_pct=None)}
    qualified, draft = _apply_v3_gates(df, stats)
    assert len(qualified) == 1
    assert draft.empty


def test_high_tight_flags_spec_zero() -> None:
    """HTF: sách BE 0% → rào 0%; fail 0 → qualified, fail >0 → draft."""
    df = _df([{"pattern_id": "high_tight_flags", "symbol": "AAA"}])
    ok = _apply_v3_gates(df, {"high_tight_flags": _stats(failure_busted_rate_pct=0.0)})
    assert len(ok[0]) == 1 and ok[1].empty
    bad = _apply_v3_gates(df, {"high_tight_flags": _stats(failure_busted_rate_pct=0.5)})
    assert bad[0].empty and len(bad[1]) == 1


def test_profile_cu_khong_tier_giu_hanh_vi_cu() -> None:
    """Profile cũ/chưa build (không có field tier) → qualified hết, mail không rỗng."""
    df = _df([{"pattern_id": "cup_with_handle", "symbol": "AAA"}])
    qualified, draft = _apply_v3_gates(df, {"cup_with_handle": {"failure_busted_rate_pct": 99.0}})
    assert len(qualified) == 1
    assert draft.empty
