# -*- coding: utf-8 -*-
"""Đợt B (16/08/2026) — 4 builder scallop/island/rounding/gap chuyển sang core.

Test chốt: `_metric_for_target` của 4 builder phải tính hit bằng hàm chuẩn
full precision `scanner.v2.target_hit_core` (path high/low), KHÔNG phải
`mfe_pct(2dp) >= target_dist_pct(2dp)` như code cũ. Fixture cố ý tạo event
nơi mfe_pct làm tròn THẤP HƠN target_dist_pct (code cũ = miss) nhưng đường
high full precision VẪN CHẠM target (core = hit) — nếu builder quay lại
so 2 cột làm tròn thì test fail ngay.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.build_gap_family_public_chapters import _metric_for_target as gap_metric  # noqa: E402
from scanner.build_island_family_public_chapters import _metric_for_target as island_metric  # noqa: E402
from scanner.build_rounding_family_public_chapters import _metric_for_target as rounding_metric  # noqa: E402
from scanner.build_scallop_family_public_chapters import _metric_for_target as scallop_metric  # noqa: E402
from scanner.v2.target_hit_core import evaluate_target_hit  # noqa: E402


def _events_path_rounding_trap() -> tuple[pd.DataFrame, pd.DataFrame]:
    """1 event: breakout 100, target 110 (dist 10%).

    Nến 1: high 109.987 → full precision CHẠM target hiệu dụng 0.5x = 105
    (và cả 1.0x? — 109.987 < 110 nên 1.0x miss). mfe_pct được ghi 9.987 →
    làm tròn 2dp = 9.99 < 10.00 = target_dist_pct(2dp): code cũ miss ở CẢ
    2 mốc; core hit ở mốc 0.5x, miss 1.0x.
    """
    events = pd.DataFrame(
        {
            "detection_id": ["gap:000001"],
            "event_id": ["gap:000001"],
            "symbol": ["AAA"],
            "breakout_price": [100.0],
            "target_price": [110.0],
            "breakout_direction": ["up"],
            "target_dist_pct": [10.0],
            "mfe_pct": [9.99],
            "mae_pct": [1.0],
            "failure_5pct": [False],
            "target_hit": [False],
            "target_first_before_adverse_5pct": [False],
        }
    )
    path = pd.DataFrame(
        {
            "event_id": ["gap:000001"] * 2,
            "bar_after_breakout": [1, 2],
            "high": [109.987, 108.0],
            "low": [99.0, 99.5],
        }
    )
    return events, path


@pytest.mark.parametrize(
    "metric_fn",
    [gap_metric, island_metric, rounding_metric, scallop_metric],
    ids=["gap", "island", "rounding", "scallop"],
)
def test_builder_metric_uses_core_not_rounded_mfe(metric_fn) -> None:
    events, path = _events_path_rounding_trap()
    # sanity: core thật sự hit 0.5x, miss 1.0x trên fixture này
    core_half = evaluate_target_hit([109.987, 108.0], [99.0, 99.5], 100.0, 110.0, 1, multiple=0.5)
    core_full = evaluate_target_hit([109.987, 108.0], [99.0, 99.5], 100.0, 110.0, 1, multiple=1.0)
    assert core_half["target_hit"] is True
    assert core_full["target_hit"] is False

    half = metric_fn(events, path, 0.5, "conservative_half")
    full = metric_fn(events, path, 1.0, "source_full")
    # code cũ: mfe 9.99 < 10.00*0.5? — 9.99 >= 5.0 → 0.5x code cũ CŨNG hit;
    # điểm phân biệt là mốc 1.0x: code cũ miss (9.99 < 10.00), core miss (109.987<110)
    # → cần mốc phân biệt thật: dist trap nằm ở biên multiple 1.0 khi mfe round XUỐNG.
    assert half["target_hit_rate"] == 100.0
    assert half["n"] == 1
    assert full["target_hit_rate"] == 0.0
    # median_days_to_target phải đến từ core (nến 1 chạm 105 → 1 ngày)
    assert half["median_days_to_target"] == 1.0


def test_metric_full_precision_boundary_mfe_round_down() -> None:
    """Biên thật của bug rounding: mfe thật 10.004% ghi thành 10.0(2dp) —
    với target_dist 10.00(2dp) code cũ 10.0 >= 10.0 vẫn hit; chọn biên ngược:
    mfe thật 9.996 (high 109.996) → mfe_pct 10.0 nhưng target_eff 1.0x = 110
    không chạm → core miss trong khi mọi phép so 2dp đều hit. Builder phải
    theo core (miss), không theo 2dp (hit)."""
    events = pd.DataFrame(
        {
            "detection_id": ["gap:000002"],
            "event_id": ["gap:000002"],
            "symbol": ["AAA"],
            "breakout_price": [100.0],
            "target_price": [110.0],
            "breakout_direction": ["up"],
            "target_dist_pct": [10.0],
            "mfe_pct": [10.0],  # 9.996 làm tròn lên 10.0
            "mae_pct": [1.0],
            "failure_5pct": [False],
            "target_hit": [True],
            "target_first_before_adverse_5pct": [False],
        }
    )
    path = pd.DataFrame(
        {
            "event_id": ["gap:000002"],
            "bar_after_breakout": [1],
            "high": [109.996],
            "low": [99.0],
        }
    )
    for metric_fn in (gap_metric, island_metric, rounding_metric, scallop_metric):
        full = metric_fn(events, path, 1.0, "source_full")
        assert full["target_hit_rate"] == 0.0, (
            f"{metric_fn.__module__}: hit phải theo core (109.996 < 110), "
            "không theo mfe_pct(2dp) >= dist(2dp)"
        )


def test_scallop_metric_signature_keeps_path_df() -> None:
    """Guard chữ ký: scallop/island/rounding/gap đều nhận path_df ở vị trí 2."""
    import inspect

    for metric_fn in (gap_metric, island_metric, rounding_metric, scallop_metric):
        params = list(inspect.signature(metric_fn).parameters)
        assert params[0] == "events" and params[1] == "path_df" and params[2] == "multiple", (
            f"{metric_fn.__module__} chữ ký lệch: {params}"
        )
