"""Fixture test BARR bottom target — LOCAL PIVOT cơ chế (BLOCKER 1, đợt A2 16/08/2026).

Sách ECP Table 7.8: "The highest high in the pattern is the target"; pattern
bắt đầu tại OLD HIGH — "the old high (which is the start of the formation)".

Cơ chế (thay cơ chế argmax cửa sổ 60 của đợt A — Sol duyệt có điều kiện):
old high = pivot HIGH GẦN NHẤT trước lead_start neo được vào CÙNG đường
lead-in: fit mở rộng [pivot, lead_end] trên close vẫn đạt hướng dốc (slope<0)
và tiêu chuẩn fit của detector (r2 >= lead_in_min_r2). Duyệt pivot GẦN → XA
trong cap kỹ thuật 250 bars (cap KHÔNG phải định nghĩa). Không pivot hợp lệ →
event BỊ LOẠI (barr_bottom_target trả None) — không được thay bằng cực đại
cửa sổ.

Fixture 8 events thật (tests/fixtures/barr_bottom_targets.json):
- 6 events có pivot hợp lệ, dist 5/30/31/120/121/249 (đa dạng gần→xa)
- 2 events CÓ candidate pivots nhưng mọi pivot fail fit → no_valid_anchor

Với code đợt A (argmax cửa sổ 60) các test này FAIL — chữ ký và ngữ nghĩa khác
(chứng minh fail-trước bằng TypeError trước khi sửa code).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scanner.v2.bump_and_run import (
    BARR_OLD_HIGH_SEARCH_CAP_BARS,
    barr_bottom_target,
)

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/barr_bottom_targets.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _frame(event: dict) -> pd.DataFrame:
    bars = pd.DataFrame(event["bars"])
    bars["date"] = pd.to_datetime(bars["date"])
    bars["symbol"] = event["symbol"]
    return bars.reset_index(drop=True)


def test_barr_bottom_target_anchors_at_local_pivot_on_frozen_real_events() -> None:
    data = _load_fixture()
    assert BARR_OLD_HIGH_SEARCH_CAP_BARS == 250  # cap kỹ thuật, không phải định nghĩa
    checked_ok = 0
    for event in data["events"]:
        df = _frame(event)
        ls = int(event["lead_start_idx_in_segment"])
        le = int(event["lead_end_idx_in_segment"])
        bi = int(event["bump_idx_in_segment"])
        pivots = [int(i) for i in event["high_pivot_indices_in_segment"]]
        res = barr_bottom_target(df, ls, le, bi, pivots)
        if event["expected_status"] == "no_valid_anchor":
            assert res is None, (
                f"{event['detection_id']}: phải None (no_valid_anchor) nhưng được {res}"
            )
            continue
        checked_ok += 1
        target, old_idx, old_high = res
        assert old_idx == int(event["expected_old_high_idx_in_segment"]), event["detection_id"]
        assert abs(old_high - float(event["expected_old_high"])) < 1e-9, event["detection_id"]
        assert abs(target - float(event["expected_target"])) < 1e-9, event["detection_id"]
    assert checked_ok == 6


def test_barr_bottom_pivot_far_beyond_legacy_window_raises_target() -> None:
    """Events pivot cách lead_start > 60 (ngoài cửa sổ cũ): target pivot PHẢI cao
    hơn legacy max(high[lead_start-2 .. bump]) — đây là các case mà cơ chế cửa sổ
    60 của đợt A sai (bỏ sót old high thật)."""
    data = _load_fixture()
    checked = 0
    for event in data["events"]:
        if event["expected_status"] != "ok":
            continue
        ls = int(event["lead_start_idx_in_segment"])
        if ls - int(event["expected_old_high_idx_in_segment"]) <= 60:
            continue
        checked += 1
        df = _frame(event)
        le = int(event["lead_end_idx_in_segment"])
        bi = int(event["bump_idx_in_segment"])
        pivots = [int(i) for i in event["high_pivot_indices_in_segment"]]
        target, _, _ = barr_bottom_target(df, ls, le, bi, pivots)
        legacy = float(df.iloc[max(0, ls - 2) : bi + 1]["high"].max())
        assert target > legacy, event["detection_id"]
    assert checked >= 3  # dist 120/121/249


def test_barr_bottom_no_valid_anchor_is_rejected_not_window_max() -> None:
    """2 events có candidate pivots nhưng mọi pivot fail fit lead-in: kết quả
    PHẢI là None (caller loại event) — không được im lặng thay bằng cực đại
    cửa sổ (yêu cầu Sol BLOCKER 1)."""
    data = _load_fixture()
    checked = 0
    for event in data["events"]:
        if event["expected_status"] != "no_valid_anchor":
            continue
        checked += 1
        assert len(event["high_pivot_indices_in_segment"]) > 0, "fixture phải có candidates để test fit-fail"
        df = _frame(event)
        res = barr_bottom_target(
            df,
            int(event["lead_start_idx_in_segment"]),
            int(event["lead_end_idx_in_segment"]),
            int(event["bump_idx_in_segment"]),
            [int(i) for i in event["high_pivot_indices_in_segment"]],
        )
        assert res is None, event["detection_id"]
    assert checked == 2
