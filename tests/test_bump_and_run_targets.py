"""Fixture test BARR bottom target — sách ECP Table 7.8 (đợt A round-2, Sol NO-GO 15/08/2026).

Fixture đóng băng từ dữ liệu thật: 8 events của
artifacts/scanner_v2/bump_and_run_family/bump_and_run_reversal_bottoms/db_active/events.csv
+ OHLC daily normalized từ market_cache/stock_ohlcv/latest.sqlite.

Quy tắc sách: "The highest high in the pattern is the target" và pattern bắt đầu
tại OLD HIGH (đỉnh trước đà giảm lead-in — "the old high (which is the start of
the formation)"). old_high_idx = argmax(high) trong 60 nến trước lead_start
(cửa sổ chốt bằng đo 2.116 events: 100% có old_high xác định, p50 dist=33, p90=60;
xem _meta trong fixtures/barr_bottom_targets.json).

6 events đầu: đỉnh cũ XA lead_start (dist 45-60) — code cũ (max high
[lead_start-2 .. bump]) cho target THẤP hơn sách → fixture FAIL với code cũ.
2 events cuối: đỉnh cũ TRÙNG pattern — target không đổi (bảo vệ case ổn định).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scanner.v2.bump_and_run import (
    BARR_OLD_HIGH_LOOKBACK_BARS,
    barr_bottom_target,
)

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/barr_bottom_targets.json"


def _load_fixture() -> dict:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


def _frame(event: dict) -> pd.DataFrame:
    bars = pd.DataFrame(event["bars"])
    bars["date"] = pd.to_datetime(bars["date"])
    bars["symbol"] = event["symbol"]
    return bars.reset_index(drop=True)


def test_barr_bottom_target_matches_book_on_frozen_real_events() -> None:
    data = _load_fixture()
    assert BARR_OLD_HIGH_LOOKBACK_BARS == 60  # cửa sổ chốt từ đo 2.116 events (xem _meta)
    for event in data["events"]:
        df = _frame(event)
        ls = int(event["lead_start_idx_in_segment"])
        bi = int(event["bump_idx_in_segment"])
        target, old_idx, old_high = barr_bottom_target(df, ls, bi)
        assert old_idx == int(event["expected_old_high_idx_in_segment"]), (
            f"{event['detection_id']}: old_high_idx {old_idx} != {event['expected_old_high_idx_in_segment']}"
        )
        assert abs(old_high - float(event["expected_old_high"])) < 1e-9, event["detection_id"]
        assert abs(target - float(event["expected_target"])) < 1e-9, (
            f"{event['detection_id']}: target {target} != sách {event['expected_target']}"
        )


def test_barr_bottom_far_old_high_raises_target_vs_legacy_window() -> None:
    """Các event đỉnh cũ xa: target sách PHẢI CAO hơn code cũ (max [ls-2..bump])
    — chính là các case mà fixture FAIL với code hiện tại trước khi sửa."""
    data = _load_fixture()
    for event in data["events"]:
        if float(event["target_current_code"]) == float(event["expected_target"]):
            continue  # 2 events trùng — kiểm ở test dưới
        assert float(event["expected_target"]) > float(event["target_current_code"]), event["detection_id"]
        df = _frame(event)
        ls = int(event["lead_start_idx_in_segment"])
        bi = int(event["bump_idx_in_segment"])
        target, _, _ = barr_bottom_target(df, ls, bi)
        legacy = float(df.iloc[max(0, ls - 2) : bi + 1]["high"].max())
        assert target > legacy, event["detection_id"]


def test_barr_bottom_old_high_inside_pattern_keeps_target_stable() -> None:
    """2 events đỉnh cũ nằm TRONG [ls-2..bump]: target sách == target code cũ —
    sửa không được làm thay đổi các case vốn đã đúng."""
    data = _load_fixture()
    checked = 0
    for event in data["events"]:
        if float(event["target_current_code"]) != float(event["expected_target"]):
            continue
        checked += 1
        df = _frame(event)
        ls = int(event["lead_start_idx_in_segment"])
        bi = int(event["bump_idx_in_segment"])
        target, old_idx, old_high = barr_bottom_target(df, ls, bi)
        assert abs(target - float(event["expected_target"])) < 1e-9, event["detection_id"]
        # old high có thể xa hơn 2 nến nhưng KHÔNG vượt đỉnh pattern
        # (bằng hoặc thấp hơn — target giữ nguyên)
        assert old_high <= target + 1e-9, event["detection_id"]
    assert checked == 2
