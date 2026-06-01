from __future__ import annotations

import json
from pathlib import Path

from scanner.run_realtime_scan_watchlist import build_realtime_scan_plan, build_watchlist_from_artifacts


def test_realtime_scan_plan_covers_current_final_pattern_set() -> None:
    plan = build_realtime_scan_plan()

    pattern_ids = {row["pattern_id"] for row in plan["jobs"]}

    manifest = json.loads(Path("artifacts/final_chapters/final_chapters_manifest.json").read_text(encoding="utf-8"))
    assert len(pattern_ids) == len(manifest["chapters"])
    assert "bull_flags" in pattern_ids
    assert "double_bottoms_adam_adam" in pattern_ids
    assert "head_and_shoulders_tops" in pattern_ids
    assert all(row["refresh_command"] for row in plan["jobs"])


def test_realtime_watchlist_uses_available_event_artifacts() -> None:
    plan = build_realtime_scan_plan(patterns=["bull_flags"])
    watchlist = build_watchlist_from_artifacts(plan, lookback_days=30)

    assert set(watchlist.columns).issuperset({"pattern_id", "symbol", "event_date", "quality_tier"})
    assert Path(plan["jobs"][0]["event_source"]).exists()
