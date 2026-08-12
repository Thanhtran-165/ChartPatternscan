from __future__ import annotations

from scanner.run_buy_setup_scan_watchlist import (
    VN100_MARKET_GROUPS,
    _load_buy_setup_specs,
    dedupe_buy_setups,
    scan_buy_setups,
)


def test_buy_setup_specs_cover_buy_allowed_patterns() -> None:
    specs = _load_buy_setup_specs()
    pattern_ids = {spec.pattern_id for spec in specs}

    assert len(specs) == 15
    assert "bull_flags" in pattern_ids
    assert "measured_move_up" in pattern_ids
    assert "triangles_ascending" in pattern_ids
    assert all(spec.detector_family for spec in specs)
    assert next(spec for spec in specs if spec.pattern_id == "measured_move_up").family == "measured_move_family"


def test_buy_setup_scan_is_vn100_pre_breakout_only() -> None:
    setups, meta = scan_buy_setups(limit_per_pattern=2)

    assert meta["workflow_id"] == "buy_setup_scan_watchlist_v1"
    assert meta["pattern_count"] == 15
    assert meta["symbols_scanned"] == 100
    assert set(setups["market_group"]).issubset(VN100_MARKET_GROUPS)
    if not setups.empty:
        assert set(setups["buy_stage"]) == {"BUY_SETUP"}
        assert set(setups["is_confirmed_breakout"]) == {False}
        assert (setups["last_close"] < setups["trigger_price"]).all()


def test_buy_setup_deduped_output_keeps_one_operational_view_per_geometry() -> None:
    setups, _ = scan_buy_setups(patterns=["double_bottoms_adam_adam", "double_bottoms_adam_eve"], limit_per_pattern=5)
    deduped = dedupe_buy_setups(setups)

    assert len(deduped) <= len(setups)
    if not deduped.empty:
        keys = (
            deduped["symbol"].astype(str)
            + "|"
            + deduped["detector_family"].astype(str)
            + "|"
            + deduped["trigger_price"].round(2).astype(str)
        )
        assert keys.is_unique
