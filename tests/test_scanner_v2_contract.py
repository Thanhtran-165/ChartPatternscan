from __future__ import annotations

import copy

import pytest

from scanner.v2 import (
    CORE_PATTERN_KEYS,
    ContractError,
    ScannerV2Engine,
    canonical_spec_hash,
    load_core_registry,
    load_taxonomy_lineage,
    validate_official_pattern,
    validate_pattern_provenance,
    run_bull_flags_fixture,
    run_broadening_bottoms_fixture,
    verify_pattern_source_alignment,
)


def test_core_registry_has_required_provenance_fields() -> None:
    registry = load_core_registry()
    patterns = registry["patterns"]

    assert set(CORE_PATTERN_KEYS).issubset(patterns)
    for key in CORE_PATTERN_KEYS:
        assert validate_pattern_provenance(key, patterns[key]) == []


def test_core_patterns_compile_with_only_broadening_bottoms_official() -> None:
    engine = ScannerV2Engine()
    compiled = engine.compile_core_patterns(require_official=False)

    assert set(compiled) == set(CORE_PATTERN_KEYS)
    for key, pattern in compiled.items():
        metadata = pattern.result_metadata()
        assert metadata["scanner_pattern_key"] == f"v2:{key}"
        assert len(metadata["spec_hash"]) == 64
        assert all(row["status"] == "implemented" for row in metadata["coverage"])
        if key in {"broadening_bottoms", "bull_flags"}:
            assert metadata["official_ready"] is True
            assert validate_official_pattern(key) == []
        else:
            assert metadata["official_ready"] is False
            assert any("golden_fixtures required" in err for err in validate_official_pattern(key))


def test_require_official_allows_broadening_bottoms() -> None:
    engine = ScannerV2Engine()

    compiled = engine.compile_pattern("broadening_bottoms", require_official=True)
    assert compiled.official_ready is True


def test_require_official_allows_bull_flags() -> None:
    engine = ScannerV2Engine()

    compiled = engine.compile_pattern("bull_flags", require_official=True)
    assert compiled.official_ready is True


def test_require_official_blocks_remaining_draft_patterns() -> None:
    engine = ScannerV2Engine()

    with pytest.raises(ContractError, match="golden_fixtures required"):
        engine.compile_pattern("double_bottoms", require_official=True)


def test_unknown_rule_type_fails_closed() -> None:
    registry = load_core_registry()
    lineage = load_taxonomy_lineage()
    mutated = copy.deepcopy(registry)
    mutated["patterns"]["broadening_bottoms"]["rules"][0]["rule_type"] = "visual_guess"

    with pytest.raises(ContractError, match="unimplemented rule_type visual_guess"):
        ScannerV2Engine(mutated, lineage).compile_pattern("broadening_bottoms")


def test_missing_provenance_fails_closed() -> None:
    registry = load_core_registry()
    lineage = load_taxonomy_lineage()
    mutated = copy.deepcopy(registry)
    del mutated["patterns"]["broadening_bottoms"]["rules"][0]["source_page"]

    with pytest.raises(ContractError, match="source_page is required"):
        ScannerV2Engine(mutated, lineage).compile_pattern("broadening_bottoms")


def test_official_gate_fails_when_source_excerpt_does_not_align() -> None:
    registry = load_core_registry()
    lineage = load_taxonomy_lineage()
    mutated = copy.deepcopy(registry)
    mutated["patterns"]["broadening_bottoms"]["rules"][0]["evidence_excerpt"] = "not present in the source page"

    errors = validate_official_pattern("broadening_bottoms", mutated, lineage)

    assert any("excerpt_not_found_on_claimed_pdf_page" in err for err in errors)


def test_full_spec_hash_changes_when_rule_changes() -> None:
    registry = load_core_registry()
    original = registry["patterns"]["broadening_bottoms"]
    changed = copy.deepcopy(original)
    changed["rules"][0]["numeric_threshold"]["value"] = "sideways"

    assert canonical_spec_hash(original) != canonical_spec_hash(changed)


def test_taxonomy_lineage_matches_core_keys() -> None:
    lineage = load_taxonomy_lineage()["lineage"]

    assert set(CORE_PATTERN_KEYS) == set(lineage)
    for key in CORE_PATTERN_KEYS:
        assert lineage[key]["scanner_pattern_key"] == f"v2:{key}"
        assert lineage[key]["bulkowski_chapters"]
        assert lineage[key]["result_payload_contract"] == "schemas/scanner_v2/result_payload.schema.json"


def test_broadening_bottoms_golden_fixtures_match_expected_outcomes() -> None:
    fixtures = load_core_registry()["patterns"]["broadening_bottoms"]["golden_fixtures"]

    outcomes = {fixture["fixture_id"]: run_broadening_bottoms_fixture(fixture).to_dict() for fixture in fixtures}

    assert outcomes["bb_valid_up_breakout"]["matched"] is True
    assert outcomes["bb_valid_up_breakout"]["breakout_direction"] == "up"
    assert outcomes["bb_valid_down_breakout"]["matched"] is True
    assert outcomes["bb_valid_down_breakout"]["breakout_direction"] == "down"
    assert outcomes["bb_reject_flat_bottom_raw_v1_error"]["matched"] is False
    assert "lows_not_falling" in outcomes["bb_reject_flat_bottom_raw_v1_error"]["reasons"]
    assert outcomes["bb_reject_missing_prior_downtrend"]["matched"] is False
    assert "prior_trend_not_down" in outcomes["bb_reject_missing_prior_downtrend"]["reasons"]
    assert outcomes["bb_reject_no_close_breakout"]["matched"] is False
    assert outcomes["bb_reject_no_close_breakout"]["reasons"] == ["no_close_beyond_boundary"]


def test_bull_flags_golden_fixtures_match_expected_outcomes() -> None:
    fixtures = load_core_registry()["patterns"]["bull_flags"]["golden_fixtures"]

    outcomes = {fixture["fixture_id"]: run_bull_flags_fixture(fixture).to_dict() for fixture in fixtures}

    assert outcomes["bf_valid_up_breakout"]["matched"] is True
    assert outcomes["bf_valid_up_breakout"]["breakout_direction"] == "up"
    assert outcomes["bf_reject_weak_prior_advance"]["matched"] is False
    assert "prior_advance_not_steep" in outcomes["bf_reject_weak_prior_advance"]["reasons"]
    assert outcomes["bf_reject_no_up_breakout"]["matched"] is False
    assert outcomes["bf_reject_no_up_breakout"]["reasons"] == ["no_close_above_upper_trendline"]


def test_broadening_bottoms_evidence_excerpts_align_to_claimed_pdf_pages() -> None:
    alignment = verify_pattern_source_alignment("broadening_bottoms")

    assert alignment["aligned"] is True
    assert alignment["errors"] == []
    assert len(alignment["rule_checks"]) == len(load_core_registry()["patterns"]["broadening_bottoms"]["rules"])


def test_bull_flags_evidence_excerpts_align_to_claimed_pdf_pages() -> None:
    alignment = verify_pattern_source_alignment("bull_flags")

    assert alignment["aligned"] is True
    assert alignment["errors"] == []
    assert len(alignment["rule_checks"]) == len(load_core_registry()["patterns"]["bull_flags"]["rules"])
