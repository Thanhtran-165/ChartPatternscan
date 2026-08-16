from scanner.canonical_chapter_content import _locked_fact_sections, _remove_unlocked_numeric_claims
from scanner.pattern_publication_core import _public_term


def test_locked_fact_sections_use_current_payload_values() -> None:
    payload = {
        "pattern_id": "synthetic_pattern",
        "chapter_reference": {
            "events": 123,
            "median_mfe_pct": 31.34,
            "median_mae_pct": 14.67,
            "failure_5pct_rate": 9.3,
        },
        "target_calibration": {
            "base_target": {
                "target_multiple": 1.0,
                "target_hit_rate": 78.9,
                "target_first_before_adverse_5pct_rate": 44.7,
                "failure_5pct_rate": 9.3,
            },
            "legacy_target": {"target_multiple": 1.0, "target_hit_rate": 78.9},
        },
    }
    quick = " ".join(_locked_fact_sections(payload)["quick_read"])
    assert "78,90%" in quick
    assert "74,23%" not in quick
    assert "31,34%" in quick


def test_unlocked_numeric_claims_are_removed_from_explanatory_sections() -> None:
    sections = {
        "tour": ["Tỷ lệ cũ 74,23% không còn được dùng."],
        "size_volume": ["Biên độ cũ 12,69% chỉ là số đời trước."],
        "tactics": ["Mốc cũ 0,5x không còn là fact khóa."],
        "checklist": ["Đọc mốc cũ 1,0x trước khi hành động."],
    }
    cleaned = _remove_unlocked_numeric_claims(sections)
    assert all("74,23%" not in value for value in cleaned["tour"])
    assert all("12,69%" not in value for value in cleaned["size_volume"])
    assert all("0,5x" not in value for value in cleaned["tactics"])
    assert all("1,0x" not in value for value in cleaned["checklist"])


def test_internal_example_and_quality_tokens_are_reader_safe() -> None:
    assert "textbook_success" not in _public_term("textbook_success")
    assert "zero and stale" not in _public_term("zero_and_stale")
    assert "layer missing" not in _public_term("tradable_layer_missing")
