from __future__ import annotations

import json
from pathlib import Path

from scanner.build_bull_flag_investor_chapter import build_ai_editorial_narrative, build_investor_chapter


def _source_notes(tmp_path: Path) -> Path:
    path = tmp_path / "source_notes.json"
    path.write_text(
        json.dumps(
            {
                "source_grounding_id": "bull_flag_bulkowski_source_grounding_v1",
                "status": "PASS",
                "local_source": {
                    "core_patterns_path": "scanner/v2/core_patterns.json",
                    "book_chapters": [{"chapter": 21, "name": "Flags", "source_page_start": 335}],
                },
                "source_rules": [
                    {
                        "rule_id": "bf.prior_trend.steep_up",
                        "source_page": 338,
                        "short_excerpt": "Steep, quick price trend",
                        "implementation_mapping": "Require a steep, quick advance before a Bull Flag formation.",
                    },
                    {
                        "rule_id": "bf.measure.pole_height_legacy",
                        "source_page": 347,
                        "short_excerpt": "Calculate the price difference between the start of the trend and the formation.",
                        "implementation_mapping": "Compute the legacy pole-height measure rule.",
                    },
                ],
                "bulkowski_book_2e_stats": {
                    "upward_breakouts": {
                        "break_even_failure_rate_bull_bear_pct": [4, 3],
                        "average_rise_bull_bear_pct": [23, 17],
                        "percentage_meeting_price_target_bull_bear_pct": [64, 55],
                    },
                    "downward_breakouts": {
                        "break_even_failure_rate_bull_bear_pct": [2, 0],
                        "average_decline_bull_bear_pct": [16, 25],
                        "percentage_meeting_price_target_bull_bear_pct": [47, 54],
                    },
                },
                "thepatternsite_2020_stats": {
                    "url": "https://thepatternsite.com/flags.html",
                    "break_even_failure_rate_up_down_pct": [44, 45],
                    "average_rise_decline_up_down_pct": [9, 8],
                    "percentage_meeting_price_target_up_down_pct": [46, 46],
                },
                "thepatternsite_measure_rule": {
                    "url": "https://thepatternsite.com/measure.html",
                    "flags_up_breakout_rule": "Flag low + ((Flagpole height) * 46%)",
                    "flags_down_breakout_rule": "Flag high - ((Flagpole height) * 46%)",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _payload(tmp_path: Path) -> Path:
    path = tmp_path / "payload.json"
    path.write_text(
        json.dumps(
            {
                "classification": "bull_flag_tradable_research_candidate_95",
                "chapter_reference": {
                    "median_mfe_pct": 12.72,
                    "median_mae_pct": 8.18,
                    "legacy_target_hit_rate": 39.09,
                    "failure_5pct_rate": 24.55,
                },
                "target_calibration": {
                    "base_target": {"target_hit_rate": 70.0, "target_first_before_adverse_5pct_rate": 42.73},
                    "rows": [
                        {"target_multiple": 0.46, "target_role": "bulkowski_adjusted_base", "n": 110, "target_hit_rate": 70.0, "target_first_before_adverse_5pct_rate": 42.73, "failure_5pct_rate": 24.55}
                    ],
                },
                "release_candidate": {
                    "conservative_score": 95.78,
                    "fresh": {"score": 98.67},
                },
                "tradable_setup": {
                    "selected_strategy_id": "bf_v2",
                    "selected_metrics": {
                        "entry_delay_bars": 3,
                        "stop_loss_pct": 7.0,
                        "max_holding_days": 60,
                        "target_multiple": 0.46,
                        "trades": 62,
                        "total_return_pct": 19.42,
                        "validation_total_return_pct": 3.73,
                        "holdout_total_return_pct": 5.54,
                        "position_size_pct": 0.1,
                        "max_positions": 10,
                        "commission_bps_per_side": 15.0,
                        "slippage_bps_per_side": 10.0,
                        "sell_tax_bps": 10.0,
                        "median_adtv_participation_pct": 3.11,
                    },
                    "walk_forward_summary": {"positive_fold_rate_pct": 100.0},
                },
                "fresh_candidate": {
                    "summary": {"trades": 67, "total_return_pct": 18.15, "holdout_total_return_pct": 9.08},
                    "walk_forward_summary": {"positive_fold_rate_pct": 100.0},
                },
                "supporting_robustness": {"profiles": []},
                "data_scope_and_caveats": {"remaining_caveats": ["corporate_action_audit"]},
                "scanner_contract": {"detector_config": {}},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_ai_editorial_narrative_is_data_bound(tmp_path: Path) -> None:
    payload = json.loads(_payload(tmp_path).read_text(encoding="utf-8"))

    narrative = build_ai_editorial_narrative(payload, json.loads(_source_notes(tmp_path).read_text(encoding="utf-8")))

    assert narrative["status"] == "source_grounded_ai_editorial_layer"
    assert "source notes" in narrative["executive_summary"][0]
    assert "95.78" in narrative["executive_summary"][1]
    assert "0.46x" in narrative["executive_summary"][2]


def test_investor_chapter_writer_emits_pdf_and_narrative(tmp_path: Path) -> None:
    paths = build_investor_chapter(
        payload_path=_payload(tmp_path),
        out_dir=tmp_path / "out",
        source_notes_path=_source_notes(tmp_path),
    )

    assert paths["pdf"].exists()
    assert paths["pdf"].stat().st_size > 0
    assert paths["narrative"].exists()
    assert "AI Editorial Narrative" in paths["narrative"].read_text(encoding="utf-8")
    assert "Source grounding" in paths["narrative"].read_text(encoding="utf-8")
