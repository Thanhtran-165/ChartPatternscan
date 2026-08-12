from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from scanner.realtime_scan_history import evaluate_setup_outcome, update_realtime_scan_history


def _write_test_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE stock_price_history (
                symbol TEXT,
                time TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
            """
        )
        rows = [
            ("AAA", "2026-01-01", 10, 10.2, 9.8, 10, 1000),
            ("AAA", "2026-01-02", 10.2, 10.5, 10.1, 10.4, 1000),
            ("AAA", "2026-01-03", 10.6, 11.2, 10.5, 11.1, 1000),
            ("AAA", "2026-01-04", 11.2, 12.3, 11.0, 12.1, 1000),
        ]
        conn.executemany("INSERT INTO stock_price_history VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def test_evaluate_setup_outcome_marks_trigger_and_target() -> None:
    df = pd.DataFrame(
        [
            {"date": "2026-01-01", "open": 10, "high": 10.2, "low": 9.8, "close": 10},
            {"date": "2026-01-02", "open": 10.2, "high": 10.5, "low": 10.1, "close": 10.4},
            {"date": "2026-01-03", "open": 10.6, "high": 11.2, "low": 10.5, "close": 11.1},
            {"date": "2026-01-04", "open": 11.2, "high": 12.3, "low": 11.0, "close": 12.1},
        ]
    )
    row = {
        "candidate_date": "2026-01-01",
        "trigger_price": 11,
        "invalidation_price": 9.5,
        "target_price": 12,
        "reference_price": 10,
    }

    outcome = evaluate_setup_outcome(row, df)

    assert outcome["outcome_status"] == "target_hit_after_trigger"
    assert outcome["triggered"] is True
    assert outcome["target_hit"] is True
    assert outcome["trigger_date"] == "2026-01-03"


def test_update_realtime_scan_history_writes_ledger_without_duplicate_candidates(tmp_path: Path) -> None:
    db_path = tmp_path / "prices.sqlite"
    _write_test_db(db_path)
    setups = pd.DataFrame(
        [
            {
                "workflow_id": "buy_setup_scan_watchlist_v1",
                "detector_version": "setup_proxy_v1",
                "pattern_id": "bull_flags",
                "family": "flag_family",
                "symbol": "AAA",
                "market_group": "VN30",
                "latest_date": "2026-01-01",
                "setup_start_date": "2025-12-20",
                "last_close": 10,
                "trigger_price": 11,
                "invalidation_price": 9.5,
                "target_price": 12,
                "distance_to_trigger_pct": 10,
                "potential_profit_pct": 20,
                "setup_quality_score": 80,
                "setup_quality_tier": "watchlist_setup",
                "setup_reason": "test",
                "detector_family": "flaglike_continuation",
                "volume_quality_label": "strong",
                "volume_warning_label": "none",
                "volume_ratio_20": 1.8,
                "price_volume_phase": "up_confirmed",
                "pattern_volume_contraction_ratio": 0.8,
            }
        ]
    )

    first = update_realtime_scan_history(
        buy_setups=setups,
        watchlist=pd.DataFrame(),
        db_path=db_path,
        history_dir=tmp_path / "history",
        generated_at="2026-01-05T09:00:00",
    )
    second = update_realtime_scan_history(
        buy_setups=setups,
        watchlist=pd.DataFrame(),
        db_path=db_path,
        history_dir=tmp_path / "history",
        generated_at="2026-01-06T09:00:00",
    )

    ledger = pd.read_csv(first["paths"]["ledger"])
    assert len(ledger) == 1
    ledger_after_second = pd.read_csv(second["paths"]["ledger"])
    assert int(ledger_after_second.iloc[0]["seen_count"]) == 2
    assert second["history"]["total_candidates"] == 1
    assert second["history"]["setup_conversion"]["target_hit_after_trigger"] == 1
    assert ledger_after_second.iloc[0]["volume_quality_label"] == "strong"
    assert float(ledger_after_second.iloc[0]["volume_ratio_20"]) == 1.8
    assert second["history"]["by_volume_quality"]["strong"] == 1
