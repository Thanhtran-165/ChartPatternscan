"""Tests for stale-setup filtering and breakout-today detection in the realtime scan email.

These cover the three classification buckets the email now produces:
- new / recurring setups (still shown in box 0)
- confirmed breakout today (highlighted in box 0a)
- stale setups (>= stale_threshold consecutive days, hidden from the email body)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from scanner.send_realtime_scan_email import (
    _classify_setups,
    _detect_breakouts_today,
    _load_setup_persistence,
    render_html_email,
    render_text_email,
    summarize_watchlist,
)


def _make_setup_row(symbol: str, pattern_id: str = "bull_flags", trigger_price: float = 10.0) -> dict:
    return {
        "symbol": symbol,
        "pattern_id": pattern_id,
        "family": "flag_family",
        "market_group": "VN100 ex VN30",
        "latest_date": "2026-07-17",
        "last_close": 9.5,
        "trigger_price": trigger_price,
        "invalidation_price": 8.5,
        "target_price": 11.0,
        "distance_to_trigger_pct": 5.0,
        "potential_profit_pct": 15.0,
        "setup_quality_score": 85.0,
        "setup_quality_tier": "strong_setup",
        "setup_reason": "test reason",
        "volume_quality_label": "healthy",
        "volume_warning_label": "none",
    }


def _write_ledger(history_dir: Path, rows: list[dict]) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = history_dir / "candidate_ledger.csv"
    if rows:
        pd.DataFrame(rows).to_csv(ledger_path, index=False)
    else:
        # Empty ledger with the expected columns
        pd.DataFrame(columns=["candidate_id", "stage", "symbol", "pattern_id", "observed_at"]).to_csv(ledger_path, index=False)
    return ledger_path


def _make_test_db(db_path: Path, symbol_to_bars: dict[str, list[dict]]) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_price_history (
                id INTEGER PRIMARY KEY,
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
        for symbol, bars in symbol_to_bars.items():
            for bar in bars:
                conn.execute(
                    "INSERT INTO stock_price_history (symbol, time, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        symbol,
                        bar["date"],
                        bar["open"],
                        bar["high"],
                        bar["low"],
                        bar["close"],
                        bar["volume"],
                    ),
                )
        conn.commit()
    finally:
        conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Persistence loader
# ---------------------------------------------------------------------------


def test_persistence_counts_consecutive_scan_days_ending_at_generated_at(tmp_path: Path) -> None:
    """A streak counts consecutive *scan days* (distinct observed_at in the ledger),
    not calendar days, so weekend/market-holiday gaps don't reset the counter."""
    generated_at = "2026-07-17T17:03:28"
    # Global scan days present in the ledger (sorted reverse): 2026-07-17, 07-16, 07-15, 07-10
    rows = [
        # AAA appears on the three most recent scan days (17, 16, 15) but NOT on 07-10
        {"symbol": "AAA", "pattern_id": "bull_flags", "observed_at": "2026-07-15"},
        {"symbol": "AAA", "pattern_id": "bull_flags", "observed_at": "2026-07-16"},
        {"symbol": "AAA", "pattern_id": "bull_flags", "observed_at": "2026-07-17"},
        # BBB appears on the latest scan day (17) and the oldest (10) but missed 16 and 15
        {"symbol": "BBB", "pattern_id": "bull_flags", "observed_at": "2026-07-10"},
        {"symbol": "BBB", "pattern_id": "bull_flags", "observed_at": "2026-07-17"},
    ]
    _write_ledger(tmp_path, rows)

    persistence = _load_setup_persistence(tmp_path, generated_at)

    # AAA: present on 17, 16, 15 (3 consecutive scan days); missing on 10 stops the streak
    assert persistence[("AAA", "bull_flags")] == 3
    # BBB: present on 17 (anchor), but missing on 16 -> streak breaks at 1
    assert persistence[("BBB", "bull_flags")] == 1


def test_persistence_skips_weekend_calendar_gap_between_scan_days(tmp_path: Path) -> None:
    """Friday -> Monday gap (no weekend scan) must NOT reset a streak."""
    generated_at = "2026-07-13T17:03:28"  # Monday
    rows = [
        {"symbol": "AAA", "pattern_id": "bull_flags", "observed_at": "2026-07-10"},  # Friday
        {"symbol": "AAA", "pattern_id": "bull_flags", "observed_at": "2026-07-13"},  # Monday
    ]
    _write_ledger(tmp_path, rows)

    persistence = _load_setup_persistence(tmp_path, generated_at)

    # Two consecutive scan days (Fri, Mon) even though 3 calendar days apart
    assert persistence[("AAA", "bull_flags")] == 2


def test_persistence_returns_empty_dict_when_no_ledger(tmp_path: Path) -> None:
    persistence = _load_setup_persistence(tmp_path, "2026-07-17T17:03:28")
    assert persistence == {}


# ---------------------------------------------------------------------------
# Breakout detection
# ---------------------------------------------------------------------------


def test_breakout_today_detected_when_close_crosses_above_trigger(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    # Two bars: yesterday close below trigger (10.0), today close above
    bars = {
        "AAA": [
            {"date": "2026-07-16", "open": 9.2, "high": 9.8, "low": 9.0, "close": 9.5, "volume": 1000},
            {"date": "2026-07-17", "open": 9.8, "high": 10.5, "low": 9.7, "close": 10.2, "volume": 1500},
        ],
    }
    _make_test_db(db_path, bars)

    setups = pd.DataFrame([_make_setup_row("AAA", trigger_price=10.0)])
    breakouts = _detect_breakouts_today(setups, db_path)

    assert ("AAA", "bull_flags") in breakouts


def test_breakout_not_detected_when_already_above_trigger_yesterday(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    bars = {
        "AAA": [
            {"date": "2026-07-16", "open": 10.0, "high": 10.5, "low": 9.9, "close": 10.3, "volume": 1000},
            {"date": "2026-07-17", "open": 10.3, "high": 10.6, "low": 10.1, "close": 10.4, "volume": 1500},
        ],
    }
    _make_test_db(db_path, bars)

    setups = pd.DataFrame([_make_setup_row("AAA", trigger_price=10.0)])
    breakouts = _detect_breakouts_today(setups, db_path)

    assert ("AAA", "bull_flags") not in breakouts


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_new_setup_classified_as_new() -> None:
    setups = pd.DataFrame([_make_setup_row("AAA")])
    persistence: dict[tuple[str, str], int] = {}  # not seen before -> new
    breakouts: set[tuple[str, str]] = set()

    classified = _classify_setups(setups, persistence, breakouts, stale_threshold=10)

    assert classified.iloc[0]["status_today"] == "new"
    assert classified.iloc[0]["persistence_days"] == 0


def test_recurring_setup_below_threshold_classified_as_recurring() -> None:
    setups = pd.DataFrame([_make_setup_row("AAA")])
    persistence = {("AAA", "bull_flags"): 5}  # 5 consecutive days, below threshold
    breakouts: set[tuple[str, str]] = set()

    classified = _classify_setups(setups, persistence, breakouts, stale_threshold=10)

    assert classified.iloc[0]["status_today"] == "recurring"
    assert classified.iloc[0]["persistence_days"] == 5


def test_stale_setup_at_threshold_classified_as_stale() -> None:
    setups = pd.DataFrame([_make_setup_row("AAA")])
    persistence = {("AAA", "bull_flags"): 10}  # exactly at threshold
    breakouts: set[tuple[str, str]] = set()

    classified = _classify_setups(setups, persistence, breakouts, stale_threshold=10)

    assert classified.iloc[0]["status_today"] == "stale"


def test_breakout_today_takes_priority_over_stale() -> None:
    """Even a stale setup should surface if it broke out today."""
    setups = pd.DataFrame([_make_setup_row("AAA")])
    persistence = {("AAA", "bull_flags"): 15}  # would be stale
    breakouts = {("AAA", "bull_flags")}  # but broke out today

    classified = _classify_setups(setups, persistence, breakouts, stale_threshold=10)

    assert classified.iloc[0]["status_today"] == "confirmed_breakout_today"


# ---------------------------------------------------------------------------
# summarize_watchlist integration
# ---------------------------------------------------------------------------


def _empty_watchlist() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "pattern_id",
            "symbol",
            "event_date",
            "after_buy_action",
            "market_group",
        ]
    )


def test_summarize_splits_setup_section_when_filtering_active(tmp_path: Path) -> None:
    # Three setups: one new, one stale (10 days), one breakout-today
    setups = pd.DataFrame(
        [
            _make_setup_row("NEW", trigger_price=10.0),
            _make_setup_row("STALE", trigger_price=10.0),
            _make_setup_row("BREAK", trigger_price=10.0),
        ]
    )
    persistence = {
        ("STALE", "bull_flags"): 10,
    }
    # Build DB so BREAK has a breakout today
    db_path = tmp_path / "test.db"
    _make_test_db(
        db_path,
        {
            "NEW": [
                {"date": "2026-07-16", "open": 9.0, "high": 9.5, "low": 8.9, "close": 9.3, "volume": 1000},
                {"date": "2026-07-17", "open": 9.3, "high": 9.6, "low": 9.1, "close": 9.4, "volume": 1100},
            ],
            "STALE": [
                {"date": "2026-07-16", "open": 9.0, "high": 9.5, "low": 8.9, "close": 9.3, "volume": 1000},
                {"date": "2026-07-17", "open": 9.3, "high": 9.6, "low": 9.1, "close": 9.4, "volume": 1100},
            ],
            "BREAK": [
                {"date": "2026-07-16", "open": 9.0, "high": 9.5, "low": 8.9, "close": 9.3, "volume": 1000},
                {"date": "2026-07-17", "open": 9.8, "high": 10.5, "low": 9.7, "close": 10.2, "volume": 1500},
            ],
        },
    )

    # Patch _load_setup_persistence via history_dir (empty ledger + explicit persistence not needed)
    # We rely on summarize_watchlist calling _load_setup_persistence(history_dir, ...).
    # So write a ledger that yields the persistence above.
    ledger_rows = []
    for day in [
        "2026-07-04", "2026-07-05", "2026-07-06", "2026-07-07", "2026-07-08",
        "2026-07-09", "2026-07-10", "2026-07-13", "2026-07-14", "2026-07-15",
    ]:
        ledger_rows.append(
            {"symbol": "STALE", "pattern_id": "bull_flags", "observed_at": day}
        )
    # Plus today (2026-07-17) so streak ends today
    ledger_rows.append({"symbol": "STALE", "pattern_id": "bull_flags", "observed_at": "2026-07-17"})
    _write_ledger(tmp_path, ledger_rows)

    summary = summarize_watchlist(
        _empty_watchlist(),
        buy_setups=setups,
        history_dir=tmp_path,
        db_path=db_path,
        stale_threshold=10,
    )

    sections = summary["sections"]
    new_symbols = {row["symbol"] for row in sections["buy_setup_new"]}
    breakout_symbols = {row["symbol"] for row in sections["buy_setup_breakout_today"]}
    stale_symbols = {row["symbol"] for row in sections["buy_setup_stale_hidden"]}

    assert new_symbols == {"NEW"}
    assert breakout_symbols == {"BREAK"}
    assert stale_symbols == {"STALE"}

    assert summary["counts"]["buy_setup_new"] == 1
    assert summary["counts"]["buy_setup_breakout_today"] == 1
    assert summary["counts"]["buy_setup_stale_hidden"] == 1


def test_summarize_keeps_legacy_buy_setup_key_without_filter() -> None:
    """When history_dir/db_path are None, behavior matches the old contract."""
    setups = pd.DataFrame([_make_setup_row("AAA")])
    summary = summarize_watchlist(_empty_watchlist(), buy_setups=setups)

    # Legacy key still present and contains the row
    assert "buy_setup" in summary["sections"]
    legacy_symbols = {row["symbol"] for row in summary["sections"]["buy_setup"]}
    assert legacy_symbols == {"AAA"}


def test_render_hides_stale_section_in_text_and_html(tmp_path: Path) -> None:
    setups = pd.DataFrame(
        [
            _make_setup_row("NEW", trigger_price=10.0),
            _make_setup_row("STALE", trigger_price=10.0),
        ]
    )
    db_path = tmp_path / "test.db"
    _make_test_db(
        db_path,
        {
            "NEW": [
                {"date": "2026-07-16", "open": 9.0, "high": 9.5, "low": 8.9, "close": 9.3, "volume": 1000},
                {"date": "2026-07-17", "open": 9.3, "high": 9.6, "low": 9.1, "close": 9.4, "volume": 1100},
            ],
            "STALE": [
                {"date": "2026-07-16", "open": 9.0, "high": 9.5, "low": 8.9, "close": 9.3, "volume": 1000},
                {"date": "2026-07-17", "open": 9.3, "high": 9.6, "low": 9.1, "close": 9.4, "volume": 1100},
            ],
        },
    )
    ledger_rows = [
        {"symbol": "STALE", "pattern_id": "bull_flags", "observed_at": day}
        for day in [
            "2026-07-04", "2026-07-05", "2026-07-06", "2026-07-07", "2026-07-08",
            "2026-07-09", "2026-07-10", "2026-07-13", "2026-07-14", "2026-07-15",
        ]
    ]
    ledger_rows.append({"symbol": "STALE", "pattern_id": "bull_flags", "observed_at": "2026-07-17"})
    _write_ledger(tmp_path, ledger_rows)

    summary = summarize_watchlist(
        _empty_watchlist(),
        buy_setups=setups,
        history_dir=tmp_path,
        db_path=db_path,
        stale_threshold=10,
    )

    text = render_text_email(summary)
    html = render_html_email(summary)

    assert "NEW" in text
    assert "STALE" not in text
    assert "NEW" in html
    assert "STALE" not in html
    # Footer note should mention hidden stale count
    assert "đã ẩn" in text.lower() or "ẩn khỏi mail" in text.lower()


def test_render_shows_breakout_today_box(tmp_path: Path) -> None:
    setups = pd.DataFrame([_make_setup_row("BREAK", trigger_price=10.0)])
    db_path = tmp_path / "test.db"
    _make_test_db(
        db_path,
        {
            "BREAK": [
                {"date": "2026-07-16", "open": 9.0, "high": 9.5, "low": 8.9, "close": 9.3, "volume": 1000},
                {"date": "2026-07-17", "open": 9.8, "high": 10.5, "low": 9.7, "close": 10.2, "volume": 1500},
            ],
        },
    )
    _write_ledger(tmp_path, [])

    summary = summarize_watchlist(
        _empty_watchlist(),
        buy_setups=setups,
        history_dir=tmp_path,
        db_path=db_path,
        stale_threshold=10,
    )

    text = render_text_email(summary)
    html = render_html_email(summary)

    assert "BREAK" in text
    assert "xác nhận breakout" in text.lower() or "đã xác nhận" in text.lower()
    assert "BREAK" in html
