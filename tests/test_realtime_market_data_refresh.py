from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from scanner import refresh_realtime_market_data as refresh


def _write_stock_db(path: Path, rows: list[tuple[str, str]]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE stock_price_history (
                symbol TEXT,
                time TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY(symbol, time)
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO stock_price_history(symbol, time, open, high, low, close, volume)
            VALUES (?, ?, 10, 11, 9, 10.5, 1000)
            """,
            rows,
        )


def test_inspect_stock_db_and_freshness_status(tmp_path: Path) -> None:
    db_path = tmp_path / "prices.sqlite"
    _write_stock_db(db_path, [("AAA", "2026-06-01"), ("BBB", "2026-06-02")])

    snapshot = refresh.inspect_stock_db(db_path)
    fresh = refresh.freshness_status(snapshot, today=date(2026, 6, 5), staleness_days=3)
    stale = refresh.freshness_status(snapshot, today=date(2026, 6, 6), staleness_days=3)

    assert snapshot.row_count == 2
    assert snapshot.symbol_count == 2
    assert snapshot.max_date == "2026-06-02"
    assert fresh["status"] == "FRESH"
    assert stale["status"] == "STALE"


def test_build_update_command_includes_provider_and_symbols(tmp_path: Path) -> None:
    command = refresh.build_update_command(
        db_path=tmp_path / "prices.sqlite",
        source="VCI",
        rpm=120,
        max_errors=5,
        timeout_seconds=7,
        end="2026-06-05",
        symbols=["aaa", "BBB"],
        python_executable="/python",
        update_script=Path("/update_latest_stock_ohlcv.py"),
    )

    assert command[:2] == ["/python", "/update_latest_stock_ohlcv.py"]
    assert "--stock-db" in command
    assert command[command.index("--source") + 1] == "VCI"
    assert command[command.index("--rpm") + 1] == "120"
    assert command[-4:] == ["--symbol", "AAA", "--symbol", "BBB"]


def test_refresh_report_blocks_when_provider_missing(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "prices.sqlite"
    _write_stock_db(db_path, [("AAA", "2026-05-29")])
    monkeypatch.setattr(refresh, "provider_available", lambda **_: False)

    report = refresh.refresh_realtime_market_data(
        db_path=db_path,
        out_dir=tmp_path / "refresh",
        staleness_days=3,
        check_only=False,
    )

    assert report["status"] == "BLOCKED_MISSING_PROVIDER"
    assert report["provider"]["available"] is False
    assert Path(report["report_path"]).exists()


def test_check_only_never_blocks_on_missing_provider(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "prices.sqlite"
    _write_stock_db(db_path, [("AAA", "2026-05-29")])
    monkeypatch.setattr(refresh, "provider_available", lambda **_: False)

    report = refresh.refresh_realtime_market_data(
        db_path=db_path,
        out_dir=tmp_path / "refresh",
        check_only=True,
    )

    assert report["status"].startswith("CHECK_ONLY_")
    assert report["update_run"] is None


def test_regenerates_market_stats_when_db_is_fresh_but_web_artifact_is_stale(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "prices.sqlite"
    today = date.today().isoformat()
    _write_stock_db(db_path, [("AAA", today)])
    stale_artifact = refresh.MarketStatsArtifactSnapshot(
        path=str(tmp_path / "market_stats_data.json"),
        exists=True,
        stock_latest_date="2026-05-29",
    )
    fresh_artifact = refresh.MarketStatsArtifactSnapshot(
        path=str(tmp_path / "market_stats_data.json"),
        exists=True,
        stock_latest_date=today,
    )
    artifact_calls = [stale_artifact, stale_artifact, fresh_artifact]
    runs = []

    monkeypatch.setattr(refresh, "provider_available", lambda **_: True)
    monkeypatch.setattr(refresh, "choose_provider_python", lambda *_: "/python")
    monkeypatch.setattr(refresh, "inspect_market_stats_artifact", lambda: artifact_calls.pop(0) if artifact_calls else fresh_artifact)

    def fake_run_command(command, *, cwd, timeout=None):
        runs.append(command)
        return {
            "command": command,
            "cwd": str(cwd),
            "started_at": "2026-06-05T09:00:00",
            "finished_at": "2026-06-05T09:00:01",
            "returncode": 0,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(refresh, "_run_command", fake_run_command)

    report = refresh.refresh_realtime_market_data(
        db_path=db_path,
        out_dir=tmp_path / "refresh",
        staleness_days=3,
        regenerate_market_stats=True,
    )

    assert report["status"] == "SKIPPED_FRESH"
    assert report["market_stats_artifact_needs_regeneration"] is True
    assert report["market_stats_regeneration_run"]["returncode"] == 0
    assert runs and "generate_simple_stats.py" in runs[0][1]
    assert report["market_stats_artifact_after"]["stock_latest_date"] == today


def test_refresh_marks_stale_when_provider_completes_but_db_remains_old(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "prices.sqlite"
    _write_stock_db(db_path, [("AAA", "2026-05-29")])

    neutral_artifact = refresh.MarketStatsArtifactSnapshot(
        path=str(tmp_path / "market_stats_data.json"),
        exists=True,
        stock_latest_date="2026-05-29",
    )

    monkeypatch.setattr(refresh, "provider_available", lambda **_: True)
    monkeypatch.setattr(refresh, "choose_provider_python", lambda *_: "/python")
    monkeypatch.setattr(refresh, "inspect_market_stats_artifact", lambda: neutral_artifact)

    def fake_run_command(command, *, cwd, timeout=None):
        return {
            "command": command,
            "cwd": str(cwd),
            "started_at": "2026-06-05T09:00:00",
            "finished_at": "2026-06-05T09:00:01",
            "returncode": 0,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(refresh, "_run_command", fake_run_command)

    report = refresh.refresh_realtime_market_data(
        db_path=db_path,
        out_dir=tmp_path / "refresh",
        staleness_days=0,
        regenerate_market_stats=False,
    )

    assert report["status"] == "REFRESHED_STALE"
    assert report["freshness_after"]["is_stale"] is True
    assert "still older" in report["reason"]
