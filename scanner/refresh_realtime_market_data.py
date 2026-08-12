"""Refresh/audit the OHLCV cache before realtime pattern scans.

The pattern scanner reads a local SQLite snapshot. This module makes that
dependency explicit: it can inspect freshness, call the Market Stats refresher
when requested, and write a small audit report for each realtime run.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, asdict, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
MAIN_SONET_ROOT = ROOT.parent
WORKFLOW_ID = "realtime_market_data_refresh_v1"
DEFAULT_STOCK_DB = MAIN_SONET_ROOT / "market_cache" / "stock_ohlcv" / "latest.sqlite"
DEFAULT_SHARED_VENV_PYTHON = MAIN_SONET_ROOT / ".venv" / "bin" / "python"


def existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_MARKET_STATS_UPDATE_SCRIPT = existing_path(
    MAIN_SONET_ROOT / "market_stats" / "update_latest_stock_ohlcv.py",
    MAIN_SONET_ROOT / "market_stats" / "update_latest_stock_ohlcv 2.py",
)
DEFAULT_MARKET_STATS_GENERATE_SCRIPT = existing_path(
    MAIN_SONET_ROOT / "market_stats" / "generate_simple_stats.py",
    MAIN_SONET_ROOT / "market_stats" / "generate_simple_stats 2.py",
)
DEFAULT_MARKET_STATS_DATA_JSON = existing_path(
    MAIN_SONET_ROOT / "market_stats" / "web" / "market_stats_data.json",
    MAIN_SONET_ROOT / "market_stats" / "web" / "market_stats_data 2.json",
)
DEFAULT_OUT_DIR = ROOT / "artifacts" / "realtime_scan" / "latest" / "data_refresh"
FAIL_STATUSES = {
    "MISSING_DB",
    "MISSING_TABLE",
    "BLOCKED_MISSING_PROVIDER",
    "REFRESH_FAILED",
    "REFRESHED_STALE",
    "REGENERATION_FAILED",
}


@dataclass(frozen=True)
class StockDbSnapshot:
    db_path: str
    exists: bool
    has_stock_price_history: bool
    row_count: int | None = None
    min_date: str | None = None
    max_date: str | None = None
    symbol_count: int | None = None
    latest_date_symbol_count: int | None = None

    @property
    def max_date_value(self) -> date | None:
        if not self.max_date:
            return None
        return datetime.strptime(str(self.max_date)[:10], "%Y-%m-%d").date()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketStatsArtifactSnapshot:
    path: str
    exists: bool
    stock_latest_date: str | None = None
    stock_data_latest_date: str | None = None
    stock_analysis_date: str | None = None
    generated_at: str | None = None
    mtime: str | None = None
    error: str | None = None

    @property
    def comparable_stock_date_value(self) -> date | None:
        comparable = self.stock_data_latest_date or self.stock_latest_date
        if not comparable:
            return None
        return datetime.strptime(str(comparable)[:10], "%Y-%m-%d").date()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_stock_db(db_path: Path) -> StockDbSnapshot:
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        return StockDbSnapshot(str(path), exists=False, has_stock_price_history=False)
    with sqlite3.connect(path) as conn:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_price_history'"
        ).fetchone()
        if not has_table:
            return StockDbSnapshot(str(path), exists=True, has_stock_price_history=False)
        row = conn.execute(
            "SELECT COUNT(*), MIN(time), MAX(time), COUNT(DISTINCT symbol) FROM stock_price_history"
        ).fetchone()
        latest_date = str(row[2])[:10] if row[2] else None
        latest_date_symbol_count = None
        if latest_date:
            latest_date_symbol_count = conn.execute(
                "SELECT COUNT(DISTINCT symbol) FROM stock_price_history WHERE time = ?",
                (latest_date,),
            ).fetchone()[0]
    return StockDbSnapshot(
        db_path=str(path),
        exists=True,
        has_stock_price_history=True,
        row_count=int(row[0] or 0),
        min_date=str(row[1])[:10] if row[1] else None,
        max_date=latest_date,
        symbol_count=int(row[3] or 0),
        latest_date_symbol_count=int(latest_date_symbol_count or 0) if latest_date_symbol_count is not None else None,
    )


def scoped_latest_symbol_count(db_path: Path, snapshot: StockDbSnapshot, symbols: Sequence[str] | None) -> StockDbSnapshot:
    if not symbols or not snapshot.max_date or not snapshot.exists or not snapshot.has_stock_price_history:
        return snapshot
    normalized = sorted({str(symbol).upper() for symbol in symbols if str(symbol).strip()})
    if not normalized:
        return snapshot
    placeholders = ",".join("?" for _ in normalized)
    params = [snapshot.max_date, *normalized]
    with sqlite3.connect(Path(db_path).expanduser().resolve()) as conn:
        row = conn.execute(
            f"""
            SELECT COUNT(DISTINCT symbol)
            FROM stock_price_history
            WHERE time = ?
              AND UPPER(symbol) IN ({placeholders})
            """,
            params,
        ).fetchone()
    return replace(snapshot, latest_date_symbol_count=int(row[0] or 0))


def inspect_market_stats_artifact(path: Path = DEFAULT_MARKET_STATS_DATA_JSON) -> MarketStatsArtifactSnapshot:
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists():
        return MarketStatsArtifactSnapshot(path=str(artifact_path), exists=False)
    mtime = datetime.fromtimestamp(artifact_path.stat().st_mtime).isoformat(timespec="seconds")
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit should report malformed artifacts.
        return MarketStatsArtifactSnapshot(path=str(artifact_path), exists=True, mtime=mtime, error=f"{type(exc).__name__}: {exc}")
    sources = data.get("sources") if isinstance(data.get("sources"), dict) else {}
    stock_source = sources.get("stock_ohlcv") if isinstance(sources.get("stock_ohlcv"), dict) else {}
    summary_cards = data.get("summary_cards") if isinstance(data.get("summary_cards"), dict) else {}
    stock_latest_date = (
        stock_source.get("latest_date")
        or stock_source.get("analysis_date")
        or summary_cards.get("stock_latest_date")
    )
    return MarketStatsArtifactSnapshot(
        path=str(artifact_path),
        exists=True,
        stock_latest_date=str(stock_latest_date)[:10] if stock_latest_date else None,
        stock_data_latest_date=str(stock_source.get("data_latest_date"))[:10] if stock_source.get("data_latest_date") else None,
        stock_analysis_date=str(stock_source.get("analysis_date"))[:10] if stock_source.get("analysis_date") else None,
        generated_at=str(data.get("generated_at")) if data.get("generated_at") else None,
        mtime=mtime,
    )


def market_stats_artifact_needs_regeneration(
    stock_snapshot: StockDbSnapshot,
    artifact_snapshot: MarketStatsArtifactSnapshot,
) -> bool:
    db_latest = stock_snapshot.max_date_value
    artifact_latest = artifact_snapshot.comparable_stock_date_value
    if db_latest is None:
        return False
    if not artifact_snapshot.exists or artifact_latest is None:
        return True
    return artifact_latest < db_latest


def provider_available(*, python_executable: str = sys.executable, module_name: str = "vnstock_data", timeout: int = 45) -> bool:
    try:
        result = subprocess.run(
            [python_executable, "-c", f"import {module_name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=int(timeout),
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def choose_provider_python(python_executable: str | None = None) -> str:
    """Choose a Python runtime that can import vnstock_data.

    The pattern project may use its own Python 3.14 venv for tests, while
    Market Stats/Vnstock subscription tooling lives in the shared workspace
    venv. Prefer the caller's runtime when possible, then fall back to the
    shared workspace runtime.
    """
    candidates = [python_executable or sys.executable]
    if str(DEFAULT_SHARED_VENV_PYTHON) not in candidates and DEFAULT_SHARED_VENV_PYTHON.exists():
        candidates.append(str(DEFAULT_SHARED_VENV_PYTHON))
    for candidate in candidates:
        if provider_available(python_executable=str(candidate)):
            return str(candidate)
    return str(candidates[0])


def freshness_status(
    snapshot: StockDbSnapshot,
    *,
    today: date | None = None,
    staleness_days: int = 0,
    min_latest_symbols: int = 0,
) -> dict[str, Any]:
    current_day = today or date.today()
    if not snapshot.exists:
        return {"status": "MISSING_DB", "is_stale": True, "days_stale": None, "fresh_after": None}
    if not snapshot.has_stock_price_history:
        return {"status": "MISSING_TABLE", "is_stale": True, "days_stale": None, "fresh_after": None}
    latest = snapshot.max_date_value
    if latest is None:
        return {"status": "EMPTY_DB", "is_stale": True, "days_stale": None, "fresh_after": None}
    days_stale = (current_day - latest).days
    fresh_after = current_day - timedelta(days=max(0, int(staleness_days)))
    latest_symbol_count = int(snapshot.latest_date_symbol_count or 0)
    if latest >= fresh_after:
        if int(min_latest_symbols) > 0 and latest_symbol_count < int(min_latest_symbols):
            return {
                "status": "INCOMPLETE_LATEST_DATE",
                "is_stale": True,
                "days_stale": days_stale,
                "fresh_after": fresh_after.isoformat(),
                "latest_date_symbol_count": latest_symbol_count,
                "min_latest_symbols": int(min_latest_symbols),
            }
        return {
            "status": "FRESH",
            "is_stale": False,
            "days_stale": days_stale,
            "fresh_after": fresh_after.isoformat(),
            "latest_date_symbol_count": latest_symbol_count,
            "min_latest_symbols": int(min_latest_symbols),
        }
    return {
        "status": "STALE",
        "is_stale": True,
        "days_stale": days_stale,
        "fresh_after": fresh_after.isoformat(),
        "latest_date_symbol_count": latest_symbol_count,
        "min_latest_symbols": int(min_latest_symbols),
    }


def build_update_command(
    *,
    db_path: Path,
    source: str,
    rpm: int,
    max_errors: int,
    timeout_seconds: int,
    end: str | None = None,
    symbols: Sequence[str] | None = None,
    python_executable: str | None = None,
    update_script: Path = DEFAULT_MARKET_STATS_UPDATE_SCRIPT,
) -> list[str]:
    command = [
        python_executable,
        str(update_script),
        "--stock-db",
        str(Path(db_path).expanduser().resolve()),
        "--source",
        source,
        "--rpm",
        str(int(rpm)),
        "--max-errors",
        str(int(max_errors)),
        "--timeout-seconds",
        str(int(timeout_seconds)),
    ]
    if end:
        command.extend(["--end", str(end)])
    for symbol in symbols or []:
        command.extend(["--symbol", str(symbol).upper()])
    return command


def _run_command(command: list[str], *, cwd: Path, timeout: int | None = None) -> dict[str, Any]:
    started = datetime.now().isoformat(timespec="seconds")
    result = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "cwd": str(cwd),
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def refresh_realtime_market_data(
    *,
    db_path: Path = DEFAULT_STOCK_DB,
    out_dir: Path = DEFAULT_OUT_DIR,
    check_only: bool = False,
    force: bool = False,
    strict: bool = False,
    source: str = "VCI",
    rpm: int = 180,
    max_errors: int = 80,
    timeout_seconds: int = 10,
    command_timeout_seconds: int = 3600,
    staleness_days: int = 0,
    min_latest_symbols: int = 0,
    end: str | None = None,
    symbols: Sequence[str] | None = None,
    regenerate_market_stats: bool = True,
    python_executable: str = sys.executable,
    update_script: Path = DEFAULT_MARKET_STATS_UPDATE_SCRIPT,
    generate_script: Path = DEFAULT_MARKET_STATS_GENERATE_SCRIPT,
) -> dict[str, Any]:
    db_path = Path(db_path).expanduser().resolve()
    update_script = Path(update_script).expanduser().resolve()
    generate_script = Path(generate_script).expanduser().resolve()
    selected_python = choose_provider_python(python_executable)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_before = inspect_stock_db(db_path)
    snapshot_before = scoped_latest_symbol_count(db_path, snapshot_before, symbols)
    market_stats_artifact_before = inspect_market_stats_artifact()
    freshness_before = freshness_status(
        snapshot_before,
        staleness_days=staleness_days,
        min_latest_symbols=min_latest_symbols,
    )
    provider_ok = provider_available(python_executable=selected_python)
    update_command = build_update_command(
        db_path=db_path,
        source=source,
        rpm=rpm,
        max_errors=max_errors,
        timeout_seconds=timeout_seconds,
        end=end,
        symbols=symbols,
        python_executable=selected_python,
        update_script=update_script,
    )
    report: dict[str, Any] = {
        "workflow_id": WORKFLOW_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": freshness_before["status"],
        "check_only": bool(check_only),
        "force": bool(force),
        "strict": bool(strict),
        "provider": {
            "module": "vnstock_data",
            "available": bool(provider_ok),
            "python_executable": selected_python,
        },
        "freshness": freshness_before,
        "snapshot_before": snapshot_before.to_dict(),
        "snapshot_after": None,
        "market_stats_artifact_before": market_stats_artifact_before.to_dict(),
        "market_stats_artifact_after": None,
        "market_stats_artifact_needs_regeneration": market_stats_artifact_needs_regeneration(
            snapshot_before,
            market_stats_artifact_before,
        ),
        "update_command": update_command,
        "update_run": None,
        "market_stats_regeneration_run": None,
    }
    should_refresh = bool(force or freshness_before.get("is_stale"))
    if check_only:
        report["status"] = "CHECK_ONLY_" + str(freshness_before["status"])
    elif not should_refresh:
        report["status"] = "SKIPPED_FRESH"
    elif not provider_ok:
        report["status"] = "BLOCKED_MISSING_PROVIDER"
        report["reason"] = "Python environment cannot import vnstock_data, so Market Stats provider refresh cannot run."
    else:
        update_run = _run_command(update_command, cwd=MAIN_SONET_ROOT, timeout=int(command_timeout_seconds))
        report["update_run"] = update_run
        if int(update_run["returncode"]) != 0:
            report["status"] = "REFRESH_FAILED"
        else:
            snapshot_after = inspect_stock_db(db_path)
            snapshot_after = scoped_latest_symbol_count(db_path, snapshot_after, symbols)
            report["snapshot_after"] = snapshot_after.to_dict()
            freshness_after = freshness_status(
                snapshot_after,
                staleness_days=staleness_days,
                min_latest_symbols=min_latest_symbols,
            )
            report["freshness_after"] = freshness_after
            if freshness_after.get("is_stale"):
                report["status"] = "REFRESHED_STALE"
                report["reason"] = (
                    "Provider refresh completed, but the local OHLCV database is still older "
                    "than the configured freshness policy."
                )
            else:
                report["status"] = "REFRESHED"
    effective_snapshot = snapshot_before
    if isinstance(report.get("snapshot_after"), dict):
        snapshot_after_payload = report["snapshot_after"]
        effective_snapshot = StockDbSnapshot(
            db_path=str(snapshot_after_payload.get("db_path")),
            exists=bool(snapshot_after_payload.get("exists")),
            has_stock_price_history=bool(snapshot_after_payload.get("has_stock_price_history")),
            row_count=snapshot_after_payload.get("row_count"),
            min_date=snapshot_after_payload.get("min_date"),
            max_date=snapshot_after_payload.get("max_date"),
            symbol_count=snapshot_after_payload.get("symbol_count"),
            latest_date_symbol_count=snapshot_after_payload.get("latest_date_symbol_count"),
        )
    market_stats_artifact_current = inspect_market_stats_artifact()
    needs_regeneration = market_stats_artifact_needs_regeneration(effective_snapshot, market_stats_artifact_current)
    report["market_stats_artifact_needs_regeneration"] = needs_regeneration
    if not check_only and regenerate_market_stats and report["status"] not in FAIL_STATUSES and needs_regeneration:
        report["market_stats_regeneration_run"] = _run_command(
            [selected_python, str(generate_script)],
            cwd=MAIN_SONET_ROOT,
            timeout=int(command_timeout_seconds),
        )
        if int(report["market_stats_regeneration_run"]["returncode"]) != 0:
            report["status"] = "REGENERATION_FAILED"
    report["market_stats_artifact_after"] = inspect_market_stats_artifact().to_dict()
    report_path = out_dir / "data_refresh_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if strict and report["status"] in FAIL_STATUSES:
        raise RuntimeError(f"Realtime market data refresh failed: {report['status']}. See {report_path}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect/refresh the realtime scanner OHLCV cache.")
    parser.add_argument("--db", default=str(DEFAULT_STOCK_DB))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--source", default="VCI", choices=["VND", "VCI", "KBS", "MAS"])
    parser.add_argument("--rpm", type=int, default=180)
    parser.add_argument("--max-errors", type=int, default=80)
    parser.add_argument("--timeout-seconds", type=int, default=10)
    parser.add_argument("--command-timeout-seconds", type=int, default=3600)
    parser.add_argument("--staleness-days", type=int, default=0)
    parser.add_argument("--min-latest-symbols", type=int, default=0)
    parser.add_argument("--end")
    parser.add_argument("--symbol", action="append", default=[])
    parser.add_argument("--skip-regenerate-market-stats", action="store_true", help="Do not regenerate market_stats web artifacts after DB refresh.")
    parser.add_argument("--python-executable", help="Override the Python runtime used for vnstock_data refresh.")
    args = parser.parse_args(argv)
    report = refresh_realtime_market_data(
        db_path=Path(args.db),
        out_dir=Path(args.out_dir),
        check_only=bool(args.check_only),
        force=bool(args.force),
        strict=bool(args.strict),
        source=str(args.source),
        rpm=int(args.rpm),
        max_errors=int(args.max_errors),
        timeout_seconds=int(args.timeout_seconds),
        command_timeout_seconds=int(args.command_timeout_seconds),
        staleness_days=int(args.staleness_days),
        min_latest_symbols=int(args.min_latest_symbols),
        end=args.end,
        symbols=list(args.symbol) or None,
        regenerate_market_stats=not bool(args.skip_regenerate_market_stats),
        python_executable=args.python_executable,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["status"] in FAIL_STATUSES else 0


if __name__ == "__main__":
    raise SystemExit(main())
