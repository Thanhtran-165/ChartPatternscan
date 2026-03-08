from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _latest_run_stats(db_path: Path) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT run_id FROM scanner_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            raise SystemExit(f"No scanner_runs found in {db_path}")
        run_id = str(row[0])
        detections = int(
            conn.execute("SELECT COUNT(*) FROM pattern_detections WHERE run_id = ?", (run_id,)).fetchone()[0]
        )
        evals = int(
            conn.execute("SELECT COUNT(*) FROM post_breakout_results WHERE run_id = ?", (run_id,)).fetchone()[0]
        )
        return {
            "db_path": str(db_path.resolve()),
            "run_id": run_id,
            "detections": detections,
            "evals": evals,
        }
    finally:
        conn.close()


def _row_by_name(rows: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for row in rows:
        if str(row.get("name")) == name:
            return row
    return {}


def _variant_rows(payload: Dict[str, Any], split: str, pattern: str) -> List[Dict[str, Any]]:
    rows = payload.get(split, {}).get("rows", [])
    return [row for row in rows if str(row.get("pattern_name")) == pattern]


def build_snapshot(
    *,
    final_valid_db: Path,
    final_calib_db: Path,
    candidate_summary: Path,
    measured_report: Path,
    islands_report: Path,
    gaps_report: Path,
    benchmark_summary: Path,
    phase3_summary: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    final_valid = _latest_run_stats(final_valid_db)
    final_calib = _latest_run_stats(final_calib_db)
    candidate = _read_json(candidate_summary)
    measured = _read_json(measured_report)
    islands = _read_json(islands_report)
    gaps = _read_json(gaps_report)
    benchmark = _read_json(benchmark_summary)
    phase3 = _read_json(phase3_summary)

    payload: Dict[str, Any] = {
        "final_valid": final_valid,
        "final_calib": final_calib,
        "broadening_bottoms": {
            "all": _row_by_name(candidate.get("cohorts", []), "all"),
            "narrower_core": _row_by_name(candidate.get("cohorts", []), "narrower_core"),
        },
        "measured_move": {
            "valid": _variant_rows(measured, "valid", "measured_move_up") + _variant_rows(measured, "valid", "measured_move_down"),
            "calib": _variant_rows(measured, "calib", "measured_move_up") + _variant_rows(measured, "calib", "measured_move_down"),
        },
        "islands": {
            "valid": _variant_rows(islands, "valid", "island_reversals") + _variant_rows(islands, "valid", "islands_long"),
            "calib": _variant_rows(islands, "calib", "island_reversals") + _variant_rows(islands, "calib", "islands_long"),
        },
        "gaps": {
            "valid": _variant_rows(gaps, "valid", "gaps"),
            "calib": _variant_rows(gaps, "calib", "gaps"),
        },
        "benchmark": benchmark,
        "phase3": phase3,
        "source_reports": {
            "candidate_summary": str(candidate_summary.resolve()),
            "measured_report": str(measured_report.resolve()),
            "islands_report": str(islands_report.resolve()),
            "gaps_report": str(gaps_report.resolve()),
            "benchmark_summary": str(benchmark_summary.resolve()),
            "phase3_summary": str(phase3_summary.resolve()),
        },
    }

    _write_json(out_dir / "final_overhaul_snapshot.json", payload)
    _write_text(out_dir / "final_overhaul_snapshot.md", _render(payload))
    return payload


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _render(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Final Overhaul Snapshot")
    lines.append("")
    lines.append("## Unified Runs")
    lines.append("")
    lines.append(f"- valid_run_id: `{payload['final_valid']['run_id']}`")
    lines.append(f"- valid: `{payload['final_valid']['detections']} detections / {payload['final_valid']['evals']} evals`")
    lines.append(f"- calib_run_id: `{payload['final_calib']['run_id']}`")
    lines.append(f"- calib: `{payload['final_calib']['detections']} detections / {payload['final_calib']['evals']} evals`")
    lines.append("")
    lines.append("## Broadening Bottoms")
    lines.append("")
    lines.append("| Cohort | Split | Evals | Move | Fail<5 | Target |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for name in ("all", "narrower_core"):
        row = payload["broadening_bottoms"][name]
        for split in ("valid", "calib"):
            stat = row.get(split, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        split,
                        str(int(stat.get("evals") or 0)),
                        _fmt(stat.get("median_move_pct")),
                        _fmt(stat.get("failure_rate_5pct")),
                        _fmt(stat.get("target_hit_pct")),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("## Measured Move Rewrite")
    lines.append("")
    lines.append("| Split | Pattern | Variant | Detections | Evals | Move | Fail<5 | Target |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for split in ("valid", "calib"):
        for row in payload["measured_move"][split]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        split,
                        str(row.get("pattern_name")),
                        str(row.get("variant_code")),
                        str(int(row.get("detections") or 0)),
                        str(int(row.get("evals") or 0)),
                        _fmt(row.get("median_move_pct")),
                        _fmt(row.get("failure_rate_5pct")),
                        _fmt(row.get("target_hit_pct")),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("## Islands Recalibration")
    lines.append("")
    lines.append("| Split | Pattern | Variant | Detections | Evals | Move | Fail<5 | Target |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for split in ("valid", "calib"):
        for row in payload["islands"][split]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        split,
                        str(row.get("pattern_name")),
                        str(row.get("variant_code")),
                        str(int(row.get("detections") or 0)),
                        str(int(row.get("evals") or 0)),
                        _fmt(row.get("median_move_pct")),
                        _fmt(row.get("failure_rate_5pct")),
                        _fmt(row.get("target_hit_pct")),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("## Gaps Stratification")
    lines.append("")
    lines.append("| Split | Variant | Detections | Evals | Move | Fail<5 | Target |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for split in ("valid", "calib"):
        rows = sorted(payload["gaps"][split], key=lambda row: int(row.get("detections") or 0), reverse=True)
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        split,
                        str(row.get("variant_code")),
                        str(int(row.get("detections") or 0)),
                        str(int(row.get("evals") or 0)),
                        _fmt(row.get("median_move_pct")),
                        _fmt(row.get("failure_rate_5pct")),
                        _fmt(row.get("target_hit_pct")),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("## Governance / Benchmark")
    lines.append("")
    lines.append(f"- benchmark_status_counts: `{payload['benchmark'].get('status_counts')}`")
    lines.append(f"- phase3_status_counts: `{payload['phase3'].get('phase3_status_counts')}`")
    lines.append(f"- strategy_gate_counts: `{payload['phase3'].get('strategy_gate_counts')}`")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-valid-db", required=True)
    parser.add_argument("--final-calib-db", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--measured-report", required=True)
    parser.add_argument("--islands-report", required=True)
    parser.add_argument("--gaps-report", required=True)
    parser.add_argument("--benchmark-summary", required=True)
    parser.add_argument("--phase3-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    payload = build_snapshot(
        final_valid_db=Path(args.final_valid_db).resolve(),
        final_calib_db=Path(args.final_calib_db).resolve(),
        candidate_summary=Path(args.candidate_summary).resolve(),
        measured_report=Path(args.measured_report).resolve(),
        islands_report=Path(args.islands_report).resolve(),
        gaps_report=Path(args.gaps_report).resolve(),
        benchmark_summary=Path(args.benchmark_summary).resolve(),
        phase3_summary=Path(args.phase3_summary).resolve(),
        out_dir=Path(args.out_dir).resolve(),
    )
    print("=== Final Overhaul Snapshot ===")
    print(f"out_dir: {Path(args.out_dir).resolve()}")
    print(f"valid_run_id: {payload['final_valid']['run_id']}")
    print(f"calib_run_id: {payload['final_calib']['run_id']}")


if __name__ == "__main__":
    main()
