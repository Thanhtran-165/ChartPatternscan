from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

try:
    from .pattern_set_metadata import base_metadata_for_pattern_set  # type: ignore
except Exception:  # pragma: no cover
    from pattern_set_metadata import base_metadata_for_pattern_set  # type: ignore


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:
        return None
    return v


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _latest_run_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT run_id FROM scanner_runs ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row:
        raise SystemExit("No scanner_runs found.")
    return str(row[0])


def _load_phase3_matrix(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["pattern_key"]): row for row in rows}


def _load_benchmark_matrix(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["pattern_key"]): row for row in rows}


def _pattern_metrics(db_path: Path) -> Dict[str, Dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        run_id = _latest_run_id(conn)
        det_rows = conn.execute(
            """
            SELECT pattern_name, symbol, COALESCE(variant_code, '<null>') AS variant_code
            FROM pattern_detections
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
        eval_rows = conn.execute(
            """
            SELECT
                d.pattern_name,
                d.symbol,
                p.max_favorable_excursion_pct,
                p.boundary_invalidated,
                p.target_achieved_intraday,
                p.throwback_pullback_occurred
            FROM pattern_detections d
            JOIN post_breakout_results p
              ON p.run_id = d.run_id AND p.pattern_id = d.pattern_id
            WHERE d.run_id = ?
            """,
            (run_id,),
        ).fetchall()
    finally:
        conn.close()

    out: Dict[str, Dict[str, Any]] = {}
    symbols_by_pattern: Dict[str, set[str]] = defaultdict(set)
    for row in det_rows:
        pat = str(row["pattern_name"])
        cur = out.setdefault(pat, {"detections": 0})
        cur["detections"] = int(cur["detections"]) + 1
        symbols_by_pattern[pat].add(str(row["symbol"]))

    buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"move": [], "boundary": [], "target": [], "tbpb": []})
    eval_symbols: Dict[str, set[str]] = defaultdict(set)
    for row in eval_rows:
        pat = str(row["pattern_name"])
        eval_symbols[pat].add(str(row["symbol"]))
        move = _safe_float(row["max_favorable_excursion_pct"])
        if move is not None:
            buckets[pat]["move"].append(move)
        for col, key in (
            ("boundary_invalidated", "boundary"),
            ("target_achieved_intraday", "target"),
            ("throwback_pullback_occurred", "tbpb"),
        ):
            val = _safe_float(row[col])
            if val is not None:
                buckets[pat][key].append(val)

    for pat in set(out) | set(buckets):
        cur = out.setdefault(pat, {"detections": 0})
        cur["symbol_count"] = len(symbols_by_pattern.get(pat, set()))
        cur["eval_symbol_count"] = len(eval_symbols.get(pat, set()))
        moves = buckets.get(pat, {}).get("move", [])
        boundary = buckets.get(pat, {}).get("boundary", [])
        target = buckets.get(pat, {}).get("target", [])
        tbpb = buckets.get(pat, {}).get("tbpb", [])
        cur["evals"] = len(moves) or max(len(boundary), len(target), len(tbpb))
        cur["median_move_pct"] = float(median(moves)) if moves else None
        cur["failure_rate_5pct"] = (sum(1.0 for x in moves if float(x) < 5.0) / len(moves) * 100.0) if moves else None
        cur["boundary_pct"] = (sum(boundary) / len(boundary) * 100.0) if boundary else None
        cur["target_hit_pct"] = (sum(target) / len(target) * 100.0) if target else None
        cur["tbpb_pct"] = (sum(tbpb) / len(tbpb) * 100.0) if tbpb else None
    return out


def _family_aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    family_map: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row["canonical_key"])
        valid = row.get("valid") or {}
        calib = row.get("calib") or {}
        cur = family_map.setdefault(
            key,
            {
                "canonical_key": key,
                "family_label": str(row["family_label"]),
                "pattern_count": 0,
                "valid_detections": 0,
                "valid_evals": 0,
                "calib_detections": 0,
                "calib_evals": 0,
                "valid_symbols": 0,
            },
        )
        cur["pattern_count"] += 1
        cur["valid_detections"] += int(valid.get("detections") or 0)
        cur["valid_evals"] += int(valid.get("evals") or 0)
        cur["calib_detections"] += int(calib.get("detections") or 0)
        cur["calib_evals"] += int(calib.get("evals") or 0)
        cur["valid_symbols"] += int(valid.get("symbol_count") or 0)
    return sorted(family_map.values(), key=lambda x: (-int(x["valid_evals"]), -int(x["valid_detections"]), str(x["canonical_key"])))


def build_report(
    *,
    valid_db: Path,
    calib_db: Path,
    phase3_matrix: Path,
    benchmark_matrix: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    meta = base_metadata_for_pattern_set("bulkowski_53_strict")
    phase3 = _load_phase3_matrix(phase3_matrix)
    benchmark = _load_benchmark_matrix(benchmark_matrix)
    valid = _pattern_metrics(valid_db)
    calib = _pattern_metrics(calib_db)

    rows: List[Dict[str, Any]] = []
    for pattern_key, pattern_meta in sorted(meta.items()):
        p3 = phase3.get(pattern_key, {})
        bm = benchmark.get(pattern_key, {})
        valid_row = valid.get(pattern_key, {})
        calib_row = calib.get(pattern_key, {})
        rows.append(
            {
                "pattern_key": pattern_key,
                "bulkowski_name": pattern_meta.get("bulkowski_name"),
                "canonical_key": pattern_meta.get("canonical_key"),
                "family_label": pattern_meta.get("canonical_key"),
                "chapter": pattern_meta.get("bulkowski_chapter"),
                "phase3_status": p3.get("phase3_status"),
                "strategy_gate": p3.get("strategy_gate"),
                "benchmark_status": bm.get("benchmark_status"),
                "valid": valid_row,
                "calib": calib_row,
            }
        )

    families = _family_aggregate(rows)
    rows_by_valid_evals = sorted(rows, key=lambda row: (-int((row["valid"] or {}).get("evals") or 0), -int((row["valid"] or {}).get("detections") or 0), str(row["pattern_key"])))
    rows_by_valid_symbols = sorted(rows, key=lambda row: (-int((row["valid"] or {}).get("symbol_count") or 0), -int((row["valid"] or {}).get("detections") or 0), str(row["pattern_key"])))
    rows_ex_gaps = [row for row in rows if str(row["pattern_key"]) != "gaps"]
    rows_by_strength = sorted(
        [row for row in rows_ex_gaps if int((row["valid"] or {}).get("evals") or 0) >= 20],
        key=lambda row: (
            -float((row["valid"] or {}).get("median_move_pct") or -1e9),
            float((row["valid"] or {}).get("failure_rate_5pct") or 1e9),
            -int((row["valid"] or {}).get("evals") or 0),
        ),
    )

    payload = {
        "summary": {
            "valid_db": str(valid_db.resolve()),
            "calib_db": str(calib_db.resolve()),
            "pattern_count": len(rows),
            "family_count": len(families),
            "phase3_status_counts": {
                key: sum(1 for row in rows if str(row.get("phase3_status")) == key)
                for key in sorted({str(row.get("phase3_status")) for row in rows})
            },
            "benchmark_status_counts": {
                key: sum(1 for row in rows if str(row.get("benchmark_status")) == key)
                for key in sorted({str(row.get("benchmark_status")) for row in rows})
            },
        },
        "top_patterns_by_valid_evals": rows_by_valid_evals[:20],
        "top_patterns_by_symbol_count": rows_by_valid_symbols[:20],
        "top_patterns_by_strength_ex_gaps": rows_by_strength[:20],
        "family_prevalence": families[:20],
        "candidate_and_watchlist": [row for row in rows if str(row.get("strategy_gate")) in {"candidate", "watchlist"}],
        "pattern_matrix": rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "vietnam_research_report.json", payload)
    _write_text(out_dir / "vietnam_research_report.md", _render(payload))
    return payload


def _render(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    summary = payload["summary"]
    lines.append("# Vietnam Pattern Research Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- pattern_count: `{summary['pattern_count']}`")
    lines.append(f"- family_count: `{summary['family_count']}`")
    lines.append(f"- phase3_status_counts: `{summary['phase3_status_counts']}`")
    lines.append(f"- benchmark_status_counts: `{summary['benchmark_status_counts']}`")
    lines.append("")

    lines.append("## Candidate / Watchlist")
    lines.append("")
    lines.append("| Pattern | Family | Phase 3 | Strategy | Valid evals | Valid move | Fail<5 | Target |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for row in payload["candidate_and_watchlist"]:
        valid = row["valid"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["pattern_key"]),
                    str(row["canonical_key"]),
                    str(row["phase3_status"]),
                    str(row["strategy_gate"]),
                    str(int(valid.get("evals") or 0)),
                    _fmt(valid.get("median_move_pct")),
                    _fmt(valid.get("failure_rate_5pct")),
                    _fmt(valid.get("target_hit_pct")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Common Patterns In Vietnam (By Valid Evals)")
    lines.append("")
    lines.append("| Pattern | Family | Valid evals | Symbols | Move | Fail<5 | Target | Benchmark |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in payload["top_patterns_by_valid_evals"]:
        valid = row["valid"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["pattern_key"]),
                    str(row["canonical_key"]),
                    str(int(valid.get("evals") or 0)),
                    str(int(valid.get("eval_symbol_count") or 0)),
                    _fmt(valid.get("median_move_pct")),
                    _fmt(valid.get("failure_rate_5pct")),
                    _fmt(valid.get("target_hit_pct")),
                    str(row.get("benchmark_status")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Common Patterns In Vietnam (By Symbol Coverage)")
    lines.append("")
    lines.append("| Pattern | Family | Symbols | Detections | Valid evals | Move |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in payload["top_patterns_by_symbol_count"]:
        valid = row["valid"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["pattern_key"]),
                    str(row["canonical_key"]),
                    str(int(valid.get("symbol_count") or 0)),
                    str(int(valid.get("detections") or 0)),
                    str(int(valid.get("evals") or 0)),
                    _fmt(valid.get("median_move_pct")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Stronger Patterns Excluding Gaps")
    lines.append("")
    lines.append("| Pattern | Family | Valid evals | Move | Fail<5 | Target | Strategy |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for row in payload["top_patterns_by_strength_ex_gaps"]:
        valid = row["valid"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["pattern_key"]),
                    str(row["canonical_key"]),
                    str(int(valid.get("evals") or 0)),
                    _fmt(valid.get("median_move_pct")),
                    _fmt(valid.get("failure_rate_5pct")),
                    _fmt(valid.get("target_hit_pct")),
                    str(row.get("strategy_gate")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Family Prevalence")
    lines.append("")
    lines.append("| Family | Pattern count | Valid detections | Valid evals | Calib detections | Calib evals |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in payload["family_prevalence"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["canonical_key"]),
                    str(int(row["pattern_count"])),
                    str(int(row["valid_detections"])),
                    str(int(row["valid_evals"])),
                    str(int(row["calib_detections"])),
                    str(int(row["calib_evals"])),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-db", required=True)
    parser.add_argument("--calib-db", required=True)
    parser.add_argument("--phase3-pattern-matrix", required=True)
    parser.add_argument("--benchmark-pattern-matrix", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    payload = build_report(
        valid_db=Path(args.valid_db).resolve(),
        calib_db=Path(args.calib_db).resolve(),
        phase3_matrix=Path(args.phase3_pattern_matrix).resolve(),
        benchmark_matrix=Path(args.benchmark_pattern_matrix).resolve(),
        out_dir=Path(args.out_dir).resolve(),
    )
    print("=== Vietnam Pattern Research Report ===")
    print(f"out_dir: {Path(args.out_dir).resolve()}")
    print(f"phase3_status_counts: {payload['summary']['phase3_status_counts']}")
    print(f"benchmark_status_counts: {payload['summary']['benchmark_status_counts']}")


if __name__ == "__main__":
    main()
