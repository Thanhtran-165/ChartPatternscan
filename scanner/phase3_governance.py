from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_PATTERN_OUTPUT = Path("scan_results/spec_audit_20260306/phase3")


TARGETED_PATCHES = [
    {
        "patterns": [
            "double_bottoms_adam_adam",
            "double_bottoms_adam_eve",
            "double_bottoms_eve_adam",
            "double_bottoms_eve_eve",
            "double_tops_adam_adam",
            "double_tops_adam_eve",
            "double_tops_eve_adam",
            "double_tops_eve_eve",
        ],
        "valid_db": Path("scan_results/double_family_pass4_valid_20260308.sqlite"),
        "calib_db": Path("scan_results/double_family_pass4_calib_20260308.sqlite"),
        "checkpoint": "double_family_pass4",
    },
    {
        "patterns": [
            "head_and_shoulders_tops",
            "head_and_shoulders_tops_complex",
        ],
        "valid_db": Path("scan_results/hs_family_valid_refactor_v2_20260306.sqlite"),
        "calib_db": Path("scan_results/hs_family_calib_refactor_v2_20260306.sqlite"),
        "checkpoint": "hs_batch2_refactor_v2",
    },
    {
        "patterns": [
            "head_and_shoulders_bottoms",
            "head_and_shoulders_bottoms_complex",
        ],
        "valid_db": Path("scan_results/hs_bottoms_pass3_valid_20260308.sqlite"),
        "calib_db": Path("scan_results/hs_bottoms_pass3_calib_20260308.sqlite"),
        "checkpoint": "hs_bottoms_pass3",
    },
    {
        "patterns": [
            "triangles_ascending",
            "triangles_descending",
            "triangles_symmetrical",
        ],
        "valid_db": Path("scan_results/triangles_family_valid_refactor_20260306.sqlite"),
        "calib_db": Path("scan_results/triangles_family_calib_refactor_20260306.sqlite"),
        "checkpoint": "triangles_batch3_refactor",
    },
    {
        "patterns": [
            "scallops_ascending",
            "scallops_ascending_inverted",
            "scallops_descending",
            "scallops_descending_inverted",
        ],
        "valid_db": Path("scan_results/scallops_family_pass3_valid_20260308.sqlite"),
        "calib_db": Path("scan_results/scallops_family_pass3_calib_20260308.sqlite"),
        "checkpoint": "scallops_family_pass3",
    },
    {
        "patterns": [
            "cup_with_handle",
            "cup_with_handle_inverted",
        ],
        "valid_db": Path("scan_results/post_phase3_backlog_valid_20260306.sqlite"),
        "calib_db": Path("scan_results/post_phase3_backlog_calib_20260306.sqlite"),
        "checkpoint": "post_phase3_cup_round2",
    },
    {
        "patterns": [
            "broadening_wedges_ascending",
            "broadening_wedges_descending",
        ],
        "valid_db": Path("scan_results/bw_family_valid_refactor_20260306.sqlite"),
        "calib_db": Path("scan_results/bw_family_calib_refactor_20260306.sqlite"),
        "checkpoint": "bw_family_batch6_refactor",
    },
    {
        "patterns": [
            "horn_bottoms",
            "horn_tops",
        ],
        "valid_db": Path("scan_results/horn_family_valid_refactor_20260306.sqlite"),
        "calib_db": Path("scan_results/horn_family_calib_refactor_20260306.sqlite"),
        "checkpoint": "horn_family_batch7_refactor",
    },
    {
        "patterns": [
            "pipe_bottoms",
            "pipe_tops",
        ],
        "valid_db": Path("scan_results/pipe_family_valid_refactor_20260306.sqlite"),
        "calib_db": Path("scan_results/pipe_family_calib_refactor_20260306.sqlite"),
        "checkpoint": "pipe_family_batch8_refactor",
    },
    {
        "patterns": [
            "rounding_bottoms",
            "rounding_tops",
        ],
        "valid_db": Path("scan_results/post_phase3_backlog_valid_20260306.sqlite"),
        "calib_db": Path("scan_results/post_phase3_backlog_calib_20260306.sqlite"),
        "checkpoint": "post_phase3_rounding_round2",
    },
    {
        "patterns": [
            "triple_bottoms",
            "triple_tops",
        ],
        "valid_db": Path("scan_results/triple_family_valid_refactor_20260306.sqlite"),
        "calib_db": Path("scan_results/triple_family_calib_refactor_20260306.sqlite"),
        "checkpoint": "triple_family_batch10_refactor",
    },
    {
        "patterns": [
            "flags",
            "flags_high_tight",
        ],
        "valid_db": Path("scan_results/flags_family_valid_refactor_20260306.sqlite"),
        "calib_db": Path("scan_results/flags_family_calib_refactor_20260306.sqlite"),
        "checkpoint": "flags_family_batch11_refactor",
    },
]


PHASE3_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "double_bottoms_adam_adam": {
        "strategy_gate": "watchlist",
        "manual_note": "Pass 4 keeps AA as the densest double-bottom branch while allowing only extra-rounded midpoint shapes to drift toward Eve. The AA branch remains watchlist-only, not a production strategy input.",
        "checkpoint_sources": ["double_family_pass4"],
    },
    "double_tops_adam_adam": {
        "strategy_gate": "watchlist",
        "manual_note": "Pass 4 keeps AA as the dominant double-top branch while refusing to revive weak midpoint-width Adam calls. It still owns the watchlist, but the family benchmark drift remains too large for promotion.",
        "checkpoint_sources": ["double_family_pass4"],
    },
    "double_bottoms_eve_adam": {
        "phase3_status": "research_only",
        "manual_note": "EA still has no surviving branch-level evidence after Pass 4, but it remains part of the reference taxonomy instead of being retired outright.",
        "checkpoint_sources": ["double_family_pass4"],
    },
    "double_bottoms_eve_eve": {
        "phase3_status": "recalibrate",
        "research_lane": "recalibration_backlog",
        "manual_note": "EE still only produces a calibration-side remnant after Pass 4. Keep it in recalibration backlog rather than treating it as usable research coverage.",
        "checkpoint_sources": ["double_family_pass4"],
    },
    "double_tops_eve_eve": {
        "phase3_status": "research_only",
        "manual_note": "Top-side EE survives only as a very thin reference branch after Pass 4. Keep it in the research layer, not in strategy work.",
        "checkpoint_sources": ["double_family_pass4"],
    },
    "head_and_shoulders_tops": {
        "phase3_status": "research_only",
        "manual_note": "Batch 2 made the standard/complex split defensible, but tops still remain a research-only family.",
        "checkpoint_sources": ["hs_batch2_refactor_v2"],
    },
    "head_and_shoulders_tops_complex": {
        "phase3_status": "research_only",
        "manual_note": "Complex tops remain research-only after the family refactor.",
        "checkpoint_sources": ["hs_batch2_refactor_v2"],
    },
    "head_and_shoulders_bottoms": {
        "phase3_status": "research_only",
        "manual_note": "The neckline-specific waiver keeps inverse H&S bottoms research-only: calibration still has only one standard survivor, and valid remains complex-only. The branch is cleaner, but still too thin for strategy work.",
        "checkpoint_sources": ["hs_bottoms_pass3"],
    },
    "head_and_shoulders_bottoms_complex": {
        "phase3_status": "research_only",
        "manual_note": "Complex inverse H&S still contains only a tiny valid-side remainder after the neckline-specific pass. It belongs in the research lane, not in strategy work.",
        "checkpoint_sources": ["hs_bottoms_pass3"],
    },
    "triangles_ascending": {
        "phase3_status": "research_only",
        "manual_note": "Batch 3 restored credible ascending-triangle structure and removed the earlier retire decision, but the refactored branch still shows weak valid-side performance. Keep it in the research layer.",
        "checkpoint_sources": ["triangles_batch3_refactor"],
    },
    "triangles_descending": {
        "phase3_status": "research_only",
        "manual_note": "Descending triangles now look structurally defensible after Batch 3, but valid-side KPI drift remains too large for strategy use.",
        "checkpoint_sources": ["triangles_batch3_refactor"],
    },
    "triangles_symmetrical": {
        "phase3_status": "research_only",
        "manual_note": "Symmetrical triangles survived the family refactor with plausible geometry, yet the calibration-valid drift still blocks promotion beyond research.",
        "checkpoint_sources": ["triangles_batch3_refactor"],
    },
    "scallops_ascending": {
        "phase3_status": "research_only",
        "manual_note": "The third scallop pass keeps the bullish ascending branch dense and stable while pushing more mixed noise into the weaker bearish branches. It remains research-only because the family still overlaps rounding semantics.",
        "checkpoint_sources": ["scallops_family_pass3"],
    },
    "scallops_ascending_inverted": {
        "phase3_status": "recalibrate",
        "research_lane": "recalibration_backlog",
        "manual_note": "The third branch gate cut ascending-inverted detections again, but calibration remains too weak and unstable. Keep it in the recalibration backlog.",
        "checkpoint_sources": ["scallops_family_pass3"],
    },
    "scallops_descending": {
        "phase3_status": "recalibrate",
        "research_lane": "recalibration_backlog",
        "manual_note": "The third scallop pass removed another slice of descending false positives, but bearish KPI quality is still too weak for promotion beyond recalibration backlog.",
        "checkpoint_sources": ["scallops_family_pass3"],
    },
    "scallops_descending_inverted": {
        "phase3_status": "research_only",
        "manual_note": "Descending-inverted remains the cleanest reverse-bullish scallop branch after the third pass, with materially better stability than the other mixed scallop branches. Keep it research-only.",
        "checkpoint_sources": ["scallops_family_pass3"],
    },
    "cup_with_handle": {
        "phase3_status": "research_only",
        "manual_note": "The bullish cup branch stays intact after the post-phase-3 pass and remains one of the cleaner research families, but its sample is still too thin for strategy use.",
        "checkpoint_sources": ["post_phase3_cup_round2"],
    },
    "cup_with_handle_inverted": {
        "phase3_status": "retire_from_strategy",
        "strategy_gate": "retired",
        "research_lane": "reference_only",
        "manual_note": "The original-space handle gate reduced inverted cups to only two valid-side survivors and zero calibration survivors. Move this branch to reference-only until a new bearish cup detector earns support.",
        "checkpoint_sources": ["post_phase3_cup_round2"],
    },
    "broadening_wedges_ascending": {
        "phase3_status": "research_only",
        "manual_note": "Batch 6 replaces the loose slope splitter with a same-direction diverging-boundary detector. The surviving ascending wedges now look structurally plausible and belong in the research layer.",
        "checkpoint_sources": ["bw_family_batch6_refactor"],
    },
    "broadening_wedges_descending": {
        "phase3_status": "research_only",
        "manual_note": "Descending broadening wedges survived Batch 6 with coherent divergence and acceptable eval volume, but they still need more benchmarking before strategy use.",
        "checkpoint_sources": ["bw_family_batch6_refactor"],
    },
    "horn_bottoms": {
        "phase3_status": "research_only",
        "manual_note": "Batch 7 restored horn semantics to short, sharp V-shaped reversals instead of breakout-direction proxies. Bottoms now look good enough for research, not production signals.",
        "checkpoint_sources": ["horn_family_batch7_refactor"],
    },
    "horn_tops": {
        "phase3_status": "research_only",
        "manual_note": "Horn tops now come from a proper spike-reversal detector with prior uptrend and symmetric horns. The family is clean enough for research only.",
        "checkpoint_sources": ["horn_family_batch7_refactor"],
    },
    "pipe_bottoms": {
        "phase3_status": "research_only",
        "manual_note": "Batch 8 keeps only compact, balanced pipe bottoms with timely breakouts. The branch looks materially cleaner and strong enough for research benchmarking.",
        "checkpoint_sources": ["pipe_family_batch8_refactor"],
    },
    "pipe_tops": {
        "phase3_status": "research_only",
        "manual_note": "Pipe tops now require symmetric vertical legs and prompt downside confirmation, which removes much of the old noise. Keep them in research only for now.",
        "checkpoint_sources": ["pipe_family_batch8_refactor"],
    },
    "rounding_bottoms": {
        "phase3_status": "research_only",
        "manual_note": "Rounding bottoms remain structurally credible after the top-side tightening pass and stay in the research lane.",
        "checkpoint_sources": ["post_phase3_rounding_round2"],
    },
    "rounding_tops": {
        "phase3_status": "research_only",
        "manual_note": "The second top-side pass cut rounding tops sharply while preserving a usable research sample. Keep the branch in research-only rather than moving it back into recalibration backlog.",
        "checkpoint_sources": ["post_phase3_rounding_round2"],
    },
    "triple_bottoms": {
        "phase3_status": "research_only",
        "manual_note": "Batch 10 converts triple bottoms from a loose pivot proxy into a flat-boundary, near-equal three-touch detector. The branch is now credible research coverage.",
        "checkpoint_sources": ["triple_family_batch10_refactor"],
    },
    "triple_tops": {
        "phase3_status": "research_only",
        "manual_note": "Triple tops now require flat, near-equal highs and balanced spacing. They remain noisier than bottoms, but still clear the bar for research-only use.",
        "checkpoint_sources": ["triple_family_batch10_refactor"],
    },
    "flags": {
        "phase3_status": "research_only",
        "manual_note": "Batch 11 reintroduces the missing flagpole and parallel-channel semantics. The surviving flags look much closer to textbook continuation structures and belong in the research lane.",
        "checkpoint_sources": ["flags_family_batch11_refactor"],
    },
    "flags_high_tight": {
        "phase3_status": "retire_from_strategy",
        "strategy_gate": "retired",
        "research_lane": "reference_only",
        "manual_note": "The dedicated flag-family pass no longer leaves surviving high-tight flags on this dataset. Keep the chapter as reference-only until a separate detector earns support.",
        "checkpoint_sources": ["flags_family_batch11_refactor"],
    },
}


NEXT_BATCH_PRIORITY = [
    {
        "canonical_key": "double_bottoms",
        "reason": "Double bottoms remain on the watchlist after the second resolver pass, but only AA has meaningful density and the Eve-side variants still need more evidence.",
    },
    {
        "canonical_key": "double_tops",
        "reason": "Double tops mirror the same post-phase-3 watchlist state as double bottoms: AA is usable for benchmarking, but the non-AA branches remain too thin for promotion.",
    },
    {
        "canonical_key": "scallop_ascending_descending",
        "reason": "Scallops improved materially, yet ascending-inverted and descending still diverge from the cleaner bullish branches and need another recalibration pass.",
    },
    {
        "canonical_key": "head_and_shoulders_bottom",
        "reason": "Inverse head-and-shoulders still has only one calibration-side standard survivor, and recall-search smoke found just one clean near-miss blocked by neckline slope. The next pass should stay neckline-specific, not broaden the whole detector.",
    },
    {
        "canonical_key": "rounding_bottoms_tops",
        "reason": "Rounding tops are cleaner after the second pass, but the family is still a good candidate for later benchmark tuning once the higher-risk watchlist families are addressed.",
    },
]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, default=str)


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


def _fmt_pct(x: Any) -> str:
    v = _safe_float(x)
    return "" if v is None else f"{v:.2f}"


def _load_metrics(results_db_path: Path) -> Dict[str, Dict[str, Any]]:
    if not results_db_path.exists():
        return {}
    conn = sqlite3.connect(str(results_db_path))
    try:
        row = conn.execute("SELECT run_id FROM scanner_runs ORDER BY created_at DESC LIMIT 1").fetchone()
        if not row:
            return {}
        run_id = str(row[0])

        detections: Dict[str, Dict[str, Any]] = {}
        for pat, det, conf in conn.execute(
            """
            SELECT
                pattern_name,
                COUNT(*) AS detections,
                SUM(CASE WHEN breakout_date IS NOT NULL AND breakout_price IS NOT NULL THEN 1 ELSE 0 END) AS confirmed
            FROM pattern_detections
            WHERE run_id = ?
            GROUP BY pattern_name
            """,
            (run_id,),
        ).fetchall():
            detections[str(pat)] = {
                "detections": int(det or 0),
                "confirmed": int(conf or 0),
            }

        eval_rows: Dict[str, List[tuple[Any, Any, Any, Any]]] = defaultdict(list)
        for row in conn.execute(
            """
            SELECT
                pattern_name,
                max_favorable_excursion_pct,
                boundary_invalidated,
                target_achieved_intraday,
                throwback_pullback_occurred
            FROM post_breakout_results
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall():
            eval_rows[str(row[0])].append(row[1:])

        out: Dict[str, Dict[str, Any]] = {}
        for pat in sorted(set(detections) | set(eval_rows)):
            base = dict(detections.get(pat, {}))
            rows = eval_rows.get(pat, [])
            moves = [_safe_float(r[0]) for r in rows if _safe_float(r[0]) is not None]
            boundary = [_safe_float(r[1]) for r in rows if _safe_float(r[1]) is not None]
            target = [_safe_float(r[2]) for r in rows if _safe_float(r[2]) is not None]
            tbpb = [_safe_float(r[3]) for r in rows if _safe_float(r[3]) is not None]

            base["evals"] = len(rows)
            base["median_move_pct"] = float(median(moves)) if moves else None
            base["failure_rate_5pct"] = (sum(1 for x in moves if float(x) < 5.0) / len(moves) * 100.0) if moves else None
            base["boundary_pct"] = (sum(float(x) for x in boundary) / len(boundary) * 100.0) if boundary else None
            base["target_hit_pct"] = (sum(float(x) for x in target) / len(target) * 100.0) if target else None
            base["tbpb_pct"] = (sum(float(x) for x in tbpb) / len(tbpb) * 100.0) if tbpb else None
            out[pat] = base
        return out
    finally:
        conn.close()


def _default_lane(status: str) -> tuple[str, str]:
    if status == "candidate_after_review":
        return "benchmark_candidate", "candidate"
    if status == "recalibrate":
        return "recalibration_backlog", "blocked"
    if status == "research_only":
        return "active_research", "blocked"
    return "reference_only", "retired"


def _drift_flags(calib: Dict[str, Any], valid: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    move_c = _safe_float(calib.get("median_move_pct"))
    move_v = _safe_float(valid.get("median_move_pct"))
    if move_c is not None and move_v is not None and abs(move_v - move_c) >= 5.0:
        flags.append("move_drift")

    fail_c = _safe_float(calib.get("failure_rate_5pct"))
    fail_v = _safe_float(valid.get("failure_rate_5pct"))
    if fail_c is not None and fail_v is not None and abs(fail_v - fail_c) >= 10.0:
        flags.append("fail5_drift")

    tgt_c = _safe_float(calib.get("target_hit_pct"))
    tgt_v = _safe_float(valid.get("target_hit_pct"))
    if tgt_c is not None and tgt_v is not None and abs(tgt_v - tgt_c) >= 15.0:
        flags.append("target_drift")

    boundary_c = _safe_float(calib.get("boundary_pct"))
    boundary_v = _safe_float(valid.get("boundary_pct"))
    if boundary_c is not None and boundary_v is not None and abs(boundary_v - boundary_c) >= 15.0:
        flags.append("boundary_drift")
    return flags


def _family_phase3_status(group: List[Dict[str, Any]]) -> str:
    strategy_gates = {str(r["strategy_gate"]) for r in group}
    statuses = {str(r["phase3_status"]) for r in group}
    if "candidate" in strategy_gates:
        return "candidate_family"
    if "watchlist" in strategy_gates:
        return "watchlist_family"
    if "recalibrate" in statuses:
        return "recalibration_family"
    if strategy_gates == {"retired"}:
        return "reference_only_family"
    return "research_family"


def _family_action(group: List[Dict[str, Any]]) -> str:
    strategy_gates = {str(r["strategy_gate"]) for r in group}
    statuses = {str(r["phase3_status"]) for r in group}
    if "candidate" in strategy_gates:
        return "benchmark_then_strategy"
    if "watchlist" in strategy_gates:
        return "family_recalibration_then_watchlist"
    if "recalibrate" in statuses:
        return "family_recalibration"
    if strategy_gates == {"retired"}:
        return "reference_only"
    return "keep_research_only"


def _priority_score(group: List[Dict[str, Any]]) -> int:
    evals = sum(int(r.get("valid", {}).get("evals") or 0) for r in group)
    recalibrate_n = sum(1 for r in group if r["phase3_status"] == "recalibrate")
    research_n = sum(1 for r in group if r["phase3_status"] == "research_only")
    retired_n = sum(1 for r in group if r["phase3_status"] == "retire_from_strategy")
    watchlist_n = sum(1 for r in group if r["strategy_gate"] == "watchlist")
    return recalibrate_n * 6 + research_n * 3 + retired_n * 2 + watchlist_n * 4 + min(evals, 80) // 10


def build_phase3(
    *,
    audit_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    base_rows = _read_json(audit_dir / "recalibration_matrix.json")
    if not isinstance(base_rows, list):
        raise SystemExit("recalibration_matrix.json is not a list")

    rows: List[Dict[str, Any]] = []
    by_key: Dict[str, Dict[str, Any]] = {}
    for raw in base_rows:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row["base_status"] = row.get("status")
        row["phase3_status"] = row.get("status")
        row["checkpoint_sources"] = []
        rows.append(row)
        by_key[str(row["pattern_key"])] = row

    for patch in TARGETED_PATCHES:
        valid_metrics = _load_metrics(patch["valid_db"])
        calib_metrics = _load_metrics(patch["calib_db"])
        for pattern_key in patch["patterns"]:
            row = by_key.get(pattern_key)
            if row is None:
                continue
            # Targeted family reruns are authoritative for the patterns they cover.
            # If a rerun yields no surviving rows, that should overwrite the older
            # phase-2 baseline instead of silently preserving stale metrics.
            row["valid"] = dict(valid_metrics.get(pattern_key, {}))
            row["calib"] = dict(calib_metrics.get(pattern_key, {}))
            row["checkpoint_sources"] = sorted(set(row["checkpoint_sources"] + [patch["checkpoint"]]))

    for row in rows:
        override = PHASE3_OVERRIDES.get(str(row["pattern_key"]), {})
        if "phase3_status" in override:
            row["phase3_status"] = override["phase3_status"]

        research_lane, strategy_gate = _default_lane(str(row["phase3_status"]))
        row["research_lane"] = override.get("research_lane", research_lane)
        row["strategy_gate"] = override.get("strategy_gate", strategy_gate)
        row["manual_note"] = override.get("manual_note")
        row["checkpoint_sources"] = sorted(set(row["checkpoint_sources"] + list(override.get("checkpoint_sources", []))))
        row["phase3_drift_flags"] = _drift_flags(row.get("calib") or {}, row.get("valid") or {})
        row["phase3_drift_score"] = len(row["phase3_drift_flags"])

    family_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["canonical_key"])].append(row)

    for canonical_key, group in sorted(grouped.items()):
        status_counts = dict(Counter(str(r["phase3_status"]) for r in group))
        research_lane_counts = dict(Counter(str(r["research_lane"]) for r in group))
        strategy_gate_counts = dict(Counter(str(r["strategy_gate"]) for r in group))
        family_rows.append(
            {
                "canonical_key": canonical_key,
                "patterns": [str(r["pattern_key"]) for r in sorted(group, key=lambda x: int(x["chapter"]))],
                "pattern_count": len(group),
                "valid_evals_total": sum(int(r.get("valid", {}).get("evals") or 0) for r in group),
                "calib_evals_total": sum(int(r.get("calib", {}).get("evals") or 0) for r in group),
                "phase3_status_counts": status_counts,
                "research_lane_counts": research_lane_counts,
                "strategy_gate_counts": strategy_gate_counts,
                "phase3_family_status": _family_phase3_status(group),
                "family_action": _family_action(group),
                "priority_score": _priority_score(group),
                "watchlist_patterns": [str(r["pattern_key"]) for r in group if str(r["strategy_gate"]) == "watchlist"],
                "candidate_patterns": [str(r["pattern_key"]) for r in group if str(r["strategy_gate"]) == "candidate"],
                "checkpoint_sources": sorted({src for r in group for src in r["checkpoint_sources"]}),
            }
        )

    candidate_patterns = [r for r in rows if str(r["strategy_gate"]) == "candidate"]
    watchlist_patterns = [r for r in rows if str(r["strategy_gate"]) == "watchlist"]
    retired_patterns = [r for r in rows if str(r["strategy_gate"]) == "retired"]

    family_lookup = {str(r["canonical_key"]): r for r in family_rows}
    next_batches: List[Dict[str, Any]] = []
    for rank, item in enumerate(NEXT_BATCH_PRIORITY, start=1):
        fam = family_lookup.get(item["canonical_key"])
        if fam is None:
            continue
        next_batches.append(
            {
                "rank": rank,
                "canonical_key": item["canonical_key"],
                "reason": item["reason"],
                "phase3_family_status": fam["phase3_family_status"],
                "family_action": fam["family_action"],
                "priority_score": fam["priority_score"],
                "patterns": fam["patterns"],
            }
        )

    summary = {
        "pattern_count": len(rows),
        "family_count": len(family_rows),
        "phase3_status_counts": dict(Counter(str(r["phase3_status"]) for r in rows)),
        "research_lane_counts": dict(Counter(str(r["research_lane"]) for r in rows)),
        "strategy_gate_counts": dict(Counter(str(r["strategy_gate"]) for r in rows)),
        "candidate_patterns": [str(r["pattern_key"]) for r in candidate_patterns],
        "watchlist_patterns": [str(r["pattern_key"]) for r in watchlist_patterns],
        "retired_patterns": [str(r["pattern_key"]) for r in retired_patterns],
    }

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            {"candidate_after_review": 0, "recalibrate": 1, "research_only": 2, "retire_from_strategy": 3}.get(str(r["phase3_status"]), 9),
            {"candidate": 0, "watchlist": 1, "blocked": 2, "retired": 3}.get(str(r["strategy_gate"]), 9),
            int(r["chapter"]),
            str(r["pattern_key"]),
        ),
    )
    family_rows_sorted = sorted(family_rows, key=lambda r: (-int(r["priority_score"]), str(r["canonical_key"])))

    payload = {
        "summary": summary,
        "pattern_matrix": rows_sorted,
        "family_matrix": family_rows_sorted,
        "strategy_matrix": {
            "candidate": candidate_patterns,
            "watchlist": watchlist_patterns,
            "retired": retired_patterns,
        },
        "next_batches": next_batches,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "phase3_summary.json", summary)
    _write_json(out_dir / "phase3_pattern_matrix.json", rows_sorted)
    _write_json(out_dir / "phase3_family_matrix.json", family_rows_sorted)
    _write_json(out_dir / "phase3_strategy_matrix.json", payload["strategy_matrix"])
    _write_json(out_dir / "phase3_next_batches.json", next_batches)
    _write_text(out_dir / "phase3_governance_report.md", _render_report(payload))
    return payload


def _render_report(payload: Dict[str, Any]) -> str:
    summary = payload["summary"]
    pattern_matrix = payload["pattern_matrix"]
    family_matrix = payload["family_matrix"]
    next_batches = payload["next_batches"]

    lines: List[str] = []
    lines.append("# Phase 3 Governance")
    lines.append("")
    lines.append("## Intent")
    lines.append("")
    lines.append(
        "Phase 3 chuyển trọng tâm từ sửa sâu từng detector sang quyết định hệ thống: "
        "pattern nào thuộc strategy gate, pattern nào chỉ nên ở research layer, "
        "và family nào là batch recalibration kế tiếp."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- patterns: `{summary['pattern_count']}`")
    lines.append(f"- families: `{summary['family_count']}`")
    lines.append(f"- phase3_status_counts: `{summary['phase3_status_counts']}`")
    lines.append(f"- research_lane_counts: `{summary['research_lane_counts']}`")
    lines.append(f"- strategy_gate_counts: `{summary['strategy_gate_counts']}`")
    lines.append(f"- candidate_patterns: `{summary['candidate_patterns']}`")
    lines.append(f"- watchlist_patterns: `{summary['watchlist_patterns']}`")
    lines.append("")

    lines.append("## Strategy Layer")
    lines.append("")
    lines.append("| Pattern | Family | Phase 3 status | Strategy gate | Why |")
    lines.append("|---|---|---|---|---|")
    for row in [r for r in pattern_matrix if str(r["strategy_gate"]) in {"candidate", "watchlist"}]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["pattern_key"]),
                    str(row["canonical_key"]),
                    str(row["phase3_status"]),
                    str(row["strategy_gate"]),
                    str(row.get("manual_note") or row.get("visual_summary") or "").replace("\n", " ").strip(),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Family Matrix")
    lines.append("")
    lines.append("| Family | Family status | Action | Valid evals | Status counts | Strategy gates |")
    lines.append("|---|---|---|---:|---|---|")
    for row in family_matrix[:15]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["canonical_key"]),
                    str(row["phase3_family_status"]),
                    str(row["family_action"]),
                    str(int(row["valid_evals_total"] or 0)),
                    str(row["phase3_status_counts"]),
                    str(row["strategy_gate_counts"]),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Pattern Matrix")
    lines.append("")
    lines.append("| Chap | Pattern | Family | Phase 3 status | Research lane | Strategy gate | Valid evals | Drift |")
    lines.append("|---:|---|---|---|---|---|---:|---|")
    for row in pattern_matrix[:25]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["chapter"]),
                    str(row["pattern_key"]),
                    str(row["canonical_key"]),
                    str(row["phase3_status"]),
                    str(row["research_lane"]),
                    str(row["strategy_gate"]),
                    str(int(row.get("valid", {}).get("evals") or 0)),
                    ", ".join(row["phase3_drift_flags"]) if row["phase3_drift_flags"] else "",
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Next Batches")
    lines.append("")
    lines.append("| Rank | Family | Family status | Action | Why next |")
    lines.append("|---:|---|---|---|---|")
    for row in next_batches:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rank"]),
                    str(row["canonical_key"]),
                    str(row["phase3_family_status"]),
                    str(row["family_action"]),
                    str(row["reason"]),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Key Overrides")
    lines.append("")
    lines.append("- `double_bottoms_adam_adam` và `double_tops_adam_adam` vẫn là hai branch duy nhất ở `watchlist`; phase 3 chưa promote thêm pattern nào vào strategy gate.")
    lines.append("- `broadening_wedges`, `horns`, `pipe`, `rounding`, `triple`, và `flags` đã đi qua full family rerun trong phase 3, và hiện được giữ ở `research_only` thay vì backlog cũ.")
    lines.append("- `flags_high_tight` không còn survivor sau family scanner mới cho `flags`, nên chapter này được giữ ở `reference_only` cho tới khi có detector riêng thuyết phục.")
    lines.append("- `head_and_shoulders_bottoms` hiện dùng metric từ `hs_bottoms_pass3`: pass neckline-specific chỉ cứu lại đúng một calibration-side standard survivor và giữ valid ở complex-only, nên branch này vẫn quá mỏng cho strategy use.")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", default="scan_results/spec_audit_20260306")
    parser.add_argument("--out-dir", default=str(DEFAULT_PATTERN_OUTPUT))
    args = parser.parse_args()

    payload = build_phase3(
        audit_dir=Path(args.audit_dir).resolve(),
        out_dir=Path(args.out_dir).resolve(),
    )
    print("=== Phase 3 Governance ===")
    print(f"out_dir: {Path(args.out_dir).resolve()}")
    print(f"phase3_status_counts: {payload['summary']['phase3_status_counts']}")
    print(f"strategy_gate_counts: {payload['summary']['strategy_gate_counts']}")


if __name__ == "__main__":
    main()
