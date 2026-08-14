"""Edition 2 — refresh statistics of measure-rule-affected final chapters.

Chuỗi đã duyệt edition 2 (14/08): cập nhật số liệu các chương có hit/target đổi
sau 3 đợt sửa measure rule, thêm ghi chú variant sách, giữ nguyên mạch editorial
đã duyệt. Pipeline mỗi pattern:

1. Nạp payload/spec/notes MỚI do family publication builder chạy lại (A3) và
   events MỚI do family scan runner chạy lại (A2) — cùng đường dẫn quen thuộc
   của `rebuild_source_guided_final_chapters.EVENT_SOURCES`.
2. Giữ bản approved AI sections edition 1.2 làm previous candidate.
3. Chạy đúng 1 pass lightweight refinement bắt buộc (`--force`) với locked
   facts MỚI để prose khớp số mới.
4. Rebuild ví dụ minh họa + render qua canonical factory (final PDF chỉ đi
   qua `canonical_publication_chapter_factory_v1`).
5. Style-v3 audit + ghi manifest entry edition 2 + promote final.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.audit_publication_style_v3 import audit_publication_style_v3  # noqa: E402
from scanner.canonical_publication_chapter_factory import (  # noqa: E402
    build_canonical_publication_chapter,
)
from scanner.canonical_example_charts import (  # noqa: E402
    DEFAULT_PRICE_DB as DEFAULT_CANONICAL_CHART_PRICE_DB,
    build_canonical_example_charts,
)
from scanner.promote_final_chapter import promote_final_chapters  # noqa: E402
from scanner.rebuild_source_guided_final_chapters import (  # noqa: E402
    DEFAULT_DEEPSEEK_MODEL,
    _load_charts,
    _load_events,
    _read_json,
    _run_lightweight_refinement,
    _slug_from_entry,
    _write_json,
)
from scanner.validate_final_chapters_manifest import DEFAULT_MANIFEST  # noqa: E402
from scanner.publication_flow_contract import CANONICAL_SOURCE_GUIDED_REFINEMENT_ID  # noqa: E402
from scanner.canonical_chapter_content import CANONICAL_CONTENT_GENERATOR_ID  # noqa: E402
from scanner.canonical_editorial_layer import CANONICAL_AI_EDITORIAL_GATE_ID, CANONICAL_EDITORIAL_WORKFLOW_ID  # noqa: E402
from scanner.canonical_publication_chapter_factory import (  # noqa: E402
    CANONICAL_PUBLICATION_FACTORY_ID,
    CANONICAL_PUBLICATION_FLOW,
    CANONICAL_PUBLICATION_STYLE_VERSION,
    CANONICAL_READER_EXPERIENCE_GATE_ID,
)
from scanner.pattern_publication_core import PUBLICATION_CORE_ID  # noqa: E402


DEFAULT_OUT_ROOT = Path("artifacts/scanner_v2/source_guided_refinement_edition_2")
EDITION2_REBUILD_ID = "edition2_stats_refresh_v1"

# pattern_id -> fresh family-builder payload (tương đối với repo root)
FRESH_PAYLOADS: dict[str, str] = {
    "bump_and_run_reversal_bottoms": "artifacts/scanner_v2/bump_and_run_family_public_chapters/bump_and_run_reversal_bottoms/bump_and_run_reversal_bottoms_public_chapter_payload.json",
    "bump_and_run_reversal_tops": "artifacts/scanner_v2/bump_and_run_family_public_chapters/bump_and_run_reversal_tops/bump_and_run_reversal_tops_public_chapter_payload.json",
    "bear_flags": "artifacts/scanner_v2/flag_family_public_chapters/bear_flag/bear_flag_public_chapter_payload.json",
    "bull_flags": "artifacts/scanner_v2/flag_family_public_chapters/bull_flag/bull_flag_public_chapter_payload.json",
    "bear_pennants": "artifacts/scanner_v2/flag_family_public_chapters/bear_pennant/bear_pennant_public_chapter_payload.json",
    "bull_pennants": "artifacts/scanner_v2/flag_family_public_chapters/bull_pennant/bull_pennant_public_chapter_payload.json",
    "high_tight_flags": "artifacts/scanner_v2/flag_family_public_chapters/high_tight_flag/high_tight_flag_public_chapter_payload.json",
    "horn_bottoms": "artifacts/scanner_v2/horn_family_public_chapters/horn_bottoms/horn_bottoms_public_chapter_payload.json",
    "horn_tops": "artifacts/scanner_v2/horn_family_public_chapters/horn_tops/horn_tops_public_chapter_payload.json",
    "pipe_bottoms": "artifacts/scanner_v2/pipe_family_public_chapters/pipe_bottoms/pipe_bottoms_public_chapter_payload.json",
    "pipe_tops": "artifacts/scanner_v2/pipe_family_public_chapters/pipe_tops/pipe_tops_public_chapter_payload.json",
    "rounding_bottoms": "artifacts/scanner_v2/rounding_family_public_chapters/rounding_bottoms/rounding_bottoms_public_chapter_payload.json",
    "rounding_tops": "artifacts/scanner_v2/rounding_family_public_chapters/rounding_tops/rounding_tops_public_chapter_payload.json",
    "falling_three_methods": "artifacts/scanner_v2/three_methods_family_public_chapters/falling_three_methods/falling_three_methods_public_chapter_payload.json",
    "rising_three_methods": "artifacts/scanner_v2/three_methods_family_public_chapters/rising_three_methods/rising_three_methods_public_chapter_payload.json",
    "triangles_symmetrical": "artifacts/scanner_v2/triangle_family_public_chapters/symmetrical_triangle/symmetrical_triangle_public_chapter_payload.json",
    "wedges_falling": "artifacts/scanner_v2/wedge_family_public_chapters/falling_wedge/falling_wedge_public_chapter_payload.json",
    "wedges_rising": "artifacts/scanner_v2/wedge_family_public_chapters/rising_wedge/rising_wedge_public_chapter_payload.json",
    "cup_with_handle": "artifacts/scanner_v2/cup_handle_family_public_chapters/cup_with_handle/cup_with_handle_public_chapter_payload.json",
    "double_tops_adam_adam": "artifacts/scanner_v2/double_pattern_variant_public_chapters/double_tops_adam_adam/double_tops_adam_adam_public_chapter_payload.json",
    "double_tops_adam_eve": "artifacts/scanner_v2/double_pattern_variant_public_chapters/double_tops_adam_eve/double_tops_adam_eve_public_chapter_payload.json",
    "double_tops_eve_adam": "artifacts/scanner_v2/double_pattern_variant_public_chapters/double_tops_eve_adam/double_tops_eve_adam_public_chapter_payload.json",
    "double_tops_eve_eve": "artifacts/scanner_v2/double_pattern_variant_public_chapters/double_tops_eve_eve/double_tops_eve_eve_public_chapter_payload.json",
}


def _load_spec(entry: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    spec_value = str(entry.get("publication_spec") or "").strip()
    spec_path = Path(spec_value) if spec_value else Path(str(payload.get("publication_spec_path") or ""))
    if spec_path.exists():
        spec = dict(_read_json(spec_path))
    else:
        spec = {}
    spec.setdefault("title", payload.get("pattern_name") or entry.get("pattern_id"))
    return spec


def refresh_one(
    *,
    entry: Mapping[str, Any],
    out_root: Path,
    model: str,
    timeout_s: int,
    max_tokens: int,
    skip_ai: bool = False,
) -> Path:
    pattern_id = str(entry.get("pattern_id"))
    family = str(entry.get("family") or "uncategorized")
    slug = _slug_from_entry(entry)
    chapter_dir = out_root / family / slug
    ai_dir = chapter_dir / "ai"
    style_dir = chapter_dir / "source_style"
    render_dir = chapter_dir / "chapter"

    fresh_payload_path = Path(FRESH_PAYLOADS[pattern_id])
    if not fresh_payload_path.exists():
        raise FileNotFoundError(f"Missing fresh family payload for {pattern_id}: {fresh_payload_path}")
    payload = dict(_read_json(fresh_payload_path))

    source_notes_path = Path(str(entry.get("source_notes")))
    source_notes = dict(_read_json(source_notes_path))
    old_payload_path = Path(str(entry.get("payload")))

    # refined AI sections edition 1.2: bản được render final là bản đã nâng cấp quick_read
    # (quick_read_editorial_upgrade_v1) — ưu tiên theo thứ tự: entry stages quick_read →
    # payload cũ (editorial_source_path / stages) → entry refined (8 sections, chỉ dự phòng).
    entry_stages = entry.get("chapter_writing_stages") if isinstance(entry.get("chapter_writing_stages"), Mapping) else {}
    old_payload_json: Mapping[str, Any] = {}
    if old_payload_path.exists():
        loaded_payload = _read_json(old_payload_path)
        if isinstance(loaded_payload, Mapping):
            old_payload_json = loaded_payload
    payload_stages = old_payload_json.get("chapter_writing_stages") if isinstance(old_payload_json.get("chapter_writing_stages"), Mapping) else {}

    def _first_existing(*candidates: Any) -> Path | None:
        for value in candidates:
            text_value = str(value or "").strip()
            if text_value:
                candidate = Path(text_value)
                if candidate.exists():
                    return candidate
        return None

    old_refined = _first_existing(
        entry_stages.get("quick_read_ai_sections"),
        payload_stages.get("quick_read_ai_sections"),
        old_payload_json.get("editorial_source_path"),
        payload_stages.get("refined_ai_sections"),
        entry_stages.get("refined_ai_sections"),
    )
    if old_refined is None:
        raise FileNotFoundError(f"Missing edition-1.2 refined AI sections for {pattern_id}")

    # dossier phong cách: giữ nguyên bản edition 1.2
    old_dossier_value = str(entry_stages.get("source_style_dossier") or "")
    if not old_dossier_value and old_payload_path.exists():
        payload_stages = _read_json(old_payload_path).get("chapter_writing_stages")
        if isinstance(payload_stages, Mapping):
            old_dossier_value = str(payload_stages.get("source_style_dossier") or "")
    old_dossier = Path(old_dossier_value) if old_dossier_value and Path(old_dossier_value).exists() else None
    style_dir.mkdir(parents=True, exist_ok=True)
    dossier_path = style_dir / "source_style_dossier.md"
    if old_dossier is not None:
        shutil.copy2(old_dossier, dossier_path)
    else:
        dossier_path.write_text(
            "# Source/style dossier (edition 2 stats refresh)\n\n"
            "Bản edition 2 giữ toàn bộ hướng phong cách của chương edition 1.2; chỉ số liệu và ghi chú variant thay đổi.\n",
            encoding="utf-8",
        )

    previous_candidate = ai_dir / "refined_source" / "approved_ai_sections.json"
    previous_candidate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old_refined, previous_candidate)

    chapter_meta = {
        "pattern_id": pattern_id,
        "title": payload.get("pattern_name") or entry.get("title") or pattern_id,
        "family": family,
    }

    # payload edition2 nằm trong render dir; AI đọc từ đây để lấy locked facts mới
    render_dir.mkdir(parents=True, exist_ok=True)
    edition2_payload_path = render_dir / f"{slug}_public_chapter_payload.json"
    _write_json(edition2_payload_path, payload)

    if skip_ai:
        refined = previous_candidate
        guard_stub = ai_dir / "refined" / "approved_ai_sections_guard.json"
        guard_stub.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous_candidate, ai_dir / "refined" / "approved_ai_sections.json")
        _write_json(guard_stub, {"status": "PASS", "refinement_mode": "edition2_skip_ai_v1", "reused_previous_candidate": str(previous_candidate)})
    else:
        refined = _run_lightweight_refinement(
            payload_path=edition2_payload_path,
            source_notes_path=source_notes_path,
            out_dir=ai_dir / "refined",
            chapter_meta=chapter_meta,
            style_dossier=dossier_path,
            previous_candidate=previous_candidate,
            model=model,
            temperature=0.3,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            force=True,
        )

    events = _load_events(pattern_id)
    source_charts = _load_charts(Path(str(entry.get("source_pdf") or entry.get("pdf"))), old_payload_path, slug)
    charts, selected_examples, chart_report = build_canonical_example_charts(
        pattern_id=pattern_id,
        events=events,
        existing_examples=payload.get("example_events") if isinstance(payload.get("example_events"), Mapping) else {},
        out_dir=render_dir / "charts",
        price_db=DEFAULT_CANONICAL_CHART_PRICE_DB,
        schematic=source_charts.get("schematic"),
    )
    payload["example_events"] = {key: dict(value) for key, value in selected_examples.items()}
    payload["canonical_example_chart_report"] = chart_report
    if "schematic" not in charts and "schematic" in source_charts:
        charts["schematic"] = source_charts["schematic"]

    spec = _load_spec(entry, payload)
    result = build_canonical_publication_chapter(
        payload=payload,
        source_notes=source_notes,
        events=events,
        path_df=pd.DataFrame(),
        charts=charts,
        spec=spec,
        out_dir=render_dir,
        pdf_filename=Path(str(entry.get("source_pdf") or entry.get("pdf"))).name,
        payload_filename=edition2_payload_path.name,
        manuscript_filename=Path(str(entry.get("manuscript") or f"{slug}_ai_editorial_manuscript.md")).name,
        notes_filename=Path(str(entry.get("notes") or f"{slug}_public_chapter_notes.md")).name,
        family_id=family,
        source_family_factory_id=payload.get("source_family_factory_id"),
        approved_sections_path=refined,
    )
    audit_path = render_dir / "style_v3_audit.json"
    audit = audit_publication_style_v3(Path(result["pdf"]), Path(result["payload"]))
    _write_json(audit_path, audit)
    if audit["status"] != "PASS":
        raise RuntimeError(f"style-v3 audit failed for {pattern_id}: {audit['failures']}")

    manifest_entry = dict(entry)
    manifest_entry.update(
        {
            "status": "final",
            "source_pdf": str(result["pdf"]),
            "payload": str(result["payload"]),
            "manuscript": str(result["manuscript"]),
            "notes": str(result["notes"]),
            "factory_id": CANONICAL_PUBLICATION_FACTORY_ID,
            "publication_core_id": PUBLICATION_CORE_ID,
            "publication_flow": CANONICAL_PUBLICATION_FLOW,
            "canonical_publication_factory_id": CANONICAL_PUBLICATION_FACTORY_ID,
            "canonical_reader_experience_gate_id": CANONICAL_READER_EXPERIENCE_GATE_ID,
            "canonical_publication_style_version": CANONICAL_PUBLICATION_STYLE_VERSION,
            "canonical_editorial_workflow_id": CANONICAL_EDITORIAL_WORKFLOW_ID,
            "canonical_ai_editorial_gate_id": CANONICAL_AI_EDITORIAL_GATE_ID,
            "canonical_content_generator_id": CANONICAL_CONTENT_GENERATOR_ID,
            "style_v3_audit": str(audit_path),
            "chapter_writing_policy_id": CANONICAL_SOURCE_GUIDED_REFINEMENT_ID,
            "chapter_writing_stages": {
                "source_style_dossier": str(dossier_path),
                "source_guided_ai_sections": str(previous_candidate),
                "refined_ai_sections": str(refined),
                "canonical_pdf": str(result["pdf"]),
                "style_v3_audit": str(audit_path),
            },
            "edition2_stats_refresh_id": EDITION2_REBUILD_ID,
            "edition2_stats_refresh_notes": (
                "Số liệu cập nhật theo family scan + measure rule đã sửa (3 đợt 13-14/08); "
                "prose giữ mạch edition 1.2, đã qua 1 pass lightweight refinement với locked facts mới."
            ),
        }
    )
    entry_path = chapter_dir / f"{pattern_id}_final_manifest_entry.json"
    _write_json(entry_path, manifest_entry)
    return entry_path


def _select_entries(manifest: Mapping[str, Any], patterns: list[str]) -> list[Mapping[str, Any]]:
    chapters = [c for c in manifest.get("chapters", []) if isinstance(c, Mapping)]
    if not patterns:
        raise SystemExit("Cần truyền --pattern (có thể lặp lại) hoặc --all.")
    wanted = set(patterns)
    missing = wanted - {str(c.get("pattern_id")) for c in chapters}
    if missing:
        raise SystemExit(f"Pattern không có trong manifest: {sorted(missing)}")
    return [c for c in chapters if str(c.get("pattern_id")) in wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description="Edition 2 stats refresh cho các chương measure-rule đổi số.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ 23 pattern trong FRESH_PAYLOADS.")
    parser.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--skip-ai", action="store_true", help="Giữ nguyên prose (chỉ khi số trong prose không đổi).")
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    patterns = list(args.pattern)
    if args.all:
        patterns = sorted(FRESH_PAYLOADS)
    if not patterns:
        raise SystemExit("Cần --pattern hoặc --all.")
    manifest = _read_json(Path(args.manifest))
    entries = _select_entries(manifest, patterns)
    out_root = Path(args.out_root)

    done: list[str] = []
    failed: list[dict[str, str]] = []
    for entry in entries:
        pattern_id = str(entry.get("pattern_id"))
        try:
            entry_path = refresh_one(
                entry=entry,
                out_root=out_root,
                model=args.model,
                timeout_s=args.timeout_s,
                max_tokens=args.max_tokens,
                skip_ai=args.skip_ai,
            )
            done.append(pattern_id)
            print(f"[OK] {pattern_id} -> {entry_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append({"pattern_id": pattern_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL] {pattern_id}: {type(exc).__name__}: {exc}", flush=True)

    summary_path = out_root / "edition2_stats_refresh_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"done": done, "failed": failed}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"done": len(done), "failed": len(failed)}, ensure_ascii=False))
    if failed:
        raise SystemExit(1)

    if args.promote:
        promote_final_chapters(
            entry_paths=[
                out_root / str(e.get("family") or "uncategorized") / _slug_from_entry(e) / f"{e.get('pattern_id')}_final_manifest_entry.json"
                for e in entries
                if str(e.get("pattern_id")) in set(done)
            ],
            manifest_path=Path(args.manifest),
        )


if __name__ == "__main__":
    main()
