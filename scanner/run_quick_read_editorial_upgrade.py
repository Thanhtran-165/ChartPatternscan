"""Add the required quick_read section through the canonical AI editorial layer.

This command does not rescan patterns, change statistics, or render PDFs. It
only upgrades approved editorial artifacts so the public renderer can consume a
human-readable `quick_read` section instead of generating template prose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.canonical_chapter_content import prepare_canonical_chapter_content  # noqa: E402
from scanner.canonical_deepseek_editorial_adapter import (  # noqa: E402
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    _repair_missing_editorial_sections_with_ai,
    build_deepseek_dossier,
    load_dotenv,
)
from scanner.canonical_editorial_layer import validate_canonical_editorial_sections  # noqa: E402
from scanner.rebuild_source_guided_final_chapters import _read_json, _write_json  # noqa: E402
from scanner.validate_final_chapters_manifest import DEFAULT_MANIFEST  # noqa: E402


DEFAULT_OUT_ROOT = Path("artifacts/scanner_v2/quick_read_editorial_upgrade_v1")


def _select_entries(manifest: Mapping[str, Any], patterns: list[str]) -> list[Mapping[str, Any]]:
    chapters = [chapter for chapter in manifest.get("chapters", []) if isinstance(chapter, Mapping)]
    if not patterns:
        return chapters
    wanted = set(patterns)
    return [chapter for chapter in chapters if chapter.get("pattern_id") in wanted]


def _resolve(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    if not path.is_absolute():
        path = ROOT / path
    return path if path.exists() and path.is_file() else None


def _approved_source_path(entry: Mapping[str, Any], payload: Mapping[str, Any]) -> Path:
    candidates = [
        payload.get("editorial_source_path"),
        ((entry.get("chapter_writing_stages") or {}) if isinstance(entry.get("chapter_writing_stages"), Mapping) else {}).get("refined_ai_sections"),
        ((entry.get("chapter_writing_stages") or {}) if isinstance(entry.get("chapter_writing_stages"), Mapping) else {}).get("source_guided_ai_sections"),
    ]
    for candidate in candidates:
        path = _resolve(str(candidate)) if candidate else None
        if path:
            return path
    raise FileNotFoundError(f"No approved editorial artifact for {entry.get('pattern_id')}")


def _load_approved(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Approved artifact is not a JSON object: {path}")
    return data


def _public_text_context(entry: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in ("manuscript", "notes"):
        path = _resolve(str(entry.get(key))) if entry.get(key) else None
        if path:
            paths.append(path)
    return paths


def _relative_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def upgrade_one(
    entry: Mapping[str, Any],
    *,
    out_root: Path,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    update_payload: bool,
) -> dict[str, Any]:
    pattern_id = str(entry.get("pattern_id"))
    family = str(entry.get("family") or "uncategorized")
    payload_path = Path(str(entry.get("payload")))
    source_notes_path = Path(str(entry.get("source_notes")))
    payload = dict(_read_json(payload_path))
    source_notes = dict(_read_json(source_notes_path))
    approved_path = _approved_source_path(entry, payload)
    approved = _load_approved(approved_path)
    out_dir = out_root / family / pattern_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dossier = build_deepseek_dossier(
        payload=payload,
        source_notes=source_notes,
        chapter_meta=entry,
        extra_context_paths=_public_text_context(entry),
        style_profile={
            "quick_read_goal": (
                "Viết phần mở đầu 'Đọc nhanh chương này' như lời dẫn sách: tự nhiên, giàu diễn giải, "
                "không giống bảng kết quả và không khuyến nghị giao dịch."
            ),
            "forbidden_quick_read_style": [
                "không dùng cấu trúc Mục/Kết quả chính",
                "không viết câu placeholder hoặc template",
                "không nhắc scanner, pipeline, payload, factory",
            ],
        },
    )
    (out_dir / "quick_read_dossier.json").write_text(
        json.dumps(dossier, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    upgraded = _repair_missing_editorial_sections_with_ai(
        approved=approved,
        dossier=dossier,
        out_dir=out_dir,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )
    upgraded_path = out_dir / "approved_ai_sections.json"
    upgraded_path.write_text(json.dumps(upgraded, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    prepared = prepare_canonical_chapter_content(payload, approved_sections_path=upgraded_path)
    gate = validate_canonical_editorial_sections(prepared)
    if gate["status"] != "PASS":
        raise RuntimeError(f"quick_read gate failed for {pattern_id}: {gate['failures']}")
    if update_payload:
        payload["editorial_source_path"] = str(upgraded_path.resolve())
        payload.setdefault("chapter_writing_stages", {})
        if isinstance(payload["chapter_writing_stages"], dict):
            payload["chapter_writing_stages"]["quick_read_ai_sections"] = _relative_or_abs(upgraded_path)
        _write_json(payload_path, payload)
    return {
        "pattern_id": pattern_id,
        "family": family,
        "status": "PASS",
        "previous_approved": str(approved_path),
        "upgraded_approved": str(upgraded_path),
        "payload_updated": update_payload,
        "quick_read_paragraphs": len((upgraded.get("editorial_sections") or {}).get("quick_read") or []),
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate missing quick_read sections through DeepSeek.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_DEEPSEEK_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--update-payload", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")
    manifest = _read_json(Path(args.manifest))
    entries = _select_entries(manifest, list(args.pattern))
    if not entries:
        raise SystemExit("No chapters selected.")
    out_root = Path(args.out_root)
    report = {"status": "PASS", "chapters": []}
    for index, entry in enumerate(entries, start=1):
        print(f"[{index}/{len(entries)}] quick_read {entry.get('pattern_id')}", flush=True)
        report["chapters"].append(
            upgrade_one(
                entry,
                out_root=out_root,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout_s,
                update_payload=args.update_payload,
            )
        )
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "quick_read_upgrade_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "report": str(report_path), "count": len(report["chapters"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
