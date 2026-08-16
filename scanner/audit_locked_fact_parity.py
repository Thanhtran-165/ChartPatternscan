"""Fail-closed parity gate for locked chapter facts versus rendered PDFs.

This gate deliberately checks only headline facts that are expected to be
shared by the payload, reader prose, and the chapter card. Dates, prices,
quantiles, and example-specific figures are allowed to differ because they
describe different rows of the chapter.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts/final_chapters/final_chapters_manifest.json"
INTERNAL_TOKENS = (
    "conservative_half_",
    "source_full_",
    "textbook_success",
    "middle_case",
    "layer missing",
    "zero and stale",
    "failure 5pct",
    "Đã kiểm tra 0 biểu đồ",
)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, Mapping) else {}


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 2) -> str | None:
    number = _num(value)
    if number is None:
        return None
    return f"{number:.{digits}f}".replace(".", ",")


def _count(value: Any) -> str | None:
    number = _num(value)
    if number is None:
        return None
    return f"{int(round(number)):,}".replace(",", ".")


def _pdf_text(path: Path) -> str:
    return subprocess.check_output(["pdftotext", str(path), "-"], text=True, errors="replace")


def _facts(payload: Mapping[str, Any]) -> dict[str, str]:
    ref = payload.get("chapter_reference") if isinstance(payload.get("chapter_reference"), Mapping) else {}
    target = payload.get("target_calibration") if isinstance(payload.get("target_calibration"), Mapping) else {}
    base = target.get("base_target") if isinstance(target.get("base_target"), Mapping) else {}
    facts: dict[str, str] = {}
    for label, value in (
        ("n", _count(ref.get("public_grade_events") or ref.get("events") or ref.get("evaluated_events"))),
        ("target_hit", _fmt(base.get("target_hit_rate"))),
        ("failure_5pct", _fmt(base.get("failure_5pct_rate") or ref.get("failure_5pct_rate"))),
        ("mfe", _fmt(ref.get("median_mfe_pct"))),
        ("mae", _fmt(ref.get("median_mae_pct"))),
    ):
        if value:
            facts[label] = value
    return facts


def audit(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    failures: list[dict[str, Any]] = []
    checked = 0
    chapters = [row for row in manifest.get("chapters", []) if isinstance(row, Mapping)]
    for entry in chapters:
        pattern_id = str(entry.get("pattern_id") or "")
        payload_path = Path(str(entry.get("payload") or ""))
        pdf_path = Path(str(entry.get("pdf") or entry.get("source_pdf") or ""))
        if not payload_path.is_absolute():
            payload_path = ROOT / payload_path
        if not pdf_path.is_absolute():
            pdf_path = ROOT / pdf_path
        if not payload_path.exists() or not pdf_path.exists():
            failures.append({"pattern_id": pattern_id, "check": "artifact_exists"})
            continue
        payload = _read_json(payload_path)
        report = payload.get("canonical_content_generation_report")
        rewritten = report.get("locked_fact_sections_rewritten") if isinstance(report, Mapping) else None
        if set(rewritten or []) != {"failure", "post_breakout", "quick_read", "statistics", "summary"}:
            failures.append({"pattern_id": pattern_id, "check": "locked_fact_rewrite", "value": rewritten})
        text = _pdf_text(pdf_path)
        lower = text.lower()
        for token in INTERNAL_TOKENS:
            if token.lower() in lower:
                failures.append({"pattern_id": pattern_id, "check": "internal_token", "token": token})
        facts = _facts(payload)
        for label, value in facts.items():
            if value not in text and value.replace(",", ".") not in text:
                failures.append({"pattern_id": pattern_id, "check": "headline_fact_missing", "fact": label, "value": value})
        checked += 1
    return {"status": "PASS" if not failures and checked == len(chapters) else "FAIL", "checked": checked, "chapters": len(chapters), "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()
    result = audit(Path(args.manifest))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
