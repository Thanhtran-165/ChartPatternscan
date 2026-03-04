"""
Validate Vietnamese book output produced by `scanner/build_book_vi.py`.

This is a lightweight quality gate to catch:
  - missing chapters / missing sections
  - malformed AI narrative (wrong headings, digits/percent signs, tables)

Usage:
  python3 scanner/validate_book_vi.py --book-dir scan_results/book_vi
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Optional


RE_HAS_DIGIT = re.compile(r"\d")
RE_HAS_PERCENT = re.compile(r"%")
RE_HAS_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$")


@dataclass(frozen=True)
class ChapterIssue:
    chapter_path: str
    issue: str


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _find_required(text: str, required: List[str]) -> List[str]:
    missing: List[str] = []
    for r in required:
        if r not in text:
            missing.append(r)
    return missing


def _extract_narrative(text: str) -> str:
    """
    Return text after the "## Lời bình / Narrative" section heading (best-effort).
    """
    marker = "## Lời bình / Narrative"
    if marker not in text:
        return ""
    tail = text.split(marker, 1)[1]
    return tail.strip()


def validate_book_dir(book_dir: str) -> tuple[bool, List[ChapterIssue]]:
    book_dir = os.path.abspath(book_dir)
    issues: List[ChapterIssue] = []

    # Basic structure
    required_paths = [
        os.path.join(book_dir, "book.md"),
        os.path.join(book_dir, "book_meta.json"),
        os.path.join(book_dir, "chapters"),
        os.path.join(book_dir, "reports"),
        os.path.join(book_dir, "context"),
    ]
    for p in required_paths:
        if not os.path.exists(p):
            issues.append(ChapterIssue(chapter_path=book_dir, issue=f"missing: {p}"))

    chapters_dir = os.path.join(book_dir, "chapters")
    if not os.path.isdir(chapters_dir):
        return (False, issues)

    chapter_files = sorted([os.path.join(chapters_dir, f) for f in os.listdir(chapters_dir) if f.endswith(".md")])
    if not chapter_files:
        issues.append(ChapterIssue(chapter_path=chapters_dir, issue="no chapter .md files found"))
        return (False, issues)

    for ch in chapter_files:
        try:
            text = _read_text(ch)
        except Exception as e:
            issues.append(ChapterIssue(chapter_path=ch, issue=f"read error: {e}"))
            continue

        required_sections = [
            "## Định nghĩa scan (từ digitized spec)",
            "## Kết quả thống kê",
            "## Lời bình / Narrative",
        ]
        miss = _find_required(text, required_sections)
        for m in miss:
            issues.append(ChapterIssue(chapter_path=ch, issue=f"missing section: {m}"))

        # Narrative rules
        narrative = _extract_narrative(text)
        if not narrative:
            issues.append(ChapterIssue(chapter_path=ch, issue="missing narrative content"))
            continue

        if "_(Chưa sinh narrative)_" in narrative:
            # Placeholder: allowed, but report as warning-level issue.
            issues.append(ChapterIssue(chapter_path=ch, issue="narrative placeholder (AI not generated)"))
            continue

        # If AI narrative exists, enforce structure and anti-hallucination guards.
        required_headings = [
            "### Nhận xét nhanh",
            "### Cách nhận diện trên dữ liệu OHLCV (theo logic scanner)",
            "### Lưu ý sai lệch/false positives cần calibration",
            "### Ý tưởng nghiên cứu tiếp theo trên dữ liệu Việt Nam",
        ]
        miss_h = _find_required(narrative, required_headings)
        for m in miss_h:
            issues.append(ChapterIssue(chapter_path=ch, issue=f"narrative missing heading: {m}"))

        if RE_HAS_DIGIT.search(narrative):
            issues.append(ChapterIssue(chapter_path=ch, issue="narrative contains digits (0-9)"))
        if RE_HAS_PERCENT.search(narrative):
            issues.append(ChapterIssue(chapter_path=ch, issue="narrative contains % sign"))

        # Disallow new tables in narrative (stats must come from deterministic tables).
        for line in narrative.splitlines():
            if RE_HAS_TABLE_ROW.match(line):
                issues.append(ChapterIssue(chapter_path=ch, issue="narrative contains a markdown table row"))
                break

    ok = len([i for i in issues if not i.issue.startswith("narrative placeholder")]) == 0
    return (ok, issues)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book-dir", required=True, help="Output directory produced by build_book_vi.py")
    args = parser.parse_args()

    ok, issues = validate_book_dir(str(args.book_dir))
    warnings = [i for i in issues if i.issue.startswith("narrative placeholder")]
    errors = [i for i in issues if not i.issue.startswith("narrative placeholder")]
    print("=== Book Validation (VI) ===")
    print(f"book_dir: {os.path.abspath(str(args.book_dir))}")
    print(f"ok: {ok}")
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")
    shown: List[ChapterIssue] = errors + warnings[: min(25, len(warnings))]
    for i in shown[:200]:
        print(f"- {i.chapter_path}: {i.issue}")
    remaining = len(issues) - len(shown[:200])
    if remaining > 0:
        print(f"... ({remaining} more)")

    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
