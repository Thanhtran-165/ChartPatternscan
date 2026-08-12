"""Split and audit Edition 1 book prose quality.

This is a read-only QA helper. It does not change scanner output, statistics,
chart examples, or chapter PDFs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader


BOOK_DIR = Path("artifacts/book_level/edition_1")
BOOK_PDF = BOOK_DIR / "bulkowski_vietnam_edition_1.pdf"
BOOK_MANIFEST = BOOK_DIR / "bulkowski_vietnam_edition_1_manifest.json"
TEXT_QA_DIR = BOOK_DIR / "text_qa"
SPLIT_DIR = TEXT_QA_DIR / "split_current"
REPORT_PATH = BOOK_DIR / "edition1_prose_quality_audit.json"


TECHNICAL_TERMS = (
    "walk-forward",
    "fold",
    "preflight",
    "tradable layer",
    "target-first",
    "pipeline",
    "fallback",
    "scanner",
    "payload",
    "factory",
    "branch_id",
    "publication_quality_tier",
)

TRADING_DIRECTIVE_TERMS = (
    "khuyến nghị mua",
    "khuyến nghị bán",
    "khuyến nghị mua/bán",
    "vào lệnh",
    "cắt lỗ",
    "dừng lỗ",
    "short setup",
    "BUY signal",
)

AI_FLAT_PHRASES = (
    "Dữ liệu cho thấy",
    "Điều này cho thấy",
    "Có thể thấy rằng",
    "Nhìn chung",
    "Tóm lại",
)

INTERNAL_REPORT_PHRASES = (
    "báo cáo nội bộ",
    "artifact",
    "manifest",
    "QA",
    "gate",
    "audit",
    "test suite",
)


@dataclass(frozen=True)
class SectionText:
    key: str
    title: str
    family: str
    start_page: int
    end_page: int
    text: str


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _page_texts(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _paragraphs(text: str) -> list[str]:
    paras: list[str] = []
    chunks = re.split(r"\n\s*\n", text)
    for chunk in chunks:
        cleaned = _norm(chunk)
        if cleaned:
            paras.append(cleaned)
    return paras


def _sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[.!?。]+|\n", text) if part.strip()])


def _count_terms(text: str, terms: tuple[str, ...]) -> dict[str, int]:
    lower = text.lower()
    out: dict[str, int] = {}
    for term in terms:
        count = lower.count(term.lower())
        if count:
            out[term] = count
    return out


def _long_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n", text) if part.strip()]
    out: list[str] = []
    for sent in sentences:
        words = sent.split()
        if len(words) >= 65:
            out.append(sent[:260])
    return out[:10]


def _repeated_paragraphs(paragraphs: list[str]) -> list[dict[str, Any]]:
    counts = Counter(paragraphs)
    return [
        {"count": count, "text": para[:260]}
        for para, count in counts.most_common()
        if count >= 2 and len(para) >= 80
    ][:10]


def _edit_level(score: int, term_hits: int, repeats: int, long_sentences: int) -> str:
    if score == 0:
        return "pass"
    if term_hits >= 1 or score >= 14:
        return "heavy_edit"
    if repeats >= 1 or long_sentences >= 3 or score >= 7:
        return "medium_edit"
    return "light_edit"


def _audit_section(section: SectionText) -> dict[str, Any]:
    paras = _paragraphs(section.text)
    tech = _count_terms(section.text, TECHNICAL_TERMS)
    directives = _count_terms(section.text, TRADING_DIRECTIVE_TERMS)
    ai_flat = _count_terms(section.text, AI_FLAT_PHRASES)
    internal = _count_terms(section.text, INTERNAL_REPORT_PHRASES)
    long_sents = _long_sentences(section.text)
    repeated = _repeated_paragraphs(paras)
    score = (
        8 * sum(tech.values())
        + 10 * sum(directives.values())
        + 2 * sum(ai_flat.values())
        + 4 * sum(internal.values())
        + 2 * len(long_sents)
        + 3 * len(repeated)
    )
    term_hits = sum(tech.values()) + sum(directives.values())
    return {
        "key": section.key,
        "title": section.title,
        "family": section.family,
        "start_page": section.start_page,
        "end_page": section.end_page,
        "paragraph_count": len(paras),
        "sentence_count": _sentence_count(section.text),
        "char_count": len(section.text),
        "technical_terms": tech,
        "trading_directive_terms": directives,
        "flat_ai_phrases": ai_flat,
        "internal_report_phrases": internal,
        "long_sentences": long_sents,
        "repeated_paragraphs": repeated,
        "style_risk_score": score,
        "edit_level": _edit_level(score, term_hits, len(repeated), len(long_sents)),
    }


def _write_split(sections: list[SectionText]) -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in SPLIT_DIR.glob("*.txt"):
        old_path.unlink()
    for idx, section in enumerate(sections):
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", section.key).strip("_").lower()
        path = SPLIT_DIR / f"{idx:02d}_{slug}.txt"
        header = (
            f"title: {section.title}\n"
            f"family: {section.family}\n"
            f"pages: {section.start_page}-{section.end_page}\n\n"
        )
        path.write_text(header + section.text.strip() + "\n", encoding="utf-8")


def main() -> int:
    manifest = _load_json(BOOK_MANIFEST)
    pages = _page_texts(BOOK_PDF)
    sections: list[SectionText] = []

    front_pages = int(manifest.get("front_matter_pages") or 0)
    if front_pages:
        sections.append(
            SectionText(
                key="front_matter",
                title="Front matter",
                family="book",
                start_page=1,
                end_page=front_pages,
                text="\n\n".join(pages[:front_pages]),
            )
        )

    for item in manifest.get("items") or []:
        if item.get("kind") != "chapter":
            continue
        start = int(item["start_page"])
        end = int(item["end_page"])
        sections.append(
            SectionText(
                key=str(Path(item["source_pdf"]).stem),
                title=str(item["title"]),
                family=str(item["family"]),
                start_page=start,
                end_page=end,
                text="\n\n".join(pages[start - 1 : end]),
            )
        )

    _write_split(sections)
    rows = [_audit_section(section) for section in sections]
    levels = Counter(row["edit_level"] for row in rows if row["key"] != "front_matter")
    report = {
        "status": "PASS",
        "book_pdf": str(BOOK_PDF),
        "book_pages": len(pages),
        "front_matter_pages": front_pages,
        "chapter_count": len([row for row in rows if row["key"] != "front_matter"]),
        "split_dir": str(SPLIT_DIR),
        "edit_level_counts_chapters": dict(sorted(levels.items())),
        "terms_checked": {
            "technical_terms": TECHNICAL_TERMS,
            "trading_directive_terms": TRADING_DIRECTIVE_TERMS,
            "ai_flat_phrases": AI_FLAT_PHRASES,
            "internal_report_phrases": INTERNAL_REPORT_PHRASES,
        },
        "rows": rows,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "book_pages", "chapter_count", "split_dir", "edit_level_counts_chapters")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
