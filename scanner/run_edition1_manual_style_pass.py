"""Manual Vietnamese style pass for Edition 1 public chapters.

This pass edits only reader-facing prose inside canonical editorial sections.
It does not change metrics, scanners, examples, charts, targets, or labels.
The same edits are applied to both the rendered payload and the approved
editorial artifact referenced by ``editorial_source_path`` so render-only builds
cannot silently fall back to older wording.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "artifacts/final_chapters/final_chapters_manifest.json"
DEFAULT_REPORT = ROOT / "artifacts/book_level/edition_1/manual_style_pass_report.json"

SECTION_KEYS = ("summary", "tour", "failure", "statistics", "post_breakout", "size_volume", "tactics", "checklist")

DIRECT_REPLACEMENTS = (
    ("khuyến nghị mua/bán", "lời chỉ dẫn hành động"),
    ("khuyến nghị mua, bán hay bán khống", "lời chỉ dẫn mua, bán hay bán khống"),
    ("khuyến nghị mua, bán hay nắm giữ", "lời chỉ dẫn mua, bán hay nắm giữ"),
    ("khuyến nghị mua hay bán", "lời chỉ dẫn mua hay bán"),
    ("khuyến nghị mua hoặc bán", "lời chỉ dẫn mua hoặc bán"),
    ("khuyến nghị mua bán", "lời chỉ dẫn mua bán"),
    ("khuyến nghị mua", "lời mời mua"),
    ("khuyến nghị bán", "lời mời bán"),
    ("mức cắt lỗ", "ngưỡng phòng vệ"),
    ("mức cảnh báo cắt lỗ", "mức cảnh báo phòng vệ"),
    ("điểm cắt lỗ", "ngưỡng phòng vệ"),
    ("kỷ luật cắt lỗ", "kỷ luật phòng vệ"),
    ("cắt lỗ/thoát vị thế", "giảm rủi ro hoặc đánh giá lại vị thế"),
    ("cắt lỗ máy móc", "phòng vệ máy móc"),
    ("dừng lỗ", "giảm rủi ro"),
    ("vào hay thoát khỏi một vị thế", "tăng hay giảm rủi ro với một vị thế"),
    ("điểm vào lệnh", "điểm hành động"),
    ("vào lệnh", "hành động"),
    ("điểm mua", "điểm hành động"),
    ("tín hiệu mua mới", "lý do mua mới"),
    ("tín hiệu giảm đã yếu đi", "nhịp giảm đã kém sạch hơn"),
)

FLAT_PHRASE_REPLACEMENTS = {
    "Dữ liệu cho thấy": (
        "Trong mẫu lịch sử này,",
        "Nhìn vào các lần xuất hiện đã đo được,",
        "Các quan sát lịch sử gợi ý rằng",
        "Ở tập mẫu này,",
    ),
    "Điều này cho thấy": (
        "Nói cách khác,",
        "Cách đọc thực tế là",
        "Với người đọc biểu đồ,",
        "Điểm đáng chú ý là",
        "Ở đây,",
    ),
    "Có thể thấy rằng": (
        "Có thể đọc là",
        "Điểm dễ nhận ra là",
    ),
    "Nhìn chung,": (
        "Ở bức tranh tổng thể,",
        "Nếu thu gọn lại,",
    ),
    "Tóm lại:": (
        "Điểm cần nhớ:",
        "Cách đọc ngắn gọn:",
    ),
    "Tóm lại,": (
        "Điểm cần nhớ là",
        "Nói gọn lại,",
        "Với mẫu này,",
    ),
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _sections(payload: Mapping[str, Any]) -> dict[str, list[Any]]:
    value = payload.get("editorial_sections")
    return value if isinstance(value, dict) else {}


def _style_edit_text(text: str, *, title: str, section: str, index: int) -> tuple[str, list[str]]:
    original = text
    edits: list[str] = []
    out = text
    for old, new in DIRECT_REPLACEMENTS:
        if old in out:
            out = out.replace(old, new)
            edits.append(f"replace:{old}")

    for phrase, choices in FLAT_PHRASE_REPLACEMENTS.items():
        if phrase not in out:
            continue
        seed = sum(ord(ch) for ch in f"{title}|{section}|{index}|{phrase}")
        counter = -1

        def repl(_: re.Match[str]) -> str:
            nonlocal counter
            counter += 1
            return choices[(seed + counter) % len(choices)]

        out = re.sub(re.escape(phrase), repl, out)
        edits.append(f"soften_flat:{phrase}")

    if out.startswith("Cách đọc mới của chương này là quản lý bẫy giảm sau phá vỡ"):
        match = re.search(r"Trong\s+(.+)$", out)
        tail = match.group(1) if match else out
        out = (
            f"Với {title.lower()}, điểm cần đọc kỹ là nhịp quay lại sau phá vỡ giảm. "
            f"Mẫu không chỉ trả lời câu hỏi giá có giảm tiếp hay không, mà còn cho biết phá vỡ ấy có sạch hay nhanh chóng biến thành bẫy giảm. "
            f"Trong {tail}"
        )
        edits.append("rewrite:bear_trap_summary")

    if out.startswith("Thất bại quan trọng nhất của một phá vỡ giảm không chỉ là giá không giảm tiếp"):
        out = out.replace(
            "Thất bại quan trọng nhất của một phá vỡ giảm không chỉ là giá không giảm tiếp, mà là cú quay lại vùng phá vỡ quá nhanh khiến người đọc nhầm phá vỡ giảm thành tín hiệu sạch.",
            "Một phá vỡ giảm gây rắc rối nhất khi giá quay lại vùng vừa bị xuyên thủng quá nhanh. Khi đó, vấn đề không chỉ là giá không giảm tiếp, mà là người đọc dễ tưởng nhịp giảm ban đầu sạch hơn thực tế.",
        )
        edits.append("rewrite:bear_trap_failure")

    if out.startswith("Bảng cảnh báo cắt lỗ bổ sung một câu hỏi thực dụng"):
        out = out.replace(
            "Bảng cảnh báo cắt lỗ bổ sung một câu hỏi thực dụng cho số liệu hậu phá vỡ:",
            "Bảng cảnh báo bẫy giảm bổ sung một câu hỏi thực dụng cho phần sau phá vỡ:",
        )
        edits.append("rewrite:bear_trap_statistics")

    if out.startswith("Khi phá vỡ giảm bị quay lại vùng phá vỡ"):
        out = out.replace(
            "Khi phá vỡ giảm bị quay lại vùng phá vỡ, chương nên được đọc như hồ sơ kiểm tra độ sạch của nhịp giảm.",
            "Khi giá quay lại vùng phá vỡ, chương nên được đọc như một phép thử độ sạch của nhịp giảm.",
        )
        edits.append("rewrite:bear_trap_post_breakout")

    if out.startswith("Quy tắc sử dụng thực tế: theo dõi cửa sổ 5/10/20 phiên sau phá vỡ giảm."):
        out = out.replace(
            "Quy tắc sử dụng thực tế: theo dõi cửa sổ 5/10/20 phiên sau phá vỡ giảm.",
            "Cách dùng gọn: quan sát các cửa sổ 5/10/20 phiên sau phá vỡ giảm.",
        )
        edits.append("rewrite:bear_trap_tactics")

    if "không đưa ra một" in out and "cụ thể" in out:
        out = out.replace("không đưa ra một", "không ấn định một")
        edits.append("soften:not_specify")

    # Remove a few report-like labels while keeping the meaning.
    out = out.replace("Ranh giới giữa tham khảo và hành động:", "Cần giữ ranh giới rõ:")
    out = out.replace("Nhấn mạnh lần cuối:", "Điểm cần nhớ:")

    if out != original and not edits:
        edits.append("style:minor")
    return out, edits


def _edit_sections(payload: dict[str, Any], *, title: str) -> tuple[int, Counter[str]]:
    sections = _sections(payload)
    changed = 0
    edit_counts: Counter[str] = Counter()
    for section in SECTION_KEYS:
        values = sections.get(section)
        if not isinstance(values, list):
            continue
        for index, value in enumerate(values):
            if not isinstance(value, str):
                continue
            edited, edits = _style_edit_text(value, title=title, section=section, index=index)
            if edited != value:
                values[index] = edited
                changed += 1
                edit_counts.update(edits)
    return changed, edit_counts


def _chapter_paths(row: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in ("payload", "refined_ai_sections"):
        value = row.get(key)
        if value:
            path = ROOT / str(value)
            if path.exists():
                paths.append(path)
    stages = row.get("chapter_writing_stages")
    if isinstance(stages, Mapping):
        for key in ("source_guided_ai_sections", "refined_ai_sections"):
            value = stages.get(key)
            if value:
                path = ROOT / str(value)
                if path.exists():
                    paths.append(path)
    payload_path = ROOT / str(row.get("payload") or "")
    if payload_path.exists():
        payload = _read_json(payload_path)
        source = payload.get("editorial_source_path") if isinstance(payload, Mapping) else None
        if source:
            source_path = ROOT / str(source)
            if source_path.exists():
                paths.append(source_path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def run(*, inventory_path: Path, report_path: Path, apply: bool) -> dict[str, Any]:
    inventory = _read_json(inventory_path)
    rows = inventory.get("chapters") if isinstance(inventory, Mapping) else []
    report_rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    files_changed = 0
    chapters_changed = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        title = str(row.get("title") or row.get("pattern_id") or "")
        chapter_changed = 0
        chapter_counts: Counter[str] = Counter()
        changed_files: list[str] = []
        for path in _chapter_paths(row):
            payload = _read_json(path)
            if not isinstance(payload, dict):
                continue
            changed, edit_counts = _edit_sections(payload, title=title)
            if changed:
                chapter_changed += changed
                chapter_counts.update(edit_counts)
                changed_files.append(str(path.relative_to(ROOT)))
                if apply:
                    _write_json(path, payload)
                    files_changed += 1
        if chapter_changed:
            chapters_changed += 1
        totals.update(chapter_counts)
        report_rows.append(
            {
                "pattern_id": row.get("pattern_id"),
                "title": title,
                "family": row.get("family"),
                "changed_paragraphs": chapter_changed,
                "edit_counts": dict(chapter_counts),
                "files": changed_files,
            }
        )
    report = {
        "status": "PASS",
        "mode": "apply" if apply else "dry_run",
        "chapter_count": len([row for row in rows if isinstance(row, Mapping)]),
        "chapters_changed": chapters_changed,
        "files_changed": files_changed if apply else 0,
        "changed_paragraphs": sum(int(row["changed_paragraphs"]) for row in report_rows),
        "edit_counts": dict(totals),
        "rows": report_rows,
    }
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual style pass for Edition 1 chapter prose.")
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = run(inventory_path=Path(args.inventory), report_path=Path(args.report), apply=bool(args.apply))
    print(json.dumps({k: report[k] for k in ("status", "mode", "chapter_count", "chapters_changed", "changed_paragraphs", "files_changed", "edit_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
