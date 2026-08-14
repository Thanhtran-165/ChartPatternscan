"""Build Edition 2 book PDF (64 chương, gồm Harami Family mới).

Tái dùng toàn bộ pipeline merge của `build_edition1_book`, chỉ override:
- thư mục/namen output -> artifacts/book_level/edition_2
- nhãn ấn bản -> "Ấn bản 2" / "thứ hai" / bulkowski_vietnam_edition_2
- FAMILY_ORDER/FAMILY_LABELS -> thêm harami_family (giữ bảng chữ cái A-Z).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner.build_edition1_book as base  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "book_level" / "edition_2"
base.OUT_DIR = OUT_DIR
base.EDITION_PDF = OUT_DIR / "bulkowski_vietnam_edition_2.pdf"
base.EDITION_MANIFEST = OUT_DIR / "bulkowski_vietnam_edition_2_manifest.json"
base.FRONT_MATTER_PDF = OUT_DIR / "edition_2_front_matter.pdf"
base.COVER_PDF = OUT_DIR / "edition_2_cover.pdf"
base.EDITION_ID = "bulkowski_vietnam_edition_2"
base.EDITION_LABEL = "Ấn bản 2"
base.EDITION_ORDINAL = "thứ hai"

_EXTRA_FAMILY_ORDER = ["harami_family"]
base.FAMILY_ORDER = sorted(set(base.FAMILY_ORDER) | set(_EXTRA_FAMILY_ORDER))
base.FAMILY_LABELS = {
    **dict(base.FAMILY_LABELS),
    "harami_family": "Harami Family",
}


def main() -> None:
    manifest = base.build_edition()
    print(
        json.dumps(
            {
                "status": "PASS",
                "pdf": manifest["pdf"],
                "total_pages": manifest["total_pages"],
                "chapters": manifest["chapter_count"],
                "appendices_removed": sum(1 for item in manifest["items"] if item.get("appendix_removed")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
