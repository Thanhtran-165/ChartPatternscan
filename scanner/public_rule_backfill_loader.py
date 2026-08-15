"""Tải prose quy tắc nhận diện công khai đã duyệt (backfill 14/08) cho chương sách.

Bối cảnh đợt B (15/08): publication core đòi `source_rules_public` NGAY khi
render payload, trong khi flow 14/08 backfill (DeepSeek) chạy SAU render và ghi
vào payload in-place. Kết quả backfill AI đã duyệt vẫn còn nguyên tại
`artifacts/governance/final_chapters/governance/public_rule_backfill/<pattern_id>/parsed.json`.

Builder tái dùng nguyên văn bản đó — không gọi AI lại, văn bản chương không
đổi ngoài số liệu đợt B.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
BACKFILL_DIR = ROOT / "artifacts/governance/final_chapters/governance/public_rule_backfill"
BACKFILL_ID = "final_chapter_public_rule_backfill_v1"


def load_backfill_public_rules(pattern_id: str) -> List[Dict[str, str]]:
    """Trả rows quy tắc công khai từ parsed.json đã duyệt; [] nếu không có/lỗi.

    Chuẩn hoá giống `_normalize_rules` của backfill_final_chapter_public_rules.py
    để rows giống hệt những gì backfill ghi vào payload 14/08.
    """
    parsed_path = BACKFILL_DIR / str(pattern_id) / "parsed.json"
    if not parsed_path.exists():
        return []
    try:
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, Mapping):
        return []
    raw = parsed.get("source_rules_public")
    if not isinstance(raw, list):
        raw = parsed.get("rules")
    rows: List[Dict[str, str]] = []
    if not isinstance(raw, list):
        return rows
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        rule = str(item.get("rule") or item.get("public_rule") or item.get("public_description") or "").strip()
        application = str(item.get("application") or item.get("how_to_apply") or item.get("importance") or "").strip()
        avoid = str(item.get("avoid") or item.get("common_mistake") or item.get("common_mistakes") or "").strip()
        if not rule or not application:
            continue
        row: Dict[str, str] = {
            "rule_id": str(item.get("rule_id") or "").strip(),
            "rule": rule,
            "application": application,
        }
        if avoid:
            row["avoid"] = avoid
        rows.append(row)
    return rows[:8]


def apply_backfill_public_rules(payload: Dict[str, Any], pattern_id: str) -> None:
    """Gán `source_rules_public` + provenance vào payload nếu còn thiếu.

    Không ghi đè nếu payload đã có rules (tôn trọng nguồn khác ưu tiên hơn).
    """
    if payload.get("source_rules_public"):
        return
    rows = load_backfill_public_rules(pattern_id)
    if not rows:
        return
    payload["source_rules_public"] = rows
    payload["source_rules_public_provenance"] = {
        "backfill_id": BACKFILL_ID,
        "artifact": str((BACKFILL_DIR / str(pattern_id) / "parsed.json").relative_to(ROOT)),
        "reused_by": "dotb_builder_backfill_reuse_v1",
    }
