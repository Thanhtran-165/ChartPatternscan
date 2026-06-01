"""Canonical public-chapter content preparation.

This module is the single entry point for public editorial content. Pattern and
family scanners may compute facts, examples, and source notes, but they should
not own separate logic for mapping AI/human prose into `editorial_sections`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scanner.canonical_editorial_layer import (
    CANONICAL_AI_EDITORIAL_GATE_ID,
    CANONICAL_EDITORIAL_WORKFLOW_ID,
    REQUIRED_EDITORIAL_SECTIONS,
)


CANONICAL_CONTENT_GENERATOR_ID = "canonical_chapter_content_generator_v1"

CANONICAL_CONTENT_CONTRACT = {
    "generator_id": CANONICAL_CONTENT_GENERATOR_ID,
    "editorial_workflow_id": CANONICAL_EDITORIAL_WORKFLOW_ID,
    "ai_editorial_gate_id": CANONICAL_AI_EDITORIAL_GATE_ID,
    "purpose": "prepare public editorial sections from approved AI/human content before canonical PDF rendering",
    "required_sections": list(REQUIRED_EDITORIAL_SECTIONS),
    "allowed_source_kinds": [
        "approved_ai_sections",
        "canonical_test_sections",
    ],
    "must_not_do": [
        "generate different editorial section schemas per pattern",
        "let family builders map approved AI files themselves",
        "treat inline legacy payload sections as final content",
    ],
}


SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "intro", "tom-tat", "overview"),
    "tour": ("tour", "how_it_works", "mau-hinh-hoat-dong"),
    "failure": ("failure", "that-bai"),
    "statistics": ("statistics", "thong-ke"),
    "post_breakout": ("post_breakout", "hanh-vi-sau-pha-vo"),
    "size_volume": ("size_volume", "kich-thuoc-va-khoi-luong"),
    "tactics": ("tactics", "usage", "cach-su-dung"),
    "checklist": ("checklist_callout", "checklist", "usage_callout"),
}

PUBLIC_TEXT_REPLACEMENTS = {
    "tổng mẫu quét lịch sử": "tổng mẫu lịch sử",
    "Tổng mẫu quét lịch sử": "Tổng mẫu lịch sử",
    "mẫu quét lịch sử": "mẫu lịch sử",
    "Mẫu quét lịch sử": "Mẫu lịch sử",
    "tổng mẫu quét": "tổng mẫu lịch sử",
    "Tổng mẫu quét": "Tổng mẫu lịch sử",
    "mẫu quét": "mẫu lịch sử",
    "Mẫu quét": "Mẫu lịch sử",
    "source_full_pipe": "mốc đầy đủ",
    "Tham số hiện tại": "Dấu hiệu cần thấy",
    "Spike": "Cú xuyên giá",
    "spike": "cú xuyên giá",
    "Overlap": "Vùng chồng lấn",
    "overlap": "vùng chồng lấn",
    "target-first-before-adverse": "mục tiêu đến trước nhịp kéo ngược bất lợi",
    "Target-first-before-adverse": "mục tiêu đến trước nhịp kéo ngược bất lợi",
    "target-first": "đạt mục tiêu trước kéo ngược",
    "Target-first": "đạt mục tiêu trước kéo ngược",
    "target-hit": "tỷ lệ đạt mục tiêu",
    "Target-hit": "tỷ lệ đạt mục tiêu",
    "MFE": "mức đi thuận lợi tốt nhất",
    "MAE": "mức kéo ngược sâu nhất",
    "mfe": "mức đi thuận lợi tốt nhất",
    "mae": "mức kéo ngược sâu nhất",
    "breakout": "phá vỡ",
    "scanner": "bộ nhận diện",
    "pipeline": "quy trình",
    "payload": "bộ dữ liệu chương",
    "factory": "bộ dựng chương",
    "available-series": "phạm vi dữ liệu hiện có",
    "research-only": "tham khảo nghiên cứu",
    "setup": "cấu trúc mẫu",
    "Flag Family": "nhóm cờ",
    "Corporate actions": "sự kiện điều chỉnh giá",
    "delisted/halted": "mã hủy niêm yết hoặc tạm ngừng",
    "status tape": "lịch sử trạng thái giao dịch",
    "historical VN30/VN100 membership": "lịch sử thành phần VN30/VN100",
    "point-in-time universe": "danh sách cổ phiếu theo từng thời điểm",
    "low-liquidity": "thanh khoản thấp",
    "data_limited": "thiếu dữ liệu",
    "branch_id": "nhánh đọc",
    "regime": "bối cảnh thị trường",
    "bucket": "nhóm",
    "Premium": "Nhóm tốt",
    "premium": "nhóm tốt",
    "Standard": "Nhóm chuẩn",
    "standard": "nhóm chuẩn",
    "audit": "kiểm tra",
    "stop-loss": "ngưỡng rủi ro",
    "(lead-in)": "",
    "lead-in": "nhịp dẫn",
    "Lead-in": "Nhịp dẫn",
    "trendline": "đường xu hướng",
    "Trendline": "Đường xu hướng",
    "short setup": "hồ sơ bán khống",
    "short cấu trúc mẫu": "hồ sơ bán khống",
    "short cấu hình": "hồ sơ bán khống",
    "long-watchlist": "hồ sơ theo dõi hướng tăng",
    "long-theo dõi": "hồ sơ theo dõi hướng tăng",
    "vào lệnh": "xem xét tín hiệu",
    "dừng lỗ": "ngưỡng rủi ro",
}

SECTION_MIN_ITEMS = {
    "summary": 3,
    "tour": 2,
    "failure": 2,
    "statistics": 2,
    "post_breakout": 2,
    "size_volume": 2,
    "tactics": 2,
    "checklist": 5,
}

SECTION_MIN_CHARS = {
    "summary": 720,
    "tour": 420,
    "failure": 460,
    "statistics": 480,
    "post_breakout": 400,
    "size_volume": 380,
    "tactics": 420,
    "checklist": 180,
}

SECTION_READER_BRIDGES = {
    "summary": "Người đọc nên xem phần này như bản tóm tắt cách hành xử của mẫu hình sau khi được xác nhận. Vì vậy, kết luận không nằm ở một tỷ lệ đơn lẻ, mà ở sự kết hợp giữa mức đi thuận lợi, mức kéo ngược, thất bại và bối cảnh nơi mẫu xuất hiện. Điều này giúp chương đọc giống một tài liệu tham khảo biểu đồ hơn là một bảng số liệu rời rạc, đồng thời nhắc rằng một mẫu hình chỉ đáng tin hơn khi hình thái, xác nhận và đường đi sau đó cùng ủng hộ nhau.",
    "tour": "Cách đọc đúng là đi từ hình học sang xác nhận rồi mới tới số liệu. Nếu hình dạng có vẻ giống nhưng thiếu bối cảnh trước mẫu hoặc thiếu phiên xác nhận rõ, người đọc nên giảm độ tin cậy của toàn bộ phần thống kê phía sau.",
    "failure": "Phần thất bại giúp người đọc tránh chỉ nhìn các ví dụ đẹp. Một mẫu hợp lệ vẫn có thể không đi đủ xa, đi quá chậm, hoặc kéo ngược sâu trước khi đạt mục tiêu; những trường hợp đó là một phần của hồ sơ mẫu hình.",
    "statistics": "Các con số trong bảng nên được đọc như bản đồ xác suất có điều kiện. Điều này cho thấy người đọc cần hỏi mẫu có tạo được bất đối xứng đủ rõ giữa hướng thuận lợi và hướng bất lợi hay không; vì vậy không nên chỉ dựa vào một dòng kết quả đẹp.",
    "post_breakout": "Sau phá vỡ, thứ tự đường đi quan trọng ngang với độ lớn cuối cùng. Điều này cho thấy một mẫu đạt mục tiêu trước khi kéo ngược sâu có chất lượng khác với mẫu chỉ đạt mục tiêu sau một đoạn nhiễu dài; vì vậy người đọc cần xem tốc độ và trật tự đường đi cùng với tỷ lệ đạt mục tiêu.",
    "size_volume": "Kích thước, độ gọn và thanh khoản là phần giúp tách mẫu có thể đọc được khỏi vùng dao động nhiễu. Khi các điều kiện này yếu, người đọc nên xem kết quả như tham khảo thận trọng hơn.",
    "tactics": "Cách sử dụng phù hợp là biến chương thành checklist đọc biểu đồ: kiểm hình học, kiểm xác nhận, kiểm đường đi sau phá vỡ và kiểm bối cảnh. Điều này cho thấy chương không thay thế một hệ thống giao dịch có quy tắc thực thi riêng; vì vậy người đọc nên dùng nó để tăng hoặc giảm độ tin cậy của mẫu, không dùng như mệnh lệnh hành động.",
}

SECTION_READER_PADDING = {
    "summary": "Nói cách khác, phần mở đầu phải giúp người đọc hiểu mẫu này nên được dùng để quan sát điều gì, không chỉ biết nó có bao nhiêu mẫu. Nếu số liệu đẹp nhưng hình thái hoặc đường đi không sạch, chương vẫn phải giữ giọng thận trọng.",
    "tour": "Cách nhìn này cũng giúp tránh nhầm lẫn giữa một vùng giá ngẫu nhiên và một mẫu hình đã đủ điều kiện thống kê.",
    "failure": "Vì vậy, thất bại không phải phụ lục xấu đi kèm chương; nó là phần cho biết ranh giới sử dụng của mẫu.",
    "statistics": "Khi hai chỉ số mâu thuẫn, ví dụ tỷ lệ đạt mục tiêu khá nhưng kéo ngược cũng sâu, diễn giải nên ưu tiên sự cân bằng hơn là kết luận một chiều.",
    "post_breakout": "Đọc theo trình tự này giúp người đọc thấy khác biệt giữa mẫu chạy gọn ngay sau phá vỡ và mẫu cần nhiều nhiễu mới đi đúng hướng.",
    "size_volume": "Những điều kiện này không làm thay đổi tên mẫu hình, nhưng làm thay đổi độ tin cậy khi áp dụng vào một cổ phiếu cụ thể.",
    "tactics": "Do đó, phần sử dụng nên được đọc như một khung ra quyết định thận trọng: xác nhận thêm khi điều kiện đẹp, giảm kỳ vọng khi điều kiện thiếu.",
    "checklist": "Nếu một mục trong checklist không đạt, người đọc nên giảm độ tin cậy của kết luận thay vì cố ép mẫu vào câu chuyện có sẵn.",
}


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, Mapping) else {}


def _items(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def canonicalize_editorial_sections(raw_sections: Mapping[str, Any]) -> dict[str, list[str]]:
    """Map arbitrary approved section ids into the canonical eight-section schema."""

    normalized: dict[str, list[str]] = {}
    for canonical_id, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            values = [_clean_public_text(item) for item in _items(raw_sections.get(alias))]
            if values:
                normalized[canonical_id] = _ensure_reader_depth(canonical_id, values)
                break
    return normalized


def _clean_public_text(value: Any) -> str:
    out = str(value or "").strip()
    for old, new in PUBLIC_TEXT_REPLACEMENTS.items():
        out = out.replace(old, new)
    return " ".join(out.split())


def _clean_public_object(value: Any) -> Any:
    if isinstance(value, str):
        return _clean_public_text(value)
    if isinstance(value, list):
        return [_clean_public_object(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_public_object(item) for key, item in value.items()}
    return value


def _ensure_reader_depth(section: str, values: list[str]) -> list[str]:
    out = [item for item in values if item.strip()]
    min_items = SECTION_MIN_ITEMS.get(section, 1)
    min_chars = SECTION_MIN_CHARS.get(section, 0)
    bridge = SECTION_READER_BRIDGES.get(section)
    if bridge and bridge not in out:
        out.append(bridge)
    padding = SECTION_READER_PADDING.get(section)
    if padding and sum(len(item) for item in out) < min_chars and padding not in out:
        out.append(padding)
    if section == "checklist" and len(out) < min_items:
        out.extend(
            [
                "Kiểm tra hình học trước khi đọc kết quả.",
                "Chỉ đọc mẫu sau khi có xác nhận rõ.",
                "Luôn đặt kết quả thuận lợi cạnh mức kéo ngược.",
                "Giảm độ tin cậy nếu thanh khoản hoặc đường giá kém sạch.",
                "Không biến chương thành khuyến nghị giao dịch tự động.",
            ][: max(0, min_items - len(out))]
        )
    return out


def load_approved_editorial_sections(path: Path) -> dict[str, Any]:
    """Load an approved AI/human editorial artifact.

    Supported input shapes:
    - `{"approved_sections": [{"id": "intro", "paragraphs": [...], "callout": {"bullets": [...]}}]}`
    - `{"editorial_sections": {"summary": [...]}}`
    - `{"sections": {"summary": [...]}}`
    """

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    data = _read_json(path)
    raw_sections: dict[str, list[str]] = {}

    approved = data.get("approved_sections")
    if isinstance(approved, list):
        for section in approved:
            if not isinstance(section, Mapping):
                continue
            section_id = str(section.get("id") or "").strip()
            paragraphs = _items(section.get("paragraphs"))
            if section_id and paragraphs:
                raw_sections[section_id] = paragraphs
            callout = section.get("callout")
            if isinstance(callout, Mapping):
                bullets = _items(callout.get("bullets"))
                if section_id and bullets:
                    raw_sections[f"{section_id}_callout"] = bullets

    for key in ("editorial_sections", "sections"):
        candidate = data.get(key)
        if isinstance(candidate, Mapping):
            for section_id, value in candidate.items():
                values = _items(value)
                if values:
                    raw_sections[str(section_id)] = values

    captions = _clean_public_object(data.get("example_captions")) if isinstance(data.get("example_captions"), Mapping) else {}
    extras: dict[str, Any] = {}
    for key in ("source_rules_public", "recognition_mistakes", "section_hints"):
        value = data.get(key)
        if value:
            extras[key] = _clean_public_object(value)
    return {
        "sections": canonicalize_editorial_sections(raw_sections),
        "captions": dict(captions),
        "extras": extras,
        "source_path": str(path),
    }


def prepare_canonical_chapter_content(
    payload: Mapping[str, Any],
    *,
    approved_sections_path: Path | None = None,
    editorial_sections: Mapping[str, Any] | None = None,
    source_kind: str | None = None,
) -> dict[str, Any]:
    """Attach canonical editorial content and provenance to a payload."""

    if approved_sections_path is not None and editorial_sections is not None:
        raise ValueError("Provide either approved_sections_path or editorial_sections, not both")

    out = dict(payload)
    captions: Mapping[str, Any] = {}
    source_path = ""
    if approved_sections_path is not None:
        loaded = load_approved_editorial_sections(approved_sections_path)
        sections = loaded["sections"]
        captions = loaded["captions"]
        for key, value in (loaded.get("extras") or {}).items():
            out[key] = value
        source_path = str(approved_sections_path)
        resolved_source_kind = source_kind or "approved_ai_sections"
    elif editorial_sections is not None:
        if source_kind != "canonical_test_sections":
            raise ValueError(
                "Inline editorial_sections are not allowed for public chapters; "
                "provide an approved AI/refinement artifact instead"
            )
        sections = canonicalize_editorial_sections(editorial_sections)
        resolved_source_kind = source_kind
    else:
        raise ValueError("Canonical chapter content requires approved_sections_path or editorial_sections")

    if resolved_source_kind not in CANONICAL_CONTENT_CONTRACT["allowed_source_kinds"]:
        raise ValueError(f"Unsupported canonical content source kind: {resolved_source_kind}")

    missing = [section for section in REQUIRED_EDITORIAL_SECTIONS if not sections.get(section)]
    if missing:
        raise ValueError("Canonical chapter content missing sections: " + ", ".join(missing))

    out["editorial_sections"] = sections
    if captions:
        out["example_captions"] = dict(captions)
    if source_path:
        out["editorial_source_path"] = source_path
    out["canonical_content_generator_id"] = CANONICAL_CONTENT_GENERATOR_ID
    out["canonical_content_source_kind"] = resolved_source_kind
    out["canonical_content_contract"] = CANONICAL_CONTENT_CONTRACT
    out["canonical_content_generation_report"] = {
        "status": "PASS",
        "generator_id": CANONICAL_CONTENT_GENERATOR_ID,
        "source_kind": resolved_source_kind,
        "source_path": source_path,
        "section_count": len(sections),
        "sections": list(sections.keys()),
    }
    return out
