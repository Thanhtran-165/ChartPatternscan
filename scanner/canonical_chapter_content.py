"""Canonical public-chapter content preparation.

This module is the single entry point for public editorial content. Pattern and
family scanners may compute facts, examples, and source notes, but they should
not own separate logic for mapping AI/human prose into `editorial_sections`.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from scanner.canonical_editorial_layer import (
    CANONICAL_AI_EDITORIAL_GATE_ID,
    CANONICAL_EDITORIAL_WORKFLOW_ID,
    REQUIRED_EDITORIAL_SECTIONS,
    validate_canonical_editorial_sections,
)


CANONICAL_CONTENT_GENERATOR_ID = "canonical_chapter_content_generator_v1"
UNLOCKED_NUMERIC_CLAIM_RE = re.compile(r"\b\d+[.,]\d+\s*%|\b\d+[.,]\d+x\b")

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
        "pad thin or missing public prose with deterministic fallback text",
    ],
}


SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "quick_read": ("quick_read", "doc_nhanh_chuong_nay", "doc-nhanh-chuong-nay", "quick-read", "fast_read"),
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
    "được chuyển thành câu chuyện thị giác": "được diễn giải bằng ngôn ngữ biểu đồ",
    "được chuyển thành": "được diễn giải thành",
    "chúng tôi kỳ vọng bạn": "người đọc nên",
    "chúng tôi": "chương này",
    "tổng mẫu quét lịch sử": "tổng mẫu lịch sử",
    "Tổng mẫu quét lịch sử": "Tổng mẫu lịch sử",
    "mẫu quét lịch sử": "mẫu lịch sử",
    "Mẫu quét lịch sử": "Mẫu lịch sử",
    "tổng mẫu quét": "tổng mẫu lịch sử",
    "Tổng mẫu quét": "Tổng mẫu lịch sử",
    "mẫu quét": "mẫu lịch sử",
    "Mẫu quét": "Mẫu lịch sử",
    "source_full_pipe": "mốc đầy đủ",
    "source_full_height": "mốc đầy đủ",
    "source_full_pole": "mốc đầy đủ",
    "conservative_half_": "mốc thận trọng",
    "textbook_success": "ví dụ đạt mục tiêu",
    "middle_case": "ví dụ trung vị",
    "zero and stale": "không có dữ liệu hợp lệ",
    "failure 5pct rate": "tỷ lệ thất bại 5%",
    "target first before adverse": "đạt mục tiêu trước kéo ngược",
    "layer missing": "chưa có lớp kiểm tra tương ứng",
    "unknown": "chưa đủ dữ liệu",
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
    "watchlist-reference": "tham khảo theo dõi",
    "watchlist": "theo dõi",
    "vào lệnh": "xem xét tín hiệu",
    "dừng lỗ": "ngưỡng rủi ro",
}


def _fact_number(value: Any, digits: int = 2) -> str:
    """Format a locked payload fact for reader-facing Vietnamese prose."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "chưa đủ dữ liệu"
    if not math.isfinite(number):
        return "chưa đủ dữ liệu"
    if digits == 0:
        return f"{number:.0f}".replace(".", ",")
    return f"{number:.{digits}f}".replace(".", ",")


def _fact_count(value: Any) -> str:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return "chưa đủ dữ liệu"
    return f"{number:,}".replace(",", ".")


def _locked_fact_sections(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    """Create concise fact-led paragraphs from the current locked payload.

    Approved editorial prose can outlive a rescan.  The publication contract
    therefore regenerates only the statistic-heavy sections from the current
    payload while keeping the approved explanatory sections intact.  This is a
    deterministic rewrite; it never calls an external model.
    """

    ref = payload.get("chapter_reference") if isinstance(payload.get("chapter_reference"), Mapping) else {}
    target = payload.get("target_calibration") if isinstance(payload.get("target_calibration"), Mapping) else {}
    base = target.get("base_target") if isinstance(target.get("base_target"), Mapping) else {}
    legacy = target.get("legacy_target") if isinstance(target.get("legacy_target"), Mapping) else {}
    name = str(payload.get("pattern_name") or payload.get("pattern_id") or "mẫu hình").strip()
    n = _fact_count(ref.get("public_grade_events") or ref.get("events") or ref.get("evaluated_events"))
    base_multiple = _fact_number(base.get("target_multiple"), 1)
    base_hit = _fact_number(base.get("target_hit_rate"))
    base_first = _fact_number(base.get("target_first_before_adverse_5pct_rate"))
    failure = _fact_number(base.get("failure_5pct_rate") or ref.get("failure_5pct_rate"))
    mfe = _fact_number(ref.get("median_mfe_pct"))
    mae = _fact_number(ref.get("median_mae_pct"))
    legacy_hit = _fact_number(legacy.get("target_hit_rate"))
    legacy_multiple = _fact_number(legacy.get("target_multiple"), 1)

    quick_read = [
        f"{name} xuất hiện {n} lần trong nhóm dữ liệu đủ điều kiện công bố hiện tại. Đây là quy mô của mẫu được dùng cho phần diễn giải, không phải số lần mọi bộ lọc nhận diện từng ghi nhận.",
        f"Ở mốc cơ sở {base_multiple}x chiều cao mẫu, tỷ lệ đạt mục tiêu là {base_hit}%. Tỷ lệ thất bại 5% là {failure}%, nghĩa là hình thái đúng vẫn có thể đi chậm hoặc quay đầu sau phá vỡ.",
        f"Tỷ lệ đạt mục tiêu trước khi chịu nhịp kéo ngược 5% là {base_first}%. Mức đi thuận lợi tốt nhất trung vị là {mfe}%, còn mức kéo ngược sâu nhất trung vị là {mae}%.",
        "Các con số này nên được đọc như bản đồ hành vi sau phá vỡ: hình thái là điều kiện đầu vào, còn chất lượng đường đi và bối cảnh mới quyết định mức độ đáng chú ý của từng biểu đồ.",
    ]
    summary = [
        f"Trong phạm vi dữ liệu công bố, {name} có {n} mẫu đủ điều kiện để đọc thống kê. Con số này là mẫu số của chương hiện tại và cần được giữ nguyên khi so sánh các bảng trong cùng chương. Nó không đại diện cho mọi cổ phiếu hay mọi giai đoạn thị trường, mà chỉ cho phần dữ liệu đã qua các điều kiện công bố của chương.",
        f"Mốc cơ sở {base_multiple}x đạt {base_hit}% và thất bại 5% ở {failure}%. Mốc đầy đủ {legacy_multiple}x đạt {legacy_hit}%, nếu có, trả lời một câu hỏi xa hơn và không nên được dùng thay cho mốc cơ sở. Hai mốc cùng tồn tại để người đọc phân biệt kỳ vọng thực tế với kịch bản chạy xa, thay vì trộn chúng thành một tỷ lệ duy nhất.",
        f"Trung vị đường đi thuận lợi là {mfe}%, trong khi kéo ngược sâu nhất là {mae}%. Vì vậy, người đọc nên chuẩn bị cho một hành trình không thẳng và kiểm tra cả kịch bản mẫu không đạt mục tiêu. Phần ví dụ và phụ lục được giữ lại để cho thấy cùng một hình thái có thể dẫn tới những đường đi rất khác nhau sau phiên xác nhận.",
    ]
    failure_section = [
        f"Thất bại 5% của {name} ở mức {failure}% trong mẫu số hiện tại. Đây là tỷ lệ đường đi không đạt ngưỡng tối thiểu sau phá vỡ, không phải tỷ lệ hình thái bị nhận diện sai.",
        f"Chỉ {base_first}% đạt mốc cơ sở trước khi chịu nhịp kéo ngược 5%. Điều đó nhắc rằng tỷ lệ đạt cuối cửa sổ không mô tả đầy đủ thứ tự diễn biến mà người đọc phải trải qua.",
        "Khi giá không tiếp tục theo hướng dự kiến, hãy quay lại kiểm tra nến xác nhận, vùng phá vỡ và cấu trúc đỉnh đáy gần nhất. Một mẫu đẹp không loại bỏ được rủi ro thất bại.",
        "Ví dụ thất bại trong chương được giữ lại để chỉ rõ ranh giới của thống kê, không phải để tạo một ngoại lệ gây ấn tượng.",
    ]
    statistics = [
        f"Mốc cơ sở {base_multiple}x đạt {base_hit}% trong {n} mẫu. Đây là xác suất mô tả theo cửa sổ đo của chương, không phải lời hứa rằng mọi biểu đồ tương lai sẽ chạm mục tiêu.",
        f"Tỷ lệ đạt mục tiêu trước kéo ngược 5% là {base_first}%; thất bại 5% là {failure}%. Hai tỷ lệ này bổ sung cho nhau: một tỷ lệ nói về thứ tự đường đi, tỷ lệ kia nói về việc có đi đủ ngưỡng tối thiểu hay không.",
        f"Mức đi thuận lợi tốt nhất trung vị {mfe}% đặt cạnh mức kéo ngược sâu nhất trung vị {mae}% cho thấy biên độ cơ hội và nhiễu có thể cùng lớn. Người đọc không nên chỉ chọn mốc có tỷ lệ đẹp hơn.",
        f"Nếu bảng có thêm mốc đầy đủ {legacy_multiple}x với tỷ lệ {legacy_hit}%, hãy xem đó là thước đo độ xa của nhịp đi sau phá vỡ; mốc cơ sở vẫn là điểm neo để đọc trước.",
    ]
    post_breakout = [
        f"Sau phá vỡ, {base_first}% mẫu đạt mốc cơ sở trước khi chịu kéo ngược 5%. Điều này cho thấy đường đi gập ghềnh là trạng thái thường gặp chứ không phải ngoại lệ.",
        f"Mức đi thuận lợi tốt nhất trung vị {mfe}% và mức kéo ngược sâu nhất trung vị {mae}% nên được đọc cùng nhau. Nếu giá quay lại vùng phá vỡ, người đọc cần đánh giá lại cấu trúc thay vì mặc định mẫu đã hỏng.",
        "Thời gian chạm mục tiêu phụ thuộc cửa sổ quan sát và dữ liệu hậu phá vỡ. Mục tiêu là mốc tham khảo để theo dõi, không phải thời hạn bắt buộc của một giao dịch.",
    ]
    return {
        "quick_read": quick_read,
        "summary": summary,
        "failure": failure_section,
        "statistics": statistics,
        "post_breakout": post_breakout,
    }


def _remove_unlocked_numeric_claims(sections: dict[str, list[str]]) -> dict[str, list[str]]:
    """Remove old-generation scalar claims from explanatory sections.

    Example captions and generated tables retain event-level facts.  This
    guard applies only to approved prose sections outside the deterministic
    fact blocks, where an old percentage can otherwise survive a rescan.
    """

    replacements = {
        "tour": "Người đọc nên đi theo thứ tự hình thái, phiên xác nhận và đường đi sau phá vỡ; một tỷ lệ riêng lẻ không thay thế việc kiểm tra biểu đồ.",
        "size_volume": "Độ gọn, khối lượng và vị trí phá vỡ chỉ là các lớp bối cảnh. Người đọc nên kết hợp chúng với hình thái và đường đi thực tế thay vì suy luận từ một con số cũ.",
        "tactics": "Các mốc mục tiêu và tỷ lệ lịch sử chỉ là tham khảo trong phạm vi chương. Người đọc không nên biến chúng thành bảo đảm cho biểu đồ mới hoặc quyết định đầu tư.",
        "checklist": "Người đọc nên xem mục tiêu như mốc tham khảo của chương; sau đó xác nhận lại hình thái, đường đi và bối cảnh trên biểu đồ thực tế.",
    }
    for section_id, fallback in replacements.items():
        values = sections.get(section_id) or []
        sections[section_id] = [
            fallback if UNLOCKED_NUMERIC_CLAIM_RE.search(item) else item
            for item in values
        ]
    return sections

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
                normalized[canonical_id] = values
                break
    return normalized


def _clean_public_text(value: Any) -> str:
    out = str(value or "").strip()
    for old, new in PUBLIC_TEXT_REPLACEMENTS.items():
        out = out.replace(old, new)
    out = re.sub(r"\b(?:conservative_half|source_full|local)_[A-Za-z0-9_]+\b", "mốc tham chiếu", out)
    out = re.sub(r"\b(?:textbook_success|middle_case|failure)\b", "ví dụ minh họa", out)
    out = out.replace("chất lượng đường giá clean", "chất lượng đường giá sạch")
    return " ".join(out.split())


def _clean_public_object(value: Any) -> Any:
    if isinstance(value, str):
        return _clean_public_text(value)
    if isinstance(value, list):
        return [_clean_public_object(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_public_object(item) for key, item in value.items()}
    return value


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

    # Rescans may update locked payload facts while an approved editorial
    # artifact still contains the previous generation's percentages.  Replace
    # only the statistic-heavy reader sections with deterministic prose built
    # from the current payload; explanatory sections remain editorially stable.
    locked_sections = _locked_fact_sections(out)
    for section_id, paragraphs in locked_sections.items():
        if paragraphs:
            sections[section_id] = [_clean_public_text(item) for item in paragraphs]
    sections = _remove_unlocked_numeric_claims(sections)

    gate_report = validate_canonical_editorial_sections({"editorial_sections": sections})
    if gate_report["status"] != "PASS":
        raise ValueError(f"Canonical chapter content editorial gate failed: {gate_report['failures']}")

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
        "locked_fact_sections_rewritten": sorted(locked_sections),
        "editorial_gate_report": gate_report,
    }
    return out
