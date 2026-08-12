"""Publication-semantic specs for Triangle Family public chapters."""

from __future__ import annotations

from typing import Any, Mapping

from scanner.publication_semantic_contract import PUBLICATION_SEMANTIC_GATE_ID


TRIANGLE_PUBLICATION_SPEC_VERSION = "triangle_family_publication_spec_v1"

PUBLIC_FORBIDDEN_TERMS = (
    "Temporal robustness",
    "Regime x liquidity interaction",
    "Manual visual",
    "pass rate",
    "Publication core",
    "release gate",
    "source grounding",
    "source_alignment",
    "publication_quality_tier",
    "data_limited",
    "headline",
    "candidate",
    "public-grade",
    "premium",
    "standard",
    "loose",
    "low-liquidity",
    "aggregate",
    "audit",
)


PATTERN_REQUIRED_PHRASES = {
    "triangles_ascending": (
        "Tam giác tăng",
        "kháng cự",
        "đáy sau cao hơn đáy trước",
        "giá đóng cửa vượt",
        "0,5x",
    ),
    "triangles_descending": (
        "Tam giác giảm",
        "hỗ trợ",
        "đỉnh sau thấp hơn đỉnh trước",
        "giá đóng cửa phá xuống",
        "0,5x",
    ),
    "triangles_symmetrical": (
        "Tam giác cân",
        "hai biên hội tụ",
        "tách hướng phá vỡ",
        "giá đóng cửa phá",
        "0,5x",
    ),
}

PATTERN_PUBLIC_RULE_ROWS = {
    "triangles_ascending": [
        ["Kháng cự phía trên tương đối ngang.", "Các đỉnh trong mẫu phải chạm cùng một vùng giá đủ rõ; nếu vùng này quá dốc hoặc quá rộng, mẫu bị hạ chất lượng."],
        ["Đáy sau cao hơn đáy trước.", "Đường hỗ trợ đi lên cho thấy lực mua nâng dần dưới vùng kháng cự."],
        ["Biên độ có xu hướng nén lại.", "Mẫu đáng tin hơn khi không gian dao động hẹp dần trước phiên xác nhận."],
        ["Chỉ xác nhận khi giá đóng cửa vượt kháng cự.", "Nến xuyên trong phiên nhưng đóng cửa quay lại dưới vùng kháng cự không được xem là phá vỡ đầy đủ."],
        ["Mục tiêu đo bằng chiều cao tam giác.", "Mốc 1,0x là mốc nguồn của chương; 0,5x được dùng như lớp đọc thận trọng."],
    ],
    "triangles_descending": [
        ["Hỗ trợ phía dưới tương đối ngang.", "Các đáy trong mẫu phải cùng nằm quanh một vùng đỡ đủ rõ."],
        ["Đỉnh sau thấp hơn đỉnh trước.", "Đường kháng cự đi xuống cho thấy lực bán ép dần vào vùng hỗ trợ."],
        ["Biên độ có xu hướng nén lại.", "Mẫu thiếu nén dễ trở thành vùng dao động rộng hơn là tam giác giảm."],
        ["Chỉ xác nhận khi giá đóng cửa phá xuống hỗ trợ.", "Không dùng cú xuyên trong phiên làm xác nhận nếu giá đóng cửa hồi lại vào thân mẫu."],
        ["Mục tiêu đo bằng chiều cao tam giác.", "Mốc đầy đủ giữ vai trò nguồn; các mốc thấp hơn giúp đọc mức thận trọng trong dữ liệu Việt Nam."],
    ],
    "triangles_symmetrical": [
        ["Hai biên phải hội tụ.", "Đỉnh thấp dần và đáy cao dần cùng ép giá vào vùng hẹp hơn."],
        ["Mẫu có thể phá vỡ theo hai hướng.", "Kết luận phải tách hướng phá vỡ; không gộp phá lên và phá xuống thành một câu chung."],
        ["Chờ giá đóng cửa phá khỏi biên.", "Chỉ xác nhận khi giá đóng cửa nằm rõ ngoài biên theo hướng phá vỡ."],
        ["Đường đi sau phá vỡ quyết định chất lượng.", "Đọc tỷ lệ đạt trước kéo ngược cùng MAE, không chỉ nhìn tỷ lệ chạm mục tiêu."],
        ["Mục tiêu đo bằng chiều cao vùng nén.", "Mốc 0,5x là mốc thận trọng; 1,0x là tham chiếu đầy đủ khi phá vỡ đi xa."],
    ],
}


def build_triangle_publication_spec(*, pattern_id: str, title: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build a machine-checkable semantic spec for one Triangle chapter."""

    required = list(PATTERN_REQUIRED_PHRASES.get(pattern_id, (title, "0,5x")))
    return {
        "status": "PASS",
        "semantic_gate_id": PUBLICATION_SEMANTIC_GATE_ID,
        "publication_spec_id": f"{pattern_id}_{TRIANGLE_PUBLICATION_SPEC_VERSION}",
        "pattern_id": pattern_id,
        "family": "triangle_family",
        "spec_scope": "pattern_chapter",
        "variant_specific": True,
        "public_required_phrases": required,
        "public_forbidden_terms": list(PUBLIC_FORBIDDEN_TERMS),
        "public_rule_rows": PATTERN_PUBLIC_RULE_ROWS.get(pattern_id, []),
        "title": title,
        "source_chapter": spec.get("local_source_chapter"),
        "public_story_contract": {
            "must_read_like": "public investor-facing chart-pattern chapter",
            "must_not_read_like": "internal audit, scanner QA, or release-gate report",
            "allowed_data_scope": "available-series scope with explicit caveats",
        },
    }


def sanitize_triangle_public_text(text: Any) -> str:
    """Remove internal English publication terms from public Triangle prose."""

    out = str(text)
    replacements = {
        "premium + standard": "nhóm tốt nhất + nhóm chuẩn",
        "premium/standard": "nhóm tốt nhất/chuẩn",
        "premium và standard": "nhóm tốt nhất và nhóm chuẩn",
        "Premium tier": "Nhóm tốt nhất",
        "premium tier": "nhóm tốt nhất",
        "premium": "nhóm tốt nhất",
        "standard": "nhóm chuẩn",
        "public-grade": "đủ điều kiện công bố",
        "data_limited": "thiếu dữ liệu",
        "loose": "lỏng",
        "audit nền": "kiểm tra nền",
        "audit": "kiểm tra",
        "headline": "kết luận chính",
        "candidate": "ứng viên",
        "target-first-before-adverse": "đạt mục tiêu trước kéo ngược",
        "target-first": "đạt trước kéo ngược",
        "target-hit": "tỷ lệ đạt mục tiêu",
        "Target hit": "Tỷ lệ đạt mục tiêu",
        "breakdown risk": "rủi ro phá vỡ xuống",
        "breakdown": "phá vỡ xuống",
        "low-liquidity": "thanh khoản thấp",
        "aggregate": "toàn mẫu",
        "bull regime": "bối cảnh thị trường tăng",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out
