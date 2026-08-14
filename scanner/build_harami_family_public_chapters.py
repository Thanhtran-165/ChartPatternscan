"""Build source-grounded Harami Family public-chapter seed artifacts.

Chapter 64 (edition 2). Nguyên liệu deterministic theo chuẩn canonical:
- scanner: `scanner/v2/harami.py` (variant-aware, EC ch.43-46)
- nguồn sách: Encyclopedia of Candlestick ch.43-46 (offset PDF +24),
  trích trong `docs/project/pdf_review/m5/family_harami_20260813.md`
  và `scanner/v2/measurement_registry.py` (`_PDF_EXTRAS["harami"]`).

Builder chỉ cung cấp nguyên liệu; prose final phải qua
`canonical_source_guided_refinement_v1`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.build_horn_family_public_chapters import (  # noqa: E402
    _enrich_events_for_target,
    _events_for_scope,
    _fmt,
    _metric_for_target,
    _select_examples,
)
from scanner.harami_family_publication_specs import build_harami_publication_spec  # noqa: E402
from scanner.publication_flow_contract import SOURCE_GROUNDED_PUBLICATION_GATE_ID  # noqa: E402


DEFAULT_OUT_DIR = Path("artifacts/scanner_v2/harami_family_public_chapters")
SOURCE_PDF = "references/Thomas N. Bulkowski - Encyclopedia of Candlestick.pdf"

PATTERNS: dict[str, dict[str, Any]] = {
    "harami": {
        "slug": "harami",
        "title": "Harami",
        "subtitle": "Nến mẹ dài, nến con nằm trong thân nến mẹ, chờ đóng cửa xác nhận",
        "scan_dir": Path("artifacts/scanner_v2/harami_family/harami/db_active"),
        "source_chapter": "Harami (EC ch.43-46)",
        "source_name": "Harami, Bearish / Bullish / Harami Cross Bearish / Bullish",
        "scope_tier": "premium+standard",
        "classification": "hồ sơ tham khảo hai hướng trong phạm vi dữ liệu hiện có",
        "claim_level": "đọc như mẫu hai nến đảo chiều: nến con nằm trong nến mẹ, sau đó chờ đóng cửa vượt biên trong một trong hai hướng",
        "public_classification_sentence": "Trong phạm vi dữ liệu hiện có, Harami là chương hai nến rất phổ biến: đảo chiều chỉ nhỉnh hơn ngẫu nhiên, nên giá trị chính là đọc nén thân nến cùng xác nhận đóng cửa, không phải tín hiệu tự động.",
        "morphology": "Harami gồm một nến mẹ dài và một nến con nhỏ: thân nến con nằm gọn trong thân nến mẹ, bỏ qua bóng nến; biến thể harami cross dùng nến con doji nên đo theo biên độ cao-thấp của hai nến. Mẫu chỉ có hiệu lực khi giá đóng cửa vượt lên trên đỉnh phạm vi hai nến hoặc đóng cửa xuống dưới đáy phạm vi đó; trước thời điểm đó, nó chỉ là trạng thái nén thân nến.",
        "role_note": "Dùng như hồ sơ nén hai nến sau xác nhận; không đoán hướng trước khi có đóng cửa vượt biên.",
        "base_target_multiple": 0.5,
        "legacy_target_multiple": 1.0,
    },
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _plot_schematic(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=180)
    x = np.array([0, 1, 2, 3])
    opens = np.array([15.6, 17.2, 17.0, 18.3])
    closes = np.array([17.9, 17.9, 17.4, 18.9])
    lows = np.array([15.3, 16.9, 16.6, 17.9])
    highs = np.array([18.2, 18.1, 17.6, 19.2])
    labels = ["nến mẹ", "nến con", "xác nhận lên", "đi sau xác nhận"]
    for i, (o, c, lo, hi) in enumerate(zip(opens, closes, lows, highs)):
        color = "#1b8a5a" if c >= o else "#c44e52"
        ax.vlines(x[i], lo, hi, color="#222222", linewidth=2.0)
        ax.add_patch(Rectangle((x[i] - 0.16, min(o, c)), 0.32, max(abs(c - o), 0.06), facecolor=color, edgecolor=color, alpha=0.9))
        if labels[i]:
            ax.text(x[i], hi + 0.18, labels[i], ha="center", fontsize=8, color="#245b5a")
    mother_top, mother_bottom = 17.9, 15.6
    ax.axhline(mother_top, color="#7A5195", linestyle="--", linewidth=1.0)
    ax.axhline(mother_bottom, color="#7A5195", linestyle="--", linewidth=1.0)
    ax.text(0.08, mother_top + 0.08, "đỉnh thân nến mẹ", fontsize=8, color="#7A5195")
    ax.text(0.08, mother_bottom - 0.30, "đáy thân nến mẹ", fontsize=8, color="#7A5195")
    ax.axhspan(mother_bottom, mother_top, xmin=0.10, xmax=0.42, color="#6baed6", alpha=0.14)
    ax.set_title("Giải phẫu Harami: thân nến con trong thân nến mẹ", loc="left", fontsize=10)
    ax.set_ylim(min(lows) - 0.65, max(highs) + 0.75)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _source_notes(pattern_id: str, meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "PASS",
        "source_grounding_policy_id": SOURCE_GROUNDED_PUBLICATION_GATE_ID,
        "source_grounding_level": "publication_aligned",
        "local_source": {"pattern_key": pattern_id, "chapter": meta["source_chapter"], "name": meta["source_name"]},
        "direct_pdf_review": {
            "status": "PASS",
            "review_id": "harami_source_extraction_review_v1",
            "pdf_path": SOURCE_PDF,
            "book_chapter": meta["source_chapter"],
            "book_pages_checked": [374, 382, 383, 391, 392, 400, 408],
            "pdf_pages_checked": [398, 406, 407, 415, 416, 424, 432],
            "pdf_offset_note": "số trang in + 24 = số trang PDF (kiểm chứng p398 = in p374, mở ch.43)",
            "target_rule_summary": (
                "EC ch.43-46: target = giá phá vỡ ± (chiều cao mẫu × multiplier theo chương). "
                "Multipliers cột bull market: bearish harami 63% (lên) / 64% (xuống); bullish harami 69%/59%; "
                "cross bearish 69%/68%; cross bullish 74%/68%. Sách không công bố failure rate cho candlestick."
            ),
            "review_note": (
                "Đối chiếu từ docs/project/pdf_review/m5/family_harami_20260813.md (tự trích pdftotext, offset +24) "
                "và scanner/v2/measurement_registry.py _PDF_EXTRAS['harami']."
            ),
        },
        "source_rules": [
            {"rule_id": "harami.body_containment", "short_excerpt": "The small black candle on the second day must have a body that fits inside the body of the white candle", "implementation_mapping": "harami thường: thân nến con nằm trong thân nến mẹ (cho phép một mép bằng, cấm cả hai mép bằng)"},
            {"rule_id": "harami.cross_range_containment", "short_excerpt": "A doji with a trading range inside the price range of the prior day", "implementation_mapping": "harami cross (nến con doji): đo theo biên độ cao-thấp của hai nến thay vì thân"},
            {"rule_id": "harami.tops_or_bottoms_equal", "short_excerpt": "Either the tops or the bottoms of the bodies can be equal but not both", "implementation_mapping": "một mép thân được bằng, hai mép cùng bằng thì loại"},
            {"rule_id": "harami.candle_color", "short_excerpt": "Some ignore the candle color, but I don't", "implementation_mapping": "màu nến theo hướng đảo chiều của từng biến thể; tổ hợp màu khác không tính"},
            {"rule_id": "harami.breakout_close", "short_excerpt": "price closes above the two-candle high or below the two-candle low", "implementation_mapping": "xác nhận chỉ tính khi đóng cửa vượt đỉnh hoặc thủng đáy phạm vi hai nến"},
            {"rule_id": "harami.measure_rule_multiplier", "short_excerpt": "Compute the height of the candle pattern and multiply it by the appropriate percentage shown in the table; then apply it to the breakout price", "implementation_mapping": "mục tiêu = giá phá vỡ ± (chiều cao mẫu × 58-74% tùy chương và hướng)"},
            {"rule_id": "harami.no_failure_rate", "short_excerpt": "EC không công bố break-even failure rate cho candlestick", "implementation_mapping": "thất bại 5% là ngưỡng nội bộ của pipeline Việt Nam, không phải số sách"},
        ],
    }


def _spec(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "title": meta["title"],
        "subtitle": meta["subtitle"],
        "base_target_multiple": float(meta["base_target_multiple"]),
        "base_target_label": "0,5x",
        "legacy_target_multiple": float(meta["legacy_target_multiple"]),
        "legacy_target_label": "1,0x",
        "target_unit_label": "khoảng mục tiêu theo multiplier chương sách",
        "target_focus_title": "Mốc cơ sở 0,5x",
        "target_focus_caption": "mốc 0,5x khoảng mục tiêu sách",
        "target_focus_reading": "mốc thận trọng cho mẫu hai nến",
        "target_full_title": "Mốc đầy đủ 1,0x",
        "target_full_reading": "mốc bằng khoảng mục tiêu theo multiplier chương sách",
        "morphology_sentence": meta["morphology"],
        "role_note": meta["role_note"],
        "classification_sentence": meta["public_classification_sentence"],
        "headline_scope": "Harami là mẫu hai nến rất phổ biến trên dữ liệu Việt Nam; chương đọc nó như tín hiệu nén thân nến kèm xác nhận đóng cửa, không phải tín hiệu đảo chiều tự động.",
        "local_source_chapter": meta["source_chapter"],
        "schematic_caption": "Sơ đồ minh họa Harami: nến mẹ dài, thân nến con nằm gọn trong thân nến mẹ, rồi đóng cửa xác nhận vượt biên phạm vi hai nến.",
        "how_subtitle": "Nến mẹ dài, nến con nằm trong, đóng cửa xác nhận.",
        "labels": {"favorable_move": "mức đi thuận lợi", "adverse_move": "mức kéo ngược bất lợi"},
        "source_rule_ids": ["harami.body_containment", "harami.cross_range_containment", "harami.tops_or_bottoms_equal", "harami.candle_color", "harami.breakout_close", "harami.measure_rule_multiplier", "harami.no_failure_rate"],
        "public_rule_rows": [
            ["Thân nến con phải nằm trong thân nến mẹ.", "Harami thường so thân nến (bỏ qua bóng); đây là điểm khác Inside Day vốn so biên độ cao-thấp."],
            ["Một mép thân được bằng, nhưng không phải cả hai.", "Nếu cả đỉnh và đáy thân trùng nhau, nến con không còn nhỏ hơn và mẫu bị loại."],
            ["Harami cross: nến con doji thì đo theo biên độ.", "Doji có thân gần bằng 0 nên sách chuyển sang so biên độ cao-thấp của hai nến."],
            ["Màu nến theo hướng đảo chiều của biến thể.", "Sách công nhận màu nến mẹ và nến con ngược chiều nhau; không gộp tổ hợp màu khác."],
            ["Xác nhận bằng đóng cửa vượt biên phạm vi hai nến.", "Trước khi có đóng cửa vượt đỉnh hoặc thủng đáy, mẫu chỉ là trạng thái nén."],
            ["Mục tiêu theo multiplier chương sách.", "Mốc = giá phá vỡ ± (chiều cao mẫu × 58-74% tùy chương và hướng), không dùng nguyên chiều cao."],
            ["Thất bại 5% là ngưỡng nội bộ.", "Sách candlestick không công bố failure rate; ngưỡng 5% chỉ để so sánh nội bộ các chương."],
        ],
        "quick_question_rows": [
            ["Bối cảnh", "Có nến mẹ dài nổi bật trước đó không?"],
            ["Thân mẫu", "Thân nến con có nằm trong thân nến mẹ (hoặc biên độ, nếu doji) không?"],
            ["Xác nhận", "Giá đã đóng cửa vượt đỉnh hoặc thủng đáy phạm vi hai nến chưa?"],
            ["Đường đi sau đó", "Mốc 0,5x khoảng mục tiêu sách có đến trước kéo ngược 5% không?"],
        ],
        "component_rows": [
            ["Nến mẹ", "Nến dài tạo khung giá cho mẫu.", "Định khung mục tiêu"],
            ["Nến con", "Nến nhỏ nằm trong (thân hoặc biên độ nếu doji).", "Tín hiệu nén"],
            ["Xác nhận", "Đóng cửa vượt đỉnh hoặc đáy phạm vi hai nến.", "Kích hoạt hai hướng"],
            ["Đường đi sau đó", "Đo mức đi thuận lợi, kéo ngược và thời gian đạt mục tiêu.", "Kiểm chứng sau xác nhận"],
        ],
        "reject_bullets": [
            "Thân nến con chỉ nằm trong biên độ nhưng tràn thân nến mẹ (và không phải doji cross).",
            "Cả đỉnh và đáy thân hai nến trùng nhau.",
            "Màu nến không theo hướng đảo chiều của biến thể.",
            "Chưa có đóng cửa vượt biên phạm vi hai nến.",
            "Không đủ dữ liệu hậu phá vỡ để đo đường đi sau xác nhận.",
        ],
        "identification_paragraphs": [meta["morphology"]],
        "example_intro": ["Ba ví dụ dưới đây minh họa Harami như một case study hai nến: một mẫu đạt mốc cơ sở, một mẫu gần trung vị và một mẫu thất bại. Điểm cần nhìn là nến mẹ, nến con, đóng cửa xác nhận và đường đi sau đó."],
        "failure_bullets": [
            "Thất bại 5% là mẫu không đi đủ xa sau xác nhận, không phải stop-loss thực chiến.",
            "Mẫu hai nến dễ nhiễu: đảo chiều của Harami chỉ nhỉnh hơn ngẫu nhiên trong số liệu sách.",
            "Tỷ lệ đạt mục tiêu cần đọc cùng target-first-before-adverse vì giá có thể chạm mốc sau khi đã kéo ngược sâu.",
        ],
        "target_paragraph": "Mục tiêu của Harami lấy chiều cao mẫu hai nến nhân với multiplier chương sách (58-74% tùy biến thể và hướng) rồi chiếu từ giá phá vỡ; chương giữ 0,5x khoảng mục tiêu đó làm mốc cơ sở thận trọng và 1,0x làm mốc nguồn đầy đủ.",
        "measure_rule_variant_notes": [
            "Sách gốc (EC ch.43-46) ghi multiplier theo từng chương: harami giảm 63/64%, harami tăng 69/59%, cross giảm 69/68%, cross tăng 74/68% (cột thị trường tăng). Bản in này dùng đúng bộ multiplier đó theo từng biến thể và hướng phá vỡ; sách còn cặp số thị trường giảm để đối chiếu thêm.",
        ],
        "quick_conclusion_rows": [
            ["Mẫu này dùng để đọc gì?", "Tín hiệu nén hai nến kèm xác nhận đóng cửa; đảo chiều chỉ nhỉnh hơn ngẫu nhiên."],
            ["Mốc đọc chính?", "0,5x khoảng mục tiêu theo multiplier chương sách."],
            ["Mốc tham chiếu?", "1,0x khoảng mục tiêu theo multiplier chương sách."],
            ["Khi nào thận trọng?", "Khi nến mẹ không nổi bật, nến con tràn thân, hoặc kéo ngược đến trước khi chạm mốc."],
        ],
        "identification_bridge": (
            "Các quy tắc nhận diện nên được đọc theo đúng thứ tự: trước hết phải có nến mẹ dài, sau đó kiểm tra thân nến con nằm gọn trong thân nến mẹ "
            "(hoặc biên độ nếu là doji cross), rồi mới chờ đóng cửa vượt biên phạm vi hai nến. Nếu đảo thứ tự này, người đọc rất dễ gọi một cặp nến chồng lấn bất kỳ là Harami."
        ),
        "caveat_bullets": [
            "Không tuyên bố đây là nghiên cứu toàn thị trường đúng từng ngày lịch sử.",
            "Không dùng historical VN30/VN100 membership làm kết luận chính.",
            "Corporate actions và delisted/halted hiện là kiểm tra thay thế, chưa phải status tape chính thức.",
            "Chương là tài liệu tham khảo hậu xác nhận, không phải khuyến nghị mua bán.",
        ],
        "quantile_specs": [
            ("Chiều cao mẫu", "pattern_height_pct", "%"),
            ("Tỷ lệ thân nến con/mẹ", "body_ratio", "lần"),
            ("Thân nến mẹ", "mother_body_pct", "%"),
            ("Thân nến con", "child_body_pct", "%"),
            ("Thời gian xác nhận", "breakout_lag_bars", "phiên"),
            ("Mức đi thuận lợi", "mfe_pct", "%"),
            ("Mức đi ngược bất lợi", "mae_pct", "%"),
            ("Ngày chạm mốc cơ sở", "days_to_target", "phiên"),
        ],
        "skip_condition_specs": [
            ("Xác nhận quá trễ", "breakout_lag_bars", "q75", None, "Mẫu hai nến mất độ sắc nếu giá chờ quá lâu mới xác nhận."),
            ("Nến con quá lớn", "body_ratio", "q75", None, "Thân nến con chiếm gần hết thân nến mẹ thì tín hiệu nén yếu đi."),
            ("Nến mẹ quá mỏng", "mother_body_pct", "q25", None, "Nến mẹ không đủ dài thì khung mục tiêu không còn đáng tin."),
            ("Kéo ngược quá sâu", "mae_pct", "q75", None, "Đường đi sau xác nhận không còn gọn."),
        ],
        "general_stat_specs": [
            ("Chiều cao mẫu", "pattern_height_pct", "%", "Chiều cao hai nến là gốc của khung mục tiêu."),
            ("Tỷ lệ thân nến con/mẹ", "body_ratio", "lần", "Càng nhỏ càng giống harami nguồn."),
            ("Thân nến mẹ", "mother_body_pct", "%", "Nến mẹ càng dài, khung mẫu càng rõ."),
            ("Thời gian xác nhận", "breakout_lag_bars", "phiên", "Harami xác nhận nhanh; chờ lâu là dấu hiệu yếu."),
        ],
        "best_condition_specs": [
            ("Nhóm hình thái tốt", "publication_quality_tier", "==", "premium", "Hai nến rõ ràng, đường giá sạch."),
            ("Nhóm chuẩn", "publication_quality_tier", "==", "standard", "Đủ dùng trong thống kê nhưng không phải lúc nào cũng đẹp để minh họa."),
            ("Đường giá sạch", "path_quality_bucket", "==", "clean", "Ít thiếu phiên và ít chuỗi đứng giá."),
            ("Thanh khoản tốt hơn", "liquidity_bucket", "in", "mid/high", "Giảm nhiễu ở các cặp nến kém giao dịch."),
        ],
        "conclusion_bullets": [
            "Harami chỉ được đọc như mẫu hai nến: nén thân nến kèm xác nhận đóng cửa, đúng với EC ch.43-46.",
            "Mục tiêu nguồn là khoảng mục tiêu theo multiplier chương sách; chương dùng 0,5x làm mốc cơ sở thận trọng.",
            meta["role_note"],
        ],
    }


def _publication_payload(pattern_id: str, meta: Mapping[str, Any], events: pd.DataFrame, all_events: pd.DataFrame, path_df: pd.DataFrame) -> dict[str, Any]:
    base = _metric_for_target(events, path_df, 0.5, "conservative_half_measure")
    full = _metric_for_target(events, path_df, 1.0, "source_measure_rule")
    variants = events["variant"].value_counts().to_dict() if "variant" in events.columns else {}
    return {
        "publication_id": f"{pattern_id}_publication_chapter_v1",
        "pattern_id": pattern_id,
        "pattern_name": meta["title"],
        "status": "PASS",
        "classification": meta["classification"],
        "chapter_reference": {
            "scope": "nhóm hình thái tốt + nhóm chuẩn",
            "all_scanner_events": int(len(all_events)),
            "public_grade_events": int(len(events)),
            "public_grade_share_pct": round(float(len(events)) / max(len(all_events), 1) * 100.0, 2),
            "events": int(len(events)),
            "symbols_scanned": int(all_events["symbol"].nunique()) if "symbol" in all_events.columns else None,
            "evaluated_events": int(events["mfe_pct"].notna().sum()) if "mfe_pct" in events.columns else int(len(events)),
            "median_mfe_pct": base.get("median_mfe_pct"),
            "median_mae_pct": base.get("median_mae_pct"),
            "mfe_mae_median_ratio": base.get("mfe_mae_median_ratio"),
            "failure_5pct_rate": base.get("failure_5pct_rate"),
            "legacy_target_hit_rate": full.get("target_hit_rate"),
            "legacy_target_first_before_adverse_5pct_rate": full.get("target_first_before_adverse_5pct_rate"),
            "median_body_ratio": _fmt(pd.to_numeric(events.get("body_ratio"), errors="coerce").median()),
            "median_mother_body_pct": _fmt(pd.to_numeric(events.get("mother_body_pct"), errors="coerce").median()),
            "variant_distribution": variants,
        },
        "target_calibration": {
            "target_family": {"conservative_half_measure": 0.5, "source_measure_rule": 1.0},
            "selected_base_target_multiple": 0.5,
            "selected_base_target_role": "conservative_half_measure",
            "base_target": base,
            "stretch_target": full,
            "legacy_target": full,
            "rows": [base, full],
            "interpretation": "Mốc 0,5x giữ vai trò cơ sở thận trọng cho mẫu hai nến; 1,0x là khoảng mục tiêu theo multiplier chương sách.",
        },
        "data_scope_and_caveats": {
            "remaining_caveats": [
                "Không claim point-in-time universe toàn thị trường.",
                "Không dùng historical VN30/VN100 membership làm kết luận chính.",
                "Corporate actions và delisted/halted hiện là kiểm tra thay thế, chưa phải status tape chính thức.",
            ]
        },
    }


def build_one_harami_chapter(*, pattern_id: str, out_dir: Path) -> dict[str, Path]:
    meta = PATTERNS[pattern_id]
    chapter_dir = out_dir / str(meta["slug"])
    if chapter_dir.exists():
        shutil.rmtree(chapter_dir)
    chapter_dir.mkdir(parents=True, exist_ok=True)
    all_events = pd.read_csv(meta["scan_dir"] / "events.csv")
    if "event_id" not in all_events.columns:
        all_events["event_id"] = all_events["detection_id"]
    path_df = pd.read_csv(meta["scan_dir"] / "post_breakout_path.csv")
    events = _events_for_scope(all_events, str(meta["scope_tier"]))
    events = _enrich_events_for_target(events, path_df, float(meta["base_target_multiple"]))
    payload = _publication_payload(pattern_id, meta, events, all_events, path_df)
    spec = _spec(meta)
    publication_spec = build_harami_publication_spec(pattern_id=pattern_id, title=str(meta["title"]), spec=spec)
    payload["publication_spec_id"] = publication_spec["publication_spec_id"]
    payload["source_rules_public"] = [{"rule": row[0], "application": row[1]} for row in spec.get("public_rule_rows", [])]
    selected_examples = _select_examples(events)
    payload["example_events"] = {role: {**event.to_dict(), "example_role": role} for role, event in selected_examples.items()}
    charts_dir = chapter_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    schematic = charts_dir / "harami_schematic.png"
    _plot_schematic(schematic)
    source_notes = _source_notes(pattern_id, meta)
    payload_path = chapter_dir / f"{meta['slug']}_public_chapter_payload.json"
    source_notes_path = chapter_dir / f"{meta['slug']}_source_notes.json"
    publication_spec_path = chapter_dir / f"{meta['slug']}_publication_spec.json"
    _write_json(payload_path, payload)
    _write_json(source_notes_path, source_notes)
    _write_json(publication_spec_path, publication_spec)
    style_dossier = chapter_dir / "source_style_dossier.md"
    style_dossier.write_text(
        "# Source-Guided Style Dossier - harami\n\n"
        "Harami là mẫu hai nến: thân nến con nằm trong thân nến mẹ (cross: doji đo theo biên độ), rồi chỉ có hiệu lực khi đóng cửa vượt biên phạm vi hai nến. "
        "Dossier giữ thứ tự đọc: nến mẹ, nến con, đóng cửa xác nhận, thất bại 5%, mục tiêu theo multiplier chương sách và cách dùng thận trọng. "
        "Số liệu sách (EC ch.43-46): multipliers 63/64, 69/59, 69/68, 74/68 (cột bull market), mỗi chương giới hạn 20.000 mẫu; sách không công bố failure rate. "
        "Không sao chép hoặc dịch lại tài liệu gốc; số liệu Việt Nam lấy từ payload đã khóa.\n",
        encoding="utf-8",
    )
    entry = {
        "family": "harami_family",
        "pattern_id": pattern_id,
        "title": meta["title"],
        "status": "source_seed",
        "classification": meta["classification"],
        "score": None,
        "claim_level": meta["claim_level"],
        "pdf": f"artifacts/final_chapters/harami_family/{meta['slug']}_final.pdf",
        "source_pdf": f"artifacts/final_chapters/harami_family/{meta['slug']}_final.pdf",
        "payload": str(payload_path),
        "source_notes": str(source_notes_path),
        "publication_spec": str(publication_spec_path),
        "source_grounding_required": True,
        "source_grounding_policy_id": SOURCE_GROUNDED_PUBLICATION_GATE_ID,
        "direct_source_review_required": True,
        "publication_semantic_required": True,
        "publication_semantic_gate_id": publication_spec["semantic_gate_id"],
        "canonical_rebuild_required": True,
        "chapter_writing_stages": {"source_style_dossier": str(style_dossier)},
        "chapter_writing_notes": "Seed artifact only. Final public prose must be generated by source-guided AI refinement and canonical publication factory.",
        "note": "Harami Family dùng scanner hai nến body-aware riêng (EC ch.43-46); builder này chỉ cung cấp nguyên liệu, không render hoặc approve PDF final.",
    }
    entry_path = chapter_dir / f"{meta['slug']}_final_manifest_entry.json"
    _write_json(entry_path, entry)
    return {"payload": payload_path, "source_notes": source_notes_path, "publication_spec": publication_spec_path, "entry": entry_path, "chart_schematic": schematic}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Harami Family public-chapter seed artifacts.")
    parser.add_argument("--pattern", choices=sorted(PATTERNS), default="harami")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    result = build_one_harami_chapter(pattern_id=str(args.pattern), out_dir=Path(args.out_dir))
    print(json.dumps({key: str(value) for key, value in result.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
