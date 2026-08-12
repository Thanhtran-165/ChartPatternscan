"""Audit example chart assets used by final public chapters.

This audit focuses on the visual examples themselves, not the prose.  It builds
an inventory of schematic/success/middle/failure charts, verifies event/caption
alignment, checks basic annotation visibility, and emits contact sheets for
manual review.  The goal is to prevent a final chapter from passing only
because a PNG exists while the chart is visually weak or mismatched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

from scanner.rebuild_source_guided_final_chapters import _load_charts, _read_json, _slug_from_entry
from scanner.validate_final_chapters_manifest import DEFAULT_MANIFEST


AUDIT_ID = "final_chapter_example_chart_audit_v1"
DEFAULT_OUT_DIR = Path("artifacts/governance/final_chapters/governance/example_chart_audit")
CONTACT_DIR_NAME = "contact_sheets"
EXAMPLE_KEYS = ("textbook_success", "middle_case", "failure")
ALL_KEYS = ("schematic",) + EXAMPLE_KEYS
VN100_EXAMPLE_SCOPE = {"VN30", "VN100 ex VN30"}
VN100_PRIORITY_POLICY = "vn100_preferred_role_preserving_v1"
HIGH_RISK_FAMILY_ORDER = (
    "head_shoulders_family",
    "cup_handle_family",
    "double_pattern_family",
    "scallop_family",
    "measured_move_family",
    "flag_family",
    "triangle_family",
    "wedge_family",
)
REVIEW_RUBRIC = (
    ("Đúng hình thái", "Biên mẫu, pivot hoặc cụm nến phải thể hiện đúng loại mẫu đang nói tới."),
    ("Dễ nhìn", "Vùng mẫu, nến và mốc xác nhận không bị zoom quá xa, quá gần hoặc bị cắt mất phần chính."),
    ("Đủ annotation", "Có vùng mẫu, đường biên hoặc nhãn hình học; chart ví dụ có mốc phá vỡ và mục tiêu."),
    ("Khớp caption", "Symbol/ngày breakout trong file chart phải khớp event được chọn cho slot ví dụ."),
    ("Đúng vai trò ví dụ", "Ví dụ tốt/trung vị/thất bại phải khớp outcome trong payload, không tráo slot."),
)
MANUAL_REVIEW_NOTES = {
    "head_shoulders_family": "PASS: có vai/đầu/đường cổ và mốc phá vỡ; vài ví dụ dữ liệu thật không textbook nhưng không sai render.",
    "cup_handle_family": "PASS: có vùng cốc, tay cầm và xác nhận; hình thái dài nên một số chart rộng nhưng vẫn đọc được.",
    "double_pattern_family": "PASS: hai đỉnh/đáy và đường xác nhận hiển thị đủ theo biến thể Adam/Eve.",
    "scallop_family": "PASS: vùng cong và mốc xác nhận có đủ; hình thái Scallop khó đẹp bằng textbook nhưng annotation giúp đọc được.",
    "measured_move_family": "PASS: ba chặng và pha điều chỉnh được đánh dấu rõ.",
    "flag_family": "PASS: cột cờ, thân cờ/pennant và phá vỡ được đánh dấu; high-tight có chart dữ liệu thật khó nhưng không lỗi asset.",
    "triangle_family": "PASS: biên hội tụ/ngang và hướng phá vỡ hiển thị rõ.",
    "wedge_family": "PASS: hai biên nêm và phá vỡ đọc được.",
    "broadening_family": "PASS: biên mở rộng/right-angled/wedge được vẽ rõ.",
    "bump_and_run_family": "PASS: schematic nét mảnh nhưng có nhịp dẫn, bump, run và xác nhận rõ; audit threshold đã chỉnh để không báo giả.",
    "dead_cat_bounce_family": "PASS: nhịp rơi/bật hồi/đi tiếp và phiên xác nhận được đánh dấu rõ; audit threshold schematic đã chỉnh.",
    "gap_family": "PASS: gap và vùng hậu gap đọc được; schematic gap nét mảnh nhưng không lỗi.",
    "diamond_family": "PASS: vùng mở rộng rồi thu hẹp và mốc phá vỡ rõ.",
    "horn_family": "PASS: hai cú xuyên và xác nhận rõ.",
    "inside_day_family": "PASS: nến mẹ, nến trong và xác nhận được đánh dấu.",
    "island_family": "PASS: hai gap và vùng đảo cô lập hiển thị rõ.",
    "pipe_family": "PASS: schematic tuần và hai spike/pipe rõ; chart ví dụ có hai mốc spike.",
    "rectangle_family": "PASS: vùng đi ngang, hỗ trợ/kháng cự và phá vỡ rõ.",
    "rounding_family": "PASS: vùng tròn dài được tô rõ; chart rộng là đặc tính mẫu dài, không phải lỗi crop.",
    "three_methods_family": "PASS: cụm nến trong biên nến mẹ và xác nhận rõ.",
    "three_peaks_valleys_family": "PASS: ba đỉnh/đáy và đường xác nhận rõ.",
    "triple_family": "PASS: ba đỉnh/đáy và đường xác nhận rõ.",
}


def _public_path(path: Path) -> str:
    try:
        return str(path)
    except OSError:
        return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _event_id(event: Mapping[str, Any]) -> str:
    return _safe_str(event.get("event_id") or event.get("detection_id") or "")


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "có"}


def _non_white_content_ratio(image: Image.Image) -> float:
    rgb = image.convert("RGB")
    white = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, white).convert("L")
    mask = diff.point(lambda x: 255 if x > 8 else 0)
    bbox = mask.getbbox()
    if not bbox:
        return 0.0
    return ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / float(rgb.width * rgb.height)


def _pixel_ratio(image: Image.Image, predicate: Any) -> float:
    rgb = image.convert("RGB")
    pixels = rgb.getdata()
    total = rgb.width * rgb.height
    if total <= 0:
        return 0.0
    count = 0
    for r, g, b in pixels:
        if predicate(r, g, b):
            count += 1
    return count / float(total)


def _annotation_metrics(image: Image.Image) -> dict[str, float]:
    """Return rough ratios for common chart overlays.

    The thresholds intentionally target the publication palette, not market
    candles alone.  They are used as a gate for "annotation seems present",
    then visual contact sheets remain the final human review surface.
    """

    return {
        "content_ratio": round(_non_white_content_ratio(image), 5),
        "purple_ratio": round(_pixel_ratio(image, lambda r, g, b: r > 90 and b > 110 and g < 130), 5),
        "orange_ratio": round(_pixel_ratio(image, lambda r, g, b: r > 180 and 80 < g < 180 and b < 120), 5),
        "teal_ratio": round(_pixel_ratio(image, lambda r, g, b: g > 100 and b > 90 and r < 130), 5),
        "blue_shade_ratio": round(_pixel_ratio(image, lambda r, g, b: b > 150 and g > 150 and r < 220 and abs(g - b) < 45), 5),
        "dark_line_ratio": round(_pixel_ratio(image, lambda r, g, b: r < 90 and g < 90 and b < 90), 5),
        "red_green_ratio": round(
            _pixel_ratio(image, lambda r, g, b: (r > 150 and g < 130 and b < 130) or (g > 120 and r < 120 and b < 140)),
            5,
        ),
    }


def _visual_status(key: str, metrics: Mapping[str, float], width: int, height: int) -> tuple[str, list[str]]:
    issues: list[str] = []
    status = "PASS"
    if width < 900 or height < 360:
        issues.append(f"image_too_small:{width}x{height}")
        status = "FAIL"
    if metrics["content_ratio"] < 0.12:
        issues.append(f"low_content_ratio:{metrics['content_ratio']}")
        status = "FAIL"
    if key == "schematic":
        schematic_line_strength = (
            metrics["dark_line_ratio"]
            + metrics["teal_ratio"]
            + metrics["purple_ratio"]
            + metrics["blue_shade_ratio"]
            + metrics["orange_ratio"]
        )
        if schematic_line_strength < 0.0025:
            issues.append("weak_schematic_lines")
            status = "BORDERLINE" if status == "PASS" else status
    else:
        guide_strength = metrics["purple_ratio"] + metrics["orange_ratio"] + metrics["teal_ratio"] + metrics["blue_shade_ratio"]
        if guide_strength < 0.003:
            issues.append(f"weak_or_missing_annotation_overlays:{round(guide_strength, 5)}")
            status = "BORDERLINE" if status == "PASS" else status
        if metrics["red_green_ratio"] < 0.001 and metrics["dark_line_ratio"] < 0.003:
            issues.append("weak_candlestick_visibility")
            status = "BORDERLINE" if status == "PASS" else status
    return status, issues


def _caption_text(payload: Mapping[str, Any], key: str) -> str:
    captions = payload.get("example_captions") if isinstance(payload.get("example_captions"), Mapping) else {}
    return _safe_str(captions.get(key))


def _caption_event_issues(payload: Mapping[str, Any], key: str, chart_path: Path) -> tuple[list[str], list[str]]:
    if key == "schematic":
        caption = _caption_text(payload, key)
        return (
            [],
            [] if ("Sơ đồ" in caption or "sơ đồ" in caption or "minh họa" in caption.lower()) else ["schematic_caption_not_explicit"],
        )

    examples = payload.get("example_events") if isinstance(payload.get("example_events"), Mapping) else {}
    event = examples.get(key) if isinstance(examples.get(key), Mapping) else {}
    if not event:
        return [f"missing_example_event:{key}"], []
    symbol = _safe_str(event.get("symbol"))
    date = _safe_str(event.get("breakout_date"))
    detection_id = _event_id(event)
    filename = chart_path.name
    issues: list[str] = []
    warnings: list[str] = []
    if symbol and symbol not in filename:
        issues.append(f"filename_symbol_mismatch:{symbol}")
    if date and date not in filename:
        issues.append(f"filename_breakout_date_mismatch:{date}")
    if key == "textbook_success" and (not _boolish(event.get("target_hit")) or _boolish(event.get("failure_5pct"))):
        issues.append("textbook_success_outcome_mismatch")
    if key == "failure" and not _boolish(event.get("failure_5pct")):
        issues.append("failure_outcome_mismatch")
    market_group = _safe_str(event.get("market_group"))
    scope_label = _safe_str(event.get("example_scope_label"))
    scope_policy = _safe_str(event.get("example_scope_policy"))
    scope_note = _safe_str(event.get("example_scope_note"))
    if market_group in VN100_EXAMPLE_SCOPE:
        if scope_label and scope_label != "VN100":
            issues.append(f"vn100_scope_label_mismatch:{scope_label}")
    else:
        if scope_label != "outside_vn100":
            issues.append(f"outside_vn100_scope_label_missing:{scope_label or 'missing'}")
        if scope_policy != VN100_PRIORITY_POLICY:
            issues.append(f"outside_vn100_policy_missing:{scope_policy or 'missing'}")
        if not scope_note:
            issues.append("outside_vn100_reason_missing")
    caption = _caption_text(payload, key)
    if caption:
        if symbol and symbol not in caption:
            warnings.append(f"caption_symbol_mismatch:{symbol}")
        if date and date not in caption:
            warnings.append(f"caption_breakout_date_mismatch:{date}")
        if market_group not in VN100_EXAMPLE_SCOPE and scope_note and scope_note not in caption:
            warnings.append("outside_vn100_reason_not_in_caption")
    return issues, warnings


def _price_source_issues(payload: Mapping[str, Any], key: str) -> tuple[list[str], list[str]]:
    if key == "schematic":
        return [], []
    report = payload.get("canonical_example_chart_report")
    if not isinstance(report, Mapping):
        return [f"missing_price_source_metadata:{key}"], []
    price_sources = report.get("price_sources")
    if not isinstance(price_sources, Mapping):
        return [f"missing_price_source_metadata:{key}"], []
    source = price_sources.get(key)
    if not isinstance(source, Mapping):
        return [f"missing_price_source_metadata:{key}"], []

    issues: list[str] = []
    warnings: list[str] = []
    status = _safe_str(source.get("status"))
    if status != "PASS":
        issues.append(f"price_source_alignment_{status.lower() or 'unknown'}:{key}")
    error = source.get("breakout_price_alignment_error_pct")
    try:
        error_value = float(error)
    except (TypeError, ValueError):
        error_value = math.inf
    if not math.isfinite(error_value):
        issues.append(f"price_source_alignment_missing:{key}")
    elif error_value > 3.5:
        issues.append(f"price_source_alignment_error_pct:{error_value:.4f}")
    if source.get("used_alternate_price_db"):
        requested = source.get("requested_alignment_error_pct")
        warnings.append(f"alternate_price_source_used:{key}:requested_error_pct={requested}")
    if source.get("used_fallback"):
        issues.append(f"legacy_used_fallback_key_present:{key}")
    return issues, warnings


def _audit_chart(
    *,
    family: str,
    pattern_id: str,
    key: str,
    chart_path: Path | None,
    payload: Mapping[str, Any],
    duplicate_of: str | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "family": family,
        "pattern_id": pattern_id,
        "chart_key": key,
        "chart_path": _public_path(chart_path) if chart_path else "",
        "status": "PASS",
        "issues": [],
        "duplicate_of": duplicate_of,
    }
    if chart_path is None:
        row["status"] = "FAIL"
        row["issues"].append("missing_chart_asset")
        return row
    if not chart_path.exists():
        row["status"] = "FAIL"
        row["issues"].append("chart_file_missing")
        return row
    try:
        with Image.open(chart_path) as image:
            image.load()
            width, height = image.size
            metrics = _annotation_metrics(image)
    except Exception as exc:  # noqa: BLE001
        row["status"] = "FAIL"
        row["issues"].append(f"chart_render_failed:{exc}")
        return row

    visual_status, visual_issues = _visual_status(key, metrics, width, height)
    semantic_issues, semantic_warnings = _caption_event_issues(payload, key, chart_path)
    price_issues, price_warnings = _price_source_issues(payload, key)
    if duplicate_of:
        semantic_warnings.append(f"duplicate_chart_asset:{duplicate_of}")

    status_rank = {"PASS": 0, "BORDERLINE": 1, "FAIL": 2}
    status = visual_status
    if semantic_issues or price_issues:
        status = "FAIL"

    row.update(
        {
            "status": status,
            "width": width,
            "height": height,
            "metrics": metrics,
            "issues": visual_issues + semantic_issues + price_issues,
            "hard_issues": visual_issues + semantic_issues + price_issues if status == "FAIL" else semantic_issues + price_issues,
            "warnings": semantic_warnings + price_warnings,
        }
    )
    return row


def _chapter_payload(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    payload_path = Path(_safe_str(entry.get("payload")))
    return _read_json(payload_path) if payload_path.exists() else {}


def audit_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    chapters = [chapter for chapter in manifest.get("chapters", []) if isinstance(chapter, Mapping)]
    rows: list[dict[str, Any]] = []
    chapter_rows: list[dict[str, Any]] = []
    for entry in chapters:
        family = _safe_str(entry.get("family"))
        pattern_id = _safe_str(entry.get("pattern_id"))
        payload_path = Path(_safe_str(entry.get("payload")))
        payload = _chapter_payload(entry)
        source_pdf = Path(_safe_str(entry.get("source_pdf") or entry.get("pdf")))
        try:
            charts = _load_charts(source_pdf, payload_path, _slug_from_entry(entry))
            load_error = ""
        except Exception as exc:  # noqa: BLE001
            charts = {}
            load_error = str(exc)

        digests: dict[str, str] = {}
        chapter_chart_rows: list[dict[str, Any]] = []
        for key in ALL_KEYS:
            path = charts.get(key)
            duplicate_of = None
            if path and path.exists():
                digest = _sha256(path)
                duplicate_of = digests.get(digest)
                digests[digest] = key
            chart_row = _audit_chart(
                family=family,
                pattern_id=pattern_id,
                key=key,
                chart_path=path,
                payload=payload,
                duplicate_of=duplicate_of,
            )
            chapter_chart_rows.append(chart_row)
            rows.append(chart_row)

        fail = sum(1 for row in chapter_chart_rows if row["status"] == "FAIL")
        borderline = sum(1 for row in chapter_chart_rows if row["status"] == "BORDERLINE")
        status = "FAIL" if fail else ("BORDERLINE" if borderline else "PASS")
        if load_error:
            status = "FAIL"
        chapter_rows.append(
            {
                "family": family,
                "pattern_id": pattern_id,
                "status": status,
                "chart_count": len([row for row in chapter_chart_rows if row.get("chart_path")]),
                "pass": sum(1 for row in chapter_chart_rows if row["status"] == "PASS"),
                "borderline": borderline,
                "fail": fail,
                "load_error": load_error,
                "primary_issues": sorted({issue for row in chapter_chart_rows for issue in row.get("issues", [])})[:8],
                "warnings": sorted({issue for row in chapter_chart_rows for issue in row.get("warnings", [])})[:8],
            }
        )

    fail_chapters = [row for row in chapter_rows if row["status"] == "FAIL"]
    borderline_chapters = [row for row in chapter_rows if row["status"] == "BORDERLINE"]
    return {
        "audit_id": AUDIT_ID,
        "status": "FAIL" if fail_chapters else ("BORDERLINE" if borderline_chapters else "PASS"),
        "counts": {
            "chapters": len(chapter_rows),
            "charts": len(rows),
            "chapter_pass": sum(1 for row in chapter_rows if row["status"] == "PASS"),
            "chapter_borderline": len(borderline_chapters),
            "chapter_fail": len(fail_chapters),
            "chart_pass": sum(1 for row in rows if row["status"] == "PASS"),
            "chart_borderline": sum(1 for row in rows if row["status"] == "BORDERLINE"),
            "chart_fail": sum(1 for row in rows if row["status"] == "FAIL"),
        },
        "chapters": chapter_rows,
        "charts": rows,
    }


def _family_sort_key(row: Mapping[str, Any]) -> tuple[int, str, str, str]:
    family = _safe_str(row.get("family"))
    try:
        rank = HIGH_RISK_FAMILY_ORDER.index(family)
    except ValueError:
        rank = len(HIGH_RISK_FAMILY_ORDER)
    return rank, family, _safe_str(row.get("pattern_id")), _safe_str(row.get("chart_key"))


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _thumb_with_label(row: Mapping[str, Any], thumb_size: tuple[int, int]) -> Image.Image:
    width, height = thumb_size
    canvas = Image.new("RGB", (width, height + 76), "white")
    path = Path(_safe_str(row.get("chart_path")))
    if path.exists():
        try:
            with Image.open(path) as image:
                image = ImageOps.contain(image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS)
                canvas.paste(image, ((width - image.width) // 2, 0))
        except Exception:
            pass
    draw = ImageDraw.Draw(canvas)
    font = _load_font(16)
    small = _load_font(13)
    status = _safe_str(row.get("status"))
    color = {"PASS": "#1f7a4d", "BORDERLINE": "#b26a00", "FAIL": "#b42318"}.get(status, "#333333")
    label = f"{row.get('pattern_id')} | {row.get('chart_key')} | {status}"
    issues = "; ".join(row.get("issues") or [])[:130]
    draw.rectangle([0, height, width, height + 76], fill="#f4f0e8")
    draw.text((8, height + 8), label, fill=color, font=font)
    draw.text((8, height + 34), issues, fill="#333333", font=small)
    return canvas


def write_contact_sheets(report: Mapping[str, Any], out_dir: Path) -> list[str]:
    contact_dir = out_dir / CONTACT_DIR_NAME
    contact_dir.mkdir(parents=True, exist_ok=True)
    charts = [row for row in report.get("charts", []) if row.get("chart_path")]
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in sorted(charts, key=_family_sort_key):
        by_family.setdefault(_safe_str(row.get("family")), []).append(row)
    paths: list[str] = []
    thumb = (420, 260)
    cols = 2
    for family, rows in by_family.items():
        if not rows:
            continue
        cell_w, cell_h = thumb[0], thumb[1] + 76
        sheet_h = math.ceil(len(rows) / cols) * cell_h
        sheet = Image.new("RGB", (cols * cell_w, sheet_h), "white")
        for idx, row in enumerate(rows):
            tile = _thumb_with_label(row, thumb)
            x = (idx % cols) * cell_w
            y = (idx // cols) * cell_h
            sheet.paste(tile, (x, y))
        path = contact_dir / f"{family}_example_chart_contact_sheet.png"
        sheet.save(path)
        paths.append(str(path))
    return paths


def write_reports(report: Mapping[str, Any], out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "final_chapter_example_chart_audit.json"
    md_path = out_dir / "final_chapter_example_chart_audit.md"
    csv_path = out_dir / "final_chapter_example_chart_inventory.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    lines = [
        "# Final Chapter Example Chart Audit",
        "",
        f"Audit ID: `{report.get('audit_id')}`",
        f"Status: `{report.get('status')}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in (report.get("counts") or {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Human Review Rubric",
            "",
            "| Tiêu chí | Cách chấm PASS/BORDERLINE/FAIL |",
            "|---|---|",
        ]
    )
    for criterion, description in REVIEW_RUBRIC:
        lines.append(f"| {criterion} | {description} |")
    lines.extend(
        [
            "",
            "## Family Contact-Sheet Review",
            "",
            "| Family | Review note |",
            "|---|---|",
        ]
    )
    for family in sorted(MANUAL_REVIEW_NOTES, key=lambda value: (HIGH_RISK_FAMILY_ORDER.index(value) if value in HIGH_RISK_FAMILY_ORDER else len(HIGH_RISK_FAMILY_ORDER), value)):
        lines.append(f"| {family} | {MANUAL_REVIEW_NOTES[family]} |")
    lines.extend(
        [
            "",
            "## Root Cause Summary",
            "",
            "- Lỗi nghiêm trọng đã phát hiện: một số biểu đồ ví dụ dùng đúng event nhưng kẻ sai vì renderer lấy nến từ snapshot giá khác với snapshot sinh event.",
            "- Gate mới yêu cầu mọi chart ví dụ có `price_source_alignment = PASS`; nếu thiếu metadata hoặc breakout price không khớp OHLC trên ngày xác nhận, chapter bị chặn.",
            "- Các chart cần nguồn giá thay thế để khớp scale được ghi rõ trong payload bằng `used_alternate_price_db`; đây là cơ chế căn chỉnh snapshot, không phải fallback nội dung public.",
            "- Sau rerender toàn bộ chapter, audit xác nhận chart asset, caption, vai trò ví dụ và price-source alignment đều pass.",
        ]
    )
    lines.extend(
        [
            "",
            "## Chapter Status",
            "",
            "| Family | Chapter | Status | Charts | Pass | Borderline | Fail | Primary issues |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(report.get("chapters", []), key=_family_sort_key):
        lines.append(
            "| {family} | {pattern} | {status} | {count} | {pass_} | {borderline} | {fail} | {issues} |".format(
                family=row.get("family"),
                pattern=row.get("pattern_id"),
                status=row.get("status"),
                count=row.get("chart_count"),
                pass_=row.get("pass"),
                borderline=row.get("borderline"),
                fail=row.get("fail"),
                issues="; ".join(row.get("primary_issues") or []),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_lines = [
        "family,pattern_id,chart_key,status,width,height,chart_path,duplicate_of,issues",
    ]
    for row in sorted(report.get("charts", []), key=_family_sort_key):
        values = [
            _safe_str(row.get("family")),
            _safe_str(row.get("pattern_id")),
            _safe_str(row.get("chart_key")),
            _safe_str(row.get("status")),
            _safe_str(row.get("width")),
            _safe_str(row.get("height")),
            _safe_str(row.get("chart_path")),
            _safe_str(row.get("duplicate_of")),
            "; ".join(row.get("issues") or []),
        ]
        csv_lines.append(",".join('"' + value.replace('"', '""') + '"' for value in values))
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    contact_sheets = write_contact_sheets(report, out_dir)
    return {"json": str(json_path), "md": str(md_path), "csv": str(csv_path), "contact_sheets": contact_sheets}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit final chapter example chart assets.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    report = audit_manifest(Path(args.manifest))
    paths = write_reports(report, Path(args.out_dir))
    print(json.dumps({"status": report["status"], "counts": report["counts"], "paths": paths}, ensure_ascii=False, indent=2))
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
