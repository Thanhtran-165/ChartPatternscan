"""Build a PDF detail report for the realtime pattern scan email.

The email stays light; this PDF carries the chart-level review material.  The
report is deterministic and uses the same canonical chart renderer as the
publication pipeline so chart overlays stay aligned with scanner event data.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.canonical_example_charts import DEFAULT_PRICE_DB, render_canonical_example_chart  # noqa: E402
from scanner.rebuild_source_guided_final_chapters import EVENT_SOURCES  # noqa: E402


WORKFLOW_ID = "realtime_scan_pdf_report_v1"
DEFAULT_PDF_OUT = Path("artifacts/realtime_scan/latest/email/realtime_scan_detail.pdf")
DEFAULT_CHART_DIR = Path("artifacts/realtime_scan/latest/email/charts")
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
)


def _font_path() -> Path | None:
    return next((path for path in FONT_CANDIDATES if path.exists()), None)


def _register_fonts() -> tuple[str, str]:
    path = _font_path()
    if not path:
        return "Helvetica", "Helvetica-Bold"
    pdfmetrics.registerFont(TTFont("AtlasSans", str(path)))
    pdfmetrics.registerFont(TTFont("AtlasSansBold", str(path)))
    return "AtlasSans", "AtlasSansBold"


def _clean(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    return text if text else default


def _pct(value: Any, *, absolute: bool = False) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
        number = float(value)
        if absolute:
            number = abs(number)
        return f"{number:.2f}%"
    except (TypeError, ValueError):
        return _clean(value)


def _prob(value: Any) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
        number = float(value)
        if 0 <= number <= 1:
            number *= 100
        return f"{number:.0f}%"
    except (TypeError, ValueError):
        return _clean(value)


def _volume_label(value: Any) -> str:
    labels = {
        "strong": "mạnh",
        "healthy": "ổn",
        "weak": "yếu",
        "risky": "cần thận trọng",
        "unknown": "chưa rõ",
        "none": "không",
        "missing_volume": "thiếu dữ liệu khối lượng",
        "adverse_volume_spike": "giá yếu đi kèm volume lớn",
        "noisy_setup_volume": "volume trong mẫu hơi nhiễu",
        "thin_value": "giá trị giao dịch mỏng",
    }
    text = str(value or "").strip()
    return labels.get(text, text or "-")


def _date_label(value: Any) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return _clean(value)
    return dt.date().isoformat()


def _source_id(row: Mapping[str, Any]) -> str:
    return str(row.get("source_event_id") or row.get("event_id") or row.get("detection_id") or "")


def _event_source_for_pattern(pattern_id: str) -> Path | None:
    source = EVENT_SOURCES.get(pattern_id)
    if not source:
        return None
    return Path(source[0])


def _load_event_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    pattern_id = str(row.get("pattern_id") or "")
    source_path = _event_source_for_pattern(pattern_id)
    source_id = _source_id(row)
    if not source_path or not source_path.exists() or not source_id:
        return None
    try:
        events = pd.read_csv(source_path, low_memory=False)
    except Exception:
        return None
    candidates = []
    for column in ("event_id", "detection_id"):
        if column in events.columns:
            candidates.append(events.loc[events[column].astype(str).eq(source_id)])
    match = next((frame for frame in candidates if not frame.empty), None)
    if match is None or match.empty:
        return None
    event = match.iloc[0].to_dict()
    event.setdefault("pattern_id", pattern_id)
    return event


def _render_chart(row: Mapping[str, Any], chart_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    event = _load_event_row(row)
    if not event:
        return None, {"status": "FAIL", "reason": "missing_source_event", "source_event_id": _source_id(row)}
    pattern_id = str(row.get("pattern_id") or event.get("pattern_id") or "")
    symbol = _clean(row.get("symbol") or event.get("symbol"), "UNKNOWN")
    date = _date_label(row.get("event_date") or event.get("breakout_date"))
    chart_path = chart_dir / f"{symbol}_{pattern_id}_{date}.png"
    title = f"{symbol} - {_clean(row.get('pattern_label') or pattern_id)} ({date})"
    rendered, report = render_canonical_example_chart(
        price_db=DEFAULT_PRICE_DB,
        event=event,
        pattern_id=pattern_id,
        out_path=chart_path,
        title=title,
    )
    return (chart_path if rendered else None), {**report, "source_event_id": _source_id(row)}


def _styles() -> dict[str, ParagraphStyle]:
    regular, bold = _register_fonts()
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName=bold, fontSize=22, leading=28, textColor=colors.HexColor("#153f3a"), spaceAfter=8),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName=bold, fontSize=15, leading=19, textColor=colors.HexColor("#1f6b61"), spaceBefore=8, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName=bold, fontSize=12, leading=15, textColor=colors.HexColor("#153f3a"), spaceBefore=5, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName=regular, fontSize=9.5, leading=13.5, textColor=colors.HexColor("#253330")),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName=regular, fontSize=8.2, leading=11.5, textColor=colors.HexColor("#52605b")),
        "bold": ParagraphStyle("bold", parent=base["BodyText"], fontName=bold, fontSize=9.5, leading=13.5, textColor=colors.HexColor("#153f3a")),
    }


def _summary_table(summary: Mapping[str, Any], styles: Mapping[str, ParagraphStyle]) -> Table:
    counts = summary.get("counts", {})
    data = [
        [Paragraph("Nhóm", styles["bold"]), Paragraph("Số lượng", styles["bold"]), Paragraph("Cách đọc", styles["bold"])],
        [Paragraph("Setup đang hình thành", styles["body"]), Paragraph(str(counts.get("buy_setup", 0)), styles["body"]), Paragraph("Mẫu trong VN100/VN30 đang gần vùng xác nhận, chưa phải tín hiệu đã phá vỡ.", styles["small"])],
        [Paragraph("Ứng viên BUY tiềm năng", styles["body"]), Paragraph(str(counts.get("buy_candidates", 0)), styles["body"]), Paragraph("Mã VN100/VN30 đã qua bộ lọc BUY hiện tại.", styles["small"])],
        [Paragraph("Watchlist theo dõi thêm", styles["body"]), Paragraph(str(counts.get("watchlist", 0)), styles["body"]), Paragraph("Mã VN100/VN30 cần mở chart theo dõi thêm.", styles["small"])],
    ]
    table = Table(data, colWidths=[55 * mm, 28 * mm, 95 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eee8dc")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d8d0c4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _setup_fact_table(row: Mapping[str, Any], styles: Mapping[str, ParagraphStyle]) -> Table:
    data = [
        [Paragraph("Mẫu đang hình thành", styles["bold"]), Paragraph(_clean(row.get("pattern_label") or row.get("pattern_id")), styles["body"])],
        [Paragraph("Ngày quan sát", styles["bold"]), Paragraph(_date_label(row.get("latest_date")), styles["body"])],
        [Paragraph("Nhóm", styles["bold"]), Paragraph(_clean(row.get("market_group")), styles["body"])],
        [Paragraph("Giá hiện tại", styles["bold"]), Paragraph(_clean(row.get("last_close")), styles["body"])],
        [Paragraph("Điểm xác nhận", styles["bold"]), Paragraph(_clean(row.get("trigger_price")), styles["body"])],
        [Paragraph("Còn cách xác nhận", styles["bold"]), Paragraph(_pct(row.get("distance_to_trigger_pct")), styles["body"])],
        [Paragraph("Mục tiêu tham khảo", styles["bold"]), Paragraph(_clean(row.get("target_price")), styles["body"])],
        [Paragraph("Dư địa tham khảo", styles["bold"]), Paragraph(_pct(row.get("potential_profit_pct")), styles["body"])],
        [Paragraph("Lý do vào radar", styles["bold"]), Paragraph(_clean(row.get("setup_reason")), styles["body"])],
    ]
    table = Table(data, colWidths=[54 * mm, 124 * mm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#ddd5ca")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2ede4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _volume_fact_table(row: Mapping[str, Any], styles: Mapping[str, ParagraphStyle]) -> Table:
    ratio = row.get("volume_ratio_20")
    contraction = row.get("pattern_volume_contraction_ratio")
    mfi = row.get("mfi_14")
    ratio_text = f"{float(ratio):.2f}x nền 20 phiên" if ratio not in (None, "") and not pd.isna(ratio) else "-"
    contraction_text = f"{float(contraction):.2f}x vùng trước mẫu" if contraction not in (None, "") and not pd.isna(contraction) else "-"
    mfi_text = f"{float(mfi):.1f}" if mfi not in (None, "") and not pd.isna(mfi) else "-"
    data = [
        [Paragraph("Câu hỏi", styles["bold"]), Paragraph("Cách đọc", styles["bold"])],
        [Paragraph("Sức khối lượng", styles["body"]), Paragraph(_volume_label(row.get("volume_quality_label")), styles["body"])],
        [Paragraph("Cảnh báo", styles["body"]), Paragraph(_volume_label(row.get("volume_warning_label")), styles["body"])],
        [Paragraph("Volume so với nền", styles["body"]), Paragraph(ratio_text, styles["body"])],
        [Paragraph("Volume trong mẫu", styles["body"]), Paragraph(contraction_text, styles["body"])],
        [Paragraph("Pha giá-khối lượng", styles["body"]), Paragraph(_clean(row.get("price_volume_phase")), styles["body"])],
        [Paragraph("MFI 14", styles["body"]), Paragraph(mfi_text, styles["body"])],
    ]
    table = Table(data, colWidths=[54 * mm, 124 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eee8dc")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#ddd5ca")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _row_fact_table(row: Mapping[str, Any], styles: Mapping[str, ParagraphStyle]) -> Table:
    data = [
        [Paragraph("Mẫu hình", styles["bold"]), Paragraph(_clean(row.get("pattern_label") or row.get("pattern_id")), styles["body"])],
        [Paragraph("Ngày xác nhận", styles["bold"]), Paragraph(_date_label(row.get("event_date")), styles["body"])],
        [Paragraph("Nhóm", styles["bold"]), Paragraph(f"{_clean(row.get('market_group'))} - thanh khoản {_clean(row.get('liquidity_bucket'))}", styles["body"])],
        [Paragraph("Lợi nhuận tiềm năng lịch sử", styles["bold"]), Paragraph(_prob(row.get("potential_profit_pct")), styles["body"])],
        [Paragraph("Xác suất đạt mục tiêu", styles["bold"]), Paragraph(_prob(row.get("target_success_probability")), styles["body"])],
        [Paragraph("Đạt mục tiêu trước kéo ngược 5%", styles["bold"]), Paragraph(_prob(row.get("clean_path_probability")), styles["body"])],
        [Paragraph("Đường đi từ xác nhận", styles["bold"]), Paragraph(f"Đã tăng tốt nhất {_pct(row.get('mfe_pct'))}; kéo ngược sâu nhất {_pct(row.get('mae_pct'), absolute=True)}.", styles["body"])],
    ]
    table = Table(data, colWidths=[54 * mm, 124 * mm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#ddd5ca")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2ede4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _case_note(row: Mapping[str, Any]) -> str:
    return (
        f"{_clean(row.get('symbol'))} là một ứng viên để mở chart kiểm tra, không phải lệnh mua. "
        f"Điểm cần xem là giá còn giữ được vùng xác nhận hay không, mức kéo ngược có vượt ngưỡng chịu đựng hay không, "
        f"và bối cảnh VNINDEX/thanh khoản có ủng hộ mẫu hình tiếp diễn hay không."
    )


def _setup_case_note(row: Mapping[str, Any]) -> str:
    volume_quality = _volume_label(row.get("volume_quality_label"))
    volume_warning = _volume_label(row.get("volume_warning_label"))
    return (
        f"{_clean(row.get('symbol'))} mới là mẫu đang hình thành, chưa xác nhận. "
        f"Điểm cần xem là giá có đóng cửa vượt vùng xác nhận hay không. "
        f"Khối lượng hiện đọc là {volume_quality}; cảnh báo khối lượng: {volume_warning}. "
        "Nếu giá chưa vượt điểm xác nhận, dòng này chỉ nên dùng để đưa vào danh sách mở chart theo dõi."
    )


def _build_story(summary: Mapping[str, Any], styles: Mapping[str, ParagraphStyle], chart_dir: Path) -> tuple[list[Any], dict[str, Any]]:
    story: list[Any] = []
    chart_reports: list[dict[str, Any]] = []
    sections = [
        ("Ứng viên BUY tiềm năng", summary.get("sections", {}).get("buy_candidates", [])),
        ("Watchlist theo dõi thêm", summary.get("sections", {}).get("watchlist", [])),
    ]
    story.append(Paragraph("BUY Candidate Scan - Hồ sơ cơ hội VN100", styles["title"]))
    story.append(Paragraph(f"Thời điểm tạo: {_clean(summary.get('generated_at') or datetime.now().isoformat(timespec='seconds'))}", styles["small"]))
    story.append(Spacer(1, 5 * mm))
    story.append(
        Paragraph(
            "PDF này là phần chi tiết đính kèm mail nhanh. Phạm vi BUY và Watchlist bắt buộc chỉ lấy VN100/VN30. "
            "Các xác suất là thống kê lịch sử của mẫu hình, không phải cam kết cho từng mã.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(_summary_table(summary, styles))
    story.append(PageBreak())
    setup_rows = summary.get("sections", {}).get("buy_setup", [])
    story.append(Paragraph("Đang hình thành trước xác nhận", styles["h1"]))
    if not setup_rows:
        story.append(Paragraph("Không có setup VN100/VN30 đủ gần điểm xác nhận trong vòng quét này.", styles["body"]))
        story.append(PageBreak())
    else:
        story.append(
            Paragraph(
                "Các dòng dưới đây là radar trước phá vỡ. Chúng giúp chọn mã để mở chart quan sát, không thay thế điều kiện xác nhận bằng giá đóng cửa.",
                styles["body"],
            )
        )
        story.append(Spacer(1, 4 * mm))
        for index, row in enumerate(setup_rows, start=1):
            story.append(Paragraph(f"{index}. {_clean(row.get('symbol'))} - {_clean(row.get('pattern_label') or row.get('pattern_id'))}", styles["h1"]))
            story.append(_setup_fact_table(row, styles))
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph("Khối lượng nói gì?", styles["h2"]))
            story.append(_volume_fact_table(row, styles))
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph(_setup_case_note(row), styles["body"]))
            if index != len(setup_rows):
                story.append(PageBreak())
        story.append(PageBreak())
    for title, rows in sections:
        story.append(Paragraph(title, styles["h1"]))
        if not rows:
            story.append(Paragraph("Không có mã phù hợp trong phạm vi VN100/VN30 ở cửa sổ quét này.", styles["body"]))
            story.append(PageBreak())
            continue
        for index, row in enumerate(rows, start=1):
            story.append(Paragraph(f"{index}. {_clean(row.get('symbol'))} - {_clean(row.get('pattern_label') or row.get('pattern_id'))}", styles["h1"]))
            story.append(_row_fact_table(row, styles))
            story.append(Spacer(1, 4 * mm))
            chart_path, chart_report = _render_chart(row, chart_dir)
            chart_reports.append({"symbol": row.get("symbol"), "pattern_id": row.get("pattern_id"), **chart_report})
            if chart_path and chart_path.exists():
                story.append(Image(str(chart_path), width=178 * mm, height=91.5 * mm))
            else:
                story.append(Paragraph("Không render được chart cho mã này; cần mở chart thủ công từ dữ liệu nguồn.", styles["small"]))
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph(_case_note(row), styles["body"]))
            story.append(Spacer(1, 4 * mm))
            story.append(
                Paragraph(
                    "Mốc đọc nhanh: giá phá vỡ là vùng xác nhận; mục tiêu là mốc tham chiếu; kéo ngược sâu hơn vùng cảnh báo làm chất lượng cơ hội giảm.",
                    styles["small"],
                )
            )
            if not (title == sections[-1][0] and index == len(rows)):
                story.append(PageBreak())
    return story, {
        "workflow_id": WORKFLOW_ID,
        "status": "PASS",
        "chart_reports": chart_reports,
        "chart_count": sum(1 for item in chart_reports if item.get("status") in {"PASS", "WARN"}),
    }


def build_realtime_scan_pdf_report(
    summary: Mapping[str, Any],
    *,
    pdf_path: Path = DEFAULT_PDF_OUT,
    chart_dir: Path = DEFAULT_CHART_DIR,
) -> dict[str, Any]:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    for stale in chart_dir.glob("*.png"):
        stale.unlink()
    styles = _styles()
    story, report = _build_story(summary, styles, chart_dir)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=13 * mm,
        bottomMargin=13 * mm,
        title="BUY Candidate Scan - Hồ sơ cơ hội VN100",
        author="Bloger Chim Cut",
    )
    doc.build(story)
    report = {**report, "pdf_path": str(pdf_path), "chart_dir": str(chart_dir)}
    report_path = pdf_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build realtime scan detail PDF from an email summary JSON.")
    parser.add_argument("--summary-json", default="artifacts/realtime_scan/latest/email/realtime_scan_email_summary.json")
    parser.add_argument("--pdf-out", default=str(DEFAULT_PDF_OUT))
    parser.add_argument("--chart-dir", default=str(DEFAULT_CHART_DIR))
    args = parser.parse_args()
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    report = build_realtime_scan_pdf_report(summary, pdf_path=Path(args.pdf_out), chart_dir=Path(args.chart_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
