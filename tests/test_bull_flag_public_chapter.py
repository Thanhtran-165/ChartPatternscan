from __future__ import annotations

from pathlib import Path
import json

from pypdf import PdfReader

from scanner.build_bull_flag_public_chapter import build_public_chapter


def test_public_chapter_emits_reader_facing_pdf_with_examples(tmp_path: Path) -> None:
    paths = build_public_chapter(out_dir=tmp_path / "public")

    assert paths["pdf"].exists()
    assert paths["pdf"].stat().st_size > 100_000
    assert paths["chart_schematic"].exists()
    assert paths["chart_textbook_success"].exists()
    assert paths["chart_middle_case"].exists()
    assert paths["chart_failure"].exists()
    assert paths["manuscript"].exists()
    assert paths["content_parity_audit_json"].exists()
    assert paths["content_parity_audit_md"].exists()

    reader = PdfReader(str(paths["pdf"]))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert "Kết quả quan trọng" in text
    assert "Cách nhận diện" in text
    assert "Ví dụ trong VN100" in text
    assert "Mẫu hình hoạt động ra sao" in text
    assert "Khi nên đọc thận trọng" in text
    assert "Diễn biến mẫu hoàn chỉnh" in text
    assert "Tập trung vào thất bại" in text
    assert "Cách đọc kết quả quan trọng" in text
    assert "Mục tiêu giá" in text
    assert "Vùng thường gặp và vùng cực trị" in text
    assert "Hành vi sau phá vỡ" in text
    assert "Kích thước và khối lượng" in text
    assert "Phụ lục kỹ thuật" in text
    assert "Phụ lục bối cảnh" in text
    assert "Cách sử dụng thực tế" in text
    assert "Khi mẫu đáng chú ý hơn" in text
    assert "Tóm tắt thực hành" in text
    assert "Checklist đọc mẫu" in text
    assert "So với Bulkowski" not in text
    assert "Đối chiếu lại tài liệu gốc" not in text
    for forbidden in [
        "MFE",
        "MAE",
        "breakout",
        "stop loss",
        "half-staff",
        "swing",
        "path dữ liệu",
        "research",
        "setup",
        "proxy",
        "available",
        "biên thuận lợi",
        "biên bất lợi",
        "hạ trọng số",
        "point-in-time",
    ]:
        assert forbidden not in text
    assert "SHB" in text
    assert "MBB" in text
    assert "MWG" in text

    payload = json.loads(paths["payload"].read_text(encoding="utf-8"))
    groups = {event["market_group"] for event in payload["example_events"].values()}
    assert groups <= {"VN30", "VN100 ex VN30"}

    audit = json.loads(paths["content_parity_audit_json"].read_text(encoding="utf-8"))
    covered = {row["source_section"] for row in audit["status"]}
    assert {"Kết quả quan trọng", "Tour mẫu hình", "Điều kiện đọc thận trọng", "Focus on failures", "Vùng phân bố kết quả", "Kích thước và khối lượng", "Chiến thuật giao dịch", "For best performance", "Tóm tắt thực hành"} <= covered


def test_public_chapter_accepts_approved_ai_sections(tmp_path: Path) -> None:
    ai_sections = tmp_path / "approved_ai_sections.json"
    ai_sections.write_text(
        json.dumps(
            {
                "title": "Cờ tăng",
                "approved_sections": [
                    {
                        "id": "overview",
                        "title": "Tóm tắt",
                        "paragraphs": ["Đoạn thử nghiệm từ lớp biên tập AI đã qua hậu kiểm."],
                        "callout": None,
                        "claims_used": [],
                    }
                ],
                "example_captions": {
                    "textbook_success": "Chú thích AI cho SHB đã qua hậu kiểm.",
                    "middle_case": "Chú thích AI cho MBB đã qua hậu kiểm.",
                    "failure": "Chú thích AI cho MWG đã qua hậu kiểm.",
                },
                "final_caveat": "Không phải lời khuyên giao dịch.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    paths = build_public_chapter(out_dir=tmp_path / "public_ai", ai_sections_path=ai_sections)
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(paths["pdf"])).pages)

    assert "Đoạn thử nghiệm từ lớp biên tập AI đã qua hậu kiểm." in text
    assert "Chú thích AI cho SHB đã qua hậu kiểm." in text
    payload = json.loads(paths["payload"].read_text(encoding="utf-8"))
    assert payload["ai_sections_source"] == str(ai_sections)
