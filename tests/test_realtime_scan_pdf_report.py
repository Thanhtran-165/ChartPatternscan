from __future__ import annotations

from pathlib import Path

from scanner.build_realtime_scan_pdf_report import build_realtime_scan_pdf_report


def test_realtime_scan_pdf_includes_buy_setup_volume_section(tmp_path: Path) -> None:
    summary = {
        "generated_at": "2026-06-16T17:00:00",
        "counts": {
            "buy_setup": 1,
            "buy_candidates": 0,
            "watchlist": 0,
        },
        "sections": {
            "buy_setup": [
                {
                    "symbol": "AAA",
                    "pattern_id": "bull_flags",
                    "pattern_label": "Cờ tăng",
                    "market_group": "VN30",
                    "latest_date": "2026-06-16",
                    "last_close": 10.5,
                    "trigger_price": 11.0,
                    "distance_to_trigger_pct": 4.76,
                    "target_price": 12.0,
                    "potential_profit_pct": 14.29,
                    "setup_reason": "Mẫu đang nén gần vùng xác nhận.",
                    "volume_quality_label": "strong",
                    "volume_warning_label": "none",
                    "volume_ratio_20": 1.8,
                    "pattern_volume_contraction_ratio": 0.7,
                    "price_volume_phase": "up_confirmed",
                    "mfi_14": 58.2,
                }
            ],
            "buy_candidates": [],
            "watchlist": [],
        },
    }

    report = build_realtime_scan_pdf_report(
        summary,
        pdf_path=tmp_path / "realtime_scan_detail.pdf",
        chart_dir=tmp_path / "charts",
    )

    assert report["status"] == "PASS"
    assert Path(report["pdf_path"]).exists()
    assert Path(report["report_path"]).exists()
    assert report["chart_count"] == 0
