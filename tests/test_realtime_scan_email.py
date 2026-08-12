from __future__ import annotations

import pandas as pd

from scanner.send_realtime_scan_email import (
    ACTIONABLE,
    RISK,
    WATCHLIST,
    render_html_email,
    render_text_email,
    summarize_watchlist,
)


def test_realtime_scan_email_groups_candidate_watchlist_and_risk() -> None:
    watchlist = pd.DataFrame(
        [
            {
                "pattern_id": "bull_flags",
                "symbol": "AAA",
                "event_date": "2026-06-01",
                "market_group": "VN100 ex VN30",
                "liquidity_bucket": "high",
                "mfe_pct": 4.2,
                "mae_pct": 1.1,
                "after_buy_action": ACTIONABLE,
            },
                {
                    "pattern_id": "bull_pennants",
                    "symbol": "BBB",
                    "event_date": "2026-06-01",
                    "market_group": "VN100 ex VN30",
                    "liquidity_bucket": "mid",
                    "mfe_pct": 2.2,
                    "mae_pct": 1.7,
                "after_buy_action": WATCHLIST,
            },
            {
                "pattern_id": "bear_flags",
                "symbol": "CCC",
                "event_date": "2026-06-01",
                "market_group": "VN30",
                "liquidity_bucket": "high",
                "mfe_pct": None,
                "mae_pct": None,
                "after_buy_action": RISK,
            },
        ]
    )

    summary = summarize_watchlist(watchlist)

    assert summary["counts"]["buy_candidates"] == 1
    assert summary["counts"]["watchlist"] == 1
    assert summary["counts"]["risk_context"] == 1
    assert summary["sections"]["buy_candidates"][0]["symbol"] == "AAA"
    assert summary["sections"]["watchlist"][0]["symbol"] == "BBB"
    assert summary["sections"]["risk_context"][0]["symbol"] == "CCC"


def test_realtime_scan_email_text_keeps_non_advice_boundary() -> None:
    summary = summarize_watchlist(
        pd.DataFrame(
            [
                {
                    "pattern_id": "double_bottoms_adam_adam",
                    "symbol": "AAA",
                    "event_date": "2026-06-01",
                    "after_buy_action": ACTIONABLE,
                }
            ]
        )
    )

    text = render_text_email(summary)

    assert "không phải khuyến nghị mua bán" in text
    assert "Ứng viên BUY tiềm năng" in text
    assert "Cảnh báo tránh mua" not in text
    assert "Risk / avoid-buy" not in text
    assert "BUY Candidate Scan" in text
    assert "Khả năng chạm trước kéo ngược mạnh" in text


def test_realtime_scan_email_html_uses_reader_friendly_column_names() -> None:
    summary = summarize_watchlist(
        pd.DataFrame(
            [
                {
                    "pattern_id": "bull_flags",
                    "symbol": "AAA",
                    "event_date": "2026-06-01",
                    "market_group": "VN100 ex VN30",
                    "mae_pct": 1.0,
                    "failure_5pct": False,
                    "target_hit": False,
                    "after_buy_action": ACTIONABLE,
                }
            ]
        )
    )

    html = render_html_email(summary)

    assert "Khả năng chạm trước kéo ngược mạnh" in html
    assert "Đã tăng / đã kéo ngược" in html
    assert "Xác suất đường đi sạch" not in html
    assert "Đường đi hiện tại" not in html


def test_realtime_scan_email_displays_adverse_move_as_magnitude() -> None:
    summary = summarize_watchlist(
        pd.DataFrame(
            [
                {
                    "pattern_id": "rectangle_bottoms",
                    "symbol": "NEG",
                    "event_date": "2026-06-01",
                    "market_group": "VN100 ex VN30",
                    "mfe_pct": 2.5,
                    "mae_pct": -0.98,
                    "failure_5pct": False,
                    "target_hit": False,
                    "after_buy_action": WATCHLIST,
                }
            ]
        )
    )

    text = render_text_email(summary)

    assert "kéo ngược sâu nhất 0.98%" in text
    assert "kéo ngược sâu nhất -0.98%" not in text


def test_realtime_scan_email_displays_data_freshness_when_available() -> None:
    summary = summarize_watchlist(pd.DataFrame())
    summary["data_refresh"] = {
        "status": "REFRESHED",
        "max_date": "2026-06-05",
        "days_stale": 0,
    }

    text = render_text_email(summary)
    html = render_html_email(summary)

    assert "Dữ liệu giá: trạng thái REFRESHED, mới nhất 2026-06-05, độ trễ 0 ngày." in text
    assert "Dữ liệu giá" in html
    assert "2026-06-05" in html


def test_realtime_scan_email_demotes_deep_adverse_buy_candidate_to_risk() -> None:
    summary = summarize_watchlist(
        pd.DataFrame(
            [
                {
                    "pattern_id": "double_bottoms_adam_adam",
                    "symbol": "RISKY",
                    "event_date": "2026-06-01",
                    "mae_pct": 8.5,
                    "failure_5pct": False,
                    "target_hit": False,
                    "after_buy_action": ACTIONABLE,
                }
            ]
        )
    )

    assert summary["counts"]["buy_candidates"] == 0
    assert summary["counts"]["risk_context"] == 1


def test_realtime_scan_email_requires_vn100_for_buy_candidates() -> None:
    summary = summarize_watchlist(
        pd.DataFrame(
            [
                {
                    "pattern_id": "bull_flags",
                    "symbol": "OUT",
                    "event_date": "2026-06-01",
                    "market_group": "Outside VN100",
                    "mae_pct": 1.0,
                    "failure_5pct": False,
                    "target_hit": False,
                    "after_buy_action": ACTIONABLE,
                },
                {
                    "pattern_id": "bull_flags",
                    "symbol": "VN100",
                    "event_date": "2026-06-01",
                    "market_group": "VN100 ex VN30",
                    "mae_pct": 1.0,
                    "failure_5pct": False,
                    "target_hit": False,
                    "after_buy_action": ACTIONABLE,
                },
            ]
        )
    )

    assert summary["display_scope"]["buy_candidates"] == "vn100_required"
    assert summary["counts"]["buy_candidates"] == 1
    assert summary["sections"]["buy_candidates"][0]["symbol"] == "VN100"


def test_realtime_scan_email_does_not_fallback_to_outside_vn100() -> None:
    summary = summarize_watchlist(
        pd.DataFrame(
            [
                {
                    "pattern_id": "bull_flags",
                    "symbol": "OUT",
                    "event_date": "2026-06-01",
                    "market_group": "Outside VN100",
                    "mae_pct": 1.0,
                    "failure_5pct": False,
                    "target_hit": False,
                    "after_buy_action": ACTIONABLE,
                }
            ]
        )
    )

    assert summary["display_scope"]["buy_candidates"] == "vn100_required_no_rows"
    assert summary["counts"]["buy_candidates"] == 0
    assert summary["counts"]["buy_candidates_all_market_before_vn100_filter"] == 1
    assert summary["sections"]["buy_candidates"] == []
