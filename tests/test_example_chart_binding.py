from __future__ import annotations

import pandas as pd

from scanner.canonical_example_charts import _slice_window
from scanner.pattern_publication_core import _example_caption


def _prices(n: int = 140) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.0,
            "volume": 1_000.0,
        }
    )


def test_example_window_reaches_late_target() -> None:
    frame = _prices()
    event = {
        "formation_start_date": "2020-01-05",
        "formation_end_date": "2020-01-10",
        "breakout_date": "2020-01-11",
        "target_hit": True,
        "days_to_target": 80,
        "evaluated_bars": 90,
    }
    window, offset = _slice_window(frame, event)
    breakout_local = 10 - offset
    assert len(window) > breakout_local + 85


def test_caption_ignores_stale_ai_lead() -> None:
    event = {
        "symbol": "SCS",
        "breakout_date": "2018-04-27",
        "pattern_width_bars": 10,
        "pattern_height_pct": 12.3,
        "prior_trend_pct": 8.0,
        "mfe_pct": 20.0,
        "mae_pct": 2.0,
        "target_hit": True,
        "failure_5pct": False,
        "days_to_target": 12,
        "path_quality_bucket": "clean",
    }
    caption = _example_caption(
        key="textbook_success",
        fallback="Trên biểu đồ GMD năm 2021, mẫu cũ đã đạt mục tiêu.",
        event=event,
        spec={"title": "Nêm mở rộng tăng", "favorable_move": "mức đi thuận lợi", "adverse_move": "mức kéo ngược"},
    )
    assert "GMD" not in caption
    assert "SCS xác nhận ngày 2018-04-27" in caption
