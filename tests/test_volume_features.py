from __future__ import annotations

import pandas as pd

from scanner.volume_features import compute_latest_volume_features


def test_volume_features_detect_up_confirmed_phase() -> None:
    rows = []
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    for idx in range(30):
        rows.append(
            {
                "date": dates[idx],
                "open": 10 + idx * 0.1,
                "high": 10.5 + idx * 0.1,
                "low": 9.8 + idx * 0.1,
                "close": 10 + idx * 0.1,
                "volume": 1000 + idx * 10,
            }
        )
    rows[-1]["close"] = rows[-2]["close"] * 1.03
    rows[-1]["volume"] = 2500
    df = pd.DataFrame(rows)

    features = compute_latest_volume_features(df, setup_start_date="2026-01-20")

    assert features["price_volume_phase"] == "up_confirmed"
    assert features["volume_quality_label"] == "strong"
    assert features["volume_ratio_20"] > 1.25
    assert features["mfi_14"] is not None
    assert "obv_slope_20" in features


def test_volume_features_warn_on_noisy_setup_volume() -> None:
    rows = []
    dates = pd.date_range("2026-02-01", periods=40, freq="D")
    for idx in range(40):
        volume = 1000 if idx < 20 else 2000
        rows.append(
            {
                "date": dates[idx],
                "open": 10,
                "high": 10.5,
                "low": 9.8,
                "close": 10 + idx * 0.01,
                "volume": volume,
            }
        )
    df = pd.DataFrame(rows)

    features = compute_latest_volume_features(df, setup_start_date="2026-02-21")

    assert features["pattern_volume_contraction_ratio"] >= 1.35
    assert features["volume_warning_label"] == "noisy_setup_volume"
