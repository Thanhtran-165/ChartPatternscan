from __future__ import annotations

"""Gate parity target_hit — Sol BLOCKER 3 (đợt A2).

Release gate bắt buộc: "raw events target_hit == canonical core target_hit,
mismatch = 0 trên toàn bộ events được xuất bản".

Test dựng artifacts synthetic (events.csv + post_breakout_path.csv) bằng CHÍNH
`_evaluate_detection` của detector (pipes — giờ gọi target_hit_core) rồi chạy
`scanner.audit_target_hit_core_parity.run_parity_check`:
  1. parity_pass — events do detector sinh ra phải khớp core: mismatch = 0.
  2. gate_catches_flipped_hit — bóp ngược 1 event target_hit → gate phải bắt
     (mismatch = 1, gate FAIL) — chứng minh gate không phải vô điều kiện PASS.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.audit_target_hit_core_parity import run_parity_check
from scanner.v2.pipes import _evaluate_detection


def _frame(values: list[tuple[float, float, float, float]], symbol: str = "AAA") -> pd.DataFrame:
    rows = []
    for i, (open_, high, low, close) in enumerate(values):
        rows.append(
            {
                "symbol": symbol,
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 100_000 + i * 1_000,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_case() -> tuple[pd.DataFrame, list[dict]]:
    """2 detection: 1 hit (vượt target sau 3 nến), 1 miss (chụm dưới target)."""
    df = _frame(
        [(10.0, 10.2, 9.8, 10.0)] * 3
        + [(10.1, 12.0, 10.0, 11.8), (11.9, 12.5, 11.4, 12.2), (12.2, 13.1, 12.0, 12.9), (13.0, 13.4, 12.8, 13.2)]
        + [(13.1, 13.3, 12.9, 13.0)] * 8
    )
    detections = [
        {
            "pattern_key": "pipe_bottoms",
            "event_id": "EV-HIT",
            "symbol": "AAA",
            "breakout_idx": 2,
            "breakout_price": 10.0,
            "target_price": 13.0,
            "breakout_direction": "up",
        },
        {
            "pattern_key": "pipe_bottoms",
            "event_id": "EV-MISS",
            "symbol": "AAA",
            "breakout_idx": 2,
            "breakout_price": 10.0,
            "target_price": 14.0,
            "breakout_direction": "up",
        },
    ]
    return df, detections


def _write_artifacts(base: Path, df: pd.DataFrame, detections: list[dict]) -> Path:
    out_dir = base / "gate_demo"
    out_dir.mkdir(parents=True)
    event_rows = []
    path_rows = []
    for det in detections:
        res = _evaluate_detection(df, det)
        event_rows.append({**det, **res})
        forward = df.iloc[int(det["breakout_idx"]) + 1 :]
        for bar, (_, row) in enumerate(forward.iterrows(), start=1):
            path_rows.append(
                {
                    "event_id": det["event_id"],
                    "symbol": det["symbol"],
                    "bar_after_breakout": bar,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
    pd.DataFrame(event_rows).to_csv(out_dir / "events.csv", index=False)
    pd.DataFrame(path_rows).to_csv(out_dir / "post_breakout_path.csv", index=False)
    return out_dir


def test_parity_pass_detector_events_match_core(tmp_path: Path) -> None:
    df, detections = _synthetic_case()
    _write_artifacts(tmp_path, df, detections)
    summary = run_parity_check(tmp_path)
    assert summary["pairs_scanned"] == 1
    assert summary["compared"] == 2
    assert summary["mismatch"] == 0
    assert summary["gate"] == "PASS"


def test_gate_catches_flipped_hit(tmp_path: Path) -> None:
    df, detections = _synthetic_case()
    out_dir = _write_artifacts(tmp_path, df, detections)
    events = pd.read_csv(out_dir / "events.csv")
    assert bool(events.loc[events["event_id"] == "EV-MISS", "target_hit"].iloc[0]) is False
    events.loc[events["event_id"] == "EV-MISS", "target_hit"] = True
    events.to_csv(out_dir / "events.csv", index=False)
    summary = run_parity_check(tmp_path)
    assert summary["mismatch"] == 1
    assert summary["gate"] == "FAIL"
    assert summary["sample_mismatches"][0]["event_id"] == "EV-MISS"


def test_gate_respects_evaluated_bars_horizon(tmp_path: Path) -> None:
    """Gate phải cắt path theo horizon detector đã evaluate (`evaluated_bars`).

    EV-HIT thực sự vượt target ở bar 3, nhưng events.csv mô phỏng detector chỉ
    evaluate 2 bars (target_hit=False). Nếu gate quét full path sẽ SAI báo
    mismatch; đúng phải tôn trọng evaluated_bars=2 → core=False → parity PASS.
    """
    df, detections = _synthetic_case()
    out_dir = _write_artifacts(tmp_path, df, detections)
    events = pd.read_csv(out_dir / "events.csv")
    events["evaluated_bars"] = 2
    events["target_hit"] = False
    events.to_csv(out_dir / "events.csv", index=False)
    summary = run_parity_check(tmp_path)
    assert summary["compared"] == 2
    assert summary["mismatch"] == 0
    assert summary["gate"] == "PASS"
