"""Đo đối chứng H3: busted tính bằng WICK (low/high — pipeline hiện tại, 03 §2.2)
so với CLOSE (đề xuất của V4 Pro review 13/08) — KHÔNG sửa pipeline, chỉ tái tính
từ events V3 có sẵn + DB giá, để lượng hóa ảnh hưởng của lựa chọn mốc busted.

Chạy: python3 scripts/h3_wick_vs_close_comparison.py
Kết quả in: failure rate wick vs close + số event đổi trạng thái (theo pattern)."""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scanner.ohlcv_normalizer import OHLCVNormalizer  # noqa: E402
from scanner.v2.failure_logic import _is_up, _pick_failure_level  # noqa: E402
from scanner.v2.measurement_registry import failure_threshold_pct, family_of, lookahead_bars  # noqa: E402

ROOT = str(Path(__file__).resolve().parents[1])
DBS = [
    ("repo", f"{ROOT}/vietnam_stocks.db"),
    ("market_cache", "/Users/bobo/dev/market_stats_v2/market_cache/stock_ohlcv/latest.sqlite"),
]
EVENTS_ROOT = f"{ROOT}/artifacts/scanner_v2_v3"
PATTERNS = ["cup_with_handle", "inside_day"]


def _load_series(con: sqlite3.Connection, cache: dict, symbol: str) -> pd.DataFrame:
    if symbol not in cache:
        raw = pd.read_sql(
            "SELECT symbol, time AS date, open, high, low, close, volume "
            "FROM stock_price_history WHERE symbol=? ORDER BY time",
            con,
            params=(symbol,),
        )
        df, _ = OHLCVNormalizer().normalize(raw)
        cache[symbol] = df
    return cache[symbol]


def _busted(future: pd.DataFrame, up: bool, edge: float, target: float, use_close: bool) -> bool:
    """Tái tạo vòng lặp _evaluate (failure_logic.py) với mốc busted đổi được."""
    for _, r in future.iterrows():
        high = float(r["high"])
        low = float(r["low"])
        hit_target = (high >= target) if up else (low <= target)
        if use_close:
            crossed = (float(r["close"]) <= edge) if up else (float(r["close"]) >= edge)
        else:
            crossed = (low <= edge) if up else (high >= edge)
        if hit_target:
            return False
        if crossed:
            return True
    return False


def main() -> None:
    for db_name, db_path in DBS:
        con = sqlite3.connect(db_path)
        print(f"=== DB: {db_name} ({db_path}) ===")
        for pat in PATTERNS:
            csv_path = f"{EVENTS_ROOT}/{pat}/db_active/events.csv"
            with open(csv_path, encoding="utf-8") as f:
                events = list(csv.DictReader(f))
            if not events:
                print(f"{pat}: không có events")
                continue
            cache: dict[str, pd.DataFrame] = {}
            n_wick = n_close = n_changed = n_recompute_mismatch = n_skip = 0
            for ev in events:
                sym = ev.get("symbol") or ""
                try:
                    breakout_idx = int(ev["breakout_idx"])
                except (ValueError, KeyError, TypeError):
                    n_skip += 1
                    continue
                df = _load_series(con, cache, sym)
                if breakout_idx >= len(df) - 1:
                    n_skip += 1
                    continue
                lookahead = lookahead_bars(ev.get("pattern_key") or pat) or 252
                future = df.iloc[breakout_idx + 1 : min(len(df), breakout_idx + 1 + lookahead)]
                if future.empty:
                    n_skip += 1
                    continue
                pk = ev.get("pattern_key") or pat
                up = _is_up(ev.get("breakout_direction", "up"))
                level = _pick_failure_level(ev, family_of(pk), up) or float(ev["breakout_price"])
                if level <= 0:
                    n_skip += 1
                    continue
                threshold = float(failure_threshold_pct(pk))
                edge = level * (1.0 - threshold / 100.0) if up else level * (1.0 + threshold / 100.0)
                target = float(ev["target_price"])
                busted_w = _busted(future, up, edge, target, use_close=False)
                busted_c = _busted(future, up, edge, target, use_close=True)
                csv_w = str(ev.get("failure_busted") or "").strip().lower() in ("true", "1", "yes")
                if busted_w != csv_w:
                    n_recompute_mismatch += 1
                if busted_w:
                    n_wick += 1
                if busted_c:
                    n_close += 1
                if busted_w != busted_c:
                    n_changed += 1
            total = len(events)
            print(
                f"{pat}: total={total} (skip={n_skip}) | "
                f"wick={n_wick}/{total} ({n_wick/total*100:.1f}%) | "
                f"close={n_close}/{total} ({n_close/total*100:.1f}%) | "
                f"đổi trạng thái={n_changed} ({n_changed/total*100:.1f}%) | "
                f"recompute≠csv(wick)={n_recompute_mismatch}"
            )
    con.close()


if __name__ == "__main__":
    main()
