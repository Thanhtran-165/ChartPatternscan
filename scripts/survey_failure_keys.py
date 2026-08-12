#!/usr/bin/env python3
"""Khảo sát tên khóa mức giá THỰC của từng family detection — phục vụ map _FAMILY_LEVEL_KEYS.

Chạy scan 1 mã (CTD) qua mọi detector có thể, in các key chứa low/high/extreme/level/
neckline/handle/flag/rim/support/resistance/boundary/edge + giá trị tương ứng.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sqlite3  # noqa: E402
import pandas as pd  # noqa: E402

DB = Path.home() / "dev/market_stats_v2/market_cache/stock_ohlcv/latest.sqlite"
from scanner.run_bear_flag_db_source_parity_audit import _load_symbol_from_db  # noqa: E402

LEVEL_KEYWORDS = ("low", "high", "extreme", "level", "neckline", "handle", "flag",
                  "rim", "support", "resistance", "boundary", "edge", "base", "stop",
                  "height", "width", "gap", "rise", "fall")

MODULES = [
    ("ascending_triangles", {}),
    ("descending_triangles", {}),
    ("symmetrical_triangles", {}),
    ("flags_experiment", {}),
    ("measured_moves", {"pattern_key": "measured_move_up"}),
    ("double_patterns", {"family": "double_bottom"}),
    ("rectangles", {"pattern_key": "rectangle_bottoms"}),
    ("scallops", {"pattern_key": "scallop_ascending"}),
    ("bump_and_run", {"pattern_key": "bump_and_run_bottom"}),
    ("broadening_patterns", {"pattern_key": "broadening_bottoms"}),
    ("diamonds", {"pattern_key": "diamond_bottom"}),
    ("dead_cat_bounce", {"pattern_key": "dead_cat_bounce"}),
    ("horns", {"pattern_key": "horn_bottoms"}),
    ("inside_days", {}),
    ("rounding", {"pattern_key": "rounding_bottoms"}),
    ("three_methods", {"pattern_key": "rising_three_methods"}),
    ("three_peaks_valleys", {"pattern_key": "three_peaks"}),
    ("triple_patterns", {"pattern_key": "triple_bottoms"}),
    ("high_tight_flags", {"pattern_key": "high_tight_flags"}),
    ("pennants", {"pattern_key": "pennants"}),
    ("falling_wedges", {"pattern_key": "wedges_falling"}),
    ("rising_wedges", {"pattern_key": "wedges_rising"}),
    ("cup_with_handle", {"variant": "cup_with_handle"}),
    ("pipes", {"pattern_key": "pipe_bottoms"}),
    ("gaps", {}),
]

SYMBOLS = ["VCB", "CTD", "FPT", "VNM", "HPG", "SSI"]


def main() -> None:
    conn = sqlite3.connect(str(DB))
    frames = {s: _load_symbol_from_db(conn, s) for s in SYMBOLS}
    conn.close()
    for mod_name, kwargs in MODULES:
        all_rows = []
        try:
            mod = __import__(f"scanner.v2.{mod_name}", fromlist=["scan_symbol"])
            for sym in SYMBOLS:
                try:
                    detections, _ = mod.scan_symbol(frames[sym], **kwargs)
                    all_rows.extend(detections or [])
                except Exception as exc:
                    print(f"  ⚠️ {mod_name} {sym}: {type(exc).__name__}: {str(exc)[:60]}")
        except Exception as exc:
            print(f"### {mod_name}: LỖI {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if not all_rows:
            print(f"### {mod_name}: 0 event")
            continue
        row = all_rows[0]
        pk = row.get("pattern_key") or row.get("variant") or mod_name
        print(f"### {mod_name} (pk={pk}, n={len(all_rows)}) breakout={row.get('breakout_price')} target={row.get('target_price')}")
        for k, v in sorted(row.items()):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                print(f"    {k} = {v}")
        print()


if __name__ == "__main__":
    main()
