#!/usr/bin/env python3
"""M3 — Re-scan toàn DB (1599 mã) → events.csv mới có cột V3
(weak_move_5pct, failure_busted, days_to_bust, target_dist_pct).

Ghi vào artifacts/scanner_v2_v3/<pattern>/db_active/events.csv (THƯ MỤC MỚI,
không đụng artifact cũ — rollback an toàn). Chạy nền:
    python3 scripts/rescan_pattern_events_v3.py > /tmp/rescan_v3.log 2>&1
"""
import csv
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = Path.home() / "dev/market_stats_v2/market_cache/stock_ohlcv/latest.sqlite"
OUT = ROOT / "artifacts" / "scanner_v2_v3"

from scanner.run_bear_flag_db_source_parity_audit import _load_symbol_from_db, _symbols_in_db  # noqa: E402
from scanner.v2 import (  # noqa: E402
    ascending_triangles, descending_triangles, symmetrical_triangles,
    cup_with_handle, flags_experiment, high_tight_flags, pennants, pipes, gaps,
    inside_days, rectangles, dead_cat_bounce, horns, rounding, broadening_patterns,
    double_patterns, measured_moves, scallops, three_methods, three_peaks_valleys,
    bump_and_run, diamonds, falling_wedges, rising_wedges, harami,
)

# runner: (tên pattern, hàm scan(frame) -> list rows)
RUNNERS = {
    "harami": lambda f: harami.scan_symbol(f)[0],
    "inside_day": lambda f: inside_days.scan_symbol(f)[0],
    "bull_flags": lambda f: flags_experiment.scan_symbol(f)[0],
    "high_tight_flags": lambda f: high_tight_flags.scan_symbol(f)[0],
    "pennants": lambda f: pennants.scan_symbol(f)[0],
    "cup_with_handle": lambda f: cup_with_handle.scan_symbol(f, variant="cup_with_handle")[0],
    "cup_with_handle_inverted": lambda f: cup_with_handle.scan_symbol(f, variant="cup_with_handle_inverted")[0],
    "triangles_ascending": lambda f: ascending_triangles.scan_symbol(f)[0],
    "triangles_descending": lambda f: descending_triangles.scan_symbol(f)[0],
    "triangles_symmetrical": lambda f: symmetrical_triangles.scan_symbol(f)[0],
    "wedges_falling": lambda f: falling_wedges.scan_symbol(f)[0],
    "wedges_rising": lambda f: rising_wedges.scan_symbol(f)[0],
    "gaps": lambda f: gaps.scan_symbol(f)[0],
    "double_bottoms": lambda f: double_patterns.scan_symbol(f, family="double_bottoms")[0],
    "double_tops": lambda f: double_patterns.scan_symbol(f, family="double_tops")[0],
    "scallops_ascending": lambda f: scallops.scan_symbol(f, pattern_key="scallops_ascending")[0],
    "scallops_descending": lambda f: scallops.scan_symbol(f, pattern_key="scallops_descending")[0],
    "pipe_bottoms": lambda f: pipes.scan_symbol(f, pattern_key="pipe_bottoms")[0],
    "pipe_tops": lambda f: pipes.scan_symbol(f, pattern_key="pipe_tops")[0],
    "horn_bottoms": lambda f: horns.scan_symbol(f, pattern_key="horn_bottoms")[0],
    "horn_tops": lambda f: horns.scan_symbol(f, pattern_key="horn_tops")[0],
    "rounding_bottoms": lambda f: rounding.scan_symbol(f, pattern_key="rounding_bottoms")[0],
    "rounding_tops": lambda f: rounding.scan_symbol(f, pattern_key="rounding_tops")[0],
    "rectangle_bottoms": lambda f: rectangles.scan_symbol(f, pattern_key="rectangle_bottoms")[0],
    "rectangle_tops": lambda f: rectangles.scan_symbol(f, pattern_key="rectangle_tops")[0],
    "dead_cat_bounce": lambda f: dead_cat_bounce.scan_symbol(f, pattern_key="dead_cat_bounce")[0],
    "dead_cat_bounce_inverted": lambda f: dead_cat_bounce.scan_symbol(f, pattern_key="dead_cat_bounce_inverted")[0],
    "bump_and_run_reversal_bottoms": lambda f: bump_and_run.scan_symbol(f, pattern_key="bump_and_run_reversal_bottoms")[0],
    "bump_and_run_reversal_tops": lambda f: bump_and_run.scan_symbol(f, pattern_key="bump_and_run_reversal_tops")[0],
    "measured_move_up": lambda f: measured_moves.scan_symbol(f, pattern_key="measured_move_up")[0],
    "measured_move_down": lambda f: measured_moves.scan_symbol(f, pattern_key="measured_move_down")[0],
    "broadening_bottoms": lambda f: broadening_patterns.scan_symbol(f, pattern_key="broadening_bottoms")[0],
    "broadening_tops": lambda f: broadening_patterns.scan_symbol(f, pattern_key="broadening_tops")[0],
    "three_methods_rising": lambda f: three_methods.scan_symbol(f, pattern_key="rising_three_methods")[0],
    "three_methods_falling": lambda f: three_methods.scan_symbol(f, pattern_key="falling_three_methods")[0],
    "three_falling_peaks": lambda f: three_peaks_valleys.scan_symbol(f, pattern_key="three_falling_peaks")[0],
    "three_rising_valleys": lambda f: three_peaks_valleys.scan_symbol(f, pattern_key="three_rising_valleys")[0],
    "diamond_bottoms": lambda f: diamonds.scan_symbol(f, pattern_key="diamond_bottoms")[0],
    "diamond_tops": lambda f: diamonds.scan_symbol(f, pattern_key="diamond_tops")[0],
}


def write_events(pattern: str, rows: list[dict]) -> None:
    if not rows:
        return
    out_dir = OUT / pattern / "db_active"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(out_dir / "events.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    conn = sqlite3.connect(str(DB))
    symbols = _symbols_in_db(DB, None)
    print(f"symbols: {len(symbols)}  patterns: {len(RUNNERS)}", flush=True)
    buckets = {k: [] for k in RUNNERS}
    for i, sym in enumerate(symbols):
        if i % 200 == 0:
            print(f"  {i}/{len(symbols)}...", flush=True)
        try:
            frame = _load_symbol_from_db(conn, sym)
        except Exception:
            continue
        for name, fn in RUNNERS.items():
            try:
                rows = fn(frame)
                buckets[name].extend(rows or [])
            except Exception as exc:
                if i == 0:
                    print(f"  ⚠️ {name}: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
    conn.close()
    print("\n=== RE-SCAN V3 ===", flush=True)
    for name, rows in buckets.items():
        write_events(name, rows)
        print(f"  {name:<24} {len(rows):>7} events", flush=True)
    total = sum(len(v) for v in buckets.values())
    print(f"\n✅ Xong: {total} events, {len(buckets)} pattern → {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
