#!/usr/bin/env python3
"""M2 verify TOÀN DB (1599 mã) — số failure_busted ổn định cho K3-2.

Chạy nền: python3 scripts/verify_m2_full.py > /tmp/verify_m2_full.log 2>&1
"""
import json
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = Path.home() / "dev/market_stats_v2/market_cache/stock_ohlcv/latest.sqlite"

from scanner.run_bear_flag_db_source_parity_audit import _load_symbol_from_db, _symbols_in_db  # noqa: E402
from scanner.v2 import (  # noqa: E402
    ascending_triangles, cup_with_handle, flags_experiment, pipes, gaps,
    inside_days, rectangles, dead_cat_bounce, horns, rounding, broadening_patterns,
    double_patterns, measured_moves, scallops, pennants, three_methods,
)


def summarize(name: str, detections: list[dict]) -> dict:
    evals = [r for r in detections if r.get("mfe_pct") is not None]
    n = len(evals)
    if n == 0:
        return {"pattern": name, "events": len(detections), "evaluated": 0}
    busted = [r for r in evals if r.get("failure_busted")]
    days = [r["days_to_bust"] for r in busted if r.get("days_to_bust")]
    return {
        "pattern": name,
        "events": len(detections),
        "evaluated": n,
        "failure_5pct_pct": round(100.0 * sum(bool(r.get("failure_5pct")) for r in evals) / n, 2),
        "weak_move_5pct_pct": round(100.0 * sum(bool(r.get("weak_move_5pct")) for r in evals) / n, 2),
        "failure_busted_pct": round(100.0 * len(busted) / n, 2),
        "busted_n": len(busted),
        "median_days_to_bust": round(float(statistics.median(days)), 1) if days else None,
        "median_target_dist_pct": round(float(statistics.median(
            float(r["target_dist_pct"]) for r in evals if r.get("target_dist_pct") is not None)), 2),
        "median_mfe_pct": round(float(statistics.median(float(r["mfe_pct"]) for r in evals)), 2),
    }


def main() -> int:
    conn = sqlite3.connect(str(DB))
    symbols = _symbols_in_db(DB, None)
    print(f"symbols: {len(symbols)}", flush=True)
    runners = {
        "bull_flags": lambda f: flags_experiment.scan_symbol(f)[0],
        "cup_with_handle": lambda f: cup_with_handle.scan_symbol(f, variant="cup_with_handle")[0],
        "triangles_ascending": lambda f: ascending_triangles.scan_symbol(f)[0],
        "pipe_bottoms": lambda f: pipes.scan_symbol(f, pattern_key="pipe_bottoms")[0],
        "gaps": lambda f: gaps.scan_symbol(f)[0],
        "inside_day": lambda f: inside_days.scan_symbol(f)[0],
        "rectangle_bottoms": lambda f: rectangles.scan_symbol(f, pattern_key="rectangle_bottoms")[0],
        "dead_cat_bounce": lambda f: dead_cat_bounce.scan_symbol(f, pattern_key="dead_cat_bounce")[0],
        "horn_bottoms": lambda f: horns.scan_symbol(f, pattern_key="horn_bottoms")[0],
        "rounding_bottoms": lambda f: rounding.scan_symbol(f, pattern_key="rounding_bottoms")[0],
        "broadening_bottoms": lambda f: broadening_patterns.scan_symbol(f, pattern_key="broadening_bottoms")[0],
        "double_bottoms": lambda f: double_patterns.scan_symbol(f, family="double_bottoms")[0],
        "measured_move_up": lambda f: measured_moves.scan_symbol(f, pattern_key="measured_move_up")[0],
        "scallops_ascending": lambda f: scallops.scan_symbol(f, pattern_key="scallops_ascending")[0],
        "pennants": lambda f: pennants.scan_symbol(f)[0],
        "rising_three_methods": lambda f: three_methods.scan_symbol(f, pattern_key="rising_three_methods")[0],
    }
    buckets = {k: [] for k in runners}
    for i, sym in enumerate(symbols):
        if i % 200 == 0:
            print(f"  {i}/{len(symbols)}...", flush=True)
        try:
            frame = _load_symbol_from_db(conn, sym)
        except Exception:
            continue
        for name, fn in runners.items():
            try:
                rows = fn(frame)
                buckets[name].extend(rows or [])
            except Exception as exc:
                if i == 0:
                    print(f"  ⚠️ {name}: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
    conn.close()
    results = [summarize(k, v) for k, v in buckets.items()]
    print("\n=== VERIFY M2 TOÀN DB ===")
    print(f"{'pattern':<24}{'events':>7}{'eval':>6}{'f5%':>8}{'weak%':>8}{'busted%':>9}{'n_bust':>7}{'med_days':>9}{'med_tgt%':>9}{'med_mfe':>8}")
    for r in results:
        if r.get("evaluated", 0) == 0:
            print(f"{r['pattern']:<24}{r['events']:>7}  (không có event evaluated)")
            continue
        print(
            f"{r['pattern']:<24}{r['events']:>7}{r['evaluated']:>6}"
            f"{r['failure_5pct_pct']:>8}{r['weak_move_5pct_pct']:>8}{r['failure_busted_pct']:>9}"
            f"{r['busted_n']:>7}{str(r['median_days_to_bust']):>9}{r['median_target_dist_pct']:>9}{r['median_mfe_pct']:>8}"
        )
    out = Path("/tmp/verify_m2_full.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Đã lưu {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
