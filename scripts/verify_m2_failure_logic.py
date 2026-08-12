#!/usr/bin/env python3
"""M2 verify: chạy scan thật trên DB chuẩn, kiểm failure_busted/weak_move_5pct/days_to_bust/target_dist_pct.

Mục tiêu (K3-1):
- bull_flags failure_busted ≈ 5,5% ± 3%
- cup_with_handle failure_busted ≈ 5%
- weak_move_5pct phải bằng failure_5pct mọi row (chỉ đổi tên)
- days_to_bust phân phối hợp lý (median > 0 khi có busted)
"""
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = Path.home() / "dev/market_stats_v2/market_cache/stock_ohlcv/latest.sqlite"
SYMBOLS = ["VCB", "CTD", "FPT", "VNM", "HPG", "ACB", "BID", "MWG", "TCB", "SSI"]

from scanner.run_bear_flag_db_source_parity_audit import _load_symbol_from_db  # noqa: E402
from scanner.v2 import ascending_triangles as at  # noqa: E402
from scanner.v2 import cup_with_handle as cup  # noqa: E402
from scanner.v2 import flags_experiment as fe  # noqa: E402
from scanner.v2 import pipes as pipes_mod  # noqa: E402
from scanner.v2 import gaps as gaps_mod  # noqa: E402


def summarize(name: str, detections: list[dict]) -> dict:
    evals = [r for r in detections if r.get("mfe_pct") is not None]
    n = len(evals)
    if n == 0:
        return {"pattern": name, "events": len(detections), "evaluated": 0}
    cons = all(
        bool(r.get("weak_move_5pct")) == bool(r.get("failure_5pct"))
        for r in evals
        if r.get("weak_move_5pct") is not None and r.get("failure_5pct") is not None
    )
    busted = [r for r in evals if r.get("failure_busted")]
    days = [r["days_to_bust"] for r in busted if r.get("days_to_bust")]
    return {
        "pattern": name,
        "events": len(detections),
        "evaluated": n,
        "failure_5pct_pct": round(100.0 * sum(bool(r.get("failure_5pct")) for r in evals) / n, 2),
        "weak_move_5pct_pct": round(100.0 * sum(bool(r.get("weak_move_5pct")) for r in evals) / n, 2),
        "weak==failure_5pct": cons,
        "failure_busted_pct": round(100.0 * len(busted) / n, 2),
        "busted_n": len(busted),
        "median_days_to_bust": round(float(statistics.median(days)), 1) if days else None,
        "median_target_dist_pct": round(float(statistics.median(
            float(r["target_dist_pct"]) for r in evals if r.get("target_dist_pct") is not None)), 2),
        "median_mfe_pct": round(float(statistics.median(float(r["mfe_pct"]) for r in evals)), 2),
    }


def main() -> int:
    if not DB.exists():
        print(f"❌ Không thấy DB: {DB}")
        return 1
    conn = sqlite3.connect(str(DB))
    results: list[dict] = []
    try:
        frames = {s: _load_symbol_from_db(conn, s) for s in SYMBOLS}
        for s, f in frames.items():
            print(f"  {s}: {len(f)} rows ({f['date'].iloc[0]} → {f['date'].iloc[-1]})")
    finally:
        conn.close()

    def collect(name: str, fn) -> list[dict]:
        rows: list[dict] = []
        for sym in SYMBOLS:
            try:
                detections, _ = fn(frames[sym])
                rows.extend(detections or [])
            except Exception as exc:
                print(f"  ⚠️ {name} {sym}: {exc}")
        return rows

    # 1) bull_flags (flags_experiment) — mục tiêu ≈5,5%±3
    det = collect("flags", lambda f: fe.scan_symbol(f))
    results.append(summarize("bull_flags (flags_experiment)", det))

    # 2) cup_with_handle — mục tiêu ≈5%
    det = collect("cup", lambda f: cup.scan_symbol(f, variant="cup_with_handle"))
    results.append(summarize("cup_with_handle", det))

    # 3) ascending_triangles
    det = collect("tri_asc", lambda f: at.scan_symbol(f))
    results.append(summarize("triangles_ascending", det))

    # 4) pipes (weekly — lookahead_weeks)
    det = collect("pipes", lambda f: pipes_mod.scan_symbol(f, pattern_key="pipe_bottoms"))
    results.append(summarize("pipe_bottoms (weekly)", det))

    # 5) gaps
    det = collect("gaps", lambda f: gaps_mod.scan_symbol(f))
    results.append(summarize("gaps", det))

    print("\n=== KẾT QUẢ VERIFY M2 ===")
    print(f"{'pattern':<30}{'events':>7}{'eval':>6}{'f5%':>8}{'weak%':>8}{'==old':>7}{'busted%':>9}{'n_bust':>7}{'med_days':>9}{'med_tgt%':>9}{'med_mfe':>8}")
    for r in results:
        if r.get("evaluated", 0) == 0:
            print(f"{r['pattern']:<30}{r['events']:>7}  (không có event evaluated)")
            continue
        print(
            f"{r['pattern']:<30}{r['events']:>7}{r['evaluated']:>6}"
            f"{r['failure_5pct_pct']:>8}{r['weak_move_5pct_pct']:>8}{str(r['weak==failure_5pct']):>7}"
            f"{r['failure_busted_pct']:>9}{r['busted_n']:>7}{str(r['median_days_to_bust']):>9}"
            f"{r['median_target_dist_pct']:>9}{r['median_mfe_pct']:>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
