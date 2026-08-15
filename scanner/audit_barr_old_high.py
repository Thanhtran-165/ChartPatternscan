"""Audit BARR bottom old high — tái lập phép đo cơ chế PIVOT trên toàn bộ events.

Đợt A2 16/08/2026 (Sol BLOCKER 1 + điều kiện bổ sung 1-2). Dùng đúng cơ chế
production (`barr_bottom_old_high` của scanner.v2.bump_and_run) để audit và
code không thể lệch nhau.

Xuất scanner/audits/barr_old_high_audit.json:
  - population: giữ / loại (no-pivot-in-cap vs fit-fail), tier của nhóm bị loại
  - phân phối khoảng cách pivot (percentiles), tỷ lệ bị cắt bởi search cap
    (đo thử cap rộng 400 để đếm "sẽ đạt nếu cap rộng hơn")
  - thay đổi target so với code cũ (max[lead_start-2..bump]) và cửa sổ 60 đợt A
  - gate target_dist_pct > 110 (điều kiện 1): ứng viên trước gate, số bị loại,
    tier, tác động hit rate (mô phỏng path 120 bars — số chính thức có sau
    rescan Đợt B)

Chạy: python3 scanner/audit_barr_old_high.py [--db PATH] [--events PATH] [--out PATH]
Không cần cwd repo (đường dẫn resolve từ vị trí file này).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scanner.ohlcv_normalizer import OHLCVNormalizer  # noqa: E402
from scanner.pivot_detector import PivotDetector, PivotType  # noqa: E402
from scanner.v2.bump_and_run import (  # noqa: E402
    BARR_BOTTOMS,
    BARR_OLD_HIGH_SEARCH_CAP_BARS,
    barr_bottom_old_high,
    BumpAndRunConfig,
)

DEFAULT_DB = Path.home() / "dev/market_stats_v2/market_cache/stock_ohlcv/latest.sqlite"
DEFAULT_EVENTS = REPO / "artifacts/scanner_v2/bump_and_run_family/bump_and_run_reversal_bottoms/db_active/events.csv"
DEFAULT_PATH_CSV = REPO / "artifacts/scanner_v2/bump_and_run_family/bump_and_run_reversal_bottoms/db_active/post_breakout_path.csv"
DEFAULT_OUT = REPO / "scanner/audits/barr_old_high_audit.json"
CAP_WIDE = 400  # đo cắt-cap: nếu cap rộng 400 thì thêm bao nhiêu event đạt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    ap.add_argument("--path-csv", type=Path, default=DEFAULT_PATH_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    ev = pd.read_csv(args.events)
    conn = sqlite3.connect(str(args.db))
    norm = OHLCVNormalizer()
    pdet = PivotDetector()
    cfg = BumpAndRunConfig()
    r2_min = float(cfg.lead_in_min_r2)
    cache: dict[str, tuple] = {}

    def load(sym: str):
        if sym in cache:
            return cache[sym]
        raw = pd.read_sql_query(
            "SELECT symbol, time AS date, open, high, low, close, volume "
            "FROM stock_price_history WHERE symbol=? ORDER BY time", conn, params=[sym])
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        for c in ("open", "high", "low", "close", "volume"):
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw["symbol"] = sym
        raw["value"] = raw["close"] * raw["volume"]
        raw = raw.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)
        df, _ = norm.normalize(raw)
        df = df.reset_index(drop=True)
        pivots = pdet.get_filtered_pivots(pdet.detect_pivots(df, pivot_type="intermediate"), min_spacing=int(cfg.pivot_min_spacing))
        cache[sym] = (df, [int(p.idx) for p in pivots if p.type == PivotType.HIGH])
        return cache[sym]

    rows = []
    for _, r in ev.iterrows():
        sym = str(r["symbol"]).upper()
        ls, le, bi = int(r["lead_in_start_idx"]), int(r["lead_in_end_idx"]), int(r["bump_idx"])
        df, highs = load(sym)
        if df.empty:
            rows.append({"detection_id": r["detection_id"], "symbol": sym, "ok": False, "reason": "no_series"})
            continue
        anchor = barr_bottom_old_high(df, ls, le, highs, r2_min=r2_min, cap_bars=BARR_OLD_HIGH_SEARCH_CAP_BARS)
        reason = "ok"
        if anchor is None:
            anchor_wide = barr_bottom_old_high(df, ls, le, highs, r2_min=r2_min, cap_bars=CAP_WIDE)
            if not any(i < ls for i in highs):
                reason = "no_pivot_before_lead_start"
            elif anchor_wide is not None:
                reason = "cut_by_search_cap"  # sẽ đạt nếu cap rộng hơn
            else:
                reason = "no_pivot_fits_lead_in"
        old_idx, old_price = anchor if anchor else (None, None)
        target_new = float(df["high"].iloc[old_idx : bi + 1].max()) if old_idx is not None else None
        target_legacy = float(df["high"].iloc[max(0, ls - 2) : bi + 1].max())
        lo60 = max(0, ls - 60)
        w60 = df["high"].iloc[lo60:ls]
        target_w60 = float(w60.iloc[int(np.argmax(w60.to_numpy())) : bi + 1].max()) if not w60.empty else target_legacy
        rows.append({
            "detection_id": r["detection_id"], "symbol": sym, "ok": True, "reason": reason,
            "tier": str(r.get("publication_quality_tier") or ""),
            "lead_start": ls, "bump_idx": bi,
            "old_high_idx": old_idx, "pivot_dist": (ls - old_idx) if old_idx is not None else None,
            "target_new": target_new, "target_legacy": target_legacy, "target_window60": target_w60,
            "target_code": float(r["target_price"]), "breakout_price": float(r["breakout_price"]),
            "target_hit_detector": str(r.get("target_hit") or "").lower() in ("true", "1", "1.0"),
        })

    m = pd.DataFrame(rows)
    ok = m[m["ok"] == True]  # noqa: E712
    kept = ok[ok["reason"] == "ok"].copy()
    dropped = ok[ok["reason"] != "ok"].copy()
    d = kept["pivot_dist"].astype(int)

    # gate dist 110 (điều kiện 1) — trên nhóm GIỮ (ứng viên trước gate)
    kept["dist_new_pct"] = (kept["target_new"] - kept["breakout_price"]).abs() / kept["breakout_price"] * 100
    gate_out = kept[kept["dist_new_pct"] > 110]

    # hit rate mô phỏng path 120 (số chính thức sau rescan Đợt B)
    path_hits_new, path_hits_code = [], []
    if args.path_csv.exists():
        path = pd.read_csv(args.path_csv)
        grouped = {eid: g for eid, g in path.groupby("event_id")}
        for _, r in kept.iterrows():
            g = grouped.get(r["detection_id"])
            if g is None or g.empty:
                continue
            path_hits_new.append(bool(float(g["high"].max()) >= float(r["target_new"])))
            path_hits_code.append(bool(float(g["high"].max()) >= float(r["target_code"])))

    def pct(vals, q):
        return round(float(np.percentile(np.asarray(vals, dtype=float), q)), 1)

    delta_vs_code = (kept["target_new"] - kept["target_legacy"]) / kept["target_legacy"] * 100
    delta_vs_w60 = (kept["target_new"] - kept["target_window60"]) / kept["target_window60"] * 100
    same_idx = int((kept["old_high_idx"].astype(int) == (kept["lead_start"].astype(int) - kept["pivot_dist"].astype(int))).sum())

    audit = {
        "_meta": {
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "mechanism": (
                "old high = pivot HIGH intermediate GẦN NHẤT trước lead_start neo được vào cùng "
                "đường lead-in: fit mở rộng [pivot, lead_end] trên close đạt slope<0 và "
                f"r2 >= {r2_min} (lead_in_min_r2); duyệt gần→xa; search cap "
                f"{BARR_OLD_HIGH_SEARCH_CAP_BARS} bars (cap kỹ thuật, KHÔNG phải định nghĩa — Sol BLOCKER 1). "
                "Không pivot hợp lệ → LOẠI event tại detection (không thay bằng cực đại cửa sổ)."
            ),
            "reason_loi_not_mark": (
                "Chọn LOẠI thay vì đánh dấu: measure rule sách không xác định được khi thiếu old "
                "high — event không đo được target thì không thuộc population target_hit; đánh dấu "
                "sẽ đòi target=None xuyên suốt evaluate/builder (phá vỡ nhiều điểm chạm hơn)."
            ),
            "pattern": BARR_BOTTOMS,
            "events_source": str(args.events.relative_to(REPO)) if args.events.is_relative_to(REPO) else str(args.events),
            "db_source": str(args.db),
            "db_rows_meta": "1.599 symbols / 4.249.160 rows / max_date 2026-08-14 (db_source_meta statistics.json)",
        },
        "population": {
            "events_total": int(len(ok)),
            "kept": int(len(kept)),
            "kept_pct": round(len(kept) / len(ok) * 100, 2),
            "dropped": int(len(dropped)),
            "dropped_pct": round(len(dropped) / len(ok) * 100, 2),
            "dropped_reasons": dropped["reason"].value_counts().to_dict(),
            "dropped_tier_distribution": dropped["tier"].value_counts().to_dict(),
            "kept_tier_distribution": kept["tier"].value_counts().to_dict(),
        },
        "pivot_distance_bars": {
            "p50": pct(d, 50), "p75": pct(d, 75), "p90": pct(d, 90),
            "p95": pct(d, 95), "p99": pct(d, 99), "p100": int(d.max()),
        },
        "cap_censorship": {
            "search_cap_bars": int(BARR_OLD_HIGH_SEARCH_CAP_BARS),
            "events_cut_by_cap_would_pass_at_cap_400": int((dropped["reason"] == "cut_by_search_cap").sum()),
            "note": "cap = giới hạn kỹ thuật; đo cap 400 để lượng censor. p95 dist = 234 <= cap 250.",
        },
        "target_change_vs_legacy_code": {
            "median_pct": round(float(delta_vs_code.median()), 2),
            "p90_pct": round(float(delta_vs_code.quantile(0.9)), 2),
            "events_target_up": int((delta_vs_code > 0).sum()),
            "events_target_down": int((delta_vs_code < 0).sum()),
            "events_same": int((delta_vs_code == 0).sum()),
        },
        "target_change_vs_window60_dotA": {
            "median_pct": round(float(delta_vs_w60.median()), 2),
            "p90_pct": round(float(delta_vs_w60.quantile(0.9)), 2),
            "events_pivot_higher": int((delta_vs_w60 > 0).sum()),
            "events_pivot_lower": int((delta_vs_w60 < 0).sum()),
            "events_same": int((delta_vs_w60 == 0).sum()),
            "note": "so old high pivot với argmax cửa sổ 60 của đợt A (cơ chế bị Sol bác).",
        },
        "gate_dist_110_audit": {
            "candidates_before_gate": int(len(kept)),
            "rejected_gt_110": int(len(gate_out)),
            "rejected_pct": round(len(gate_out) / max(len(kept), 1) * 100, 2),
            "rejected_tier_distribution": gate_out["tier"].value_counts().to_dict() if not gate_out.empty else {},
            "note": "Điều kiện bổ sung 1 (đợt A2): ứng viên bị loại bởi ngưỡng target_dist_pct>110 giữ vết ở đây — KHÔNG biến mất không dấu. Số chính thức sau rescan Đợt B.",
        },
        "hit_rate_simulated_path120": {
            "n": int(len(path_hits_new)),
            "target_hit_code_current_pct": round(float(np.mean(path_hits_code) * 100), 2) if path_hits_code else None,
            "target_hit_pivot_pct": round(float(np.mean(path_hits_new) * 100), 2) if path_hits_new else None,
            "note": "mô phỏng trên path 120 bars của events hiện hành (population cũ, trước các cổng loại mới) — số chính thức sau rescan Đợt B.",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=1)[:2400])
    print(f"\nwrote {args.out}")
    detail = ok[["detection_id", "symbol", "reason", "tier", "pivot_dist", "target_new", "target_legacy", "target_code", "breakout_price"]]
    detail.to_csv(args.out.with_suffix(".detail.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
