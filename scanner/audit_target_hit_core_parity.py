# -*- coding: utf-8 -*-
"""Release gate parity target_hit — Sol BLOCKER 3 (đợt A2, 16/08/2026).

Tuyên bố gate (nguyên văn Sol): "raw events target_hit == canonical core
target_hit, mismatch = 0 trên toàn bộ events được xuất bản".

Cơ chế: quét mọi cặp `events.csv` + `post_breakout_path.csv` dưới thư mục
artifacts; với MỖI event có đủ (breakout_price, target_price,
breakout_direction, target_hit không rỗng), tính lại bằng HÀM CHUẨN
`scanner.v2.target_hit_core.evaluate_target_hit` (multiple 1.0, adverse 5%,
precision chuẩn: target/breakout 4dp so giá path full precision — xem docstring
core) rồi so với cột `target_hit` trong events.csv. Event thiếu path hoặc thiếu
giá được đếm riêng (skipped) — KHÔNG tính mismatch vì events.csv của chúng cũng
rỗng/None theo cùng quy ước.

Exit code 0 khi mismatch == 0; 1 khi có mismatch (gate FAIL).
Kết quả chi tiết ghi JSON (mặc định scanner/audits/target_hit_core_parity.json).

Chạy:  python3 scanner/audit_target_hit_core_parity.py \
           [--artifacts-dir artifacts/scanner_v2] [--out scanner/audits/...json]
Test:  tests/test_target_hit_core_parity.py (synthetic qua _evaluate_detection).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.v2.target_hit_core import evaluate_target_hit  # noqa: E402

DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "scanner_v2"
DEFAULT_OUT = ROOT / "scanner" / "audits" / "target_hit_core_parity.json"
PATH_CSV_NAME = "post_breakout_path.csv"
EVENTS_CSV_NAME = "events.csv"
MAX_SAMPLE_MISMATCH = 20


def _horizon_of(event: pd.Series, group: pd.DataFrame, pattern_key: str | None = None) -> tuple[int, str]:
    """Số bars mà DETECTOR đã evaluate — gate phải cắt path đúng ngưỡng này.

    Thứ tự ưu tiên:
      1. Cột `evaluated_bars` trong events.csv — con số detector thực sự dùng.
      2. `measurement_registry.lookahead_bars(pattern_key)` — chuẩn đo lường
         mà detector gọi (đợt B 15/08/2026: path parity dirs bull/bear_flags
         ghi 120 bars trong khi detector evaluate 25 = registry; gate cắt sai
         120 → 34-47 mismatch giả).
      3. Độ dài path của chính event (fallback cuối — chỉ đúng khi module ghi
         path đúng horizon detector, đã chuẩn hoá sau đợt B).

    Trả (horizon, nguồn) — không bao giờ None (path luôn có ≥ 1 bar).
    """
    eb = event.get("evaluated_bars")
    if eb is not None and not pd.isna(eb):
        try:
            val = int(eb)
            if val > 0:
                return val, "evaluated_bars"
        except (TypeError, ValueError):
            pass
    if pattern_key:
        from scanner.v2.measurement_registry import lookahead_bars

        reg = lookahead_bars(pattern_key)
        if reg:
            return int(reg), "registry"
    return int(len(group)), "path_length"


def _direction_of(event: pd.Series) -> int:
    raw = str(event.get("breakout_direction") or "").strip().lower()
    if raw in ("up", "1", "bull", "bottom"):
        return 1
    if raw in ("down", "-1", "bear", "top"):
        return -1
    return 1 if float(event["target_price"]) >= float(event["breakout_price"]) else -1


def _truthy(raw) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("true", "1", "1.0", "yes")


def run_parity_check(artifacts_dir: Path, published_only: bool = False) -> dict:
    """Quét mọi cặp events+path dưới `artifacts_dir`, trả summary parity.

    published_only=True (đợt B, 16/08/2026): chỉ kiểm các cặp events.csv thuộc
    EVENT_SOURCES của `rebuild_source_guided_final_chapters` — tức events THỰC
    SỰ nuôi sách. Cổng release của Sol nói "mismatch = 0 trên toàn bộ events
    được XUẤT BẢN"; các thư mục grid/smoke/thử nghiệm trong artifacts là di
    sản nghiên cứu, không xuất bản — chạy mặc định (full) để tham khảo, chạy
    --published-only làm cổng phát hành.
    """
    artifacts_dir = Path(artifacts_dir)
    key_by_path: dict[Path, str] = {}
    if published_only:
        from scanner.rebuild_source_guided_final_chapters import DOUBLE_VARIANTS, EVENT_SOURCES

        published_events = set()
        for key, (events_path, _filters) in EVENT_SOURCES.items():
            resolved = (ROOT / Path(events_path)).resolve()
            published_events.add(resolved)
            key_by_path[resolved] = key
        # Đợt B (15/08/2026): 8 chương sách double variants (AA/AE/EA/EE × tops/bottoms)
        # đọc CHUNG 2 events.csv gốc này — cũng là events xuất bản, phải qua gate.
        for base, _variant in DOUBLE_VARIANTS.values():
            resolved = (ROOT / f"artifacts/scanner_v2/double_pattern_family/{base}/db_active/events.csv").resolve()
            published_events.add(resolved)
            key_by_path.setdefault(resolved, base)
    else:
        published_events = None
    pairs: list[tuple[Path, Path]] = []
    for events_path in sorted(artifacts_dir.rglob(EVENTS_CSV_NAME)):
        if published_events is not None and events_path.resolve() not in published_events:
            continue
        path_csv = events_path.parent / PATH_CSV_NAME
        if path_csv.exists():
            pairs.append((events_path, path_csv))

    total_events = 0
    compared = 0
    skipped_no_path = 0
    skipped_blank_hit = 0
    skipped_missing_fields = 0
    skipped_unknown_horizon = 0
    mismatch = 0
    mismatch_reasons: dict[str, int] = {}
    samples: list[dict] = []
    per_dir: dict[str, dict] = {}

    for events_path, path_csv in pairs:
        rel = str(events_path.parent.relative_to(artifacts_dir))
        events = pd.read_csv(events_path, low_memory=False)
        path_df = pd.read_csv(path_csv, low_memory=False)
        grouped: dict[str, pd.DataFrame] = {}
        if "event_id" in getattr(path_df, "columns", []):
            for event_id, group in path_df.groupby("event_id"):
                grouped[str(event_id)] = group.sort_values("bar_after_breakout")
        key_col = "event_id" if "event_id" in events.columns else ("detection_id" if "detection_id" in events.columns else None)
        dir_stat = {"events": 0, "compared": 0, "mismatch": 0}
        for _, event in events.iterrows():
            total_events += 1
            dir_stat["events"] += 1
            bp, tp = event.get("breakout_price"), event.get("target_price")
            raw_hit = event.get("target_hit")
            if pd.isna(bp) or pd.isna(tp):
                skipped_missing_fields += 1
                continue
            group = grouped.get(str(event.get(key_col))) if key_col else None
            if group is None or group.empty:
                skipped_no_path += 1
                continue
            if pd.isna(raw_hit):
                skipped_blank_hit += 1
                continue
            horizon, horizon_src = _horizon_of(event, group, pattern_key=key_by_path.get(events_path.resolve()))
            bars = pd.to_numeric(group["bar_after_breakout"], errors="coerce")
            window = group[bars <= horizon]
            if window.empty:
                skipped_unknown_horizon += 1
                continue
            core = evaluate_target_hit(
                pd.to_numeric(window["high"], errors="coerce").to_numpy(),
                pd.to_numeric(window["low"], errors="coerce").to_numpy(),
                float(bp),
                float(tp),
                _direction_of(event),
            )
            compared += 1
            dir_stat["compared"] += 1
            if bool(core["target_hit"]) != _truthy(raw_hit):
                mismatch += 1
                dir_stat["mismatch"] += 1
                # Phân loại nguyên nhân: artifacts cũ của family flags từng tính
                # target_hit = mfe_pct(2dp) >= target_dist_pct(2dp) — bug Sol
                # đích danh cho BARR, family flags dính cùng cơ chế.
                reason = "unexplained"
                mfe_c, dist_c = event.get("mfe_pct"), event.get("target_dist_pct")
                if mfe_c is not None and dist_c is not None and not pd.isna(mfe_c) and not pd.isna(dist_c):
                    legacy = float(mfe_c) >= float(dist_c)
                    if legacy == _truthy(raw_hit):
                        reason = "legacy_mfe_vs_dist_rounded"
                    elif abs(float(mfe_c) - float(dist_c)) < 1e-9:
                        # Cạnh rounding: mfe_pct == dist_pct sau round 2dp — code
                        # cũ so favorable FULL vs dist ROUNDED nên lệch biên
                        # (cùng cơ chế legacy, không tái hiện được bằng 2 số 2dp).
                        reason = "legacy_mfe_vs_dist_rounded_edge"
                mismatch_reasons[reason] = mismatch_reasons.get(reason, 0) + 1
                if len(samples) < MAX_SAMPLE_MISMATCH:
                    samples.append(
                        {
                            "dir": rel,
                            "event_id": str(event.get(key_col)),
                            "symbol": str(event.get("symbol")),
                            "breakout_price": float(bp),
                            "target_price": float(tp),
                            "direction": _direction_of(event),
                            "csv_target_hit": _truthy(raw_hit),
                            "core_target_hit": bool(core["target_hit"]),
                            "days_to_target": core["days_to_target"],
                            "reason": reason,
                        }
                    )
        if dir_stat["events"]:
            per_dir[rel] = dir_stat

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts_dir": str(artifacts_dir),
        "published_only": bool(published_only),
        "pairs_scanned": len(pairs),
        "total_events": total_events,
        "compared": compared,
        "mismatch": mismatch,
        "mismatch_reasons": dict(mismatch_reasons),
        "skipped_no_path": skipped_no_path,
        "skipped_blank_hit": skipped_blank_hit,
        "skipped_missing_fields": skipped_missing_fields,
        "skipped_unknown_horizon": skipped_unknown_horizon,
        "gate": "PASS" if mismatch == 0 else "FAIL",
        "sample_mismatches": samples,
        "per_dir": per_dir,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Release gate parity target_hit (Sol BLOCKER 3)")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--published-only",
        action="store_true",
        help="Chỉ kiểm các events.csv thuộc EVENT_SOURCES nuôi sách (release gate); mặc định quét toàn bộ artifacts (tham khảo).",
    )
    args = parser.parse_args(argv)

    summary = run_parity_check(Path(args.artifacts_dir), published_only=bool(args.published_only))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[parity] scanned {summary['pairs_scanned']} cặp events+path · {summary['total_events']} events")
    print(f"[parity] compared={summary['compared']} · mismatch={summary['mismatch']} "
          f"(skip: no_path={summary['skipped_no_path']}, blank_hit={summary['skipped_blank_hit']}, "
          f"missing_fields={summary['skipped_missing_fields']}, unknown_horizon={summary['skipped_unknown_horizon']})")
    if summary["mismatch"]:
        print(f"[parity] GATE FAIL — {summary['mismatch']} event lệch. Chi tiết: {out}")
        for s in summary["sample_mismatches"][:10]:
            print(f"  {s['dir']} · {s['symbol']} · event {s['event_id']}: csv={s['csv_target_hit']} core={s['core_target_hit']}")
        return 1
    print(f"[parity] GATE PASS — mismatch = 0. Chi tiết: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
