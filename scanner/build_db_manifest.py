# -*- coding: utf-8 -*-
"""Build DB manifest — Sol HIGH-3 (đợt B, 16/08/2026).

Manifest khai báo ĐÚNG phạm vi dữ liệu nguồn của toàn bộ bộ scan xuất bản:
đường dẫn + SHA-256 file DB, nguồn/provider (đọc từ stock_ohlcv_meta +
cột source), số symbols/rows/min-max date, phạm vi adjusted ĐƯỢC PHÉP TUYÊN
BỐ (provider-adjusted, chưa có factor audit — theo data_gate_audit), cách xử
lí close=0 (delisted/halted bị detector loại), và issue đang chặn refresh
(ISS-001: updater chỉ UPSERT không DELETE hàng cũ).

Chạy:  python3 scanner/build_db_manifest.py [--db <path>] [--out <json>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DB = Path("/Users/bobo/dev/market_stats_v2/market_cache/stock_ohlcv/latest.sqlite")
DEFAULT_OUT = ROOT / "artifacts" / "scanner_v2" / "db_manifest.json"
DATA_GATE_AUDIT = ROOT / "artifacts" / "scanner_v2" / "bull_flags_localized" / "data_gate_audit.json"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _git(field: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", f"--git-dir={ROOT / '.git'}", field],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_manifest(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(time), MAX(time) FROM stock_price_history").fetchone()
        n_rows, n_symbols, min_date, max_date = rows
        n_zero = conn.execute("SELECT COUNT(*) FROM stock_price_history WHERE close IS NULL OR close <= 0").fetchone()[0]
        n_zero_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM stock_price_history WHERE close <= 0").fetchone()[0]
        refresh_raw = None
        try:
            refresh_raw = conn.execute("SELECT value FROM stock_ohlcv_meta WHERE key='market_stats_latest_refresh'").fetchone()[0]
        except sqlite3.Error:
            pass
        source_providers = [
            {"source": s or "(null)", "rows": n}
            for s, n in conn.execute(
                "SELECT COALESCE(source,'(null)'), COUNT(*) FROM stock_price_history "
                "WHERE source IS NOT NULL GROUP BY source ORDER BY 2 DESC LIMIT 5"
            )
        ]
    finally:
        conn.close()

    refresh = json.loads(refresh_raw) if refresh_raw else None
    gate = {}
    if DATA_GATE_AUDIT.exists():
        gate_doc = json.loads(DATA_GATE_AUDIT.read_text(encoding="utf-8"))
        corp = next((g for g in gate_doc.get("gates", []) if g.get("gate_id") == "corporate_action_audit"), {})
        pit = next((g for g in gate_doc.get("gates", []) if g.get("gate_id") == "point_in_time_universe"), {})
        gate = {
            "standard_version": gate_doc.get("standard_version"),
            "corporate_action_audit_status": corp.get("status"),
            "price_basis": corp.get("evidence", {}).get("price_basis"),
            "adjustment": corp.get("evidence", {}).get("adjustment"),
            "adjustment_guardrail": corp.get("evidence", {}).get("adjustment_guardrail"),
            "point_in_time_universe_status": pit.get("status"),
            "membership_mode": pit.get("evidence", {}).get("membership_mode"),
        }

    stat = db_path.stat()
    return {
        "manifest_version": "db_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Khai báo phạm vi dữ liệu nguồn của bộ scan xuất bản (Sol HIGH-3) — mọi phát biểu về dữ liệu trong sách phải không rộng hơn manifest này",
        "database": {
            "path": str(db_path),
            "file_size_bytes": stat.st_size,
            "sha256": _sha256(db_path),
            "symbols": int(n_symbols),
            "rows": int(n_rows),
            "min_date": str(min_date),
            "max_date": str(max_date),
        },
        "source_provider": {
            "library": "vnstock_data (Market Quote)",
            "primary_source": "VCI",
            "fallback_sources": ["VND", "KBS", "MAS"],
            "sample_source_tags": source_providers,
            "last_refresh_meta": refresh,
            "updater": "market_stats/update_latest_stock_ohlcv.py (repo ~/dev/market_stats_v2)",
        },
        "adjusted_claim_allowed": {
            "claim": "OHLCV do nhà cung cấp mô tả là ĐÃ ĐIỀU CHỈNH toàn tuyến (provider-adjusted); dự án KHÔNG có factor audit độc lập để kiểm chứng hệ số điều chỉnh",
            "price_basis": gate.get("price_basis", "provider_adjusted_ohlcv"),
            "adjustment_status": gate.get("adjustment", "provider_adjusted_without_factor_audit"),
            "guardrail": gate.get(
                "adjustment_guardrail",
                "OHLCV lấy từ vnstock_data; nguồn mô tả dữ liệu đã điều chỉnh nhưng không trả kèm hệ số kiểm chứng.",
            ),
            "point_in_time_universe": gate.get("membership_mode", "current_snapshot"),
            "downgraded_statements_required": [
                "KHÔNG nói 'full-reload toàn bộ lịch sử adjusted cho mọi mã' — refresh gần nhất chỉ chạm 1.348/1.599 mã (xem last_refresh_meta).",
                "KHÔNG nói 'toàn bộ dữ liệu đã kiểm chứng điều chỉnh' — chỉ provider-adjusted chưa audit factor.",
                "Universe là available-series descriptive + membership snapshot hiện tại, không phải point-in-time.",
            ],
        },
        "close_zero_handling": {
            "rows_close_le_zero": int(n_zero),
            "symbols_affected": int(n_zero_symbols),
            "treatment": "Hàng close<=0 (đánh dấu delisted/halted của nguồn) bị các detector loại tại chỗ khi đọc chuỗi giá (vd gaps.py close<=0, inside_days prev_close==0); không đi vào events xuất bản.",
        },
        "open_issues": [
            {
                "id": "ISS-001",
                "title": "update_latest_stock_ohlcv.py chỉ UPSERT, không DELETE hàng cũ",
                "path": "scanner/issues/ISS-001-full-reload-replace-stale-rows.md",
                "impact": "Hàng lỗi/quá hạn có thể tồn tại sau refresh; CHẶN refresh kế tiếp cho tới khi xử lí xong.",
            }
        ],
        "rescan_context": {
            "branch": _git("rev-parse --abbrev-ref HEAD"),
            "commit": _git("rev-parse --short HEAD"),
            "note": "Đợt B rescan toàn thị trường bằng code mới (target_hit_core full precision) — mọi events.csv xuất bản tái sinh từ DB ở trên.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DB manifest (Sol HIGH-3)")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    manifest = build_manifest(Path(args.db))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[db_manifest] {out}")
    print(f"  sha256={manifest['database']['sha256'][:16]}… · {manifest['database']['symbols']} symbols · {manifest['database']['rows']} rows · max {manifest['database']['max_date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
