"""
Run a full (or partial) scan over the source OHLCV SQLite DB and persist results
to a separate results DB with run_id tracking.

Usage examples:
  python3 scanner/run_full_scan.py --limit 50
  python3 scanner/run_full_scan.py --patterns double_tops,head_and_shoulders_top
  python3 scanner/run_full_scan.py --results-db scan_results/pattern_scans.sqlite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Allow running as a script (imports are local-module style in this folder)
import sys

SCANNER_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCANNER_DIR)

from pattern_scanner import PatternScanner, ScannerConfig  # noqa: E402
from post_breakout_analyzer import (  # noqa: E402
    PostBreakoutEvaluator,
    EvaluationConfig,
)
from results_db import (  # noqa: E402
    connect_results_db,
    ensure_schema,
    generate_run_id,
    insert_detections,
    insert_post_breakout_results,
    record_error,
    upsert_run,
    upsert_run_statistics,
)


SOURCE_TABLE = "stock_price_history"


def _md5_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _parse_patterns(s: Optional[str], available: Sequence[str]) -> List[str]:
    if not s:
        return list(available)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    unknown = [p for p in parts if p not in available]
    if unknown:
        raise SystemExit(f"Unknown patterns: {unknown}. Available: {sorted(available)}")
    return parts


def _load_symbols(conn: sqlite3.Connection, min_rows: int, limit: Optional[int]) -> List[str]:
    df = pd.read_sql_query(
        """
        SELECT symbol, COUNT(*) AS cnt
        FROM stock_price_history
        GROUP BY symbol
        HAVING cnt >= ?
        ORDER BY symbol
        """,
        conn,
        params=[min_rows],
    )
    symbols = df["symbol"].tolist()
    if limit is not None:
        symbols = symbols[: int(limit)]
    return symbols


def _load_symbol_df(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT symbol, time as date, open, high, low, close, volume "
        "FROM stock_price_history WHERE symbol = ? ORDER BY time",
        conn,
        params=[symbol],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def _compute_run_statistics(conn: sqlite3.Connection, run_id: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"run_id": run_id}

    def _scalar(q: str, params: Tuple[Any, ...] = ()) -> Any:
        cur = conn.execute(q, params)
        row = cur.fetchone()
        return row[0] if row else None

    stats["detections_total"] = _scalar(
        "SELECT COUNT(*) FROM pattern_detections WHERE run_id = ?",
        (run_id,),
    )
    stats["detections_with_breakout"] = _scalar(
        "SELECT COUNT(*) FROM pattern_detections WHERE run_id = ? AND breakout_date IS NOT NULL",
        (run_id,),
    )
    stats["evaluations_total"] = _scalar(
        "SELECT COUNT(*) FROM post_breakout_results WHERE run_id = ?",
        (run_id,),
    )

    # Overall rates (only rows with non-null metric participate)
    def _rate(col: str) -> Optional[float]:
        v = _scalar(
            f"SELECT AVG({col}) FROM post_breakout_results WHERE run_id = ? AND {col} IS NOT NULL",
            (run_id,),
        )
        return None if v is None else float(v) * 100.0

    stats["bust_failure_rate_5pct"] = _rate("bust_failure_5pct")
    stats["bust_failure_rate_10pct"] = _rate("bust_failure_10pct")
    stats["boundary_invalidation_rate_pct"] = _rate("boundary_invalidated")
    stats["throwback_pullback_rate_pct"] = _rate("throwback_pullback_occurred")
    stats["target_achievement_rate_intraday_pct"] = _rate("target_achieved_intraday")
    stats["target_achievement_rate_close_pct"] = _rate("target_achieved_close")

    stats["avg_days_to_ultimate"] = _scalar(
        "SELECT AVG(days_to_ultimate) FROM post_breakout_results WHERE run_id = ? AND days_to_ultimate IS NOT NULL",
        (run_id,),
    )
    stats["avg_mfe_pct"] = _scalar(
        "SELECT AVG(max_favorable_excursion_pct) FROM post_breakout_results "
        "WHERE run_id = ? AND max_favorable_excursion_pct IS NOT NULL AND ABS(max_favorable_excursion_pct) < 1e308",
        (run_id,),
    )
    stats["avg_mae_pct"] = _scalar(
        "SELECT AVG(max_adverse_excursion_pct) FROM post_breakout_results "
        "WHERE run_id = ? AND max_adverse_excursion_pct IS NOT NULL AND ABS(max_adverse_excursion_pct) < 1e308",
        (run_id,),
    )

    # By pattern
    by_pattern: Dict[str, Any] = {}
    for (pattern_name,) in conn.execute(
        "SELECT DISTINCT pattern_name FROM post_breakout_results WHERE run_id = ? ORDER BY pattern_name",
        (run_id,),
    ).fetchall():
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS n,
                AVG(throwback_pullback_occurred) * 100.0,
                AVG(target_achieved_intraday) * 100.0,
                AVG(bust_failure_5pct) * 100.0,
                AVG(boundary_invalidated) * 100.0,
                AVG(CASE WHEN max_favorable_excursion_pct IS NOT NULL AND ABS(max_favorable_excursion_pct) < 1e308 THEN max_favorable_excursion_pct END),
                AVG(CASE WHEN max_adverse_excursion_pct IS NOT NULL AND ABS(max_adverse_excursion_pct) < 1e308 THEN max_adverse_excursion_pct END),
                AVG(days_to_ultimate)
            FROM post_breakout_results
            WHERE run_id = ? AND pattern_name = ?
            """,
            (run_id, pattern_name),
        ).fetchone()
        if not row:
            continue
        by_pattern[pattern_name] = {
            "count": int(row[0]),
            "throwback_pullback_rate_pct": float(row[1]) if row[1] is not None else None,
            "target_achievement_rate_intraday_pct": float(row[2]) if row[2] is not None else None,
            "bust_failure_rate_5pct": float(row[3]) if row[3] is not None else None,
            "boundary_invalidation_rate_pct": float(row[4]) if row[4] is not None else None,
            "avg_mfe_pct": float(row[5]) if row[5] is not None else None,
            "avg_mae_pct": float(row[6]) if row[6] is not None else None,
            "avg_days_to_ultimate": float(row[7]) if row[7] is not None else None,
        }

    stats["by_pattern"] = by_pattern

    # By variant (double tops)
    by_variant: Dict[str, Any] = {}
    for (variant,) in conn.execute(
        """
        SELECT DISTINCT variant
        FROM post_breakout_results
        WHERE run_id = ? AND pattern_name = 'double_tops' AND variant IS NOT NULL
        ORDER BY variant
        """,
        (run_id,),
    ).fetchall():
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS n,
                AVG(CASE WHEN max_favorable_excursion_pct IS NOT NULL AND ABS(max_favorable_excursion_pct) < 1e308 THEN max_favorable_excursion_pct END),
                AVG(CASE WHEN max_adverse_excursion_pct IS NOT NULL AND ABS(max_adverse_excursion_pct) < 1e308 THEN max_adverse_excursion_pct END),
                AVG(bust_failure_5pct) * 100.0,
                AVG(target_achieved_intraday) * 100.0
            FROM post_breakout_results
            WHERE run_id = ? AND pattern_name = 'double_tops' AND variant = ?
            """,
            (run_id, variant),
        ).fetchone()
        if not row:
            continue
        by_variant[variant] = {
            "count": int(row[0]),
            "avg_mfe_pct": float(row[1]) if row[1] is not None else None,
            "avg_mae_pct": float(row[2]) if row[2] is not None else None,
            "bust_failure_rate_5pct": float(row[3]) if row[3] is not None else None,
            "target_achievement_rate_intraday_pct": float(row[4]) if row[4] is not None else None,
        }

    if by_variant:
        stats["double_tops_by_variant"] = by_variant

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default=os.path.join(os.path.abspath(os.path.join(SCANNER_DIR, "..")), "vietnam_stocks.db"),
        help="Path to source price DB (SQLite).",
    )
    parser.add_argument(
        "--results-db",
        default=os.path.join(os.path.abspath(os.path.join(SCANNER_DIR, "..")), "scan_results", "pattern_scans.sqlite"),
        help="Path to results DB (SQLite). Will be created if missing.",
    )
    parser.add_argument("--run-id", default=None, help="Optional run_id. If omitted, one is generated.")
    parser.add_argument("--min-rows", type=int, default=500, help="Minimum bars per symbol to scan.")
    parser.add_argument(
        "--patterns",
        default=None,
        help="Comma-separated list of patterns to scan (default: all available in scanner).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit symbols (for testing).")
    parser.add_argument("--commit-every", type=int, default=25, help="Commit every N symbols.")
    parser.add_argument("--notes", default=None, help="Optional notes saved with the run.")
    args = parser.parse_args()

    source_db_path = os.path.abspath(args.db)
    results_db_path = os.path.abspath(args.results_db)
    run_id = args.run_id or generate_run_id(prefix="scan")

    scanner = PatternScanner(ScannerConfig())
    available_patterns = list(scanner.scanners.keys())
    patterns = _parse_patterns(args.patterns, available_patterns)

    eval_config = EvaluationConfig(
        lookahead_bars=252,
        reversal_threshold_pct=20.0,
        throwback_tolerance_pct=1.0,
        invalidation_return_threshold_pct=3.0,
        invalidation_within_bars=10,
        variant_peak_width_tolerance_pct=2.0,
        variant_peak_width_window_bars=15,
        variant_adam_max_peak_width_bars=3,
        variant_eve_min_peak_width_bars=7,
    )
    evaluator = PostBreakoutEvaluator(eval_config)

    run_config: Dict[str, Any] = {
        "scanner_config": scanner.config.__dict__,
        "evaluation_config": eval_config.__dict__,
        "min_rows": int(args.min_rows),
        "patterns": patterns,
        "pivot_min_spacing_bars": 10,
    }
    run_config_hash = _md5_json(run_config)

    # Connections
    src_conn = sqlite3.connect(source_db_path)
    src_conn.execute("PRAGMA query_only=ON;")
    res_conn = connect_results_db(results_db_path)
    ensure_schema(res_conn)

    symbols = _load_symbols(src_conn, min_rows=int(args.min_rows), limit=args.limit)

    upsert_run(
        res_conn,
        run_id=run_id,
        source_db_path=source_db_path,
        source_table=SOURCE_TABLE,
        patterns=patterns,
        run_config=run_config,
        run_config_hash=run_config_hash,
        symbols_planned=len(symbols),
        symbols_scanned=0,
        symbols_failed=0,
        notes=args.notes,
    )
    res_conn.commit()

    detections_total = 0
    eval_total = 0
    scanned = 0
    failed = 0

    try:
        for i, symbol in enumerate(symbols, start=1):
            if i == 1 or (i % 50 == 0):
                print(f"[{run_id}] Scanning {i}/{len(symbols)}: {symbol}")

            try:
                df = _load_symbol_df(src_conn, symbol)
                if len(df) < int(args.min_rows):
                    continue

                # Normalize once here so detection + evaluation see the same bars.
                df_norm, _ = scanner.normalizer.normalize(df)
                if len(df_norm) < int(args.min_rows):
                    scanned += 1
                    continue

                raw_pivots = scanner.pivot_detector.detect_pivots(df_norm, scanner.config.pivot_type)
                pivots = scanner.pivot_detector.get_filtered_pivots(raw_pivots, min_spacing=10)
                if len(pivots) < 3:
                    scanned += 1
                    continue

                detections = []
                for p in patterns:
                    s = scanner.scanners[p]
                    try:
                        detections.extend(
                            s.scan(
                                symbol=symbol,
                                df=df_norm,
                                pivots_filtered=pivots,
                                pivots_raw=raw_pivots,
                            )
                        )
                    except TypeError:
                        # Legacy signature
                        detections.extend(s.scan(symbol, df_norm, pivots, raw_pivots))

                detections_pd = [scanner._to_detection(d) for d in detections]
                if detections:
                    # Ensure PatternDetection objects for persistence layer.
                    detections_total += insert_detections(res_conn, run_id=run_id, detections=detections_pd)

                # Evaluate confirmed breakouts only (look-ahead metrics)
                confirmed = [d for d in detections_pd if d.breakout_date is not None and d.breakout_price is not None]
                if confirmed:
                    results = [evaluator.evaluate(d, df_norm) for d in confirmed]
                    eval_total += insert_post_breakout_results(res_conn, run_id=run_id, results=results)

                scanned += 1

            except Exception as e:
                failed += 1
                record_error(
                    res_conn,
                    run_id=run_id,
                    symbol=symbol,
                    error=str(e),
                    traceback_text=traceback.format_exc(),
                )

            if (i % int(args.commit_every)) == 0:
                upsert_run(
                    res_conn,
                    run_id=run_id,
                    source_db_path=source_db_path,
                    source_table=SOURCE_TABLE,
                    patterns=patterns,
                    run_config=run_config,
                    run_config_hash=run_config_hash,
                    symbols_planned=len(symbols),
                    symbols_scanned=scanned,
                    symbols_failed=failed,
                    notes=args.notes,
                )
                res_conn.commit()

    finally:
        # Finalize run record + stats
        upsert_run(
            res_conn,
            run_id=run_id,
            source_db_path=source_db_path,
            source_table=SOURCE_TABLE,
            patterns=patterns,
            run_config=run_config,
            run_config_hash=run_config_hash,
            symbols_planned=len(symbols),
            symbols_scanned=scanned,
            symbols_failed=failed,
            notes=args.notes,
        )
        res_conn.commit()

        stats = _compute_run_statistics(res_conn, run_id)
        upsert_run_statistics(res_conn, run_id=run_id, stats=stats)
        res_conn.commit()

        src_conn.close()
        res_conn.close()

    print("\n=== Completed ===")
    print(f"run_id: {run_id}")
    print(f"results_db: {results_db_path}")
    print(f"symbols planned: {len(symbols)} scanned: {scanned} failed: {failed}")
    print(f"detections inserted: {detections_total}")
    print(f"evaluations inserted: {eval_total}")


if __name__ == "__main__":
    main()
