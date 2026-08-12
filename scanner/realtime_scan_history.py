"""Persistent outcome ledger for realtime BUY scans.

The scanner produces candidates; this module records them over time and checks
what happened later in OHLCV data.  It is deliberately deterministic and does
not change scanner decisions.  Its job is to create feedback data for improving
BUY_SETUP, BUY_PULLBACK, and tradable-layer rules over time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.run_bear_flag_db_source_parity_audit import DEFAULT_DB, _load_symbol_from_db  # noqa: E402
from scanner.run_buy_setup_scan_watchlist import dedupe_buy_setups  # noqa: E402


WORKFLOW_ID = "realtime_scan_history_v1"
DEFAULT_HISTORY_DIR = Path("artifacts/realtime_scan/history")
DEFAULT_LATEST_DIR = Path("artifacts/realtime_scan/latest")
ACTIONABLE = "actionable_long_cash_candidate_after_buy_confirmed"
WATCHLIST = "watchlist_only_do_not_promote_until_fold_improves"
VN100_GROUPS = {"VN30", "VN100 ex VN30"}
VOLUME_LEDGER_FIELDS = [
    "volume_quality_label",
    "volume_warning_label",
    "volume_ratio_20",
    "value_ratio_20",
    "volume_trend_slope_20",
    "price_volume_phase",
    "pattern_volume_contraction_ratio",
    "obv_slope_20",
    "vpt_slope_20",
    "mfi_14",
    "vwma_fast_minus_slow",
    "vwma_trend_confirmed",
]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _date(value: Any) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return _clean(value)
    return dt.date().isoformat()


def _candidate_id(parts: list[Any]) -> str:
    raw = "|".join(_clean(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"rt_{digest}"


def _setup_rows(setups: pd.DataFrame) -> list[dict[str, Any]]:
    if setups.empty:
        return []
    rows: list[dict[str, Any]] = []
    for row in dedupe_buy_setups(setups).to_dict("records"):
        candidate_date = _date(row.get("latest_date"))
        candidate_id = _candidate_id(
            [
                "BUY_SETUP",
                row.get("symbol"),
                row.get("pattern_id"),
                row.get("setup_start_date"),
                candidate_date,
                round(float(row.get("trigger_price") or 0.0), 4),
            ]
        )
        observation = {
            "candidate_id": candidate_id,
            "stage": "BUY_SETUP",
            "symbol": _clean(row.get("symbol")).upper(),
            "pattern_id": row.get("pattern_id"),
            "family": row.get("family"),
            "market_group": row.get("market_group"),
            "candidate_date": candidate_date,
            "setup_start_date": _date(row.get("setup_start_date")),
            "trigger_price": row.get("trigger_price"),
            "invalidation_price": row.get("invalidation_price"),
            "target_price": row.get("target_price"),
            "reference_price": row.get("last_close"),
            "distance_to_trigger_pct": row.get("distance_to_trigger_pct"),
            "potential_profit_pct": row.get("potential_profit_pct"),
            "setup_quality_score": row.get("setup_quality_score"),
            "setup_quality_tier": row.get("setup_quality_tier"),
            "source_event_id": "",
            "after_buy_action": "",
            "source_workflow_id": row.get("workflow_id"),
            "detector_version": row.get("detector_version"),
            "reason": row.get("setup_reason"),
        }
        for field in VOLUME_LEDGER_FIELDS:
            observation[field] = row.get(field, "")
        rows.append(observation)
    return rows


def _pullback_rows(watchlist: pd.DataFrame) -> list[dict[str, Any]]:
    if watchlist.empty or "after_buy_action" not in watchlist.columns:
        return []
    rows: list[dict[str, Any]] = []
    subset = watchlist.loc[
        watchlist["after_buy_action"].isin({ACTIONABLE, WATCHLIST})
        & watchlist.get("market_group", pd.Series(index=watchlist.index, dtype=object)).isin(VN100_GROUPS)
    ].copy()
    for row in subset.to_dict("records"):
        action = _clean(row.get("after_buy_action"))
        stage = "BUY_PULLBACK" if action == ACTIONABLE else "BUY_WATCHLIST"
        candidate_date = _date(row.get("event_date"))
        candidate_id = _candidate_id(
            [
                stage,
                row.get("symbol"),
                row.get("pattern_id"),
                row.get("source_event_id"),
                candidate_date,
            ]
        )
        observation = {
            "candidate_id": candidate_id,
            "stage": stage,
            "symbol": _clean(row.get("symbol")).upper(),
            "pattern_id": row.get("pattern_id"),
            "family": row.get("family"),
            "market_group": row.get("market_group"),
            "candidate_date": candidate_date,
            "setup_start_date": "",
            "trigger_price": "",
            "invalidation_price": "",
            "target_price": "",
            "reference_price": "",
            "distance_to_trigger_pct": "",
            "potential_profit_pct": "",
            "setup_quality_score": "",
            "setup_quality_tier": "",
            "source_event_id": row.get("source_event_id"),
            "after_buy_action": action,
            "source_workflow_id": "realtime_scan_watchlist_v1",
            "detector_version": "",
            "reason": "",
            "mfe_pct_at_scan": row.get("mfe_pct"),
            "mae_pct_at_scan": row.get("mae_pct"),
            "target_hit_at_scan": row.get("target_hit"),
            "failure_5pct_at_scan": row.get("failure_5pct"),
        }
        for field in VOLUME_LEDGER_FIELDS:
            observation[field] = row.get(field, "")
        rows.append(observation)
    return rows


def build_observations(
    *,
    buy_setups: pd.DataFrame | None = None,
    watchlist: pd.DataFrame | None = None,
    scan_run_id: str | None = None,
    generated_at: str | None = None,
) -> pd.DataFrame:
    generated_at = generated_at or datetime.now().isoformat(timespec="seconds")
    scan_run_id = scan_run_id or generated_at.replace(":", "").replace("-", "").replace("T", "_")
    rows = []
    rows.extend(_setup_rows(buy_setups if buy_setups is not None else pd.DataFrame()))
    rows.extend(_pullback_rows(watchlist if watchlist is not None else pd.DataFrame()))
    for row in rows:
        row["scan_run_id"] = scan_run_id
        row["observed_at"] = generated_at
    return pd.DataFrame(rows)


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _event_dates(frame: pd.DataFrame, condition: pd.Series) -> tuple[str | None, int | None]:
    hits = frame.loc[condition.fillna(False)]
    if hits.empty:
        return None, None
    first = hits.iloc[0]
    return _date(first["date"]), int(frame.index.get_loc(first.name) + 1)


def evaluate_setup_outcome(row: Mapping[str, Any], df: pd.DataFrame, *, horizon_days: int = 120) -> dict[str, Any]:
    candidate_date = pd.to_datetime(row.get("candidate_date"), errors="coerce")
    trigger = _num(row.get("trigger_price"))
    invalidation = _num(row.get("invalidation_price"))
    target = _num(row.get("target_price"))
    reference = _num(row.get("reference_price"))
    if df.empty or pd.isna(candidate_date) or trigger is None or invalidation is None or target is None:
        return {"outcome_status": "insufficient_data"}
    future = df.loc[pd.to_datetime(df["date"], errors="coerce") > candidate_date].copy()
    if horizon_days > 0:
        future = future.head(int(horizon_days))
    if future.empty:
        return {"outcome_status": "pending_no_future_data"}
    future = future.reset_index(drop=True)
    trigger_date, bars_to_trigger = _event_dates(future, pd.to_numeric(future["close"], errors="coerce") >= trigger)
    invalid_date, bars_to_invalid = _event_dates(future, pd.to_numeric(future["low"], errors="coerce") <= invalidation)
    max_high = pd.to_numeric(future["high"], errors="coerce").max()
    min_low = pd.to_numeric(future["low"], errors="coerce").min()
    mfe = ((float(max_high) / reference - 1.0) * 100.0) if reference and not pd.isna(max_high) else None
    mae = ((1.0 - float(min_low) / reference) * 100.0) if reference and not pd.isna(min_low) else None
    if trigger_date is None:
        if invalid_date is not None:
            status = "setup_invalidated_pre_trigger"
        else:
            status = "setup_waiting"
        return {
            "outcome_status": status,
            "triggered": False,
            "target_hit": False,
            "invalidated": invalid_date is not None,
            "trigger_date": "",
            "target_date": "",
            "invalid_date": invalid_date or "",
            "bars_to_trigger": "",
            "bars_to_target": "",
            "bars_to_invalid": bars_to_invalid or "",
            "mfe_since_candidate_pct": round(mfe, 2) if mfe is not None else "",
            "mae_since_candidate_pct": round(mae, 2) if mae is not None else "",
        }
    trigger_ts = pd.to_datetime(trigger_date)
    after_trigger = future.loc[pd.to_datetime(future["date"], errors="coerce") >= trigger_ts].reset_index(drop=True)
    target_date, bars_after_trigger_to_target = _event_dates(after_trigger, pd.to_numeric(after_trigger["high"], errors="coerce") >= target)
    invalid_after_date, bars_after_trigger_to_invalid = _event_dates(
        after_trigger,
        pd.to_numeric(after_trigger["low"], errors="coerce") <= invalidation,
    )
    if target_date and (not invalid_after_date or pd.to_datetime(target_date) <= pd.to_datetime(invalid_after_date)):
        status = "target_hit_after_trigger"
    elif invalid_after_date:
        status = "failed_after_trigger"
    else:
        status = "triggered_open"
    return {
        "outcome_status": status,
        "triggered": True,
        "target_hit": status == "target_hit_after_trigger",
        "invalidated": status == "failed_after_trigger",
        "trigger_date": trigger_date,
        "target_date": target_date or "",
        "invalid_date": invalid_after_date or "",
        "bars_to_trigger": bars_to_trigger or "",
        "bars_to_target": (bars_to_trigger or 0) + (bars_after_trigger_to_target or 0) - 1 if target_date else "",
        "bars_to_invalid": (bars_to_trigger or 0) + (bars_after_trigger_to_invalid or 0) - 1 if invalid_after_date else "",
        "mfe_since_candidate_pct": round(mfe, 2) if mfe is not None else "",
        "mae_since_candidate_pct": round(mae, 2) if mae is not None else "",
    }


def evaluate_pullback_outcome(row: Mapping[str, Any]) -> dict[str, Any]:
    target_hit = _as_bool(row.get("target_hit_at_scan"))
    failure = _as_bool(row.get("failure_5pct_at_scan"))
    mae = _num(row.get("mae_pct_at_scan"))
    if target_hit:
        status = "target_hit_observed"
    elif failure or (mae is not None and mae > 5.0):
        status = "failed_5pct_observed"
    else:
        status = "active_post_breakout"
    return {
        "outcome_status": status,
        "triggered": True,
        "target_hit": bool(target_hit),
        "invalidated": bool(status == "failed_5pct_observed"),
        "trigger_date": row.get("candidate_date") or "",
        "target_date": "",
        "invalid_date": "",
        "bars_to_trigger": 0,
        "bars_to_target": "",
        "bars_to_invalid": "",
        "mfe_since_candidate_pct": row.get("mfe_pct_at_scan") or "",
        "mae_since_candidate_pct": abs(float(mae)) if mae is not None else "",
    }


def _non_empty_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    series = frame[column].map(_clean)
    series = series.loc[series.ne("")]
    return {str(k): int(v) for k, v in series.value_counts().to_dict().items()}


def evaluate_observations(observations: pd.DataFrame, *, db_path: Path = DEFAULT_DB, horizon_days: int = 120) -> pd.DataFrame:
    if observations.empty:
        return observations.copy()
    cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    conn = sqlite3.connect(str(db_path))
    try:
        for row in observations.to_dict("records"):
            symbol = _clean(row.get("symbol")).upper()
            if symbol not in cache:
                cache[symbol] = _load_symbol_from_db(conn, symbol)
            if row.get("stage") == "BUY_SETUP":
                outcome = evaluate_setup_outcome(row, cache[symbol], horizon_days=horizon_days)
            else:
                outcome = evaluate_pullback_outcome(row)
            rows.append({**row, **outcome})
    finally:
        conn.close()
    return pd.DataFrame(rows)


def merge_ledger(existing: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return existing.copy()
    if existing.empty:
        base = observations.copy()
        base["first_seen_at"] = base["observed_at"]
        base["last_seen_at"] = base["observed_at"]
        base["seen_count"] = 1
        return base.sort_values(["last_seen_at", "stage", "symbol"], ascending=[False, True, True]).reset_index(drop=True)
    old = existing.copy()
    new = observations.copy()
    combined = pd.concat([old, new], ignore_index=True, sort=False)
    combined["observed_at_sort"] = pd.to_datetime(combined.get("observed_at"), errors="coerce")
    latest = (
        combined.sort_values(["candidate_id", "observed_at_sort"], ascending=[True, True])
        .drop_duplicates("candidate_id", keep="last")
        .drop(columns=["observed_at_sort"], errors="ignore")
    )
    old_seen = pd.to_numeric(old.get("seen_count", pd.Series(dtype="float64")), errors="coerce").fillna(1)
    old_counts = old.assign(_seen_count=old_seen).groupby("candidate_id")["_seen_count"].max()
    new_counts = new.groupby("candidate_id").size()
    seen_counts = old_counts.add(new_counts, fill_value=0).astype(int)
    old_first = old.set_index("candidate_id").get("first_seen_at", pd.Series(dtype=object))
    new_first = new.groupby("candidate_id")["observed_at"].min()
    first_seen = old_first.combine_first(new_first)
    new_last = new.groupby("candidate_id")["observed_at"].max()
    old_last = old.set_index("candidate_id").get("last_seen_at", old.set_index("candidate_id").get("observed_at", pd.Series(dtype=object)))
    last_seen = old_last.combine_first(new_last)
    last_seen.update(new_last)
    latest["first_seen_at"] = latest["candidate_id"].map(first_seen)
    latest["last_seen_at"] = latest["candidate_id"].map(last_seen)
    latest["seen_count"] = latest["candidate_id"].map(seen_counts).fillna(1).astype(int)
    return latest.sort_values(["last_seen_at", "stage", "symbol"], ascending=[False, True, True]).reset_index(drop=True)


def summarize_ledger(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"total_candidates": 0, "by_stage": {}, "by_outcome": {}, "setup_conversion": {}, "by_volume_quality": {}, "by_volume_warning": {}}
    by_stage = ledger.groupby("stage").size().sort_values(ascending=False).to_dict()
    by_outcome = ledger.groupby(["stage", "outcome_status"]).size().to_dict()
    setup = ledger.loc[ledger["stage"].eq("BUY_SETUP")].copy() if "stage" in ledger.columns else pd.DataFrame()
    if setup.empty:
        setup_conversion = {"setup_total": 0}
    else:
        setup_conversion = {
            "setup_total": int(len(setup)),
            "triggered": int(setup.get("triggered", pd.Series(dtype=object)).map(_as_bool).eq(True).sum()),
            "target_hit_after_trigger": int(setup.get("outcome_status", pd.Series(dtype=object)).eq("target_hit_after_trigger").sum()),
            "invalidated_pre_trigger": int(setup.get("outcome_status", pd.Series(dtype=object)).eq("setup_invalidated_pre_trigger").sum()),
            "failed_after_trigger": int(setup.get("outcome_status", pd.Series(dtype=object)).eq("failed_after_trigger").sum()),
            "waiting": int(setup.get("outcome_status", pd.Series(dtype=object)).isin({"setup_waiting", "pending_no_future_data"}).sum()),
        }
    setup_volume_quality = _non_empty_counts(setup, "volume_quality_label")
    setup_volume_warning = _non_empty_counts(setup, "volume_warning_label")
    return {
        "total_candidates": int(len(ledger)),
        "by_stage": {str(k): int(v) for k, v in by_stage.items()},
        "by_outcome": {f"{k[0]}::{k[1]}": int(v) for k, v in by_outcome.items()},
        "setup_conversion": setup_conversion,
        "by_volume_quality": _non_empty_counts(ledger, "volume_quality_label"),
        "by_volume_warning": _non_empty_counts(ledger, "volume_warning_label"),
        "setup_volume_quality": setup_volume_quality,
        "setup_volume_warning": setup_volume_warning,
    }


def update_realtime_scan_history(
    *,
    buy_setups: pd.DataFrame | None = None,
    watchlist: pd.DataFrame | None = None,
    db_path: Path = DEFAULT_DB,
    history_dir: Path = DEFAULT_HISTORY_DIR,
    horizon_days: int = 120,
    generated_at: str | None = None,
) -> dict[str, Any]:
    history_dir.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or datetime.now().isoformat(timespec="seconds")
    scan_run_id = generated_at.replace(":", "").replace("-", "").replace("T", "_")
    observations = build_observations(buy_setups=buy_setups, watchlist=watchlist, scan_run_id=scan_run_id, generated_at=generated_at)
    evaluated = evaluate_observations(observations, db_path=db_path, horizon_days=horizon_days)
    observation_path = history_dir / "scan_observations.csv"
    ledger_path = history_dir / "candidate_ledger.csv"
    summary_path = history_dir / "history_summary.json"
    existing_observations = _load_existing(observation_path)
    all_observations = pd.concat([existing_observations, evaluated], ignore_index=True, sort=False) if not existing_observations.empty else evaluated
    all_observations.to_csv(observation_path, index=False)
    existing_ledger = _load_existing(ledger_path)
    ledger = merge_ledger(existing_ledger, evaluated)
    ledger.to_csv(ledger_path, index=False)
    summary = {
        "workflow_id": WORKFLOW_ID,
        "scan_run_id": scan_run_id,
        "generated_at": generated_at,
        "new_observations": int(len(evaluated)),
        "history": summarize_ledger(ledger),
        "paths": {
            "observations": str(observation_path),
            "ledger": str(ledger_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def _read_latest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Update realtime BUY scan candidate history and outcome ledger.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--latest-dir", default=str(DEFAULT_LATEST_DIR))
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--horizon-days", type=int, default=120)
    args = parser.parse_args()
    latest = Path(args.latest_dir)
    buy_setups = _read_latest(latest / "buy_setup" / "buy_setup_watchlist.csv")
    watchlist = _read_latest(latest / "realtime_watchlist.csv")
    result = update_realtime_scan_history(
        buy_setups=buy_setups,
        watchlist=watchlist,
        db_path=Path(args.db),
        history_dir=Path(args.history_dir),
        horizon_days=int(args.horizon_days),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
