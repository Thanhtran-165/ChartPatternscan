"""Build and optionally send the realtime BUY-candidate watchlist email.

This layer is intentionally deterministic: it formats scanner/tradable outputs
into a concise review email. It does not use AI and does not create buy/sell
recommendations.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import smtplib
import ssl
import sys
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MAIN_SONET_ROOT = ROOT.parent
if str(MAIN_SONET_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_SONET_ROOT))

from scanner.run_realtime_scan_watchlist import (  # noqa: E402
    DEFAULT_AFTER_BUY_CONFIG,
    DEFAULT_DB,
    DEFAULT_OUT_DIR,
    build_realtime_scan_plan,
    build_watchlist_from_artifacts,
    write_realtime_outputs,
)
from scanner.build_realtime_scan_pdf_report import (  # noqa: E402
    DEFAULT_PDF_OUT,
    build_realtime_scan_pdf_report,
)
from scanner.run_buy_setup_scan_watchlist import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_BUY_SETUP_OUT_DIR,
    dedupe_buy_setups,
    scan_buy_setups,
    write_buy_setup_outputs,
)
from scanner.realtime_scan_history import (  # noqa: E402
    DEFAULT_HISTORY_DIR,
    update_realtime_scan_history,
)
from scanner.refresh_realtime_market_data import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_DATA_REFRESH_OUT_DIR,
    refresh_realtime_market_data,
)
from scanner.run_bear_flag_db_source_parity_audit import _load_symbol_from_db  # noqa: E402
import sqlite3  # noqa: E402


WORKFLOW_ID = "realtime_scan_email_v1"
DEFAULT_EMAIL_OUT_DIR = Path("artifacts/realtime_scan/latest/email")
DEFAULT_PREFLIGHT_MATRIX = Path("artifacts/governance/final_chapters/governance/chapter_tradable_preflight_matrix.json")
# Profile V3 (M3): metadata.patterns_stats = failure_busted_rate/tier/median_target_dist/n per pattern.
DEFAULT_V3_PROFILE = Path(__file__).resolve().parents[2] / "market_stats" / "web" / "stock_pattern_profiles.json"
ACTIONABLE = "actionable_long_cash_candidate_after_buy_confirmed"
WATCHLIST = "watchlist_only_do_not_promote_until_fold_improves"
RISK = "avoid_buy_or_exit_warning"
VN100_GROUPS = {"VN30", "VN100 ex VN30"}


@dataclass(frozen=True)
class EmailConfig:
    smtp_host: str
    smtp_port: int
    smtp_user: str | None
    smtp_password: str | None
    mail_from: str
    mail_to: list[str]
    use_tls: bool = True
    use_ssl: bool = False
    timeout: int = 30


def _as_bool(value: str | None, *, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _dotenv_value(key: str) -> str | None:
    for path in (MAIN_SONET_ROOT / ".env.local", MAIN_SONET_ROOT / ".env", ROOT / ".env"):
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            raw_key, raw_value = stripped.split("=", 1)
            if raw_key.strip() == key:
                value = raw_value.strip().strip('"').strip("'")
                return value or None
    return None


def _config_value(key: str) -> str | None:
    return os.environ.get(key) or _dotenv_value(key)


def load_email_config_from_env() -> EmailConfig:
    central = _load_central_email_settings()
    host = (
        _config_value("REALTIME_SCAN_SMTP_HOST")
        or _config_value("SMTP_HOST")
        or _config_value("SMTP_SERVER")
        or (central or {}).get("host")
    )
    to_raw = (
        _config_value("REALTIME_SCAN_EMAIL_TO")
        or _config_value("TEST_EMAIL")
        or _config_value("ASSET_REPORT_TO")
        or _config_value("EMAIL_TO")
    )
    mail_from = (
        _config_value("REALTIME_SCAN_EMAIL_FROM")
        or _config_value("SMTP_FROM")
        or _config_value("SMTP_USER")
        or _config_value("EMAIL_SENDER")
        or (central or {}).get("sender")
    )
    if not host:
        raise RuntimeError("Missing SMTP host. Set REALTIME_SCAN_SMTP_HOST, SMTP_HOST, SMTP_SERVER, or central EMAIL_* config.")
    if not to_raw:
        raise RuntimeError("Missing recipient. Set REALTIME_SCAN_EMAIL_TO, TEST_EMAIL, ASSET_REPORT_TO, or EMAIL_TO.")
    if not mail_from:
        raise RuntimeError("Missing sender. Set REALTIME_SCAN_EMAIL_FROM, SMTP_FROM, SMTP_USER, EMAIL_SENDER, or central EMAIL_* config.")
    return EmailConfig(
        smtp_host=str(host),
        smtp_port=int(_config_value("REALTIME_SCAN_SMTP_PORT") or _config_value("SMTP_PORT") or (central or {}).get("port") or "587"),
        smtp_user=_config_value("REALTIME_SCAN_SMTP_USER") or _config_value("SMTP_USER") or _config_value("EMAIL_USERNAME") or (central or {}).get("username"),
        smtp_password=_config_value("REALTIME_SCAN_SMTP_PASSWORD") or _config_value("SMTP_PASSWORD") or _config_value("EMAIL_PASSWORD") or (central or {}).get("password"),
        mail_from=str(mail_from),
        mail_to=[item.strip() for item in to_raw.split(",") if item.strip()],
        use_tls=_as_bool(_config_value("REALTIME_SCAN_SMTP_TLS") or _config_value("SMTP_TLS"), default=bool((central or {}).get("use_tls", True))),
        use_ssl=_as_bool(_config_value("REALTIME_SCAN_SMTP_SSL") or _config_value("SMTP_SSL"), default=bool((central or {}).get("use_ssl", False))),
        timeout=int(_config_value("REALTIME_SCAN_SMTP_TIMEOUT") or _config_value("EMAIL_TIMEOUT") or (central or {}).get("timeout") or "30"),
    )


def _load_central_email_settings() -> dict[str, Any] | None:
    try:
        from config.email_config import get_email_config  # type: ignore
    except Exception:
        return None
    try:
        smtp = get_email_config().smtp
    except Exception:
        return None
    return {
        "host": smtp.host,
        "port": smtp.port,
        "username": smtp.username,
        "password": smtp.password,
        "sender": smtp.sender,
        "use_tls": smtp.use_tls,
        "use_ssl": smtp.use_ssl,
        "timeout": smtp.timeout,
    }


def _clean(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    return text if text else default


def _pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return _clean(value)


def _pct_abs(value: Any) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
        return f"{abs(float(value)):.2f}%"
    except (TypeError, ValueError):
        return _clean(value)


def _prob(value: Any) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
        number = float(value)
        if 0 <= number <= 1:
            number *= 100
        return f"{number:.0f}%"
    except (TypeError, ValueError):
        return _clean(value)


def _date_label(value: Any) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return _clean(value)
    return dt.date().isoformat()


def _pattern_label(pattern_id: Any) -> str:
    labels = {
        "bull_flags": "Cờ tăng",
        "bull_pennants": "Cờ đuôi nheo tăng",
        "high_tight_flags": "Cờ cao và chặt",
        "double_bottoms_adam_adam": "Hai đáy Adam-Adam",
        "double_bottoms_adam_eve": "Hai đáy Adam-Eve",
        "double_bottoms_eve_adam": "Hai đáy Eve-Adam",
        "double_bottoms_eve_eve": "Hai đáy Eve-Eve",
        "measured_move_up": "Measured Move tăng",
        "rectangle_bottoms": "Chữ nhật đáy",
        "triangles_ascending": "Tam giác tăng",
        "triangles_symmetrical": "Tam giác cân",
        "triangles_descending": "Tam giác giảm",
        "broadening_bottoms": "Mở rộng đáy",
        "wedges_falling": "Nêm giảm",
        "cup_with_handle": "Cốc tay cầm",
    }
    text = str(pattern_id or "")
    return labels.get(text, text.replace("_", " ").title() if text else "-")


def _volume_label(value: Any) -> str:
    labels = {
        "strong": "mạnh",
        "healthy": "ổn",
        "weak": "yếu",
        "risky": "cần thận trọng",
        "unknown": "chưa rõ",
        "none": "không",
        "missing_volume": "thiếu dữ liệu khối lượng",
        "adverse_volume_spike": "giá yếu đi kèm volume lớn",
        "noisy_setup_volume": "volume trong mẫu hơi nhiễu",
        "thin_value": "giá trị giao dịch mỏng",
    }
    text = str(value or "").strip()
    return labels.get(text, text or "-")


def _load_pattern_context(path: Path = DEFAULT_PREFLIGHT_MATRIX) -> dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = data.get("chapters") if isinstance(data.get("chapters"), list) else []
    return {str(row.get("pattern_id")): row for row in rows if isinstance(row, Mapping) and row.get("pattern_id")}


def _load_v3_pattern_stats(path: Path | None = None) -> dict[str, Mapping[str, Any]]:
    """patterns_stats từ profile V3 (M3) — failure_busted_rate/tier/median_target_dist/n per pattern.

    Mặc định đọc market_stats/web; có thể trỏ staging (web_v3) qua env
    REALTIME_SCAN_V3_PROFILE — dùng khi chạy mail mẫu trước khi chuyển V3."""
    if path is None:
        path = Path(os.environ.get("REALTIME_SCAN_V3_PROFILE") or DEFAULT_V3_PROFILE)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    stats = (data.get("metadata") or {}).get("patterns_stats") or {}
    return {str(k): v for k, v in stats.items() if isinstance(v, Mapping)}


MANIFEST_PATH = Path(__file__).resolve().parents[1] / "scanner" / "v2" / "pattern_family_manifest.json"
BOOK_STATUSES = {"publication_final", "active"}


def _load_manifest_status() -> dict[str, str]:
    """pattern_id -> status từ pattern_family_manifest.json (M2 — V4 Pro 13/08).

    Pattern sách (bear_flags/bull_pennants/bear_pennants...) chưa có events V3 →
    không có trong patterns_stats → mail hạ nhầm xuống 'Bản nháp — chưa kiểm định'.
    Fallback manifest để nhãn trung thực: pattern đã đối chiếu PDF thì gắn đúng
    nhãn 'Đã đối chiếu PDF', chỉ thiếu số đo V3 (n=0 → không qualified)."""
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        d = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, str] = {}
    for finfo in (d.get("families") or {}).values():
        for pkey, pinfo in (finfo.get("patterns") or {}).items():
            out[str(pkey)] = pinfo.get("status", "unknown")
    return out


def _v3_or_manifest_stats(v3_stats: Mapping[str, Mapping[str, Any]], pattern_id: Any,
                          manifest: Mapping[str, str]) -> dict[str, Any]:
    """Thống kê V3 cho 1 pattern; pattern sách chưa đo V3 → nhãn theo manifest (M2)."""
    s = dict(v3_stats.get(str(pattern_id)) or {})
    if s.get("tier") is not None:
        return s
    if manifest.get(str(pattern_id)) in BOOK_STATUSES:
        return {"tier": 3, "tier_label": "Đã đối chiếu PDF",
                "tier_note": "chuẩn sách Bulkowski — chưa có events V3 để đo", "n": 0}
    return s


def _enrich_rows_with_pattern_context(rows: list[Mapping[str, Any]], context: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    v3 = _load_v3_pattern_stats()
    manifest = _load_manifest_status()
    enriched: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        pattern_context = context.get(str(row.get("pattern_id"))) or {}
        v3_stats = _v3_or_manifest_stats(v3, row.get("pattern_id"), manifest)
        out["pattern_label"] = _pattern_label(row.get("pattern_id"))
        out["potential_profit_pct"] = pattern_context.get("median_mfe_pct")
        out["target_success_probability"] = pattern_context.get("target_hit_rate")
        out["clean_path_probability"] = pattern_context.get("target_first_before_adverse_5pct_rate")
        out["confidence_score"] = pattern_context.get("preflight_score")
        # V3 (M3): thống kê chuẩn Bulkowski — failure_busted, nấc, mục tiêu, cỡ mẫu.
        out["failure_busted_rate_pct"] = v3_stats.get("failure_busted_rate_pct")
        out["weak_move_5pct_rate_pct"] = v3_stats.get("weak_move_5pct_rate_pct")
        out["median_target_dist_pct"] = v3_stats.get("median_target_dist_pct")
        out["pattern_n"] = v3_stats.get("n")
        out["pattern_tier"] = v3_stats.get("tier")
        out["pattern_tier_label"] = v3_stats.get("tier_label")
        enriched.append(out)
    return enriched


def _enrich_setup_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["pattern_label"] = _pattern_label(row.get("pattern_id"))
        enriched.append(out)
    return enriched


def _section(df: pd.DataFrame, action: str, limit: int) -> pd.DataFrame:
    if df.empty or "after_buy_action" not in df.columns:
        return df.head(0).copy()
    out = df.loc[df["after_buy_action"].eq(action)].copy()
    if "event_date" in out.columns:
        out["event_date_sort"] = pd.to_datetime(out["event_date"], errors="coerce")
        sort_cols = [column for column in ["event_date_sort", "market_group", "liquidity_bucket", "symbol"] if column in out.columns]
        ascending = [False if column == "event_date_sort" else True for column in sort_cols]
        out = out.sort_values(sort_cols, ascending=ascending)
        out = out.drop(columns=["event_date_sort"])
    return out.head(limit)


def _bool_like(value: Any) -> bool | None:
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
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _bool_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype=object)
    return df[column].map(_bool_like)


def _failure_busted_column(watchlist: pd.DataFrame) -> pd.Series:
    """failure_busted (V3) nếu có cột; fallback failure_5pct cũ khi artifacts chưa re-scan."""
    busted = _bool_column(watchlist, "failure_busted")
    if busted.notna().any():
        return busted
    return _bool_column(watchlist, "failure_5pct")


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _event_level_buy_candidate_mask(watchlist: pd.DataFrame, *, max_mae_pct: float = 5.0) -> pd.Series:
    if watchlist.empty or "after_buy_action" not in watchlist.columns:
        return pd.Series([False] * len(watchlist), index=watchlist.index)
    action_ok = watchlist["after_buy_action"].eq(ACTIONABLE)
    failure = _failure_busted_column(watchlist)
    target_hit = _bool_column(watchlist, "target_hit")
    mae = _numeric_column(watchlist, "mae_pct")
    # Missing forward fields are tolerated for fresh events. Explicit failure or
    # already-hit targets are not treated as current BUY candidates in the email.
    not_failed = ~failure.eq(True)
    not_already_target_hit = ~target_hit.eq(True)
    not_adverse_too_deep = mae.isna() | mae.le(float(max_mae_pct))
    return action_ok & not_failed & not_already_target_hit & not_adverse_too_deep


def _event_level_watchlist_mask(watchlist: pd.DataFrame, buy_mask: pd.Series) -> pd.Series:
    if watchlist.empty or "after_buy_action" not in watchlist.columns:
        return pd.Series([False] * len(watchlist), index=watchlist.index)
    action = watchlist["after_buy_action"]
    failure = _failure_busted_column(watchlist)
    return (action.eq(WATCHLIST) | action.eq(ACTIONABLE)) & ~buy_mask & ~failure.eq(True)


def _event_level_risk_mask(watchlist: pd.DataFrame) -> pd.Series:
    if watchlist.empty:
        return pd.Series([False] * len(watchlist), index=watchlist.index)
    action = watchlist.get("after_buy_action", pd.Series(index=watchlist.index, dtype=object))
    failure = _failure_busted_column(watchlist)
    mae = _numeric_column(watchlist, "mae_pct")
    return action.eq(RISK) | failure.eq(True) | mae.gt(5.0)


def _mask_section(df: pd.DataFrame, mask: pd.Series, limit: int) -> pd.DataFrame:
    out = df.loc[mask].copy()
    if out.empty:
        return out
    if "event_date" in out.columns:
        out["event_date_sort"] = pd.to_datetime(out["event_date"], errors="coerce")
        sort_cols = [column for column in ["event_date_sort", "market_group", "liquidity_bucket", "symbol"] if column in out.columns]
        ascending = [False if column == "event_date_sort" else True for column in sort_cols]
        out = out.sort_values(sort_cols, ascending=ascending).drop(columns=["event_date_sort"])
    return out.head(limit)


def _vn100_only(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if df.empty or "market_group" not in df.columns:
        return df.head(0).copy(), "vn100_required_no_market_group"
    vn100 = df.loc[df["market_group"].isin(VN100_GROUPS)].copy()
    if not vn100.empty:
        return vn100, "vn100_required"
    return df.head(0).copy(), "vn100_required_no_rows"


def _load_setup_persistence(history_dir: Path, generated_at: str) -> dict[tuple[str, str], int]:
    """Read candidate_ledger.csv and compute consecutive scan-day streaks ending at generated_at.

    Returns {(symbol, pattern_id): streak_days}. A streak counts how many of the
    most recent scan days (ending exactly on the generated_at date) the setup
    appeared in without missing a day. "Scan day" = any distinct observed_at date
    present in the ledger, so weekend/market-holiday gaps do not break a streak
    the way calendar-day counting would.
    """
    ledger_path = history_dir / "candidate_ledger.csv"
    if not ledger_path.exists():
        return {}
    try:
        ledger = pd.read_csv(ledger_path, low_memory=False)
    except Exception:
        return {}
    if ledger.empty or "symbol" not in ledger.columns or "pattern_id" not in ledger.columns or "observed_at" not in ledger.columns:
        return {}
    today = pd.to_datetime(generated_at, errors="coerce")
    if pd.isna(today):
        return {}
    today_date = today.date()
    # Keep only BUY_SETUP rows (those are what buy_setup scan produces)
    stage_col = ledger.get("stage")
    if stage_col is not None:
        ledger = ledger.loc[ledger["stage"].fillna("BUY_SETUP").eq("BUY_SETUP") | ledger["stage"].isna()].copy()
    # Build the global ordered list of scan days from the entire ledger so that
    # gaps for one symbol (e.g. a day it didn't qualify) still count as a break
    # while natural market closures (no scan ran) do not.
    all_scan_days = sorted(
        set(pd.to_datetime(ledger["observed_at"], errors="coerce").dropna().dt.date),
        reverse=True,
    )
    if not all_scan_days or today_date not in all_scan_days:
        # If today is not yet in the ledger (e.g. history update was skipped),
        # fall back to the most recent scan day as the streak anchor.
        anchor_index = 0
    else:
        anchor_index = all_scan_days.index(today_date)
    persistence: dict[tuple[str, str], int] = {}
    for (symbol, pattern_id), group in ledger.groupby(["symbol", "pattern_id"], dropna=False):
        symbol_clean = str(symbol).strip().upper() if symbol is not None and not (isinstance(symbol, float) and pd.isna(symbol)) else ""
        pattern_clean = str(pattern_id).strip() if pattern_id is not None and not (isinstance(pattern_id, float) and pd.isna(pattern_id)) else ""
        if not symbol_clean or not pattern_clean:
            continue
        setup_days = set(pd.to_datetime(group["observed_at"], errors="coerce").dropna().dt.date)
        if not setup_days:
            continue
        # Only count a streak if the setup actually appears on the anchor day.
        anchor_day = all_scan_days[anchor_index]
        if anchor_day not in setup_days:
            persistence[(symbol_clean, pattern_clean)] = 0
            continue
        # all_scan_days is sorted newest-first, so walking forward from the
        # anchor means walking back in time. Stop at the first scan day the
        # setup did NOT appear in.
        streak = 0
        for i in range(anchor_index, len(all_scan_days)):
            if all_scan_days[i] in setup_days:
                streak += 1
            else:
                break
        persistence[(symbol_clean, pattern_clean)] = streak
    return persistence


def _detect_breakouts_today(setups: pd.DataFrame, db_path: Path) -> set[tuple[str, str]]:
    """Detect setups whose latest OHLCV bar closed above trigger while the previous bar closed below.

    Returns {(symbol, pattern_id)}. Reads OHLCV via the shared _load_symbol_from_db
    helper, caching per symbol to avoid re-querying.
    """
    if setups.empty or "trigger_price" not in setups.columns or "symbol" not in setups.columns:
        return set()
    breakouts: set[tuple[str, str]] = set()
    cache: dict[str, pd.DataFrame] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        for row in setups.to_dict("records"):
            symbol = str(row.get("symbol") or "").strip().upper()
            pattern_id = str(row.get("pattern_id") or "").strip()
            trigger = row.get("trigger_price")
            try:
                trigger_price = float(trigger) if trigger is not None and not pd.isna(trigger) else None
            except (TypeError, ValueError):
                trigger_price = None
            if not symbol or not pattern_id or trigger_price is None:
                continue
            if symbol not in cache:
                cache[symbol] = _load_symbol_from_db(conn, symbol)
            df = cache[symbol]
            if df.empty or len(df) < 2:
                continue
            close = pd.to_numeric(df["close"], errors="coerce")
            latest_close = close.iloc[-1]
            prev_close = close.iloc[-2]
            if pd.isna(latest_close) or pd.isna(prev_close):
                continue
            if latest_close >= trigger_price and prev_close < trigger_price:
                breakouts.add((symbol, pattern_id))
    finally:
        conn.close()
    return breakouts


def _classify_setups(
    setups: pd.DataFrame,
    persistence_map: Mapping[tuple[str, str], int],
    breakout_set: set[tuple[str, str]],
    *,
    stale_threshold: int = 10,
) -> pd.DataFrame:
    """Enrich each setup row with persistence_days and status_today.

    status_today is one of: new | recurring | stale | confirmed_breakout_today.
    Breakout takes priority over stale so a long-watched name still surfaces on
    the day it finally breaks out.
    """
    if setups.empty:
        out = setups.copy()
        out["persistence_days"] = pd.Series(dtype="int64")
        out["status_today"] = pd.Series(dtype="object")
        return out
    out = setups.copy()
    persistence_days_col: list[int] = []
    status_col: list[str] = []
    for row in out.to_dict("records"):
        symbol = str(row.get("symbol") or "").strip().upper()
        pattern_id = str(row.get("pattern_id") or "").strip()
        key = (symbol, pattern_id)
        days = int(persistence_map.get(key, 0))
        persistence_days_col.append(days)
        if key in breakout_set:
            status_col.append("confirmed_breakout_today")
        elif days >= stale_threshold:
            status_col.append("stale")
        elif days == 0:
            status_col.append("new")
        else:
            status_col.append("recurring")
    out["persistence_days"] = persistence_days_col
    out["status_today"] = status_col
    return out


# Spec "Break-even failure" (BE% — tỷ lệ fail ở mốc 5%) theo Bulkowski gốc,
# nguồn: docs/project/pdf_review/PDF_REVIEW_20260812.md (§2 chi tiết + §3 bảng tóm tắt).
# Cách chọn spec: variant THẤP NHẤT (tốt nhất) trong các thị trường/hướng breakout
# (vd pipe_bottoms 5% bull / 4% bear → spec 4.0) — đại diện chuẩn pattern đáng đạt.
# Pattern KHÔNG có spec ở đây → KHÔNG gate failure (không bịa số; gồm inside_day
# lệch định nghĩa Harami, dead_cat_bounce không có BE, broadening/flags/gaps... chưa trích).
# Rào mail (phán quyết chủ đầu tư 13/08/2026 — H2): failure_busted_rate_pct ≤ 2×spec.
PATTERN_FAILURE_SPECS_PCT: dict[str, float] = {
    "pipe_bottoms": 4.0,          # PDF 5% (bull) / 4% (bear)
    "pipe_tops": 2.0,             # PDF 11% / 2%
    "horn_bottoms": 7.0,          # PDF 9% / 7%
    "horn_tops": 2.0,             # PDF 7% / 2%
    "cup_with_handle": 5.0,       # PDF 5% / 7%
    "scallops_ascending": 10.0,   # PDF 10 / 16 / 27 / 14
    "scallops_descending": 8.0,   # PDF 22 / 20 / 15 / 8
    "rectangle_bottoms": 4.0,     # PDF 10 / 11 / 16 / 4
    "rectangle_tops": 9.0,        # PDF 9 / 16 / 11 / 9
    "three_falling_peaks": 4.0,   # PDF 12% / 4%
    "three_rising_valleys": 5.0,  # PDF 5% / 9%
    "high_tight_flags": 0.0,      # PDF 0% / 0% (sách chưa thấy busted)
    "head_and_shoulders_bottoms": 4.0,  # PDF 4% / 8%
    "head_and_shoulders_tops": 1.0,     # PDF 4% / 1%
    "triple_bottoms": 4.0,              # PDF 4% / 8%
    "triple_tops": 5.0,                 # PDF 10% / 5%
}


def _apply_v3_gates(df: pd.DataFrame, v3_stats: Mapping[str, Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rào mail V3 (08 §11 + phán quyết chủ đầu tư 13/08/2026 H1/H2):
    qualified = pattern Nấc 2+ (tier≥2) + n≥30 toàn thị trường + median_target_dist_pct≥5%
    (loại inside_bar 2.3% khỏi top) + failure_busted_rate_pct ≤ 2×spec chuẩn sách
    (PATTERN_FAILURE_SPECS_PCT — loại pattern thực đo VN tệ hơn 2 lần chuẩn Bulkowski);
    draft = Nấc 1 (bản nháp) + các pattern bị rào (section riêng cuối mail, không biến mất).

    Chỉ gate khi profile V3 thật (có field tier) — profile cũ/chưa build không có tier
    thì giữ hành vi cũ (qualified hết) để mail không bị rỗng."""
    if df.empty or "pattern_id" not in df.columns:
        return df.head(0).copy(), df.head(0).copy()
    v3_active = any(isinstance(s, Mapping) and s.get("tier") is not None for s in v3_stats.values())
    if not v3_active:
        return df.copy(), df.head(0).copy()
    qualified, draft = [], []
    manifest = _load_manifest_status()
    for _, row in df.iterrows():
        s = _v3_or_manifest_stats(v3_stats, row.get("pattern_id"), manifest)
        tier = int(s.get("tier") or 1)
        n = int(s.get("n") or 0)
        tgt = s.get("median_target_dist_pct")
        fail = s.get("failure_busted_rate_pct")
        spec = PATTERN_FAILURE_SPECS_PCT.get(str(row.get("pattern_id")))
        failure_gated = spec is not None and fail is not None and float(fail) > 2.0 * spec
        if failure_gated:
            # Rào H2: thực đo VN tệ hơn 2× chuẩn sách → không lên mail chính,
            # hạ xuống draft kèm lý do (pattern chưa đạt chuẩn để giới thiệu).
            row["v3_gate_note"] = (
                f"failure {float(fail):.1f}% > 2×spec ({spec:.1f}%) — chưa đạt chuẩn sách"
            )
            draft.append(row)
        elif tier >= 2 and n >= 30 and (tgt is None or float(tgt) >= 5.0):
            qualified.append(row)
        else:
            # Nấc 1 (bản nháp) + Nấc 2+ nhưng mẫu nhỏ hoặc mục tiêu thấp
            # (VD inside_day 2.3%) → section phụ cuối mail, không biến mất.
            draft.append(row)
    q = pd.DataFrame(qualified, columns=df.columns) if qualified else df.head(0).copy()
    # Không ép columns=df.columns cho draft: giữ cột phụ v3_gate_note (lý do bị rào H2).
    d = pd.DataFrame(draft) if draft else df.head(0).copy()
    return q, d


def summarize_watchlist(
    watchlist: pd.DataFrame,
    *,
    limit_each: int = 20,
    buy_setups: pd.DataFrame | None = None,
    history_dir: Path | None = None,
    db_path: Path | None = None,
    stale_threshold: int = 10,
    generated_at: str | None = None,
) -> dict[str, Any]:
    buy_mask = _event_level_buy_candidate_mask(watchlist)
    watchlist_mask = _event_level_watchlist_mask(watchlist, buy_mask)
    risk_mask = _event_level_risk_mask(watchlist)
    v3_stats = _load_v3_pattern_stats()
    buy_section_raw, buy_draft_raw = _apply_v3_gates(_mask_section(watchlist, buy_mask, limit_each * 5), v3_stats)
    watchlist_section_raw, watch_draft_raw = _apply_v3_gates(_mask_section(watchlist, watchlist_mask, limit_each * 5), v3_stats)
    buy_section, buy_scope = _vn100_only(buy_section_raw)
    watchlist_section, watchlist_scope = _vn100_only(watchlist_section_raw)
    risk_section = _mask_section(watchlist, risk_mask, limit_each)
    setup_section = dedupe_buy_setups(buy_setups) if buy_setups is not None else pd.DataFrame()
    if not setup_section.empty:
        setup_section = setup_section.sort_values(
            ["setup_quality_score", "distance_to_trigger_pct", "potential_profit_pct", "symbol"],
            ascending=[False, True, False, True],
        ).head(limit_each)

    # Stale-filtering enrichment. When history_dir and db_path are both provided
    # we classify each setup into new/recurring/stale/confirmed_breakout_today
    # and split the legacy buy_setup section into three display buckets. When
    # either is missing we fall back to the legacy single-section contract.
    filter_active = bool(history_dir is not None and db_path is not None and not setup_section.empty)
    setup_new_section = pd.DataFrame()
    setup_breakout_section = pd.DataFrame()
    setup_stale_hidden_section = pd.DataFrame()
    stale_hidden_count = 0
    if filter_active:
        ts = generated_at or datetime.now().isoformat(timespec="seconds")
        persistence_map = _load_setup_persistence(history_dir, ts)
        breakout_set = _detect_breakouts_today(setup_section, db_path)
        classified = _classify_setups(
            setup_section,
            persistence_map,
            breakout_set,
            stale_threshold=stale_threshold,
        )
        # Keep display order: new/recurring first, then breakout, stale hidden
        setup_new_section = classified.loc[classified["status_today"].isin(["new", "recurring"])].copy()
        setup_breakout_section = classified.loc[classified["status_today"].eq("confirmed_breakout_today")].copy()
        setup_stale_hidden_section = classified.loc[classified["status_today"].eq("stale")].copy()
        stale_hidden_count = int(len(setup_stale_hidden_section))
        # The legacy buy_setup section is kept as new+recurring (no stale, no breakout)
        # so existing callers/tests that read sections["buy_setup"] still see the
        # human-visible rows in the same shape they expect.
        setup_section = setup_new_section.copy()
        if not setup_new_section.empty:
            setup_new_section = setup_new_section.sort_values(
                ["persistence_days", "setup_quality_score", "distance_to_trigger_pct", "symbol"],
                ascending=[True, False, True, True],
            ).head(limit_each)
        if not setup_breakout_section.empty:
            setup_breakout_section = setup_breakout_section.sort_values(
                ["setup_quality_score", "distance_to_trigger_pct", "symbol"],
                ascending=[False, True, True],
            ).head(limit_each)

    context = _load_pattern_context()
    draft_rows = pd.concat([buy_draft_raw, watch_draft_raw], ignore_index=True) if not (buy_draft_raw.empty and watch_draft_raw.empty) else pd.DataFrame()
    counts = {
        "buy_candidates": int(len(buy_section)),
        "watchlist": int(len(watchlist_section)),
        "risk_context": int(risk_mask.sum()),
        "buy_setup": int(len(setup_section)),
        "buy_candidates_all_market_before_vn100_filter": int(buy_mask.sum()),
        "watchlist_all_market_before_vn100_filter": int(watchlist_mask.sum()),
        "draft_patterns": int(len(draft_rows)),
    }
    sections = {
        "buy_candidates": _enrich_rows_with_pattern_context(buy_section.head(limit_each).to_dict("records"), context),
        "watchlist": _enrich_rows_with_pattern_context(watchlist_section.head(limit_each).to_dict("records"), context),
        "risk_context": _enrich_rows_with_pattern_context(risk_section.to_dict("records"), context),
        "buy_setup": _enrich_setup_rows(setup_section.to_dict("records")),
        "draft_patterns": _enrich_rows_with_pattern_context(draft_rows.head(limit_each).to_dict("records"), context),
    }
    if filter_active:
        counts["buy_setup_new"] = int(len(setup_new_section))
        counts["buy_setup_breakout_today"] = int(len(setup_breakout_section))
        counts["buy_setup_stale_hidden"] = stale_hidden_count
        sections["buy_setup_new"] = _enrich_setup_rows(setup_new_section.to_dict("records"))
        sections["buy_setup_breakout_today"] = _enrich_setup_rows(setup_breakout_section.to_dict("records"))
        sections["buy_setup_stale_hidden"] = _enrich_setup_rows(setup_stale_hidden_section.to_dict("records"))
    return {
        "workflow_id": WORKFLOW_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stale_threshold": stale_threshold if filter_active else None,
        "total_rows": int(len(watchlist)),
        "counts": counts,
        "display_scope": {
            "buy_candidates": buy_scope,
            "watchlist": watchlist_scope,
        },
        "sections": sections,
    }


def _row_text(row: Mapping[str, Any]) -> str:
    potential = _prob(row.get("potential_profit_pct"))
    probability = _prob(row.get("target_success_probability"))
    path_prob = _prob(row.get("clean_path_probability"))
    current_path = f"đã đi tốt nhất {_pct(row.get('mfe_pct'))}, kéo ngược sâu nhất {_pct_abs(row.get('mae_pct'))}"
    return (
        f"- {_clean(row.get('symbol'))}: {_clean(row.get('pattern_label'))}, xác nhận {_date_label(row.get('event_date'))}. "
        f"Lợi nhuận tiềm năng lịch sử khoảng {potential}; xác suất đạt mục tiêu mẫu khoảng {probability}; "
        f"xác suất đi tới mục tiêu trước khi bất lợi mạnh khoảng {path_prob}. "
        f"Nhóm {_clean(row.get('market_group'))}, thanh khoản {_clean(row.get('liquidity_bucket'))}; {current_path}."
    )


def _setup_row_text(row: Mapping[str, Any]) -> str:
    volume_quality = _volume_label(row.get("volume_quality_label"))
    volume_warning = _volume_label(row.get("volume_warning_label"))
    return (
        f"- {_clean(row.get('symbol'))}: {_clean(row.get('pattern_label'))}, còn cách xác nhận "
        f"{_pct(row.get('distance_to_trigger_pct'))}. Giá hiện tại {_clean(row.get('last_close'))}, "
        f"điểm xác nhận {_clean(row.get('trigger_price'))}, mục tiêu tham khảo {_clean(row.get('target_price'))} "
        f"(dư địa khoảng {_pct(row.get('potential_profit_pct'))}). "
        f"Khối lượng: {volume_quality}; cảnh báo: {volume_warning}. "
        f"Lý do: {_clean(row.get('setup_reason'))}"
    )


def _setup_row_text_v2(row: Mapping[str, Any]) -> str:
    """Setup row text with persistence badge. Breakout rows use a dedicated prefix."""
    base = _setup_row_text(row)
    persistence = row.get("persistence_days")
    status = row.get("status_today")
    if status == "confirmed_breakout_today":
        return f"- {_clean(row.get('symbol'))}: {_clean(row.get('pattern_label'))} — ĐÃ XÁC NHẬN BREAKOUT HÔM NAY (đóng trên {_clean(row.get('trigger_price'))}). {base.lstrip('- ')[len(_clean(row.get('symbol'))) + 2:]}"
    if persistence is None or str(persistence) == "0" or str(persistence) == "":
        badge = "mới"
    else:
        badge = f"theo dõi {persistence} ngày"
    return f"{base} [{badge}]"


def render_text_email(summary: Mapping[str, Any], *, include_risk_details: bool = True) -> str:
    counts = summary["counts"]
    data_refresh = summary.get("data_refresh") if isinstance(summary.get("data_refresh"), Mapping) else None
    stale_threshold = summary.get("stale_threshold")
    filter_active = stale_threshold is not None
    stale_hidden = int(counts.get("buy_setup_stale_hidden", 0)) if filter_active else 0
    buy_setup_rows = summary["sections"].get("buy_setup_new") if filter_active else summary["sections"].get("buy_setup", [])
    breakout_rows = summary["sections"].get("buy_setup_breakout_today", []) if filter_active else []
    lines = [
        "BUY Candidate Scan - VN100 Watchlist",
        "",
        "CẢNH BÁO: Đây là quét hình học từ dữ liệu lịch sử, KHÔNG phải khuyến nghị mua bán. Mỗi mẫu hình có tỉ lệ thất bại thực tế kèm theo. Thống kê gồm cả mã đã ngừng giao dịch.",
        "",
        "⚠️ LƯU Ý: hệ thống đang chuẩn hóa lại số liệu mẫu hình (nâng cấp V3) — các con số thống kê trong mail này là tham khảo tạm, chưa theo chuẩn cuối.",
        "",
        f"Generated at: {summary['generated_at']}",
        f"Total rows: {summary['total_rows']}",
        f"BUY candidates: {counts['buy_candidates']}",
        f"Watchlist: {counts['watchlist']}",
    ]
    if filter_active:
        lines.append(f"BUY setup (mới/vẫn theo dõi): {counts.get('buy_setup_new', 0)}")
        lines.append(f"BUY setup đã xác nhận breakout hôm nay: {counts.get('buy_setup_breakout_today', 0)}")
        if stale_hidden:
            lines.append(f"BUY setup đã ẩn (theo dõi >= {stale_threshold} ngày): {stale_hidden}")
    else:
        lines.append(f"BUY setup đang hình thành: {counts.get('buy_setup', 0)}")
    lines.extend(
        [
            "",
            "Nguyên tắc: đây là danh sách ứng viên để mở chart kiểm tra, không phải khuyến nghị mua bán.",
            "Phạm vi bắt buộc: hai nhóm BUY và Watchlist chỉ lấy VN100/VN30; nếu không có mã phù hợp thì để trống.",
            "Cách đọc: lợi nhuận tiềm năng và xác suất là thống kê lịch sử của mẫu hình, không phải cam kết cho từng mã.",
            "Khả năng chạm trước kéo ngược mạnh: trong lịch sử, mẫu này chạm mục tiêu trước khi bị kéo ngược bất lợi 5% với tần suất bao nhiêu.",
            "Đã tăng / đã kéo ngược: mức tăng tốt nhất và mức kéo ngược sâu nhất đã xảy ra từ ngày xác nhận tới hiện tại.",
        ]
    )
    if data_refresh:
        lines.append(
            "Dữ liệu giá: "
            f"trạng thái {_clean(data_refresh.get('status'))}, "
            f"mới nhất {_clean(data_refresh.get('max_date'))}, "
            f"độ trễ {_clean(data_refresh.get('days_stale'))} ngày."
        )
    if filter_active:
        lines.extend(["", "0a. Đã xác nhận breakout hôm nay"])
        for row in breakout_rows:
            lines.append(_setup_row_text_v2(row))
        if not breakout_rows:
            lines.append("- Không có setup nào đóng vượt điểm xác nhận trong phiên hôm nay.")
        lines.extend(["", "0. Đang hình thành (mới / vẫn theo dõi)"])
        for row in buy_setup_rows:
            lines.append(_setup_row_text_v2(row))
        if not buy_setup_rows:
            lines.append("- Không có setup VN100 mới hoặc đang theo dõi (< ngưỡng ẩn) trong vòng quét hiện tại.")
    else:
        lines.extend(["", "0. Đang hình thành trước xác nhận"])
        for row in buy_setup_rows:
            lines.append(_setup_row_text(row))
        if not buy_setup_rows:
            lines.append("- Không có setup VN100 đủ gần điểm xác nhận trong vòng quét hiện tại.")
    lines.extend(["", "1. Ứng viên BUY tiềm năng"])
    for row in summary["sections"]["buy_candidates"]:
        lines.append(_row_text(row))
    if not summary["sections"]["buy_candidates"]:
        lines.append("- Không có ứng viên BUY mới trong cửa sổ quét.")
    lines.extend(["", "2. Watchlist theo dõi thêm"])
    for row in summary["sections"]["watchlist"]:
        lines.append(_row_text(row))
    if not summary["sections"]["watchlist"]:
        lines.append("- Không có watchlist mới.")
    draft_rows = summary["sections"].get("draft_patterns", [])
    if draft_rows:
        lines.extend(["", f"3. Không đạt chuẩn tín hiệu ({len(draft_rows)} mã, chỉ tham khảo)"])
        for row in draft_rows:
            lines.append(_row_text(row))
        lines.append("- Nhóm này gồm: bản nháp Nấc 1 chưa kiểm định, hoặc mẫu nhỏ dưới 30 lần toàn thị trường, hoặc mục tiêu giữa mẫu dưới 5%, hoặc tỉ lệ vỡ mẫu vượt 2 lần chuẩn sách — không dùng làm tín hiệu, chỉ tham khảo.")
    if filter_active and stale_hidden:
        lines.extend(
            [
                "",
                f"Ghi chú: {stale_hidden} mã đã theo dõi liên tục >= {stale_threshold} ngày được ẩn khỏi mail để giảm tiếng ồn; xem đầy đủ trong CSV đính kèm.",
            ]
        )
    lines.extend(["", "Checklist đọc thủ công: chart, thanh khoản, VNINDEX/regime, tin tức, vị trí giá, kế hoạch rủi ro."])
    return "\n".join(lines) + "\n"


def _html_rows(rows: list[Mapping[str, Any]]) -> str:
    if not rows:
        return "<p class=\"muted\">Không có dòng mới trong nhóm này.</p>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(_clean(row.get('symbol')))}</td>"
            f"<td>{html.escape(_clean(row.get('pattern_label')))}</td>"
            f"<td>{html.escape(_date_label(row.get('event_date')))}</td>"
            f"<td>{html.escape(_clean(row.get('market_group')))}</td>"
            f"<td>{html.escape(_prob(row.get('potential_profit_pct')))}</td>"
            f"<td>{html.escape(_prob(row.get('target_success_probability')))}</td>"
            f"<td>{html.escape(_prob(row.get('clean_path_probability')))}</td>"
            f"<td>{html.escape(_pct(row.get('mfe_pct')))} / {html.escape(_pct_abs(row.get('mae_pct')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Mã</th><th>Mẫu hình tiềm năng</th><th>Ngày xác nhận</th><th>Nhóm</th>"
        "<th>Lợi nhuận tiềm năng</th><th>Xác suất đạt mục tiêu</th><th>Khả năng chạm trước kéo ngược mạnh</th><th>Đã tăng / đã kéo ngược</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _html_setup_rows(rows: list[Mapping[str, Any]]) -> str:
    if not rows:
        return "<p class=\"muted\">Không có setup VN100 đủ gần điểm xác nhận trong vòng quét hiện tại.</p>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(_clean(row.get('symbol')))}</td>"
            f"<td>{html.escape(_clean(row.get('pattern_label')))}</td>"
            f"<td>{html.escape(_clean(row.get('market_group')))}</td>"
            f"<td>{html.escape(_pct(row.get('distance_to_trigger_pct')))}</td>"
            f"<td>{html.escape(_clean(row.get('last_close')))}</td>"
            f"<td>{html.escape(_clean(row.get('trigger_price')))}</td>"
            f"<td>{html.escape(_clean(row.get('target_price')))}</td>"
            f"<td>{html.escape(_pct(row.get('potential_profit_pct')))}</td>"
            f"<td>{html.escape(_volume_label(row.get('volume_quality_label')))}</td>"
            f"<td>{html.escape(_volume_label(row.get('volume_warning_label')))}</td>"
            f"<td>{html.escape(_clean(row.get('setup_reason')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Mã</th><th>Mẫu đang hình thành</th><th>Nhóm</th><th>Còn cách xác nhận</th>"
        "<th>Giá hiện tại</th><th>Điểm xác nhận</th><th>Mục tiêu tham khảo</th><th>Dư địa tham khảo</th><th>Sức khối lượng</th><th>Cảnh báo</th><th>Lý do</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _html_setup_rows_v2(rows: list[Mapping[str, Any]], *, empty_message: str | None = None) -> str:
    """Setup table variant with a 'Theo dõi' badge column reflecting persistence/status_today."""
    if not rows:
        return f"<p class=\"muted\">{html.escape(empty_message or 'Không có setup trong nhóm này.')}</p>"
    body = []
    for row in rows:
        persistence = row.get("persistence_days")
        status = row.get("status_today")
        if status == "confirmed_breakout_today":
            badge = '<span class="badge badge-breakout">⚡ Đã xác nhận breakout hôm nay</span>'
        elif persistence is None or str(persistence) == "0" or str(persistence) == "":
            badge = '<span class="badge badge-new">Mới</span>'
        else:
            badge = f'<span class="badge badge-recurring">Theo dõi {html.escape(str(persistence))} ngày</span>'
        body.append(
            "<tr>"
            f"<td>{html.escape(_clean(row.get('symbol')))}</td>"
            f"<td>{html.escape(_clean(row.get('pattern_label')))}</td>"
            f"<td>{html.escape(_clean(row.get('market_group')))}</td>"
            f"<td>{html.escape(_pct(row.get('distance_to_trigger_pct')))}</td>"
            f"<td>{html.escape(_clean(row.get('last_close')))}</td>"
            f"<td>{html.escape(_clean(row.get('trigger_price')))}</td>"
            f"<td>{html.escape(_clean(row.get('target_price')))}</td>"
            f"<td>{html.escape(_pct(row.get('potential_profit_pct')))}</td>"
            f"<td>{html.escape(_volume_label(row.get('volume_quality_label')))}</td>"
            f"<td>{html.escape(_volume_label(row.get('volume_warning_label')))}</td>"
            f"<td>{badge}</td>"
            f"<td>{html.escape(_clean(row.get('setup_reason')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Mã</th><th>Mẫu đang hình thành</th><th>Nhóm</th><th>Còn cách xác nhận</th>"
        "<th>Giá hiện tại</th><th>Điểm xác nhận</th><th>Mục tiêu tham khảo</th><th>Dư địa tham khảo</th><th>Sức khối lượng</th><th>Cảnh báo</th><th>Trạng thái hôm nay</th><th>Lý do</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def render_html_email(summary: Mapping[str, Any], *, include_risk_details: bool = True) -> str:
    counts = summary["counts"]
    data_refresh = summary.get("data_refresh") if isinstance(summary.get("data_refresh"), Mapping) else None
    data_refresh_html = ""
    if data_refresh:
        data_refresh_html = (
            "<p class=\"note\"><b>Dữ liệu giá</b>: "
            f"trạng thái {html.escape(_clean(data_refresh.get('status')))}, "
            f"mới nhất {html.escape(_clean(data_refresh.get('max_date')))}, "
            f"độ trễ {html.escape(_clean(data_refresh.get('days_stale')))} ngày.</p>"
        )
    stale_threshold = summary.get("stale_threshold")
    filter_active = stale_threshold is not None
    stale_hidden = int(counts.get("buy_setup_stale_hidden", 0)) if filter_active else 0
    buy_setup_rows = summary["sections"].get("buy_setup_new") if filter_active else summary["sections"].get("buy_setup", [])
    breakout_rows = summary["sections"].get("buy_setup_breakout_today", []) if filter_active else []
    setup_card_count = counts.get("buy_setup_new", 0) + counts.get("buy_setup_breakout_today", 0) if filter_active else counts.get("buy_setup", 0)
    extra_style = (
        """
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
    .badge-new {{ background: #e7f3ec; color: #1f6b61; }}
    .badge-recurring {{ background: #fbf1d6; color: #8a6a1f; }}
    .badge-breakout {{ background: #1f6b61; color: #ffffff; }}
    .breakout-box {{ border-left: 4px solid #1f6b61; padding: 10px 12px; background: #f1f7f4; margin-top: 8px; }}
    .stale-note {{ background: #fbf1d6; border-left: 4px solid #c9a227; padding: 10px 12px; margin-top: 16px; font-size: 13px; color: #4f5c57; }}
""" if filter_active else ""
    )
    breakout_box_html = ""
    if filter_active:
        breakout_count = counts.get("buy_setup_breakout_today", 0)
        breakout_box_html = (
            f'<h2>0a. Đã xác nhận breakout hôm nay ({breakout_count})</h2>\n'
            f'    {_html_setup_rows_v2(breakout_rows, empty_message="Không có setup nào đóng vượt điểm xác nhận trong phiên hôm nay.")}'
        )
    setup_heading = "0. Đang hình thành (mới / vẫn theo dõi)" if filter_active else "0. Đang hình thành trước xác nhận"
    setup_table_html = _html_setup_rows_v2(buy_setup_rows) if filter_active else _html_setup_rows(buy_setup_rows)
    stale_note_html = (
        f'<p class="stale-note"><b>Đã ẩn {stale_hidden} mã</b> theo dõi liên tục &ge; {html.escape(str(stale_threshold))} ngày để giảm tiếng ồn; xem đầy đủ trong CSV đính kèm.</p>'
        if filter_active and stale_hidden
        else ""
    )
    draft_html = ""
    if summary["sections"].get("draft_patterns"):
        draft_html = (
            '<h2>3. Không đạt chuẩn tín hiệu (chỉ tham khảo)</h2>\n'
            f'{_html_rows(summary["sections"]["draft_patterns"])}'
            '<p class="note">Nhóm này gồm: bản nháp Nấc 1 chưa kiểm định, hoặc mẫu nhỏ dưới 30 lần toàn thị trường, hoặc mục tiêu giữa mẫu dưới 5%, hoặc tỉ lệ vỡ mẫu vượt 2 lần chuẩn sách — không dùng làm tín hiệu, chỉ tham khảo.</p>'
        )
    cards_html = (
        f"""
    <div class="cards">
      <div class="card"><div class="num">{counts["buy_candidates"]}</div><div>Ứng viên BUY tiềm năng</div></div>
      <div class="card"><div class="num">{counts["watchlist"]}</div><div>Watchlist theo dõi thêm</div></div>
      <div class="card"><div class="num">{counts.get("buy_setup_new", 0)}</div><div>Setup mới / vẫn theo dõi</div></div>
      <div class="card"><div class="num">{counts.get("buy_setup_breakout_today", 0)}</div><div>Đã breakout hôm nay</div></div>
    </div>"""
        if filter_active
        else f"""
    <div class="cards">
      <div class="card"><div class="num">{counts["buy_candidates"]}</div><div>Ứng viên BUY tiềm năng</div></div>
      <div class="card"><div class="num">{counts["watchlist"]}</div><div>Watchlist theo dõi thêm</div></div>
      <div class="card"><div class="num">{counts.get("buy_setup", 0)}</div><div>Setup đang hình thành</div></div>
    </div>"""
    )
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #153f3a; background: #fbf7ef; margin: 0; padding: 24px; }}
    .wrap {{ max-width: 960px; margin: 0 auto; background: #fffdf8; border: 1px solid #ded4c4; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    .muted {{ color: #6f746d; }}
    .cards {{ display: flex; gap: 12px; margin: 18px 0; }}
    .card {{ flex: 1; border: 1px solid #ded4c4; padding: 12px; background: #f6f1e8; }}
    .num {{ font-size: 28px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5ddd0; padding: 8px; text-align: left; }}
    th {{ background: #eee8dc; }}
    .notice {{ border-left: 4px solid #1f6b61; padding: 10px 12px; background: #f1f7f4; }}
    .note {{ color: #4f5c57; font-size: 13px; line-height: 1.45; }}{extra_style}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>BUY Candidate Scan - VN100 Watchlist</h1>
    <p class="muted">Generated at: {html.escape(str(summary["generated_at"]))}</p>
    <div class="notice"><b>CẢNH BÁO:</b> Đây là quét hình học từ dữ liệu lịch sử, KHÔNG phải khuyến nghị mua bán. Mỗi mẫu hình có tỉ lệ thất bại thực tế kèm theo. Thống kê gồm cả mã đã ngừng giao dịch.</div>
    <div class="notice"><b>⚠️ Lưu ý:</b> hệ thống đang chuẩn hóa lại số liệu mẫu hình (nâng cấp V3) — các con số thống kê trong mail này là tham khảo tạm, chưa theo chuẩn cuối.</div>
    <div class="notice">Đây là danh sách ứng viên để mở chart kiểm tra, không phải khuyến nghị mua bán.</div>
    <p class="note">Phạm vi bắt buộc: hai nhóm BUY và Watchlist chỉ lấy VN100/VN30; nếu không có mã phù hợp thì để trống. Lợi nhuận tiềm năng và xác suất là thống kê lịch sử của mẫu hình, không phải cam kết cho từng mã.</p>
    <p class="note"><b>Khả năng chạm trước kéo ngược mạnh</b> là tần suất lịch sử mẫu chạm mục tiêu trước khi bị kéo ngược bất lợi 5%. <b>Đã tăng / đã kéo ngược</b> là mức tăng tốt nhất và mức kéo ngược sâu nhất từ ngày xác nhận tới hiện tại.</p>
    {data_refresh_html}{cards_html}
    {breakout_box_html}
    <h2>{setup_heading}</h2>
    {setup_table_html}
    {stale_note_html}
    <h2>1. Ứng viên BUY tiềm năng</h2>
    {_html_rows(summary["sections"]["buy_candidates"])}
    <h2>2. Watchlist theo dõi thêm</h2>
    {_html_rows(summary["sections"]["watchlist"])}
    {draft_html}
    <p class="muted">Checklist đọc thủ công: chart, thanh khoản, VNINDEX/regime, tin tức, vị trí giá, kế hoạch rủi ro.</p>
  </div>
</body>
</html>
"""


def write_email_artifacts(summary: Mapping[str, Any], out_dir: Path = DEFAULT_EMAIL_OUT_DIR, *, include_risk_details: bool = True) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "realtime_scan_email_summary.json"
    text_path = out_dir / "realtime_scan_email.txt"
    html_path = out_dir / "realtime_scan_email.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    text_path.write_text(render_text_email(summary, include_risk_details=include_risk_details), encoding="utf-8")
    html_path.write_text(render_html_email(summary, include_risk_details=include_risk_details), encoding="utf-8")
    return {"summary_json": str(json_path), "text": str(text_path), "html": str(html_path)}


def send_email(
    summary: Mapping[str, Any],
    config: EmailConfig,
    *,
    include_risk_details: bool = True,
    attachment_paths: list[Path] | None = None,
) -> None:
    counts = summary["counts"]
    subject = (
        f"[BUY Scan] VN100 Setup {counts.get('buy_setup', 0)} | BUY {counts['buy_candidates']} | Watchlist {counts['watchlist']}"
    )
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.mail_from
    msg["To"] = ", ".join(config.mail_to)
    msg.set_content(render_text_email(summary, include_risk_details=include_risk_details))
    msg.add_alternative(render_html_email(summary, include_risk_details=include_risk_details), subtype="html")
    for path in attachment_paths or []:
        if not path.exists():
            continue
        msg.add_attachment(
            path.read_bytes(),
            maintype="application",
            subtype="pdf",
            filename=path.name,
        )

    context = ssl.create_default_context()
    smtp_cls = smtplib.SMTP_SSL if config.use_ssl else smtplib.SMTP
    with smtp_cls(config.smtp_host, config.smtp_port, timeout=config.timeout) as smtp:
        if config.use_tls:
            smtp.starttls(context=context)
        if config.smtp_user:
            smtp.login(config.smtp_user, config.smtp_password or "")
        smtp.send_message(msg)


def _compact_data_refresh_report(report: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    snapshot = report.get("snapshot_after") if isinstance(report.get("snapshot_after"), Mapping) else None
    if not snapshot:
        snapshot = report.get("snapshot_before") if isinstance(report.get("snapshot_before"), Mapping) else {}
    freshness = report.get("freshness_after") if isinstance(report.get("freshness_after"), Mapping) else None
    if not freshness:
        freshness = report.get("freshness") if isinstance(report.get("freshness"), Mapping) else {}
    provider = report.get("provider") if isinstance(report.get("provider"), Mapping) else {}
    return {
        "status": report.get("status"),
        "max_date": snapshot.get("max_date"),
        "row_count": snapshot.get("row_count"),
        "symbol_count": snapshot.get("symbol_count"),
        "days_stale": freshness.get("days_stale"),
        "provider_available": provider.get("available"),
        "report_path": report.get("report_path"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and optionally send realtime scan candidate email.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--email-out-dir", default=str(DEFAULT_EMAIL_OUT_DIR))
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--limit-each", type=int, default=20)
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--after-buy-config", default=str(DEFAULT_AFTER_BUY_CONFIG))
    parser.add_argument("--send", action="store_true", help="Send email using SMTP env vars. Without this, only writes preview artifacts.")
    parser.add_argument("--focus-buy-watchlist", action="store_true", help="Keep risk/caution as a count only so the email focuses on BUY candidates and watchlist rows.")
    parser.add_argument("--attach-pdf", action="store_true", help="Render and attach the detailed realtime scan PDF report.")
    parser.add_argument("--pdf-out", default=str(DEFAULT_PDF_OUT), help="Output path for the detailed PDF report when --attach-pdf is used.")
    parser.add_argument("--pdf-chart-dir", help="Output directory for PDF charts. Defaults to a charts folder beside --pdf-out.")
    parser.add_argument("--include-buy-setup", action="store_true", help="Also scan current VN100 for pre-breakout BUY_SETUP candidates.")
    parser.add_argument("--buy-setup-out-dir", default=str(DEFAULT_BUY_SETUP_OUT_DIR))
    parser.add_argument("--buy-setup-limit-per-pattern", type=int, default=8)
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR), help="Persistent candidate ledger directory.")
    parser.add_argument("--history-horizon-days", type=int, default=120)
    parser.add_argument("--skip-history", action="store_true", help="Do not update persistent realtime scan history.")
    parser.add_argument("--stale-threshold", type=int, default=10, help="Hide setups seen for >= N consecutive days from the email body (0 disables).")
    parser.add_argument("--refresh-data", action="store_true", help="Refresh/audit the Market Stats OHLCV cache before scanning.")
    parser.add_argument("--refresh-data-out-dir", default=str(DEFAULT_DATA_REFRESH_OUT_DIR))
    parser.add_argument("--refresh-source", default="VCI", choices=["VND", "VCI", "KBS", "MAS"])
    parser.add_argument("--refresh-rpm", type=int, default=180)
    parser.add_argument("--refresh-max-errors", type=int, default=80)
    parser.add_argument("--refresh-timeout-seconds", type=int, default=10)
    parser.add_argument("--refresh-command-timeout-seconds", type=int, default=3600)
    parser.add_argument("--refresh-staleness-days", type=int, default=0)
    parser.add_argument("--refresh-min-latest-symbols", type=int, default=0)
    parser.add_argument("--refresh-end")
    parser.add_argument("--refresh-symbol", action="append", default=[])
    parser.add_argument("--refresh-force", action="store_true", help="Force provider refresh even when the local DB looks fresh.")
    parser.add_argument("--strict-refresh", action="store_true", help="Fail the email scan when data refresh is stale/blocked/failed.")
    parser.add_argument("--skip-regenerate-market-stats", action="store_true", help="Do not regenerate market_stats web artifacts after DB refresh.")
    parser.add_argument("--refresh-python-executable", help="Override the Python runtime used for vnstock_data refresh.")
    args = parser.parse_args()

    data_refresh_report: dict[str, Any] | None = None
    if args.refresh_data:
        data_refresh_report = refresh_realtime_market_data(
            db_path=Path(args.db),
            out_dir=Path(args.refresh_data_out_dir),
            force=bool(args.refresh_force),
            strict=bool(args.strict_refresh),
            source=str(args.refresh_source),
            rpm=int(args.refresh_rpm),
            max_errors=int(args.refresh_max_errors),
            timeout_seconds=int(args.refresh_timeout_seconds),
            command_timeout_seconds=int(args.refresh_command_timeout_seconds),
            staleness_days=int(args.refresh_staleness_days),
            min_latest_symbols=int(args.refresh_min_latest_symbols),
            end=args.refresh_end,
            symbols=list(args.refresh_symbol) or None,
            regenerate_market_stats=not bool(args.skip_regenerate_market_stats),
            python_executable=args.refresh_python_executable,
        )

    plan = build_realtime_scan_plan(db_path=Path(args.db), out_root=Path(args.out_dir), patterns=list(args.pattern) or None)
    watchlist = build_watchlist_from_artifacts(
        plan,
        lookback_days=int(args.lookback_days),
        after_buy_config_path=Path(args.after_buy_config),
    )
    realtime_paths = write_realtime_outputs(plan, watchlist, Path(args.out_dir))
    buy_setups = pd.DataFrame()
    setup_paths: dict[str, str] = {}
    setup_meta: dict[str, Any] | None = None
    if args.include_buy_setup:
        buy_setups, setup_meta = scan_buy_setups(
            db_path=Path(args.db),
            after_buy_config_path=Path(args.after_buy_config),
            limit_per_pattern=int(args.buy_setup_limit_per_pattern),
        )
        setup_paths = {f"buy_setup_{key}": value for key, value in write_buy_setup_outputs(buy_setups, setup_meta, Path(args.buy_setup_out_dir)).items()}

    # Pin a single timestamp for this run so history-update and the email summary
    # stay in sync. The history ledger must include today's row before we read
    # persistence back into the email, so update_history runs first.
    generated_at = datetime.now().isoformat(timespec="seconds")
    history_report: dict[str, Any] | None = None
    if not args.skip_history:
        history_report = update_realtime_scan_history(
            buy_setups=buy_setups if args.include_buy_setup else pd.DataFrame(),
            watchlist=watchlist,
            db_path=Path(args.db),
            history_dir=Path(args.history_dir),
            horizon_days=int(args.history_horizon_days),
            generated_at=generated_at,
        )

    stale_filter_active = bool(args.include_buy_setup and args.stale_threshold > 0 and not args.skip_history)
    summary = summarize_watchlist(
        watchlist,
        limit_each=int(args.limit_each),
        buy_setups=buy_setups if args.include_buy_setup else None,
        history_dir=Path(args.history_dir) if stale_filter_active else None,
        db_path=Path(args.db) if stale_filter_active else None,
        stale_threshold=int(args.stale_threshold),
        generated_at=generated_at,
    )
    if data_refresh_report:
        summary["data_refresh"] = _compact_data_refresh_report(data_refresh_report)
    include_risk_details = not bool(args.focus_buy_watchlist)
    email_paths = write_email_artifacts(summary, Path(args.email_out_dir), include_risk_details=include_risk_details)
    pdf_report: dict[str, Any] | None = None
    attachment_paths: list[Path] = []
    if args.attach_pdf:
        pdf_chart_dir = Path(args.pdf_chart_dir) if args.pdf_chart_dir else Path(args.pdf_out).parent / "charts"
        pdf_report = build_realtime_scan_pdf_report(summary, pdf_path=Path(args.pdf_out), chart_dir=pdf_chart_dir)
        email_paths["detail_pdf"] = str(pdf_report["pdf_path"])
        email_paths["detail_pdf_report"] = str(pdf_report["report_path"])
        attachment_paths.append(Path(str(pdf_report["pdf_path"])))

    sent = False
    if args.send:
        send_email(
            summary,
            load_email_config_from_env(),
            include_risk_details=include_risk_details,
            attachment_paths=attachment_paths,
        )
        sent = True
    print(
        json.dumps(
            {
                "workflow_id": WORKFLOW_ID,
                "status": "PASS",
                "sent": sent,
                "counts": summary["counts"],
                "data_refresh_report": data_refresh_report,
                "buy_setup_meta": setup_meta,
                "history_report": history_report,
                "pdf_report": pdf_report,
                "paths": {**realtime_paths, **setup_paths, **email_paths},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
