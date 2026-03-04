"""
Build a Vietnamese Bulkowski-style "book" from scan results.

This script assembles:
  - deterministic tables/statistics (from results DBs)
  - optional AI-written narrative (via DeepSeek API)
  - optional figures (candlestick windows for selected cases)
  - a single Markdown file ready for PDF export (via pandoc if installed)

Typical workflow:
  1) Run scans (calibration + validation) with `scanner/run_full_scan.py`
  2) Generate book Markdown (and optionally PDF):
       export DEEPSEEK_API_KEY=...
       python3 scanner/build_book_vi.py \\
         --results-db-valid scan_results/valid_2022_2025_vn30_eval.sqlite \\
         --results-db-calib scan_results/calib_2018_2021_vn30_eval.sqlite \\
         --price-db vietnam_stocks.db \\
         --index-symbol VN30 \\
         --out-dir scan_results/book_vi

Notes:
  - If pandoc isn't installed, the script will still generate Markdown outputs.
  - Narrative generation is designed to avoid inventing statistics: numbers live in tables
    we generate deterministically; the AI is instructed to write only qualitative text.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from .bulkowski_report import generate_bulkowski_payload  # type: ignore
except Exception:  # pragma: no cover
    from bulkowski_report import generate_bulkowski_payload  # type: ignore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: str) -> str:
    p = os.path.abspath(path)
    os.makedirs(p, exist_ok=True)
    return p


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str, ensure_ascii=False)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _breakable_key(s: Any) -> str:
    """
    Insert zero-width spaces after underscores so long identifiers can wrap in PDF.
    """
    return str(s).replace("_", "_\u200b")


def _bulkowski_base_name(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return ""
    if " (" in s:
        s = s.split(" (", 1)[0].strip()
    if "," in s:
        s = s.split(",", 1)[0].strip()
    return s


def _run_id_from_results_db(results_db_path: str) -> str:
    conn = sqlite3.connect(os.path.abspath(results_db_path))
    try:
        row = conn.execute("SELECT run_id FROM scanner_runs ORDER BY created_at DESC LIMIT 1").fetchone()
        if not row:
            raise SystemExit(f"No runs in scanner_runs for results_db={results_db_path}")
        return str(row[0])
    finally:
        conn.close()


def _pattern_meta_map_from_results_db(results_db_path: str, run_id: str) -> Dict[str, Dict[str, Any]]:
    conn = sqlite3.connect(os.path.abspath(results_db_path))
    try:
        row = conn.execute("SELECT run_config_json FROM scanner_runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row or not row[0]:
            return {}
        try:
            cfg = json.loads(row[0])
        except Exception:
            return {}
        meta = cfg.get("pattern_metadata")
        if not isinstance(meta, dict):
            return {}
        pats = meta.get("patterns")
        return pats if isinstance(pats, dict) else {}
    finally:
        conn.close()


def _group_order_from_meta(meta_map: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Return canonical keys ordered by minimum Bulkowski chapter (stable book ordering).
    """
    groups: Dict[str, List[int]] = {}
    for pk, m in meta_map.items():
        if not isinstance(m, dict):
            continue
        ck = str(m.get("canonical_key") or m.get("spec_key") or pk)
        chap = m.get("bulkowski_chapter")
        if isinstance(chap, int):
            groups.setdefault(ck, []).append(int(chap))
        else:
            groups.setdefault(ck, [])
    ordered = sorted(groups.items(), key=lambda kv: (min(kv[1]) if kv[1] else 10**9, kv[0]))
    return [k for k, _ in ordered]


def _pattern_order_from_meta(meta_map: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Return pattern_keys ordered by Bulkowski chapter (stable chapter ordering).
    Falls back to pattern_key lexical order when chapter is missing.
    """
    rows: List[Tuple[int, str]] = []
    for pk, m in meta_map.items():
        if not isinstance(m, dict):
            continue
        chap = m.get("bulkowski_chapter")
        c = int(chap) if isinstance(chap, int) else 10**9
        rows.append((c, str(pk)))
    rows.sort(key=lambda x: (x[0], x[1]))
    return [pk for _, pk in rows]


def _summary_groups_by(
    payload: Dict[str, Any],
    *,
    group_by: str,
) -> List[Dict[str, Any]]:
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(summary, dict):
        return []
    groups = summary.get("groups")
    if not isinstance(groups, list):
        return []
    # payload already represents a group_by; caller uses separate payloads.
    return [g for g in groups if isinstance(g, dict)]


def _pivot_spec_dir() -> str:
    return os.path.abspath(os.path.join("extraction_phase_1", "digitization", "patterns_digitized"))


def _load_digitized_spec(spec_key: str) -> Optional[Dict[str, Any]]:
    spec_dir = _pivot_spec_dir()
    path = os.path.join(spec_dir, f"{spec_key}_digitized.json")
    if not os.path.exists(path):
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _spec_summary_vi(spec_key: str) -> List[str]:
    spec = _load_digitized_spec(spec_key) or {}
    geom = spec.get("geometry_constraints") if isinstance(spec, dict) else {}
    sig = spec.get("detection_signature") if isinstance(spec, dict) else {}
    brk = spec.get("breakout_confirmation") if isinstance(spec, dict) else {}
    piv = spec.get("pivot_requirements") if isinstance(spec, dict) else {}

    lines: List[str] = []
    if spec.get("pattern_name"):
        lines.append(f"- Tên spec: {_breakable_key(spec_key)} ({spec.get('pattern_name')})")
    else:
        lines.append(f"- Tên spec: {_breakable_key(spec_key)}")

    if isinstance(sig, dict):
        seq = sig.get("pivot_sequence")
        if isinstance(seq, list) and seq:
            lines.append(f"- Pivot signature: `{''.join(str(x) for x in seq)}` (min_pivots={sig.get('min_pivots')})")

    if isinstance(piv, dict):
        pt = piv.get("pivot_type")
        lb = piv.get("pivot_lookback")
        la = piv.get("pivot_lookahead")
        if pt or lb or la:
            lines.append(f"- Pivot config: type={pt}, lookback={lb}, lookahead={la}")

    if isinstance(geom, dict):
        wmin, wmax = geom.get("width_min_bars"), geom.get("width_max_bars")
        hmin, hmax = geom.get("height_ratio_min"), geom.get("height_ratio_max")
        if wmin is not None or wmax is not None:
            lines.append(f"- Độ rộng (bars): {wmin} → {wmax}")
        if hmin is not None or hmax is not None:
            lines.append(f"- Độ cao (%): {hmin} → {hmax}")
        if geom.get("near_equal_tolerance_pct") is not None:
            lines.append(f"- Near-equal tolerance: {geom.get('near_equal_tolerance_pct')}%")

    if isinstance(brk, dict):
        thr = brk.get("breakout_threshold_pct")
        vol_req = brk.get("volume_required")
        vol_mult = brk.get("volume_multiplier_min")
        if thr is not None:
            lines.append(f"- Breakout: close vượt biên {thr}%")
        if vol_req is not None:
            lines.append(f"- Volume confirm: {'có' if bool(vol_req) else 'không'} (min_mult={vol_mult})")

    return lines


def _format_group_table_md(groups: List[Dict[str, Any]], *, title: str) -> str:
    if not groups:
        return f"### {title}\n\n(Không có dữ liệu)\n"

    # Split into two narrower tables to reduce PDF layout overflow (Overfull \\hbox).
    cols_a = [
        ("market_regime", "Regime"),
        ("breakout_direction", "Dir"),
        ("n_evaluated", "n_eval"),
        ("n_confirmed", "n_conf"),
        ("median_move_pct", "MedMove%"),
        ("median_days_to_ultimate", "MedDays"),
    ]
    cols_b = [
        ("market_regime", "Regime"),
        ("breakout_direction", "Dir"),
        ("throwback_pullback_rate_pct", "TB/PB%"),
        ("failure_lt_5pct_rate_pct", "Fail<5%"),
        ("target_hit_rate_intraday_pct", "TgtHit%"),
        ("boundary_invalidation_rate_pct", "Bound%"),
        ("busted_5pct_rate_pct", "Bust5%"),
    ]

    def _fmt(x: Any) -> str:
        if x is None:
            return ""
        try:
            if isinstance(x, bool):
                return "1" if x else "0"
            if isinstance(x, (int, np.integer)):
                return str(int(x))
            v = float(x)
        except Exception:
            return str(x)
        if not np.isfinite(v):
            return ""
        # integers (days) vs pct
        if abs(v) >= 1000:
            return f"{v:.0f}"
        if float(int(v)) == v:
            return f"{v:.0f}"
        return f"{v:.2f}"

    def _render(cols: List[Tuple[str, str]]) -> str:
        header = "| " + " | ".join(h for _, h in cols) + " |\n"
        sep = "| " + " | ".join("---" for _ in cols) + " |\n"
        rows = []
        for g in groups:
            row = []
            for key, _ in cols:
                row.append(_fmt(g.get(key)))
            rows.append("| " + " | ".join(row) + " |")
        return header + sep + "\n".join(rows) + "\n"

    return "### " + title + "\n\n" + _render(cols_a) + "\n" + _render(cols_b)


def _deepseek_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout_s: int = 120,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {"model": model, "messages": messages, "temperature": 0.2}
    data = json.dumps(body).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"DeepSeek HTTPError {getattr(e, 'code', '?')}: {msg}")
    except URLError as e:
        raise RuntimeError(f"DeepSeek URLError: {e}")

    try:
        return str(payload["choices"][0]["message"]["content"])
    except Exception:
        raise RuntimeError(f"Unexpected DeepSeek response shape: {payload}")


def _generate_narrative_vi(
    *,
    context: Dict[str, Any],
    table_md: str,
    api_key: str,
    base_url: str,
    model: str,
) -> str:
    """
    Generate qualitative Vietnamese narrative only (avoid introducing numbers).
    """
    pat_name = str(context.get("pattern_display_name") or context.get("canonical_key") or "").strip()
    spec_key = str(context.get("spec_key") or "").strip()

    system = (
        "Bạn là trợ lý nghiên cứu thị trường tài chính. "
        "Hãy viết tiếng Việt, giọng văn trung tính, phục vụ nghiên cứu. "
        "Tuyệt đối không bịa số liệu; tránh dùng chữ số (0-9). "
        "Không trích dẫn nguyên văn sách có bản quyền."
    )

    user = (
        f"Hãy viết phần lời bình (narrative) cho pattern: {pat_name}.\n\n"
        "Thông tin kỹ thuật (spec tóm tắt):\n"
        + "\n".join(str(x) for x in context.get("spec_summary_vi", []))
        + "\n\n"
        "Bảng thống kê (đã có sẵn, không cần nhắc số cụ thể):\n"
        + table_md
        + "\n\n"
        "Yêu cầu đầu ra:\n"
        "- Viết 4 mục (dùng Markdown headings cấp 3):\n"
        "  1) Nhận xét nhanh\n"
        "  2) Cách nhận diện trên dữ liệu OHLCV (theo logic scanner)\n"
        "  3) Lưu ý sai lệch/false positives cần calibration\n"
        "  4) Ý tưởng nghiên cứu tiếp theo trên dữ liệu Việt Nam\n"
        "- Không dùng bất kỳ chữ số nào (0-9). Nếu bắt buộc, viết bằng chữ.\n"
        "- Không thêm bảng số liệu mới.\n"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return _deepseek_chat_completion(base_url=base_url, api_key=api_key, model=model, messages=messages)


def _load_dotenv_if_present() -> None:
    """
    Best-effort dotenv loader (no external dependencies).

    Loads variables from repo-root `.env` (sibling of `scanner/`) if present.
    Does NOT override existing environment variables.
    """

    def _parse_line(line: str) -> Optional[Tuple[str, str]]:
        s = line.strip()
        if not s or s.startswith("#"):
            return None
        if s.startswith("export "):
            s = s[len("export ") :].strip()
        if "=" not in s:
            return None
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            return None
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1]
        return (k, v)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    dotenv_path = os.path.join(repo_root, ".env")
    if not os.path.exists(dotenv_path):
        return
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                kv = _parse_line(line)
                if not kv:
                    continue
                k, v = kv
                if k not in os.environ:
                    os.environ[k] = v
    except Exception:
        return


def _placeholder_narrative_md(*, note: Optional[str] = None) -> str:
    note_s = ""
    if note:
        note_s = " ".join(str(note).splitlines()).strip()
        if len(note_s) > 500:
            note_s = note_s[:500].rstrip() + "..."

    parts: List[str] = []
    parts.extend(
        [
            "### Nhận xét nhanh",
            "",
            "_(Chưa sinh narrative)_",
        ]
    )
    if note_s:
        parts.extend(["", f"_(Ghi chú: {note_s})_"])

    parts.extend(
        [
            "",
            "### Cách nhận diện trên dữ liệu OHLCV (theo logic scanner)",
            "",
            "_(Chưa sinh narrative)_",
            "",
            "### Lưu ý sai lệch/false positives cần calibration",
            "",
            "_(Chưa sinh narrative)_",
            "",
            "### Ý tưởng nghiên cứu tiếp theo trên dữ liệu Việt Nam",
            "",
            "_(Chưa sinh narrative)_",
        ]
    )
    return "\n".join(parts).strip() + "\n"


def _normalize_narrative_headings_md(text: str) -> str:
    """
    Normalize DeepSeek narrative headings to the canonical 4-section layout.

    This keeps content intact but makes validation + book consistency stable,
    even if the model omits small parentheticals or has minor typos.
    """
    if not text:
        return text

    # Small VI typo fixes (post-editing).
    raw = str(text).replace("giai đoịan", "giai đoạn").replace("đoịan", "đoạn")

    out_lines: List[str] = []
    re_calib = re.compile(r"\bcalib(?:ration)?\b", flags=re.IGNORECASE)

    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("### "):
            head = s[4:].strip()
            h = head.lower()

            if "nhận xét" in h:
                out_lines.append("### Nhận xét nhanh")
                continue
            if "cách nhận diện" in h:
                out_lines.append("### Cách nhận diện trên dữ liệu OHLCV (theo logic scanner)")
                continue
            if ("lưu ý" in h) or ("false positives" in h) or ("sai lệch" in h):
                out_lines.append("### Lưu ý sai lệch/false positives cần calibration")
                continue
            if ("nghiên cứu" in h) and ("tưởng" in h or "y tuong" in h or "ý" in h):
                out_lines.append("### Ý tưởng nghiên cứu tiếp theo trên dữ liệu Việt Nam")
                continue

        if s and (not s.startswith("### ")):
            line = re_calib.sub("hiệu chỉnh", line)
        out_lines.append(line)

    return "\n".join(out_lines).strip() + "\n"


@dataclass(frozen=True)
class CaseRow:
    pattern_id: str
    symbol: str
    pattern_key: str
    formation_start: str
    formation_end: str
    breakout_date: Optional[str]
    breakout_direction: Optional[str]
    breakout_price: Optional[float]
    target_price: Optional[float]
    stop_loss_price: Optional[float]
    confidence_score: int


def _select_cases(
    *,
    results_db_path: str,
    run_id: str,
    pattern_keys: List[str],
    max_cases_per_direction: int = 1,
) -> List[CaseRow]:
    conn = sqlite3.connect(os.path.abspath(results_db_path))
    try:
        q = f"""
        SELECT
            d.pattern_id,
            d.symbol,
            d.pattern_name,
            d.formation_start,
            d.formation_end,
            d.breakout_date,
            d.breakout_direction,
            d.breakout_price,
            d.target_price,
            d.stop_loss_price,
            d.confidence_score
        FROM pattern_detections d
        WHERE d.run_id = ?
          AND d.pattern_name IN ({",".join(["?"] * len(pattern_keys))})
          AND d.breakout_date IS NOT NULL
          AND d.breakout_price IS NOT NULL
        ORDER BY d.confidence_score DESC, d.pattern_width_bars DESC, d.pattern_id
        """
        rows = conn.execute(q, (run_id, *pattern_keys)).fetchall()
    finally:
        conn.close()

    up: List[CaseRow] = []
    down: List[CaseRow] = []
    for r in rows:
        cr = CaseRow(
            pattern_id=str(r[0]),
            symbol=str(r[1]),
            pattern_key=str(r[2]),
            formation_start=str(r[3]),
            formation_end=str(r[4]),
            breakout_date=str(r[5]) if r[5] is not None else None,
            breakout_direction=str(r[6]) if r[6] is not None else None,
            breakout_price=float(r[7]) if r[7] is not None else None,
            target_price=float(r[8]) if r[8] is not None else None,
            stop_loss_price=float(r[9]) if r[9] is not None else None,
            confidence_score=int(r[10] or 0),
        )
        if (cr.breakout_direction or "").lower() == "up":
            if len(up) < int(max_cases_per_direction):
                up.append(cr)
        elif (cr.breakout_direction or "").lower() == "down":
            if len(down) < int(max_cases_per_direction):
                down.append(cr)
        if len(up) >= int(max_cases_per_direction) and len(down) >= int(max_cases_per_direction):
            break

    return up + down


def _load_symbol_ohlcv(price_db_path: str, symbol: str) -> pd.DataFrame:
    conn = sqlite3.connect(os.path.abspath(price_db_path))
    try:
        df = pd.read_sql_query(
            "SELECT time as date, open, high, low, close, volume FROM stock_price_history WHERE symbol = ? ORDER BY time",
            conn,
            params=[symbol],
        )
    finally:
        conn.close()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    return df


def _slice_window(
    df: pd.DataFrame,
    *,
    formation_start: str,
    formation_end: str,
    breakout_date: Optional[str],
    pre_bars: int = 30,
    post_bars: int = 30,
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, Optional[pd.Timestamp]]:
    fs = pd.to_datetime(formation_start, errors="coerce")
    fe = pd.to_datetime(formation_end, errors="coerce")
    bd = pd.to_datetime(breakout_date, errors="coerce") if breakout_date else pd.NaT
    bd = bd if pd.notna(bd) else pd.NaT

    if df.empty or pd.isna(fs) or pd.isna(fe):
        return df.iloc[:0].copy(), fs, fe, (bd if pd.notna(bd) else None)

    # Find nearest indices for slicing.
    idx_start = int(df["date"].searchsorted(fs, side="left"))
    idx_end = int(df["date"].searchsorted(fe, side="right"))
    if idx_end <= idx_start:
        idx_end = min(len(df), idx_start + 1)

    w0 = max(0, idx_start - int(pre_bars))
    w1 = min(len(df), idx_end + int(post_bars))
    out = df.iloc[w0:w1].copy().reset_index(drop=True)
    return out, fs, fe, (bd.to_pydatetime() if pd.notna(bd) else None)


def _plot_candles(
    df: pd.DataFrame,
    *,
    formation_start: pd.Timestamp,
    formation_end: pd.Timestamp,
    breakout_date: Optional[pd.Timestamp],
    breakout_direction: Optional[str],
    target_price: Optional[float],
    stop_loss_price: Optional[float],
    title: str,
    out_png: str,
) -> None:
    if df.empty:
        return

    g = df.copy()
    g = g.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)
    if g.empty:
        return

    x = np.arange(len(g))
    dates = g["date"].tolist()

    fig_w = max(10.0, min(18.0, len(g) / 10.0))
    fig, ax = plt.subplots(figsize=(fig_w, 5.0), dpi=140)

    up_color = "#2ca02c"
    down_color = "#d62728"
    wick_color = "#111111"

    for i, row in g.iterrows():
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        color = up_color if c >= o else down_color
        ax.vlines(x[i], l, h, color=wick_color, linewidth=0.8, alpha=0.9)
        y0 = min(o, c)
        height = max(1e-9, abs(c - o))
        rect = Rectangle((x[i] - 0.3, y0), 0.6, height, facecolor=color, edgecolor=color, linewidth=0.6, alpha=0.85)
        ax.add_patch(rect)

    # Formation shading
    def _nearest_idx(ts: pd.Timestamp) -> Optional[int]:
        if ts is None or pd.isna(ts):
            return None
        j = int(g["date"].searchsorted(ts, side="left"))
        if j < 0:
            return 0
        if j >= len(g):
            return len(g) - 1
        return j

    i0 = _nearest_idx(pd.to_datetime(formation_start))
    i1 = _nearest_idx(pd.to_datetime(formation_end))
    if i0 is not None and i1 is not None and i1 >= i0:
        ax.axvspan(i0 - 0.5, i1 + 0.5, color="#1f77b4", alpha=0.08, label="Formation")

    # Breakout marker
    if breakout_date is not None and not pd.isna(breakout_date):
        ib = _nearest_idx(pd.to_datetime(breakout_date))
        if ib is not None:
            ax.axvline(ib, color="#9467bd", linewidth=1.2, alpha=0.9)
            lbl = f"Breakout ({(breakout_direction or '').lower() or '?'})"
            ax.text(ib + 0.2, float(g["high"].max()), lbl, fontsize=8, color="#9467bd", va="top")

    # Target / stop
    if target_price is not None and np.isfinite(float(target_price)):
        ax.axhline(float(target_price), color="#ff7f0e", linestyle="--", linewidth=1.0, alpha=0.9, label="Target")
    if stop_loss_price is not None and np.isfinite(float(stop_loss_price)):
        ax.axhline(float(stop_loss_price), color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.9, label="Stop")

    ax.set_title(title)
    ax.set_xlim(-1, len(g))
    ax.grid(True, alpha=0.15)

    # Sparse date ticks
    step = max(1, int(len(g) / 10))
    ticks = list(range(0, len(g), step))
    labels = [pd.to_datetime(dates[i]).strftime("%Y-%m-%d") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    ax.legend(loc="upper left", fontsize=8, frameon=False)

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _assemble_book_md(chapter_paths: List[str], *, title: str) -> str:
    parts: List[str] = []
    parts.append("---")
    parts.append(f'title: "{title}"')
    parts.append('lang: "vi-VN"')
    parts.append(f'date: "{datetime.now().strftime("%Y-%m-%d")}"')
    parts.append("---")
    parts.append("")
    parts.append("# Lời nói đầu")
    parts.append("")
    parts.append(
        "Tài liệu này được sinh tự động từ kết quả quét mô hình giá (OHLCV) và lớp đánh giá hậu breakout (look-ahead) "
        "trên dữ liệu Việt Nam. Mục tiêu là tái hiện cấu trúc nghiên cứu kiểu Bulkowski theo hướng có thể tái lập."
    )
    parts.append("")
    parts.append("\\newpage")
    parts.append("")

    for p in chapter_paths:
        text = open(p, "r", encoding="utf-8").read()
        parts.append(text.rstrip())
        parts.append("")
        parts.append("\\newpage")
        parts.append("")

    return "\n".join(parts).strip() + "\n"


def _try_build_pdf(*, book_md_path: str, out_pdf_path: str, mainfont: Optional[str]) -> Tuple[bool, str]:
    pandoc = "pandoc"
    try:
        subprocess.run([pandoc, "--version"], check=True, capture_output=True, text=True)
    except Exception:
        return (False, "pandoc not found; generated Markdown only.")

    # Prefer a lightweight, self-contained engine if available.
    pdf_engine = None
    for candidate in ["tectonic", "xelatex", "lualatex", "pdflatex"]:
        if shutil.which(candidate):
            pdf_engine = candidate
            break
    if not pdf_engine:
        return (False, "No PDF engine found (tectonic/xelatex/lualatex/pdflatex). Generated Markdown only.")

    book_md_abs = os.path.abspath(book_md_path)
    book_dir = os.path.dirname(book_md_abs)
    book_name = os.path.basename(book_md_abs)

    preamble_path = os.path.join(book_dir, "_pandoc_preamble.tex")
    _write_text(
        preamble_path,
        "\n".join(
            [
                "% Auto-generated by scanner/build_book_vi.py",
                "\\usepackage{etoolbox}",
                "\\sloppy",
                "\\setlength{\\emergencystretch}{3em}",
                "\\setlength{\\tabcolsep}{2pt}",
                "\\renewcommand{\\arraystretch}{0.85}",
                "\\AtBeginEnvironment{longtable}{\\small}",
                "\\AtBeginEnvironment{table}{\\small}",
                "",
            ]
        )
        + "\n",
    )

    cmd = [
        pandoc,
        book_name,
        "-o",
        os.path.abspath(out_pdf_path),
        f"--pdf-engine={pdf_engine}",
        "--resource-path",
        book_dir,
        "-H",
        preamble_path,
        "--toc",
        "--toc-depth=2",
    ]
    if mainfont:
        cmd.extend(["-V", f"mainfont={mainfont}"])

    try:
        subprocess.run(cmd, check=True, cwd=book_dir)
    except Exception as e:
        return (False, f"pandoc failed (engine={pdf_engine}): {e}")
    return (True, f"Wrote PDF: {os.path.abspath(out_pdf_path)}")


def main() -> None:
    _load_dotenv_if_present()

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-db-valid", required=True, help="Validation results DB (SQLite) from run_full_scan.py")
    parser.add_argument("--results-db-calib", default=None, help="Optional calibration results DB (SQLite)")
    parser.add_argument("--price-db", default="vietnam_stocks.db", help="Price DB used for market regime and figures")
    parser.add_argument("--index-symbol", default="VN30", help="Index symbol for bull/bear regime (default: VN30)")
    parser.add_argument("--out-dir", default=os.path.join("scan_results", "book_vi"), help="Output directory")
    parser.add_argument(
        "--patterns",
        default=None,
        help="Optional comma-separated pattern_key list to include (default: all patterns in run metadata).",
    )
    parser.add_argument(
        "--anchor",
        choices=["formation_start", "breakout_date"],
        default="formation_start",
        help="Anchor date for 18-month bull/bear regime (default: formation_start)",
    )
    parser.add_argument(
        "--overlap-policy",
        choices=["none", "bulkowski"],
        default="bulkowski",
        help="Overlap handling for stats tables (default: bulkowski)",
    )
    parser.add_argument("--min-breakout-price", type=float, default=None, help="Optional breakout price floor for stats")
    parser.add_argument(
        "--skip-ai",
        action="store_true",
        help="Do not call DeepSeek (reuse cached narrative if present; otherwise keep placeholders).",
    )
    parser.add_argument("--deepseek-base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--deepseek-model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--deepseek-api-key", default=os.getenv("DEEPSEEK_API_KEY"))
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--cases-per-direction", type=int, default=1, help="Cases per direction (up/down) per pattern")
    parser.add_argument("--pre-bars", type=int, default=30, help="Bars before formation in figure window")
    parser.add_argument("--post-bars", type=int, default=30, help="Bars after formation in figure window")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF build even if pandoc exists")
    parser.add_argument("--pdf-mainfont", default=None, help="Optional mainfont for pandoc/xelatex (e.g. 'Noto Serif')")
    args = parser.parse_args()

    out_dir = _ensure_dir(str(args.out_dir))
    reports_dir = _ensure_dir(os.path.join(out_dir, "reports"))
    context_dir = _ensure_dir(os.path.join(out_dir, "context"))
    chapters_dir = _ensure_dir(os.path.join(out_dir, "chapters"))
    cases_dir = _ensure_dir(os.path.join(out_dir, "cases"))
    figures_dir = _ensure_dir(os.path.join(out_dir, "figures"))

    # === Generate deterministic reports (JSON) ===
    valid_run_id = _run_id_from_results_db(str(args.results_db_valid))
    valid_meta = _pattern_meta_map_from_results_db(str(args.results_db_valid), valid_run_id)

    payload_valid_canon = generate_bulkowski_payload(
        results_db_path=str(args.results_db_valid),
        price_db_path=str(args.price_db),
        index_symbol=str(args.index_symbol),
        run_id=valid_run_id,
        anchor=str(args.anchor),
        group_by="canonical_key",
        overlap_policy=str(args.overlap_policy),
        min_breakout_price=float(args.min_breakout_price) if args.min_breakout_price is not None else None,
    )
    payload_valid_pat = generate_bulkowski_payload(
        results_db_path=str(args.results_db_valid),
        price_db_path=str(args.price_db),
        index_symbol=str(args.index_symbol),
        run_id=valid_run_id,
        anchor=str(args.anchor),
        group_by="pattern_key",
        overlap_policy=str(args.overlap_policy),
        min_breakout_price=float(args.min_breakout_price) if args.min_breakout_price is not None else None,
    )

    _write_json(os.path.join(reports_dir, "valid_canonical.json"), payload_valid_canon)
    _write_json(os.path.join(reports_dir, "valid_pattern_key.json"), payload_valid_pat)

    payload_calib_canon: Optional[Dict[str, Any]] = None
    payload_calib_pat: Optional[Dict[str, Any]] = None
    calib_run_id: Optional[str] = None
    if args.results_db_calib:
        calib_run_id = _run_id_from_results_db(str(args.results_db_calib))
        payload_calib_canon = generate_bulkowski_payload(
            results_db_path=str(args.results_db_calib),
            price_db_path=str(args.price_db),
            index_symbol=str(args.index_symbol),
            run_id=calib_run_id,
            anchor=str(args.anchor),
            group_by="canonical_key",
            overlap_policy=str(args.overlap_policy),
            min_breakout_price=float(args.min_breakout_price) if args.min_breakout_price is not None else None,
        )
        payload_calib_pat = generate_bulkowski_payload(
            results_db_path=str(args.results_db_calib),
            price_db_path=str(args.price_db),
            index_symbol=str(args.index_symbol),
            run_id=calib_run_id,
            anchor=str(args.anchor),
            group_by="pattern_key",
            overlap_policy=str(args.overlap_policy),
            min_breakout_price=float(args.min_breakout_price) if args.min_breakout_price is not None else None,
        )
        _write_json(os.path.join(reports_dir, "calib_canonical.json"), payload_calib_canon)
        _write_json(os.path.join(reports_dir, "calib_pattern_key.json"), payload_calib_pat)

    book_meta = {
        "built_at": _utc_now_iso(),
        "out_dir": out_dir,
        "price_db": os.path.abspath(str(args.price_db)),
        "index_symbol": str(args.index_symbol),
        "anchor": str(args.anchor),
        "overlap_policy": str(args.overlap_policy),
        "min_breakout_price": float(args.min_breakout_price) if args.min_breakout_price is not None else None,
        "valid": {"results_db": os.path.abspath(str(args.results_db_valid)), "run_id": valid_run_id},
        "calib": {"results_db": os.path.abspath(str(args.results_db_calib)), "run_id": calib_run_id} if calib_run_id else None,
        "deepseek": {
            "enabled": (not bool(args.skip_ai)) and bool(args.deepseek_api_key),
            "base_url": str(args.deepseek_base_url),
            "model": str(args.deepseek_model),
        },
    }
    _write_json(os.path.join(out_dir, "book_meta.json"), book_meta)

    # Determine chapter order from valid run metadata.
    pattern_order = _pattern_order_from_meta(valid_meta)
    if args.patterns:
        wanted = {p.strip() for p in str(args.patterns).split(",") if p.strip()}
        unknown = sorted([p for p in wanted if p not in set(pattern_order)])
        if unknown:
            raise SystemExit(f"Unknown --patterns: {unknown}")
        pattern_order = [p for p in pattern_order if p in wanted]

    valid_groups = _summary_groups_by(payload_valid_canon, group_by="canonical_key")
    valid_by_ck: Dict[str, List[Dict[str, Any]]] = {}
    for g in valid_groups:
        ck = str(g.get("pattern_name") or "")
        if ck:
            valid_by_ck.setdefault(ck, []).append(g)

    # For per-pattern-key breakdown inside chapter.
    valid_pat_groups = _summary_groups_by(payload_valid_pat, group_by="pattern_key")
    valid_by_pk: Dict[str, List[Dict[str, Any]]] = {}
    for g in valid_pat_groups:
        pk = str(g.get("pattern_name") or "")
        if pk:
            valid_by_pk.setdefault(pk, []).append(g)

    calib_by_pk: Dict[str, List[Dict[str, Any]]] = {}
    if payload_calib_pat:
        for g in _summary_groups_by(payload_calib_pat, group_by="pattern_key"):
            pk = str(g.get("pattern_name") or "")
            if pk:
                calib_by_pk.setdefault(pk, []).append(g)

    calib_by_ck: Dict[str, List[Dict[str, Any]]] = {}
    if payload_calib_canon:
        for g in _summary_groups_by(payload_calib_canon, group_by="canonical_key"):
            ck = str(g.get("pattern_name") or "")
            if ck:
                calib_by_ck.setdefault(ck, []).append(g)

    chapter_paths: List[str] = []

    for pk in pattern_order:
        meta = valid_meta.get(pk)
        if not isinstance(meta, dict):
            continue

        chap = meta.get("bulkowski_chapter")
        chap_num = int(chap) if isinstance(chap, int) else None
        display = str(meta.get("bulkowski_name") or pk)
        canonical_key = str(meta.get("canonical_key") or meta.get("spec_key") or pk)
        spec_key = str(meta.get("spec_key") or pk)
        variant = meta.get("variant")

        # Deterministic stats tables:
        #   1) Chapter-level (pattern_key)
        pk_groups = valid_by_pk.get(str(pk), [])
        table_md = _format_group_table_md(pk_groups, title="Thống kê (VALID, theo pattern_key)")

        #   2) Optional canonical aggregation (for variants)
        canon_groups = valid_by_ck.get(canonical_key, [])
        canon_md = ""
        if canon_groups:
            canon_md = _format_group_table_md(canon_groups, title="Tổng hợp (VALID, theo canonical_key)")

        # Optional: build calibration-vs-validation delta (very lightweight).
        calib_groups = calib_by_ck.get(str(canonical_key), [])
        delta_md = ""
        if calib_groups:
            def _safe_int(x: Any) -> int:
                try:
                    return int(x or 0)
                except Exception:
                    return 0

        calib_eval_total_ck = sum(_safe_int(g.get("n_evaluated")) for g in calib_groups)
        valid_eval_total_ck = sum(_safe_int(g.get("n_evaluated")) for g in canon_groups)

        calib_pk_groups = calib_by_pk.get(str(pk), [])
        calib_eval_total_pk = sum(_safe_int(g.get("n_evaluated")) for g in calib_pk_groups)
        valid_eval_total_pk = sum(_safe_int(g.get("n_evaluated")) for g in pk_groups)
        delta_md = (
            "### Calibration vs Validation (tóm tắt)\n\n"
            f"- Theo pattern_key: calib n_eval={calib_eval_total_pk}, valid n_eval={valid_eval_total_pk}\n"
            f"- Theo canonical_key: calib n_eval={calib_eval_total_ck}, valid n_eval={valid_eval_total_ck}\n\n"
            "_Gợi ý: dùng phần này để kiểm tra độ ổn định KPI sau khi hiệu chỉnh._\n"
        )

        spec_summary = _spec_summary_vi(str(spec_key))
        if variant is not None:
            spec_summary.append(f"- Variant: `{variant}`")

        context = {
            "pattern_key": str(pk),
            "pattern_display_name": display,
            "bulkowski_chapter": chap_num,
            "canonical_key": canonical_key,
            "spec_key": spec_key,
            "variant": variant,
            "spec_summary_vi": spec_summary,
            "valid": {"results_db": os.path.abspath(str(args.results_db_valid)), "run_id": valid_run_id},
            "calib": {"results_db": os.path.abspath(str(args.results_db_calib)), "run_id": calib_run_id} if calib_run_id else None,
        }
        _write_json(os.path.join(context_dir, f"{pk}.json"), context)

        # === Select cases + figures (optional) ===
        selected_cases: List[CaseRow] = []
        fig_md_lines: List[str] = []
        if not bool(args.skip_figures):
            try:
                selected_cases = _select_cases(
                    results_db_path=str(args.results_db_valid),
                    run_id=valid_run_id,
                    pattern_keys=[str(pk)],
                    max_cases_per_direction=int(args.cases_per_direction),
                )
            except Exception:
                selected_cases = []

        if selected_cases and not bool(args.skip_figures):
            case_payload = [c.__dict__ for c in selected_cases]
            _write_json(os.path.join(cases_dir, f"{pk}.json"), case_payload)
            for c in selected_cases:
                df_sym = _load_symbol_ohlcv(str(args.price_db), c.symbol)
                df_win, fs, fe, bd = _slice_window(
                    df_sym,
                    formation_start=c.formation_start,
                    formation_end=c.formation_end,
                    breakout_date=c.breakout_date,
                    pre_bars=int(args.pre_bars),
                    post_bars=int(args.post_bars),
                )

                safe_id = base64.urlsafe_b64encode(c.pattern_id.encode("utf-8")).decode("utf-8").rstrip("=")
                out_png = os.path.join(figures_dir, f"{pk}_{safe_id}.png")
                title = f"{display} | {c.symbol} | conf={c.confidence_score}"
                _plot_candles(
                    df_win,
                    formation_start=fs,
                    formation_end=fe,
                    breakout_date=pd.to_datetime(c.breakout_date, errors="coerce") if c.breakout_date else None,
                    breakout_direction=c.breakout_direction,
                    target_price=c.target_price,
                    stop_loss_price=c.stop_loss_price,
                    title=title,
                    out_png=out_png,
                )
                rel_png = os.path.relpath(out_png, out_dir)
                fig_md_lines.append(f"- `{c.symbol}`:\n\n  ![]({rel_png})")

        fig_md = ""
        if fig_md_lines:
            fig_md = "### Ví dụ minh hoạ\n\n" + "\n\n".join(fig_md_lines) + "\n"

        # === Narrative (DeepSeek) ===
        narrative_md = _placeholder_narrative_md()
        cache_path = os.path.join(chapters_dir, f"{pk}.ai.json")
        cached = None
        if os.path.exists(cache_path):
            try:
                cached = _read_json(cache_path)
            except Exception:
                cached = None
        cached_content = str(cached.get("content")) if isinstance(cached, dict) and cached.get("content") else ""

        # Offline mode: do not call DeepSeek, but reuse cache if present.
        if bool(args.skip_ai) or (not args.deepseek_api_key):
            if cached_content:
                narrative_md = _normalize_narrative_headings_md(cached_content)
        else:
            prompt_fingerprint = _sha256_text(
                json.dumps({"context": context, "table_md": table_md}, sort_keys=True, default=str)
            )
            if isinstance(cached, dict) and cached.get("fingerprint") == prompt_fingerprint and cached_content:
                ai_text = cached_content
            else:
                # Retry/backoff (very small)
                last_err = None
                ai_text = ""
                for attempt in range(1, 4):
                    try:
                        ai_text = _generate_narrative_vi(
                            context=context,
                            table_md=table_md,
                            api_key=str(args.deepseek_api_key),
                            base_url=str(args.deepseek_base_url),
                            model=str(args.deepseek_model),
                        )
                        break
                    except Exception as e:
                        last_err = e
                        time.sleep(2 * attempt)
                if not ai_text and last_err is not None:
                    ai_text = _placeholder_narrative_md(note=f"Lỗi gọi DeepSeek: {last_err}")
                _write_json(
                    cache_path,
                    {
                        "fingerprint": prompt_fingerprint,
                        "generated_at": _utc_now_iso(),
                        "model": str(args.deepseek_model),
                        "base_url": str(args.deepseek_base_url),
                        "content": ai_text,
                    },
                )

            narrative_md = _normalize_narrative_headings_md(ai_text)

        # === Chapter markdown ===
        lines: List[str] = []
        chap_prefix = f"{chap_num:02d}. " if isinstance(chap_num, int) else ""
        lines.append(f"# {chap_prefix}{display}")
        lines.append("")
        lines.append(f"- Chapter: {chap_num}")
        lines.append(f"- pattern_key: {_breakable_key(pk)}")
        lines.append(f"- canonical_key: {_breakable_key(canonical_key)}")
        lines.append(f"- spec_key: {_breakable_key(spec_key)}")
        lines.append("")
        lines.append("## Định nghĩa scan (từ digitized spec)")
        lines.append("")
        lines.extend(spec_summary)
        lines.append("")
        lines.append("## Kết quả thống kê")
        lines.append("")
        lines.append(table_md.strip())
        lines.append("")
        if canon_md:
            lines.append(canon_md.strip())
            lines.append("")
        if delta_md:
            lines.append(delta_md.strip())
            lines.append("")
        if fig_md:
            lines.append(fig_md.strip())
            lines.append("")
        lines.append("## Lời bình / Narrative")
        lines.append("")
        lines.append(narrative_md.strip())
        lines.append("")

        filename = f"chap_{chap_num:02d}_{pk}.md" if isinstance(chap_num, int) else f"chap_{pk}.md"
        chapter_path = os.path.join(chapters_dir, filename)
        _write_text(chapter_path, "\n".join(lines).strip() + "\n")
        chapter_paths.append(chapter_path)

    # Assemble final book markdown
    title = "Bách khoa mô hình giá (phiên bản dữ liệu Việt Nam)"
    book_md = _assemble_book_md(chapter_paths, title=title)
    book_md_path = os.path.join(out_dir, "book.md")
    _write_text(book_md_path, book_md)

    pdf_msg = ""
    if not bool(args.skip_pdf):
        ok, msg = _try_build_pdf(
            book_md_path=book_md_path,
            out_pdf_path=os.path.join(out_dir, "book.pdf"),
            mainfont=str(args.pdf_mainfont) if args.pdf_mainfont else None,
        )
        pdf_msg = msg

    print("=== Book Build (VI) ===")
    print(f"out_dir: {out_dir}")
    print(f"chapters: {len(chapter_paths)}")
    print(f"book_md: {os.path.abspath(book_md_path)}")
    if pdf_msg:
        print(pdf_msg)


if __name__ == "__main__":
    main()
