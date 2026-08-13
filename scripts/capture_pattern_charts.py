#!/usr/bin/env python3
"""Máy chụp chart cho nghiệm thu trực quan scanner (13/08/2026).

Vẽ nến OHLCV thật quanh mỗi event phát hiện mẫu hình, đánh dấu:
- vùng hình thành mẫu hình (vàng mờ)
- giá breakout (xanh lá, nét đứt) + ngày breakout (mũi tên đỏ)
- giá target (cam) — nếu có
- MFE / MAE (% theo breakout) — nếu có
Mục đích: để worker GLM có khả năng đọc ảnh soi xem mẫu hình "tạo hình
trên đồ thị thật" có đúng không (logic số học đúng ≠ hình vẽ đúng).

Dùng: python capture_pattern_charts.py --pattern inside_day_family/inside_day/db_active \
        --limit 20 --out artifacts/chart_capture/inside_day
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sqlite3
import sys
from pathlib import Path

import mplfinance as mpf
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Thị trường VN: xanh = tăng, đỏ = giảm
MC = mpf.make_marketcolors(up="#26a69a", down="#ef5350", edge="inherit",
                           wick="inherit", volume="in")
STYLE = mpf.make_mpf_style(marketcolors=MC, gridstyle=":", y_on_right=True)

DEFAULT_DB = "/Users/bobo/dev/market_stats_v2/market_cache/stock_ohlcv/latest.sqlite"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _num(v) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def load_events(events_csv: Path) -> list[dict]:
    with open(events_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_sample(events: list[dict], limit: int, seed: int,
                sort_by: str = "none", variant: str | None = None,
                tier: str | None = None) -> list[dict]:
    """Chọn đa dạng. sort_by: none|mfe_asc|mae_desc|random."""
    pool = events
    if variant:
        pool = [e for e in pool if (e.get("variant") or "") == variant]
    if tier:
        pool = [e for e in pool if (e.get("pattern_quality_tier") or "").lower() == tier.lower()]
    if not pool:
        return []
    if len(pool) <= limit:
        return pool
    if sort_by == "mfe_asc":
        pool = sorted(pool, key=lambda e: _num(e.get("mfe_pct")) or 0.0)[: max(limit * 3, limit)]
    elif sort_by == "mae_desc":
        pool = sorted(pool, key=lambda e: _num(e.get("mae_pct")) or 0.0, reverse=True)[: max(limit * 3, limit)]
    elif sort_by == "random":
        rng = random.Random(seed)
        return rng.sample(pool, limit)
    picked: list[dict] = []
    # 1/2 cách đều theo thứ tự file
    even = pool[:: max(1, len(pool) // (limit // 2))]
    # 1/2 random có seed (ổn định giữa các lần chạy)
    rng = random.Random(seed)
    rest = rng.sample(pool, min(limit - len(even[: limit // 2]), len(pool)))
    picked = list(even[: limit // 2]) + rest
    # đảm bảo có cả target_hit True và False nếu có thể
    hits = [e for e in picked if str(e.get("target_hit", "")).strip().lower() == "true"]
    fails = [e for e in picked if str(e.get("target_hit", "")).strip().lower() == "false"]
    if hits and not fails:
        candidate_fails = [e for e in pool if e not in picked
                           and str(e.get("target_hit", "")).strip().lower() == "false"]
        if candidate_fails:
            picked[-1] = rng.choice(candidate_fails)
    return picked[:limit]


def fetch_ohlcv(db: sqlite3.Connection, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    cur = db.execute(
        """
        SELECT time, open, high, low, close, volume
        FROM stock_price_history
        WHERE symbol = ? AND time >= ? AND time <= ?
        ORDER BY time
        """,
        (symbol, start_date, end_date),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df


def plot_event(db: sqlite3.Connection, ev: dict, out_path: Path, before_cal: int, after_cal: int) -> str | None:
    symbol = (ev.get("symbol") or "").strip()
    if not symbol:
        return "no_symbol"
    formation_start = ev.get("formation_start_date") or ""
    breakout_date = ev.get("breakout_date") or ""
    if not formation_start or not breakout_date:
        return "no_dates"

    start = pd.to_datetime(formation_start) - pd.Timedelta(days=before_cal)
    end = pd.to_datetime(breakout_date) + pd.Timedelta(days=after_cal)
    df = fetch_ohlcv(db, symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if len(df) < 10:
        return f"insufficient_data({len(df)})"

    breakout_price = _num(ev.get("breakout_price"))
    target_price = _num(ev.get("target_price"))
    mfe = _num(ev.get("mfe_pct"))
    mae = _num(ev.get("mae_pct"))

    lines = []
    if breakout_price is not None:
        lines.append(mpf.make_addplot([breakout_price] * len(df), color="#2e7d32",
                                      linestyle="--", width=1.2))
    if target_price is not None:
        lines.append(mpf.make_addplot([target_price] * len(df), color="#ef6c00",
                                      linestyle="-.", width=1.1))
    mfe_line = None
    mae_line = None
    if breakout_price is not None and mfe is not None and mfe > 0:
        mfe_line = breakout_price * (1 + mfe / 100.0)
        lines.append(mpf.make_addplot([mfe_line] * len(df), color="#26a69a",
                                      linestyle=":", width=0.9))
    if breakout_price is not None and mae is not None and mae > 0:
        mae_line = breakout_price * (1 - mae / 100.0)
        lines.append(mpf.make_addplot([mae_line] * len(df), color="#c62828",
                                      linestyle=":", width=0.9))

    title = (f"{ev.get('detection_id','?')} {symbol} {ev.get('variant','')} "
             f"tier={ev.get('pattern_quality_tier','?')} dir={ev.get('breakout_direction','?')}")
    fig, axlist = mpf.plot(df, type="candle", volume=True, style=STYLE,
                           addplot=lines, figsize=(13, 7), returnfig=True,
                           title=title, tight_layout=False)
    ax = axlist[0]

    fs = pd.to_datetime(formation_start)
    fe = pd.to_datetime(ev.get("formation_end_date") or formation_start)
    bd = pd.to_datetime(breakout_date)
    ax.axvspan(fs, fe, color="#f9a825", alpha=0.18, zorder=0)
    ax.axvline(bd, color="#e53935", alpha=0.9, linewidth=1.4, zorder=1)

    ymin, ymax = ax.get_ylim()
    ax.annotate("breakout", xy=(bd, breakout_price if breakout_price else ymax),
                xytext=(bd, ymax * 0.985), fontsize=9, color="#e53935",
                ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color="#e53935", lw=1))

    for label, price, color in (("target", target_price, "#ef6c00"),
                                ("mfe", mfe_line, "#26a69a"),
                                ("mae", mae_line, "#c62828")):
        if price is None:
            continue
        ax.text(df.index[-1], price, f" {label} {price:.2f}",
                fontsize=8, color=color, va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Chụp chart nến quanh các event mẫu hình")
    ap.add_argument("--pattern", required=True,
                    help="đường dẫn tương đối tới thư mục chứa events.csv (vd inside_day_family/inside_day/db_active)")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--before-cal", type=int, default=60, help="ngày lịch trước formation start")
    ap.add_argument("--after-cal", type=int, default=90, help="ngày lịch sau breakout")
    ap.add_argument("--sort-by", default="none", choices=["none", "mfe_asc", "mae_desc", "random"],
                    help="cách chọn mẫu nghi án")
    ap.add_argument("--variant", default=None, help="lọc theo variant")
    ap.add_argument("--tier", default=None, help="lọc theo pattern_quality_tier")
    ap.add_argument("--out", default=None, help="thư mục xuất PNG")
    args = ap.parse_args()

    artifact_dir = REPO_ROOT / "artifacts" / "scanner_v2"
    pattern_dir = artifact_dir / args.pattern
    events_csv = pattern_dir / "events.csv"
    if not events_csv.exists():
        print(f"Không tìm thấy events.csv tại {events_csv}", file=sys.stderr)
        return 2

    events = load_events(events_csv)
    print(f"events.csv: {len(events)} dòng tại {events_csv}")

    # tên gọn: bỏ hậu tố db_active (chung cho mọi pattern)
    _parts = [p for p in args.pattern.split("/") if p]
    if _parts and _parts[-1] == "db_active":
        _parts = _parts[:-1]
    default_name = _parts[-1] if _parts else "pattern"
    out_dir = Path(args.out) if args.out else REPO_ROOT / "artifacts" / "chart_capture" / default_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = pick_sample(events, args.limit, args.seed, sort_by=args.sort_by,
                         variant=args.variant, tier=args.tier)
    print(f"Chụp {len(sample)} event vào {out_dir}")

    db = sqlite3.connect(args.db)
    ok = 0
    failed = 0
    for ev in sample:
        det_id = (ev.get("detection_id") or "unknown").replace(":", "_").replace("/", "_")
        out_path = out_dir / f"{det_id}.png"
        err = plot_event(db, ev, out_path, args.before_cal, args.after_cal)
        if err is None:
            ok += 1
        else:
            failed += 1
            print(f"  [skip] {det_id}: {err}")
    db.close()
    print(f"Xong: {ok} PNG thành công, {failed} bỏ qua. Thư mục: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
