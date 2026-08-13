"""Render chart snapshots of sampled pattern detections for the "eye exam" (Nấc 1).

Vẽ nến thật cho các tín hiệu lấy mẫu ngẫu nhiên từ events V3, tô vùng mẫu hình
(vàng nhạt) + đường breakout (đứt nét) để GLM vision chấm: đúng mẫu hình / nhầm / không rõ.

KHÔNG sửa dữ liệu, KHÔNG ghi vào DB — chỉ đọc và vẽ ra thư mục ảnh.
Idx trong events.csv là vị trí trong frame `_load_symbol_from_db` (dropna + reset_index)
— script này tái tạo y hệt để idx khớp.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

DB_DEFAULT = Path.home() / "dev/market_stats_v2/market_cache/stock_ohlcv/latest.sqlite"
EVENTS_ROOT = Path("artifacts/scanner_v2_v3")
OUT_ROOT = Path("artifacts/eye_exam")

# màu nến + vùng
UP_COLOR = "#1a7f37"   # nến tăng
DOWN_COLOR = "#c62828"  # nến giảm
ZONE_COLOR = (1.0, 0.85, 0.3, 0.28)  # nền vùng mẫu hình
BO_COLOR = "#1565c0"   # đường breakout


def load_frame(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    """Tái tạo đúng cách rescan load dữ liệu (để idx khớp events.csv)."""
    frame = pd.read_sql_query(
        "SELECT symbol, time AS date, open, high, low, close, volume "
        "FROM stock_price_history WHERE symbol = ? ORDER BY time",
        conn, params=[symbol],
    )
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)


def render_one(family: str, row: pd.Series, frame: pd.DataFrame, out_dir: Path) -> dict | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    sym = str(row["symbol"]).upper()
    s = int(row["formation_start_idx"])
    e = int(row["formation_end_idx"])
    bo = int(row["breakout_idx"])
    # pattern dài (tam giác 20-95 nến) cần nhìn đủ hình: lùi tối đa 45 nến về trước
    width = int(row.get("pattern_width_bars") or 0)
    lookback = max(15, min(width or 15, 45))
    w0 = max(0, s - lookback)
    w1 = min(len(frame) - 1, bo + 15)
    if w1 <= w0 + 10 or bo >= len(frame):
        return None
    win = frame.iloc[w0 : w1 + 1]

    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=100)
    # nền vùng mẫu hình — tọa độ phải đổi về window index (0-based) cho khớp nến
    ax.add_patch(Rectangle((s - w0 - 0.5, 0), e - s + 1, 1, transform=ax.get_xaxis_transform(),
                           facecolor=ZONE_COLOR, edgecolor="none", zorder=0))
    # nến
    for i, (_, r) in enumerate(win.iterrows()):
        color = UP_COLOR if r["close"] >= r["open"] else DOWN_COLOR
        ax.plot([i, i], [r["low"], r["high"]], color=color, linewidth=0.8, zorder=2)
        body_lo = min(r["open"], r["close"])
        body_hi = max(r["open"], r["close"])
        ax.add_patch(Rectangle((i - 0.32, body_lo), 0.64, max(body_hi - body_lo, 1e-9),
                               facecolor=color, edgecolor=color, zorder=3))
    # đường breakout
    bo_price = float(row["breakout_price"])
    ax.hlines(bo_price, bo - w0, w1 - w0, colors=BO_COLOR, linestyles="--", linewidth=1.2, zorder=4)
    ax.annotate("breakout", xy=(bo - w0, bo_price), xytext=(bo - w0, ax.get_ylim()[1]),
                fontsize=8, color=BO_COLOR, ha="center", zorder=5)
    direction = str(row.get("breakout_direction") or "")
    ax.set_title(f"{sym} — {family} — {direction} breakout @ {win.iloc[bo - w0]['date'].date()}", fontsize=10)
    ticks = [0, (bo - w0), (w1 - w0)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(win.iloc[t]["date"].date()) for t in ticks], fontsize=8)
    ax.set_ylabel("price", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    fname = f"{sym}_{s}_{bo}.png"
    fig.savefig(out_dir / fname)
    plt.close(fig)
    return {"img": str((out_dir / fname).relative_to(OUT_ROOT)),
            "symbol": sym, "breakout_direction": direction,
            "breakout_date": str(win.iloc[bo - w0]["date"].date()),
            "formation_dates": f"{win.iloc[0]['date'].date()}..{win.iloc[s - w0]['date'].date()}",
            "pattern_quality_tier": str(row.get("pattern_quality_tier") or "")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--db", type=Path, default=DB_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    args = ap.parse_args()

    events_csv = EVENTS_ROOT / args.family / "db_active" / "events.csv"
    if not events_csv.exists():
        print(f"KHÔNG có events cho {args.family}: {events_csv}")
        return 2
    events = pd.read_csv(events_csv)
    sample = events.sample(n=min(args.n, len(events)), random_state=args.seed)
    out_dir = args.out / "pilot" / args.family
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(args.db))
    manifest = []
    for _, row in sample.iterrows():
        try:
            frame = load_frame(conn, str(row["symbol"]).upper())
            entry = render_one(args.family, row, frame, out_dir)
            if entry:
                manifest.append(entry)
        except Exception as exc:
            print(f"  ⚠️ {row.get('symbol')}: {type(exc).__name__} {str(exc)[:60]}")
    conn.close()
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ {args.family}: {len(manifest)} ảnh → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
