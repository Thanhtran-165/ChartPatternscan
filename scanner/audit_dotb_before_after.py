# -*- coding: utf-8 -*-
"""Đợt B — bảng TRƯỚC/SAU 3 chương + báo cáo nhóm BARR dist>110 bị loại.

TRƯỚC = events.csv.bak_pre_dotb (bản trước rescan đợt B — code cũ).
SAU   = events.csv sau rescan toàn thị trường bằng code mới (target_hit_core).

3 chương: double_tops (neo neckline), bump_and_run_reversal_bottoms (old high
pivot), inside_day (full precision). Chỉ số: số events, target_hit (cột
detector, multiple 1.0), failure_5pct, median target_dist_pct.

Kèm nhóm BARR dist>110: tổng ứng viên, số bị loại (>110%), tier, tác động hit
rate (có vs không nhóm này) — cập nhật số audit đợt A2 sau rescan.

Chạy:  python3 scanner/audit_dotb_before_after.py
Xuất:  scanner/audits/dotb_before_after_3chapters.{json,md}
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHAPTERS = [
    ("double_tops", "Double Tops (neo neckline — Sol HIGH-1)"),
    ("bump_and_run_reversal_bottoms", "BARR Bottom (old high pivot — Sol HIGH-2)"),
    ("inside_day", "Inside Day (full precision — Sol MEDIUM-1)"),
]
SOURCES = {
    "double_tops": ROOT / "artifacts/scanner_v2/double_pattern_family/double_tops/db_active",
    "bump_and_run_reversal_bottoms": ROOT / "artifacts/scanner_v2/bump_and_run_family/bump_and_run_reversal_bottoms/db_active",
    "inside_day": ROOT / "artifacts/scanner_v2/inside_day_family/inside_day/db_active",
}
OUT_JSON = ROOT / "scanner/audits/dotb_before_after_3chapters.json"
OUT_MD = ROOT / "scanner/audits/dotb_before_after_3chapters.md"
BARR_GATE_PCT = 110.0


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def _evaluated(series: pd.Series) -> pd.Series:
    """dotC (Sol round-2 option a): mask dòng outcome CÓ giá trị — dòng rỗng
    (N/A, chưa có forward bar sau breakout) bị loại khỏi mẫu số mọi tỉ lệ."""
    raw = series.dropna()
    return raw[~raw.astype(str).str.strip().isin(["", "None", "nan"])]


def _stats(df: pd.DataFrame) -> dict:
    n = int(len(df))
    hit_eval = _evaluated(df.get("target_hit", pd.Series(dtype=object)))
    fail_eval = _evaluated(df.get("failure_5pct", pd.Series(dtype=object)))
    hit = _truthy(hit_eval)
    fail = _truthy(fail_eval)
    dist = pd.to_numeric(df.get("target_dist_pct"), errors="coerce")
    return {
        "n_events": n,
        "n_evaluated_events": int(len(hit_eval)),
        "n_na_no_forward_bars": int(n - len(hit_eval)),
        "target_hit_rate_pct": round(float(hit.mean() * 100.0), 2) if len(hit_eval) else None,
        "failure_5pct_rate_pct": round(float(fail.mean() * 100.0), 2) if len(fail_eval) else None,
        "median_target_dist_pct": round(float(dist.median()), 2) if dist.notna().any() else None,
    }


def _scoped(df: pd.DataFrame) -> pd.DataFrame:
    if "publication_quality_tier" in df.columns:
        sub = df[df["publication_quality_tier"].astype(str).str.lower().isin(["premium", "standard"])]
        if len(sub) >= 30:
            return sub
    return df


def _before_path(d: Path) -> tuple[Path, str]:
    """Chọn file TRƯỚC: ưu tiên .bak_pre_dotb (backup đợt B); double không thuộc
    EVENT_SOURCES nên backup đợt B bỏ sót → fallback .bak_vintage_pre_rerun
    (14/08, 782 events — CÙNG tập event_id với bản mới, khác mỗi cột đánh giá:
    cô lập đúng tác động đo của đợt B) trước, rồi mới .bak_pre_edition2 (802
    events, đời cũ hơn)."""
    preferred = d / "events.csv.bak_pre_dotb"
    if preferred.exists():
        return preferred, "events.csv.bak_pre_dotb (backup ngay trước rescan đợt B)"
    for fallback_name in ("events.csv.bak_vintage_pre_rerun", "events.csv.bak_pre_edition2"):
        fallback = d / fallback_name
        if fallback.exists():
            extra = " — cùng tập event_id với bản SAU, khác mỗi cột đánh giá" if "vintage" in fallback_name else ""
            return fallback, f"{fallback_name} (đời code cũ, double không thuộc EVENT_SOURCES nên backup đợt B không phủ{extra})"
    raise FileNotFoundError(f"Không tìm thấy file TRƯỚC cho {d}")


def _chapter_block(pattern_id: str) -> dict:
    d = SOURCES[pattern_id]
    before_path, before_note = _before_path(d)
    before = pd.read_csv(before_path, low_memory=False)
    after = pd.read_csv(d / "events.csv", low_memory=False)
    return {
        "pattern_id": pattern_id,
        "before_source": before_note,
        "raw": {"before": _stats(before), "after": _stats(after)},
        "scoped_premium_standard": {"before": _stats(_scoped(before)), "after": _stats(_scoped(after))},
    }


def _barr_block() -> dict:
    d = SOURCES["bump_and_run_reversal_bottoms"]
    df = pd.read_csv(d / "events.csv", low_memory=False)
    dist = pd.to_numeric(df.get("target_dist_pct"), errors="coerce")
    gate_mask = dist > BARR_GATE_PCT
    excluded = df[gate_mask.fillna(False)]
    kept = df[~gate_mask.fillna(False)]
    tier_dist = (
        excluded["publication_quality_tier"].value_counts(dropna=False).to_dict()
        if "publication_quality_tier" in excluded.columns
        else {}
    )
    # dotC (option a): tỉ lệ hit tính trên nhóm outcome CÓ giá trị (loại N/A).
    hit_kept = _truthy(_evaluated(kept.get("target_hit", pd.Series(dtype=object))))
    hit_all = _truthy(_evaluated(df.get("target_hit", pd.Series(dtype=object))))
    return {
        "barr_bottoms_total_events": int(len(df)),
        "gate_dist_pct": BARR_GATE_PCT,
        "excluded_over_gate": int(len(excluded)),
        "excluded_share_pct": round(len(excluded) / max(len(df), 1) * 100.0, 2),
        "excluded_tier_distribution": {str(k): int(v) for k, v in tier_dist.items()},
        "excluded_dist_pct_min": round(float(dist[gate_mask.fillna(False)].min()), 2) if len(excluded) else None,
        "excluded_dist_pct_max": round(float(dist[gate_mask.fillna(False)].max()), 2) if len(excluded) else None,
        "hit_rate_all_pct": round(float(hit_all.mean() * 100.0), 2),
        "hit_rate_kept_pct": round(float(hit_kept.mean() * 100.0), 2),
        "hit_rate_impact_pp": round(float(hit_kept.mean() * 100.0 - hit_all.mean() * 100.0), 2),
        "note": (
            "Nhóm dist>110 bị cổng publication loại theo quyết định đợt A2 (target quá xa so khoảng cách đo sách). "
            "Ngoài ra events không vào được chuỗi neo pivot lead-in đã bị detector loại TRƯỚC khi ghi events.csv — "
            "xem scanner/audits/barr_old_high_audit.json."
        ),
    }


def _md(chapters: list[dict], barr: dict) -> str:
    lines = [
        "# Đợt B — Bảng TRƯỚC/SAU 3 chương (rescan toàn thị trường 16/08/2026)",
        "",
        "TRƯỚC = `events.csv.bak_pre_dotb` (code cũ). SAU = rescan bằng `target_hit_core` full precision.",
        "",
        "Nguồn TRƯỚC từng chương:",
    ]
    for ch in chapters:
        lines.append(f"- {ch['pattern_id']}: {ch['before_source']}")
    lines += [
        "",
        "| Chương | Phạm vi | N TRƯỚC | N SAU | Hit TRƯỚC | Hit SAU | Δ pp | Fail5 TRƯỚC | Fail5 SAU | Median dist TRƯỚC | SAU |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ch in chapters:
        for scope_label, key in (("toàn bộ", "raw"), ("premium+standard", "scoped_premium_standard")):
            b, a = ch[key]["before"], ch[key]["after"]
            delta = round((a["target_hit_rate_pct"] or 0) - (b["target_hit_rate_pct"] or 0), 2)
            lines.append(
                f"| {ch['pattern_id']} | {scope_label} | {b['n_events']} | {a['n_events']} | "
                f"{b['target_hit_rate_pct']}% | {a['target_hit_rate_pct']}% | {delta:+.2f} | "
                f"{b['failure_5pct_rate_pct']}% | {a['failure_5pct_rate_pct']}% | "
                f"{b['median_target_dist_pct']}% | {a['median_target_dist_pct']}% |"
            )
    lines += [
        "",
        "## Nhóm BARR Bottom dist > 110% bị cổng loại (sau rescan)",
        "",
        f"- Tổng events BARR bottoms: **{barr['barr_bottoms_total_events']}**",
        f"- Bị loại dist > {barr['gate_dist_pct']:.0f}%: **{barr['excluded_over_gate']}** ({barr['excluded_share_pct']}%)",
        f"- Phân bố tier nhóm bị loại: {barr['excluded_tier_distribution']}",
        f"- Khoảng dist nhóm bị loại: {barr['excluded_dist_pct_min']}% – {barr['excluded_dist_pct_max']}%",
        f"- Hit rate toàn bộ: {barr['hit_rate_all_pct']}% · sau loại: {barr['hit_rate_kept_pct']}% (tác động {barr['hit_rate_impact_pp']:+.2f} pp)",
        f"- Ghi chú: {barr['note']}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    chapters = [_chapter_block(pid) for pid, _ in CHAPTERS]
    barr = _barr_block()
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Bằng chứng đợt B: trước/sau rescan 3 chương Sol đích danh + nhóm BARR dist>110",
        "chapters": [
            {"pattern_id": ch["pattern_id"], "label": label, **ch}
            for ch, (_, label) in zip(chapters, CHAPTERS)
        ],
        "barr_dist110_excluded": barr,
    }
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_md(chapters, barr), encoding="utf-8")
    print(f"[dotb] {OUT_JSON}")
    print(f"[dotb] {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
