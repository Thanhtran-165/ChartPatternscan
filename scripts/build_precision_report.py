"""Thước đo SỐ (deterministic) cho độ chính xác hình học của bộ quét — Nấc 1 "khám mắt".

Với mỗi family, kiểm tra các tiêu chí hình học ĐỊNH NGHĨA mẫu hình (theo sách Bulkowski)
trên TOÀN TẬP events — không lấy mẫu, không dùng vision. Đầu ra: bảng % đạt từng tiêu chí
+ % đạt TẤT CẢ = "precision hình học" của detector.

Ngưỡng sơ bộ được đánh dấu (*) — sẽ đối chiếu lại với bảng tiêu chí lập từ các file review
(artifacts/eye_exam/geometry_criteria.md) khi subagent Flash hoàn thành.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EVENTS_ROOT = Path("artifacts/scanner_v2_v3")
OUT_ROOT = Path("artifacts/eye_exam/precision")


def _c(col: str) -> str:
    return col


# Mỗi tiêu chí: (tên, biểu thức, kind) — kind: "shape" = định nghĩa bắt buộc của mẫu hình;
# "quality" = điểm chất lượng (không bắt buộc để là mẫu hình đúng, dùng xếp tier).
CRITERIA: dict[str, list[tuple[str, str, str]]] = {
    "inside_day": [
        ("con nằm trọn trong mẹ", "(inside_day_high <= mother_bar_high) & (inside_day_low >= mother_bar_low)", "shape"),
        ("mẹ không phải doji", "mother_body_pct >= 0.1", "shape"),
    ],
    "triangles_ascending": [
        ("đỉnh gần phẳng (|upper|<=5°)", "upper_slope_deg.abs() <= 5", "shape"),
        ("đáy dốc lên (lower>=10°)", "lower_slope_deg >= 10", "shape"),
    ],
    "triangles_descending": [
        ("đáy gần phẳng (|lower|<=5°)", "lower_slope_deg.abs() <= 5", "shape"),
        ("đỉnh dốc xuống (upper<=-10°)", "upper_slope_deg <= -10", "shape"),
    ],
    "triangles_symmetrical": [
        ("đỉnh dốc xuống (upper<=-10°)", "upper_slope_deg <= -10", "shape"),
        ("đáy dốc lên (lower>=10°)", "lower_slope_deg >= 10", "shape"),
    ],
    "bull_flags": [
        ("cột cờ đủ mạnh (|pole_move_pct|>=10%)", "pole_move_pct.abs() >= 10", "shape"),
        ("hướng cột cờ khớp hướng breakout", "((breakout_direction == 'up') & (pole_price < (flag_upper_price0 + flag_lower_price0) / 2)) | ((breakout_direction == 'down') & (pole_price > (flag_upper_price0 + flag_lower_price0) / 2))", "shape"),
        ("cờ hẹp (height<=20%) (*)", "pattern_height_pct <= 20", "shape"),
        ("cờ gọn (|slope_gap|<=10°) (*)", "slope_gap_deg.abs() <= 10", "shape"),
        ("volume xác nhận", "volume_confirmed == True", "quality"),
    ],
}


def build_report(families: list[str]) -> dict:
    report = {"generated_by": "build_precision_report_v1", "families": {}}
    for fam in families:
        csv = EVENTS_ROOT / fam / "db_active" / "events.csv"
        if not csv.exists():
            report["families"][fam] = {"error": "no events"}
            continue
        df = pd.read_csv(csv)
        n = len(df)
        crits = []
        shape_pass = pd.Series(True, index=df.index)
        for name, expr, kind in CRITERIA[fam]:
            mask = df.eval(expr).fillna(False)
            crits.append({"criteria": name, "kind": kind, "pass": int(mask.sum()), "pct": round(100.0 * mask.mean(), 1)})
            if kind == "shape":
                shape_pass &= mask
        report["families"][fam] = {
            "n": n,
            "precision_shape_pct": round(100.0 * shape_pass.mean(), 1),
            "criteria": crits,
        }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append", help="chỉ chạy family này (lặp lại được)")
    args = ap.parse_args()
    families = args.family or list(CRITERIA.keys())
    report = build_report(families)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "precision_report_latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'family':22s} {'n':>7s} {'hình học':>9s}")
    for fam, r in report["families"].items():
        if "error" in r:
            print(f"{fam:22s} {'-':>7s} {r['error']}")
            continue
        print(f"{fam:22s} {r['n']:7d} {r['precision_shape_pct']:8.1f}%")
        for c in r["criteria"]:
            tag = "hình" if c["kind"] == "shape" else "chất"
            print(f"    [{tag}] {c['criteria']:44s} {c['pass']:7d} ({c['pct']:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
