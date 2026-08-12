#!/usr/bin/env python3
"""M2: nối failure_logic (failure_busted close-based + weak_move_5pct + days_to_bust)
vào 12 file định nghĩa _evaluate trong scanner/v2 + thêm cột mới vào COLUMNS.

Chỉ sửa 13 hàm gán (12 file) — consumer lan tự động qua import.
Chạy: python scripts/patch_m2_failure_logic.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "scanner" / "v2"

# --- Chuẩn bị: 3 dòng thay thế dùng chung ---
EMPTY_3 = (
    '            "weak_move_5pct": None,\n'
    '            "failure_busted": None,\n'
    '            "days_to_bust": None,\n'
)

def busted_lines(det_arg: str, fut_var: str) -> str:
    return (
        f'        "weak_move_5pct": bool(float(mfe) < 5.0),\n'
        f'        "failure_busted": failure_busted_flag({det_arg}, {fut_var}, breakout_price=breakout_price, target_price=target, mfe_pct=float(mfe)),\n'
        f'        "days_to_bust": failure_busted_days({det_arg}, {fut_var}, breakout_price=breakout_price, target_price=target, mfe_pct=float(mfe)),\n'
    )

# --- Danh sách file chuẩn: (tên file, detection_arg, future_var, số lần patch) ---
STD_FILES = [
    ("ascending_triangles.py", "detection", "future", 2),
    ("descending_triangles.py", "detection", "future", 2),
    ("symmetrical_triangles.py", "detection", "future", 2),
    ("flags_experiment.py", "detection", "future", 2),
    ("measured_moves.py", "detection", "future", 2),
    ("head_shoulders.py", "detection", "future", 2),
    ("double_patterns.py", "detection", "future", 2),
    ("rectangles.py", "detection", "future", 2),
    ("scallops.py", "detection", "future", 2),
    ("pipes.py", "detection", "future", 2),
    ("gaps.py", "row", "future", 2),  # _evaluate_gap dùng row
]

# --- Các file consumer chỉ cần thêm cột COLUMNS ---
COLUMN_ONLY = [
    "broadening_patterns.py", "cup_with_handle.py", "diamonds.py", "dead_cat_bounce.py",
    "horns.py", "inside_days.py", "rounding.py", "three_methods.py",
    "falling_wedges.py", "rising_wedges.py", "three_peaks_valleys.py", "triple_patterns.py",
]

COL_ADD = '"failure_5pct", "weak_move_5pct", "failure_busted", "days_to_bust",'

report = []

def patch_file(fname: str, replacements: list[tuple[str, str, int]]) -> None:
    p = V2 / fname
    src = p.read_text(encoding="utf-8")
    for old, new, expected in replacements:
        n = src.count(old)
        if n != expected:
            raise SystemExit(f"[{fname}] pattern sai: {old[:60]!r} → count {n}, kỳ vọng {expected}")
        src = src.replace(old, new)
    p.write_text(src, encoding="utf-8")
    report.append(f"✅ {fname}: {sum(r[2] for r in replacements)} chỗ")

# 1) Nhóm chuẩn: import + empty + chính
for fname, det_arg, fut, _ in STD_FILES:
    p = V2 / fname
    src = p.read_text(encoding="utf-8")
    import_old = None
    for line in src.splitlines():
        if "measurement_registry import" in line and line.strip().startswith("from "):
            import_old = line
            break
    if import_old is None:
        raise SystemExit(f"[{fname}] không tìm thấy import measurement_registry")
    import_new = import_old + "\nfrom scanner.v2.failure_logic import failure_busted_days, failure_busted_flag"
    if fname == "flags_experiment.py":
        import_new = import_old + "\nfrom .failure_logic import failure_busted_days, failure_busted_flag"
    n = src.count(import_old)
    if n != 1:
        raise SystemExit(f"[{fname}] import lặp {n} lần")
    src = src.replace(import_old, import_new)
    src = src.replace('            "failure_5pct": None,\n', '            "failure_5pct": None,\n' + EMPTY_3, 1)
    src = src.replace(
        '        "failure_5pct": bool(float(mfe) < 5.0),\n',
        '        "failure_5pct": bool(float(mfe) < 5.0),\n' + busted_lines(det_arg, fut),
        1,
    )
    p.write_text(src, encoding="utf-8")
    report.append(f"✅ {fname}: import + empty + chính (det={det_arg})")

# 2) bump_and_run: empty 1-dòng + chính (biến det/forward, mfe không float())
p = V2 / "bump_and_run.py"
src = p.read_text(encoding="utf-8")
old_empty = '        return {"mfe_pct": None, "mae_pct": None, "target_hit": False, "failure_5pct": True, "evaluated_bars": 0}'
new_empty = (
    '        return {\n'
    '            "mfe_pct": None,\n'
    '            "mae_pct": None,\n'
    '            "target_hit": False,\n'
    '            "failure_5pct": True,\n'
    '            "weak_move_5pct": None,\n'
    '            "failure_busted": None,\n'
    '            "days_to_bust": None,\n'
    '            "evaluated_bars": 0,\n'
    '        }'
)
assert src.count(old_empty) == 1, "bump empty"
src = src.replace(old_empty, new_empty)
old_main = '        "failure_5pct": bool(mfe < 5.0),\n'
new_main = (
    '        "failure_5pct": bool(mfe < 5.0),\n'
    '        "weak_move_5pct": bool(mfe < 5.0),\n'
    '        "failure_busted": failure_busted_flag(det, forward, breakout_price=breakout_price, target_price=target, mfe_pct=float(mfe)),\n'
    '        "days_to_bust": failure_busted_days(det, forward, breakout_price=breakout_price, target_price=target, mfe_pct=float(mfe)),\n'
)
assert src.count(old_main) == 1, "bump main"
src = src.replace(old_main, new_main)
# import
old_imp = "from scanner.v2.measurement_registry import lookahead_bars as _registry_lookahead"
assert src.count(old_imp) == 1, "bump import"
src = src.replace(old_imp, old_imp + "\nfrom scanner.v2.failure_logic import failure_busted_days, failure_busted_flag")
p.write_text(src, encoding="utf-8")
report.append("✅ bump_and_run.py: empty 1-dòng → multi-line + chính + import")

# 3) islands: empty + mfe_series + pattern_key
p = V2 / "islands.py"
src = p.read_text(encoding="utf-8")
old_sig = "def _evaluate_island(df: pd.DataFrame, row: Mapping[str, Any], *, lookahead: int) -> dict[str, Any]:\n    idx = int(row[\"breakout_idx\"])"
new_sig = (
    "def _evaluate_island(df: pd.DataFrame, row: Mapping[str, Any], *, lookahead: int) -> dict[str, Any]:\n"
    "    row = dict(row)  # bản sao — không sửa row gốc\n"
    '    row.setdefault("pattern_key", "island_reversals")\n'
    '    idx = int(row["breakout_idx"])'
)
assert src.count(old_sig) == 1, "islands sig"
src = src.replace(old_sig, new_sig)
src = src.replace('            "failure_5pct": None,\n', '            "failure_5pct": None,\n' + EMPTY_3, 1)
old_main = '        "failure_5pct": bool(float(mfe_series.max()) < 5.0) if not mfe_series.dropna().empty else None,\n'
new_main = (
    '        "failure_5pct": bool(float(mfe_series.max()) < 5.0) if not mfe_series.dropna().empty else None,\n'
    '        "weak_move_5pct": bool(float(mfe_series.max()) < 5.0) if not mfe_series.dropna().empty else None,\n'
    '        "failure_busted": failure_busted_flag(row, future, breakout_price=breakout_price, target_price=target, mfe_pct=float(mfe_series.max()) if not mfe_series.dropna().empty else None),\n'
    '        "days_to_bust": failure_busted_days(row, future, breakout_price=breakout_price, target_price=target, mfe_pct=float(mfe_series.max()) if not mfe_series.dropna().empty else None),\n'
)
assert src.count(old_main) == 1, "islands main"
src = src.replace(old_main, new_main)
old_imp = "from scanner.v2.measurement_registry import lookahead_bars as _registry_lookahead"
assert src.count(old_imp) == 1, "islands import"
src = src.replace(old_imp, old_imp + "\nfrom scanner.v2.failure_logic import failure_busted_days, failure_busted_flag")
p.write_text(src, encoding="utf-8")
report.append("✅ islands.py: pattern_key + empty + mfe_series + import")

# 4) Cột COLUMNS cho mọi file có "failure_5pct", (12 file định nghĩa + consumer)
column_files = [f for f, *_ in STD_FILES] + COLUMN_ONLY + ["bump_and_run.py", "islands.py"]
for fname in column_files:
    p = V2 / fname
    src = p.read_text(encoding="utf-8")
    n = src.count('"failure_5pct",')
    if n == 0:
        report.append(f"⚠️ {fname}: không có cột failure_5pct (bỏ qua)")
        continue
    src = src.replace('"failure_5pct",', COL_ADD)
    p.write_text(src, encoding="utf-8")
    report.append(f"✅ {fname}: cột COLUMNS thêm 3 ({n} dòng)")

# 5) Registry: gaps threshold 2.0 (fill gần breakout — K3-1 "breakaway fill = failure")
p = V2 / "measurement_registry.py"
src = p.read_text(encoding="utf-8")
old = '    "spike_formation": 3.0,\n'
assert src.count(old) == 1, "registry spike"
src = src.replace(old, old + '    "gaps": 2.0,  # fill gap = close quay lại sát breakout (K3-1: breakaway fill = failure)\n')
p.write_text(src, encoding="utf-8")
report.append("✅ measurement_registry.py: gaps threshold 2.0")

# 6) failure_logic: gaps/islands failure reference → gap_edge (khớp registry _FAILURE_REFERENCE)
p = V2 / "failure_logic.py"
src = p.read_text(encoding="utf-8")
old = '    "gaps": ("breakout_price",),\n    "islands": ("breakout_price",),\n'
assert src.count(old) == 1, "failure_logic gaps"
src = src.replace(old, '    "gaps": ("gap_edge", "breakout_price"),\n    "islands": ("gap_edge", "breakout_price"),\n')
p.write_text(src, encoding="utf-8")
report.append("✅ failure_logic.py: gaps/islands → gap_edge")

print("\n".join(report))
print(f"\nHOÀN TẤT: {len(report)} bước")
