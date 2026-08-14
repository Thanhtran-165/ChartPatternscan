"""measurement_registry.py — NGUỒN CHUẨN ĐO LƯỜNG DUY NHẤT (V3, mốc M1).

Mọi thành phần (detector scan, build profile, dashboard, mail tín hiệu) ĐỌC
chuẩn đo lường từ đây — không hardcode ở nơi khác. Đổi chuẩn = sửa file này
(cùng spec JSON), KHÔNG sửa detector.

Thứ tự ưu tiên nguồn số liệu:
  1. pdf_review — số đọc trực tiếp từ sách Bulkowski (PDF_REVIEW_20260812.md)
  2. digitized  — spec JSON đã trích (extraction_phase_1/digitization/...)
  3. detector_legacy — chưa có spec, giữ số detector hiện tại + cờ chờ M5

Mỗi mục ghi rõ `source` + `note` để biết số từ đâu mà không cần mở file khác.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIGITIZED_DIRS = [
    _REPO_ROOT / "extraction_phase_1" / "digitization" / "patterns_digitized",
    _REPO_ROOT / "extraction_phase_1" / "digitization" / "patterns_digitized_pdfreview",
]

# ---------------------------------------------------------------------------
# 1. SỐ PDF (đọc trực tiếp từ sách — nguồn cao nhất, xem PDF_REVIEW_20260812.md)
#    lookahead = "Days to ultimate high/low" (bull/bear market theo sách).
#    Detector dùng 1 số → lấy giá trị BULL MARKET (dòng đầu bảng sách), ghi rõ.
# ---------------------------------------------------------------------------
_PDF_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "pipe_bottoms": {"lookahead_bull": 194, "lookahead_bear": 133, "sample": 1152},
    "pipe_tops": {"lookahead_bull": 75, "lookahead_bear": 54, "sample": 830},
    "cup_with_handle": {"lookahead_bull": 167, "lookahead_bear": 63, "sample": 471},
    "head_and_shoulders_bottom": {"lookahead_bull": 176, "lookahead_bear": 107, "sample": 672},
    "head_and_shoulders_top": {"lookahead_bull": 62, "lookahead_bear": 41, "sample": 814},
    "scallops_ascending": {"lookahead_bull": 162, "lookahead_bear": 68, "note": "4 variants 162/68/44/35 — dùng UA (bull)"},
    "scallops_descending": {"lookahead_bull": 106, "lookahead_bear": 70, "note": "4 variants 106/70/47/30 — dùng UA (bull)"},
    "high_tight_flags": {"lookahead_bull": 39, "lookahead_bear": 25, "sample": 307},
    "triple_bottoms": {"lookahead_bull": 165, "lookahead_bear": 80, "sample": 602},
    "triple_tops": {"lookahead_bull": 60, "lookahead_bear": 42, "sample": 627},
    "three_falling_peaks": {"lookahead_bull": 36, "lookahead_bear": 34, "sample": 527},
    "three_rising_valleys": {"lookahead_bull": 125, "lookahead_bear": 94, "sample": 496},
    # --- M5 13/08 (docs/project/pdf_review/m5/): lookahead digitized SAI 2–10 lần → số PDF.
    # Quy ước: bull = Bull/Up (dòng đầu bảng sách), bear = Bear/Down (dòng cuối).
    "double_bottoms": {"lookahead_bull": 136, "lookahead_bear": 77, "sample": 1383,
        "note": "M5 PDF ch.13–16: bull/UA 136–170 · bear/UD 77–101 (digitized 76 SAI)"},
    "double_tops": {"lookahead_bull": 43, "lookahead_bear": 32, "sample": 1413,
        "note": "M5 PDF ch.17–20: bull/UD 43–51 · bear/UD 32–45 (digitized 71 SAI)"},
    "bump_and_run_reversal": {"lookahead_bull": 68, "lookahead_bear": 39, "sample": 1309,
        "note": "M5 PDF ch.8 Tops: ultimate low bull 68 / bear 39 (digitized 252 SAI 4-6 lần)"},
    "wedges_ascending_descending": {"lookahead_bull": 116, "lookahead_bear": 32, "sample": 1163,
        "note": "M5 PDF ch.52: Falling bull/UA 116 · bear/UD 32 (rising 127/60/38/38 — digitized 126 gần khớp)"},
    "islands": {"lookahead_bull": 128, "lookahead_bear": 34, "sample": 1837,
        "note": "M5 PDF ch.30 Reversals: bull/Up 128 · bear/Down 34 (Long 67/33/27/26 — digitized 14 SAI)"},
    "rounding_bottoms_tops": {"lookahead_bull": 189, "lookahead_bear": 105, "sample": 1229,
        "note": "M5 PDF ch.39 RdB: ultimate high bull 189 / bear 105 (RdT 161/77/45/25 — digitized 84/63 SAI)"},
    "measured_move_down_up": {"lookahead_bull": 153, "lookahead_bear": 113, "sample": 1721,
        "note": "M5 PDF ch.32: KHÔNG có ultimate — dùng pattern length MMD 153/113 (digitized 63 SAI)"},
    "rising_falling_three_methods": {"lookahead_bull": 7, "lookahead_bear": 13, "sample": 166,
        "note": "M5 PDF EC ch.73: KHÔNG có ultimate — dùng candle-end→trend-end median 7/11 (digitized 20 SAI)"},
    "broadening_bottoms": {"lookahead_bull": 112, "lookahead_bear": 65, "sample": 237,
        "note": "M5 PDF ch.1: ultimate high bull 112 / bear 65 (digitized 84 lệch)"},
    "broadening_tops": {"lookahead_bull": 50, "lookahead_bear": 29, "sample": 493,
        "note": "M5 PDF ch.4: ultimate low bull 50 / bear 29 (digitized 56 lệch)"},
    "broadening_wedges": {"lookahead_bull": 131, "lookahead_bear": 23, "sample": 719,
        "note": "M5 PDF ch.6 BWD: bull/Up 131 · bear/Down 23 (BWA 161/78/63/51)"},
    "diamond_bottom": {"lookahead_bull": 119, "lookahead_bear": 72, "sample": 295,
        "note": "M5 PDF ch.11: ultimate high bull 119 / bear 72 (digitized 77 gần bear, sai bull)"},
    "diamond_top": {"lookahead_bull": 52, "lookahead_bear": 43, "sample": 375,
        "note": "M5 PDF ch.12: ultimate low bull 52 / bear 43 (digitized 63 lệch nhẹ)"},
    # Harami (EC candlestick book ch.43-46, family_harami_20260813.md): KHÔNG có
    # "days to ultimate" — sách đo "candle end → trend end" median 6-9 ngày.
    # Dùng 10 (bao phủ trend end + buffer) như đề xuất file family_harami.
    "harami": {"lookahead_bull": 10, "lookahead_bear": 10, "sample": 20000,
        "note": "M5 PDF EC ch.43-46: KHÔNG ultimate — candle-end→trend-end median 6-9d → dùng 10 (digitized inside_day 10 gần khớp nhưng khác metric)"},
    # inside_day: PDF lệch ĐỊNH NGHĨA (body Harami vs range) → KHÔNG dùng số PDF.
    # dead_cat: event-driven, không có "days to ultimate" kiểu chart pattern.
    # horn/rectangle tách theo pattern_key con (bảng _VARIANT_LOOKAHEAD).
}

# Lookahead theo pattern_key CON (những family gộp nhiều hướng có số khác nhau).
# Ưu tiên CAO HƠN _PDF_OVERRIDES family — key chính xác nhất thắng.
_VARIANT_LOOKAHEAD: Dict[str, Dict[str, Any]] = {
    # PDF_REVIEW_20260812: horn bottoms 180/90, tops 67/64
    "horn_bottoms": {"lookahead_bull": 180, "lookahead_bear": 90, "source": "pdf", "note": "PDF horn_bottoms 180/90"},
    "horn_tops": {"lookahead_bull": 67, "lookahead_bear": 64, "source": "pdf", "note": "PDF horn_tops 67/64"},
    # PDF_REVIEW_20260812: rect bottoms 177/81/41/33, tops 170/75/56/40
    "rectangle_bottoms": {"lookahead_bull": 177, "lookahead_bear": 81, "source": "pdf", "note": "PDF rect_bottoms 177/81"},
    "rectangle_tops": {"lookahead_bull": 170, "lookahead_bear": 75, "source": "pdf", "note": "PDF rect_tops 170/75"},
    # M5 PDF 13/08 (family_rounding_20260813.md): digitized average_days_bottom/top 84/63 SAI
    # (thực ra là "time to target" bịa). PDF: RdB ultimate high 189/105 · RdT-UA 161/77 · RdT-UD 45/25.
    "rounding_bottoms": {"lookahead_bull": 189, "lookahead_bear": 105, "source": "pdf", "note": "M5 PDF ch.39 RdB: 189/105 (digitized 84 SAI)"},
    "rounding_tops": {"lookahead_bull": 161, "lookahead_bear": 25, "source": "pdf", "note": "M5 PDF ch.40 RdT: UA 161/77 · UD 45/25 (digitized 63 SAI)"},
    # M5 PDF 14/08 (bar_bottoms_measure_rule_deepdive_20260814.md): family-level
    # "bump_and_run_reversal" 68/39 là số TOPS (ch.8) — SAI cho Bottoms (ch.7).
    "bump_and_run_reversal_bottoms": {"lookahead_bull": 186, "lookahead_bear": 109, "source": "pdf",
        "note": "M5 PDF ch.7 Bottoms: ultimate high bull 186 / bear 109 (trước dùng nhầm 68/39 của Tops)"},
    "bump_and_run_reversal_tops": {"lookahead_bull": 68, "lookahead_bear": 39, "source": "pdf",
        "note": "M5 PDF ch.8 Tops: ultimate low bull 68 / bear 39 (tách khỏi family-level cho tường minh)"},
    # M5 PDF 13/08 (family_gaps_20260813.md): gaps KHÔNG có "days to ultimate" —
    # sách đo "Average time to close the gap" (thời gian gap bị lấp). Số digitized
    # (42/21/5/63) là BỊA. Dùng time-to-close làm lookahead đo lường (nguồn pdf).
    # Thứ tự cột sách: bull/Up · bear/Up · bull/Down · bear/Down → bull = Bull/Up, bear = Bear/Down.
    "breakaway_gaps": {"lookahead_bull": 136, "lookahead_bear": 111, "source": "pdf",
        "note": "M5 PDF ch23: avg time to close 136/61/168/111 — KHÔNG có days-to-ultimate (BỊA digitized)"},
    "continuation_gaps": {"lookahead_bull": 98, "lookahead_bear": 91, "source": "pdf",
        "note": "M5 PDF ch23: avg time to close 98/43/77/91 — KHÔNG có days-to-ultimate (BỊA digitized)"},
    "exhaustion_gaps": {"lookahead_bull": 9, "lookahead_bear": 10, "source": "pdf",
        "note": "M5 PDF ch23: avg time to close 9/7/14/10 — KHÔNG có days-to-ultimate (BỊA digitized)"},
    "area_gaps": {"lookahead_bull": 3, "lookahead_bear": 3, "source": "pdf",
        "note": "M5 PDF ch23: avg time to close 3d (cả 4 tổ hợp) — KHÔNG có days-to-ultimate (BỊA digitized)"},
    # M5 PDF 13/08 (docs/project/pdf_review/m5/family_triangles_20260813.md):
    # days to ultimate theo 4 tổ hợp Bull/UA · Bear/UA · Bull/UD · Bear/UD.
    # Quy ước: lookahead_bull = Bull/UA (dòng đầu bảng sách), bear = Bear/UD (dòng cuối).
    "triangles_ascending": {"lookahead_bull": 185, "lookahead_bear": 39, "source": "pdf",
        "note": "M5 PDF ch47: Bull/UA 185 · Bear/UA 97 · Bull/UD 64 · Bear/UD 39 — pending_rescan (events hiện đo 126)"},
    "triangles_descending": {"lookahead_bull": 178, "lookahead_bear": 32, "source": "pdf",
        "note": "M5 PDF ch48: Bull/UA 178 · Bear/UA 86 · Bull/UD 50 · Bear/UD 32 — pending_rescan (events hiện đo 126)"},
    "triangles_symmetrical": {"lookahead_bull": 124, "lookahead_bear": 30, "source": "pdf",
        "note": "M5 PDF ch49: Bull/UA 124 · Bear/UA 77 · Bull/UD 45 · Bear/UD 30 — pending_rescan (events hiện đo 126)"},
}

# Số liệu PDF mới (M5 — đọc sách 13/08, docs/project/pdf_review/m5/) KHÔNG phải lookahead
# ultimate (flags/pennants sách đo bằng trend-end method) — field phụ: sample,
# break-even failure, target rule. Merge vào measurement output qua key "pdf_extras".
_PDF_EXTRAS: Dict[str, Dict[str, Any]] = {
    "flags": {
        "sample": 523,
        "be_failure_pct": 4,
        "days_to_trend_high": 17,
        "target_rule": "formation_end ± (trend_start − formation_start_low/high)",
        "source_file": "docs/project/pdf_review/m5/family_flags_pennants_20260813.md",
        "note": (
            "Sách đo bằng TREND-END method (17d tới minor high/low) — KHÔNG phải days-to-ultimate "
            "nên KHÔNG nạp lookahead. Digitized measure rule (breakout + flagpole height) SAI — "
            "đã thay bằng công thức PDF. BE failure 4/3/2/0% (bull 4%). Sample digitized thiếu."
        ),
    },
    "pennants": {
        "sample": 462,
        "be_failure_pct": 2,
        "days_to_trend_high": 22,
        "meet_target_pct": 63,
        "target_rule": "formation_end ± (trend_start − formation_start_low/high)",
        "source_file": "docs/project/pdf_review/m5/family_flags_pennants_20260813.md",
        "note": (
            "Trend-end method — giữ lookahead digitized 63. Digitized measure rule SAI (flagpole height) — "
            "đã thay bằng công thức PDF. BE failure 2/2/4/0% (bull 2%); % meeting target 50–63%."
        ),
    },
    "high_tight_flags": {
        "sample": 307,
        "be_failure_pct": "0% / 0% (cả 2 thị trường — xuất sắc)",
        "avg_move": "rise bull 69% / bear 42%",
        "meet_target_pct": "90% / 91%",
        "days": "ultimate high 39 / 25 (đã nạp lookahead _PDF_OVERRIDES)",
        "target_rule": (
            "ECP ch22 Table 22.8: 'Measure the rise leading to the flag and project half of it upward, "
            "using the flag low price' — detector ĐÃ dùng half-prior-move (cập nhật 14/08 theo GLM-5.3 review L2)"
        ),
        "source_file": "docs/project/pdf_review/PDF_REVIEW_20260812.md (ch.22 p373-384) + family_flags_pennants_20260813.md",
        "note": (
            "M5 13/08: digitized flags_digitized.json ghi HTF rise 47% / failure 5% — SAI (PDF 69%/42%, BE 0%). "
            "HTF cần spec riêng; số đã tách khỏi flags chung."
        ),
    },
    "triangles": {
        "sample": 3605,
        "source_file": "docs/project/pdf_review/m5/family_triangles_20260813.md",
        "note": (
            "M5 PDF ch.47–49: sample asc 1.092 / desc 1.166 / sym 1.347. Lookahead nạp per-pattern "
            "trong _VARIANT_LOOKAHEAD. Measure rule digitized KHỚP PDF (breakout ± height); "
            "symmetrical thêm biến thể halfway point."
        ),
    },
    "double_bottoms": {
        "sample": 1383,
        "be_failure_pct": "bull 4–5% / bear 4–8% (4 biến thể)",
        "avg_move": "rise bull 35–40% / bear 23–33%",
        "meet_target_pct": "47–67% (AA 66 bull / 48 bear)",
        "days": "ultimate high bull 136–170 / bear 77–101",
        "throwback_pct": "46–64%",
        "best_variant": "EE (rise bull 40%, BE 4%)",
        "target_rule": "highest_high_between_bottoms + (highest_high_between_bottoms − lowest_low_lower_bottom)",
        "source_file": "docs/project/pdf_review/m5/family_doubles_20260813.md",
        "note": (
            "M5 PDF ch.13–16: measure rule KHỚP digitized (KHÔNG chia đôi — chỉ DT mới chia). "
            "Digitized avg rise SAI (~½ PDF) và best/worst variant ĐẢO NGƯỢC (EE là best, không phải worst)."
        ),
    },
    "double_tops": {
        "sample": 1413,
        "be_failure_pct": "bull 8–14% / bear 2–11%",
        "avg_move": "decline bull 15–19% / bear 19–25%",
        "meet_target_pct": "68–79%",
        "days": "ultimate low bull 43–51 / bear 32–45",
        "pullback_pct": "48–64%",
        "best_variant": "EE (decline bear 25%, BE 2%)",
        "target_rule": "lowest_low − (highest_peak − lowest_low_between_tops) / 2  [CHIA ĐÔI height]",
        "source_file": "docs/project/pdf_review/m5/family_doubles_20260813.md",
        "note": (
            "M5 PDF ch.17–20: measure rule CHIA ĐÔI height — digitized thiếu /2 → target tính gấp đôi thực tế "
            "(lỗi critical). Best/worst variant digitized ĐẢO NGƯỢC (EE là best)."
        ),
    },
    "bump_and_run_reversal": {
        "sample": "Tops 777 + Bottoms 532 = 1309",
        "be_failure_pct": "Tops bull 5% / bear 1% · Bottoms bull 2% / bear 1%",
        "avg_move": "Tops −19% / −27% · Bottoms +38% / +31%",
        "meet_target_pct": "Tops 78% / 90% · Bottoms 68% / 64%",
        "days": "Tops ultimate low 68 / 39 · Bottoms ultimate high 186 / 109",
        "pullback_pct": "Tops 62% / 65% · Bottoms throwback 59% / 73%",
        "performance_rank": "Tops bull 3/21 · bear 4/21 (top performer)",
        "target_rule": (
            "Tops (ch.8): breakout − lead_in_height [lead-in height = HH→trendline đo PHẦN TƯ ĐẦU, "
            "KHÔNG phải bump height] · Bottoms (ch.7): target = highest high trong pattern"
        ),
        "source_file": "docs/project/pdf_review/m5/family_bump_and_run_20260813.md",
        "note": (
            "M5 PDF ch.7–8: digitized dùng bump_peak−lead_in_start (SAI dimension — bump height >> lead-in height). "
            "Digitized failure 10/5 SAI (PDF 5/1); throwback 45% SAI (PDF pullback 62/65%); rank 12 SAI (PDF 3-4)."
        ),
    },
    "broadening_bottoms": {
        "sample": 237,
        "be_failure_pct": "Up bull 10% / bear 9% · Down bull 16% / bear 9%",
        "avg_move": "rise Up bull 27% / bear 21% · decline Down 15% / 18%",
        "meet_target_pct": "Up 59% / 53% · Down 44% / 31%",
        "days": "ultimate high Up bull 112 / bear 65",
        "throwback_pct": "Up 41% / 44%",
        "failure_at_10pct": "Up bull 25% / bear 29% (digitized ghi 6 → SAI)",
        "target_rule": "Up: highest_high + (highest_high − lowest_low) · Down: đối xứng",
        "source_file": "docs/project/pdf_review/m5/family_broadening_btw_20260813.md",
        "note": (
            "M5 PDF ch.1: measure rule KHỚP digitized. Digitized failure@10% SAI lớn (6 vs 25/29); "
            "thiếu sample 237, %target, toàn bộ hướng Down."
        ),
    },
    "broadening_tops": {
        "sample": 493,
        "be_failure_pct": "Down bull 18% / bear 3% · Up bull 15% / bear 11%",
        "avg_move": "decline Down bull 15% / bear 20% · rise Up 29% / 24%",
        "meet_target_pct": "Down 37% / 32% (thấp) · Up 62% / 61%",
        "days": "ultimate low Down bull 50 / bear 29",
        "pullback_pct": "Down 48% / 62%",
        "target_rule": "Down: lowest_low − (highest_high − lowest_low) · Up: đối xứng",
        "source_file": "docs/project/pdf_review/m5/family_broadening_btw_20260813.md",
        "note": (
            "M5 PDF ch.4: measure rule KHỚP. Digitized pullback 37% SAI (PDF 48/62); thiếu sample 493, "
            "%target, hướng Up."
        ),
    },
    "broadening_wedges": {
        "sample": "Asc 255 + Desc 464 = 719",
        "be_failure_pct": "Asc Up 2%/0% · Down 11%/14% | Desc Up 6%/11% · Down 9%/2%",
        "avg_move": "Asc Up +38/+18 · Down −17/−21 | Desc Up +33/+24 · Down −20/−25",
        "meet_target_pct": "Asc Up 69/60 · Down 58/86 | Desc Up 79/58 · Down 36/32",
        "days": "Asc Up 161/78 · Down 63/51 | Desc Up 131/66 · Down 40/23",
        "throwback_pct": "Asc 50/70 · Desc 53/61",
        "target_rule": "Up: highest_high + (HH − LL) · Down: lowest_low − (HH − LL)",
        "source_file": "docs/project/pdf_review/m5/family_broadening_btw_20260813.md",
        "note": (
            "M5 PDF ch.5–6: digitized broadening_wedges THIẾU HOÀN TOÀN performance stats — nạp đầy đủ từ PDF."
        ),
    },
    "diamond_bottom": {
        "sample": 295,
        "be_failure_pct": "Up bull 4% / bear 3% · Down bull 10% / bear 0% (n=20 nhỏ)",
        "avg_move": "rise Up 36% / 36% · decline Down 21% / 44%* (n=20 — Bulkowski bảo bỏ qua)",
        "meet_target_pct": "Up 81% / 60% · Down 63% / 80%",
        "days": "Up ultimate high 119 / 72 · Down ultimate low 35 / 28",
        "throwback_pct": "Up 53% / 60% · Down pullback 71% / 40%",
        "failure_at_10pct": "Up bull 12% / bear 16% (digitized ghi 2 → SAI lớn)",
        "performance_rank": "Up bull 8/23 · bear 2/19 (xuất sắc)",
        "target_rule": "Up: highest_high + (HH − LL) · Down: lowest_low − (HH − LL)",
        "source_file": "docs/project/pdf_review/m5/family_diamonds_20260813.md",
        "note": (
            "M5 PDF ch.11: digitized failure@10% = 2 SAI (PDF 12/16); avg rise bear 25% SAI (PDF 36%); "
            "thiếu sample, %target (81/60 — top performer), hướng Down."
        ),
    },
    "diamond_top": {
        "sample": 375,
        "be_failure_pct": "Down bull 6% / bear 4% · Up bull 10% / bear 0% (n=28)",
        "avg_move": "decline Down 21% / 24% · rise Up 27% / 33%",
        "meet_target_pct": "Down 76% / 59% · Up 69% / 79%",
        "days": "Down ultimate low 52 / 43 · Up ultimate high 81 / 66",
        "pullback_pct": "Down 57% / 57% · Up throwback 59% / 54%",
        "target_rule": "Down: lowest_low − (HH − LL) · Up: highest_high + (HH − LL)",
        "source_file": "docs/project/pdf_review/m5/family_diamonds_20260813.md",
        "note": (
            "M5 PDF ch.12: digitized avg decline bull 15% SAI (PDF 21%); thiếu sample 375, %target, hướng Up."
        ),
    },
    "wedges_ascending_descending": {
        "sample": "Falling 542 + Rising 621 = 1163",
        "be_failure_pct": "Falling 11/11/15/6% · Rising 8/14/24/15% (bull-UA/bear-UA/bull-UD/bear-UD)",
        "avg_move": "Falling +32/+26/−15/−24 · Rising +28/+17/−14/−20",
        "meet_target_pct": "Falling 70/60/30/36 · Rising 58/33/46/40",
        "days": "Falling 116/77/43/32 · Rising 127/60/38/38",
        "throwback_pct": "Falling 56/61/69/72 · Rising 73/66/63/63",
        "target_rule": (
            "Falling UA = HH trong wedge · Falling UD = breakout − height · "
            "Rising UD = LL trong wedge · Rising UA = breakout + (HH − LL)  [BẤT XỨNG]"
        ),
        "source_file": "docs/project/pdf_review/m5/family_wedges_20260813.md",
        "note": (
            "M5 PDF ch.52–53: digitized dùng breakout±height đối xứng — SAI cho chiều reversal "
            "(target reversal = cực trị mẫu hình). Digitized avg rise falling 20% SAI (PDF 32/26)."
        ),
    },
    "islands": {
        "sample": "Reversals 917 + Long 920",
        "be_failure_pct": "Reversals 18/10/17/5% · Long 11/4/5/2%",
        "avg_move": "Reversals +23/+21/−17/−23 · Long +31/+25/−22/−26",
        "meet_target_pct": "Reversals 62/46/69/49 · Long 82/72/78/76",
        "days": "Reversals 128/53/53/34 · Long 67/33/27/26",
        "throwback_pct": "Reversals 70/75/65/59 · Long 67/74/54/54",
        "target_rule": (
            "Reversals (ch.30): breakout ± formation height (HH−LL trong island) · "
            "Islands Long (ch.31): breakout ± HALF formation height"
        ),
        "source_file": "docs/project/pdf_review/m5/family_islands_20260813.md",
        "note": (
            "M5 PDF ch.30–31: digitized dùng gap_height — SAI concept (PDF = formation height). "
            "Digitized avg move 10% SAI (PDF 21-26%); days 14 SAI (PDF 26-128); throwback 30% SAI (PDF 54-75%); "
            "median_move/time_to_completion/gap_size_effect/duration_effect BỊA."
        ),
    },
    "rounding_bottoms_tops": {
        "sample": "RdB 453 + RdT 776 = 1229",
        "be_failure_pct": "RdB 5/5% · RdT UA 9/16% · RdT UD 12/9%",
        "avg_move": "RdB +43/+31 · RdT UA +37/+19 · RdT UD −19/−23",
        "meet_target_pct": "RdB 57/53 · RdT-UA 61/35 · RdT-UD 24/15 (gần vô dụng)",
        "days": "RdB 189/105 · RdT UA 161/77 · RdT UD 45/25",
        "throwback_pct": "RdB 40/43 · RdT UA 53/52 · RdT UD pullback 48/57",
        "performance_rank": "RdB bull 5/23 · bear 6/19 (top)",
        "target_rule": (
            "RdB: right_saucer_lip + (right_saucer_lip − lowest_low) · "
            "RdT-UA: formation_high + (formation_high − right_rim_low) · "
            "RdT-UD: right_rim_low − (formation_high − right_rim_low)  [3 công thức KHÁC nhau]"
        ),
        "source_file": "docs/project/pdf_review/m5/family_rounding_20260813.md",
        "note": (
            "M5 PDF ch.39–40: digitized dùng generic breakout±height — SAI dimension. "
            "Digitized avg rise RdB 25% SAI (PDF 43/31); failure RdT 6% SAI (PDF 9–16); "
            "RdT-UD %target CHỈ 15-24% → cảnh báo measure rule yếu chiều này."
        ),
    },
    "rising_falling_three_methods": {
        "sample": "Rising 102 + Falling 64 (Statistics EXCLUDED)",
        "continuation_pct": "Rising bull 74% / bear 79% · Falling bull 71% / bear 67%",
        "avg_move": "Rising continuation +6.86% / +3.93% (UA) — rất nhỏ",
        "meet_target_pct": "Rising 60/23/21/33% (bull-UA/bear-UA/bull-UD/bear-UD)",
        "days": "Rising candle-end→trend-end median 7/4/11/13 · →breakout 3/2/8/8",
        "overall_rank": "Rising 94/103 · Falling 89/103 (poor) · frequency 88/91 trên 103",
        "target_rule": (
            "breakout ± candle_height × multiplier%  [Rising 60/23/21/33 · "
            "Falling KHÔNG CÓ — Statistics EXCLUDED (chỉ 64 mẫu)]"
        ),
        "source_file": "docs/project/pdf_review/m5/family_three_methods_20260813.md",
        "note": (
            "M5 PDF EC ch.73/39 (KHÔNG phải ECP): failure rate digitized 20/10 là BỊA — EC không publish. "
            "Falling: toàn bộ performance BỊA (PDF exclude). Digitized target first_bar_range SAI → candle_height × %."
        ),
    },
    "measured_move_down_up": {
        "sample": "MMD 911 + MMU 810 = 1721",
        "avg_move": "MMD legs: first −27/−36 (61/45d) · retrace 48/44 (30/22d) · last −25/−36 (62/46d) | "
                   "MMU legs: first +46/+39 (87/30d) · retrace 47/50 (32/22d) · last +32/+35 (60/33d)",
        "meet_target_pct": "MMD 35/39 · MMU 45/56 (full first leg) — half first leg đạt 83–93%",
        "meet_time_target_pct": "MMD 53/49 · MMU 38/56",
        "days": "độ dài MMD 153/113 · MMU 180/85",
        "last_vs_first_leg": "last NGẮN HƠN first 19–20% (ratio ~0.80) — digitized 0.85–1.15 SAI",
        "target_rule": (
            "corrective_phase_top/bottom ∓ first_leg (FULL) → đạt 35–56% · "
            "dùng HALF first leg → đạt 83–93%  [PDF khuyến nghị half]"
        ),
        "source_file": "docs/project/pdf_review/m5/family_measured_moves_20260813.md",
        "note": (
            "M5 PDF ch.32–33: failure rate digitized 15/8 là BỊA — Bulkowski TỪ CHỐI metric này. "
            "success_rate 72 SAI (PDF 35-56). ultimate method SAI SEMANTIC (không có ultimate cho MM)."
        ),
    },
    "gaps": {
        "sample": "Area 484 + Breakaway 737 + Continuation 495 + Exhaustion 471 = 2187",
        "close_within_week_pct": "area 89–93% · breakaway 1–9% · continuation 4–20% · exhaustion 61–78%",
        "avg_time_to_close": "area 3d · breakaway 61–168d · continuation 43–98d · exhaustion 7–14d",
        "target_rule": "KHÔNG CÓ measure rule — Performance rank: Not applicable (event pattern)",
        "source_file": "docs/project/pdf_review/m5/family_gaps_20260813.md",
        "note": (
            "M5 PDF ch.23: gaps là event pattern — failure rate / ultimate days / average move / target digitized "
            "TOÀN BỘ BỊA. Metric duy nhất: Close-within-a-week + avg time to close."
        ),
    },
    "spike_formation": {
        "source": "not_in_bulkowski",
        "source_file": "docs/project/pdf_review/m5/family_spike_20260813.md",
        "note": (
            "M5 13/08 (GLM-5.2, 3 kiểm chứng độc lập): KHÔNG có chương Spike trong ECP lẫn EC — toàn bộ "
            "spike_formation_digitized.json FABRICATED. Số đang dùng là heuristic scanner, KHÔNG phải thống kê Bulkowski. "
            "Thay thế có nguồn: Shooting Star/Takuri (EC) hoặc Pipe Bottoms/Tops (ECP)."
        ),
    },
    "harami": {
        "sample": "20,000 mỗi chương × 4 chương (capped — EC ch.43-46)",
        "reversal_rate": "bearish 53C/50R · bullish 53R/51R · cross bear 57C/56C · cross bull 55C/56C (gần random)",
        "meet_target_pct": "bearish 63/58/64/64 · bullish 69/66/59/61 · cross bear 69/67/68/66 · cross bull 74/73/68/70",
        "avg_move": "≈6-10% cả 4 chương (nhỏ — harami beat S&P nhưng move tuyệt đối thấp)",
        "days": "candle-end→breakout median 3-4d · candle-end→trend-end median 6-9d",
        "overall_rank": "bearish 72/103 (poor) · bullish 38/103 · cross bear 80/103 (poor) · cross bull 50/103",
        "target_rule": (
            "breakout ± ((HH − LL) của cả 2 nến × multiplier 58-74% theo chương) — "
            "EC KHÔNG publish failure rate cho candlestick (failure digitized inside_day là BỊA nếu áp cho harami)"
        ),
        "detection_rule": "harami thường = BODY containment (cho phép đỉnh HOẶC đáy bằng, không cả hai) · harami cross = RANGE containment + doji",
        "source_file": "docs/project/pdf_review/m5/family_harami_20260813.md",
        "note": (
            "M5 PDF EC ch.43-46 (13/08, offset +24 đã kiểm chứng): tách khỏi inside_day (range-based). "
            "Reversal rate 47-57% gần random → pattern yếu về dự báo chiều; sách khuyên trade theo "
            "primary trend + opening gap confirmation. Multiplier detector dùng cột bull market."
        ),
    },
    # Batch 3 (14/08, GLM-5.3 tự trích PDF offset +23): pipes + horns — số liệu ECP ch.28/29/35/36.
    "pipe_tops": {
        "sample": 830,
        "be_failure_pct": "bull 11% / bear 2%",
        "meet_target_pct": "bull 70% / bear 68%",
        "avg_move": "decline bull 20% / bear 27%",
        "days": "ultimate low bull 75 / bear 54",
        "target_rule": (
            "ECP ch36 NGUYÊN VĂN p582: 'Compute the formation height by subtracting the lowest low "
            "from the highest high in the pipe. Subtract the result from the breakout price "
            "(the lowest low) to get a target.' — anchor = breakout_level (lowest low); "
            "detector đã đổi anchor theo nguyên văn này (batch 3, 14/08)"
        ),
        "source_file": "docs/project/pdf_review/m5/family_htf_pipes_horns_supplement_20260814.md",
        "note": (
            "WEEKLY. Width VN 5-10 tuần (median ~6-7) vs sách 'two adjacent spikes' (2 tuần liền kề) — "
            "divergence bản địa hóa có chủ đích, KHÔNG đổi width trong batch 3 (cần pilot riêng). "
            "Bảng benchmark cũ thiếu %target 70/68 — đã nạp."
        ),
    },
    "pipe_bottoms": {
        "sample": 1152,
        "be_failure_pct": "bull 5% / bear 4%",
        "meet_target_pct": "bull 83% / bear 72%",
        "avg_move": "rise bull 45% / bear 32%",
        "days": "ultimate high bull 194 / bear 133",
        "target_rule": (
            "SUY DIỄN đối xứng — ch35 KHÔNG công bố công thức (Table 35.9 p570 chỉ 4 dòng, không có "
            "Measure rule; toàn chương không mô tả cách tính target). Anchor = breakout_level "
            "(highest high) theo (i) pipe tops p582 'breakout price (the lowest low)', "
            "(ii) horn bottoms p470 'add to the highest high', (iii) Table 35.9 confirmation "
            "'close above the highest high'. INFERRED, NOT VERBATIM."
        ),
        "source_file": "docs/project/pdf_review/m5/family_htf_pipes_horns_supplement_20260814.md",
        "note": (
            "WEEKLY — mọi số công bố (BE 5/4, rise 45/32, %target 83/72) là weekly; daily pipes "
            "BE 18%/gain 33% bị Bulkowski loại bỏ (p559-560). Scanner chạy weekly W-FRI (pipes.py "
            "_to_weekly_ohlcv) — KHỚP sách; bảng cũ ghi 'daily' là SAI, đã sửa _TIMEFRAME."
        ),
    },
    "horn_bottoms_tops": {
        "sample": "Tops 323 (266+57) + Bottoms 404 (286+118)",
        "be_failure_pct": "Tops bull 7% / bear 2% · Bottoms bull 9% / bear 7%",
        "meet_target_pct": "Tops bull 70% / bear 60% · Bottoms bull 76% / bear 61%",
        "days": "Tops ultimate low bull 67 / bear 64 · Bottoms ultimate high bull 180 / bear 90",
        "target_rule": (
            "Tops (ECP ch29 NGUYÊN VĂN p483): 'Subtract the result from the lowest low to get the "
            "target price' (70%/60%) · Bottoms (ch28 NGUYÊN VĂN p470): 'Add the difference to the "
            "highest high to get the target price' (76%/61%) — anchor-biên nhất quán cả họ; "
            "detector đã đổi anchor theo nguyên văn (batch 3, 14/08)"
        ),
        "source_file": "docs/project/pdf_review/m5/family_htf_pipes_horns_supplement_20260814.md",
        "note": (
            "WEEKLY, spikes cách 1 tuần ('separated by a week') — hình học 3 tuần của detector KHỚP sách. "
            "Bảng benchmark cũ ghi horn_tops 'sách KHÔNG có %target' + book_chapter ch28 — SAI cả hai, "
            "đã nạp 70/60 + ECP ch29."
        ),
    },
}

# Ngưỡng thất bại (% kéo ngược bất lợi so mốc tham chiếu) — bảng 03 §2.3 + spec.
_FAILURE_THRESHOLD_PCT = {
    "inside_day": 1.0,
    # M5 13/08: islands threshold digitized 2.0 BỊA — PDF đo BE mốc 5% (ch.30/31) → 5.0
    "islands": 5.0,
    "horn_bottoms_tops": 3.0,
    "pipe_bottoms": 3.0,
    "pipe_tops": 3.0,
    # M5 13/08: spike digitized FABRICATED — 3.0 là heuristic scanner (KHÔNG có trong sách)
    "spike_formation": 3.0,
    "gaps": 2.0,  # fill gap = close quay lại sát breakout (K3-1: breakaway fill = failure)
    # M5 13/08: rising_falling_three_methods threshold 2.0 digitized BỊA (EC không publish
    # failure) → xóa entry, dùng mặc định 5.0.
    # còn lại mặc định 5.0 (gaps "varies_by_type" → M2 xử lý riêng)
}

# Cap số event/mã (chống 1 mã độc chiếm artifact) — từ config detector hiện tại.
_CAP_PER_FAMILY = {
    "pipe_bottoms": 18,
    "pipe_tops": 18,
    "inside_day": 12,
    "harami": 12,  # khớp HaramiConfig.max_events_per_symbol (harami.py)
    "scallops_ascending": 14,
    "scallops_descending": 14,
    "bump_and_run_reversal": 10,
}

# Mốc tham chiếu failure per family (đáy pattern / neckline / handle low / flag high...)
_FAILURE_REFERENCE = {
    "inside_day": "low_of_inside_day",
    "pipe_bottoms": "pipe_bottom_level",
    "pipe_tops": "pipe_top_level",
    "horn_bottoms_tops": "pattern_low",
    "flags": "flag_high",
    "pennants": "flag_high",
    "cup_with_handle": "handle_low",
    "head_and_shoulders_bottom": "neckline",
    "head_and_shoulders_top": "neckline",
    "triangles": "breakout_price",
    "wedges_ascending_descending": "breakout_price",
    "rectangle_bottoms_tops": "pattern_low",
    "rounding_bottoms_tops": "pattern_low",
    "scallops_ascending": "scallop_low",
    "scallops_descending": "scallop_high",
    "three_falling_peaks": "peak_high",
    "three_rising_valleys": "valley_low",
    "triple_bottoms": "pattern_low",
    "triple_tops": "pattern_high",
    "broadening_bottoms": "pattern_low",
    "broadening_tops": "pattern_high",
    "broadening_wedges": "breakout_price",
    "broadening_formations_right_angled": "breakout_price",
    "bump_and_run_reversal": "bump_low",
    "diamond_bottom": "pattern_low",
    "diamond_top": "pattern_high",
    "double_bottoms": "pattern_low",
    "double_tops": "pattern_high",
    "measured_move_down_up": "phase1_extreme",
    "gaps": "gap_edge",
    "islands": "gap_edge",
    "high_tight_flags": "flag_high",
    "dead_cat_bounce": "event_low",
    "spike_formation": "spike_extreme",
    "rising_falling_three_methods": "first_bar_range",  # K3-1: giá quay lại trong range bar đầu (03 §2.3)
    # Harami: EC không publish failure rate — mốc tham chiếu tự nhiên là đáy nến mẹ
    # (giá phá đáy nến mẹ = pattern hỏng, đối xứng inside_day dùng low_of_inside_day).
    "harami": "mother_bar_low",
    # thiếu spec → M5 bổ sung
}

# Timeframe theo từng family (sách weekly vs scanner daily — K3 plan §4).
_TIMEFRAME = {
    # Batch 3 (14/08): pipes + horns scan WEEKLY (resample W-FRI — pipes.py _to_weekly_ohlcv,
    # horns.py weekly) — bản cũ ghi "daily" là SAI so với code đang chạy.
    "pipe_bottoms": "weekly (resample W-FRI — khớp sách)",
    "pipe_tops": "weekly (resample W-FRI — khớp sách)",
    "horn_bottoms_tops": "weekly (resample W-FRI — khớp sách)",
    "dead_cat_bounce": "daily (event-driven)",
}
_TIMEFRAME_DEFAULT = "daily"

# ---------------------------------------------------------------------------
# 2. Map pattern_key (artifact/detector) → family digitized (bảng 03 §1.2)
# ---------------------------------------------------------------------------
_PATTERN_KEY_TO_FAMILY: Dict[str, str] = {
    "inside_day": "inside_day",
    "harami": "harami",  # M5-2c (13/08): detector mới — tách khỏi inside_day (body vs range)
    "rising_three_methods": "rising_falling_three_methods",
    "falling_three_methods": "rising_falling_three_methods",
    "horn_bottoms": "horn_bottoms_tops",
    "horn_tops": "horn_bottoms_tops",
    "island_reversals": "islands",
    "islands_long": "islands",
    "bull_flags": "flags",
    "bear_flags": "flags",
    "flags_experiment": "flags",
    "bull_pennants": "pennants",
    "bear_pennants": "pennants",
    "pennants": "pennants",
    "area_gaps": "gaps",
    "breakaway_gaps": "gaps",
    "continuation_gaps": "gaps",
    "exhaustion_gaps": "gaps",
    "measured_move_up": "measured_move_down_up",
    "measured_move_down": "measured_move_down_up",
    "pipe_bottoms": "pipe_bottoms",
    "pipe_tops": "pipe_tops",
    "triangles_ascending": "triangles",
    "triangles_descending": "triangles",
    "triangles_symmetrical": "triangles",
    "wedges_falling": "wedges_ascending_descending",
    "wedges_rising": "wedges_ascending_descending",
    "broadening_bottoms": "broadening_bottoms",
    "broadening_tops": "broadening_tops",
    "broadening_formations_right_angled_ascending": "broadening_formations_right_angled",
    "broadening_formations_right_angled_descending": "broadening_formations_right_angled",
    "broadening_wedges_ascending": "broadening_wedges",
    "broadening_wedges_descending": "broadening_wedges",
    "bump_and_run_reversal_bottoms": "bump_and_run_reversal",
    "bump_and_run_reversal_tops": "bump_and_run_reversal",
    "cup_with_handle": "cup_with_handle",
    "cup_with_handle_inverted": "cup_with_handle",
    "cup_with_handle_family": "cup_with_handle",
    "diamond_bottoms": "diamond_bottom",
    "diamond_tops": "diamond_top",
    "double_bottoms_aa": "double_bottoms",
    "double_bottoms_ae": "double_bottoms",
    "double_bottoms_ea": "double_bottoms",
    "double_bottoms_ee": "double_bottoms",
    "double_tops_aa": "double_tops",
    "double_tops_ae": "double_tops",
    "double_tops_ea": "double_tops",
    "double_tops_ee": "double_tops",
    "double_bottoms": "double_bottoms",
    "double_tops": "double_tops",
    # M5-2d (13/08): alias tên variant trong pattern_family_manifest.json (adam_adam...) —
    # detector runtime chỉ dùng "double_bottoms"/"double_tops" nhưng manifest dùng tên dài.
    "double_bottoms_adam_adam": "double_bottoms",
    "double_bottoms_adam_eve": "double_bottoms",
    "double_bottoms_eve_adam": "double_bottoms",
    "double_bottoms_eve_eve": "double_bottoms",
    "double_tops_adam_adam": "double_tops",
    "double_tops_adam_eve": "double_tops",
    "double_tops_eve_adam": "double_tops",
    "double_tops_eve_eve": "double_tops",
    "head_and_shoulders_bottoms": "head_and_shoulders_bottom",
    "head_and_shoulders_bottoms_complex": "head_and_shoulders_bottom",
    "head_and_shoulders_tops": "head_and_shoulders_top",
    "head_and_shoulders_tops_complex": "head_and_shoulders_top",
    "rectangle_bottoms": "rectangle_bottoms_tops",
    "rectangle_tops": "rectangle_bottoms_tops",
    "rounding_bottoms": "rounding_bottoms_tops",
    "rounding_tops": "rounding_bottoms_tops",
    "scallops_ascending": "scallops_ascending",
    "scallops_ascending_inverted": "scallops_ascending",
    "scallops_descending": "scallops_descending",
    "scallops_descending_inverted": "scallops_descending",
    "three_falling_peaks": "three_falling_peaks",
    "three_rising_valleys": "three_rising_valleys",
    "triple_tops": "triple_tops",
    "triple_bottoms": "triple_bottoms",
    "dead_cat_bounce": "dead_cat_bounce",
    "dead_cat_bounce_inverted": "dead_cat_bounce",
    "high_tight_flags": "high_tight_flags",
    "spike_formation": "spike_formation",
}

# ---------------------------------------------------------------------------
# 3. Load chuẩn digitized từ spec JSON (1 lần, cache)
# ---------------------------------------------------------------------------
_MEASUREMENTS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _load_digitized_specs() -> Dict[str, Dict[str, Any]]:
    """Đọc toàn bộ spec JSON (pdfreview ưu tiên khi trùng tên)."""
    found: Dict[str, Path] = {}
    for d in _DIGITIZED_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*_digitized.json")):
            found[p.stem.replace("_digitized", "")] = p

    specs: Dict[str, Dict[str, Any]] = {}
    for stem, path in found.items():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        pbm = doc.get("post_breakout_measurement")
        if not isinstance(pbm, dict):
            # file một số spec thiếu pbm → cờ thiếu, M5 bổ sung
            specs[stem] = {"_incomplete": True}
            continue
        uh = pbm.get("ultimate_high_method") or {}
        ul = pbm.get("ultimate_low_method") or {}
        fd = pbm.get("failure_definition") or {}
        tc = pbm.get("target_calculation") or {}
        # average_days theo từng nguồn RIÊNG (không merge — merge sai thứ tự làm
        # uh/ul đè nhau, ví dụ triangles uh=60 nhưng ul=55). Ưu tiên uh (bullish).
        def _days(src: Any) -> Dict[str, float]:
            out: Dict[str, float] = {}
            if isinstance(src, dict):
                for k, v in src.items():
                    if "days" in k and isinstance(v, (int, float)):
                        out[k] = float(v)
            return out

        specs[stem] = {
            "lookahead_bars": pbm.get("lookahead_bars"),
            "avg_days_uh": _days(uh),
            "avg_days_ul": _days(ul),
            "failure_threshold_pct": fd.get("threshold_pct"),
            "target_method": tc.get("method"),
        }
    return specs


def _family_lookahead(spec: Dict[str, Any], family: str) -> Dict[str, Any]:
    """Chọn lookahead chuẩn cho 1 family: PDF > digitized average_days > lookahead_bars."""
    pdf = _PDF_OVERRIDES.get(family)
    if pdf:
        la = pdf.get("lookahead_bull")
        note = pdf.get("note") or (
            f"PDF_REVIEW_20260812 (bull market {la}d, bear {pdf.get('lookahead_bear')}d)"
        )
        return {
            "lookahead_bars": la,
            "lookahead_bull": pdf.get("lookahead_bull"),
            "lookahead_bear": pdf.get("lookahead_bear"),
            "source": "pdf",
            "note": note,
        }
    # Ưu tiên ultimate_high (hướng tăng — đa số detector scan breakout lên)
    avg_uh = spec.get("avg_days_uh") or {}
    avg_ul = spec.get("avg_days_ul") or {}
    if "average_days" in avg_uh:
        la = avg_uh["average_days"]
        return {
            "lookahead_bars": la, "lookahead_bull": la, "lookahead_bear": None,
            "source": "digitized",
            "note": f"digitized ultimate_high.average_days = {la}d",
        }
    # Biến thể: _bottom/_top/_ascending/_descending/_breakaway... → lấy khớp family
    for key in sorted(set(list(avg_uh) + list(avg_ul))):
        if family in key or key in family:
            la = avg_uh.get(key, avg_ul.get(key))
            if la:
                return {
                    "lookahead_bars": la, "lookahead_bull": la, "lookahead_bear": None,
                    "source": "digitized",
                    "note": f"digitized {key} = {la}d",
                }
    # Fallback: bất kỳ key days nào
    all_days = {**avg_uh, **avg_ul}
    if all_days:
        key, la = next(iter(all_days.items()))
        return {
            "lookahead_bars": la, "lookahead_bull": la, "lookahead_bear": None,
            "source": "digitized",
            "note": f"digitized {key} = {la}d",
        }
    la = spec.get("lookahead_bars")
    if la:
        return {
            "lookahead_bars": la, "lookahead_bull": la, "lookahead_bear": None,
            "source": "digitized",
            "note": f"digitized lookahead_bars = {la}",
        }
    return {
        "lookahead_bars": None, "lookahead_bull": None, "lookahead_bear": None,
        "source": "missing",
        "note": "spec thiếu lookahead — chờ M5 đọc PDF",
    }


def _build_measurements() -> Dict[str, Dict[str, Any]]:
    digitized = _load_digitized_specs()
    out: Dict[str, Dict[str, Any]] = {}
    for family in sorted(set(_PATTERN_KEY_TO_FAMILY.values())):
        # spec có thể lưu dưới tên family + hậu tố (broadening_formations_right_angled_ascending)
        spec = digitized.get(family)
        if spec is None:
            for stem, s in digitized.items():
                if stem.startswith(family):
                    spec = s
                    break
        spec = spec or {}
        la = _family_lookahead(spec, family)
        out[family] = {
            "pattern_name": family,
            "lookahead_bars": la["lookahead_bars"],
            "lookahead_bull": la["lookahead_bull"],
            "lookahead_bear": la["lookahead_bear"],
            "failure_threshold_pct": _FAILURE_THRESHOLD_PCT.get(family, 5.0),
            "failure_reference": _FAILURE_REFERENCE.get(family, "unknown"),
            "target_method": spec.get("target_method") if isinstance(spec, dict) else None,
            "timeframe": _TIMEFRAME.get(family, _TIMEFRAME_DEFAULT),
            "cap": _CAP_PER_FAMILY.get(family),
            "source": la["source"],
            "note": la["note"],
        }
        extras = _PDF_EXTRAS.get(family)
        if extras:
            out[family]["pdf_extras"] = extras
    # dead_cat: event-driven, giữ detector cũ (63 qua pipes) tới M5
    if "dead_cat_bounce" in out:
        out["dead_cat_bounce"].update({
            "lookahead_bars": 63,
            "lookahead_bull": None,
            "lookahead_bear": None,
            "source": "detector_legacy",
            "note": "event-driven — không có days-to-ultimate kiểu chart pattern; giữ 63 tới M5",
        })
    # horn family-level: digitized gộp 2 chiều average_days=14 → SAI khi dùng trực tiếp.
    # K3-1 (12/08): family-level None + note — CHỈ dùng qua variant horn_bottoms/horn_tops (PDF 180/67).
    if "horn_bottoms_tops" in out:
        out["horn_bottoms_tops"].update({
            "lookahead_bars": None,
            "lookahead_bull": None,
            "lookahead_bear": None,
            "source": "variant_only",
            "note": "digitized gộp 2 chiều (14d) — CẤM dùng family-level; dùng variant horn_bottoms=180 / horn_tops=67 (PDF_REVIEW)",
        })
    # spike: digitized FABRICATED (không có chương Spike trong ECP lẫn EC — M5 13/08, 3 kiểm chứng).
    # Giữ số heuristic để scanner chạy nhưng đánh dấu nguồn not_in_bulkowski.
    if "spike_formation" in out:
        out["spike_formation"].update({
            "source": "not_in_bulkowski",
            "note": "spike_formation_digitized.json FABRICATED (M5 13/08) — số là heuristic scanner, KHÔNG phải Bulkowski",
        })
    return out


# ---------------------------------------------------------------------------
# 4. API công khai
# ---------------------------------------------------------------------------
def _measurements() -> Dict[str, Dict[str, Any]]:
    global _MEASUREMENTS_CACHE
    if _MEASUREMENTS_CACHE is None:
        _MEASUREMENTS_CACHE = _build_measurements()
    return _MEASUREMENTS_CACHE


def family_of(pattern_key: str) -> str:
    """pattern_key artifact → family digitized (nếu chưa biết → trả nguyên pattern_key)."""
    return _PATTERN_KEY_TO_FAMILY.get(pattern_key, pattern_key)


def measurement_for(pattern_key: str) -> Dict[str, Any]:
    """Chuẩn đo lường đầy đủ cho pattern_key (dict, không None)."""
    # pattern_key CON (horn_bottoms, rounding_tops, breakaway_gaps...) → số riêng
    variant = _VARIANT_LOOKAHEAD.get(pattern_key)
    fam = family_of(pattern_key)
    m = _measurements().get(fam)
    if m is None:
        m = {
            "pattern_name": fam, "lookahead_bars": None, "lookahead_bull": None,
            "lookahead_bear": None, "failure_threshold_pct": 5.0,
            "failure_reference": "unknown", "target_method": None,
            "timeframe": _TIMEFRAME_DEFAULT, "cap": None,
            "source": "missing", "note": "family chưa có registry — cần bổ sung",
        }
    m = dict(m)  # bản sao — không sửa cache
    if variant:
        m.update({
            "lookahead_bars": variant.get("lookahead_bull"),
            "lookahead_bull": variant.get("lookahead_bull"),
            "lookahead_bear": variant.get("lookahead_bear"),
            "source": variant.get("source", m.get("source")),
            "note": variant.get("note", m.get("note")),
        })
    return m


def lookahead_bars(pattern_key: str) -> Optional[int]:
    """Số phiên đo sau breakout (chuẩn V3). Detector dùng số này.

    Luôn trả int — spec digitized lưu float (vd 5.0) gây lỗi iloc
    "indexers of type float" khi detector cắt cửa sổ tương lai.
    """
    la = measurement_for(pattern_key).get("lookahead_bars")
    return int(la) if la is not None else None


def lookahead_weeks(pattern_key: str) -> Optional[int]:
    """Lookahead quy đổi SANG TUẦN (1 tuần = 5 phiên giao dịch).

    Chuẩn Bulkowski đo bằng NGÀY GIAO DỊCH; detector pipes/horns/rounding
    scan trên dữ liệu TUẦN (mỗi bar = 1 tuần) nên phải quy đổi:
    ceil(ngày / 5). Không quy đổi → đo gấp ~5 lần, MFE phồng (vd pipe_bottom 155%).
    """
    days = lookahead_bars(pattern_key)
    if days is None:
        return None
    return -(-days // 5)  # ceil division


def failure_threshold_pct(pattern_key: str) -> float:
    return float(measurement_for(pattern_key).get("failure_threshold_pct", 5.0))


def failure_reference(pattern_key: str) -> str:
    return measurement_for(pattern_key).get("failure_reference", "unknown")


def cap(pattern_key: str) -> Optional[int]:
    return measurement_for(pattern_key).get("cap")


def all_measurements() -> Dict[str, Dict[str, Any]]:
    return dict(_measurements())


def verify_consistency() -> Dict[str, Any]:
    """Kiểm tra nội bộ: mọi pattern_key map được family + mọi family có lookahead.

    (K3-1 phán quyết 12/08: bỏ tautology — check cũ luôn rỗng vô nghĩa.)
    """
    missing = [k for k, v in _measurements().items() if v.get("lookahead_bars") is None]
    unknown_keys = [k for k in _PATTERN_KEY_TO_FAMILY if family_of(k) not in _measurements()]
    return {
        "families": len(_measurements()),
        "pattern_keys": len(_PATTERN_KEY_TO_FAMILY),
        "families_missing_lookahead": missing,
        "unknown_pattern_keys": unknown_keys,
    }


if __name__ == "__main__":
    import sys

    if "--table" in sys.argv:
        print(f"{'pattern_key':<42}{'family':<32}{'la':<6}{'fail%':<7}{'cap':<5}{'source':<18}note")
        for pk in sorted(_PATTERN_KEY_TO_FAMILY):
            m = measurement_for(pk)
            print(f"{pk:<42}{m['pattern_name']:<32}{str(m['lookahead_bars']):<6}"
                  f"{m['failure_threshold_pct']:<7}{str(m['cap']):<5}{m['source']:<18}{m['note'][:60]}")
    else:
        print(json.dumps(verify_consistency(), ensure_ascii=False, indent=2))
