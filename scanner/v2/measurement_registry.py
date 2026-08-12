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
    # inside_day: PDF lệch ĐỊNH NGHĨA (body Harami vs range) → KHÔNG dùng số PDF.
    # dead_cat: event-driven, không có "days to ultimate" kiểu chart pattern.
    # horn/rectangle/rounding tách theo pattern_key con (bảng _VARIANT_LOOKAHEAD).
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
    # digitized key biến thể: rounding average_days_bottom 84 / top 63
    "rounding_bottoms": {"lookahead_bull": 84, "lookahead_bear": None, "source": "digitized", "note": "digitized average_days_bottom = 84"},
    "rounding_tops": {"lookahead_bull": 63, "lookahead_bear": None, "source": "digitized", "note": "digitized average_days_top = 63"},
    # digitized key biến thể: gaps breakaway 42 / continuation 21 / exhaustion 5
    "breakaway_gaps": {"lookahead_bull": 42, "lookahead_bear": None, "source": "digitized", "note": "digitized breakaway_average_days = 42"},
    "continuation_gaps": {"lookahead_bull": 21, "lookahead_bear": None, "source": "digitized", "note": "digitized continuation_average_days = 21"},
    "exhaustion_gaps": {"lookahead_bull": 5, "lookahead_bear": None, "source": "digitized", "note": "digitized exhaustion_average_days = 5"},
    "area_gaps": {"lookahead_bull": 63, "lookahead_bear": None, "source": "digitized", "note": "area_gaps không có số riêng → lookahead_bars 63"},
}

# Ngưỡng thất bại (% kéo ngược bất lợi so mốc tham chiếu) — bảng 03 §2.3 + spec.
_FAILURE_THRESHOLD_PCT = {
    "inside_day": 1.0,
    "islands": 2.0,
    "rising_falling_three_methods": 2.0,
    "horn_bottoms_tops": 3.0,
    "pipe_bottoms": 3.0,
    "pipe_tops": 3.0,
    "spike_formation": 3.0,
    "gaps": 2.0,  # fill gap = close quay lại sát breakout (K3-1: breakaway fill = failure)
    # còn lại mặc định 5.0 (gaps "varies_by_type" → M2 xử lý riêng)
}

# Cap số event/mã (chống 1 mã độc chiếm artifact) — từ config detector hiện tại.
_CAP_PER_FAMILY = {
    "pipe_bottoms": 18,
    "pipe_tops": 18,
    "inside_day": 12,
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
    # thiếu spec → M5 bổ sung
}

# Timeframe theo từng family (sách weekly vs scanner daily — K3 plan §4).
_TIMEFRAME = {
    "pipe_bottoms": "daily (scanner) / weekly (sách — không so trực tiếp)",
    "pipe_tops": "daily",
    "dead_cat_bounce": "daily (event-driven)",
}
_TIMEFRAME_DEFAULT = "daily"

# ---------------------------------------------------------------------------
# 2. Map pattern_key (artifact/detector) → family digitized (bảng 03 §1.2)
# ---------------------------------------------------------------------------
_PATTERN_KEY_TO_FAMILY: Dict[str, str] = {
    "inside_day": "inside_day",
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
        return {
            "lookahead_bars": la,
            "lookahead_bull": pdf.get("lookahead_bull"),
            "lookahead_bear": pdf.get("lookahead_bear"),
            "source": "pdf",
            "note": f"PDF_REVIEW_20260812 (bull market {pdf.get('lookahead_bull')}d, bear {pdf.get('lookahead_bear')}d)",
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
