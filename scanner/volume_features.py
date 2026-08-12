"""Daily OHLCV volume diagnostics for pattern scanners.

The functions here deliberately use daily OHLCV only.  They do not attempt to
recreate tick VWAP, block money flow, or other intraday-only indicators.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b <= 0:
        return None
    return a / b


def _slope(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna().reset_index(drop=True)
    if len(clean) < 3:
        return None
    x_mean = (len(clean) - 1) / 2.0
    y_mean = float(clean.mean())
    denom = sum((i - x_mean) ** 2 for i in range(len(clean)))
    if denom == 0:
        return None
    return float(sum((i - x_mean) * (float(v) - y_mean) for i, v in enumerate(clean)) / denom)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    close_num = pd.to_numeric(close, errors="coerce")
    vol_num = pd.to_numeric(volume, errors="coerce").fillna(0.0)
    direction = close_num.diff().map(lambda v: 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)).fillna(0.0)
    return (direction * vol_num).cumsum()


def _vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    close_num = pd.to_numeric(close, errors="coerce")
    vol_num = pd.to_numeric(volume, errors="coerce").fillna(0.0)
    pct = close_num.pct_change().replace([float("inf"), -float("inf")], pd.NA).fillna(0.0)
    return (pct * vol_num).cumsum()


def _mfi(df: pd.DataFrame, window: int = 14) -> float | None:
    if len(df) < window + 1:
        return None
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    typical = (high + low + close) / 3.0
    raw_flow = typical * volume
    up = raw_flow.where(typical.diff() > 0, 0.0)
    down = raw_flow.where(typical.diff() < 0, 0.0).abs()
    pos = float(up.tail(window).sum())
    neg = float(down.tail(window).sum())
    if neg <= 0 and pos <= 0:
        return None
    if neg <= 0:
        return 100.0
    ratio = pos / neg
    return 100.0 - (100.0 / (1.0 + ratio))


def _vwma(close: pd.Series, volume: pd.Series, window: int) -> float | None:
    if len(close) < window:
        return None
    c = pd.to_numeric(close.tail(window), errors="coerce")
    v = pd.to_numeric(volume.tail(window), errors="coerce").fillna(0.0)
    denom = float(v.sum())
    if denom <= 0:
        return None
    return float((c * v).sum() / denom)


def _phase(close_now: float | None, close_prev: float | None, volume_now: float | None, volume_ref: float | None) -> str:
    if close_now is None or close_prev is None or volume_now is None or volume_ref is None or volume_ref <= 0:
        return "unknown"
    price_up = close_now > close_prev
    volume_up = volume_now > volume_ref
    if price_up and volume_up:
        return "up_confirmed"
    if price_up and not volume_up:
        return "up_weak"
    if not price_up and volume_up:
        return "down_confirmed"
    return "down_drying"


def compute_latest_volume_features(
    df: pd.DataFrame,
    *,
    setup_start_date: Any | None = None,
    lookback: int = 20,
) -> dict[str, Any]:
    """Return daily-volume diagnostics for the latest row in ``df``."""
    if df.empty or "volume" not in df.columns:
        return {
            "volume_quality_label": "unknown",
            "volume_warning_label": "missing_volume",
        }
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if work.empty:
        return {
            "volume_quality_label": "unknown",
            "volume_warning_label": "missing_volume",
        }
    latest = work.iloc[-1]
    prior = work.iloc[:-1].tail(int(lookback))
    latest_volume = _num(latest.get("volume"))
    latest_value = _num(latest.get("value"))
    if latest_value is None:
        latest_value = (_num(latest.get("close")) or 0.0) * (latest_volume or 0.0)
    prior_volume_median = _num(pd.to_numeric(prior.get("volume", pd.Series(dtype=float)), errors="coerce").median()) if not prior.empty else None
    prior_value = prior.get("value")
    if prior_value is None and {"close", "volume"}.issubset(prior.columns):
        prior_value = pd.to_numeric(prior["close"], errors="coerce") * pd.to_numeric(prior["volume"], errors="coerce")
    prior_value_median = _num(pd.to_numeric(prior_value, errors="coerce").median()) if prior_value is not None and not prior.empty else None
    volume_ratio = _safe_ratio(latest_volume, prior_volume_median)
    value_ratio = _safe_ratio(latest_value, prior_value_median)
    volume_slope = _slope(work["volume"].tail(lookback))
    obv = _obv(work["close"], work["volume"])
    vpt = _vpt(work["close"], work["volume"])
    obv_slope = _slope(obv.tail(lookback))
    vpt_slope = _slope(vpt.tail(lookback))
    mfi_14 = _mfi(work, 14)
    vwma_fast = _vwma(work["close"], work["volume"], 5)
    vwma_slow = _vwma(work["close"], work["volume"], 20)
    setup_contraction = None
    if setup_start_date is not None:
        setup_dt = pd.to_datetime(setup_start_date, errors="coerce")
        if not pd.isna(setup_dt):
            setup_window = work.loc[work["date"] >= setup_dt]
            before_setup = work.loc[work["date"] < setup_dt].tail(lookback)
            setup_med = _num(pd.to_numeric(setup_window.get("volume", pd.Series(dtype=float)), errors="coerce").median())
            before_med = _num(pd.to_numeric(before_setup.get("volume", pd.Series(dtype=float)), errors="coerce").median())
            setup_contraction = _safe_ratio(setup_med, before_med)
    close_now = _num(latest.get("close"))
    close_prev = _num(work.iloc[-2].get("close")) if len(work) >= 2 else None
    phase = _phase(close_now, close_prev, latest_volume, prior_volume_median)

    warning = "none"
    if volume_ratio is None:
        warning = "missing_volume"
    elif volume_ratio >= 2.0 and phase == "down_confirmed":
        warning = "adverse_volume_spike"
    elif setup_contraction is not None and setup_contraction >= 1.35:
        warning = "noisy_setup_volume"
    elif value_ratio is not None and value_ratio < 0.45:
        warning = "thin_value"

    quality = "unknown"
    if volume_ratio is not None:
        if phase == "up_confirmed" and volume_ratio >= 1.25:
            quality = "strong"
        elif phase in {"up_confirmed", "down_drying"} and volume_ratio >= 0.75:
            quality = "healthy"
        elif warning != "none":
            quality = "risky"
        else:
            quality = "weak"

    out = {
        "volume_ratio_20": round(float(volume_ratio), 4) if volume_ratio is not None else None,
        "value_ratio_20": round(float(value_ratio), 4) if value_ratio is not None else None,
        "volume_trend_slope_20": round(float(volume_slope), 4) if volume_slope is not None else None,
        "price_volume_phase": phase,
        "volume_quality_label": quality,
        "volume_warning_label": warning,
        "pattern_volume_contraction_ratio": round(float(setup_contraction), 4) if setup_contraction is not None else None,
        "obv_slope_20": round(float(obv_slope), 4) if obv_slope is not None else None,
        "vpt_slope_20": round(float(vpt_slope), 4) if vpt_slope is not None else None,
        "mfi_14": round(float(mfi_14), 2) if mfi_14 is not None else None,
        "vwma_fast_minus_slow": round(float(vwma_fast - vwma_slow), 4) if vwma_fast is not None and vwma_slow is not None else None,
        "vwma_trend_confirmed": bool(close_now is not None and vwma_slow is not None and close_now >= vwma_slow),
    }
    return out
