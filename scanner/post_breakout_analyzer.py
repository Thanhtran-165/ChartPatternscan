"""
Post-Breakout Evaluator
========================

EVALUATION LAYER ONLY - Uses look-ahead data (252 bars) for statistics.
Should NOT be used during detection/scanning phase.

This module evaluates detected patterns AFTER formation to compute:
- Failure rates (2 definitions per Bulkowski)
- Ultimate price and time-to-ultimate
- Throwback/pullback rates
- Target achievement

ARCHITECTURE:
┌─────────────────┐     ┌──────────────────────┐
│  Detection      │     │  Evaluation          │
│  (no lookahead) │ ──► │  (lookahead 252 bars)│
│  - Pivot detect │     │  - Failure rate      │
│  - Pattern form │     │  - Ultimate price    │
│  - Breakout     │     │  - Throwback rate    │
│    confirm only │     │  - Target achieved   │
└─────────────────┘     └──────────────────────┘

DEFINITIONS (per digitized specs):
----------------------------------
1. FAILURE RATE - Two definitions:
   a) Bust Failure: Price reverses X% against breakout direction
      - For bearish: price rises X% from breakout price
      - For bullish: price falls X% from breakout price
      - Thresholds: 5%, 10% (standard)

   b) Invalidation Failure: Price returns across pattern boundary
      - For bearish: price rises above trough/neckline
      - For bullish: price falls below peak/neckline

2. ULTIMATE PRICE:
   - Ultimate Low (bearish): Lowest price within window before:
     a) 20% reversal upward, OR
     b) New pattern forms, OR
     c) End of window (252 bars)
   - Ultimate High (bullish): Highest price within window before same stops

3. THROWBACK/PULLBACK:
   - Throwback (bearish): Price returns to breakout price level
   - Pullback (bullish): Price returns to breakout price level
   - Tolerance: within X% of breakout price (default 1%)

4. TARGET ACHIEVEMENT:
   - Price reaches target_price (measured from pattern height)
   - Can be measured as: intraday touch OR closing price
   - time_to_target: trading days from breakout to target hit
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np
import json


class FailureType(Enum):
    """Types of pattern failure"""
    NONE = "none"
    BUST_5PCT = "bust_5pct"      # Price reversed 5% against breakout
    BUST_10PCT = "bust_10pct"    # Price reversed 10% against breakout
    BOUNDARY = "boundary"        # Price crossed pattern boundary


class TargetMethod(Enum):
    """Methods for measuring target achievement"""
    INTRADAY = "intraday"  # Any touch of target level
    CLOSE = "close"        # Closing price at/above target


@dataclass
class PostBreakoutResult:
    """
    Evaluation results for a single pattern.

    All metrics use LOOK-AHEAD data and should NOT be used for detection.
    """
    # Identity
    pattern_id: str
    symbol: str
    pattern_name: str

    # Breakout info (from detection)
    breakout_date: Optional[str]
    breakout_price: Optional[float]
    breakout_direction: Optional[str]  # 'up' or 'down'
    target_price: Optional[float]
    stop_loss_price: Optional[float]

    # === FAILURE METRICS ===
    # Definition 1: Bust failure (price reversal from breakout)
    bust_failure_5pct: Optional[bool]   # Reversed >= 5% from breakout
    bust_failure_10pct: Optional[bool]  # Reversed >= 10% from breakout
    bust_failure_date: Optional[str]    # Date of first bust failure
    bust_failure_pct: Optional[float]   # Actual % reversal

    # Definition 2: Boundary invalidation
    boundary_invalidated: Optional[bool]  # Price crossed boundary
    boundary_invalidation_date: Optional[str]

    # === ULTIMATE PRICE METRICS ===
    ultimate_price: Optional[float]      # Ultimate low (bearish) or high (bullish)
    ultimate_date: Optional[str]
    days_to_ultimate: Optional[int]
    ultimate_stop_reason: Optional[str]  # '20pct_reversal', 'new_pattern', 'end_of_window'

    # === THROWBACK/PULLBACK METRICS ===
    throwback_pullback_occurred: Optional[bool]
    throwback_pullback_date: Optional[str]
    days_to_throwback_pullback: Optional[int]
    # Which level was retested
    retested_breakout_level: Optional[bool]    # Retested breakout price
    retested_boundary_level: Optional[bool]    # Retested pattern boundary

    # === TARGET METRICS ===
    target_achieved_intraday: Optional[bool]   # Target touched (intraday)
    target_achieved_close: Optional[bool]      # Target closed at/above
    target_achievement_date: Optional[str]
    days_to_target: Optional[int]

    # === EXCURSION METRICS ===
    max_favorable_excursion_pct: Optional[float]  # Max % in expected direction
    max_adverse_excursion_pct: Optional[float]    # Max % against expected

    # === VARIANT METRICS (Double Tops) ===
    variant: Optional[str] = None  # 'AA', 'AE', 'EA', 'EE' (Adam/Eve)
    peak1_width_bars: Optional[int] = None
    peak2_width_bars: Optional[int] = None

    # Analysis metadata
    evaluation_window_bars: int = 252
    evaluation_date: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationConfig:
    """Configuration for post-breakout evaluation"""
    # Window settings
    lookahead_bars: int = 252  # ~1 year of trading days

    # Ultimate price stop rules
    reversal_threshold_pct: float = 20.0  # 20% reversal stops ultimate search

    # Failure thresholds
    bust_failure_thresholds: Tuple[float, ...] = (5.0, 10.0)  # Standard thresholds

    # Throwback/Pullback settings
    throwback_tolerance_pct: float = 1.0  # Within 1% of breakout = retest

    # Boundary invalidation gating (to reduce "any-time" false invalidations)
    invalidation_return_threshold_pct: float = 3.0
    invalidation_within_bars: int = 10

    # Variant classification (double tops) - Adam/Eve by peak width
    variant_peak_width_tolerance_pct: float = 2.0
    variant_peak_width_window_bars: int = 15
    variant_adam_max_peak_width_bars: int = 3
    variant_eve_min_peak_width_bars: int = 7

    # Target settings
    target_method: TargetMethod = TargetMethod.INTRADAY


class PostBreakoutEvaluator:
    """
    Evaluates pattern performance AFTER breakout using look-ahead data.

    IMPORTANT: This evaluator uses future data and should ONLY be used
    for statistical analysis, NOT for real-time detection.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()

    def evaluate(self,
                 detection: Any,
                 df_full: pd.DataFrame,
                 pattern_boundary_price: Optional[float] = None,
                 pattern_height: Optional[float] = None) -> PostBreakoutResult:
        """
        Evaluate a pattern detection using look-ahead data.

        Args:
            detection: PatternDetection object from scanner
            df_full: Full OHLCV DataFrame (must extend beyond breakout)
            pattern_boundary_price: Trough price (bearish) or Peak price (bullish)
            pattern_height: Pattern height for calculations

        Returns:
            PostBreakoutResult with all evaluation metrics
        """
        # Find breakout index
        breakout_idx = self._find_breakout_index(detection, df_full)

        if breakout_idx is None or breakout_idx >= len(df_full) - 1:
            return self._empty_result(detection)

        # Get evaluation window
        # IMPORTANT: post-breakout metrics must use bars AFTER the breakout bar,
        # otherwise throwback/pullback and bust can be trivially triggered on the breakout bar itself.
        start_idx = breakout_idx + 1
        window_end = min(start_idx + self.config.lookahead_bars, len(df_full))
        future_df = df_full.iloc[start_idx:window_end].copy()

        if len(future_df) < 2:
            return self._empty_result(detection)

        breakout_price = detection.breakout_price or df_full.iloc[breakout_idx]['close']
        try:
            breakout_price = float(breakout_price)
        except Exception:
            return self._empty_result(detection)
        if not np.isfinite(breakout_price) or breakout_price <= 0:
            return self._empty_result(detection)

        direction = detection.breakout_direction or self._infer_direction(detection)

        # Infer boundary/height when not supplied by the caller.
        if pattern_boundary_price is None:
            pattern_boundary_price = self._infer_boundary_price(detection, df_full)
        if pattern_height is None:
            pattern_height = self._infer_pattern_height(detection, breakout_price)

        variant, peak1_w, peak2_w = self._classify_double_top_variant(detection, df_full)

        # Evaluate based on breakout direction
        if direction == 'down' or detection.pattern_type == 'reversal_bearish':
            res = self._evaluate_bearish(
                detection, future_df, breakout_idx, breakout_price,
                pattern_boundary_price, pattern_height
            )
        else:
            res = self._evaluate_bullish(
                detection, future_df, breakout_idx, breakout_price,
                pattern_boundary_price, pattern_height
            )

        res.variant = variant
        res.peak1_width_bars = peak1_w
        res.peak2_width_bars = peak2_w
        return res

    def _evaluate_bearish(self,
                          detection: Any,
                          future_df: pd.DataFrame,
                          breakout_idx: int,
                          breakout_price: float,
                          boundary_price: Optional[float],
                          pattern_height: Optional[float]) -> PostBreakoutResult:
        """Evaluate bearish pattern (expected downward move)"""

        # === FAILURE ANALYSIS ===
        # Definition 1: Bust failure (reversal from breakout)
        # Use CLOSE-based bust to match typical Bulkowski-style definitions (avoid wick noise).
        cumulative_close_high = future_df['close'].expanding().max()
        bust_pct = (cumulative_close_high - breakout_price) / breakout_price * 100

        bust_5pct = bool((bust_pct >= 5.0).any())
        bust_10pct = bool((bust_pct >= 10.0).any())

        # Find first bust date
        bust_failure_date = None
        bust_5_mask = bust_pct >= 5.0
        if bust_5_mask.any():
            first_bust_idx = bust_5_mask[bust_5_mask].index[0]
            bust_failure_date = str(future_df.loc[first_bust_idx, 'date'].date()) if 'date' in future_df.columns else None

        actual_bust_pct = bust_pct.max()

        # Definition 2: Boundary invalidation
        boundary_invalidated = None
        boundary_invalidation_date = None
        if boundary_price is not None:
            # For bearish: invalidation when close returns above boundary by a threshold,
            # within a limited time window (to avoid "any-time" invalidations).
            try:
                bp = float(boundary_price)
            except Exception:
                bp = None
            if bp is None or not np.isfinite(bp) or bp <= 0:
                bp = None
            if bp is None:
                pass
            else:
                thr = bp * (1.0 + self.config.invalidation_return_threshold_pct / 100.0)
                within = max(1, int(self.config.invalidation_within_bars))
                subset = future_df.iloc[: min(len(future_df), within + 1)]
                boundary_mask = subset['close'] >= thr
                boundary_invalidated = bool(boundary_mask.any())
                if boundary_invalidated:
                    first_invalid_idx = boundary_mask[boundary_mask].index[0]
                    boundary_invalidation_date = str(subset.loc[first_invalid_idx, 'date'].date()) if 'date' in subset.columns else None

        # === ULTIMATE LOW ===
        ultimate_price, ultimate_date, days_to_ultimate, stop_reason = \
            self._find_ultimate_bearish(future_df, breakout_price)

        # === THROWBACK ===
        # Retest of breakout price level
        throwback_occurred = None
        throwback_date = None
        days_to_throwback = None
        retested_breakout = None
        retested_boundary = None

        tolerance = self.config.throwback_tolerance_pct / 100
        # Breakout level for throwback should be the *pattern boundary* at breakout (e.g., trough/neckline),
        # not the breakout close. Using the close makes throwback almost always true.
        throw_level = breakout_price
        if boundary_price is not None:
            try:
                bp = float(boundary_price)
            except Exception:
                bp = None
            if bp is not None and np.isfinite(bp) and bp > 0:
                throw_level = bp

        breakout_retest_mask = future_df['high'] >= breakout_price * (1 - tolerance)
        boundary_retest_mask = future_df['high'] >= throw_level * (1 - tolerance)

        throwback_occurred = bool(boundary_retest_mask.any())

        if throwback_occurred:
            first_throwback_idx = boundary_retest_mask[boundary_retest_mask].index[0]
            throwback_date = str(future_df.loc[first_throwback_idx, 'date'].date()) if 'date' in future_df.columns else None
            days_to_throwback = first_throwback_idx - breakout_idx
            retested_breakout = bool(breakout_retest_mask.any())
            retested_boundary = True

        # Also check boundary retest
        if not throwback_occurred:
            retested_breakout = bool(breakout_retest_mask.any()) if breakout_retest_mask is not None else None
            retested_boundary = bool(boundary_retest_mask.any()) if boundary_retest_mask is not None else None

        # === TARGET ACHIEVEMENT ===
        target_achieved_intraday = None
        target_achieved_close = None
        target_date = None
        days_to_target = None

        if detection.target_price is not None:
            # Intraday: any touch
            intraday_mask = future_df['low'] <= detection.target_price
            target_achieved_intraday = bool(intraday_mask.any())
            if target_achieved_intraday:
                first_target_idx = intraday_mask[intraday_mask].index[0]
                target_date = str(future_df.loc[first_target_idx, 'date'].date()) if 'date' in future_df.columns else None
                days_to_target = first_target_idx - breakout_idx

            # Close: closing price at or below target
            close_mask = future_df['close'] <= detection.target_price
            target_achieved_close = bool(close_mask.any())

        # === EXCURSION ===
        ultimate_low = future_df['low'].min()
        ultimate_high = future_df['high'].max()

        max_favorable_pct = (breakout_price - ultimate_low) / breakout_price * 100
        max_adverse_pct = (ultimate_high - breakout_price) / breakout_price * 100

        return PostBreakoutResult(
            pattern_id=detection.pattern_id,
            symbol=detection.symbol,
            pattern_name=detection.pattern_name,
            breakout_date=detection.breakout_date,
            breakout_price=breakout_price,
            breakout_direction='down',
            target_price=detection.target_price,
            stop_loss_price=detection.stop_loss_price,
            # Failures
            bust_failure_5pct=bust_5pct,
            bust_failure_10pct=bust_10pct,
            bust_failure_date=bust_failure_date,
            bust_failure_pct=round(actual_bust_pct, 2) if actual_bust_pct else None,
            boundary_invalidated=boundary_invalidated,
            boundary_invalidation_date=boundary_invalidation_date,
            # Ultimate
            ultimate_price=ultimate_price,
            ultimate_date=ultimate_date,
            days_to_ultimate=days_to_ultimate,
            ultimate_stop_reason=stop_reason,
            # Throwback
            throwback_pullback_occurred=throwback_occurred,
            throwback_pullback_date=throwback_date,
            days_to_throwback_pullback=days_to_throwback,
            retested_breakout_level=retested_breakout,
            retested_boundary_level=retested_boundary,
            # Target
            target_achieved_intraday=target_achieved_intraday,
            target_achieved_close=target_achieved_close,
            target_achievement_date=target_date,
            days_to_target=days_to_target,
            # Excursion
            max_favorable_excursion_pct=round(max_favorable_pct, 2),
            max_adverse_excursion_pct=round(max_adverse_pct, 2),
            evaluation_window_bars=self.config.lookahead_bars
        )

    def _evaluate_bullish(self,
                          detection: Any,
                          future_df: pd.DataFrame,
                          breakout_idx: int,
                          breakout_price: float,
                          boundary_price: Optional[float],
                          pattern_height: Optional[float]) -> PostBreakoutResult:
        """Evaluate bullish pattern (expected upward move)"""

        # === FAILURE ANALYSIS ===
        # Definition 1: Bust failure (reversal from breakout)
        cumulative_close_low = future_df['close'].expanding().min()
        bust_pct = (breakout_price - cumulative_close_low) / breakout_price * 100

        bust_5pct = bool((bust_pct >= 5.0).any())
        bust_10pct = bool((bust_pct >= 10.0).any())

        bust_failure_date = None
        bust_5_mask = bust_pct >= 5.0
        if bust_5_mask.any():
            first_bust_idx = bust_5_mask[bust_5_mask].index[0]
            bust_failure_date = str(future_df.loc[first_bust_idx, 'date'].date()) if 'date' in future_df.columns else None

        actual_bust_pct = bust_pct.max()

        # Definition 2: Boundary invalidation
        boundary_invalidated = None
        boundary_invalidation_date = None
        if boundary_price is not None:
            try:
                bp = float(boundary_price)
            except Exception:
                bp = None
            if bp is None or not np.isfinite(bp) or bp <= 0:
                bp = None
            if bp is None:
                pass
            else:
                thr = bp * (1.0 - self.config.invalidation_return_threshold_pct / 100.0)
                within = max(1, int(self.config.invalidation_within_bars))
                subset = future_df.iloc[: min(len(future_df), within + 1)]
                # For bullish: invalidation when close returns below boundary by a threshold.
                boundary_mask = subset['close'] <= thr
                boundary_invalidated = bool(boundary_mask.any())
                if boundary_invalidated:
                    first_invalid_idx = boundary_mask[boundary_mask].index[0]
                    boundary_invalidation_date = str(subset.loc[first_invalid_idx, 'date'].date()) if 'date' in subset.columns else None

        # === ULTIMATE HIGH ===
        ultimate_price, ultimate_date, days_to_ultimate, stop_reason = \
            self._find_ultimate_bullish(future_df, breakout_price)

        # === PULLBACK ===
        tolerance = self.config.throwback_tolerance_pct / 100
        pull_level = breakout_price
        if boundary_price is not None:
            try:
                bp = float(boundary_price)
            except Exception:
                bp = None
            if bp is not None and np.isfinite(bp) and bp > 0:
                pull_level = bp

        breakout_retest_mask = future_df['low'] <= breakout_price * (1 + tolerance)
        boundary_retest_mask = future_df['low'] <= pull_level * (1 + tolerance)
        pullback_mask = boundary_retest_mask
        pullback_occurred = bool(pullback_mask.any())

        pullback_date = None
        days_to_pullback = None
        retested_breakout = None
        retested_boundary = None

        if pullback_occurred:
            first_pullback_idx = pullback_mask[pullback_mask].index[0]
            pullback_date = str(future_df.loc[first_pullback_idx, 'date'].date()) if 'date' in future_df.columns else None
            days_to_pullback = first_pullback_idx - breakout_idx
            retested_breakout = bool(breakout_retest_mask.any())
            retested_boundary = True

        if not pullback_occurred:
            retested_breakout = bool(breakout_retest_mask.any()) if breakout_retest_mask is not None else None
            retested_boundary = bool(boundary_retest_mask.any()) if boundary_retest_mask is not None else None

        # === TARGET ACHIEVEMENT ===
        target_achieved_intraday = None
        target_achieved_close = None
        target_date = None
        days_to_target = None

        if detection.target_price is not None:
            intraday_mask = future_df['high'] >= detection.target_price
            target_achieved_intraday = bool(intraday_mask.any())
            if target_achieved_intraday:
                first_target_idx = intraday_mask[intraday_mask].index[0]
                target_date = str(future_df.loc[first_target_idx, 'date'].date()) if 'date' in future_df.columns else None
                days_to_target = first_target_idx - breakout_idx

            close_mask = future_df['close'] >= detection.target_price
            target_achieved_close = bool(close_mask.any())

        # === EXCURSION ===
        ultimate_high = future_df['high'].max()
        ultimate_low = future_df['low'].min()

        max_favorable_pct = (ultimate_high - breakout_price) / breakout_price * 100
        max_adverse_pct = (breakout_price - ultimate_low) / breakout_price * 100

        return PostBreakoutResult(
            pattern_id=detection.pattern_id,
            symbol=detection.symbol,
            pattern_name=detection.pattern_name,
            breakout_date=detection.breakout_date,
            breakout_price=breakout_price,
            breakout_direction='up',
            target_price=detection.target_price,
            stop_loss_price=detection.stop_loss_price,
            bust_failure_5pct=bust_5pct,
            bust_failure_10pct=bust_10pct,
            bust_failure_date=bust_failure_date,
            bust_failure_pct=round(actual_bust_pct, 2) if actual_bust_pct else None,
            boundary_invalidated=boundary_invalidated,
            boundary_invalidation_date=boundary_invalidation_date,
            ultimate_price=ultimate_price,
            ultimate_date=ultimate_date,
            days_to_ultimate=days_to_ultimate,
            ultimate_stop_reason=stop_reason,
            throwback_pullback_occurred=pullback_occurred,
            throwback_pullback_date=pullback_date,
            days_to_throwback_pullback=days_to_pullback,
            retested_breakout_level=retested_breakout,
            retested_boundary_level=retested_boundary,
            target_achieved_intraday=target_achieved_intraday,
            target_achieved_close=target_achieved_close,
            target_achievement_date=target_date,
            days_to_target=days_to_target,
            max_favorable_excursion_pct=round(max_favorable_pct, 2),
            max_adverse_excursion_pct=round(max_adverse_pct, 2),
            evaluation_window_bars=self.config.lookahead_bars
        )

    def _find_ultimate_bearish(self, future_df: pd.DataFrame,
                               breakout_price: float) -> Tuple[Optional[float], Optional[str], Optional[int], Optional[str]]:
        """
        Find ultimate low for bearish pattern.

        Stops when:
        1. 20% reversal upward from lowest point
        2. End of window (252 bars)
        """
        if len(future_df) == 0:
            return None, None, None, None

        running_low = float('inf')
        running_low_idx = None

        for i, (idx, row) in enumerate(future_df.iterrows()):
            low = row['low']

            if low < running_low:
                running_low = low
                running_low_idx = idx

            # Check for 20% reversal
            if running_low < breakout_price:
                reversal_pct = (row['high'] - running_low) / running_low * 100
                if reversal_pct >= self.config.reversal_threshold_pct:
                    # Stopped by reversal
                    ultimate_date = str(future_df.loc[running_low_idx, 'date'].date()) if 'date' in future_df.columns else None
                    days = running_low_idx - future_df.index[0] if running_low_idx else None
                    return running_low, ultimate_date, days, '20pct_reversal'

        # End of window
        if running_low_idx is not None:
            ultimate_date = str(future_df.loc[running_low_idx, 'date'].date()) if 'date' in future_df.columns else None
            days = running_low_idx - future_df.index[0]
            return running_low, ultimate_date, days, 'end_of_window'

        return None, None, None, None

    def _find_ultimate_bullish(self, future_df: pd.DataFrame,
                               breakout_price: float) -> Tuple[Optional[float], Optional[str], Optional[int], Optional[str]]:
        """
        Find ultimate high for bullish pattern.

        Stops when:
        1. 20% reversal downward from highest point
        2. End of window (252 bars)
        """
        if len(future_df) == 0:
            return None, None, None, None

        running_high = 0
        running_high_idx = None

        for i, (idx, row) in enumerate(future_df.iterrows()):
            high = row['high']

            if high > running_high:
                running_high = high
                running_high_idx = idx

            # Check for 20% reversal
            if running_high > breakout_price:
                reversal_pct = (running_high - row['low']) / running_high * 100
                if reversal_pct >= self.config.reversal_threshold_pct:
                    ultimate_date = str(future_df.loc[running_high_idx, 'date'].date()) if 'date' in future_df.columns else None
                    days = running_high_idx - future_df.index[0] if running_high_idx else None
                    return running_high, ultimate_date, days, '20pct_reversal'

        if running_high_idx is not None:
            ultimate_date = str(future_df.loc[running_high_idx, 'date'].date()) if 'date' in future_df.columns else None
            days = running_high_idx - future_df.index[0]
            return running_high, ultimate_date, days, 'end_of_window'

        return None, None, None, None

    def _find_breakout_index(self, detection: Any, df: pd.DataFrame) -> Optional[int]:
        """Find the index of breakout in the DataFrame"""
        bi = getattr(detection, "breakout_idx", None)
        if bi is not None:
            try:
                bi_int = int(bi)
            except Exception:
                bi_int = None
            if bi_int is not None and 0 <= bi_int < len(df):
                return bi_int

        if detection.breakout_date and 'date' in df.columns:
            try:
                s = df["date"].dt.date.astype(str)
                mask = s == str(detection.breakout_date)
                if bool(mask.any()):
                    # Index is a RangeIndex in our pipeline; cast defensively.
                    return int(mask[mask].index[0])
            except Exception:
                # Fallback: slow path
                for _, (idx, row) in enumerate(df.iterrows()):
                    if str(row['date'].date()) == detection.breakout_date:
                        return int(idx) if isinstance(idx, (int, np.integer)) else None
        return None

    def _infer_direction(self, detection: Any) -> str:
        """Infer breakout direction from pattern type"""
        if detection.pattern_type == 'reversal_bearish':
            return 'down'
        elif detection.pattern_type == 'reversal_bullish':
            return 'up'
        return 'down'

    def _infer_boundary_price(self, detection: Any, df_full: pd.DataFrame) -> Optional[float]:
        """
        Infer a reasonable "pattern boundary" price level for boundary invalidation / retest metrics.

        We prefer using pivot_indices + breakout_direction so this works for both legacy and
        digitized scanners without needing extra fields persisted in detections.
        """
        pattern_name = str(getattr(detection, "pattern_name", "") or "")

        piv = getattr(detection, "pivot_indices", None)
        if isinstance(piv, str):
            try:
                piv = json.loads(piv)
            except Exception:
                piv = None
        pivots: List[int] = []
        if isinstance(piv, (list, tuple)):
            for x in piv:
                try:
                    xi = int(x)
                except Exception:
                    continue
                if 0 <= xi < len(df_full):
                    pivots.append(xi)

        bdir = getattr(detection, "breakout_direction", None) or self._infer_direction(detection)

        try:
            if pattern_name == "double_tops" and len(pivots) >= 2:
                return float(df_full.iloc[pivots[1]]["low"])
            if pattern_name == "double_bottoms" and len(pivots) >= 2:
                return float(df_full.iloc[pivots[1]]["high"])
            if pattern_name == "head_and_shoulders_top" and len(pivots) >= 4:
                nl1 = float(df_full.iloc[pivots[1]]["low"])
                nl2 = float(df_full.iloc[pivots[3]]["low"])
                return (nl1 + nl2) / 2.0
            if pattern_name == "head_and_shoulders_bottom" and len(pivots) >= 4:
                nl1 = float(df_full.iloc[pivots[1]]["high"])
                nl2 = float(df_full.iloc[pivots[3]]["high"])
                return (nl1 + nl2) / 2.0
        except Exception:
            # Fall back to generic logic below.
            pass

        try:
            if pivots and bdir == "down":
                vals = [float(df_full.iloc[i]["low"]) for i in pivots]
                return float(min(vals)) if vals else None
            if pivots and bdir == "up":
                vals = [float(df_full.iloc[i]["high"]) for i in pivots]
                return float(max(vals)) if vals else None
        except Exception:
            pass

        # Last resort: use stop_loss_price if it's available and sane.
        stop = getattr(detection, "stop_loss_price", None)
        try:
            stop_f = float(stop) if stop is not None else None
        except Exception:
            stop_f = None
        if stop_f is not None and np.isfinite(stop_f) and stop_f > 0:
            return stop_f
        return None

    def _infer_pattern_height(self, detection: Any, breakout_price: float) -> Optional[float]:
        """
        Infer pattern height in *price units*.

        Prefer using target_price when available: height_abs = |target - breakout|.
        """
        tp = getattr(detection, "target_price", None)
        if tp is None:
            return None
        try:
            tp_f = float(tp)
        except Exception:
            return None
        height = abs(tp_f - float(breakout_price))
        return height if np.isfinite(height) and height > 0 else None

    def _peak_width_bars(self, df: pd.DataFrame, peak_idx: int, peak_price: float) -> Optional[int]:
        if peak_idx < 0 or peak_idx >= len(df):
            return None
        if peak_price <= 0 or not np.isfinite(peak_price):
            return None

        tol = float(self.config.variant_peak_width_tolerance_pct) / 100.0
        window = max(1, int(self.config.variant_peak_width_window_bars))
        thr = peak_price * (1.0 - tol)

        left = peak_idx
        for k in range(1, window + 1):
            j = peak_idx - k
            if j < 0:
                break
            if float(df.iloc[j]["high"]) >= thr:
                left = j
            else:
                break

        right = peak_idx
        for k in range(1, window + 1):
            j = peak_idx + k
            if j >= len(df):
                break
            if float(df.iloc[j]["high"]) >= thr:
                right = j
            else:
                break

        return int(right - left + 1)

    def _classify_double_top_variant(
        self, detection: Any, df_full: pd.DataFrame
    ) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        if str(getattr(detection, "pattern_name", "") or "") != "double_tops":
            return None, None, None

        piv = getattr(detection, "pivot_indices", None)
        if not isinstance(piv, (list, tuple)) or len(piv) < 3:
            return None, None, None

        try:
            peak1_idx = int(piv[0])
            peak2_idx = int(piv[2])
        except Exception:
            return None, None, None

        if not (0 <= peak1_idx < len(df_full) and 0 <= peak2_idx < len(df_full)):
            return None, None, None

        try:
            peak1_price = float(df_full.iloc[peak1_idx]["high"])
            peak2_price = float(df_full.iloc[peak2_idx]["high"])
        except Exception:
            return None, None, None

        w1 = self._peak_width_bars(df_full, peak1_idx, peak1_price)
        w2 = self._peak_width_bars(df_full, peak2_idx, peak2_price)

        def _ae(width: Optional[int]) -> Optional[str]:
            if width is None:
                return None
            if width <= int(self.config.variant_adam_max_peak_width_bars):
                return "A"
            if width >= int(self.config.variant_eve_min_peak_width_bars):
                return "E"
            return None

        t1 = _ae(w1)
        t2 = _ae(w2)
        if t1 and t2:
            return f"{t1}{t2}", w1, w2
        return None, w1, w2

    def _empty_result(self, detection: Any) -> PostBreakoutResult:
        """Return empty result when evaluation cannot be performed"""
        return PostBreakoutResult(
            pattern_id=detection.pattern_id,
            symbol=detection.symbol,
            pattern_name=detection.pattern_name,
            breakout_date=detection.breakout_date,
            breakout_price=detection.breakout_price,
            breakout_direction=detection.breakout_direction,
            target_price=detection.target_price,
            stop_loss_price=detection.stop_loss_price,
            bust_failure_5pct=None,
            bust_failure_10pct=None,
            bust_failure_date=None,
            bust_failure_pct=None,
            boundary_invalidated=None,
            boundary_invalidation_date=None,
            ultimate_price=None,
            ultimate_date=None,
            days_to_ultimate=None,
            ultimate_stop_reason=None,
            throwback_pullback_occurred=None,
            throwback_pullback_date=None,
            days_to_throwback_pullback=None,
            retested_breakout_level=None,
            retested_boundary_level=None,
            target_achieved_intraday=None,
            target_achieved_close=None,
            target_achievement_date=None,
            days_to_target=None,
            max_favorable_excursion_pct=None,
            max_adverse_excursion_pct=None,
            evaluation_window_bars=self.config.lookahead_bars,
        )


class StatisticsAggregator:
    """
    Aggregates evaluation results into summary statistics.
    """

    def aggregate(self, results: List[PostBreakoutResult]) -> Dict[str, Any]:
        """
        Aggregate multiple evaluation results into summary statistics.
        """
        if not results:
            return {'total_patterns': 0}

        # Filter to results with breakouts
        with_breakout = [r for r in results if r.breakout_price is not None and r.breakout_date is not None]

        stats = {
            'total_patterns': len(results),
            'patterns_with_breakout': len(with_breakout),
            'patterns_without_breakout': len(results) - len(with_breakout),
        }

        if not with_breakout:
            return stats

        # === FAILURE RATES ===
        # Bust failure
        bust_5 = [r for r in with_breakout if r.bust_failure_5pct is True]
        bust_10 = [r for r in with_breakout if r.bust_failure_10pct is True]
        stats['bust_failure_rate_5pct'] = len(bust_5) / len(with_breakout) * 100
        stats['bust_failure_rate_10pct'] = len(bust_10) / len(with_breakout) * 100

        # Boundary invalidation
        boundary_failed = [r for r in with_breakout if r.boundary_invalidated is True]
        stats['boundary_invalidation_rate_pct'] = len(boundary_failed) / len(with_breakout) * 100

        # === TARGET ACHIEVEMENT ===
        target_intraday = [r for r in with_breakout if r.target_achieved_intraday is True]
        target_close = [r for r in with_breakout if r.target_achieved_close is True]
        stats['target_achievement_rate_intraday_pct'] = len(target_intraday) / len(with_breakout) * 100
        stats['target_achievement_rate_close_pct'] = len(target_close) / len(with_breakout) * 100

        # Days to target
        target_days = [r.days_to_target for r in with_breakout if r.days_to_target is not None]
        if target_days:
            stats['avg_days_to_target'] = np.mean(target_days)
            stats['median_days_to_target'] = np.median(target_days)

        # === THROWBACK/PULLBACK ===
        throwback = [r for r in with_breakout if r.throwback_pullback_occurred is True]
        stats['throwback_pullback_rate_pct'] = len(throwback) / len(with_breakout) * 100

        # Breakout vs boundary retest
        breakout_retest = [r for r in with_breakout if r.retested_breakout_level is True]
        boundary_retest = [r for r in with_breakout if r.retested_boundary_level is True]
        stats['breakout_retest_rate_pct'] = len(breakout_retest) / len(with_breakout) * 100
        stats['boundary_retest_rate_pct'] = len(boundary_retest) / len(with_breakout) * 100

        # === ULTIMATE ===
        days_to_ultimate = [r.days_to_ultimate for r in with_breakout if r.days_to_ultimate is not None]
        if days_to_ultimate:
            stats['avg_days_to_ultimate'] = np.mean(days_to_ultimate)
            stats['median_days_to_ultimate'] = np.median(days_to_ultimate)

        # Stop reasons
        stop_reasons = [r.ultimate_stop_reason for r in with_breakout if r.ultimate_stop_reason]
        if stop_reasons:
            stats['ultimate_stop_reasons'] = pd.Series(stop_reasons).value_counts().to_dict()

        # === EXCURSION ===
        mfe = [r.max_favorable_excursion_pct for r in with_breakout if r.max_favorable_excursion_pct is not None]
        mae = [r.max_adverse_excursion_pct for r in with_breakout if r.max_adverse_excursion_pct is not None]
        if mfe:
            stats['avg_max_favorable_excursion_pct'] = np.mean(mfe)
            stats['median_max_favorable_excursion_pct'] = np.median(mfe)
        if mae:
            stats['avg_max_adverse_excursion_pct'] = np.mean(mae)
            stats['median_max_adverse_excursion_pct'] = np.median(mae)

        # === BY PATTERN TYPE ===
        by_pattern = {}
        for pattern_name in set(r.pattern_name for r in with_breakout):
            pattern_results = [r for r in with_breakout if r.pattern_name == pattern_name]
            by_pattern[pattern_name] = self._aggregate_pattern(pattern_results)
        stats['by_pattern'] = by_pattern

        # === DOUBLE TOPS VARIANTS (AA/AE/EA/EE) ===
        dt = [r for r in with_breakout if r.pattern_name == "double_tops"]
        if dt:
            unknown = 0
            by_var: Dict[str, Dict[str, Any]] = {}
            for r in dt:
                if not r.variant:
                    unknown += 1
                    continue
                v = str(r.variant)
                bucket = by_var.setdefault(v, {"count": 0, "mfe": [], "mae": []})
                bucket["count"] += 1
                if r.max_favorable_excursion_pct is not None and np.isfinite(r.max_favorable_excursion_pct):
                    bucket["mfe"].append(float(r.max_favorable_excursion_pct))
                if r.max_adverse_excursion_pct is not None and np.isfinite(r.max_adverse_excursion_pct):
                    bucket["mae"].append(float(r.max_adverse_excursion_pct))

            if by_var:
                stats["double_tops_variant_unknown_count"] = unknown
                stats["double_tops_by_variant"] = {
                    v: {
                        "count": int(b["count"]),
                        "avg_mfe_pct": float(np.mean(b["mfe"])) if b["mfe"] else None,
                        "avg_mae_pct": float(np.mean(b["mae"])) if b["mae"] else None,
                    }
                    for v, b in sorted(by_var.items())
                }

        return stats

    def _aggregate_pattern(self, results: List[PostBreakoutResult]) -> Dict[str, Any]:
        """Aggregate statistics for a single pattern type"""
        n = len(results)
        return {
            'count': n,
            'bust_failure_rate_5pct': sum(1 for r in results if r.bust_failure_5pct) / n * 100 if n else 0,
            'bust_failure_rate_10pct': sum(1 for r in results if r.bust_failure_10pct) / n * 100 if n else 0,
            'boundary_invalidation_rate_pct': sum(1 for r in results if r.boundary_invalidated) / n * 100 if n else 0,
            'target_achievement_intraday_pct': sum(1 for r in results if r.target_achieved_intraday) / n * 100 if n else 0,
            'throwback_pullback_rate_pct': sum(1 for r in results if r.throwback_pullback_occurred) / n * 100 if n else 0,
        }


# Convenience function for batch evaluation
def evaluate_detections(detections: List[Any],
                        df_dict: Dict[str, pd.DataFrame],
                        pattern_boundaries: Optional[Dict[str, float]] = None,
                        pattern_heights: Optional[Dict[str, float]] = None,
                        config: Optional[EvaluationConfig] = None) -> Tuple[List[PostBreakoutResult], Dict[str, Any]]:
    """
    Evaluate multiple pattern detections.

    Args:
        detections: List of PatternDetection objects
        df_dict: Dictionary mapping symbol to full OHLCV DataFrame
        pattern_boundaries: Dict mapping pattern_id to boundary price
        pattern_heights: Dict mapping pattern_id to pattern height
        config: Evaluation configuration

    Returns:
        Tuple of (list of results, aggregated statistics)
    """
    evaluator = PostBreakoutEvaluator(config)
    aggregator = StatisticsAggregator()

    results = []
    for d in detections:
        df = df_dict.get(d.symbol)
        if df is None:
            continue

        boundary = pattern_boundaries.get(d.pattern_id) if pattern_boundaries else None
        height = pattern_heights.get(d.pattern_id) if pattern_heights else None

        result = evaluator.evaluate(d, df, boundary, height)
        results.append(result)

    stats = aggregator.aggregate(results)
    return results, stats


if __name__ == "__main__":
    print(__doc__)
    print("\nClasses:")
    print("  - EvaluationConfig: Configuration for evaluation")
    print("  - PostBreakoutEvaluator: Evaluates patterns with look-ahead")
    print("  - PostBreakoutResult: Result dataclass with all metrics")
    print("  - StatisticsAggregator: Aggregates results into stats")
