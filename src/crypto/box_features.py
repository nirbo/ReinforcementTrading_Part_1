"""
Intraday Dynamic Box Strategy feature components.

This module implements the box strategy where today's high/low/mid form a dynamic
trading "arena". Features track price's relationship to these levels for RL.

Box Structure:
    Today's High ─────────── CEILING (resistance/breakout)
                   UPPER ZONE (bullish bias)
    50% Midpoint ─────────── PIVOT (mean reversion)
                   LOWER ZONE (bearish bias)
    Today's Low  ─────────── FLOOR (support/breakdown)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class Zone(str, Enum):
    """Price zone classification relative to box levels."""

    ABOVE_BOX = "above_box"    # Price above today's high (breakout territory)
    UPPER = "upper"            # Price between mid and high (bullish bias)
    AT_HIGH = "at_high"        # Price at or near high level
    AT_MID = "at_mid"          # Price at or near mid level
    AT_LOW = "at_low"          # Price at or near low level
    LOWER = "lower"            # Price between low and mid (bearish bias)
    BELOW_BOX = "below_box"    # Price below today's low (breakdown territory)


@dataclass
class BoxState:
    """
    Tracks intraday box state for the dynamic box strategy.

    The box is defined by today's session high/low, with a 50% midpoint.
    This class tracks the box boundaries and interactions (touches, breakouts).

    Attributes:
        session_high: Highest high in current session
        session_low: Lowest low in current session
        session_open: Opening price of session (first bar's open)
        high_touch_count: Number of times price touched box high
        low_touch_count: Number of times price touched box low
        mid_touch_count: Number of times price touched box midpoint
        broke_high: Whether price ever broke above box_high this session
        broke_low: Whether price ever broke below box_low this session
        last_zone: Zone from previous bar (for transition detection)
        bar_count: Number of bars in current session
        _is_touching_high: Internal state for touch hysteresis
        _is_touching_low: Internal state for touch hysteresis
        _is_touching_mid: Internal state for touch hysteresis
        _last_high_touch_bar: Bar index of last high touch (for recency)
        _last_low_touch_bar: Bar index of last low touch
        _last_mid_touch_bar: Bar index of last mid touch

    Usage:
        state = BoxState()
        state.reset(first_bar_open, first_bar_high, first_bar_low)

        for bar in bars:
            state.update(bar.high, bar.low, bar.close, tolerance)
    """

    # Box boundaries
    session_high: float = 0.0
    session_low: float = float('inf')  # Start high so first bar sets it
    session_open: float = 0.0

    # Touch counts (capped for normalization)
    high_touch_count: int = 0
    low_touch_count: int = 0
    mid_touch_count: int = 0

    # Breakout state (sticky - once True, stays True)
    broke_high: bool = False
    broke_low: bool = False

    # Zone tracking
    last_zone: Zone = field(default=Zone.LOWER)

    # Session progress
    bar_count: int = 0

    # Internal hysteresis state (for touch detection)
    _is_touching_high: bool = field(default=False, repr=False)
    _is_touching_low: bool = field(default=False, repr=False)
    _is_touching_mid: bool = field(default=False, repr=False)

    # Recency tracking (bar index of last touch)
    _last_high_touch_bar: int = field(default=-1, repr=False)
    _last_low_touch_bar: int = field(default=-1, repr=False)
    _last_mid_touch_bar: int = field(default=-1, repr=False)

    # Maximum touch count (for normalization)
    MAX_TOUCH_COUNT: int = field(default=5, repr=False, init=False)

    @property
    def box_mid(self) -> float:
        """Calculate current box midpoint."""
        if self.session_high == 0.0 or self.session_low == float('inf'):
            return 0.0
        return (self.session_high + self.session_low) / 2

    @property
    def box_height(self) -> float:
        """Calculate current box height."""
        if self.session_high == 0.0 or self.session_low == float('inf'):
            return 0.0
        return self.session_high - self.session_low

    @property
    def is_valid(self) -> bool:
        """Check if box state is valid (has been initialized with data)."""
        return (
            self.session_high > 0.0 and
            self.session_low < float('inf') and
            self.session_low > 0.0 and
            self.bar_count > 0
        )

    def reset(
        self,
        first_open: float,
        first_high: float,
        first_low: float,
    ) -> None:
        """
        Reset state for new session.

        Called at session boundary (UTC midnight) with first bar data.

        Args:
            first_open: Open price of first bar
            first_high: High of first bar
            first_low: Low of first bar
        """
        self.session_high = first_high
        self.session_low = first_low
        self.session_open = first_open

        # Reset counts
        self.high_touch_count = 0
        self.low_touch_count = 0
        self.mid_touch_count = 0

        # Reset breakout flags
        self.broke_high = False
        self.broke_low = False

        # Reset zone (will be set properly on first update)
        self.last_zone = Zone.LOWER

        # Reset bar count
        self.bar_count = 1

        # Reset hysteresis
        self._is_touching_high = False
        self._is_touching_low = False
        self._is_touching_mid = False

        # Reset recency
        self._last_high_touch_bar = -1
        self._last_low_touch_bar = -1
        self._last_mid_touch_bar = -1

    def update(
        self,
        high: float,
        low: float,
        close: float,
        tolerance_pct: float = 0.001,
    ) -> Zone:
        """
        Update state with new bar data.

        Args:
            high: Bar's high price
            low: Bar's low price
            close: Bar's close price
            tolerance_pct: Tolerance for "at level" detection (default 0.1%)

        Returns:
            Current zone classification
        """
        self.bar_count += 1

        # Store OLD boundaries for breakout/zone detection
        old_high = self.session_high
        old_low = self.session_low
        old_mid = self.box_mid
        old_height = self.box_height

        # Tolerance in price terms (use old box for consistency)
        if old_height > 0:
            tolerance = old_height * tolerance_pct
        else:
            tolerance = close * tolerance_pct  # Fallback for tiny boxes

        # Check breakout state BEFORE expanding box (sticky flags)
        if close > old_high:
            self.broke_high = True
        if close < old_low:
            self.broke_low = True

        # Classify zone relative to OLD boundaries
        current_zone = self._classify_zone(close, old_high, old_low, old_mid, tolerance)
        self.last_zone = current_zone

        # Detect touches with hysteresis (relative to OLD boundaries)
        self._update_touches(close, old_high, old_low, old_mid, tolerance)

        # NOW expand box boundaries (expand only, never shrink)
        if high > self.session_high:
            self.session_high = high
        if low < self.session_low:
            self.session_low = low

        return current_zone

    def _update_touches(
        self,
        close: float,
        box_high: float,
        box_low: float,
        box_mid: float,
        tolerance: float,
    ) -> None:
        """
        Update touch counts with hysteresis to avoid double-counting.

        A touch is counted when price enters the tolerance zone from outside.
        Must leave the zone before another touch can be counted.

        Uses passed-in boundaries (old box) rather than current session_high/low.
        """
        # High touch detection
        at_high = abs(close - box_high) <= tolerance
        if at_high and not self._is_touching_high:
            # Just entered high zone
            self.high_touch_count = min(self.high_touch_count + 1, self.MAX_TOUCH_COUNT)
            self._last_high_touch_bar = self.bar_count
        self._is_touching_high = at_high

        # Low touch detection
        at_low = abs(close - box_low) <= tolerance
        if at_low and not self._is_touching_low:
            # Just entered low zone
            self.low_touch_count = min(self.low_touch_count + 1, self.MAX_TOUCH_COUNT)
            self._last_low_touch_bar = self.bar_count
        self._is_touching_low = at_low

        # Mid touch detection
        at_mid = abs(close - box_mid) <= tolerance
        if at_mid and not self._is_touching_mid:
            # Just entered mid zone
            self.mid_touch_count = min(self.mid_touch_count + 1, self.MAX_TOUCH_COUNT)
            self._last_mid_touch_bar = self.bar_count
        self._is_touching_mid = at_mid

    def _classify_zone(
        self,
        close: float,
        box_high: float,
        box_low: float,
        box_mid: float,
        tolerance: float,
    ) -> Zone:
        """
        Classify price into zone based on position relative to box levels.

        Uses passed-in boundaries (old box) rather than current session_high/low.
        """
        # Check "at level" first (highest priority)
        if abs(close - box_high) <= tolerance:
            return Zone.AT_HIGH
        if abs(close - box_low) <= tolerance:
            return Zone.AT_LOW
        if abs(close - box_mid) <= tolerance:
            return Zone.AT_MID

        # Check breakout zones
        if close > box_high:
            return Zone.ABOVE_BOX
        if close < box_low:
            return Zone.BELOW_BOX

        # Within box
        if close > box_mid:
            return Zone.UPPER
        else:
            return Zone.LOWER

    def bars_since_high_touch(self) -> int:
        """Get number of bars since last high touch (-1 if never touched)."""
        if self._last_high_touch_bar < 0:
            return -1
        return self.bar_count - self._last_high_touch_bar

    def bars_since_low_touch(self) -> int:
        """Get number of bars since last low touch (-1 if never touched)."""
        if self._last_low_touch_bar < 0:
            return -1
        return self.bar_count - self._last_low_touch_bar

    def bars_since_mid_touch(self) -> int:
        """Get number of bars since last mid touch (-1 if never touched)."""
        if self._last_mid_touch_bar < 0:
            return -1
        return self.bar_count - self._last_mid_touch_bar

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Used for environment checkpointing.
        """
        return {
            "session_high": self.session_high,
            "session_low": self.session_low if self.session_low != float('inf') else None,
            "session_open": self.session_open,
            "high_touch_count": self.high_touch_count,
            "low_touch_count": self.low_touch_count,
            "mid_touch_count": self.mid_touch_count,
            "broke_high": self.broke_high,
            "broke_low": self.broke_low,
            "last_zone": self.last_zone.value,
            "bar_count": self.bar_count,
            "_is_touching_high": self._is_touching_high,
            "_is_touching_low": self._is_touching_low,
            "_is_touching_mid": self._is_touching_mid,
            "_last_high_touch_bar": self._last_high_touch_bar,
            "_last_low_touch_bar": self._last_low_touch_bar,
            "_last_mid_touch_bar": self._last_mid_touch_bar,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoxState":
        """
        Create BoxState from dictionary.

        Used for environment checkpoint restoration.
        """
        state = cls()
        state.session_high = data.get("session_high", 0.0)
        state.session_low = data.get("session_low") or float('inf')
        state.session_open = data.get("session_open", 0.0)
        state.high_touch_count = data.get("high_touch_count", 0)
        state.low_touch_count = data.get("low_touch_count", 0)
        state.mid_touch_count = data.get("mid_touch_count", 0)
        state.broke_high = data.get("broke_high", False)
        state.broke_low = data.get("broke_low", False)
        state.last_zone = Zone(data.get("last_zone", Zone.LOWER.value))
        state.bar_count = data.get("bar_count", 0)
        state._is_touching_high = data.get("_is_touching_high", False)
        state._is_touching_low = data.get("_is_touching_low", False)
        state._is_touching_mid = data.get("_is_touching_mid", False)
        state._last_high_touch_bar = data.get("_last_high_touch_bar", -1)
        state._last_low_touch_bar = data.get("_last_low_touch_bar", -1)
        state._last_mid_touch_bar = data.get("_last_mid_touch_bar", -1)
        return state

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "BoxState":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return (
            f"BoxState(high={self.session_high:.2f}, low={self.session_low:.2f}, "
            f"mid={self.box_mid:.2f}, touches=({self.high_touch_count},{self.mid_touch_count},{self.low_touch_count}), "
            f"broke=(H:{self.broke_high},L:{self.broke_low}), zone={self.last_zone.value}, bars={self.bar_count})"
        )


@dataclass
class ValidationSignals:
    """
    Validation signals for box strategy entries.

    These signals help confirm price rejection at key levels by analyzing
    candle structure and volume context.

    Attributes:
        rejection_wick_ratio: Normalized wick asymmetry [-1, 1].
            Positive = larger lower wick (bullish rejection).
            Negative = larger upper wick (bearish rejection).
        volume_vs_session_avg: Current volume relative to session average [0.1, 5.0].
            > 1.0 indicates above-average activity.
        approach_bars: Bars since entering current zone.
            Tracks how long price has been in the current area.
        zone_entry_bar: Bar index when current zone was entered.

    Usage:
        signals = ValidationSignals()
        result = signals.calculate(open, high, low, close, volume, session_avg_vol)
    """

    rejection_wick_ratio: float = 0.0
    volume_vs_session_avg: float = 1.0
    approach_bars: int = 0
    zone_entry_bar: int = 0

    # Constants for normalization bounds
    VOLUME_RATIO_MIN: float = field(default=0.1, repr=False, init=False)
    VOLUME_RATIO_MAX: float = field(default=5.0, repr=False, init=False)
    EPS: float = field(default=1e-8, repr=False, init=False)

    def calculate_rejection_wick_ratio(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
    ) -> float:
        """
        Calculate rejection wick ratio from OHLC data.

        Measures wick asymmetry to detect price rejection at levels.
        - Upper wick = high - max(open, close)
        - Lower wick = min(open, close) - low
        - Ratio = (lower_wick - upper_wick) / candle_range

        Returns:
            Normalized ratio in [-1, 1]:
            - Positive: Larger lower wick (bullish rejection, wick pointing down)
            - Negative: Larger upper wick (bearish rejection, wick pointing up)
            - Zero: Equal wicks (doji-like)
        """
        candle_range = high - low

        # Handle zero-range candle (doji at single price)
        if candle_range < self.EPS:
            return 0.0

        body_top = max(open_price, close)
        body_bottom = min(open_price, close)

        upper_wick = high - body_top
        lower_wick = body_bottom - low

        # Ratio: (lower - upper) / range gives positive for bullish rejection
        ratio = (lower_wick - upper_wick) / (candle_range + self.EPS)

        # Clip to bounds for numerical stability
        return float(np.clip(ratio, -1.0, 1.0))

    def calculate_volume_vs_session_avg(
        self,
        current_volume: float,
        session_avg_volume: float,
    ) -> float:
        """
        Calculate current volume relative to session average.

        Args:
            current_volume: Volume of current bar
            session_avg_volume: Average volume for the session

        Returns:
            Volume ratio capped to [VOLUME_RATIO_MIN, VOLUME_RATIO_MAX].
            Values > 1.0 indicate above-average activity.
        """
        # Handle zero/negative volume edge cases
        if session_avg_volume < self.EPS:
            return 1.0  # Default to neutral if no session data

        if current_volume < 0:
            current_volume = 0.0

        ratio = current_volume / (session_avg_volume + self.EPS)

        # Cap to reasonable bounds
        return float(np.clip(ratio, self.VOLUME_RATIO_MIN, self.VOLUME_RATIO_MAX))

    def calculate_approach_bars(
        self,
        current_bar: int,
        zone_entry_bar: int,
    ) -> int:
        """
        Calculate bars since entering current zone.

        Args:
            current_bar: Current bar index (BoxState.bar_count)
            zone_entry_bar: Bar index when zone was entered

        Returns:
            Number of bars since zone entry (0 if just entered, -1 if invalid)
        """
        if zone_entry_bar < 0:
            return -1  # Zone entry not tracked

        bars_since = current_bar - zone_entry_bar
        return max(0, bars_since)

    def calculate(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        session_avg_volume: float,
        current_bar: int = 0,
        zone_entry_bar: int = 0,
    ) -> Dict[str, float]:
        """
        Calculate all validation signals for current bar.

        Args:
            open_price: Bar's open price
            high: Bar's high price
            low: Bar's low price
            close: Bar's close price
            volume: Bar's volume
            session_avg_volume: Average volume for the session
            current_bar: Current bar index (for approach_bars tracking)
            zone_entry_bar: Bar index when current zone was entered

        Returns:
            Dictionary with all validation signals:
            - rejection_wick_ratio: [-1, 1]
            - volume_vs_session_avg: [0.1, 5.0]
            - approach_bars: >= 0 or -1 if not tracked
        """
        self.rejection_wick_ratio = self.calculate_rejection_wick_ratio(
            open_price, high, low, close
        )
        self.volume_vs_session_avg = self.calculate_volume_vs_session_avg(
            volume, session_avg_volume
        )
        self.approach_bars = self.calculate_approach_bars(current_bar, zone_entry_bar)
        self.zone_entry_bar = zone_entry_bar

        return {
            "rejection_wick_ratio": self.rejection_wick_ratio,
            "volume_vs_session_avg": self.volume_vs_session_avg,
            "approach_bars": self.approach_bars,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rejection_wick_ratio": self.rejection_wick_ratio,
            "volume_vs_session_avg": self.volume_vs_session_avg,
            "approach_bars": self.approach_bars,
            "zone_entry_bar": self.zone_entry_bar,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationSignals":
        """Create ValidationSignals from dictionary."""
        signals = cls()
        signals.rejection_wick_ratio = data.get("rejection_wick_ratio", 0.0)
        signals.volume_vs_session_avg = data.get("volume_vs_session_avg", 1.0)
        signals.approach_bars = data.get("approach_bars", 0)
        signals.zone_entry_bar = data.get("zone_entry_bar", 0)
        return signals


def calculate_validation_signals(
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    session_avg_volume: float,
    current_bar: int = 0,
    zone_entry_bar: int = 0,
) -> Dict[str, float]:
    """
    Convenience function to calculate all validation signals.

    This is a module-level function that creates a ValidationSignals instance
    and computes all signals in one call.

    Args:
        open_price: Bar's open price
        high: Bar's high price
        low: Bar's low price
        close: Bar's close price
        volume: Bar's volume
        session_avg_volume: Average volume for the session
        current_bar: Current bar index (for approach_bars tracking)
        zone_entry_bar: Bar index when current zone was entered

    Returns:
        Dictionary with all validation signals:
        - rejection_wick_ratio: [-1, 1] (positive = bullish rejection)
        - volume_vs_session_avg: [0.1, 5.0] (> 1.0 = above average)
        - approach_bars: >= 0 or -1 if not tracked

    Example:
        >>> signals = calculate_validation_signals(
        ...     open_price=100.0, high=105.0, low=95.0, close=102.0,
        ...     volume=1500.0, session_avg_volume=1000.0,
        ...     current_bar=50, zone_entry_bar=45
        ... )
        >>> signals['rejection_wick_ratio']  # Bullish if positive
        0.4
        >>> signals['volume_vs_session_avg']  # 1.5x average volume
        1.5
        >>> signals['approach_bars']  # 5 bars in current zone
        5
    """
    validator = ValidationSignals()
    return validator.calculate(
        open_price=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        session_avg_volume=session_avg_volume,
        current_bar=current_bar,
        zone_entry_bar=zone_entry_bar,
    )


# =============================================================================
# BOX FEATURE EXTRACTOR
# =============================================================================

# Feature names for observation space
BOX_FEATURE_NAMES = [
    # Box dimensions (2)
    "box_height_atr_ratio",
    "session_progress",
    # Distance features (3)
    "dist_to_high_norm",
    "dist_to_mid_norm",
    "dist_to_low_norm",
    # Zone position (4)
    "zone_position",  # Continuous [-1, 1]
    "at_high",  # Binary
    "at_mid",  # Binary
    "at_low",  # Binary
    # Touch counts (3)
    "high_touch_norm",
    "mid_touch_norm",
    "low_touch_norm",
    # Touch recency (2)
    "bars_since_high_touch_norm",
    "bars_since_low_touch_norm",
    # Breakout state (3)
    "broke_high",
    "broke_low",
    "breakout_direction",
    # Validation signals (3)
    "rejection_wick_ratio",
    "volume_vs_session_avg_norm",
    "approach_bars_norm",
]


class BoxFeatureExtractor:
    """
    Extract box strategy features for RL observation space.

    This class combines BoxState tracking with feature extraction, producing
    a fixed-size feature vector suitable for neural network input.

    Features are normalized to bounded ranges:
    - Binary features: {0, 1}
    - Ratio features: [0, 1] or [-1, 1]
    - All features numerically stable (eps-safe divisions)

    Composition Pattern:
        This class does NOT inherit from FeatureExtractor. It's designed to
        be composed with the existing FeatureExtractor, appending box features
        to the existing feature set.

    Attributes:
        box_state: BoxState instance tracking session state
        validation_signals: ValidationSignals for candle analysis
        warmup_bars: Bars needed before features are valid
        touch_tolerance_pct: Tolerance for "at level" detection
        expected_session_bars: Expected bars per session (for progress normalization)

    Usage:
        extractor = BoxFeatureExtractor(warmup_bars=30)
        extractor.reset_session(open=100, high=105, low=95)

        for bar in bars:
            extractor.update(bar)
            if extractor.is_warm:
                features = extractor.extract_features(bar, atr=1.5)
    """

    # Constants for normalization
    EPS: float = 1e-8
    MAX_HEIGHT_ATR_RATIO: float = 5.0
    MAX_APPROACH_BARS: int = 50
    VOLUME_RATIO_MIN: float = 0.1
    VOLUME_RATIO_MAX: float = 5.0

    def __init__(
        self,
        warmup_bars: int = 30,
        touch_tolerance_pct: float = 0.001,
        expected_session_bars: int = 96,  # 96 bars for 15m candles in 24h
    ):
        """
        Initialize BoxFeatureExtractor.

        Args:
            warmup_bars: Bars needed before features are valid
            touch_tolerance_pct: Tolerance for "at level" detection (0.1% default)
            expected_session_bars: Expected bars per session for progress normalization
        """
        self.warmup_bars = warmup_bars
        self.touch_tolerance_pct = touch_tolerance_pct
        self.expected_session_bars = expected_session_bars

        # State tracking
        self.box_state = BoxState()
        self.validation_signals = ValidationSignals()

        # Zone entry tracking for approach_bars
        self._zone_entry_bar: int = -1
        self._last_zone: Zone = Zone.LOWER

        # Session volume tracking for volume_vs_session_avg
        self._session_volume_sum: float = 0.0
        self._session_volume_count: int = 0

    @property
    def feature_names(self) -> list:
        """Get list of feature names."""
        return BOX_FEATURE_NAMES.copy()

    @property
    def feature_dim(self) -> int:
        """Get number of features."""
        return len(BOX_FEATURE_NAMES)

    @property
    def is_warm(self) -> bool:
        """Check if warmup period is complete."""
        return self.box_state.bar_count >= self.warmup_bars

    @property
    def session_avg_volume(self) -> float:
        """Get session average volume."""
        if self._session_volume_count == 0:
            return 0.0
        return self._session_volume_sum / self._session_volume_count

    def reset_session(
        self,
        first_open: float,
        first_high: float,
        first_low: float,
        first_volume: float = 0.0,
    ) -> None:
        """
        Reset state for new trading session.

        Called at session boundary (e.g., UTC midnight) with first bar data.

        Args:
            first_open: Open price of first bar
            first_high: High of first bar
            first_low: Low of first bar
            first_volume: Volume of first bar (optional)
        """
        self.box_state.reset(first_open, first_high, first_low)

        # Reset zone tracking
        self._zone_entry_bar = 1
        self._last_zone = Zone.LOWER

        # Reset volume tracking
        self._session_volume_sum = first_volume
        self._session_volume_count = 1 if first_volume > 0 else 0

    def update(
        self,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
    ) -> Zone:
        """
        Update state with new bar data.

        Args:
            high: Bar's high price
            low: Bar's low price
            close: Bar's close price
            volume: Bar's volume (optional)

        Returns:
            Current zone classification
        """
        # Update box state
        current_zone = self.box_state.update(
            high=high,
            low=low,
            close=close,
            tolerance_pct=self.touch_tolerance_pct,
        )

        # Track zone entry for approach_bars
        if current_zone != self._last_zone:
            self._zone_entry_bar = self.box_state.bar_count
            self._last_zone = current_zone

        # Update volume tracking
        if volume > 0:
            self._session_volume_sum += volume
            self._session_volume_count += 1

        return current_zone

    def extract_features(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        atr: float,
    ) -> Dict[str, float]:
        """
        Extract all box features for current bar.

        Args:
            open_price: Bar's open price
            high: Bar's high price
            low: Bar's low price
            close: Bar's close price
            volume: Bar's volume
            atr: Current ATR value (for box_height_atr_ratio)

        Returns:
            Dictionary mapping feature names to values
        """
        features = {}
        state = self.box_state

        # Handle invalid state (before first bar)
        if not state.is_valid:
            return {name: 0.0 for name in BOX_FEATURE_NAMES}

        box_height = state.box_height
        box_mid = state.box_mid

        # ===== BOX DIMENSIONS (2) =====
        # box_height_atr_ratio: How wide is the box relative to volatility
        if atr > self.EPS:
            height_atr = box_height / atr
        else:
            height_atr = 0.0
        features["box_height_atr_ratio"] = float(np.clip(
            height_atr / self.MAX_HEIGHT_ATR_RATIO, 0.0, 1.0
        ))

        # session_progress: How far into the session [0, 1]
        features["session_progress"] = float(np.clip(
            state.bar_count / self.expected_session_bars, 0.0, 1.0
        ))

        # ===== DISTANCE FEATURES (3) =====
        # All distances normalized by box height, clipped to [-1, 1]
        if box_height > self.EPS:
            dist_to_high = (close - state.session_high) / box_height
            dist_to_mid = (close - box_mid) / box_height
            dist_to_low = (close - state.session_low) / box_height
        else:
            dist_to_high = 0.0
            dist_to_mid = 0.0
            dist_to_low = 0.0

        features["dist_to_high_norm"] = float(np.clip(dist_to_high, -1.0, 1.0))
        features["dist_to_mid_norm"] = float(np.clip(dist_to_mid, -1.0, 1.0))
        features["dist_to_low_norm"] = float(np.clip(dist_to_low, -1.0, 1.0))

        # ===== ZONE POSITION (4) =====
        # Continuous zone encoding: -1 (below_box) to +1 (above_box)
        zone = state.last_zone
        zone_map = {
            Zone.BELOW_BOX: -1.0,
            Zone.AT_LOW: -0.67,
            Zone.LOWER: -0.33,
            Zone.AT_MID: 0.0,
            Zone.UPPER: 0.33,
            Zone.AT_HIGH: 0.67,
            Zone.ABOVE_BOX: 1.0,
        }
        features["zone_position"] = zone_map.get(zone, 0.0)

        # Binary "at level" flags
        features["at_high"] = 1.0 if zone == Zone.AT_HIGH else 0.0
        features["at_mid"] = 1.0 if zone == Zone.AT_MID else 0.0
        features["at_low"] = 1.0 if zone == Zone.AT_LOW else 0.0

        # ===== TOUCH COUNTS (3) =====
        max_touches = state.MAX_TOUCH_COUNT
        features["high_touch_norm"] = state.high_touch_count / max_touches
        features["mid_touch_norm"] = state.mid_touch_count / max_touches
        features["low_touch_norm"] = state.low_touch_count / max_touches

        # ===== TOUCH RECENCY (2) =====
        # Normalize bars since touch: 0 = just touched, 1 = long ago
        max_recency = 50  # Bars for max recency
        bars_since_high = state.bars_since_high_touch()
        bars_since_low = state.bars_since_low_touch()

        if bars_since_high < 0:
            features["bars_since_high_touch_norm"] = 1.0  # Never touched
        else:
            features["bars_since_high_touch_norm"] = float(np.clip(
                bars_since_high / max_recency, 0.0, 1.0
            ))

        if bars_since_low < 0:
            features["bars_since_low_touch_norm"] = 1.0  # Never touched
        else:
            features["bars_since_low_touch_norm"] = float(np.clip(
                bars_since_low / max_recency, 0.0, 1.0
            ))

        # ===== BREAKOUT STATE (3) =====
        features["broke_high"] = 1.0 if state.broke_high else 0.0
        features["broke_low"] = 1.0 if state.broke_low else 0.0

        # Breakout direction: -1 (broke low only), 0 (neither/both), +1 (broke high only)
        if state.broke_high and not state.broke_low:
            features["breakout_direction"] = 1.0
        elif state.broke_low and not state.broke_high:
            features["breakout_direction"] = -1.0
        else:
            features["breakout_direction"] = 0.0

        # ===== VALIDATION SIGNALS (3) =====
        signals = self.validation_signals.calculate(
            open_price=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            session_avg_volume=self.session_avg_volume,
            current_bar=state.bar_count,
            zone_entry_bar=self._zone_entry_bar,
        )

        # rejection_wick_ratio: already in [-1, 1]
        features["rejection_wick_ratio"] = signals["rejection_wick_ratio"]

        # volume_vs_session_avg: rescale from [0.1, 5.0] to [0, 1]
        vol_ratio = signals["volume_vs_session_avg"]
        features["volume_vs_session_avg_norm"] = float(np.clip(
            (vol_ratio - self.VOLUME_RATIO_MIN) /
            (self.VOLUME_RATIO_MAX - self.VOLUME_RATIO_MIN),
            0.0, 1.0
        ))

        # approach_bars: normalize to [0, 1]
        approach = signals["approach_bars"]
        if approach < 0:
            features["approach_bars_norm"] = 0.0
        else:
            features["approach_bars_norm"] = float(np.clip(
                approach / self.MAX_APPROACH_BARS, 0.0, 1.0
            ))

        return features

    def extract_feature_array(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        atr: float,
    ) -> np.ndarray:
        """
        Extract features as numpy array (for direct use in observation space).

        Args:
            open_price: Bar's open price
            high: Bar's high price
            low: Bar's low price
            close: Bar's close price
            volume: Bar's volume
            atr: Current ATR value

        Returns:
            1D numpy array of shape (feature_dim,) with dtype float32
        """
        features = self.extract_features(open_price, high, low, close, volume, atr)
        return np.array(
            [features[name] for name in BOX_FEATURE_NAMES],
            dtype=np.float32
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize extractor state for checkpointing.

        Returns:
            Dictionary containing all state needed to restore
        """
        return {
            "box_state": self.box_state.to_dict(),
            "validation_signals": self.validation_signals.to_dict(),
            "zone_entry_bar": self._zone_entry_bar,
            "last_zone": self._last_zone.value,
            "session_volume_sum": self._session_volume_sum,
            "session_volume_count": self._session_volume_count,
            "warmup_bars": self.warmup_bars,
            "touch_tolerance_pct": self.touch_tolerance_pct,
            "expected_session_bars": self.expected_session_bars,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoxFeatureExtractor":
        """
        Restore extractor from checkpoint.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Restored BoxFeatureExtractor instance
        """
        extractor = cls(
            warmup_bars=data.get("warmup_bars", 30),
            touch_tolerance_pct=data.get("touch_tolerance_pct", 0.001),
            expected_session_bars=data.get("expected_session_bars", 96),
        )
        extractor.box_state = BoxState.from_dict(data.get("box_state", {}))
        extractor.validation_signals = ValidationSignals.from_dict(
            data.get("validation_signals", {})
        )
        extractor._zone_entry_bar = data.get("zone_entry_bar", -1)
        extractor._last_zone = Zone(data.get("last_zone", Zone.LOWER.value))
        extractor._session_volume_sum = data.get("session_volume_sum", 0.0)
        extractor._session_volume_count = data.get("session_volume_count", 0)
        return extractor

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return (
            f"BoxFeatureExtractor(warm={self.is_warm}, bars={self.box_state.bar_count}, "
            f"zone={self.box_state.last_zone.value}, features={self.feature_dim})"
        )
