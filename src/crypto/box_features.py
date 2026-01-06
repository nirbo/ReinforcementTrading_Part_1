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
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any

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
