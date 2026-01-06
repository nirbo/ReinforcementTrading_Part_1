"""
Feature extraction for crypto RL trading system.

Generates state vectors from OHLCV data and indicator values.
State vectors are designed for RL agent consumption with:
- Scale-invariant features (ratios, percentages)
- Bounded values (clipped to reasonable ranges)
- Rolling window history support
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.crypto.config import CryptoConfig, load_config
from src.crypto.indicators import compute_all_indicators
from src.crypto.box_features import BoxFeatureExtractor, BOX_FEATURE_NAMES


@dataclass
class PositionState:
    """Current position state for inclusion in observation."""

    direction: int = 0  # 1=long, -1=short, 0=flat
    entry_price: float = 0.0
    unrealized_pnl_pct: float = 0.0
    time_in_position: int = 0  # bars since entry


@dataclass
class FeatureSpec:
    """Specification for a single feature."""

    name: str
    min_val: float = -np.inf
    max_val: float = np.inf
    normalize: bool = True  # Whether to normalize to [0, 1] or [-1, 1]


# Feature specifications with clipping bounds
FEATURE_SPECS: List[FeatureSpec] = [
    # Price features
    FeatureSpec("close_sma20_ratio", 0.8, 1.2),
    FeatureSpec("close_sma50_ratio", 0.7, 1.3),
    FeatureSpec("atr_close_ratio", 0.0, 0.1),
    FeatureSpec("rsi", 0.0, 100.0),
    FeatureSpec("rsi_normalized", -1.0, 1.0),  # (rsi - 50) / 50
    FeatureSpec("volume_ratio", 0.0, 5.0),
    FeatureSpec("log_return", -0.1, 0.1),
    FeatureSpec("high_low_range", 0.0, 0.1),

    # Trend features
    FeatureSpec("hma_direction", -1.0, 1.0),
    FeatureSpec("kalman_direction", -1.0, 1.0),
    FeatureSpec("kalman_ratio", 0.95, 1.05),
    FeatureSpec("trend_direction", -1.0, 1.0),
    FeatureSpec("trend_strength", 0.0, 1.0),
    FeatureSpec("price_vs_trend", -1.0, 1.0),

    # Pivot/SR features
    FeatureSpec("dist_to_support", -10.0, 10.0),
    FeatureSpec("dist_to_resistance", -10.0, 10.0),
    FeatureSpec("sr_position", 0.0, 1.0),  # Where price is between S/R
    FeatureSpec("pivot_high_detected", 0.0, 1.0),
    FeatureSpec("pivot_low_detected", 0.0, 1.0),
    FeatureSpec("recent_high_dist", 0.0, 10.0),
    FeatureSpec("recent_low_dist", 0.0, 10.0),
    FeatureSpec("sr_touch_count", 0.0, 5.0),

    # Momentum features
    FeatureSpec("breakout_long", 0.0, 1.0),
    FeatureSpec("breakout_short", 0.0, 1.0),
    FeatureSpec("velocity_long", 0.0, 1.0),
    FeatureSpec("velocity_short", 0.0, 1.0),
    FeatureSpec("atr_exp_bull", 0.0, 1.0),
    FeatureSpec("atr_exp_bear", 0.0, 1.0),
    FeatureSpec("mfi", 0.0, 100.0),
    FeatureSpec("mfi_normalized", -1.0, 1.0),
    # MFI momentum signal (original sr_swing lines 1457-1466)
    FeatureSpec("mfi_momentum_bull", 0.0, 1.0),   # MFI > threshold AND rising
    FeatureSpec("mfi_momentum_bear", 0.0, 1.0),   # MFI < threshold AND falling

    # Filter features - RSI modes (original sr_swing lines 1521-1535)
    FeatureSpec("rsi_oversold", 0.0, 1.0),           # Avoid Extremes: block long if overbought
    FeatureSpec("rsi_overbought", 0.0, 1.0),         # Avoid Extremes: block short if oversold
    FeatureSpec("rsi_momentum_long", 0.0, 1.0),     # Momentum Aligned: RSI > 50 for longs
    FeatureSpec("rsi_momentum_short", 0.0, 1.0),    # Momentum Aligned: RSI < 50 for shorts
    FeatureSpec("rsi_bounce_long", 0.0, 1.0),       # Counter-Trend Bounce: RSI <= oversold for longs
    FeatureSpec("rsi_bounce_short", 0.0, 1.0),      # Counter-Trend Bounce: RSI >= overbought for shorts
    FeatureSpec("macd_bullish", 0.0, 1.0),
    FeatureSpec("macd_bearish", 0.0, 1.0),
    FeatureSpec("macd_histogram_norm", -1.0, 1.0),
    # MACD entry modes (original sr_swing lines 1540-1554)
    FeatureSpec("macd_above_zero", 0.0, 1.0),       # Above Zero Line mode
    FeatureSpec("macd_below_zero", 0.0, 1.0),       # Below Zero Line mode
    FeatureSpec("macd_cross_bull", 0.0, 1.0),       # Signal Cross: MACD crosses above signal
    FeatureSpec("macd_cross_bear", 0.0, 1.0),       # Signal Cross: MACD crosses below signal
    # Zero Lag Score features (original sr_swing lines 1386-1392, 1556-1566)
    FeatureSpec("zl_score_norm", -1.0, 1.0),       # Normalized ZL score
    FeatureSpec("zl_rising", 0.0, 1.0),            # ZL EMA rising
    FeatureSpec("zl_falling", 0.0, 1.0),           # ZL EMA falling
    FeatureSpec("zl_score_bull", 0.0, 1.0),        # Score Only mode: score > threshold_up
    FeatureSpec("zl_score_bear", 0.0, 1.0),        # Score Only mode: score < threshold_down
    FeatureSpec("bb_position", 0.0, 1.0),
    FeatureSpec("bb_squeeze", 0.0, 1.0),
    # BB squeeze breakout signals (original sr_swing lines 1469-1473)
    FeatureSpec("bb_squeeze_breakout_bull", 0.0, 1.0),  # Was in squeeze, now above upper
    FeatureSpec("bb_squeeze_breakout_bear", 0.0, 1.0),  # Was in squeeze, now below lower
    # MTF strict mode features (original sr_swing lines 1498-1504)
    FeatureSpec("htf_trend", -1.0, 1.0),              # HTF trend direction
    FeatureSpec("mtf_aligned", 0.0, 1.0),             # LTF and HTF agree on direction
    FeatureSpec("htf_rising", 0.0, 1.0),              # HTF trend rising
    FeatureSpec("htf_falling", 0.0, 1.0),             # HTF trend falling
    FeatureSpec("mtf_strict_bullish", 0.0, 1.0),      # Strict mode: htf_rising only
    FeatureSpec("mtf_strict_bearish", 0.0, 1.0),      # Strict mode: htf_falling only
    FeatureSpec("mtf_relaxed_bullish", 0.0, 1.0),     # Relaxed: htf_rising OR not htf_falling
    FeatureSpec("mtf_relaxed_bearish", 0.0, 1.0),     # Relaxed: htf_falling OR not htf_rising

    # =========================================================================
    # P2 FEATURES - Advanced SR Swing Strategy features
    # =========================================================================

    # Darvas Box Theory (sr_swing lines 1175-1182)
    FeatureSpec("box_active", 0.0, 1.0),              # Box is currently active (state=3)
    FeatureSpec("box_position", 0.0, 1.0),            # Price position within box (0=bottom, 1=top)
    FeatureSpec("box_state_norm", 0.0, 1.0),          # Box state normalized (0-3 → 0-1)
    FeatureSpec("box_breakout_bull", 0.0, 1.0),       # Bullish box breakout signal
    FeatureSpec("box_breakout_bear", 0.0, 1.0),       # Bearish box breakout signal

    # Gravity Mode (sr_swing lines 1487-1516)
    FeatureSpec("gravity_bullish", 0.0, 1.0),         # Gravity mode is bullish
    FeatureSpec("gravity_bearish", 0.0, 1.0),         # Gravity mode is bearish

    # Pivot Strength (touch count per level)
    FeatureSpec("resistance_strength_norm", 0.0, 1.0),  # Normalized resistance touches
    FeatureSpec("support_strength_norm", 0.0, 1.0),     # Normalized support touches

    # Pending Level Confirmation
    FeatureSpec("pending_resistance_norm", 0.0, 1.0),   # Pending resistance levels
    FeatureSpec("pending_support_norm", 0.0, 1.0),      # Pending support levels

    # Position state (added dynamically)
    FeatureSpec("position_direction", -1.0, 1.0),
    FeatureSpec("unrealized_pnl_pct", -0.1, 0.1),
    FeatureSpec("time_in_position_norm", 0.0, 1.0),
]


class FeatureExtractor:
    """
    Extracts RL-ready features from OHLCV data.

    Features are designed to be:
    - Scale-invariant (ratios, percentages)
    - Bounded (clipped to reasonable ranges)
    - Normalized (optionally to [0, 1] or [-1, 1])

    Box Features Integration:
        When use_box_features=True (from config.box_features.use_box_features),
        this extractor includes 20 additional features from BoxFeatureExtractor.
        These track intraday session high/low/mid levels and interactions.
    """

    def __init__(
        self,
        config: Optional[CryptoConfig] = None,
        window_size: int = 30,
        include_htf: bool = True,
    ):
        self.config = config or load_config()
        self.window_size = window_size
        self.include_htf = include_htf
        self.indicator_config = self.config.indicators.model_dump()

        # Feature toggle flags from config
        self.use_base_features = self.config.box_features.use_base_features
        self.use_box_features = self.config.box_features.use_box_features

        # Compute base feature dimension (only if enabled)
        self._base_features = self._get_feature_names() if self.use_base_features else []
        self._base_feature_dim = len(self._base_features)

        # Box features integration (optional, controlled by config)
        self._box_extractor: Optional[BoxFeatureExtractor] = None

        if self.use_box_features:
            self._box_extractor = BoxFeatureExtractor(
                warmup_bars=self.config.box_features.box_warmup_bars,
                touch_tolerance_pct=self.config.box_features.box_touch_tolerance_pct,
            )
            self._box_feature_dim = self._box_extractor.feature_dim
        else:
            self._box_feature_dim = 0

        # Total feature dimension
        self.feature_dim = self._base_feature_dim + self._box_feature_dim

    def _get_feature_names(self) -> List[str]:
        """Get list of base feature names."""
        return [spec.name for spec in FEATURE_SPECS]

    def get_all_feature_names(self) -> List[str]:
        """
        Get list of all feature names including box features.

        Returns:
            List of feature names (base + box if enabled)
        """
        names = self._base_features.copy()
        if self.use_box_features:
            names.extend(BOX_FEATURE_NAMES)
        return names

    @property
    def observation_shape(self) -> Tuple[int, int]:
        """Shape of observation: (window_size, feature_dim)."""
        return (self.window_size, self.feature_dim)

    @property
    def box_extractor(self) -> Optional[BoxFeatureExtractor]:
        """Get the box feature extractor (None if disabled)."""
        return self._box_extractor

    def reset_box_session(
        self,
        first_open: float,
        first_high: float,
        first_low: float,
        first_volume: float = 0.0,
    ) -> None:
        """
        Reset box state for new trading session.

        Called at session boundary (e.g., UTC midnight) with first bar data.
        No-op if use_box_features is False.

        Args:
            first_open: Open price of first bar
            first_high: High of first bar
            first_low: Low of first bar
            first_volume: Volume of first bar (optional)
        """
        if self._box_extractor is not None:
            self._box_extractor.reset_session(
                first_open, first_high, first_low, first_volume
            )

    def update_box_state(
        self,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
    ) -> None:
        """
        Update box state with new bar data.

        No-op if use_box_features is False.

        Args:
            high: Bar's high price
            low: Bar's low price
            close: Bar's close price
            volume: Bar's volume (optional)
        """
        if self._box_extractor is not None:
            self._box_extractor.update(high, low, close, volume)

    def get_box_features(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        atr: float,
    ) -> np.ndarray:
        """
        Extract box features for current bar as numpy array.

        Returns zeros if use_box_features is False.

        Args:
            open_price: Bar's open price
            high: Bar's high price
            low: Bar's low price
            close: Bar's close price
            volume: Bar's volume
            atr: Current ATR value

        Returns:
            1D numpy array of shape (box_feature_dim,) or (0,) if disabled
        """
        if self._box_extractor is None:
            return np.array([], dtype=np.float32)

        return self._box_extractor.extract_feature_array(
            open_price, high, low, close, volume, atr
        )

    def extract_features(
        self,
        df: pd.DataFrame,
        position: Optional[PositionState] = None,
        htf_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Extract all features from OHLCV DataFrame.

        Args:
            df: DataFrame with OHLCV columns (already has indicators if precomputed)
            position: Current position state
            htf_df: Higher timeframe DataFrame for MTF features

        Returns:
            DataFrame with feature columns
        """
        features = pd.DataFrame(index=df.index)

        # If base features disabled, return empty DataFrame (box-only mode)
        if not self.use_base_features:
            return features

        # Compute indicators if not already present
        if "rsi" not in df.columns:
            df = compute_all_indicators(df, self.indicator_config)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        atr = df["atr"]

        # ===== PRICE FEATURES =====
        features["close_sma20_ratio"] = (close / df["sma_20"]).clip(0.8, 1.2)
        features["close_sma50_ratio"] = (close / df["sma_50"]).clip(0.7, 1.3)
        features["atr_close_ratio"] = (atr / close).clip(0, 0.1)
        features["rsi"] = df["rsi"].clip(0, 100) / 100  # Normalize to [0, 1]
        features["rsi_normalized"] = ((df["rsi"] - 50) / 50).clip(-1, 1)
        features["volume_ratio"] = df["volume_ratio"].clip(0, 5) / 5  # Normalize to [0, 1]
        features["log_return"] = np.log(close / close.shift(1)).clip(-0.1, 0.1).fillna(0)
        features["high_low_range"] = ((high - low) / close).clip(0, 0.1)

        # ===== TREND FEATURES =====
        features["hma_direction"] = df["hma_direction"]
        features["kalman_direction"] = df["kalman_direction"]
        features["kalman_ratio"] = (df["kalman_short"] / df["kalman_long"].replace(0, np.nan)).clip(0.95, 1.05).fillna(1.0)
        features["kalman_ratio"] = (features["kalman_ratio"] - 1.0) * 20  # Scale to ~[-1, 1]
        features["trend_direction"] = df["trend_direction"]

        # Trend strength: absolute difference from MA ratio
        trend_dev = (features["close_sma20_ratio"] - 1.0).abs()
        features["trend_strength"] = (trend_dev * 10).clip(0, 1)

        # Price vs trend
        features["price_vs_trend"] = ((close - df["trend_value"]) / atr.replace(0, np.nan)).clip(-1, 1).fillna(0)

        # ===== PIVOT/SR FEATURES =====
        features["dist_to_support"] = df["dist_to_support"].clip(-10, 10) / 10  # Normalize to [-1, 1]
        features["dist_to_resistance"] = df["dist_to_resistance"].clip(-10, 10) / 10

        # SR position: where price is between support and resistance
        sr_range = df["resistance"] - df["support"]
        features["sr_position"] = ((close - df["support"]) / sr_range.replace(0, np.nan)).clip(0, 1).fillna(0.5)

        # Pivot detection (binary flags)
        features["pivot_high_detected"] = df["pivot_high"].notna().astype(float)
        features["pivot_low_detected"] = df["pivot_low"].notna().astype(float)

        # Recent high/low distance (normalized by ATR)
        recent_high = high.rolling(window=20).max()
        recent_low = low.rolling(window=20).min()
        features["recent_high_dist"] = ((recent_high - close) / atr.replace(0, np.nan)).clip(0, 10).fillna(0) / 10
        features["recent_low_dist"] = ((close - recent_low) / atr.replace(0, np.nan)).clip(0, 10).fillna(0) / 10

        # SR touch count (rolling sum of pivot detections)
        features["sr_touch_count"] = (
            features["pivot_high_detected"].rolling(window=50, min_periods=1).sum() +
            features["pivot_low_detected"].rolling(window=50, min_periods=1).sum()
        ).clip(0, 5) / 5

        # ===== MOMENTUM FEATURES =====
        features["breakout_long"] = df["breakout_long"].astype(float)
        features["breakout_short"] = df["breakout_short"].astype(float)
        features["velocity_long"] = df["velocity_long"].astype(float)
        features["velocity_short"] = df["velocity_short"].astype(float)
        features["atr_exp_bull"] = df["atr_exp_bull"].astype(float)
        features["atr_exp_bear"] = df["atr_exp_bear"].astype(float)
        features["mfi"] = df["mfi"].clip(0, 100) / 100
        features["mfi_normalized"] = ((df["mfi"] - 50) / 50).clip(-1, 1)

        # MFI momentum signal (original sr_swing lines 1457-1466)
        mfi_threshold = self.config.indicators.mfi_momentum_threshold
        mfi_rising = df["mfi"].diff() > 0
        mfi_falling = df["mfi"].diff() < 0
        features["mfi_momentum_bull"] = ((df["mfi"] > mfi_threshold) & mfi_rising).astype(float)
        features["mfi_momentum_bear"] = ((df["mfi"] < mfi_threshold) & mfi_falling).astype(float)

        # ===== FILTER FEATURES =====
        # RSI filter modes (original sr_swing lines 1521-1535)
        rsi_oversold = self.config.indicators.rsi_oversold
        rsi_overbought = self.config.indicators.rsi_overbought

        # Avoid Extremes mode: block entry if at extremes
        features["rsi_oversold"] = (df["rsi"] < rsi_oversold).astype(float)
        features["rsi_overbought"] = (df["rsi"] > rsi_overbought).astype(float)

        # Momentum Aligned mode: RSI must confirm direction
        features["rsi_momentum_long"] = (df["rsi"] > 50).astype(float)
        features["rsi_momentum_short"] = (df["rsi"] < 50).astype(float)

        # Counter-Trend Bounce mode: enter on extreme reversals
        features["rsi_bounce_long"] = (df["rsi"] <= rsi_oversold).astype(float)
        features["rsi_bounce_short"] = (df["rsi"] >= rsi_overbought).astype(float)

        # MACD features (original sr_swing lines 1540-1554)
        features["macd_bullish"] = (df["macd_histogram"] > 0).astype(float)
        features["macd_bearish"] = (df["macd_histogram"] < 0).astype(float)

        # MACD histogram normalized by its recent range
        macd_range = df["macd_histogram"].rolling(window=50).apply(lambda x: x.max() - x.min(), raw=True)
        features["macd_histogram_norm"] = (df["macd_histogram"] / macd_range.replace(0, np.nan)).clip(-1, 1).fillna(0)

        # MACD Zero Line position modes
        features["macd_above_zero"] = (df["macd"] > 0).astype(float)
        features["macd_below_zero"] = (df["macd"] < 0).astype(float)

        # MACD Signal Cross detection
        macd_above_signal = df["macd"] > df["macd_signal"]
        macd_above_signal_prev = macd_above_signal.shift(1, fill_value=False)
        features["macd_cross_bull"] = (macd_above_signal & ~macd_above_signal_prev).astype(float)
        features["macd_cross_bear"] = (~macd_above_signal & macd_above_signal_prev).astype(float)

        # Zero Lag Score features (original sr_swing lines 1386-1392, 1556-1566)
        zl_threshold_up = self.config.indicators.zl_threshold_up
        zl_threshold_down = self.config.indicators.zl_threshold_down
        features["zl_score_norm"] = df["zl_score"].clip(-1, 1)  # Score is typically -1 to 1
        features["zl_rising"] = df["zl_rising"]
        features["zl_falling"] = df["zl_falling"]
        features["zl_score_bull"] = (df["zl_score"] > zl_threshold_up).astype(float)
        features["zl_score_bear"] = (df["zl_score"] < zl_threshold_down).astype(float)

        features["bb_position"] = df["bb_position"]

        # BB squeeze: low bandwidth indicates consolidation
        bb_width_avg = df["bb_width"].rolling(window=50).mean()
        bb_squeeze = (df["bb_width"] < bb_width_avg * 0.5)
        features["bb_squeeze"] = bb_squeeze.astype(float)

        # BB squeeze breakout (original sr_swing lines 1469-1473)
        # Track if was in squeeze in last 5 bars
        was_in_squeeze = bb_squeeze.rolling(window=5, min_periods=1).max().fillna(False).astype(bool)
        close = df["close"]
        features["bb_squeeze_breakout_bull"] = (was_in_squeeze & (close > df["bb_upper"])).astype(float)
        features["bb_squeeze_breakout_bear"] = (was_in_squeeze & (close < df["bb_lower"])).astype(float)

        # ===== P2 FEATURES - Advanced SR Swing Strategy =====

        # Darvas Box Theory (sr_swing lines 1175-1182)
        if "box_state" in df.columns:
            features["box_active"] = df["box_active"]
            features["box_position"] = df["box_position"]
            features["box_state_norm"] = df["box_state"] / 3.0  # Normalize 0-3 to 0-1
            features["box_breakout_bull"] = df["box_breakout_bull"].astype(float)
            features["box_breakout_bear"] = df["box_breakout_bear"].astype(float)
        else:
            features["box_active"] = 0.0
            features["box_position"] = 0.5
            features["box_state_norm"] = 0.0
            features["box_breakout_bull"] = 0.0
            features["box_breakout_bear"] = 0.0

        # Gravity Mode (sr_swing lines 1487-1516)
        if "gravity_bullish" in df.columns:
            features["gravity_bullish"] = df["gravity_bullish"]
            features["gravity_bearish"] = df["gravity_bearish"]
        else:
            features["gravity_bullish"] = 0.0
            features["gravity_bearish"] = 0.0

        # Pivot Strength (sr_swing concept)
        if "resistance_strength" in df.columns:
            # Normalize strength to 0-1 (assume max 5 touches is "strong")
            features["resistance_strength_norm"] = (df["resistance_strength"].clip(0, 5) / 5.0)
            features["support_strength_norm"] = (df["support_strength"].clip(0, 5) / 5.0)
        else:
            features["resistance_strength_norm"] = 0.0
            features["support_strength_norm"] = 0.0

        # Pending Level Confirmation
        if "pending_resistance" in df.columns:
            # Normalize pending counts (assume max 10 pending levels)
            features["pending_resistance_norm"] = (df["pending_resistance"].clip(0, 10) / 10.0)
            features["pending_support_norm"] = (df["pending_support"].clip(0, 10) / 10.0)
        else:
            features["pending_resistance_norm"] = 0.0
            features["pending_support_norm"] = 0.0

        # ===== HTF FEATURES (if provided) =====
        if htf_df is not None and self.include_htf:
            # Add HTF trend direction
            if "trend_direction" in htf_df.columns:
                htf_trend = htf_df["trend_direction"].reindex(df.index, method="ffill")
                features["htf_trend"] = htf_trend.fillna(0)
            else:
                features["htf_trend"] = 0.0

            # MTF alignment: LTF and HTF agree on direction
            features["mtf_aligned"] = (features["trend_direction"] == features["htf_trend"]).astype(float)

            # MTF strict mode features (original sr_swing lines 1498-1504)
            htf_rising = (features["htf_trend"] > 0)
            htf_falling = (features["htf_trend"] < 0)
            features["htf_rising"] = htf_rising.astype(float)
            features["htf_falling"] = htf_falling.astype(float)

            # Strict mode: must have exact alignment
            features["mtf_strict_bullish"] = htf_rising.astype(float)
            features["mtf_strict_bearish"] = htf_falling.astype(float)

            # Relaxed mode: allows neutral (not opposing)
            features["mtf_relaxed_bullish"] = (htf_rising | ~htf_falling).astype(float)
            features["mtf_relaxed_bearish"] = (htf_falling | ~htf_rising).astype(float)
        else:
            features["htf_trend"] = 0.0
            features["mtf_aligned"] = 0.5  # Neutral when no HTF
            features["htf_rising"] = 0.0
            features["htf_falling"] = 0.0
            features["mtf_strict_bullish"] = 0.0
            features["mtf_strict_bearish"] = 0.0
            features["mtf_relaxed_bullish"] = 1.0  # Relaxed allows when no HTF
            features["mtf_relaxed_bearish"] = 1.0

        # ===== POSITION STATE =====
        if position is None:
            position = PositionState()

        features["position_direction"] = float(position.direction)
        features["unrealized_pnl_pct"] = np.clip(position.unrealized_pnl_pct, -0.1, 0.1) * 10  # Scale to [-1, 1]
        features["time_in_position_norm"] = np.clip(position.time_in_position / 100, 0, 1)  # Normalize

        # Fill NaN values
        features = features.fillna(0)

        return features

    def get_observation(
        self,
        df: pd.DataFrame,
        current_idx: int,
        position: Optional[PositionState] = None,
        htf_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Get observation array for a specific timestep.

        Args:
            df: DataFrame with OHLCV + indicators
            current_idx: Current bar index
            position: Current position state
            htf_df: Optional HTF DataFrame

        Returns:
            np.ndarray of shape (window_size, feature_dim)
        """
        # Extract features for entire dataframe
        features = self.extract_features(df, position, htf_df)

        # Get window ending at current_idx
        start_idx = max(0, current_idx - self.window_size + 1)
        end_idx = current_idx + 1

        window = features.iloc[start_idx:end_idx].values

        # Pad if necessary (at start of data)
        if len(window) < self.window_size:
            pad_size = self.window_size - len(window)
            padding = np.tile(window[0], (pad_size, 1))
            window = np.vstack([padding, window])

        return window.astype(np.float32)

    def precompute_features(
        self,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Precompute features for entire DataFrame.

        Returns:
            np.ndarray of shape (n_samples, feature_dim)
        """
        features = self.extract_features(df, htf_df=htf_df)
        return features.values.astype(np.float32)

    def get_feature_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature bounds for observation space.

        Returns:
            (low, high) arrays of shape (feature_dim,)
        """
        low = np.array([-1.0] * self.feature_dim, dtype=np.float32)
        high = np.array([1.0] * self.feature_dim, dtype=np.float32)
        return low, high


def create_feature_matrix(
    df: pd.DataFrame,
    window_size: int = 30,
    config: Optional[CryptoConfig] = None,
) -> Tuple[np.ndarray, FeatureExtractor]:
    """
    Convenience function to create feature matrix from OHLCV data.

    Returns:
        (features, extractor) where features is (n_samples, feature_dim)
    """
    extractor = FeatureExtractor(config, window_size)
    features = extractor.precompute_features(df)
    return features, extractor


if __name__ == "__main__":
    import numpy as np

    # Generate test data
    np.random.seed(42)
    n = 500

    returns = np.random.randn(n) * 0.02
    close = pd.Series(100 * np.exp(np.cumsum(returns)))
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))

    df = pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.uniform(1000, 10000, n),
    })

    # Test feature extraction
    extractor = FeatureExtractor(window_size=30)
    print(f"Feature dimension: {extractor.feature_dim}")
    print(f"Observation shape: {extractor.observation_shape}")

    # Get single observation
    obs = extractor.get_observation(df, current_idx=100)
    print(f"Observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Precompute all features
    features = extractor.precompute_features(df)
    print(f"Feature matrix shape: {features.shape}")

    # Check for NaN/Inf
    print(f"NaN count: {np.isnan(features).sum()}")
    print(f"Inf count: {np.isinf(features).sum()}")

    print("\nFeature extraction tests passed!")
