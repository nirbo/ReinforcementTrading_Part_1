"""
Technical indicators for crypto RL trading system.

Ported from sr_swing_strategy.py with vectorized implementations for efficiency.
All functions accept pandas Series/DataFrames and return numpy arrays or pandas Series.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# BASIC MOVING AVERAGES
# =============================================================================

def sma(data: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return data.rolling(window=length, min_periods=1).mean()


def ema(data: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average."""
    return data.ewm(span=length, adjust=False, min_periods=1).mean()


def wma(data: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, length + 1, dtype=float)
    return data.rolling(window=length, min_periods=1).apply(
        lambda x: np.dot(x[-len(weights):], weights[-len(x):]) / weights[-len(x):].sum(),
        raw=True
    )


def rma(data: pd.Series, length: int) -> pd.Series:
    """Wilder's Smoothed Moving Average (RMA)."""
    alpha = 1.0 / length
    return data.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


# =============================================================================
# ADVANCED MOVING AVERAGES
# =============================================================================

def hma(data: pd.Series, length: int) -> pd.Series:
    """
    Hull Moving Average.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    half_length = max(1, length // 2)
    sqrt_length = max(1, int(math.sqrt(length)))

    wma_half = wma(data, half_length)
    wma_full = wma(data, length)
    raw_hma = 2 * wma_half - wma_full

    return wma(raw_hma, sqrt_length)


def alma(data: pd.Series, length: int, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """
    Arnaud Legoux Moving Average.
    Uses Gaussian weighting centered at offset * (length - 1).
    """
    m = offset * (length - 1)
    s = length / sigma

    weights = np.array([math.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(length)])
    weights = weights / weights.sum()

    return data.rolling(window=length, min_periods=1).apply(
        lambda x: np.dot(x[-len(weights):], weights[-len(x):]) if len(x) >= length else x.mean(),
        raw=True
    )


def zero_lag_ema(data: pd.Series, length: int) -> pd.Series:
    """Zero Lag EMA - compensates for lag by adjusting source."""
    lag = (length - 1) // 2
    adjusted = data + (data - data.shift(lag).fillna(data))
    return ema(adjusted, length)


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return rma(tr, length)


def stdev(data: pd.Series, length: int) -> pd.Series:
    """Standard Deviation."""
    return data.rolling(window=length, min_periods=1).std()


def bollinger_bands(
    data: pd.Series,
    length: int = 20,
    mult: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    Returns: (upper, basis, lower, bandwidth)
    """
    basis = sma(data, length)
    dev = stdev(data, length) * mult
    upper = basis + dev
    lower = basis - dev
    bandwidth = (upper - lower) / basis.replace(0, np.nan)

    return upper, basis, lower, bandwidth.fillna(0)


# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================

def rsi(data: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = data.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    avg_gain = rma(gains, length)
    avg_loss = rma(losses, length)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(50)


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Money Flow Index."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    tp_delta = typical_price.diff()

    positive_flow = (raw_money_flow * (tp_delta > 0)).rolling(window=length).sum()
    negative_flow = (raw_money_flow * (tp_delta < 0)).rolling(window=length).sum()

    money_ratio = positive_flow / negative_flow.replace(0, np.nan)
    return 100 - (100 / (1 + money_ratio)).fillna(50)


def macd(
    data: pd.Series,
    fast_len: int = 12,
    slow_len: int = 26,
    signal_len: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence).
    Returns: (macd_line, signal_line, histogram)
    """
    fast_ema = ema(data, fast_len)
    slow_ema = ema(data, slow_len)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_len)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def zero_lag_score(data: pd.Series, length: int, loop_start: int = 1, loop_end: int = 70) -> pd.Series:
    """
    Zero Lag momentum score.
    Compares current zero-lag EMA to historical values.
    """
    zl = zero_lag_ema(data, length)

    def calc_score(idx: int) -> float:
        if idx < loop_end:
            return 0.0
        score = 0.0
        current = zl.iloc[idx]
        for i in range(loop_start - 1, min(loop_end, idx)):
            if current > zl.iloc[idx - i - 1]:
                score += 1
            else:
                score -= 1
        return score

    scores = [calc_score(i) for i in range(len(data))]
    return pd.Series(scores, index=data.index)


# =============================================================================
# ADVANCED FILTERS
# =============================================================================

class KalmanFilter:
    """Kalman Filter for trend detection."""

    def __init__(self, length: int, r: float = 0.01, q: float = 0.1):
        self.length = length
        self.r = r
        self.q = q
        self.estimate: Optional[float] = None
        self.error_est = 1.0
        self.error_meas = r * length

    def update(self, value: float) -> float:
        if self.estimate is None:
            self.estimate = value
            return value

        kalman_gain = self.error_est / (self.error_est + self.error_meas)
        self.estimate = self.estimate + kalman_gain * (value - self.estimate)
        self.error_est = (1 - kalman_gain) * self.error_est + self.q / self.length

        return self.estimate

    def reset(self):
        self.estimate = None
        self.error_est = 1.0


def kalman_filter(data: pd.Series, length: int) -> pd.Series:
    """Apply Kalman filter to series."""
    kf = KalmanFilter(length)
    filtered = [kf.update(v) for v in data.values]
    return pd.Series(filtered, index=data.index)


def kalman_trend(
    data: pd.Series,
    short_len: int = 18,
    long_len: int = 23,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Kalman trend detection.
    Returns: (short_kalman, long_kalman, direction)
    where direction is 1 (bullish), -1 (bearish), or 0 (neutral)
    """
    short_kf = KalmanFilter(short_len)
    long_kf = KalmanFilter(long_len)

    short_vals = []
    long_vals = []

    for v in data.values:
        short_vals.append(short_kf.update(v))
        long_vals.append(long_kf.update(v))

    short = pd.Series(short_vals, index=data.index)
    long = pd.Series(long_vals, index=data.index)
    direction = np.sign(short - long)

    return short, long, pd.Series(direction, index=data.index)


def gaussian_filter(data: pd.Series, length: int, poles: int = 2) -> pd.Series:
    """
    Gaussian Filter (Ehlers 2-pole approximation).
    """
    beta = (1 - math.cos(2 * math.pi / length)) / (math.pow(1.414, 2 / poles) - 1)
    alpha = -beta + math.sqrt(beta ** 2 + 2 * beta)

    c0 = alpha ** poles

    # Initialize output
    result = data.copy()

    # Apply recursive filter (simplified for vectorization)
    for i in range(poles, len(data)):
        weighted_sum = c0 * data.iloc[i]
        for j in range(1, min(poles + 1, i + 1)):
            coef = ((-1) ** j) * math.comb(poles, j) * ((1 - alpha) ** j)
            weighted_sum += coef * result.iloc[i - j]
        result.iloc[i] = weighted_sum

    return result


def rational_quadratic_kernel(
    data: pd.Series,
    lookback: int,
    relative_weight: float = 1.0,
) -> pd.Series:
    """
    Rational Quadratic Kernel (Nadaraya-Watson estimator).
    """
    def rq_at(idx: int) -> float:
        if idx < lookback:
            return data.iloc[idx]

        yhat = 0.0
        sum_weights = 0.0

        for i in range(lookback):
            y = data.iloc[idx - i]
            w = (1 + (i ** 2) / ((lookback ** 2) * 2 * relative_weight)) ** (-relative_weight)
            yhat += y * w
            sum_weights += w

        return yhat / sum_weights if sum_weights > 0 else data.iloc[idx]

    result = [rq_at(i) for i in range(len(data))]
    return pd.Series(result, index=data.index)


# =============================================================================
# TREND DETECTION
# =============================================================================

def calculate_trend(
    data: pd.Series,
    method: str = "HMA",
    length: int = 14,
    sigma: float = 6.0,
    poles: int = 2,
    offset: float = 0.85,
    kalman_short: int = 18,
    kalman_long: int = 23,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate trend value and direction using specified method.

    Args:
        data: Price series (typically close)
        method: One of "HMA", "Kalman", "Gaussian", "Kernel", "EMA", "SMA", "ALMA"
        length: Lookback period
        sigma: Sigma for ALMA/Gaussian
        poles: Poles for Gaussian
        offset: Offset for ALMA
        kalman_short: Short Kalman period
        kalman_long: Long Kalman period

    Returns:
        (trend_value, direction) where direction is 1/-1/0
    """
    if method == "Kalman":
        _, _, direction = kalman_trend(data, kalman_short, kalman_long)
        trend_value = kalman_filter(data, kalman_short)
    elif method == "Gaussian":
        trend_value = gaussian_filter(data, length, poles)
        direction = np.sign(trend_value.diff()).fillna(0)
    elif method == "Kernel":
        trend_value = rational_quadratic_kernel(data, length, sigma)
        direction = np.sign(trend_value.diff()).fillna(0)
    elif method == "EMA":
        trend_value = ema(data, length)
        direction = np.sign(trend_value.diff()).fillna(0)
    elif method == "SMA":
        trend_value = sma(data, length)
        direction = np.sign(trend_value.diff()).fillna(0)
    elif method == "ALMA":
        trend_value = alma(data, length, offset, sigma)
        direction = np.sign(trend_value.diff()).fillna(0)
    else:  # Default: HMA
        trend_value = hma(data, length)
        direction = np.sign(trend_value.diff()).fillna(0)

    return trend_value, pd.Series(direction, index=data.index)


# =============================================================================
# PIVOT DETECTION
# =============================================================================

def detect_pivot_high(
    high: pd.Series,
    close: pd.Series,
    left_bars: int = 4,
    right_bars: int = 3,
    use_high_low: bool = True,
) -> pd.Series:
    """
    Detect pivot highs (NO LOOK-AHEAD VERSION).

    IMPORTANT: Pivot at bar N requires N+right_bars to confirm.
    We store the pivot value at the CONFIRMATION bar (i), not the pivot bar.
    This ensures no future information leaks into the observation.

    Returns Series with pivot value at confirmation bar, NaN elsewhere.
    """
    source = high if use_high_low else close
    pivots = pd.Series(np.nan, index=source.index)

    for i in range(left_bars + right_bars, len(source)):
        pivot_idx = i - right_bars
        pivot_val = source.iloc[pivot_idx]

        # Check left bars
        is_pivot = True
        for j in range(pivot_idx - left_bars, pivot_idx):
            if source.iloc[j] >= pivot_val:
                is_pivot = False
                break

        # Check right bars
        if is_pivot:
            for j in range(pivot_idx + 1, pivot_idx + right_bars + 1):
                if j < len(source) and source.iloc[j] >= pivot_val:
                    is_pivot = False
                    break

        if is_pivot:
            # Store at confirmation bar (i), NOT pivot bar (pivot_idx)
            # This prevents look-ahead bias
            pivots.iloc[i] = pivot_val

    return pivots


def detect_pivot_low(
    low: pd.Series,
    close: pd.Series,
    left_bars: int = 4,
    right_bars: int = 3,
    use_high_low: bool = True,
) -> pd.Series:
    """
    Detect pivot lows (NO LOOK-AHEAD VERSION).

    IMPORTANT: Pivot at bar N requires N+right_bars to confirm.
    We store the pivot value at the CONFIRMATION bar (i), not the pivot bar.
    This ensures no future information leaks into the observation.

    Returns Series with pivot value at confirmation bar, NaN elsewhere.
    """
    source = low if use_high_low else close
    pivots = pd.Series(np.nan, index=source.index)

    for i in range(left_bars + right_bars, len(source)):
        pivot_idx = i - right_bars
        pivot_val = source.iloc[pivot_idx]

        # Check left bars
        is_pivot = True
        for j in range(pivot_idx - left_bars, pivot_idx):
            if source.iloc[j] <= pivot_val:
                is_pivot = False
                break

        # Check right bars
        if is_pivot:
            for j in range(pivot_idx + 1, pivot_idx + right_bars + 1):
                if j < len(source) and source.iloc[j] <= pivot_val:
                    is_pivot = False
                    break

        if is_pivot:
            # Store at confirmation bar (i), NOT pivot bar (pivot_idx)
            # This prevents look-ahead bias
            pivots.iloc[i] = pivot_val

    return pivots


def support_resistance_levels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    left_bars: int = 4,
    right_bars: int = 3,
    use_high_low: bool = True,
    max_levels: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate current support and resistance levels based on recent pivots.

    Returns:
        (support_level, resistance_level) - forward-filled from pivot detection
    """
    pivot_highs = detect_pivot_high(high, close, left_bars, right_bars, use_high_low)
    pivot_lows = detect_pivot_low(low, close, left_bars, right_bars, use_high_low)

    # Forward-fill to get current levels
    resistance = pivot_highs.ffill()
    support = pivot_lows.ffill()

    return support, resistance


# =============================================================================
# MOMENTUM BREAKOUT DETECTION
# =============================================================================

def breakout_signal(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    lookback: int = 40,
    vol_multiplier: float = 1.1,
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect breakout signals based on price and volume.

    Returns:
        (breakout_long, breakout_short) - boolean series
    """
    highest_high = high.rolling(window=lookback, min_periods=1).max()
    lowest_low = low.rolling(window=lookback, min_periods=1).min()
    avg_volume = volume.rolling(window=lookback, min_periods=1).mean()

    volume_surge = volume > (avg_volume * vol_multiplier)

    breakout_long = (close > highest_high.shift(1)) & volume_surge
    breakout_short = (close < lowest_low.shift(1)) & volume_surge

    return breakout_long.fillna(False), breakout_short.fillna(False)


def velocity_signal(
    close: pd.Series,
    threshold: float = 3.5,
    bars: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect velocity (momentum) signals using PERCENTAGE threshold.

    Original sr_swing_strategy.py (lines 1440-1443):
        price_change_pct = ((close - close[bars]) / close[bars]) * 100
        velocity_bull = price_change_pct > threshold
        velocity_bear = price_change_pct < -threshold

    Args:
        close: Close price series
        threshold: Percentage threshold (e.g., 3.5 = 3.5% move)
        bars: Number of bars to measure change over

    Returns:
        (velocity_long, velocity_short) - boolean series
    """
    # Calculate percentage change over N bars
    prev_close = close.shift(bars)
    price_change_pct = ((close - prev_close) / prev_close.replace(0, np.nan)) * 100

    velocity_long = price_change_pct > threshold
    velocity_short = price_change_pct < -threshold

    return velocity_long.fillna(False), velocity_short.fillna(False)


def atr_expansion(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    trend_direction: pd.Series,
    length: int = 14,
    multiplier: float = 3.5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect ATR expansion (volatility breakout) with TREND ALIGNMENT.

    Original sr_swing_strategy.py (lines 1446-1454):
        atr_exp_bull = current_atr > avg_atr * multiplier AND trend_is_bullish
        atr_exp_bear = current_atr > avg_atr * multiplier AND trend_is_bearish

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        trend_direction: Trend direction series (1=bullish, -1=bearish, 0=neutral)
        length: ATR period
        multiplier: ATR expansion threshold multiplier

    Returns:
        (atr_exp_bull, atr_exp_bear) - boolean series for bullish/bearish expansions
    """
    current_atr = atr(high, low, close, length)
    avg_atr = current_atr.rolling(window=length * 2).mean()

    # ATR is expanding
    atr_expanding = current_atr > avg_atr * multiplier

    # Only signal when aligned with trend
    atr_exp_bull = (atr_expanding & (trend_direction > 0)).fillna(False)
    atr_exp_bear = (atr_expanding & (trend_direction < 0)).fillna(False)

    return atr_exp_bull, atr_exp_bear


def darvas_box(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 5,
    confirmation_bars: int = 3,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Darvas Box Theory detection.

    Original sr_swing_strategy.py (lines 1175-1182):
        box_state: 0=None, 1=Building Ceiling, 2=Building Floor, 3=Active
        box_breakout_bull: close > box_top
        box_breakout_bear: close < box_bottom

    The Darvas Box identifies consolidation ranges and breakouts:
    1. New high is made (N-period high)
    2. Wait for 'confirmation_bars' consecutive bars that don't exceed the high = ceiling
    3. Find the lowest low during ceiling confirmation = floor
    4. Box is active when both top and bottom are established
    5. Breakout occurs when price closes outside the box

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        lookback: Period for detecting new highs
        confirmation_bars: Bars needed to confirm ceiling/floor

    Returns:
        (box_top, box_bottom, box_state, breakout_bull, breakout_bear)
        - box_top: Current box ceiling price
        - box_bottom: Current box floor price
        - box_state: 0=None, 1=Building Ceiling, 2=Building Floor, 3=Active
        - breakout_bull: Boolean series for bullish breakout
        - breakout_bear: Boolean series for bearish breakout
    """
    n = len(high)
    box_top = pd.Series(np.nan, index=high.index)
    box_bottom = pd.Series(np.nan, index=high.index)
    box_state = pd.Series(0, index=high.index)  # 0=None
    breakout_bull = pd.Series(False, index=high.index)
    breakout_bear = pd.Series(False, index=high.index)

    # Rolling max for N-period high detection
    rolling_max = high.rolling(window=lookback).max()

    # State machine variables
    current_top = np.nan
    current_bottom = np.nan
    state = 0  # 0=None, 1=Building Ceiling, 2=Building Floor, 3=Active
    ceiling_bars = 0
    floor_low = np.nan

    for i in range(lookback, n):
        curr_high = high.iloc[i]
        curr_low = low.iloc[i]
        curr_close = close.iloc[i]
        prev_rolling_max = rolling_max.iloc[i - 1] if i > 0 else np.nan

        if state == 0:  # No box
            # Check for new N-period high
            if not np.isnan(prev_rolling_max) and curr_high > prev_rolling_max:
                current_top = curr_high
                state = 1  # Start building ceiling
                ceiling_bars = 0
                floor_low = curr_low

        elif state == 1:  # Building Ceiling
            if curr_high > current_top:
                # New high - reset ceiling
                current_top = curr_high
                ceiling_bars = 0
                floor_low = curr_low
            else:
                ceiling_bars += 1
                floor_low = min(floor_low, curr_low)
                if ceiling_bars >= confirmation_bars:
                    # Ceiling confirmed, start building floor
                    state = 2
                    current_bottom = floor_low

        elif state == 2:  # Building Floor
            if curr_high > current_top:
                # Breakout during floor building - bullish
                breakout_bull.iloc[i] = True
                state = 0  # Reset
                current_top = np.nan
                current_bottom = np.nan
            elif curr_low < current_bottom:
                # Lower low - update floor
                current_bottom = curr_low
            else:
                # Floor confirmed when no new lows for confirmation_bars
                # For simplicity, consider floor active immediately after ceiling
                state = 3  # Box active

        elif state == 3:  # Active Box
            if curr_close > current_top:
                # Bullish breakout
                breakout_bull.iloc[i] = True
                # Start new box from this high
                current_top = curr_high
                state = 1
                ceiling_bars = 0
                floor_low = curr_low
            elif curr_close < current_bottom:
                # Bearish breakout
                breakout_bear.iloc[i] = True
                state = 0
                current_top = np.nan
                current_bottom = np.nan

        # Store state
        box_top.iloc[i] = current_top
        box_bottom.iloc[i] = current_bottom
        box_state.iloc[i] = state

    return box_top, box_bottom, box_state, breakout_bull, breakout_bear


def gravity_mode(
    close: pd.Series,
    rsi: pd.Series,
    mfi: pd.Series,
    trend_direction: pd.Series,
    momentum_threshold: float = 50.0,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
) -> pd.Series:
    """
    Gravity Mode detection from sr_swing_strategy.py.

    Original (lines 1487-1516):
        Gravity Mode combines trend direction, RSI, and MFI to determine
        market "gravity" - the dominant force pulling price:
        - "Bullish": Strong upward bias
        - "Bearish": Strong downward bias
        - "Neutral": No clear bias

    Args:
        close: Close price series
        rsi: RSI values
        mfi: MFI values
        trend_direction: Trend direction (1=bullish, -1=bearish, 0=neutral)
        momentum_threshold: MFI threshold for momentum
        rsi_oversold: RSI oversold level
        rsi_overbought: RSI overbought level

    Returns:
        Series with values: "Bullish", "Bearish", "Neutral"
    """
    # Calculate momentum conditions
    mfi_bullish = (mfi > momentum_threshold) & (mfi.diff() > 0)
    mfi_bearish = (mfi < momentum_threshold) & (mfi.diff() < 0)

    rsi_not_overbought = rsi < rsi_overbought
    rsi_not_oversold = rsi > rsi_oversold

    # Bullish gravity: trend bullish + MFI bullish + RSI not overbought
    bullish = (trend_direction > 0) & mfi_bullish & rsi_not_overbought

    # Bearish gravity: trend bearish + MFI bearish + RSI not oversold
    bearish = (trend_direction < 0) & mfi_bearish & rsi_not_oversold

    # Create result series
    result = pd.Series("Neutral", index=close.index)
    result[bullish] = "Bullish"
    result[bearish] = "Bearish"

    return result


def pivot_strength(
    pivot_high: pd.Series,
    pivot_low: pd.Series,
    high: pd.Series,
    low: pd.Series,
    tolerance_pct: float = 0.2,
    lookback: int = 100,
) -> Tuple[pd.Series, pd.Series]:
    """
    Track pivot strength by counting touches per support/resistance level.

    Original sr_swing_strategy.py concept:
        More touches = stronger level = higher probability reaction

    Args:
        pivot_high: Series with pivot high prices (NaN where no pivot)
        pivot_low: Series with pivot low prices (NaN where no pivot)
        high: High price series
        low: Low price series
        tolerance_pct: Percentage tolerance for "touch" detection
        lookback: How many bars back to count touches

    Returns:
        (resistance_strength, support_strength) - count of touches at each bar
    """
    n = len(high)
    resistance_strength = pd.Series(0, index=high.index)
    support_strength = pd.Series(0, index=high.index)

    # Track all pivot levels
    resistance_levels = []
    support_levels = []

    for i in range(n):
        # Add new pivots to tracking
        if not np.isnan(pivot_high.iloc[i]):
            resistance_levels.append((i, pivot_high.iloc[i]))
        if not np.isnan(pivot_low.iloc[i]):
            support_levels.append((i, pivot_low.iloc[i]))

        # Remove old levels outside lookback
        resistance_levels = [(idx, lvl) for idx, lvl in resistance_levels if i - idx <= lookback]
        support_levels = [(idx, lvl) for idx, lvl in support_levels if i - idx <= lookback]

        # Count touches at current bar
        curr_high = high.iloc[i]
        curr_low = low.iloc[i]

        r_touches = 0
        for _, level in resistance_levels:
            tolerance = level * tolerance_pct / 100
            if abs(curr_high - level) <= tolerance:
                r_touches += 1

        s_touches = 0
        for _, level in support_levels:
            tolerance = level * tolerance_pct / 100
            if abs(curr_low - level) <= tolerance:
                s_touches += 1

        resistance_strength.iloc[i] = r_touches
        support_strength.iloc[i] = s_touches

    return resistance_strength, support_strength


def pending_level_confirmation(
    close: pd.Series,
    pivot_high: pd.Series,
    pivot_low: pd.Series,
    confirmation_bars: int = 3,
    tolerance_pct: float = 0.3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Track pending level confirmations - levels waiting to be tested.

    Original sr_swing_strategy.py concept:
        A level becomes "confirmed" after price tests it multiple times
        or after spending time near it.

    Args:
        close: Close price series
        pivot_high: Pivot high series
        pivot_low: Pivot low series
        confirmation_bars: Bars needed near level to confirm
        tolerance_pct: Percentage tolerance for "near level"

    Returns:
        (pending_resistance, pending_support) - number of pending confirmations
    """
    n = len(close)
    pending_resistance = pd.Series(0, index=close.index)
    pending_support = pd.Series(0, index=close.index)

    # Track pending levels: (idx, price, bars_near)
    pending_r_levels = []
    pending_s_levels = []

    for i in range(n):
        curr_close = close.iloc[i]

        # Add new pivots as pending
        if not np.isnan(pivot_high.iloc[i]):
            pending_r_levels.append([i, pivot_high.iloc[i], 0])
        if not np.isnan(pivot_low.iloc[i]):
            pending_s_levels.append([i, pivot_low.iloc[i], 0])

        # Update pending resistance levels
        new_pending_r = []
        for level_data in pending_r_levels:
            idx, level, bars_near = level_data
            tolerance = level * tolerance_pct / 100
            if abs(curr_close - level) <= tolerance:
                bars_near += 1
            if bars_near < confirmation_bars:
                new_pending_r.append([idx, level, bars_near])
            # If bars_near >= confirmation_bars, level is confirmed (removed from pending)
        pending_r_levels = new_pending_r

        # Update pending support levels
        new_pending_s = []
        for level_data in pending_s_levels:
            idx, level, bars_near = level_data
            tolerance = level * tolerance_pct / 100
            if abs(curr_close - level) <= tolerance:
                bars_near += 1
            if bars_near < confirmation_bars:
                new_pending_s.append([idx, level, bars_near])
        pending_s_levels = new_pending_s

        # Store counts
        pending_resistance.iloc[i] = len(pending_r_levels)
        pending_support.iloc[i] = len(pending_s_levels)

    return pending_resistance, pending_support


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_all_indicators(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute all indicators for a DataFrame with OHLCV data.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        config: Optional dict with indicator parameters

    Returns:
        DataFrame with original data plus all indicator columns
    """
    if config is None:
        config = {}

    result = df.copy()

    # Unpack config with defaults
    trend_method = config.get("trend_method", "HMA")
    trend_length = config.get("trend_length", 14)
    kalman_short = config.get("kalman_short_len", 18)
    kalman_long = config.get("kalman_long_len", 23)
    pivot_left = config.get("pivot_left_bars", 4)
    pivot_right = config.get("pivot_right_bars", 3)
    use_high_low = config.get("use_high_low", True)
    rsi_length = config.get("rsi_length", 14)
    mfi_length = config.get("mfi_length", 14)
    atr_length = config.get("atr_length", 14)
    breakout_lookback = config.get("breakout_lookback", 40)
    velocity_threshold = config.get("velocity_threshold", 3.5)
    velocity_bars = config.get("velocity_bars", 3)
    bb_length = config.get("bb_length", 20)
    bb_mult = config.get("bb_mult", 2.0)
    zl_length = config.get("zl_length", 32)
    zl_loop_start = config.get("zl_loop_start", 1)
    zl_loop_end = config.get("zl_loop_end", 70)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Basic MAs
    result["sma_20"] = sma(close, 20)
    result["sma_50"] = sma(close, 50)
    result["ema_20"] = ema(close, 20)

    # ATR
    result["atr"] = atr(high, low, close, atr_length)

    # Trend
    trend_val, trend_dir = calculate_trend(
        close, trend_method, trend_length,
        kalman_short=kalman_short, kalman_long=kalman_long
    )
    result["trend_value"] = trend_val
    result["trend_direction"] = trend_dir

    # Kalman (always compute for state)
    short_k, long_k, kalman_dir = kalman_trend(close, kalman_short, kalman_long)
    result["kalman_short"] = short_k
    result["kalman_long"] = long_k
    result["kalman_direction"] = kalman_dir

    # HMA
    result["hma"] = hma(close, trend_length)
    result["hma_direction"] = np.sign(result["hma"].diff()).fillna(0)

    # Momentum indicators
    result["rsi"] = rsi(close, rsi_length)
    result["mfi"] = mfi(high, low, close, volume, mfi_length)

    # MACD
    macd_line, signal_line, histogram = macd(close)
    result["macd"] = macd_line
    result["macd_signal"] = signal_line
    result["macd_histogram"] = histogram

    # Zero Lag Score (oscillator filter - original sr_swing lines 1386-1392)
    result["zl_ema"] = zero_lag_ema(close, zl_length)
    result["zl_score"] = zero_lag_score(close, zl_length, zl_loop_start, zl_loop_end)
    result["zl_rising"] = (result["zl_ema"].diff() > 0).astype(float)
    result["zl_falling"] = (result["zl_ema"].diff() < 0).astype(float)

    # Bollinger Bands
    bb_upper, bb_basis, bb_lower, bb_width = bollinger_bands(close, bb_length, bb_mult)
    result["bb_upper"] = bb_upper
    result["bb_basis"] = bb_basis
    result["bb_lower"] = bb_lower
    result["bb_width"] = bb_width
    result["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    result["bb_position"] = result["bb_position"].fillna(0.5).clip(0, 1)

    # Pivots and S/R
    result["pivot_high"] = detect_pivot_high(high, close, pivot_left, pivot_right, use_high_low)
    result["pivot_low"] = detect_pivot_low(low, close, pivot_left, pivot_right, use_high_low)
    result["support"], result["resistance"] = support_resistance_levels(
        high, low, close, pivot_left, pivot_right, use_high_low
    )

    # Distance to S/R (normalized by ATR)
    result["dist_to_support"] = (close - result["support"]) / result["atr"].replace(0, np.nan)
    result["dist_to_resistance"] = (result["resistance"] - close) / result["atr"].replace(0, np.nan)
    result["dist_to_support"] = result["dist_to_support"].fillna(0).clip(-10, 10)
    result["dist_to_resistance"] = result["dist_to_resistance"].fillna(0).clip(-10, 10)

    # Breakout signals
    result["breakout_long"], result["breakout_short"] = breakout_signal(
        close, high, low, volume, breakout_lookback
    )

    # Velocity signals (uses percentage threshold, not ATR-normalized)
    result["velocity_long"], result["velocity_short"] = velocity_signal(
        close, velocity_threshold, velocity_bars
    )

    # ATR expansion (requires trend alignment)
    result["atr_exp_bull"], result["atr_exp_bear"] = atr_expansion(
        high, low, close, result["trend_direction"], atr_length
    )

    # Volume relative to average
    result["volume_ratio"] = volume / volume.rolling(window=20).mean().replace(0, np.nan)
    result["volume_ratio"] = result["volume_ratio"].fillna(1.0).clip(0, 10)

    # =========================================================================
    # P2 FEATURES - Advanced SR Swing Strategy features
    # =========================================================================

    # Darvas Box Theory (sr_swing lines 1175-1182)
    box_lookback = config.get("box_lookback", 5)
    box_confirmation = config.get("box_confirmation_bars", 3)
    result["box_top"], result["box_bottom"], result["box_state"], \
        result["box_breakout_bull"], result["box_breakout_bear"] = darvas_box(
            high, low, close, box_lookback, box_confirmation
        )
    # Box position: where price is within the box (0-1, NaN if no box)
    box_range = (result["box_top"] - result["box_bottom"]).replace(0, np.nan)
    result["box_position"] = (close - result["box_bottom"]) / box_range
    result["box_position"] = result["box_position"].clip(0, 1).fillna(0.5)
    # Box active flag
    result["box_active"] = (result["box_state"] == 3).astype(float)

    # Gravity Mode (sr_swing lines 1487-1516)
    mfi_momentum_threshold = config.get("mfi_momentum_threshold", 50.0)
    rsi_oversold = config.get("rsi_oversold", 30.0)
    rsi_overbought = config.get("rsi_overbought", 70.0)
    result["gravity_mode"] = gravity_mode(
        close, result["rsi"], result["mfi"], result["trend_direction"],
        mfi_momentum_threshold, rsi_oversold, rsi_overbought
    )
    # Encode gravity as numeric for RL
    result["gravity_bullish"] = (result["gravity_mode"] == "Bullish").astype(float)
    result["gravity_bearish"] = (result["gravity_mode"] == "Bearish").astype(float)

    # Pivot Strength (touch count per level)
    strength_tolerance = config.get("pivot_strength_tolerance_pct", 0.2)
    strength_lookback = config.get("pivot_strength_lookback", 100)
    result["resistance_strength"], result["support_strength"] = pivot_strength(
        result["pivot_high"], result["pivot_low"], high, low,
        strength_tolerance, strength_lookback
    )

    # Pending Level Confirmation
    pending_confirmation_bars = config.get("pending_confirmation_bars", 3)
    pending_tolerance = config.get("pending_tolerance_pct", 0.3)
    result["pending_resistance"], result["pending_support"] = pending_level_confirmation(
        close, result["pivot_high"], result["pivot_low"],
        pending_confirmation_bars, pending_tolerance
    )

    return result


if __name__ == "__main__":
    # Demo with random data
    import numpy as np

    np.random.seed(42)
    n = 1000

    # Generate random walk price data
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = close * (1 + np.random.randn(n) * 0.005)
    volume = np.random.uniform(1000, 10000, n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    result = compute_all_indicators(df)
    print(f"Computed {len(result.columns)} columns")
    print(f"Columns: {list(result.columns)}")
    print(f"\nSample (last 5 rows):")
    print(result[["close", "rsi", "trend_direction", "support", "resistance"]].tail())
