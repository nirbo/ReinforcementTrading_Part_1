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
    Detect pivot highs.
    Returns Series with pivot value where detected, NaN elsewhere.
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
            pivots.iloc[pivot_idx] = pivot_val

    return pivots


def detect_pivot_low(
    low: pd.Series,
    close: pd.Series,
    left_bars: int = 4,
    right_bars: int = 3,
    use_high_low: bool = True,
) -> pd.Series:
    """
    Detect pivot lows.
    Returns Series with pivot value where detected, NaN elsewhere.
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
            pivots.iloc[pivot_idx] = pivot_val

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
    atr_series: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect velocity (momentum) signals.

    Returns:
        (velocity_long, velocity_short) - boolean series
    """
    if atr_series is None:
        # Simple ATR approximation if not provided
        atr_series = close.diff().abs().rolling(window=14).mean()

    price_change = close.diff(bars)
    velocity = price_change.abs() / (atr_series * bars).replace(0, np.nan)

    velocity_long = (price_change > 0) & (velocity > threshold)
    velocity_short = (price_change < 0) & (velocity > threshold)

    return velocity_long.fillna(False), velocity_short.fillna(False)


def atr_expansion(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    multiplier: float = 3.5,
) -> pd.Series:
    """
    Detect ATR expansion (volatility breakout).

    Returns:
        Boolean series where ATR expansion detected
    """
    current_atr = atr(high, low, close, length)
    avg_atr = current_atr.rolling(window=length * 2).mean()

    return (current_atr > avg_atr * multiplier).fillna(False)


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

    # Velocity signals
    result["velocity_long"], result["velocity_short"] = velocity_signal(
        close, velocity_threshold, velocity_bars, result["atr"]
    )

    # ATR expansion
    result["atr_expansion"] = atr_expansion(high, low, close, atr_length)

    # Volume relative to average
    result["volume_ratio"] = volume / volume.rolling(window=20).mean().replace(0, np.nan)
    result["volume_ratio"] = result["volume_ratio"].fillna(1.0).clip(0, 10)

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
