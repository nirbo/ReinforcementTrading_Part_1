"""Tests for indicators module."""

import numpy as np
import pandas as pd
import pytest

from src.crypto.indicators import (
    KalmanFilter,
    alma,
    atr,
    bollinger_bands,
    compute_all_indicators,
    detect_pivot_high,
    detect_pivot_low,
    ema,
    gaussian_filter,
    hma,
    macd,
    mfi,
    rma,
    rsi,
    sma,
    support_resistance_levels,
    wma,
    zero_lag_ema,
)


class TestBasicIndicators:
    """Test basic technical indicators."""

    def test_sma(self, sample_ohlcv):
        """Test simple moving average."""
        result = sma(sample_ohlcv["close"], 20)
        assert len(result) == len(sample_ohlcv)
        # SMA uses ewm with span which has fewer NaN at start
        valid_result = result.dropna()
        assert len(valid_result) > len(sample_ohlcv) - 30
        assert result.iloc[-1] > 0

    def test_ema(self, sample_ohlcv):
        """Test exponential moving average."""
        result = ema(sample_ohlcv["close"], 20)
        assert len(result) == len(sample_ohlcv)
        # EMA should be smoother than price
        assert result.std() < sample_ohlcv["close"].std()

    def test_wma(self, sample_ohlcv):
        """Test weighted moving average."""
        result = wma(sample_ohlcv["close"], 20)
        assert len(result) == len(sample_ohlcv)
        valid = result.dropna()
        assert len(valid) > 0

    def test_rma(self, sample_ohlcv):
        """Test running moving average (Wilder's)."""
        result = rma(sample_ohlcv["close"], 14)
        assert len(result) == len(sample_ohlcv)
        # RMA should be smoother
        assert result.dropna().std() < sample_ohlcv["close"].std()


class TestAdvancedIndicators:
    """Test advanced indicators."""

    def test_hma(self, sample_ohlcv):
        """Test Hull Moving Average."""
        result = hma(sample_ohlcv["close"], 21)
        assert len(result) == len(sample_ohlcv)
        # HMA should track price closely
        valid = result.dropna()
        assert len(valid) > 0

    def test_alma(self, sample_ohlcv):
        """Test Arnaud Legoux Moving Average."""
        result = alma(sample_ohlcv["close"], 20)
        assert len(result) == len(sample_ohlcv)
        assert not np.isinf(result).any()

    def test_zero_lag_ema(self, sample_ohlcv):
        """Test Zero-Lag EMA."""
        result = zero_lag_ema(sample_ohlcv["close"], 20)
        assert len(result) == len(sample_ohlcv)
        # Should be responsive to price changes
        assert result.diff().std() > 0


class TestKalmanFilter:
    """Test Kalman filter indicator."""

    def test_kalman_filter_basic(self, sample_ohlcv):
        """Test basic Kalman filter updates."""
        kf = KalmanFilter(length=14)

        # Apply filter to data
        results = []
        for val in sample_ohlcv["close"].values:
            kf.update(val)
            results.append(kf.estimate)

        assert len(results) == len(sample_ohlcv)
        assert not any(np.isnan(r) for r in results)

    def test_kalman_filter_tracking(self, sample_ohlcv):
        """Test Kalman filter tracks the signal."""
        kf = KalmanFilter(length=14)

        results = []
        for val in sample_ohlcv["close"].values:
            kf.update(val)
            results.append(kf.estimate)

        # Estimate should be similar to input
        corr = np.corrcoef(sample_ohlcv["close"].values, results)[0, 1]
        assert corr > 0.9  # Strong positive correlation


class TestGaussianFilter:
    """Test Gaussian filter."""

    def test_gaussian_filter_basic(self, sample_ohlcv):
        """Test basic Gaussian filter."""
        result = gaussian_filter(sample_ohlcv["close"], 14, 2)
        assert len(result) == len(sample_ohlcv)
        valid = result.dropna()
        assert len(valid) > 0

    def test_gaussian_filter_different_poles(self, sample_ohlcv):
        """Test Gaussian filter with different pole counts."""
        result_1 = gaussian_filter(sample_ohlcv["close"], 14, 1)
        result_2 = gaussian_filter(sample_ohlcv["close"], 14, 2)

        # Both should produce valid output
        assert len(result_1) == len(sample_ohlcv)
        assert len(result_2) == len(sample_ohlcv)


class TestMomentumIndicators:
    """Test momentum indicators."""

    def test_rsi(self, sample_ohlcv):
        """Test RSI indicator."""
        result = rsi(sample_ohlcv["close"], 14)
        assert len(result) == len(sample_ohlcv)
        valid = result.dropna()
        # RSI should be bounded 0-100
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_mfi(self, sample_ohlcv):
        """Test MFI indicator."""
        result = mfi(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
            14,
        )
        assert len(result) == len(sample_ohlcv)
        valid = result.dropna()
        # MFI should be bounded 0-100
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_macd(self, sample_ohlcv):
        """Test MACD indicator."""
        macd_line, signal, histogram = macd(sample_ohlcv["close"])
        assert len(macd_line) == len(sample_ohlcv)
        assert len(signal) == len(sample_ohlcv)
        assert len(histogram) == len(sample_ohlcv)
        # Histogram = MACD - Signal
        valid_idx = ~(macd_line.isna() | signal.isna())
        np.testing.assert_allclose(
            histogram[valid_idx].values,
            (macd_line[valid_idx] - signal[valid_idx]).values,
            rtol=1e-10,
        )


class TestVolatilityIndicators:
    """Test volatility indicators."""

    def test_atr(self, sample_ohlcv):
        """Test ATR indicator."""
        result = atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            14,
        )
        assert len(result) == len(sample_ohlcv)
        valid = result.dropna()
        # ATR should be positive
        assert (valid > 0).all()

    def test_bollinger_bands(self, sample_ohlcv):
        """Test Bollinger Bands."""
        upper, middle, lower, width = bollinger_bands(
            sample_ohlcv["close"], 20, 2.0
        )
        assert len(upper) == len(sample_ohlcv)

        valid_idx = ~upper.isna()
        # Upper > Middle > Lower
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()
        # Width should be positive
        assert (width[valid_idx] > 0).all()


class TestPivotDetection:
    """Test pivot point detection."""

    def test_detect_pivot_high(self, sample_ohlcv):
        """Test pivot high detection."""
        result = detect_pivot_high(sample_ohlcv["high"], 5)
        assert len(result) == len(sample_ohlcv)
        # Should have some pivots
        pivot_count = result.notna().sum()
        assert pivot_count > 0
        # Pivot values should match high values
        for idx in result.dropna().index:
            assert result.loc[idx] == sample_ohlcv["high"].loc[idx]

    def test_detect_pivot_low(self, sample_ohlcv):
        """Test pivot low detection."""
        result = detect_pivot_low(sample_ohlcv["low"], 5)
        assert len(result) == len(sample_ohlcv)
        # Should have some pivots
        pivot_count = result.notna().sum()
        assert pivot_count > 0

    def test_support_resistance_levels(self, sample_ohlcv):
        """Test S/R level detection."""
        pivot_highs = detect_pivot_high(sample_ohlcv["high"], 5)
        pivot_lows = detect_pivot_low(sample_ohlcv["low"], 5)

        support, resistance = support_resistance_levels(
            sample_ohlcv["close"],
            pivot_highs,
            pivot_lows,
        )

        assert len(support) == len(sample_ohlcv)
        assert len(resistance) == len(sample_ohlcv)
        # Both should have some valid values
        assert support.notna().sum() > 0
        assert resistance.notna().sum() > 0


class TestComputeAllIndicators:
    """Test the all-in-one indicator computation."""

    def test_compute_all_indicators(self, sample_ohlcv, config):
        """Test computing all indicators at once."""
        result = compute_all_indicators(sample_ohlcv, config.indicators.model_dump())

        # Should have original columns
        assert "open" in result.columns
        assert "close" in result.columns

        # Should have basic indicators
        assert "sma_20" in result.columns
        assert "sma_50" in result.columns
        assert "rsi" in result.columns
        assert "atr" in result.columns

        # Should have MACD
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

        # Should have Bollinger Bands
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns

        # Should have pivots
        assert "pivot_high" in result.columns
        assert "pivot_low" in result.columns

    def test_no_inf_values(self, sample_ohlcv, config):
        """Test that computed indicators have no inf values."""
        result = compute_all_indicators(sample_ohlcv, config.indicators.model_dump())
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(result[col]).sum()
            assert inf_count == 0, f"Column {col} has {inf_count} inf values"

    def test_limited_nan_values(self, sample_ohlcv, config):
        """Test that most columns have limited NaN after warmup."""
        result = compute_all_indicators(sample_ohlcv, config.indicators.model_dump())
        # After 100 bars, core columns should have minimal NaNs
        late_data = result.iloc[100:]

        # Check core columns
        core_cols = ["rsi", "atr", "sma_20", "sma_50"]
        for col in core_cols:
            if col in late_data.columns:
                nan_pct = late_data[col].isna().mean()
                assert nan_pct < 0.1, f"Column {col} has {nan_pct:.0%} NaN after warmup"
