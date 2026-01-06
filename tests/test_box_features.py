"""Tests for intraday box strategy features.

This module tests session boundary detection, BoxState, and related box feature components.
"""

import json
import numpy as np
import pandas as pd
import pytest
from datetime import date

from src.crypto.data_manager import DataManager
from src.crypto.box_features import (
    BoxState,
    Zone,
    ValidationSignals,
    calculate_validation_signals,
    BoxFeatureExtractor,
    BOX_FEATURE_NAMES,
)


class TestSessionBoundaryDetection:
    """Tests for DataManager.add_session_boundaries() method."""

    def test_basic_multi_day_data(self):
        """Test basic session detection across multiple days."""
        # Create 3 days of 5-minute data (288 bars per day)
        n_days = 3
        bars_per_day = 288  # 24 * 60 / 5 = 288 bars per day at 5m
        n = n_days * bars_per_day

        index = pd.date_range(
            start="2024-01-01 00:00:00",
            periods=n,
            freq="5min",
            tz="UTC"
        )

        df = pd.DataFrame(
            {
                "open": np.random.rand(n) * 100 + 100,
                "high": np.random.rand(n) * 100 + 105,
                "low": np.random.rand(n) * 100 + 95,
                "close": np.random.rand(n) * 100 + 100,
                "volume": np.random.rand(n) * 1000,
            },
            index=index,
        )

        result = DataManager.add_session_boundaries(df)

        # Verify columns added
        assert "session_id" in result.columns
        assert "session_start" in result.columns
        assert "session_date" in result.columns

        # Verify correct number of sessions
        assert result["session_id"].nunique() == n_days

        # Verify session_id values are 0, 1, 2
        assert set(result["session_id"].unique()) == {0, 1, 2}

        # Verify first bar of each day is session_start
        assert result["session_start"].sum() == n_days

        # Verify session_id increments correctly
        session_starts = result[result["session_start"]]
        assert list(session_starts["session_id"]) == [0, 1, 2]

    def test_single_day_data(self):
        """Test with data from a single day - should have only 1 session."""
        n = 96  # 8 hours of 5-minute bars
        index = pd.date_range(
            start="2024-06-15 08:00:00",
            periods=n,
            freq="5min",
            tz="UTC"
        )

        df = pd.DataFrame(
            {
                "open": np.ones(n) * 100,
                "high": np.ones(n) * 105,
                "low": np.ones(n) * 95,
                "close": np.ones(n) * 100,
                "volume": np.ones(n) * 1000,
            },
            index=index,
        )

        result = DataManager.add_session_boundaries(df)

        # Should be exactly 1 session
        assert result["session_id"].nunique() == 1
        assert result["session_id"].iloc[0] == 0
        assert result["session_start"].sum() == 1
        assert result["session_start"].iloc[0] == True

    def test_data_gap_spanning_days(self):
        """Test that gaps spanning multiple days increment session_id by 1."""
        # Day 1: some bars
        day1_index = pd.date_range(
            start="2024-01-01 00:00:00",
            periods=10,
            freq="5min",
            tz="UTC"
        )

        # Day 5: some bars (3 days gap)
        day5_index = pd.date_range(
            start="2024-01-05 00:00:00",
            periods=10,
            freq="5min",
            tz="UTC"
        )

        index = day1_index.append(day5_index)
        n = len(index)

        df = pd.DataFrame(
            {
                "open": np.ones(n) * 100,
                "high": np.ones(n) * 105,
                "low": np.ones(n) * 95,
                "close": np.ones(n) * 100,
                "volume": np.ones(n) * 1000,
            },
            index=index,
        )

        result = DataManager.add_session_boundaries(df)

        # Should have exactly 2 sessions (not 5!)
        assert result["session_id"].nunique() == 2

        # First 10 bars: session 0
        assert all(result["session_id"].iloc[:10] == 0)

        # Last 10 bars: session 1
        assert all(result["session_id"].iloc[10:] == 1)

    def test_first_bar_always_session_start(self):
        """Test that first bar is always marked as session start."""
        # Even if it's mid-day, first bar should be session_start
        index = pd.date_range(
            start="2024-03-15 14:30:00",  # Mid-day start
            periods=50,
            freq="5min",
            tz="UTC"
        )

        df = pd.DataFrame(
            {
                "open": np.ones(50) * 100,
                "high": np.ones(50) * 105,
                "low": np.ones(50) * 95,
                "close": np.ones(50) * 100,
                "volume": np.ones(50) * 1000,
            },
            index=index,
        )

        result = DataManager.add_session_boundaries(df)

        # First bar must be session_start
        assert result["session_start"].iloc[0] == True

    def test_unsorted_data_gets_sorted(self):
        """Test that unsorted data is automatically sorted."""
        # Create intentionally unsorted data
        dates = [
            "2024-01-02 10:00:00",
            "2024-01-01 10:00:00",
            "2024-01-02 10:05:00",
            "2024-01-01 10:05:00",
        ]
        index = pd.to_datetime(dates).tz_localize("UTC")

        df = pd.DataFrame(
            {
                "open": [100, 99, 101, 98],
                "high": [105, 104, 106, 103],
                "low": [95, 94, 96, 93],
                "close": [100, 99, 101, 98],
                "volume": [1000, 1000, 1000, 1000],
            },
            index=index,
        )

        result = DataManager.add_session_boundaries(df)

        # Data should be sorted
        assert result.index.is_monotonic_increasing

        # Verify sessions are correct after sorting
        # Sorted order: Jan 1 10:00, Jan 1 10:05, Jan 2 10:00, Jan 2 10:05
        assert result["session_id"].iloc[0] == 0  # Jan 1
        assert result["session_id"].iloc[1] == 0  # Jan 1
        assert result["session_id"].iloc[2] == 1  # Jan 2
        assert result["session_id"].iloc[3] == 1  # Jan 2

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty DataFrame"):
            DataManager.add_session_boundaries(df)

    def test_non_datetime_index_raises_error(self):
        """Test that non-datetime index raises ValueError."""
        df = pd.DataFrame(
            {
                "open": [100, 101],
                "close": [100, 101],
            },
            index=[0, 1],  # Integer index, not datetime
        )

        with pytest.raises(ValueError, match="DatetimeIndex"):
            DataManager.add_session_boundaries(df)

    def test_timezone_naive_assumed_utc(self):
        """Test that timezone-naive timestamps are assumed UTC."""
        index = pd.date_range(
            start="2024-01-01",
            periods=500,
            freq="5min",
            # No tz= means timezone-naive
        )

        df = pd.DataFrame(
            {
                "open": np.ones(500) * 100,
                "high": np.ones(500) * 105,
                "low": np.ones(500) * 95,
                "close": np.ones(500) * 100,
                "volume": np.ones(500) * 1000,
            },
            index=index,
        )

        # Should not raise, should work with UTC assumption
        result = DataManager.add_session_boundaries(df)

        assert "session_id" in result.columns
        assert result["session_id"].iloc[0] == 0

    def test_session_date_column_correct(self):
        """Test that session_date column contains correct dates."""
        # 2 days of data
        n = 576  # 2 * 288
        index = pd.date_range(
            start="2024-07-20 00:00:00",
            periods=n,
            freq="5min",
            tz="UTC"
        )

        df = pd.DataFrame(
            {
                "open": np.ones(n) * 100,
                "high": np.ones(n) * 105,
                "low": np.ones(n) * 95,
                "close": np.ones(n) * 100,
                "volume": np.ones(n) * 1000,
            },
            index=index,
        )

        result = DataManager.add_session_boundaries(df)

        # Check dates
        expected_dates = [date(2024, 7, 20), date(2024, 7, 21)]
        unique_dates = result["session_date"].unique()

        assert len(unique_dates) == 2
        assert date(2024, 7, 20) in unique_dates
        assert date(2024, 7, 21) in unique_dates

    def test_session_id_dtype(self):
        """Test that session_id is int32 type."""
        index = pd.date_range(
            start="2024-01-01",
            periods=100,
            freq="5min",
            tz="UTC"
        )

        df = pd.DataFrame(
            {
                "open": np.ones(100) * 100,
                "high": np.ones(100) * 105,
                "low": np.ones(100) * 95,
                "close": np.ones(100) * 100,
                "volume": np.ones(100) * 1000,
            },
            index=index,
        )

        result = DataManager.add_session_boundaries(df)

        assert result["session_id"].dtype == np.int32

    def test_original_columns_preserved(self):
        """Test that original OHLCV columns are unchanged."""
        index = pd.date_range(
            start="2024-01-01",
            periods=100,
            freq="5min",
            tz="UTC"
        )

        original_close = np.random.rand(100) * 100 + 100

        df = pd.DataFrame(
            {
                "open": np.ones(100) * 100,
                "high": np.ones(100) * 105,
                "low": np.ones(100) * 95,
                "close": original_close,
                "volume": np.ones(100) * 1000,
            },
            index=index,
        )

        result = DataManager.add_session_boundaries(df)

        # Original columns should be identical
        np.testing.assert_array_equal(result["close"].values, original_close)
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "volume" in result.columns

    def test_does_not_modify_original(self):
        """Test that original DataFrame is not modified."""
        index = pd.date_range(
            start="2024-01-01",
            periods=100,
            freq="5min",
            tz="UTC"
        )

        df = pd.DataFrame(
            {
                "open": np.ones(100) * 100,
                "high": np.ones(100) * 105,
                "low": np.ones(100) * 95,
                "close": np.ones(100) * 100,
                "volume": np.ones(100) * 1000,
            },
            index=index,
        )

        original_columns = list(df.columns)

        _ = DataManager.add_session_boundaries(df)

        # Original should not have new columns
        assert list(df.columns) == original_columns
        assert "session_id" not in df.columns


class TestLoadOhlcvWithSessionInfo:
    """Tests for load_ohlcv() with add_session_info parameter."""

    def test_add_session_info_false_default(self, sample_ohlcv, tmp_path, config):
        """Test that session info is not added by default."""
        # Save sample data
        config.data.raw_dir = tmp_path
        dm = DataManager(config)
        dm.save_ohlcv(sample_ohlcv, "TEST/USDT:USDT", "15m")

        # Load without session info (default)
        df = dm.load_ohlcv("TEST/USDT:USDT", "15m", auto_aggregate=False)

        assert "session_id" not in df.columns
        assert "session_start" not in df.columns
        assert "session_date" not in df.columns

    def test_add_session_info_true(self, sample_ohlcv, tmp_path, config):
        """Test that session info is added when requested."""
        # Save sample data
        config.data.raw_dir = tmp_path
        dm = DataManager(config)
        dm.save_ohlcv(sample_ohlcv, "TEST/USDT:USDT", "15m")

        # Load with session info
        df = dm.load_ohlcv(
            "TEST/USDT:USDT",
            "15m",
            auto_aggregate=False,
            add_session_info=True
        )

        assert "session_id" in df.columns
        assert "session_start" in df.columns
        assert "session_date" in df.columns


class TestBoxState:
    """Tests for BoxState dataclass."""

    def test_initialization_defaults(self):
        """Test BoxState initializes with correct defaults."""
        state = BoxState()

        assert state.session_high == 0.0
        assert state.session_low == float('inf')
        assert state.session_open == 0.0
        assert state.high_touch_count == 0
        assert state.low_touch_count == 0
        assert state.mid_touch_count == 0
        assert state.broke_high == False
        assert state.broke_low == False
        assert state.bar_count == 0
        assert not state.is_valid

    def test_reset_initializes_state(self):
        """Test reset() properly initializes with first bar data."""
        state = BoxState()
        state.reset(first_open=100.0, first_high=105.0, first_low=95.0)

        assert state.session_high == 105.0
        assert state.session_low == 95.0
        assert state.session_open == 100.0
        assert state.bar_count == 1
        assert state.is_valid
        assert state.box_mid == 100.0
        assert state.box_height == 10.0

    def test_reset_clears_previous_state(self):
        """Test reset() clears all accumulated state."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)

        # Accumulate some state
        state.update(106.0, 96.0, 100.0)
        state.high_touch_count = 3
        state.broke_high = True

        # Reset
        state.reset(200.0, 210.0, 190.0)

        assert state.session_high == 210.0
        assert state.session_low == 190.0
        assert state.high_touch_count == 0
        assert state.broke_high == False
        assert state.bar_count == 1

    def test_update_expands_box(self):
        """Test update() expands box when new highs/lows are made."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)

        # New high
        state.update(110.0, 100.0, 108.0)
        assert state.session_high == 110.0
        assert state.session_low == 95.0

        # New low
        state.update(105.0, 90.0, 92.0)
        assert state.session_high == 110.0
        assert state.session_low == 90.0

    def test_box_never_shrinks(self):
        """Test that box boundaries never shrink within session."""
        state = BoxState()
        state.reset(100.0, 110.0, 90.0)

        # Update with smaller range
        state.update(105.0, 95.0, 100.0)

        # Box should not shrink
        assert state.session_high == 110.0
        assert state.session_low == 90.0

    def test_touch_detection_with_hysteresis(self):
        """Test touch detection counts once per approach."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)

        # Price at high (tolerance 1% of 10 = 0.1)
        state.update(105.0, 104.0, 104.95, tolerance_pct=0.01)
        assert state.high_touch_count == 1

        # Still at high - should not count again
        state.update(105.0, 104.0, 104.96, tolerance_pct=0.01)
        assert state.high_touch_count == 1

        # Move away
        state.update(103.0, 101.0, 102.0, tolerance_pct=0.01)
        assert state.high_touch_count == 1

        # Return to high - should count as new touch
        state.update(105.0, 104.0, 104.95, tolerance_pct=0.01)
        assert state.high_touch_count == 2

    def test_touch_count_capped(self):
        """Test touch counts are capped at MAX_TOUCH_COUNT."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)

        # Simulate 10 touches to high
        for i in range(10):
            # Touch
            state.update(105.0, 104.0, 104.95, tolerance_pct=0.01)
            # Move away
            state.update(102.0, 100.0, 101.0, tolerance_pct=0.01)

        # Should be capped at 5
        assert state.high_touch_count == state.MAX_TOUCH_COUNT

    def test_breakout_state_sticky(self):
        """Test that broke_high/broke_low are sticky flags."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)

        # Break above high
        state.update(110.0, 106.0, 108.0)
        assert state.broke_high == True

        # Come back inside box
        state.update(104.0, 100.0, 102.0)
        assert state.broke_high == True  # Still true!

        # Break below low
        state.update(94.0, 90.0, 92.0)
        assert state.broke_low == True
        assert state.broke_high == True  # Both true

    def test_zone_classification_above_box(self):
        """Test zone classification when price is above box."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)

        zone = state.update(110.0, 106.0, 108.0, tolerance_pct=0.01)
        assert zone == Zone.ABOVE_BOX

    def test_zone_classification_below_box(self):
        """Test zone classification when price is below box."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)

        zone = state.update(94.0, 90.0, 92.0, tolerance_pct=0.01)
        assert zone == Zone.BELOW_BOX

    def test_zone_classification_upper(self):
        """Test zone classification when price in upper zone."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)  # mid = 100

        zone = state.update(104.0, 101.0, 102.0, tolerance_pct=0.01)
        assert zone == Zone.UPPER

    def test_zone_classification_lower(self):
        """Test zone classification when price in lower zone."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)  # mid = 100

        zone = state.update(99.0, 96.0, 98.0, tolerance_pct=0.01)
        assert zone == Zone.LOWER

    def test_zone_classification_at_high(self):
        """Test zone classification at high level."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)

        zone = state.update(105.5, 104.5, 105.05, tolerance_pct=0.01)
        assert zone == Zone.AT_HIGH

    def test_zone_classification_at_low(self):
        """Test zone classification at low level."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)

        zone = state.update(95.5, 94.5, 94.95, tolerance_pct=0.01)
        assert zone == Zone.AT_LOW

    def test_zone_classification_at_mid(self):
        """Test zone classification at mid level."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)  # mid = 100

        zone = state.update(100.5, 99.5, 100.05, tolerance_pct=0.01)
        assert zone == Zone.AT_MID

    def test_bars_since_touch_tracking(self):
        """Test bars_since_*_touch methods."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)

        # No touches yet
        assert state.bars_since_high_touch() == -1

        # Touch high
        state.update(105.0, 104.0, 104.95, tolerance_pct=0.01)
        assert state.bars_since_high_touch() == 0

        # Move away (2 bars)
        state.update(102.0, 100.0, 101.0, tolerance_pct=0.01)
        state.update(101.0, 99.0, 100.0, tolerance_pct=0.01)
        assert state.bars_since_high_touch() == 2

    def test_serialization_to_dict(self):
        """Test to_dict() serialization."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)
        # Touch the high first (close within tolerance of 105.0)
        state.update(105.0, 104.0, 104.95, tolerance_pct=0.01)  # Touch
        # Then break above
        state.update(110.0, 106.0, 108.0, tolerance_pct=0.01)  # Breakout

        data = state.to_dict()

        assert data["session_high"] == 110.0
        assert data["session_low"] == 95.0
        assert data["high_touch_count"] == 1
        assert data["broke_high"] == True
        assert data["bar_count"] == 3

    def test_deserialization_from_dict(self):
        """Test from_dict() deserialization."""
        data = {
            "session_high": 110.0,
            "session_low": 90.0,
            "session_open": 100.0,
            "high_touch_count": 3,
            "low_touch_count": 2,
            "mid_touch_count": 1,
            "broke_high": True,
            "broke_low": False,
            "last_zone": "upper",
            "bar_count": 50,
            "_is_touching_high": False,
            "_is_touching_low": False,
            "_is_touching_mid": True,
            "_last_high_touch_bar": 45,
            "_last_low_touch_bar": 30,
            "_last_mid_touch_bar": 50,
        }

        state = BoxState.from_dict(data)

        assert state.session_high == 110.0
        assert state.session_low == 90.0
        assert state.high_touch_count == 3
        assert state.broke_high == True
        assert state.last_zone == Zone.UPPER
        assert state.bar_count == 50

    def test_json_round_trip(self):
        """Test JSON serialization round-trip preserves state."""
        original = BoxState()
        original.reset(100.0, 110.0, 90.0)
        original.update(112.0, 108.0, 111.0, tolerance_pct=0.01)
        original.update(105.0, 102.0, 103.0, tolerance_pct=0.01)

        # Round-trip
        json_str = original.to_json()
        restored = BoxState.from_json(json_str)

        assert restored.session_high == original.session_high
        assert restored.session_low == original.session_low
        assert restored.high_touch_count == original.high_touch_count
        assert restored.broke_high == original.broke_high
        assert restored.bar_count == original.bar_count

    def test_repr_string(self):
        """Test __repr__ produces readable output."""
        state = BoxState()
        state.reset(100.0, 105.0, 95.0)

        repr_str = repr(state)

        assert "BoxState" in repr_str
        assert "105.00" in repr_str  # high
        assert "95.00" in repr_str   # low
        assert "100.00" in repr_str  # mid

    def test_box_mid_property(self):
        """Test box_mid property calculation."""
        state = BoxState()
        state.reset(100.0, 110.0, 90.0)

        assert state.box_mid == 100.0

        # Expand box
        state.update(120.0, 95.0, 110.0)
        assert state.box_mid == 105.0  # (120 + 90) / 2

    def test_box_height_property(self):
        """Test box_height property calculation."""
        state = BoxState()
        state.reset(100.0, 110.0, 90.0)

        assert state.box_height == 20.0

        # Expand box
        state.update(130.0, 85.0, 100.0)
        assert state.box_height == 45.0  # 130 - 85

    def test_is_valid_property(self):
        """Test is_valid property."""
        state = BoxState()
        assert not state.is_valid

        state.reset(100.0, 105.0, 95.0)
        assert state.is_valid


class TestValidationSignals:
    """Tests for ValidationSignals dataclass and calculate_validation_signals function."""

    def test_initialization_defaults(self):
        """Test ValidationSignals initializes with correct defaults."""
        signals = ValidationSignals()

        assert signals.rejection_wick_ratio == 0.0
        assert signals.volume_vs_session_avg == 1.0
        assert signals.approach_bars == 0
        assert signals.zone_entry_bar == 0

    def test_rejection_wick_ratio_bullish_candle(self):
        """Test rejection_wick_ratio for bullish candle with large lower wick.

        Bullish candle: open=98, close=102 (bullish body)
        Range: low=95, high=105 (range=10)
        Upper wick: 105 - 102 = 3
        Lower wick: 98 - 95 = 3

        Wait - this gives equal wicks. Let's make a proper bullish rejection:
        open=100, close=104 (bullish body, body_top=104, body_bottom=100)
        Range: low=94, high=105 (range=11)
        Upper wick: 105 - 104 = 1
        Lower wick: 100 - 94 = 6
        Ratio: (6 - 1) / 11 = 0.4545... (positive = bullish rejection)
        """
        signals = ValidationSignals()

        # Bullish candle with large lower wick (hammer-like)
        ratio = signals.calculate_rejection_wick_ratio(
            open_price=100.0, high=105.0, low=94.0, close=104.0
        )

        # Expected: (6 - 1) / 11 = 0.4545
        assert ratio > 0  # Positive indicates bullish rejection
        assert abs(ratio - 0.4545) < 0.01

    def test_rejection_wick_ratio_bearish_candle(self):
        """Test rejection_wick_ratio for bearish candle with large upper wick.

        Bearish candle with shooting star pattern:
        open=104, close=100 (bearish body, body_top=104, body_bottom=100)
        Range: low=99, high=110 (range=11)
        Upper wick: 110 - 104 = 6
        Lower wick: 100 - 99 = 1
        Ratio: (1 - 6) / 11 = -0.4545... (negative = bearish rejection)
        """
        signals = ValidationSignals()

        # Bearish candle with large upper wick (shooting star)
        ratio = signals.calculate_rejection_wick_ratio(
            open_price=104.0, high=110.0, low=99.0, close=100.0
        )

        # Expected: (1 - 6) / 11 = -0.4545
        assert ratio < 0  # Negative indicates bearish rejection
        assert abs(ratio - (-0.4545)) < 0.01

    def test_rejection_wick_ratio_doji(self):
        """Test rejection_wick_ratio for doji candle with equal wicks.

        Doji: open=close=100
        Range: low=95, high=105 (range=10)
        Upper wick: 105 - 100 = 5
        Lower wick: 100 - 95 = 5
        Ratio: (5 - 5) / 10 = 0 (neutral)
        """
        signals = ValidationSignals()

        # Perfect doji with equal wicks
        ratio = signals.calculate_rejection_wick_ratio(
            open_price=100.0, high=105.0, low=95.0, close=100.0
        )

        assert abs(ratio) < 0.001  # Should be very close to 0

    def test_rejection_wick_ratio_zero_range(self):
        """Test rejection_wick_ratio handles zero-range candle (all prices equal)."""
        signals = ValidationSignals()

        # All prices equal (single tick candle)
        ratio = signals.calculate_rejection_wick_ratio(
            open_price=100.0, high=100.0, low=100.0, close=100.0
        )

        assert ratio == 0.0  # Should return 0 for degenerate case

    def test_rejection_wick_ratio_bounds(self):
        """Test rejection_wick_ratio is always bounded to [-1, 1]."""
        signals = ValidationSignals()

        # Maximum bullish (all lower wick, no upper wick)
        # open=close=high=100, low=90 => upper=0, lower=10, ratio=1.0
        ratio = signals.calculate_rejection_wick_ratio(
            open_price=100.0, high=100.0, low=90.0, close=100.0
        )
        assert ratio <= 1.0
        assert ratio > 0.9  # Should be close to 1

        # Maximum bearish (all upper wick, no lower wick)
        # open=close=low=100, high=110 => upper=10, lower=0, ratio=-1.0
        ratio = signals.calculate_rejection_wick_ratio(
            open_price=100.0, high=110.0, low=100.0, close=100.0
        )
        assert ratio >= -1.0
        assert ratio < -0.9  # Should be close to -1

    def test_volume_vs_session_avg_normal(self):
        """Test volume_vs_session_avg with normal volume levels."""
        signals = ValidationSignals()

        # Volume at session average
        ratio = signals.calculate_volume_vs_session_avg(
            current_volume=1000.0, session_avg_volume=1000.0
        )
        assert abs(ratio - 1.0) < 0.001

        # Volume 50% above average
        ratio = signals.calculate_volume_vs_session_avg(
            current_volume=1500.0, session_avg_volume=1000.0
        )
        assert abs(ratio - 1.5) < 0.001

        # Volume 50% below average
        ratio = signals.calculate_volume_vs_session_avg(
            current_volume=500.0, session_avg_volume=1000.0
        )
        assert abs(ratio - 0.5) < 0.001

    def test_volume_vs_session_avg_capping(self):
        """Test volume_vs_session_avg is capped to [0.1, 5.0]."""
        signals = ValidationSignals()

        # Very high volume (should cap at 5.0)
        ratio = signals.calculate_volume_vs_session_avg(
            current_volume=10000.0, session_avg_volume=1000.0
        )
        assert ratio == 5.0

        # Very low volume (should cap at 0.1)
        ratio = signals.calculate_volume_vs_session_avg(
            current_volume=50.0, session_avg_volume=1000.0
        )
        assert ratio == 0.1

        # Zero volume (should cap at 0.1)
        ratio = signals.calculate_volume_vs_session_avg(
            current_volume=0.0, session_avg_volume=1000.0
        )
        assert ratio == 0.1

    def test_volume_vs_session_avg_zero_session_avg(self):
        """Test volume_vs_session_avg handles zero session average."""
        signals = ValidationSignals()

        # Zero session average should return neutral 1.0
        ratio = signals.calculate_volume_vs_session_avg(
            current_volume=1000.0, session_avg_volume=0.0
        )
        assert ratio == 1.0

    def test_volume_vs_session_avg_negative_volume(self):
        """Test volume_vs_session_avg handles negative current volume."""
        signals = ValidationSignals()

        # Negative volume should be treated as 0
        ratio = signals.calculate_volume_vs_session_avg(
            current_volume=-100.0, session_avg_volume=1000.0
        )
        assert ratio == 0.1  # 0 / 1000 capped to 0.1

    def test_approach_bars_tracking(self):
        """Test approach_bars calculation."""
        signals = ValidationSignals()

        # Just entered zone
        bars = signals.calculate_approach_bars(current_bar=50, zone_entry_bar=50)
        assert bars == 0

        # In zone for 5 bars
        bars = signals.calculate_approach_bars(current_bar=55, zone_entry_bar=50)
        assert bars == 5

        # In zone for 100 bars
        bars = signals.calculate_approach_bars(current_bar=150, zone_entry_bar=50)
        assert bars == 100

    def test_approach_bars_invalid_zone_entry(self):
        """Test approach_bars with invalid zone entry (negative)."""
        signals = ValidationSignals()

        # Zone entry not tracked (negative value)
        bars = signals.calculate_approach_bars(current_bar=50, zone_entry_bar=-1)
        assert bars == -1

    def test_approach_bars_entry_after_current(self):
        """Test approach_bars handles zone_entry_bar > current_bar."""
        signals = ValidationSignals()

        # This shouldn't happen, but should return 0 (not negative)
        bars = signals.calculate_approach_bars(current_bar=50, zone_entry_bar=60)
        assert bars == 0  # max(0, -10) = 0

    def test_calculate_all_signals(self):
        """Test calculate() method returns all signals."""
        signals = ValidationSignals()

        result = signals.calculate(
            open_price=100.0,
            high=105.0,
            low=94.0,
            close=104.0,
            volume=1500.0,
            session_avg_volume=1000.0,
            current_bar=50,
            zone_entry_bar=45,
        )

        # Check all keys present
        assert "rejection_wick_ratio" in result
        assert "volume_vs_session_avg" in result
        assert "approach_bars" in result

        # Check values are reasonable
        assert result["rejection_wick_ratio"] > 0  # Bullish candle
        assert abs(result["volume_vs_session_avg"] - 1.5) < 0.001
        assert result["approach_bars"] == 5

        # Check instance variables are also set
        assert signals.rejection_wick_ratio == result["rejection_wick_ratio"]
        assert signals.volume_vs_session_avg == result["volume_vs_session_avg"]
        assert signals.approach_bars == result["approach_bars"]
        assert signals.zone_entry_bar == 45

    def test_calculate_validation_signals_function(self):
        """Test module-level calculate_validation_signals function."""
        result = calculate_validation_signals(
            open_price=100.0,
            high=105.0,
            low=94.0,
            close=104.0,
            volume=1500.0,
            session_avg_volume=1000.0,
            current_bar=50,
            zone_entry_bar=45,
        )

        # Check all keys present
        assert "rejection_wick_ratio" in result
        assert "volume_vs_session_avg" in result
        assert "approach_bars" in result

        # Check values match ValidationSignals.calculate()
        signals = ValidationSignals()
        expected = signals.calculate(
            open_price=100.0,
            high=105.0,
            low=94.0,
            close=104.0,
            volume=1500.0,
            session_avg_volume=1000.0,
            current_bar=50,
            zone_entry_bar=45,
        )

        assert result == expected

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        signals = ValidationSignals()
        signals.calculate(
            open_price=100.0,
            high=110.0,
            low=95.0,
            close=108.0,
            volume=2000.0,
            session_avg_volume=1000.0,
            current_bar=75,
            zone_entry_bar=60,
        )

        # Serialize
        data = signals.to_dict()

        # Deserialize
        restored = ValidationSignals.from_dict(data)

        assert restored.rejection_wick_ratio == signals.rejection_wick_ratio
        assert restored.volume_vs_session_avg == signals.volume_vs_session_avg
        assert restored.approach_bars == signals.approach_bars
        assert restored.zone_entry_bar == signals.zone_entry_bar

    def test_from_dict_with_missing_keys(self):
        """Test from_dict handles missing keys with defaults."""
        data = {"rejection_wick_ratio": 0.5}  # Missing other keys

        signals = ValidationSignals.from_dict(data)

        assert signals.rejection_wick_ratio == 0.5
        assert signals.volume_vs_session_avg == 1.0  # Default
        assert signals.approach_bars == 0  # Default
        assert signals.zone_entry_bar == 0  # Default

    def test_numerical_stability_small_values(self):
        """Test numerical stability with very small price values."""
        signals = ValidationSignals()

        # Very small prices (like some penny stocks or small cap crypto)
        ratio = signals.calculate_rejection_wick_ratio(
            open_price=0.0001, high=0.00015, low=0.00005, close=0.00012
        )

        # Should still produce valid bounded result
        assert -1.0 <= ratio <= 1.0
        assert not np.isnan(ratio)
        assert not np.isinf(ratio)

    def test_numerical_stability_large_values(self):
        """Test numerical stability with very large price values."""
        signals = ValidationSignals()

        # Very large prices (like BTC)
        ratio = signals.calculate_rejection_wick_ratio(
            open_price=50000.0, high=51000.0, low=49000.0, close=50500.0
        )

        # Should still produce valid bounded result
        assert -1.0 <= ratio <= 1.0
        assert not np.isnan(ratio)
        assert not np.isinf(ratio)


class TestBoxFeatureExtractor:
    """Tests for BoxFeatureExtractor class."""

    def test_initialization(self):
        """Test BoxFeatureExtractor initializes correctly."""
        extractor = BoxFeatureExtractor(warmup_bars=30)

        assert extractor.warmup_bars == 30
        assert extractor.feature_dim == len(BOX_FEATURE_NAMES)
        assert extractor.feature_dim == 20  # 20 features
        assert not extractor.is_warm
        assert extractor.session_avg_volume == 0.0

    def test_feature_names_match(self):
        """Test feature_names property returns correct names."""
        extractor = BoxFeatureExtractor()

        names = extractor.feature_names
        assert names == BOX_FEATURE_NAMES
        assert len(names) == 20

        # Verify it returns a copy (immutable)
        names.append("extra")
        assert "extra" not in extractor.feature_names

    def test_reset_session(self):
        """Test reset_session initializes state correctly."""
        extractor = BoxFeatureExtractor()
        extractor.reset_session(
            first_open=100.0,
            first_high=105.0,
            first_low=95.0,
            first_volume=1000.0,
        )

        assert extractor.box_state.session_high == 105.0
        assert extractor.box_state.session_low == 95.0
        assert extractor.box_state.session_open == 100.0
        assert extractor.box_state.bar_count == 1
        assert extractor.session_avg_volume == 1000.0

    def test_update_expands_box(self):
        """Test update expands box boundaries correctly."""
        extractor = BoxFeatureExtractor()
        extractor.reset_session(100.0, 105.0, 95.0, 1000.0)

        # Update with higher high
        extractor.update(high=110.0, low=100.0, close=108.0, volume=1200.0)

        assert extractor.box_state.session_high == 110.0
        assert extractor.box_state.session_low == 95.0
        assert extractor.box_state.bar_count == 2

    def test_warmup_tracking(self):
        """Test is_warm property tracks warmup correctly."""
        extractor = BoxFeatureExtractor(warmup_bars=5)
        extractor.reset_session(100.0, 105.0, 95.0)

        assert not extractor.is_warm

        # Add 4 more bars (total 5 = warmup)
        for i in range(4):
            extractor.update(high=105.0, low=95.0, close=100.0)

        assert extractor.is_warm

    def test_extract_features_returns_all_features(self):
        """Test extract_features returns all 20 features."""
        extractor = BoxFeatureExtractor()
        extractor.reset_session(100.0, 110.0, 90.0, 1000.0)

        features = extractor.extract_features(
            open_price=100.0,
            high=108.0,
            low=95.0,
            close=105.0,
            volume=1500.0,
            atr=5.0,
        )

        assert len(features) == 20
        for name in BOX_FEATURE_NAMES:
            assert name in features

    def test_extract_features_bounded(self):
        """Test all extracted features are within expected bounds."""
        extractor = BoxFeatureExtractor()
        extractor.reset_session(100.0, 110.0, 90.0, 1000.0)

        # Add some bars to get meaningful state
        for _ in range(10):
            extractor.update(high=108.0, low=92.0, close=100.0, volume=1000.0)

        features = extractor.extract_features(
            open_price=100.0,
            high=108.0,
            low=92.0,
            close=105.0,
            volume=1500.0,
            atr=5.0,
        )

        # Check bounds for each feature type
        assert 0.0 <= features["box_height_atr_ratio"] <= 1.0
        assert 0.0 <= features["session_progress"] <= 1.0
        assert -1.0 <= features["dist_to_high_norm"] <= 1.0
        assert -1.0 <= features["dist_to_mid_norm"] <= 1.0
        assert -1.0 <= features["dist_to_low_norm"] <= 1.0
        assert -1.0 <= features["zone_position"] <= 1.0
        assert features["at_high"] in [0.0, 1.0]
        assert features["at_mid"] in [0.0, 1.0]
        assert features["at_low"] in [0.0, 1.0]
        assert 0.0 <= features["high_touch_norm"] <= 1.0
        assert 0.0 <= features["mid_touch_norm"] <= 1.0
        assert 0.0 <= features["low_touch_norm"] <= 1.0
        assert 0.0 <= features["bars_since_high_touch_norm"] <= 1.0
        assert 0.0 <= features["bars_since_low_touch_norm"] <= 1.0
        assert features["broke_high"] in [0.0, 1.0]
        assert features["broke_low"] in [0.0, 1.0]
        assert features["breakout_direction"] in [-1.0, 0.0, 1.0]
        assert -1.0 <= features["rejection_wick_ratio"] <= 1.0
        assert 0.0 <= features["volume_vs_session_avg_norm"] <= 1.0
        assert 0.0 <= features["approach_bars_norm"] <= 1.0

    def test_extract_feature_array_shape(self):
        """Test extract_feature_array returns correct shape."""
        extractor = BoxFeatureExtractor()
        extractor.reset_session(100.0, 110.0, 90.0)

        arr = extractor.extract_feature_array(
            open_price=100.0,
            high=108.0,
            low=92.0,
            close=105.0,
            volume=1000.0,
            atr=5.0,
        )

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (20,)
        assert arr.dtype == np.float32

    def test_extract_feature_array_matches_dict(self):
        """Test feature array matches dictionary values."""
        extractor = BoxFeatureExtractor()
        extractor.reset_session(100.0, 110.0, 90.0, 1000.0)

        features_dict = extractor.extract_features(
            open_price=100.0,
            high=108.0,
            low=92.0,
            close=105.0,
            volume=1500.0,
            atr=5.0,
        )
        features_arr = extractor.extract_feature_array(
            open_price=100.0,
            high=108.0,
            low=92.0,
            close=105.0,
            volume=1500.0,
            atr=5.0,
        )

        for i, name in enumerate(BOX_FEATURE_NAMES):
            assert abs(features_arr[i] - features_dict[name]) < 1e-6

    def test_zone_position_encoding(self):
        """Test zone_position encodes zones correctly."""
        extractor = BoxFeatureExtractor()
        extractor.reset_session(100.0, 110.0, 90.0)

        # Test price in UPPER zone
        extractor.update(high=108.0, low=102.0, close=106.0)  # Above mid (100)
        features = extractor.extract_features(100.0, 108.0, 102.0, 106.0, 1000.0, 5.0)
        assert features["zone_position"] == 0.33  # UPPER zone

        # Reset and test LOWER zone
        extractor.reset_session(100.0, 110.0, 90.0)
        extractor.update(high=98.0, low=92.0, close=94.0)  # Below mid
        features = extractor.extract_features(92.0, 98.0, 92.0, 94.0, 1000.0, 5.0)
        assert features["zone_position"] == -0.33  # LOWER zone

    def test_breakout_features(self):
        """Test breakout features track correctly."""
        extractor = BoxFeatureExtractor()
        extractor.reset_session(100.0, 110.0, 90.0)

        # Initial state - no breakout
        features = extractor.extract_features(100.0, 105.0, 95.0, 100.0, 1000.0, 5.0)
        assert features["broke_high"] == 0.0
        assert features["broke_low"] == 0.0
        assert features["breakout_direction"] == 0.0

        # Break above high
        extractor.update(high=115.0, low=105.0, close=112.0)
        features = extractor.extract_features(105.0, 115.0, 105.0, 112.0, 1000.0, 5.0)
        assert features["broke_high"] == 1.0
        assert features["broke_low"] == 0.0
        assert features["breakout_direction"] == 1.0

    def test_volume_tracking(self):
        """Test session average volume tracking."""
        extractor = BoxFeatureExtractor()
        extractor.reset_session(100.0, 110.0, 90.0, first_volume=1000.0)

        # Add more volume
        extractor.update(high=108.0, low=92.0, close=100.0, volume=2000.0)
        extractor.update(high=108.0, low=92.0, close=100.0, volume=3000.0)

        # Average should be (1000 + 2000 + 3000) / 3 = 2000
        assert extractor.session_avg_volume == 2000.0

    def test_serialization_round_trip(self):
        """Test to_dict and from_dict preserve state."""
        extractor = BoxFeatureExtractor(warmup_bars=20, touch_tolerance_pct=0.002)
        extractor.reset_session(100.0, 110.0, 90.0, 1000.0)

        # Add some state
        for i in range(5):
            extractor.update(high=108.0, low=92.0, close=100.0, volume=1000.0 + i * 100)

        # Serialize
        data = extractor.to_dict()

        # Deserialize
        restored = BoxFeatureExtractor.from_dict(data)

        # Verify state matches
        assert restored.warmup_bars == extractor.warmup_bars
        assert restored.touch_tolerance_pct == extractor.touch_tolerance_pct
        assert restored.box_state.bar_count == extractor.box_state.bar_count
        assert restored.box_state.session_high == extractor.box_state.session_high
        assert restored._zone_entry_bar == extractor._zone_entry_bar
        assert restored.session_avg_volume == extractor.session_avg_volume

    def test_invalid_state_returns_zeros(self):
        """Test extract_features returns zeros before reset_session."""
        extractor = BoxFeatureExtractor()

        # Before any reset, state is invalid
        features = extractor.extract_features(100.0, 105.0, 95.0, 100.0, 1000.0, 5.0)

        # All features should be 0.0
        for name in BOX_FEATURE_NAMES:
            assert features[name] == 0.0

    def test_repr_string(self):
        """Test __repr__ returns informative string."""
        extractor = BoxFeatureExtractor(warmup_bars=10)
        extractor.reset_session(100.0, 110.0, 90.0)

        repr_str = repr(extractor)
        assert "BoxFeatureExtractor" in repr_str
        assert "warm=" in repr_str
        assert "bars=" in repr_str
        assert "features=20" in repr_str

    def test_numerical_stability_zero_atr(self):
        """Test extract_features handles zero ATR gracefully."""
        extractor = BoxFeatureExtractor()
        extractor.reset_session(100.0, 110.0, 90.0)

        # ATR = 0 should not cause division by zero
        features = extractor.extract_features(100.0, 105.0, 95.0, 100.0, 1000.0, atr=0.0)

        assert features["box_height_atr_ratio"] == 0.0
        assert not np.isnan(features["box_height_atr_ratio"])

    def test_numerical_stability_zero_box_height(self):
        """Test extract_features handles zero box height gracefully."""
        extractor = BoxFeatureExtractor()
        # Session with same high and low
        extractor.reset_session(100.0, 100.0, 100.0)

        features = extractor.extract_features(100.0, 100.0, 100.0, 100.0, 1000.0, 5.0)

        # Distance features should be 0, not NaN
        assert features["dist_to_high_norm"] == 0.0
        assert features["dist_to_mid_norm"] == 0.0
        assert features["dist_to_low_norm"] == 0.0
        assert not np.isnan(features["dist_to_high_norm"])

    def test_touch_count_features(self):
        """Test touch count normalization."""
        # Use larger tolerance (5% of box height)
        extractor = BoxFeatureExtractor(touch_tolerance_pct=0.05)  # 5% tolerance
        extractor.reset_session(100.0, 110.0, 90.0)  # box_height = 20

        # Touch the high level multiple times
        # With 5% tolerance and height=20, tolerance = 1.0
        # Close of 109.5 is 0.5 away from 110, within tolerance
        for _ in range(3):
            extractor.update(high=110.0, low=108.0, close=109.5)  # At high (0.5 < 1.0 tolerance)
            extractor.update(high=105.0, low=100.0, close=102.0)  # Move away

        features = extractor.extract_features(100.0, 105.0, 100.0, 102.0, 1000.0, 5.0)

        # Should have touched high at least once
        assert features["high_touch_norm"] > 0.0
        assert features["high_touch_norm"] <= 1.0

    def test_approach_bars_feature(self):
        """Test approach_bars_norm tracks zone duration."""
        extractor = BoxFeatureExtractor()
        extractor.reset_session(100.0, 110.0, 90.0)

        # Stay in same zone for multiple bars
        for _ in range(10):
            extractor.update(high=105.0, low=101.0, close=103.0)  # UPPER zone

        features = extractor.extract_features(101.0, 105.0, 101.0, 103.0, 1000.0, 5.0)

        # Should show time in zone
        assert features["approach_bars_norm"] > 0.0
