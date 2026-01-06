"""Tests for intraday box strategy features.

This module tests session boundary detection and related box feature components.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date

from src.crypto.data_manager import DataManager


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
