"""Tests for feature extractor module."""

import numpy as np
import pandas as pd
import pytest

from src.crypto.feature_extractor import (
    FEATURE_SPECS,
    FeatureExtractor,
    PositionState,
    create_feature_matrix,
)


class TestPositionState:
    """Test PositionState dataclass."""

    def test_default_position_state(self):
        """Test default position state."""
        state = PositionState()
        assert state.direction == 0
        assert state.entry_price == 0.0
        assert state.unrealized_pnl_pct == 0.0
        assert state.time_in_position == 0

    def test_long_position_state(self):
        """Test long position state."""
        state = PositionState(
            direction=1,
            entry_price=100.0,
            unrealized_pnl_pct=0.05,
            time_in_position=10,
        )
        assert state.direction == 1
        assert state.entry_price == 100.0
        assert state.unrealized_pnl_pct == 0.05

    def test_short_position_state(self):
        """Test short position state."""
        state = PositionState(direction=-1, entry_price=100.0)
        assert state.direction == -1


class TestFeatureExtractor:
    """Test FeatureExtractor class."""

    def test_feature_extractor_init(self, config):
        """Test feature extractor initialization."""
        extractor = FeatureExtractor(config, window_size=30)
        assert extractor.window_size == 30
        assert extractor.feature_dim > 0
        assert extractor.include_htf is True

    def test_observation_shape(self, config):
        """Test observation shape property."""
        extractor = FeatureExtractor(config, window_size=30)
        shape = extractor.observation_shape
        assert shape == (30, extractor.feature_dim)

    def test_extract_features_basic(self, sample_ohlcv, config):
        """Test basic feature extraction."""
        extractor = FeatureExtractor(config, window_size=30)
        features = extractor.extract_features(sample_ohlcv)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv)
        # Feature dim should match actual extraction
        assert features.shape[1] > 30  # We have many features

    def test_extract_features_no_nan(self, sample_ohlcv, config):
        """Test extracted features have no NaN."""
        extractor = FeatureExtractor(config, window_size=30)
        features = extractor.extract_features(sample_ohlcv)

        nan_count = features.isna().sum().sum()
        assert nan_count == 0, f"Features have {nan_count} NaN values"

    def test_extract_features_no_inf(self, sample_ohlcv, config):
        """Test extracted features have no inf."""
        extractor = FeatureExtractor(config, window_size=30)
        features = extractor.extract_features(sample_ohlcv)

        inf_count = np.isinf(features.values).sum()
        assert inf_count == 0, f"Features have {inf_count} inf values"

    def test_extract_features_bounded(self, sample_ohlcv, config):
        """Test extracted features are reasonably bounded."""
        extractor = FeatureExtractor(config, window_size=30)
        features = extractor.extract_features(sample_ohlcv)

        # Most features should be in [-5, 5] range (after normalization)
        for col in features.columns:
            min_val = features[col].min()
            max_val = features[col].max()
            assert min_val >= -10, f"{col} min={min_val}"
            assert max_val <= 10, f"{col} max={max_val}"

    def test_extract_features_with_position(self, sample_ohlcv, config):
        """Test feature extraction with position state."""
        extractor = FeatureExtractor(config, window_size=30)

        # No position
        features_flat = extractor.extract_features(sample_ohlcv, position=None)
        assert features_flat["position_direction"].iloc[-1] == 0.0

        # Long position
        long_pos = PositionState(direction=1, unrealized_pnl_pct=0.05, time_in_position=20)
        features_long = extractor.extract_features(sample_ohlcv, position=long_pos)
        assert features_long["position_direction"].iloc[-1] == 1.0

        # Short position
        short_pos = PositionState(direction=-1, unrealized_pnl_pct=-0.02, time_in_position=5)
        features_short = extractor.extract_features(sample_ohlcv, position=short_pos)
        assert features_short["position_direction"].iloc[-1] == -1.0

    def test_get_observation(self, sample_ohlcv, config):
        """Test getting single observation."""
        extractor = FeatureExtractor(config, window_size=30)

        obs = extractor.get_observation(sample_ohlcv, current_idx=100)

        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] == 30  # window_size
        assert obs.shape[1] > 30  # Many features
        assert obs.dtype == np.float32

    def test_get_observation_at_start(self, sample_ohlcv, config):
        """Test observation at start of data (padding case)."""
        extractor = FeatureExtractor(config, window_size=30)

        # At index 10, we need padding
        obs = extractor.get_observation(sample_ohlcv, current_idx=10)

        assert obs.shape[0] == 30  # window_size
        # Should be padded, not contain NaN
        assert not np.isnan(obs).any()

    def test_get_observation_at_end(self, sample_ohlcv, config):
        """Test observation at end of data."""
        extractor = FeatureExtractor(config, window_size=30)

        obs = extractor.get_observation(sample_ohlcv, current_idx=len(sample_ohlcv) - 1)

        assert obs.shape[0] == 30
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()

    def test_precompute_features(self, sample_ohlcv, config):
        """Test precomputing all features."""
        extractor = FeatureExtractor(config, window_size=30)

        features = extractor.precompute_features(sample_ohlcv)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_ohlcv)
        assert features.shape[1] > 30  # Many features
        assert features.dtype == np.float32

    def test_feature_bounds(self, config):
        """Test feature bounds."""
        extractor = FeatureExtractor(config, window_size=30)

        low, high = extractor.get_feature_bounds()

        assert len(low) == extractor.feature_dim
        assert len(high) == extractor.feature_dim
        assert (low < high).all()


class TestFeatureSpecs:
    """Test feature specifications."""

    def test_feature_specs_exist(self):
        """Test FEATURE_SPECS is defined."""
        assert len(FEATURE_SPECS) > 0

    def test_feature_specs_have_names(self):
        """Test all feature specs have unique names."""
        names = [spec.name for spec in FEATURE_SPECS]
        assert len(names) == len(set(names)), "Duplicate feature names found"

    def test_feature_specs_have_bounds(self):
        """Test feature specs have valid bounds."""
        for spec in FEATURE_SPECS:
            assert spec.min_val < spec.max_val, f"{spec.name} has invalid bounds"


class TestHTFFeatures:
    """Test higher timeframe feature integration."""

    def test_extract_features_with_htf(self, sample_ohlcv, sample_ohlcv_htf, config):
        """Test feature extraction with HTF data."""
        extractor = FeatureExtractor(config, window_size=30, include_htf=True)

        features = extractor.extract_features(sample_ohlcv, htf_df=sample_ohlcv_htf)

        assert "htf_trend" in features.columns
        assert "mtf_aligned" in features.columns

    def test_extract_features_without_htf(self, sample_ohlcv, config):
        """Test feature extraction without HTF data."""
        extractor = FeatureExtractor(config, window_size=30, include_htf=True)

        features = extractor.extract_features(sample_ohlcv, htf_df=None)

        # Should have HTF columns with default values
        assert "htf_trend" in features.columns
        assert features["htf_trend"].iloc[-1] == 0.0


class TestCreateFeatureMatrix:
    """Test create_feature_matrix convenience function."""

    def test_create_feature_matrix(self, sample_ohlcv, config):
        """Test create_feature_matrix function."""
        features, extractor = create_feature_matrix(sample_ohlcv, window_size=30, config=config)

        assert isinstance(features, np.ndarray)
        assert isinstance(extractor, FeatureExtractor)
        assert features.shape[0] == len(sample_ohlcv)
        assert features.shape[1] > 30  # Many features
