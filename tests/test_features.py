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


class TestCombinedFeatureExtraction:
    """Integration tests for FeatureExtractor with box features (task feg)."""

    @pytest.fixture
    def config_with_box(self):
        """Config with box features enabled."""
        from src.crypto.config import CryptoConfig
        config = CryptoConfig()
        config.box_features.use_box_features = True
        return config

    @pytest.fixture
    def config_without_box(self):
        """Config with box features disabled."""
        from src.crypto.config import CryptoConfig
        config = CryptoConfig()
        config.box_features.use_box_features = False
        return config

    def test_feature_dim_with_box_features(self, config_with_box):
        """Test feature dimension increases with box features enabled."""
        extractor = FeatureExtractor(config_with_box, window_size=30)

        # 76 base features + 20 box features = 96
        assert extractor.feature_dim == 96
        assert extractor._base_feature_dim == 76
        assert extractor._box_feature_dim == 20

    def test_feature_dim_without_box_features(self, config_without_box):
        """Test feature dimension unchanged without box features."""
        extractor = FeatureExtractor(config_without_box, window_size=30)

        # Only 76 base features
        assert extractor.feature_dim == 76
        assert extractor._base_feature_dim == 76
        assert extractor._box_feature_dim == 0

    def test_observation_shape_with_box(self, config_with_box):
        """Test observation shape includes box features."""
        extractor = FeatureExtractor(config_with_box, window_size=30)

        shape = extractor.observation_shape
        assert shape == (30, 96)

    def test_observation_shape_without_box(self, config_without_box):
        """Test observation shape excludes box features when disabled."""
        extractor = FeatureExtractor(config_without_box, window_size=30)

        shape = extractor.observation_shape
        assert shape == (30, 76)

    def test_box_extractor_exists_when_enabled(self, config_with_box):
        """Test BoxFeatureExtractor is created when enabled."""
        extractor = FeatureExtractor(config_with_box)

        assert extractor._box_extractor is not None
        assert extractor.box_extractor is not None
        assert extractor.use_box_features is True

    def test_box_extractor_none_when_disabled(self, config_without_box):
        """Test BoxFeatureExtractor is None when disabled."""
        extractor = FeatureExtractor(config_without_box)

        assert extractor._box_extractor is None
        assert extractor.box_extractor is None
        assert extractor.use_box_features is False

    def test_precompute_features_shape_with_box(self, sample_ohlcv, config_with_box):
        """Test precomputed features have correct shape (base only, no box)."""
        extractor = FeatureExtractor(config_with_box, window_size=30)

        # precompute_features only returns BASE features
        # Box features are computed dynamically in environment
        features = extractor.precompute_features(sample_ohlcv)

        assert features.shape[0] == len(sample_ohlcv)
        assert features.shape[1] == 76  # Only base features precomputed

    def test_get_box_features_returns_array(self, config_with_box):
        """Test get_box_features returns correctly shaped array."""
        extractor = FeatureExtractor(config_with_box)

        # Initialize box state
        extractor.reset_box_session(100.0, 105.0, 95.0, 1000.0)

        box_features = extractor.get_box_features(
            open_price=100.0,
            high=102.0,
            low=98.0,
            close=101.0,
            volume=1000.0,
            atr=2.0,
        )

        assert isinstance(box_features, np.ndarray)
        assert box_features.shape == (20,)
        assert np.isfinite(box_features).all()

    def test_get_box_features_empty_when_disabled(self, config_without_box):
        """Test get_box_features returns empty array when disabled."""
        extractor = FeatureExtractor(config_without_box)

        box_features = extractor.get_box_features(
            open_price=100.0,
            high=102.0,
            low=98.0,
            close=101.0,
            volume=1000.0,
            atr=2.0,
        )

        assert isinstance(box_features, np.ndarray)
        assert box_features.shape == (0,)

    def test_all_feature_names_with_box(self, config_with_box):
        """Test get_all_feature_names includes box features."""
        from src.crypto.box_features import BOX_FEATURE_NAMES

        extractor = FeatureExtractor(config_with_box)
        all_names = extractor.get_all_feature_names()

        # Should have 76 base + 20 box = 96 names
        assert len(all_names) == 96

        # Box feature names should be at the end
        for box_name in BOX_FEATURE_NAMES:
            assert box_name in all_names

    def test_all_feature_names_without_box(self, config_without_box):
        """Test get_all_feature_names excludes box features when disabled."""
        from src.crypto.box_features import BOX_FEATURE_NAMES

        extractor = FeatureExtractor(config_without_box)
        all_names = extractor.get_all_feature_names()

        assert len(all_names) == 76

        # Box feature names should NOT be present
        for box_name in BOX_FEATURE_NAMES:
            assert box_name not in all_names

    def test_extracted_features_no_nan(self, sample_ohlcv, config_with_box):
        """Test extracted features contain no NaN values."""
        extractor = FeatureExtractor(config_with_box, window_size=30)
        features = extractor.precompute_features(sample_ohlcv)

        assert not np.isnan(features).any(), "Features contain NaN values"

    def test_extracted_features_no_inf(self, sample_ohlcv, config_with_box):
        """Test extracted features contain no Inf values."""
        extractor = FeatureExtractor(config_with_box, window_size=30)
        features = extractor.precompute_features(sample_ohlcv)

        assert not np.isinf(features).any(), "Features contain Inf values"

    def test_box_features_bounded(self, config_with_box):
        """Test box features are within expected bounds."""
        extractor = FeatureExtractor(config_with_box)

        # Initialize and update box state
        extractor.reset_box_session(100.0, 110.0, 90.0, 1000.0)
        for _ in range(10):
            extractor.update_box_state(
                high=105.0 + np.random.rand() * 5,
                low=95.0 - np.random.rand() * 5,
                close=100.0 + np.random.randn() * 3,
                volume=1000.0 + np.random.rand() * 500,
            )

        box_features = extractor.get_box_features(
            open_price=100.0, high=103.0, low=97.0, close=101.0,
            volume=1000.0, atr=5.0,
        )

        # All features should be bounded (most are [0, 1] or [-1, 1])
        assert np.all(box_features >= -2.0), "Box features below -2"
        assert np.all(box_features <= 2.0), "Box features above 2"


class TestFeatureRegressionWithBoxToggle:
    """
    CRITICAL regression tests ensuring base features unchanged (task 5v7).

    When box features are enabled/disabled, the base 76 features must produce
    IDENTICAL values. Any discrepancy indicates a bug in the integration.
    """

    @pytest.fixture
    def config_with_box(self):
        from src.crypto.config import CryptoConfig
        config = CryptoConfig()
        config.box_features.use_box_features = True
        return config

    @pytest.fixture
    def config_without_box(self):
        from src.crypto.config import CryptoConfig
        config = CryptoConfig()
        config.box_features.use_box_features = False
        return config

    def test_base_feature_count_unchanged(self, config_with_box, config_without_box):
        """Test base feature count is consistent."""
        extractor_with = FeatureExtractor(config_with_box)
        extractor_without = FeatureExtractor(config_without_box)

        assert extractor_with._base_feature_dim == extractor_without._base_feature_dim
        assert extractor_with._base_feature_dim == 76

    def test_precomputed_features_identical(self, sample_ohlcv, config_with_box, config_without_box):
        """
        CRITICAL: Precomputed features must be identical with and without box features.

        This test verifies that enabling box features doesn't alter base feature calculation.
        """
        extractor_with = FeatureExtractor(config_with_box, window_size=30)
        extractor_without = FeatureExtractor(config_without_box, window_size=30)

        features_with = extractor_with.precompute_features(sample_ohlcv)
        features_without = extractor_without.precompute_features(sample_ohlcv)

        # Shapes should match (precompute only returns base features)
        assert features_with.shape == features_without.shape

        # Values must be EXACTLY identical
        np.testing.assert_array_equal(
            features_with, features_without,
            err_msg="Base features differ when box features enabled vs disabled"
        )

    def test_feature_by_feature_comparison(self, sample_ohlcv, config_with_box, config_without_box):
        """
        CRITICAL: Compare each feature column individually.

        This provides detailed diagnostics if regression occurs.
        """
        extractor_with = FeatureExtractor(config_with_box, window_size=30)
        extractor_without = FeatureExtractor(config_without_box, window_size=30)

        # Extract features as DataFrames for column-wise comparison
        features_with_df = extractor_with.extract_features(sample_ohlcv)
        features_without_df = extractor_without.extract_features(sample_ohlcv)

        # Get base feature names (excluding box features)
        base_names = extractor_without._get_feature_names()

        # Compare each feature column
        for name in base_names:
            with_col = features_with_df[name].values
            without_col = features_without_df[name].values

            np.testing.assert_array_equal(
                with_col, without_col,
                err_msg=f"Feature '{name}' differs when box features enabled"
            )

    def test_extract_features_same_columns(self, sample_ohlcv, config_with_box, config_without_box):
        """Test extract_features returns same base columns regardless of box toggle."""
        extractor_with = FeatureExtractor(config_with_box)
        extractor_without = FeatureExtractor(config_without_box)

        features_with = extractor_with.extract_features(sample_ohlcv)
        features_without = extractor_without.extract_features(sample_ohlcv)

        # All columns from without-box should exist in with-box
        for col in features_without.columns:
            assert col in features_with.columns, f"Column '{col}' missing when box enabled"

    def test_base_feature_names_unchanged(self, config_with_box, config_without_box):
        """Test base feature names are identical with and without box."""
        extractor_with = FeatureExtractor(config_with_box)
        extractor_without = FeatureExtractor(config_without_box)

        base_names_with = extractor_with._get_feature_names()
        base_names_without = extractor_without._get_feature_names()

        assert base_names_with == base_names_without

    def test_indicator_config_not_affected(self, config_with_box, config_without_box):
        """Test indicator config is identical regardless of box toggle."""
        extractor_with = FeatureExtractor(config_with_box)
        extractor_without = FeatureExtractor(config_without_box)

        assert extractor_with.indicator_config == extractor_without.indicator_config

    def test_htf_features_unchanged(self, sample_ohlcv, sample_ohlcv_htf, config_with_box, config_without_box):
        """Test HTF features identical with and without box toggle."""
        extractor_with = FeatureExtractor(config_with_box, include_htf=True)
        extractor_without = FeatureExtractor(config_without_box, include_htf=True)

        features_with = extractor_with.extract_features(sample_ohlcv, htf_df=sample_ohlcv_htf)
        features_without = extractor_without.extract_features(sample_ohlcv, htf_df=sample_ohlcv_htf)

        htf_columns = ["htf_trend", "mtf_aligned", "htf_rising", "htf_falling",
                       "mtf_strict_bullish", "mtf_strict_bearish",
                       "mtf_relaxed_bullish", "mtf_relaxed_bearish"]

        for col in htf_columns:
            np.testing.assert_array_equal(
                features_with[col].values, features_without[col].values,
                err_msg=f"HTF feature '{col}' differs when box features enabled"
            )

    def test_numerical_stability_preserved(self, config_with_box, config_without_box):
        """Test numerical stability is maintained with box features."""
        # Create extreme price data
        np.random.seed(42)
        n = 200
        close = pd.Series(100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.1)))
        high = close * (1 + np.abs(np.random.randn(n) * 0.05))
        low = close * (1 - np.abs(np.random.randn(n) * 0.05))

        df = pd.DataFrame({
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.uniform(100, 10000, n),
        })

        extractor_with = FeatureExtractor(config_with_box)
        extractor_without = FeatureExtractor(config_without_box)

        features_with = extractor_with.precompute_features(df)
        features_without = extractor_without.precompute_features(df)

        # Both should be numerically stable (no NaN/Inf)
        assert np.isfinite(features_with).all(), "Features with box have NaN/Inf"
        assert np.isfinite(features_without).all(), "Features without box have NaN/Inf"

        # Values should be identical
        np.testing.assert_array_equal(features_with, features_without)

    def test_seed_reproducibility(self, sample_ohlcv, config_with_box, config_without_box):
        """Test feature extraction is deterministic with same input."""
        extractor_with = FeatureExtractor(config_with_box)
        extractor_without = FeatureExtractor(config_without_box)

        # Extract twice with box enabled
        features_1 = extractor_with.precompute_features(sample_ohlcv)
        features_2 = extractor_with.precompute_features(sample_ohlcv)

        # Extract twice without box
        features_3 = extractor_without.precompute_features(sample_ohlcv)
        features_4 = extractor_without.precompute_features(sample_ohlcv)

        # All should be identical
        np.testing.assert_array_equal(features_1, features_2)
        np.testing.assert_array_equal(features_3, features_4)
        np.testing.assert_array_equal(features_1, features_3)
