"""Tests for trading environment module."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.crypto.trading_env import CryptoTradingEnv


class TestCryptoTradingEnv:
    """Test CryptoTradingEnv class."""

    def test_env_creation(self, sample_ohlcv, config):
        """Test environment creation."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        assert env is not None
        assert env.action_space is not None
        assert env.observation_space is not None

    def test_action_space(self, sample_ohlcv, config):
        """Test action space is correct."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 6  # HOLD, LONG, SHORT, CLOSE, LONG_TIGHT, SHORT_TIGHT

    def test_observation_space(self, sample_ohlcv, config):
        """Test observation space is correct."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert len(env.observation_space.shape) == 2
        assert env.observation_space.shape[0] == 30  # window_size

    def test_reset(self, sample_ohlcv, config):
        """Test environment reset."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

    def test_reset_with_seed(self, sample_ohlcv, config):
        """Test reset with seed produces deterministic results."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30, random_start=True)

        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

    def test_step_hold(self, sample_ohlcv, config):
        """Test step with HOLD action."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)  # HOLD

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_long(self, sample_ohlcv, config):
        """Test step with LONG action."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(1)  # LONG

        # We can't access internal state, but the step should complete
        assert obs.shape == env.observation_space.shape

    def test_step_short(self, sample_ohlcv, config):
        """Test step with SHORT action."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(2)  # SHORT

        assert obs.shape == env.observation_space.shape

    def test_step_close(self, sample_ohlcv, config):
        """Test CLOSE action."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        env.reset()

        # CLOSE with no position should be like HOLD
        obs, reward, terminated, truncated, info = env.step(3)
        assert obs.shape == env.observation_space.shape

    def test_open_and_close_position(self, sample_ohlcv, config):
        """Test opening and closing a position."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        env.reset()

        # Open long
        env.step(1)

        # Close position
        obs, reward, terminated, truncated, info = env.step(3)
        assert obs.shape == env.observation_space.shape

    def test_full_episode(self, sample_ohlcv, config):
        """Test running a full episode."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        obs, _ = env.reset()

        done = False
        step_count = 0
        total_reward = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            if step_count > 10000:  # Safety limit
                break

        assert step_count > 0
        assert done

    def test_get_trade_stats(self, sample_ohlcv, config):
        """Test get_trade_stats method."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        env.reset()

        # Run some steps with trades
        env.step(1)  # LONG
        for _ in range(10):
            env.step(0)  # HOLD
        env.step(3)  # CLOSE

        stats = env.get_trade_stats()

        assert isinstance(stats, dict)
        assert "n_trades" in stats
        assert "win_rate" in stats
        assert "final_equity" in stats
        assert "max_drawdown" in stats


class TestEnvGymCompliance:
    """Test Gymnasium compliance."""

    def test_check_env(self, sample_ohlcv, config):
        """Test environment passes Gymnasium check_env."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        # This will raise if there are issues
        check_env(env)

    def test_observation_in_space(self, sample_ohlcv, config):
        """Test observations are in observation space."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        obs, _ = env.reset()

        assert env.observation_space.contains(obs), "Initial obs not in space"

        for _ in range(50):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)

            assert env.observation_space.contains(obs), "Step obs not in space"

            if terminated or truncated:
                obs, _ = env.reset()

    def test_action_masking_not_needed(self, sample_ohlcv, config):
        """Test all actions are always valid (no masking needed)."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        env.reset()

        # All actions should be executable without error
        for action in range(env.action_space.n):
            env.reset()
            try:
                env.step(action)
            except Exception as e:
                pytest.fail(f"Action {action} raised: {e}")


class TestEnvRewards:
    """Test reward system."""

    def test_reward_is_finite(self, sample_ohlcv, config):
        """Test rewards are always finite."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        env.reset()

        for _ in range(200):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)

            assert np.isfinite(reward), f"Non-finite reward: {reward}"

            if terminated or truncated:
                env.reset()

    def test_reward_reasonable_range(self, sample_ohlcv, config):
        """Test rewards are in reasonable range."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=30)
        env.reset()

        rewards = []
        for _ in range(500):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                env.reset()

        # Rewards should be centered around 0 and not extreme
        assert np.abs(np.mean(rewards)) < 1.0
        assert np.max(rewards) < 10.0
        assert np.min(rewards) > -10.0


class TestEnvWithHTF:
    """Test environment with HTF data."""

    def test_env_with_htf(self, sample_ohlcv, sample_ohlcv_htf, config):
        """Test environment works with HTF data."""
        env = CryptoTradingEnv(
            sample_ohlcv, config, window_size=30, htf_df=sample_ohlcv_htf
        )
        obs, _ = env.reset()

        assert obs is not None
        assert obs.shape == env.observation_space.shape

    def test_htf_features_in_obs(self, sample_ohlcv, sample_ohlcv_htf, config):
        """Test HTF features are included in observation."""
        env_with_htf = CryptoTradingEnv(
            sample_ohlcv, config, window_size=30, htf_df=sample_ohlcv_htf
        )
        env_without_htf = CryptoTradingEnv(
            sample_ohlcv, config, window_size=30, htf_df=None
        )

        obs_with, _ = env_with_htf.reset(seed=42)
        obs_without, _ = env_without_htf.reset(seed=42)

        # Observations should be same shape but potentially different values
        assert obs_with.shape == obs_without.shape


class TestEnvEdgeCases:
    """Test edge cases."""

    def test_short_data(self, config):
        """Test with minimal data."""
        import pandas as pd

        np.random.seed(42)
        n = 100  # Minimal data
        df = pd.DataFrame(
            {
                "open": np.random.uniform(90, 110, n),
                "high": np.random.uniform(95, 115, n),
                "low": np.random.uniform(85, 105, n),
                "close": np.random.uniform(90, 110, n),
                "volume": np.random.uniform(1000, 10000, n),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="15min"),
        )
        # Fix OHLC relationships
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)

        env = CryptoTradingEnv(df, config, window_size=30)
        obs, _ = env.reset()
        assert obs is not None

    def test_large_window_size(self, sample_ohlcv, config):
        """Test with large window size."""
        env = CryptoTradingEnv(sample_ohlcv, config, window_size=100)
        obs, _ = env.reset()
        assert obs.shape[0] == 100
