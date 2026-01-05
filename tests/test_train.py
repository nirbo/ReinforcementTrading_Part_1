"""Tests for training module."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO

from src.crypto.train import (
    TradingMetricsCallback,
    create_env,
    evaluate_model,
    generate_synthetic_data,
)


# Use CPU for testing
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestGenerateSyntheticData:
    """Test synthetic data generation."""

    def test_generate_synthetic_data(self):
        """Test basic synthetic data generation."""
        df = generate_synthetic_data(1000, seed=42)

        assert len(df) == 1000
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_synthetic_data_reproducible(self):
        """Test synthetic data is reproducible with seed."""
        df1 = generate_synthetic_data(100, seed=42)
        df2 = generate_synthetic_data(100, seed=42)

        np.testing.assert_array_equal(df1["close"].values, df2["close"].values)

    def test_synthetic_data_positive_volume(self):
        """Test volume is positive."""
        df = generate_synthetic_data(1000, seed=42)
        assert (df["volume"] > 0).all()


class TestCreateEnv:
    """Test environment creation helper."""

    def test_create_train_env(self, sample_ohlcv, config):
        """Test creating training environment."""
        env = create_env(sample_ohlcv, config, train=True)
        assert env is not None

    def test_create_eval_env(self, sample_ohlcv, config):
        """Test creating evaluation environment."""
        env = create_env(sample_ohlcv, config, train=False)
        assert env is not None

    def test_create_env_with_htf(self, sample_ohlcv, sample_ohlcv_htf, config):
        """Test creating environment with HTF data."""
        env = create_env(sample_ohlcv, config, train=True, htf_df=sample_ohlcv_htf)
        assert env is not None


class TestTradingMetricsCallback:
    """Test trading metrics callback."""

    def test_callback_creation(self, sample_ohlcv, config):
        """Test callback creation."""
        env = create_env(sample_ohlcv, config, train=False)
        callback = TradingMetricsCallback(eval_env=env, eval_freq=100)

        assert callback.eval_freq == 100
        assert callback.best_equity == 0.0


class TestEvaluateModel:
    """Test model evaluation function."""

    def test_evaluate_model(self, sample_ohlcv, small_config):
        """Test model evaluation."""
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        # Create and train minimal model
        def make_env():
            env = create_env(sample_ohlcv, small_config, train=True)
            return Monitor(env)

        vec_env = DummyVecEnv([make_env])
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=32,
            batch_size=16,
            n_epochs=1,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=64, progress_bar=False)

        # Evaluate
        stats = evaluate_model(
            model,
            sample_ohlcv.iloc[:200],  # Use subset for speed
            config=small_config,
            deterministic=True,
        )

        assert isinstance(stats, dict)
        assert "n_trades" in stats
        assert "final_equity" in stats
        assert "total_reward" in stats


class TestModelPrediction:
    """Test model predictions are valid."""

    def test_model_predicts_valid_actions(self, sample_ohlcv, small_config):
        """Test model predicts valid action indices."""
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        def make_env():
            env = create_env(sample_ohlcv, small_config, train=True)
            return Monitor(env)

        vec_env = DummyVecEnv([make_env])
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=32,
            batch_size=16,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=64, progress_bar=False)

        # Test predictions
        env = create_env(sample_ohlcv, small_config, train=False)
        obs, _ = env.reset()

        for _ in range(50):
            action, _ = model.predict(obs, deterministic=True)

            # Action should be valid
            assert 0 <= action < env.action_space.n

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
