"""Pytest fixtures for crypto RL trading tests."""

import numpy as np
import pandas as pd
import pytest

from src.crypto.config import CryptoConfig, load_config


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500

    returns = np.random.randn(n) * 0.02
    close = pd.Series(100 * np.exp(np.cumsum(returns)))
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + np.random.randn(n) * 0.002)
    volume = np.abs(np.random.randn(n) * 1000 + 5000)

    index = pd.date_range(start="2024-01-01", periods=n, freq="15min", tz="UTC")

    return pd.DataFrame(
        {
            "open": open_.values,
            "high": high.values,
            "low": low.values,
            "close": close.values,
            "volume": volume,
        },
        index=index,
    )


@pytest.fixture
def sample_ohlcv_htf() -> pd.DataFrame:
    """Generate sample HTF (1h) OHLCV data for testing."""
    np.random.seed(123)
    n = 125  # 500 15-min bars = 125 1-hour bars

    returns = np.random.randn(n) * 0.03
    close = pd.Series(100 * np.exp(np.cumsum(returns)))
    high = close * (1 + np.abs(np.random.randn(n) * 0.015))
    low = close * (1 - np.abs(np.random.randn(n) * 0.015))
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + np.random.randn(n) * 0.003)
    volume = np.abs(np.random.randn(n) * 4000 + 20000)

    index = pd.date_range(start="2024-01-01", periods=n, freq="1h", tz="UTC")

    return pd.DataFrame(
        {
            "open": open_.values,
            "high": high.values,
            "low": low.values,
            "close": close.values,
            "volume": volume,
        },
        index=index,
    )


@pytest.fixture
def config() -> CryptoConfig:
    """Load default configuration."""
    return load_config()


@pytest.fixture
def small_config() -> CryptoConfig:
    """Configuration with smaller values for faster testing."""
    config = load_config()
    config.training.window_size = 10
    config.training.total_timesteps = 100
    config.training.n_steps = 32
    config.training.batch_size = 16
    return config
