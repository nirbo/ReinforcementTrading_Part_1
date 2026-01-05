"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest

from src.crypto.config import (
    CryptoConfig,
    ExchangeConfig,
    FeeConfig,
    IndicatorConfig,
    RiskConfig,
    TimeframeConfig,
    TradingPairsConfig,
    TrainingConfig,
    load_config,
)


class TestCryptoConfig:
    """Test CryptoConfig dataclass."""

    def test_default_config_loads(self):
        """Test that default config loads without errors."""
        config = load_config()
        assert config is not None
        assert isinstance(config, CryptoConfig)

    def test_config_has_required_sections(self, config):
        """Test all required sections exist."""
        assert isinstance(config.exchange, ExchangeConfig)
        assert isinstance(config.pairs, TradingPairsConfig)
        assert isinstance(config.timeframes, TimeframeConfig)
        assert isinstance(config.fees, FeeConfig)
        assert isinstance(config.risk, RiskConfig)
        assert isinstance(config.indicators, IndicatorConfig)
        assert isinstance(config.training, TrainingConfig)

    def test_exchange_config_defaults(self, config):
        """Test exchange config has sensible defaults."""
        assert config.exchange.name == "bybit"
        assert config.exchange.testnet is True
        assert config.exchange.rate_limit_per_second > 0

    def test_pairs_config(self, config):
        """Test trading pairs configuration."""
        assert len(config.pairs.pairs) > 0
        assert config.pairs.default_pair in config.pairs.pairs
        # All pairs should have correct format
        for pair in config.pairs.pairs:
            assert "/" in pair
            assert "USDT" in pair

    def test_timeframe_config(self, config):
        """Test timeframe configuration."""
        valid_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]
        assert config.timeframes.ltf in valid_timeframes
        assert config.timeframes.htf in valid_timeframes

    def test_risk_config_bounds(self, config):
        """Test risk config has valid bounds."""
        assert 0 < config.risk.max_position_pct <= 1.0
        assert 0 < config.risk.default_sl_pct <= 0.5
        assert 0 < config.risk.default_tp_pct <= 1.0
        assert 1 <= config.risk.max_leverage <= 100

    def test_indicator_config(self, config):
        """Test indicator config has valid values."""
        assert config.indicators.rsi_length > 0
        assert 0 < config.indicators.rsi_oversold < config.indicators.rsi_overbought < 100
        assert config.indicators.atr_length > 0
        assert config.indicators.trend_length > 0

    def test_training_config(self, config):
        """Test training config."""
        assert config.training.total_timesteps > 0
        assert 0 < config.training.learning_rate < 1.0
        assert config.training.n_steps > 0
        assert config.training.batch_size > 0
        assert config.training.window_size > 0

    def test_config_model_dump(self, config):
        """Test config can be dumped to dict."""
        data = config.model_dump()
        assert isinstance(data, dict)
        assert "exchange" in data
        assert "training" in data
        assert "indicators" in data
