"""Tests for evaluation/backtesting module."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.crypto.evaluate import (
    BacktestResult,
    Backtester,
    Trade,
    calculate_calmar_ratio,
    calculate_drawdown_curve,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    compare_to_benchmark,
    generate_report,
)
from src.crypto.train import create_env, generate_synthetic_data


# Use CPU for testing
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestMetricCalculations:
    """Test metric calculation functions."""

    def test_sharpe_ratio_positive(self):
        """Test Sharpe ratio with positive returns."""
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.01] * 20)
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe > 0

    def test_sharpe_ratio_negative(self):
        """Test Sharpe ratio with negative returns."""
        returns = np.array([-0.01, -0.02, -0.01, -0.015, -0.01] * 20)
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe < 0

    def test_sharpe_ratio_mixed(self):
        """Test Sharpe ratio with mixed returns."""
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01] * 20)
        sharpe = calculate_sharpe_ratio(returns)
        assert np.isfinite(sharpe)

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.01] * 20)
        sortino = calculate_sortino_ratio(returns)
        assert np.isfinite(sortino)

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        calmar = calculate_calmar_ratio(
            total_return=0.5,
            max_drawdown=-0.1,
            years=1.0,
        )
        assert calmar > 0

    def test_calmar_ratio_zero_drawdown(self):
        """Test Calmar ratio with zero drawdown."""
        calmar = calculate_calmar_ratio(
            total_return=0.5,
            max_drawdown=0,
            years=1.0,
        )
        assert calmar == 0

    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        equity = np.array([100, 110, 105, 95, 100, 90, 95, 100])
        max_dd, max_dd_pct = calculate_max_drawdown(equity)

        # Max drawdown from 110 to 90 = -20
        assert max_dd < 0
        assert max_dd_pct < 0
        assert max_dd_pct >= -1.0  # Can't lose more than 100%

    def test_max_drawdown_monotonic_up(self):
        """Test max drawdown with monotonically increasing equity."""
        equity = np.array([100, 110, 120, 130, 140])
        max_dd, max_dd_pct = calculate_max_drawdown(equity)
        assert max_dd == 0
        assert max_dd_pct == 0

    def test_drawdown_curve(self):
        """Test drawdown curve calculation."""
        equity = np.array([100, 110, 105, 95, 100, 110])
        dd_curve = calculate_drawdown_curve(equity)

        assert len(dd_curve) == len(equity)
        assert dd_curve[0] == 0  # Start at peak
        assert np.min(dd_curve) < 0  # Some drawdown occurred


class TestTrade:
    """Test Trade dataclass."""

    def test_trade_creation(self):
        """Test Trade creation."""
        from datetime import datetime

        trade = Trade(
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 2),
            direction=1,
            entry_price=100.0,
            exit_price=105.0,
            size=1.0,
            pnl=5.0,
            pnl_pct=0.05,
            bars_held=96,
            exit_reason="tp",
        )

        assert trade.direction == 1
        assert trade.pnl == 5.0


class TestBacktestResult:
    """Test BacktestResult dataclass."""

    def test_backtest_result_to_dict(self):
        """Test BacktestResult serialization."""
        from datetime import datetime

        result = BacktestResult(
            symbol="TEST/USDT",
            timeframe="15m",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            initial_equity=1.0,
            final_equity=1.5,
            total_return=0.5,
            total_return_pct=0.5,
            annualized_return=1.0,
            sharpe_ratio=2.0,
            sortino_ratio=3.0,
            calmar_ratio=5.0,
            max_drawdown=-0.05,
            max_drawdown_pct=-0.05,
            n_trades=100,
            n_winning=60,
            n_losing=40,
            win_rate=0.6,
            profit_factor=1.8,
            avg_win=0.02,
            avg_loss=-0.01,
            avg_trade=0.005,
            avg_bars_held=10.5,
            largest_win=0.1,
            largest_loss=-0.05,
            expectancy=0.008,
        )

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data["symbol"] == "TEST/USDT"
        assert data["sharpe_ratio"] == 2.0
        assert data["n_trades"] == 100

    def test_backtest_result_save(self):
        """Test saving BacktestResult to file."""
        from datetime import datetime

        result = BacktestResult(
            symbol="TEST",
            timeframe="15m",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            initial_equity=1.0,
            final_equity=1.2,
            total_return=0.2,
            total_return_pct=0.2,
            annualized_return=0.4,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=4.0,
            max_drawdown=-0.05,
            max_drawdown_pct=-0.05,
            n_trades=50,
            n_winning=30,
            n_losing=20,
            win_rate=0.6,
            profit_factor=1.5,
            avg_win=0.01,
            avg_loss=-0.005,
            avg_trade=0.004,
            avg_bars_held=8.0,
            largest_win=0.05,
            largest_loss=-0.03,
            expectancy=0.003,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            result.save(path)

            assert path.exists()

            import json
            with open(path) as f:
                loaded = json.load(f)

            assert loaded["symbol"] == "TEST"


class TestBacktester:
    """Test Backtester class."""

    @pytest.fixture
    def trained_model(self, sample_ohlcv, small_config):
        """Create a minimally trained model for testing."""
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
        return model

    def test_backtester_creation(self, trained_model, small_config):
        """Test Backtester creation."""
        backtester = Backtester(trained_model, small_config)
        assert backtester is not None

    def test_backtester_run(self, trained_model, sample_ohlcv, small_config):
        """Test running a backtest."""
        backtester = Backtester(trained_model, small_config)
        result = backtester.run(
            sample_ohlcv.iloc[:200],
            symbol="TEST",
            timeframe="15m",
        )

        assert isinstance(result, BacktestResult)
        assert result.symbol == "TEST"
        assert result.timeframe == "15m"
        assert np.isfinite(result.final_equity)
        assert np.isfinite(result.sharpe_ratio)

    def test_backtester_deterministic(self, trained_model, sample_ohlcv, small_config):
        """Test deterministic backtest gives same results."""
        backtester = Backtester(trained_model, small_config)

        result1 = backtester.run(
            sample_ohlcv.iloc[:100],
            deterministic=True,
        )
        result2 = backtester.run(
            sample_ohlcv.iloc[:100],
            deterministic=True,
        )

        assert result1.final_equity == result2.final_equity
        assert result1.n_trades == result2.n_trades


class TestCompareTooBenchmark:
    """Test benchmark comparison."""

    def test_compare_to_benchmark(self, sample_ohlcv):
        """Test compare_to_benchmark function."""
        from datetime import datetime

        # Create a mock result
        result = BacktestResult(
            symbol="TEST",
            timeframe="15m",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            initial_equity=1.0,
            final_equity=1.1,
            total_return=0.1,
            total_return_pct=0.1,
            annualized_return=0.2,
            sharpe_ratio=1.0,
            sortino_ratio=1.5,
            calmar_ratio=2.0,
            max_drawdown=-0.05,
            max_drawdown_pct=-0.05,
            n_trades=20,
            n_winning=12,
            n_losing=8,
            win_rate=0.6,
            profit_factor=1.5,
            avg_win=0.01,
            avg_loss=-0.005,
            avg_trade=0.005,
            avg_bars_held=10.0,
            largest_win=0.03,
            largest_loss=-0.02,
            expectancy=0.004,
        )

        comparison = compare_to_benchmark(result, sample_ohlcv)

        assert isinstance(comparison, dict)
        assert "strategy_return" in comparison
        assert "benchmark_return" in comparison
        assert "excess_return" in comparison
        assert "outperformed" in comparison


class TestGenerateReport:
    """Test report generation."""

    def test_generate_report(self):
        """Test generate_report function."""
        from datetime import datetime

        result = BacktestResult(
            symbol="SOL/USDT",
            timeframe="15m",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            initial_equity=1.0,
            final_equity=1.25,
            total_return=0.25,
            total_return_pct=0.25,
            annualized_return=0.5,
            sharpe_ratio=1.8,
            sortino_ratio=2.5,
            calmar_ratio=4.0,
            max_drawdown=-0.06,
            max_drawdown_pct=-0.06,
            n_trades=150,
            n_winning=90,
            n_losing=60,
            win_rate=0.6,
            profit_factor=1.8,
            avg_win=0.008,
            avg_loss=-0.004,
            avg_trade=0.00167,
            avg_bars_held=12.5,
            largest_win=0.08,
            largest_loss=-0.03,
            expectancy=0.0032,
        )

        report = generate_report(result)

        assert isinstance(report, str)
        assert "SOL/USDT" in report
        assert "Sharpe Ratio" in report
        assert "Win Rate" in report

    def test_generate_report_with_benchmark(self):
        """Test report with benchmark comparison."""
        from datetime import datetime

        result = BacktestResult(
            symbol="TEST",
            timeframe="15m",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 1),
            initial_equity=1.0,
            final_equity=1.1,
            total_return=0.1,
            total_return_pct=0.1,
            annualized_return=0.4,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=3.0,
            max_drawdown=-0.03,
            max_drawdown_pct=-0.03,
            n_trades=30,
            n_winning=18,
            n_losing=12,
            win_rate=0.6,
            profit_factor=1.6,
            avg_win=0.01,
            avg_loss=-0.005,
            avg_trade=0.0033,
            avg_bars_held=8.0,
            largest_win=0.04,
            largest_loss=-0.02,
            expectancy=0.003,
        )

        benchmark = {
            "strategy_return": 0.1,
            "benchmark_return": 0.05,
            "excess_return": 0.05,
            "strategy_sharpe": 1.5,
            "benchmark_sharpe": 0.8,
            "strategy_max_dd": -0.03,
            "benchmark_max_dd": -0.08,
            "outperformed": True,
        }

        report = generate_report(result, benchmark)

        assert "BENCHMARK COMPARISON" in report
        assert "Outperformed" in report
