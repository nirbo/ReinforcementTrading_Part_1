"""
Backtesting and evaluation engine for crypto RL trading system.

Features:
- Vectorized backtest execution (fast)
- Comprehensive performance metrics (Sharpe, Sortino, Calmar, etc.)
- Trade analysis and statistics
- Walk-forward validation support
- Equity curve analysis
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.crypto.config import CryptoConfig, load_config
from src.crypto.trading_env import CryptoTradingEnv

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""

    entry_time: datetime
    exit_time: datetime
    direction: int  # 1=long, -1=short
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    bars_held: int
    exit_reason: str = "close"  # close, sl, tp, timeout


@dataclass
class BacktestResult:
    """Complete backtest results."""

    # Basic info
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_equity: float

    # Performance metrics
    final_equity: float
    total_return: float
    total_return_pct: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float

    # Trade statistics
    n_trades: int
    n_winning: int
    n_losing: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    avg_bars_held: float
    largest_win: float
    largest_loss: float
    expectancy: float

    # Time series
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "n_trades": self.n_trades,
            "n_winning": self.n_winning,
            "n_losing": self.n_losing,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_trade": self.avg_trade,
            "avg_bars_held": self.avg_bars_held,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "expectancy": self.expectancy,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 24 * 4,  # 15-min bars
) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)

    if std_return == 0:
        return 0.0

    sharpe = mean_return / std_return * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 24 * 4,
) -> float:
    """Calculate annualized Sortino ratio (downside deviation)."""
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = np.mean(excess_returns)

    # Downside deviation
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return float("inf") if mean_return > 0 else 0.0

    downside_std = np.sqrt(np.mean(negative_returns**2))

    if downside_std == 0:
        return 0.0

    sortino = mean_return / downside_std * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_calmar_ratio(
    total_return: float,
    max_drawdown: float,
    years: float,
) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown)."""
    if max_drawdown == 0 or years == 0:
        return 0.0

    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    calmar = annualized_return / abs(max_drawdown)
    return float(calmar)


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, float]:
    """
    Calculate maximum drawdown.

    Returns:
        (max_drawdown_value, max_drawdown_pct)
    """
    if len(equity_curve) < 2:
        return 0.0, 0.0

    peak = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - peak
    drawdown_pct = drawdown / peak

    max_dd = float(np.min(drawdown))
    max_dd_pct = float(np.min(drawdown_pct))

    return max_dd, max_dd_pct


def calculate_drawdown_curve(equity_curve: np.ndarray) -> np.ndarray:
    """Calculate drawdown at each point."""
    peak = np.maximum.accumulate(equity_curve)
    return (equity_curve - peak) / peak


class Backtester:
    """
    Backtesting engine for evaluating trained models.

    Supports:
    - Single-pass evaluation
    - Walk-forward validation
    - Monte Carlo simulation
    """

    def __init__(
        self,
        model: PPO,
        config: Optional[CryptoConfig] = None,
        slippage_pct: float = 0.0001,  # 0.01% slippage
    ):
        """
        Initialize backtester.

        Args:
            model: Trained PPO model
            config: Configuration object
            slippage_pct: Slippage as percentage of price
        """
        self.model = model
        self.config = config or load_config()
        self.slippage_pct = slippage_pct

    def run(
        self,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
        symbol: str = "UNKNOWN",
        timeframe: str = "15m",
        deterministic: bool = True,
    ) -> BacktestResult:
        """
        Run backtest on data.

        Args:
            df: OHLCV DataFrame
            htf_df: Optional HTF DataFrame
            symbol: Symbol name for reporting
            timeframe: Timeframe for reporting
            deterministic: Use deterministic policy

        Returns:
            BacktestResult with full analysis
        """
        # Create environment
        env = CryptoTradingEnv(
            df=df,
            config=self.config,
            window_size=self.config.training.window_size,
            random_start=False,
            htf_df=htf_df,
        )

        # Run episode
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Get environment stats
        env_stats = env.get_trade_stats()
        trades = self._extract_trades(env)

        # Calculate performance metrics
        equity_curve = np.array(env.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        initial_equity = equity_curve[0]
        final_equity = equity_curve[-1]
        total_return = final_equity - initial_equity
        total_return_pct = total_return / initial_equity if initial_equity > 0 else 0

        # Time calculations
        if df.index.dtype == "datetime64[ns]" or hasattr(df.index[0], "timestamp"):
            start_date = df.index[0].to_pydatetime() if hasattr(df.index[0], "to_pydatetime") else df.index[0]
            end_date = df.index[-1].to_pydatetime() if hasattr(df.index[-1], "to_pydatetime") else df.index[-1]
            days = (end_date - start_date).total_seconds() / 86400
            years = days / 365.25
        else:
            start_date = datetime.now()
            end_date = datetime.now()
            years = len(df) / (252 * 24 * 4)  # Approximate

        # Annualized return
        annualized_return = (1 + total_return_pct) ** (1 / years) - 1 if years > 0 else 0

        # Risk metrics
        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        max_dd, max_dd_pct = calculate_max_drawdown(equity_curve)
        calmar = calculate_calmar_ratio(total_return_pct, max_dd_pct, years)
        drawdown_curve = calculate_drawdown_curve(equity_curve)

        # Trade statistics
        n_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        n_winning = len(winning_trades)
        n_losing = len(losing_trades)
        win_rate = n_winning / n_trades if n_trades > 0 else 0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = gross_profit / n_winning if n_winning > 0 else 0
        avg_loss = -gross_loss / n_losing if n_losing > 0 else 0
        avg_trade = total_return / n_trades if n_trades > 0 else 0
        avg_bars_held = np.mean([t.bars_held for t in trades]) if trades else 0

        largest_win = max([t.pnl for t in trades]) if trades else 0
        largest_loss = min([t.pnl for t in trades]) if trades else 0

        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) if n_trades > 0 else 0

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_equity=initial_equity,
            final_equity=final_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            n_trades=n_trades,
            n_winning=n_winning,
            n_losing=n_losing,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            avg_bars_held=avg_bars_held,
            largest_win=largest_win,
            largest_loss=largest_loss,
            expectancy=expectancy,
            equity_curve=equity_curve.tolist(),
            drawdown_curve=drawdown_curve.tolist(),
            trades=trades,
        )

    def _extract_trades(self, env: CryptoTradingEnv) -> List[Trade]:
        """Extract trade list from environment history."""
        trades = []
        trade_history = getattr(env, "trade_history", [])

        for th in trade_history:
            trades.append(
                Trade(
                    entry_time=th.get("entry_time", datetime.now()),
                    exit_time=th.get("exit_time", datetime.now()),
                    direction=th.get("direction", 0),
                    entry_price=th.get("entry_price", 0),
                    exit_price=th.get("exit_price", 0),
                    size=th.get("size", 0),
                    pnl=th.get("pnl", 0),
                    pnl_pct=th.get("pnl_pct", 0),
                    bars_held=th.get("bars_held", 0),
                    exit_reason=th.get("exit_reason", "close"),
                )
            )

        return trades

    def walk_forward(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        train_pct: float = 0.7,
        htf_df: Optional[pd.DataFrame] = None,
        symbol: str = "UNKNOWN",
        timeframe: str = "15m",
    ) -> List[BacktestResult]:
        """
        Run walk-forward validation.

        Args:
            df: Full dataset
            n_splits: Number of walk-forward periods
            train_pct: Training period percentage
            htf_df: Optional HTF data
            symbol: Symbol name
            timeframe: Timeframe

        Returns:
            List of BacktestResult for each test period
        """
        results = []
        n_samples = len(df)
        split_size = n_samples // n_splits

        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size
            if i == n_splits - 1:
                end_idx = n_samples

            # Split into train/test within this fold
            train_end = start_idx + int((end_idx - start_idx) * train_pct)
            test_df = df.iloc[train_end:end_idx]

            if len(test_df) < 100:
                continue

            # Get HTF data for test period if available
            test_htf = None
            if htf_df is not None:
                test_htf = htf_df.loc[test_df.index[0] : test_df.index[-1]]

            logger.info(f"Walk-forward fold {i + 1}/{n_splits}: testing on {len(test_df)} bars")

            result = self.run(
                test_df,
                htf_df=test_htf,
                symbol=symbol,
                timeframe=timeframe,
            )
            results.append(result)

        return results

    def monte_carlo(
        self,
        df: pd.DataFrame,
        n_simulations: int = 100,
        htf_df: Optional[pd.DataFrame] = None,
        symbol: str = "UNKNOWN",
        timeframe: str = "15m",
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation by shuffling trade order.

        Args:
            df: Test data
            n_simulations: Number of MC simulations
            htf_df: Optional HTF data
            symbol: Symbol name
            timeframe: Timeframe

        Returns:
            Dictionary with MC statistics
        """
        # First, run standard backtest to get trades
        base_result = self.run(df, htf_df, symbol, timeframe)

        if len(base_result.trades) < 5:
            logger.warning("Not enough trades for Monte Carlo simulation")
            return {"error": "Not enough trades"}

        # Extract trade PnLs
        trade_pnls = [t.pnl for t in base_result.trades]

        # Run simulations
        final_equities = []
        max_drawdowns = []
        sharpe_ratios = []

        for _ in range(n_simulations):
            # Shuffle trade order
            shuffled_pnls = np.random.permutation(trade_pnls)

            # Calculate equity curve
            equity = [base_result.initial_equity]
            for pnl in shuffled_pnls:
                equity.append(equity[-1] + pnl)

            equity = np.array(equity)
            final_equities.append(equity[-1])

            # Calculate metrics
            _, max_dd_pct = calculate_max_drawdown(equity)
            max_drawdowns.append(max_dd_pct)

            returns = np.diff(equity) / equity[:-1]
            sharpe = calculate_sharpe_ratio(returns)
            sharpe_ratios.append(sharpe)

        return {
            "n_simulations": n_simulations,
            "base_final_equity": base_result.final_equity,
            "mc_final_equity_mean": float(np.mean(final_equities)),
            "mc_final_equity_std": float(np.std(final_equities)),
            "mc_final_equity_5th": float(np.percentile(final_equities, 5)),
            "mc_final_equity_95th": float(np.percentile(final_equities, 95)),
            "mc_max_drawdown_mean": float(np.mean(max_drawdowns)),
            "mc_max_drawdown_95th": float(np.percentile(max_drawdowns, 95)),
            "mc_sharpe_mean": float(np.mean(sharpe_ratios)),
            "mc_sharpe_std": float(np.std(sharpe_ratios)),
        }


def compare_to_benchmark(
    result: BacktestResult,
    df: pd.DataFrame,
    initial_equity: float = 1.0,
) -> Dict[str, Any]:
    """
    Compare backtest result to buy-and-hold benchmark.

    Args:
        result: Backtest result
        df: OHLCV data
        initial_equity: Starting equity

    Returns:
        Dictionary with comparison metrics
    """
    # Buy and hold equity curve
    first_close = df["close"].iloc[0]
    bh_equity = initial_equity * (df["close"] / first_close)

    bh_returns = df["close"].pct_change().dropna().values
    bh_sharpe = calculate_sharpe_ratio(bh_returns)
    bh_total_return = (bh_equity.iloc[-1] - initial_equity) / initial_equity
    bh_max_dd, bh_max_dd_pct = calculate_max_drawdown(bh_equity.values)

    return {
        "strategy_return": result.total_return_pct,
        "benchmark_return": float(bh_total_return),
        "excess_return": result.total_return_pct - bh_total_return,
        "strategy_sharpe": result.sharpe_ratio,
        "benchmark_sharpe": float(bh_sharpe),
        "strategy_max_dd": result.max_drawdown_pct,
        "benchmark_max_dd": float(bh_max_dd_pct),
        "outperformed": result.total_return_pct > bh_total_return,
    }


def generate_report(
    result: BacktestResult,
    benchmark_comparison: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate text report from backtest result.

    Args:
        result: Backtest result
        benchmark_comparison: Optional benchmark comparison

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        f"BACKTEST REPORT: {result.symbol} ({result.timeframe})",
        "=" * 60,
        "",
        "PERIOD",
        "-" * 40,
        f"Start:           {result.start_date}",
        f"End:             {result.end_date}",
        "",
        "PERFORMANCE",
        "-" * 40,
        f"Initial Equity:  {result.initial_equity:.4f}",
        f"Final Equity:    {result.final_equity:.4f}",
        f"Total Return:    {result.total_return_pct:.2%}",
        f"Annualized:      {result.annualized_return:.2%}",
        "",
        "RISK METRICS",
        "-" * 40,
        f"Sharpe Ratio:    {result.sharpe_ratio:.3f}",
        f"Sortino Ratio:   {result.sortino_ratio:.3f}",
        f"Calmar Ratio:    {result.calmar_ratio:.3f}",
        f"Max Drawdown:    {result.max_drawdown_pct:.2%}",
        "",
        "TRADE STATISTICS",
        "-" * 40,
        f"Total Trades:    {result.n_trades}",
        f"Win Rate:        {result.win_rate:.2%}",
        f"Profit Factor:   {result.profit_factor:.2f}",
        f"Avg Win:         {result.avg_win:.6f}",
        f"Avg Loss:        {result.avg_loss:.6f}",
        f"Avg Trade:       {result.avg_trade:.6f}",
        f"Avg Bars Held:   {result.avg_bars_held:.1f}",
        f"Largest Win:     {result.largest_win:.6f}",
        f"Largest Loss:    {result.largest_loss:.6f}",
        f"Expectancy:      {result.expectancy:.6f}",
    ]

    if benchmark_comparison:
        lines.extend([
            "",
            "BENCHMARK COMPARISON (Buy & Hold)",
            "-" * 40,
            f"Strategy Return: {benchmark_comparison['strategy_return']:.2%}",
            f"Benchmark Return:{benchmark_comparison['benchmark_return']:.2%}",
            f"Excess Return:   {benchmark_comparison['excess_return']:.2%}",
            f"Strategy Sharpe: {benchmark_comparison['strategy_sharpe']:.3f}",
            f"Benchmark Sharpe:{benchmark_comparison['benchmark_sharpe']:.3f}",
            f"Outperformed:    {'Yes' if benchmark_comparison['outperformed'] else 'No'}",
        ])

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def run_full_evaluation(
    model_path: str,
    df: pd.DataFrame,
    config: Optional[CryptoConfig] = None,
    htf_df: Optional[pd.DataFrame] = None,
    symbol: str = "UNKNOWN",
    timeframe: str = "15m",
    output_dir: Optional[str] = None,
    run_monte_carlo: bool = True,
) -> BacktestResult:
    """
    Run full evaluation pipeline.

    Args:
        model_path: Path to trained model
        df: Test data
        config: Configuration
        htf_df: Optional HTF data
        symbol: Symbol name
        timeframe: Timeframe
        output_dir: Directory for outputs
        run_monte_carlo: Whether to run MC simulation

    Returns:
        BacktestResult
    """
    config = config or load_config()

    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Create backtester
    backtester = Backtester(model, config)

    # Run backtest
    logger.info("Running backtest...")
    result = backtester.run(df, htf_df, symbol, timeframe)

    # Compare to benchmark
    benchmark = compare_to_benchmark(result, df)

    # Generate report
    report = generate_report(result, benchmark)
    logger.info("\n" + report)

    # Monte Carlo simulation
    mc_results = None
    if run_monte_carlo and result.n_trades >= 10:
        logger.info("Running Monte Carlo simulation...")
        mc_results = backtester.monte_carlo(df, n_simulations=100, htf_df=htf_df)
        logger.info(f"MC Results: mean equity={mc_results['mc_final_equity_mean']:.4f}, "
                    f"5th percentile={mc_results['mc_final_equity_5th']:.4f}")

    # Save outputs
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        result.save(output_path / "backtest_result.json")

        with open(output_path / "report.txt", "w") as f:
            f.write(report)

        with open(output_path / "benchmark.json", "w") as f:
            json.dump(benchmark, f, indent=2)

        if mc_results:
            with open(output_path / "monte_carlo.json", "w") as f:
                json.dump(mc_results, f, indent=2)

        logger.info(f"Results saved to: {output_dir}")

    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate crypto RL trading model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data", type=str, default=None, help="Path to test data (parquet)")
    parser.add_argument("--symbol", type=str, default="SOL/USDT:USDT", help="Symbol name")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe")
    parser.add_argument("--output", type=str, default="evaluation", help="Output directory")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    config = load_config(args.config)

    # Load or generate data
    if args.synthetic:
        from src.crypto.train import generate_synthetic_data
        logger.info("Using synthetic data for evaluation")
        df = generate_synthetic_data(3000, seed=123)
    elif args.data:
        logger.info(f"Loading data from: {args.data}")
        df = pd.read_parquet(args.data)
    else:
        from src.crypto.data_manager import DataManager
        dm = DataManager(config)
        df = dm.load_ohlcv(args.symbol, args.timeframe)

        if df.empty:
            logger.warning("No data found, using synthetic")
            from src.crypto.train import generate_synthetic_data
            df = generate_synthetic_data(3000, seed=123)

    result = run_full_evaluation(
        model_path=args.model,
        df=df,
        config=config,
        symbol=args.symbol,
        timeframe=args.timeframe,
        output_dir=args.output,
    )
