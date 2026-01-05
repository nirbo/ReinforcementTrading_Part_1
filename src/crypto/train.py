"""
PPO training pipeline for crypto RL trading system.

Uses Stable-Baselines3 for training with:
- Checkpointing and TensorBoard logging
- Train/validation split for OOS evaluation
- Custom callbacks for trading metrics
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.crypto.config import CryptoConfig, load_config
from src.crypto.trading_env import CryptoTradingEnv

logger = logging.getLogger(__name__)


class TradingMetricsCallback(BaseCallback):
    """Callback to log trading-specific metrics."""

    def __init__(self, eval_env: CryptoTradingEnv, eval_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_equity = 0.0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation episode
            obs, _ = self.eval_env.reset()
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated

            stats = self.eval_env.unwrapped.get_trade_stats()

            # Log to TensorBoard
            self.logger.record("eval/n_trades", stats["n_trades"])
            self.logger.record("eval/win_rate", stats["win_rate"])
            self.logger.record("eval/final_equity", stats["final_equity"])
            self.logger.record("eval/max_drawdown", stats["max_drawdown"])
            self.logger.record("eval/avg_pnl", stats["avg_pnl"])

            if stats.get("profit_factor", 0) < float("inf"):
                self.logger.record("eval/profit_factor", stats["profit_factor"])

            # Track best model
            if stats["final_equity"] > self.best_equity:
                self.best_equity = stats["final_equity"]
                self.logger.record("eval/best_equity", self.best_equity)

            if self.verbose:
                print(f"\n[Eval] Trades={stats['n_trades']}, WinRate={stats['win_rate']:.2%}, "
                      f"Equity={stats['final_equity']:.4f}, MaxDD={stats['max_drawdown']:.2%}")

        return True


def create_env(
    df: pd.DataFrame,
    config: CryptoConfig,
    train: bool = True,
    htf_df: Optional[pd.DataFrame] = None,
) -> CryptoTradingEnv:
    """Create trading environment with appropriate settings."""
    return CryptoTradingEnv(
        df=df,
        config=config,
        window_size=config.training.window_size,
        random_start=train,  # Random starts only for training
        min_episode_bars=500,
        max_episode_bars=config.training.episode_max_steps if train else None,
        htf_df=htf_df,
    )


def train_ppo(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Optional[CryptoConfig] = None,
    output_dir: str = "models",
    htf_train_df: Optional[pd.DataFrame] = None,
    htf_val_df: Optional[pd.DataFrame] = None,
    resume_path: Optional[str] = None,
) -> Tuple[PPO, Dict[str, Any]]:
    """
    Train PPO agent on crypto trading environment.

    Args:
        train_df: Training data (OHLCV)
        val_df: Validation data (OHLCV)
        config: Configuration object
        output_dir: Directory for checkpoints and logs
        htf_train_df: Optional HTF training data
        htf_val_df: Optional HTF validation data
        resume_path: Path to resume training from

    Returns:
        Tuple of (Trained PPO model, final validation stats)
    """
    config = config or load_config()
    training = config.training

    # Create output directories
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"

    for d in [run_dir, checkpoint_dir, tensorboard_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save config
    config.to_yaml(run_dir / "config.yaml")

    logger.info(f"Training run: {run_name}")
    logger.info(f"Output directory: {run_dir}")

    # Create environments
    def make_train_env():
        env = create_env(train_df, config, train=True, htf_df=htf_train_df)
        return Monitor(env)

    def make_val_env():
        env = create_env(val_df, config, train=False, htf_df=htf_val_df)
        return Monitor(env)

    train_vec_env = DummyVecEnv([make_train_env])
    val_env = make_val_env()

    # Create or load model
    if resume_path and Path(resume_path).exists():
        logger.info(f"Resuming from: {resume_path}")
        model = PPO.load(resume_path, env=train_vec_env)
    else:
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        model = PPO(
            policy="MlpPolicy",
            env=train_vec_env,
            learning_rate=training.learning_rate,
            n_steps=training.n_steps,
            batch_size=training.batch_size,
            n_epochs=training.n_epochs,
            gamma=training.gamma,
            gae_lambda=training.gae_lambda,
            clip_range=training.clip_range,
            ent_coef=training.ent_coef,
            vf_coef=training.vf_coef,
            max_grad_norm=training.max_grad_norm,
            tensorboard_log=str(tensorboard_dir),
            verbose=1,
            seed=training.seed,
            device=device,
        )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=training.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_crypto",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    trading_callback = TradingMetricsCallback(
        eval_env=val_env,
        eval_freq=training.eval_freq,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_callback, trading_callback])

    # Train
    logger.info(f"Starting training for {training.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=training.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = run_dir / "ppo_crypto_final.zip"
    model.save(str(final_path))
    logger.info(f"Final model saved: {final_path}")

    # Final evaluation
    logger.info("Final evaluation on validation set...")
    obs, _ = val_env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = val_env.step(action)
        done = terminated or truncated

    final_stats = val_env.unwrapped.get_trade_stats()
    logger.info(f"Final stats: {final_stats}")

    # Save stats
    import json
    with open(run_dir / "final_stats.json", "w") as f:
        json.dump(final_stats, f, indent=2, default=str)

    return model, final_stats


def evaluate_model(
    model: PPO,
    df: pd.DataFrame,
    config: Optional[CryptoConfig] = None,
    htf_df: Optional[pd.DataFrame] = None,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate trained model on data.

    Returns:
        Trade statistics dictionary
    """
    config = config or load_config()
    env = create_env(df, config, train=False, htf_df=htf_df)

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    stats = env.get_trade_stats()
    stats["total_reward"] = total_reward
    stats["equity_curve"] = env.equity_curve

    return stats


def generate_synthetic_data(n_bars: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)

    # Random walk with trend
    returns = np.random.randn(n_bars) * 0.02 + 0.0001  # Slight upward drift
    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.01))
    open_ = close.copy()
    open_[1:] = close[:-1] * (1 + np.random.randn(n_bars - 1) * 0.002)
    open_[0] = close[0]

    # Volume with some autocorrelation
    base_volume = 10000
    volume = base_volume * (1 + 0.5 * np.random.randn(n_bars))
    volume = np.abs(volume)

    # Create datetime index
    index = pd.date_range(start="2023-01-01", periods=n_bars, freq="15min")

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=index)


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train crypto RL trading agent")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to OHLCV parquet file")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume from")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Load config
    config = load_config(args.config)

    # Load or generate data
    if args.synthetic:
        logger.info("Using synthetic data for testing...")
        df = generate_synthetic_data(10000)
    elif args.data:
        logger.info(f"Loading data from: {args.data}")
        df = pd.read_parquet(args.data)
    else:
        # Try to load from default location
        from src.crypto.data_manager import DataManager
        dm = DataManager(config)
        symbol = config.pairs.default_pair
        timeframe = config.timeframes.ltf
        df = dm.load_ohlcv(symbol, timeframe)

        if df.empty:
            logger.warning("No data found, using synthetic data")
            df = generate_synthetic_data(10000)

    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Train/validation split (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    logger.info(f"Training bars: {len(train_df)}")
    logger.info(f"Validation bars: {len(val_df)}")

    # Train
    model = train_ppo(
        train_df=train_df,
        val_df=val_df,
        config=config,
        output_dir=args.output,
        resume_path=args.resume,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
