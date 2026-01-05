"""
Trading environment for crypto RL system.

Gymnasium-compatible environment for crypto perpetual futures trading.
Supports 6 discrete actions with dynamic fees and percentage-based SL/TP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.crypto.config import CryptoConfig, load_config
from src.crypto.feature_extractor import FeatureExtractor, PositionState
from src.crypto.indicators import compute_all_indicators


class Action(IntEnum):
    """Trading actions."""
    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3
    # TIGHT actions removed - 1% SL triggers on noise, 38% WR vs 51% for normal


@dataclass
class Trade:
    """Trade record."""
    entry_bar: int
    entry_price: float
    direction: int  # 1=long, -1=short
    sl_price: float
    tp_price: float
    sl_pct: float
    tp_pct: float
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: float = 0.0
    fees_pct: float = 0.0
    net_pnl_pct: float = 0.0


class CryptoTradingEnv(gym.Env):
    """
    Crypto perpetual futures trading environment.

    Observation: Rolling window of features + position state
    Actions: 6 discrete actions (HOLD, LONG, SHORT, CLOSE, LONG_TIGHT, SHORT_TIGHT)
    Reward: Realized PnL (percentage) minus fees, with optional shaping
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[CryptoConfig] = None,
        window_size: int = 30,
        taker_fee: float = 0.00055,  # 0.055%
        random_start: bool = True,
        min_episode_bars: int = 500,
        max_episode_bars: Optional[int] = 2000,
        reward_scale: float = 100.0,  # Scale rewards for better learning
        unrealized_pnl_weight: float = 0.1,  # Weight for unrealized PnL shaping
        hold_penalty: float = 0.0,  # Penalty per bar for being flat (encourage trading)
        htf_df: Optional[pd.DataFrame] = None,
    ):
        super().__init__()

        self.config = config or load_config()
        self.window_size = window_size
        self.taker_fee = taker_fee
        self.random_start = random_start
        self.min_episode_bars = min_episode_bars
        self.max_episode_bars = max_episode_bars
        self.reward_scale = reward_scale
        self.unrealized_pnl_weight = unrealized_pnl_weight
        self.hold_penalty = hold_penalty

        # Precompute indicators
        self.df = compute_all_indicators(df.copy(), self.config.indicators.model_dump())
        self.htf_df = htf_df
        self.n_bars = len(self.df)

        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            config=self.config,
            window_size=window_size,
        )

        # Precompute features for efficiency
        self._precomputed_features = self.feature_extractor.precompute_features(
            self.df, htf_df=self.htf_df
        )

        # Action space: 4 discrete actions (HOLD, LONG, SHORT, CLOSE)
        self.action_space = spaces.Discrete(4)

        # Observation space: (window_size, feature_dim)
        feature_dim = self._precomputed_features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, feature_dim + 3),  # +3 for position state
            dtype=np.float32,
        )

        # SL/TP settings
        self.default_sl_pct = self.config.risk.default_sl_pct
        self.default_tp_pct = self.config.risk.default_tp_pct
        self.tight_sl_pct = self.config.risk.tight_sl_pct
        self.tight_tp_pct = self.config.risk.tight_tp_pct

        # Initialize state
        self._reset_state()

    def _reset_state(self):
        """Reset internal state."""
        self.current_bar = 0
        self.bars_in_episode = 0
        self.terminated = False
        self.truncated = False

        # Position state
        self.position = PositionState()
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None

        # Equity tracking
        self.initial_equity = 1.0  # Start with 100%
        self.equity = self.initial_equity
        self.equity_curve: List[float] = []

        # Previous unrealized for shaping
        self._prev_unrealized_pnl = 0.0

    def _get_price(self, bar: Optional[int] = None) -> float:
        """Get close price at bar."""
        if bar is None:
            bar = self.current_bar
        return float(self.df.iloc[bar]["close"])

    def _get_high(self, bar: Optional[int] = None) -> float:
        """Get high price at bar."""
        if bar is None:
            bar = self.current_bar
        return float(self.df.iloc[bar]["high"])

    def _get_low(self, bar: Optional[int] = None) -> float:
        """Get low price at bar."""
        if bar is None:
            bar = self.current_bar
        return float(self.df.iloc[bar]["low"])

    def _compute_unrealized_pnl(self) -> float:
        """Compute unrealized PnL as percentage."""
        if self.position.direction == 0 or self.current_trade is None:
            return 0.0

        current_price = self._get_price()
        entry_price = self.current_trade.entry_price

        if self.position.direction == 1:  # Long
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # Short
            pnl_pct = (entry_price - current_price) / entry_price

        return pnl_pct

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Get feature window
        start_idx = max(0, self.current_bar - self.window_size + 1)
        end_idx = self.current_bar + 1

        features = self._precomputed_features[start_idx:end_idx]

        # Pad if necessary
        if len(features) < self.window_size:
            pad_size = self.window_size - len(features)
            padding = np.tile(features[0], (pad_size, 1))
            features = np.vstack([padding, features])

        # Add position state to each row
        unrealized = self._compute_unrealized_pnl()
        position_features = np.array([
            float(self.position.direction),
            np.clip(unrealized, -0.1, 0.1) * 10,  # Scale to ~[-1, 1]
            np.clip(self.position.time_in_position / 100, 0, 1),
        ], dtype=np.float32)

        # Tile position features across window
        position_block = np.tile(position_features, (self.window_size, 1))
        obs = np.hstack([features, position_block]).astype(np.float32)

        return obs

    def _open_position(self, direction: int, tight: bool = False) -> float:
        """
        Open a new position.

        Args:
            direction: 1 for long, -1 for short
            tight: Use tight SL/TP (momentum trade)

        Returns:
            Immediate reward (negative for fees)
        """
        if self.position.direction != 0:
            return 0.0  # Already in position

        entry_price = self._get_price()
        sl_pct = self.tight_sl_pct if tight else self.default_sl_pct
        tp_pct = self.tight_tp_pct if tight else self.default_tp_pct

        if direction == 1:  # Long
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:  # Short
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)

        self.current_trade = Trade(
            entry_bar=self.current_bar,
            entry_price=entry_price,
            direction=direction,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
        )

        self.position = PositionState(
            direction=direction,
            entry_price=entry_price,
            unrealized_pnl_pct=0.0,
            time_in_position=0,
        )

        # Entry fee
        fee_cost = self.taker_fee
        return -fee_cost * self.reward_scale

    def _close_position(self, reason: str, exit_price: Optional[float] = None) -> float:
        """
        Close current position.

        Args:
            reason: Exit reason (SL_HIT, TP_HIT, MANUAL_CLOSE, etc.)
            exit_price: Exit price (defaults to current close)

        Returns:
            Realized PnL reward
        """
        if self.position.direction == 0 or self.current_trade is None:
            return 0.0

        if exit_price is None:
            exit_price = self._get_price()

        # Calculate PnL
        entry_price = self.current_trade.entry_price
        if self.position.direction == 1:  # Long
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # Short
            pnl_pct = (entry_price - exit_price) / entry_price

        # Apply fees (entry + exit)
        total_fees = 2 * self.taker_fee
        net_pnl_pct = pnl_pct - total_fees

        # Premature exit penalty: discourage closing before reaching TP zone
        premature_penalty = 0.0
        if reason in ("MANUAL_CLOSE", "FLIP"):
            tp_distance = abs(self.current_trade.tp_price - entry_price) / entry_price
            current_distance = abs(exit_price - entry_price) / entry_price
            achieved_pct = current_distance / tp_distance if tp_distance > 0 else 0

            if achieved_pct < 0.5:  # Closed before reaching 50% of TP
                # Penalty: 0.5% scaled by how early we exited
                premature_penalty = 0.005 * (1 - achieved_pct * 2)  # Max 0.5% at 0%, 0% at 50%
                net_pnl_pct -= premature_penalty

        # Update trade record
        self.current_trade.exit_bar = self.current_bar
        self.current_trade.exit_price = exit_price
        self.current_trade.exit_reason = reason
        self.current_trade.pnl_pct = pnl_pct
        self.current_trade.fees_pct = total_fees
        self.current_trade.net_pnl_pct = net_pnl_pct

        self.trades.append(self.current_trade)

        # Update equity
        self.equity *= (1 + net_pnl_pct)

        # Reset position
        self.current_trade = None
        self.position = PositionState()
        self._prev_unrealized_pnl = 0.0

        return net_pnl_pct * self.reward_scale

    def _check_sl_tp(self) -> Optional[float]:
        """
        Check if SL/TP is hit on next bar.

        Returns:
            Reward if position closed, None otherwise
        """
        if self.position.direction == 0 or self.current_trade is None:
            return None

        if self.current_bar >= self.n_bars - 1:
            return None

        # Check next bar's high/low
        next_high = self._get_high(self.current_bar + 1)
        next_low = self._get_low(self.current_bar + 1)

        sl_price = self.current_trade.sl_price
        tp_price = self.current_trade.tp_price

        if self.position.direction == 1:  # Long
            sl_hit = next_low <= sl_price
            tp_hit = next_high >= tp_price

            if sl_hit and tp_hit:
                # Conservative: assume SL hit first
                return self._close_position("SL_HIT", sl_price)
            elif sl_hit:
                return self._close_position("SL_HIT", sl_price)
            elif tp_hit:
                return self._close_position("TP_HIT", tp_price)
        else:  # Short
            sl_hit = next_high >= sl_price
            tp_hit = next_low <= tp_price

            if sl_hit and tp_hit:
                return self._close_position("SL_HIT", sl_price)
            elif sl_hit:
                return self._close_position("SL_HIT", sl_price)
            elif tp_hit:
                return self._close_position("TP_HIT", tp_price)

        return None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        self._reset_state()

        # Choose starting bar
        if self.random_start:
            max_start = self.n_bars - max(self.min_episode_bars, self.window_size) - 1
            if max_start <= self.window_size:
                self.current_bar = self.window_size
            else:
                self.current_bar = self.np_random.integers(self.window_size, max_start)
        else:
            self.current_bar = self.window_size

        obs = self._get_observation()
        info = {"equity": self.equity, "position": self.position.direction}

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return new state."""
        if self.terminated or self.truncated:
            obs = self._get_observation()
            return obs, 0.0, True, False, {}

        self.bars_in_episode += 1
        reward = 0.0
        action = Action(action)

        # Process action
        if action == Action.HOLD:
            # Small penalty for holding flat to encourage trading
            if self.position.direction == 0:
                reward -= self.hold_penalty * self.reward_scale

        elif action == Action.LONG:
            if self.position.direction == 0:
                reward += self._open_position(1, tight=False)
            elif self.position.direction == -1:
                # Close short, open long
                reward += self._close_position("FLIP")
                reward += self._open_position(1, tight=False)

        elif action == Action.SHORT:
            if self.position.direction == 0:
                reward += self._open_position(-1, tight=False)
            elif self.position.direction == 1:
                # Close long, open short
                reward += self._close_position("FLIP")
                reward += self._open_position(-1, tight=False)

        elif action == Action.CLOSE:
            if self.position.direction != 0:
                reward += self._close_position("MANUAL_CLOSE")
        # TIGHT actions (4, 5) removed from action space

        # Check SL/TP
        sl_tp_reward = self._check_sl_tp()
        if sl_tp_reward is not None:
            reward += sl_tp_reward

        # Unrealized PnL shaping (if still in position)
        if self.position.direction != 0:
            self.position.time_in_position += 1
            unrealized = self._compute_unrealized_pnl()
            self.position.unrealized_pnl_pct = unrealized

            # ASYMMETRIC shaping: only penalize holding losers, don't reward holding winners
            # This prevents the model from closing winners early to "lock in" shaping reward
            delta_unrealized = unrealized - self._prev_unrealized_pnl
            if unrealized < 0:  # Only apply when losing
                reward += delta_unrealized * self.unrealized_pnl_weight * self.reward_scale
            self._prev_unrealized_pnl = unrealized

        # Advance time
        self.current_bar += 1

        # Track equity
        current_unrealized = self._compute_unrealized_pnl()
        self.equity_curve.append(self.equity * (1 + current_unrealized))

        # Check termination
        if self.current_bar >= self.n_bars - 1:
            self.terminated = True
            # Close any open position at end
            if self.position.direction != 0:
                reward += self._close_position("END_OF_DATA")

        if self.max_episode_bars and self.bars_in_episode >= self.max_episode_bars:
            self.truncated = True

        # Get observation
        obs = self._get_observation()

        # Clip reward
        reward = np.clip(reward, -10 * self.reward_scale, 10 * self.reward_scale)

        info = {
            "equity": self.equity,
            "position": self.position.direction,
            "unrealized_pnl": self._compute_unrealized_pnl(),
            "n_trades": len(self.trades),
            "bars_in_episode": self.bars_in_episode,
        }

        if self.trades:
            last_trade = self.trades[-1]
            info["last_trade"] = {
                "pnl_pct": last_trade.net_pnl_pct,
                "reason": last_trade.exit_reason,
            }

        return obs, float(reward), self.terminated, self.truncated, info

    def get_trade_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        if not self.trades:
            return {"n_trades": 0}

        pnls = [t.net_pnl_pct for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "n_trades": len(self.trades),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "avg_pnl": np.mean(pnls) if pnls else 0,
            "total_pnl": np.sum(pnls),
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf"),
            "max_drawdown": self._compute_max_drawdown(),
            "final_equity": self.equity,
        }

    def _compute_max_drawdown(self) -> float:
        """Compute maximum drawdown from equity curve."""
        if not self.equity_curve:
            return 0.0

        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return float(drawdown.max())

    def render(self):
        """Render current state."""
        price = self._get_price()
        stats = self.get_trade_stats()
        print(
            f"Bar={self.current_bar} | Price={price:.4f} | "
            f"Equity={self.equity:.4f} | Pos={self.position.direction} | "
            f"Trades={stats['n_trades']} | WinRate={stats['win_rate']:.2%}"
        )


def make_crypto_env(
    df: pd.DataFrame,
    config: Optional[CryptoConfig] = None,
    **kwargs,
) -> CryptoTradingEnv:
    """Factory function to create crypto trading environment."""
    return CryptoTradingEnv(df, config, **kwargs)


if __name__ == "__main__":
    import numpy as np

    # Generate test data
    np.random.seed(42)
    n = 1000

    returns = np.random.randn(n) * 0.02
    close = pd.Series(100 * np.exp(np.cumsum(returns)))
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))

    df = pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.uniform(1000, 10000, n),
    })

    # Create environment
    env = CryptoTradingEnv(df, window_size=30, random_start=False)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Run random episode
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    total_reward = 0
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    stats = env.get_trade_stats()
    print(f"\nEpisode stats:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Trades: {stats['n_trades']}")
    print(f"  Win rate: {stats['win_rate']:.2%}")
    print(f"  Final equity: {stats['final_equity']:.4f}")
    print(f"  Max drawdown: {stats['max_drawdown']:.2%}")

    # Verify gym compatibility
    from gymnasium.utils.env_checker import check_env
    env2 = CryptoTradingEnv(df, window_size=30, random_start=False, max_episode_bars=100)
    check_env(env2, skip_render_check=True)
    print("\nGymnasium check passed!")
