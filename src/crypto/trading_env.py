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
    """Trading actions - 3 discrete actions only.

    CLOSE action removed - forces model to use SL/TP exits.
    This prevents premature exits and maintains proper R:R ratio.

    With CLOSE: Model exits early → 0.9% TP hit rate, R:R 0.72
    Without CLOSE: Model uses SL/TP → 28% TP hit rate, R:R 1.59
    """
    HOLD = 0
    LONG = 1
    SHORT = 2
    # CLOSE removed - model was closing winners early, destroying R:R
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

        # Action space: 3 discrete actions (HOLD, LONG, SHORT)
        # CLOSE removed - forces SL/TP exits for proper R:R
        self.action_space = spaces.Discrete(3)

        # Observation space: (window_size, feature_dim)
        # Use feature_extractor.feature_dim which includes box features if enabled
        # - Without box features: 76 base + 3 position = 79
        # - With box features: 76 base + 20 box + 3 position = 99
        self._base_feature_dim = self._precomputed_features.shape[1]
        self._box_feature_dim = self.feature_extractor._box_feature_dim
        total_feature_dim = self._base_feature_dim + self._box_feature_dim + 3  # +3 for position state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, total_feature_dim),
            dtype=np.float32,
        )

        # SL/TP settings
        self.default_sl_pct = self.config.risk.default_sl_pct
        self.default_tp_pct = self.config.risk.default_tp_pct
        self.tight_sl_pct = self.config.risk.tight_sl_pct
        self.tight_tp_pct = self.config.risk.tight_tp_pct

        # Entry filter settings (prevent trades during high uncertainty)
        self.use_entry_filter = True  # Toggle entry filter
        self.blocked_entries = 0  # Track blocked entries for logging

        # Minimum hold period before allowing flip (prevents FLIP as exit strategy)
        self.min_hold_bars = 12  # ~1 hour at 5m bars - must commit to trade

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

        # Entry filter tracking
        self.blocked_entries = 0

        # Session tracking for box features
        self._current_session_id: Optional[int] = None
        self._session_transitions: int = 0  # Count of session boundaries crossed

    @property
    def _has_session_info(self) -> bool:
        """Check if DataFrame has session_id column for session boundary detection."""
        return "session_id" in self.df.columns

    def _get_session_id(self, bar: int) -> Optional[int]:
        """Get session ID at bar, or None if session info not available."""
        if not self._has_session_info:
            return None
        return int(self.df.iloc[bar]["session_id"])

    def _find_session_start_bar(self, bar: int) -> int:
        """
        Find the first bar of the session containing the given bar.

        If session_id column is available, finds the first bar with the same session_id.
        Otherwise, returns the given bar (treat it as session start).
        """
        if not self._has_session_info:
            return bar

        session_id = self._get_session_id(bar)

        # Search backwards to find first bar of this session
        start_bar = bar
        while start_bar > 0:
            prev_session_id = self._get_session_id(start_bar - 1)
            if prev_session_id != session_id:
                break
            start_bar -= 1

        return start_bar

    def _initialize_box_for_session(self, session_start_bar: int) -> None:
        """
        Initialize box state with the first bar of a session.

        Args:
            session_start_bar: The index of the first bar in the session
        """
        if self._box_feature_dim == 0:
            return

        row = self.df.iloc[session_start_bar]
        self.feature_extractor.reset_box_session(
            first_open=float(row["open"]),
            first_high=float(row["high"]),
            first_low=float(row["low"]),
            first_volume=float(row.get("volume", 0)),
        )

        # Update session tracking
        self._current_session_id = self._get_session_id(session_start_bar)

        # Update box state for all bars from session start up to (but not including) current bar
        for bar_idx in range(session_start_bar + 1, self.current_bar + 1):
            bar_row = self.df.iloc[bar_idx]
            self.feature_extractor.update_box_state(
                high=float(bar_row["high"]),
                low=float(bar_row["low"]),
                close=float(bar_row["close"]),
                volume=float(bar_row.get("volume", 0)),
            )

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

    def _check_entry_filter(self, direction: int) -> bool:
        """
        Check if entry should be allowed based on market conditions.

        SIMPLIFIED FILTER - only block high-confidence bad entries:
        - Velocity against our direction (strong opposing momentum)
        - ATR expansion against our direction (volatility spike opposing)

        Args:
            direction: 1 for long, -1 for short

        Returns:
            True if entry is allowed, False if blocked
        """
        if not self.use_entry_filter:
            return True

        row = self.df.iloc[self.current_bar]

        # Block if velocity is against our direction (strong opposing momentum)
        velocity_long = row.get("velocity_long", False)
        velocity_short = row.get("velocity_short", False)

        if direction == 1 and velocity_short:  # Going long during short velocity
            return False
        if direction == -1 and velocity_long:  # Going short during long velocity
            return False

        # Block during ATR expansion against our direction
        atr_exp_bull = row.get("atr_exp_bull", False)
        atr_exp_bear = row.get("atr_exp_bear", False)

        if direction == 1 and atr_exp_bear:
            return False
        if direction == -1 and atr_exp_bull:
            return False

        return True

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
        # Get base feature window
        start_idx = max(0, self.current_bar - self.window_size + 1)
        end_idx = self.current_bar + 1

        base_features = self._precomputed_features[start_idx:end_idx]

        # Pad if necessary
        if len(base_features) < self.window_size:
            pad_size = self.window_size - len(base_features)
            padding = np.tile(base_features[0], (pad_size, 1))
            base_features = np.vstack([padding, base_features])

        # Add box features if enabled
        if self._box_feature_dim > 0:
            # Get current bar data for box feature extraction
            row = self.df.iloc[self.current_bar]
            box_features = self.feature_extractor.get_box_features(
                open_price=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0)),
                atr=float(row.get("atr", row["close"] * 0.02)),  # Default 2% ATR
            )
            # Tile box features across window (same for all rows since it's current state)
            box_block = np.tile(box_features, (self.window_size, 1))
        else:
            box_block = np.zeros((self.window_size, 0), dtype=np.float32)

        # Add position state to each row
        unrealized = self._compute_unrealized_pnl()
        position_features = np.array([
            float(self.position.direction),
            np.clip(unrealized, -0.1, 0.1) * 10,  # Scale to ~[-1, 1]
            np.clip(self.position.time_in_position / 100, 0, 1),
        ], dtype=np.float32)

        # Tile position features across window
        position_block = np.tile(position_features, (self.window_size, 1))

        # Concatenate: base_features + box_features + position_features
        obs = np.hstack([base_features, box_block, position_block]).astype(np.float32)

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

        # Check entry filter - block uncertain entries
        if not self._check_entry_filter(direction):
            self.blocked_entries += 1
            return 0.0  # Entry blocked by filter

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

        # Simple reward: just the PnL (no exit bonuses/penalties)
        # With CLOSE removed, exits are via SL/TP which is exactly what we want

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

        # Initialize box state if enabled
        # Find the session start for the current bar and initialize box with full session history
        if self._box_feature_dim > 0:
            session_start = self._find_session_start_bar(self.current_bar)
            self._initialize_box_for_session(session_start)

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
            # FLIP disabled - positions can only exit via SL/TP
            # This forces proper R:R and prevents early exit gaming

        elif action == Action.SHORT:
            if self.position.direction == 0:
                reward += self._open_position(-1, tight=False)
            # FLIP disabled - positions can only exit via SL/TP
        # Both CLOSE and FLIP removed - model only decides WHEN to enter and WHICH direction

        # Check SL/TP
        sl_tp_reward = self._check_sl_tp()
        if sl_tp_reward is not None:
            reward += sl_tp_reward

        # Update position time (no shaping - SL/TP handle exits)
        if self.position.direction != 0:
            self.position.time_in_position += 1
            unrealized = self._compute_unrealized_pnl()
            self.position.unrealized_pnl_pct = unrealized
            # No unrealized PnL shaping - CLOSE action removed, so model can't act on shaping
            # Exits are forced via SL/TP which maintains proper R:R

        # Advance time
        self.current_bar += 1

        # Update box state if enabled
        if self._box_feature_dim > 0:
            row = self.df.iloc[self.current_bar]

            # Check for session boundary (new trading day)
            new_session_id = self._get_session_id(self.current_bar)
            if new_session_id is not None and new_session_id != self._current_session_id:
                # Session boundary crossed - reset box state with new session's first bar
                # Note: Episode continues (no episode reset), only box state resets
                self.feature_extractor.reset_box_session(
                    first_open=float(row["open"]),
                    first_high=float(row["high"]),
                    first_low=float(row["low"]),
                    first_volume=float(row.get("volume", 0)),
                )
                self._current_session_id = new_session_id
                self._session_transitions += 1
            else:
                # Same session - update box state with new bar
                self.feature_extractor.update_box_state(
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0)),
                )

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
            "session_transitions": self._session_transitions,  # Session boundaries crossed
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
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "final_equity": self.equity,
            }

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
