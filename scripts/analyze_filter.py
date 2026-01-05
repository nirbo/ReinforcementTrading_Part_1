"""
Analyze which filter conditions are blocking the most entries.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crypto.config import load_config
from src.crypto.data_manager import DataManager
from src.crypto.trading_env import CryptoTradingEnv, Action
from src.crypto.indicators import compute_all_indicators


def analyze_filter_conditions(model_path: str):
    """Analyze which conditions block the most entries."""

    config = load_config()
    dm = DataManager(config)

    # Load data
    symbol = config.pairs.default_pair
    timeframe = config.timeframes.ltf
    df = dm.load_ohlcv(symbol, timeframe)

    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:].copy()

    # Compute indicators
    val_df = compute_all_indicators(val_df, config.indicators.model_dump())

    # Load model
    model = PPO.load(model_path)

    # Create environment WITHOUT filter
    env = CryptoTradingEnv(
        df=val_df,
        config=config,
        window_size=config.training.window_size,
        random_start=False,
        max_episode_bars=None,
    )
    env.use_entry_filter = False

    # Track attempted entries and what would block them
    block_reasons = Counter()
    entries_by_outcome = {"allowed_win": 0, "allowed_loss": 0, "blocked_would_win": 0, "blocked_would_loss": 0}

    obs, _ = env.reset()
    done = False
    pending_entry = None  # (bar, direction)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # Check if model wants to enter
        if action in [Action.LONG, Action.SHORT] and env.position.direction == 0:
            direction = 1 if action == Action.LONG else -1
            bar = env.current_bar
            row = env.df.iloc[bar]

            # Check each filter condition
            reasons = []

            # Velocity check
            velocity_long = row.get("velocity_long", False)
            velocity_short = row.get("velocity_short", False)
            if velocity_long and velocity_short:
                reasons.append("conflicting_velocity")
            if direction == 1 and velocity_short:
                reasons.append("long_against_velocity")
            if direction == -1 and velocity_long:
                reasons.append("short_against_velocity")

            # BB squeeze check
            bb_width = row.get("bb_width", 0)
            bb_sma_50 = env.df["bb_width"].rolling(50).mean().iloc[bar]
            in_squeeze = bb_width < bb_sma_50 * 0.5 if bb_sma_50 > 0 else False
            trend = row.get("trend_direction", 0)
            if in_squeeze:
                if (direction == 1 and trend <= 0) or (direction == -1 and trend >= 0):
                    reasons.append("squeeze_no_trend")

            # ATR expansion check
            atr_exp_bull = row.get("atr_exp_bull", False)
            atr_exp_bear = row.get("atr_exp_bear", False)
            if direction == 1 and atr_exp_bear:
                reasons.append("long_against_atr")
            if direction == -1 and atr_exp_bull:
                reasons.append("short_against_atr")

            # Trend alignment check
            if direction == 1 and trend < -0.5:
                reasons.append("long_against_trend")
            if direction == -1 and trend > 0.5:
                reasons.append("short_against_trend")

            would_block = len(reasons) > 0

            # Store for later outcome analysis
            pending_entry = (bar, direction, would_block, reasons)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Now analyze trades to see which would have been blocked
    print("=" * 70)
    print("FILTER CONDITION ANALYSIS")
    print("=" * 70)

    for trade in env.trades:
        bar = trade.entry_bar
        direction = trade.direction
        row = env.df.iloc[bar]

        # Recompute block reasons
        reasons = []

        velocity_long = row.get("velocity_long", False)
        velocity_short = row.get("velocity_short", False)
        if velocity_long and velocity_short:
            reasons.append("conflicting_velocity")
        if direction == 1 and velocity_short:
            reasons.append("long_against_velocity")
        if direction == -1 and velocity_long:
            reasons.append("short_against_velocity")

        bb_width = row.get("bb_width", 0)
        bb_sma_50 = env.df["bb_width"].rolling(50).mean().iloc[bar]
        in_squeeze = bb_width < bb_sma_50 * 0.5 if bb_sma_50 > 0 else False
        trend = row.get("trend_direction", 0)
        if in_squeeze:
            if (direction == 1 and trend <= 0) or (direction == -1 and trend >= 0):
                reasons.append("squeeze_no_trend")

        atr_exp_bull = row.get("atr_exp_bull", False)
        atr_exp_bear = row.get("atr_exp_bear", False)
        if direction == 1 and atr_exp_bear:
            reasons.append("long_against_atr")
        if direction == -1 and atr_exp_bull:
            reasons.append("short_against_atr")

        if direction == 1 and trend < -0.5:
            reasons.append("long_against_trend")
        if direction == -1 and trend > 0.5:
            reasons.append("short_against_trend")

        would_block = len(reasons) > 0
        is_winner = trade.net_pnl_pct > 0
        is_sl_hit = trade.exit_reason == "SL_HIT"

        if would_block:
            for r in reasons:
                block_reasons[r] += 1
            if is_winner:
                entries_by_outcome["blocked_would_win"] += 1
            else:
                entries_by_outcome["blocked_would_loss"] += 1
                if is_sl_hit:
                    block_reasons[f"BLOCKED_SL_HIT"] += 1
        else:
            if is_winner:
                entries_by_outcome["allowed_win"] += 1
            else:
                entries_by_outcome["allowed_loss"] += 1

    print("\nBlock reason frequency (how often each condition triggers):")
    print("-" * 70)
    for reason, count in block_reasons.most_common():
        print(f"  {reason:30s}: {count:4d}")

    print("\n\nOutcome analysis:")
    print("-" * 70)
    total = sum(entries_by_outcome.values())
    for outcome, count in entries_by_outcome.items():
        print(f"  {outcome:25s}: {count:4d} ({count/total*100:5.1f}%)")

    # Calculate filter effectiveness
    blocked_total = entries_by_outcome["blocked_would_win"] + entries_by_outcome["blocked_would_loss"]
    blocked_losses = entries_by_outcome["blocked_would_loss"]
    blocked_wins = entries_by_outcome["blocked_would_win"]

    print("\n\nFilter effectiveness:")
    print("-" * 70)
    if blocked_total > 0:
        print(f"  Would block {blocked_total} trades:")
        print(f"    - {blocked_losses} would be losses ({blocked_losses/blocked_total*100:.1f}%) ✓ Good")
        print(f"    - {blocked_wins} would be wins ({blocked_wins/blocked_total*100:.1f}%) ✗ Lost opportunity")
        print(f"  SL hits that would be blocked: {block_reasons.get('BLOCKED_SL_HIT', 0)}")

    allowed_total = entries_by_outcome["allowed_win"] + entries_by_outcome["allowed_loss"]
    if allowed_total > 0:
        allowed_wr = entries_by_outcome["allowed_win"] / allowed_total * 100
        print(f"\n  Allowed trades: {allowed_total} (WR would be {allowed_wr:.1f}%)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        models_dir = Path("models")
        if models_dir.exists():
            run_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])
            if run_dirs:
                latest = run_dirs[-1]
                model_path = latest / "ppo_crypto_final.zip"
                if model_path.exists():
                    analyze_filter_conditions(str(model_path))
    else:
        analyze_filter_conditions(sys.argv[1])
