"""
Analyze what's different about trades that hit SL vs profitable trades.
Goal: Find patterns that predict SL hits so we can prevent bad entries.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crypto.config import load_config
from src.crypto.data_manager import DataManager
from src.crypto.trading_env import CryptoTradingEnv, Action
from src.crypto.feature_extractor import FeatureExtractor, FEATURE_SPECS
from src.crypto.indicators import compute_all_indicators


def analyze_sl_hits(model_path: str):
    """Analyze features at entry for SL hits vs profitable trades."""

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

    # Create environment
    env = CryptoTradingEnv(
        df=val_df,
        config=config,
        window_size=config.training.window_size,
        random_start=False,
        max_episode_bars=None,
    )

    # Feature extractor for analysis
    extractor = FeatureExtractor(config=config, window_size=config.training.window_size)
    feature_names = [spec.name for spec in FEATURE_SPECS]

    # Run episode and collect entry features
    obs, _ = env.reset()
    done = False

    entry_features = []  # Store features at entry time
    current_entry_bar = None
    current_direction = None

    bar_idx = env.current_bar  # Track bar index

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # If opening a new position, record entry features
        if action in [Action.LONG, Action.SHORT] and env.position.direction == 0:
            current_entry_bar = bar_idx
            current_direction = 1 if action == Action.LONG else -1

            # Get features at this bar
            features_at_entry = env._precomputed_features[bar_idx]

        obs, reward, terminated, truncated, info = env.step(action)
        bar_idx = env.current_bar
        done = terminated or truncated

    # Now analyze trades
    sl_hit_features = []
    profitable_features = []
    manual_loss_features = []

    for trade in env.trades:
        entry_bar = trade.entry_bar
        features = env._precomputed_features[entry_bar]

        if trade.exit_reason == "SL_HIT":
            sl_hit_features.append(features)
        elif trade.net_pnl_pct > 0:
            profitable_features.append(features)
        elif trade.exit_reason == "MANUAL_CLOSE" and trade.net_pnl_pct <= 0:
            manual_loss_features.append(features)

    print(f"SL Hit trades: {len(sl_hit_features)}")
    print(f"Profitable trades: {len(profitable_features)}")
    print(f"Manual loss trades: {len(manual_loss_features)}")
    print()

    if len(sl_hit_features) == 0 or len(profitable_features) == 0:
        print("Not enough trades to analyze")
        return

    sl_hit_arr = np.array(sl_hit_features)
    profitable_arr = np.array(profitable_features)

    # Compare means
    print("=" * 70)
    print("FEATURE COMPARISON: SL_HIT vs PROFITABLE entries")
    print("=" * 70)
    print(f"{'Feature':<35} {'SL_HIT':>10} {'Profitable':>10} {'Diff':>10}")
    print("-" * 70)

    differences = []
    for i, name in enumerate(feature_names[:len(sl_hit_arr[0])]):  # Limit to actual features
        sl_mean = sl_hit_arr[:, i].mean()
        prof_mean = profitable_arr[:, i].mean()
        diff = sl_mean - prof_mean

        differences.append((name, sl_mean, prof_mean, diff, abs(diff)))

    # Sort by absolute difference
    differences.sort(key=lambda x: x[4], reverse=True)

    # Show top 20 most different features
    print("\nTOP 20 FEATURES WITH LARGEST DIFFERENCES:")
    print("-" * 70)
    for name, sl_mean, prof_mean, diff, abs_diff in differences[:20]:
        indicator = "→" if abs_diff > 0.1 else " "
        print(f"{indicator} {name:<33} {sl_mean:>10.3f} {prof_mean:>10.3f} {diff:>+10.3f}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Key insights
    key_features = [
        "trend_direction",
        "htf_trend",
        "mtf_aligned",
        "rsi_normalized",
        "macd_bullish",
        "macd_bearish",
        "hma_direction",
        "kalman_direction",
    ]

    print("\nKey directional features at entry:")
    for name in key_features:
        if name in [d[0] for d in differences]:
            for d in differences:
                if d[0] == name:
                    print(f"  {name}: SL_HIT={d[1]:.3f}, Profitable={d[2]:.3f}, diff={d[3]:+.3f}")
                    break

    print()

    # Check if SL hits are going against trend
    trend_idx = feature_names.index("trend_direction") if "trend_direction" in feature_names else None
    if trend_idx and trend_idx < sl_hit_arr.shape[1]:
        sl_trend = sl_hit_arr[:, trend_idx].mean()
        prof_trend = profitable_arr[:, trend_idx].mean()
        print(f"Average trend_direction at entry:")
        print(f"  SL_HIT trades: {sl_trend:.3f}")
        print(f"  Profitable:    {prof_trend:.3f}")
        if abs(sl_trend - prof_trend) > 0.2:
            print(f"  → SL hits are entering {'against' if sl_trend * prof_trend < 0 else 'with weaker'} trend!")

    # Check MTF alignment
    mtf_idx = feature_names.index("mtf_aligned") if "mtf_aligned" in feature_names else None
    if mtf_idx and mtf_idx < sl_hit_arr.shape[1]:
        sl_mtf = sl_hit_arr[:, mtf_idx].mean()
        prof_mtf = profitable_arr[:, mtf_idx].mean()
        print(f"\nMTF alignment at entry:")
        print(f"  SL_HIT trades: {sl_mtf:.3f}")
        print(f"  Profitable:    {prof_mtf:.3f}")
        if sl_mtf < prof_mtf:
            print(f"  → SL hits have WORSE MTF alignment!")

    print()


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
                    analyze_sl_hits(str(model_path))
    else:
        analyze_sl_hits(sys.argv[1])
