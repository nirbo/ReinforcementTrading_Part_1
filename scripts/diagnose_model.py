"""
Diagnose model performance by analyzing trade patterns.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crypto.config import load_config
from src.crypto.data_manager import DataManager
from src.crypto.trading_env import CryptoTradingEnv, Action


def run_diagnostic(model_path: str, n_episodes: int = 5):
    """Run diagnostic analysis on trained model."""

    config = load_config()
    dm = DataManager(config)

    # Load validation data (last 20%)
    symbol = config.pairs.default_pair
    timeframe = config.timeframes.ltf
    df = dm.load_ohlcv(symbol, timeframe)

    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:].copy()

    print(f"Validation data: {len(val_df)} bars")
    print(f"Date range: {val_df.index[0]} to {val_df.index[-1]}")
    print()

    # Load model
    model = PPO.load(model_path)
    print(f"Model loaded from: {model_path}")
    print()

    # Create environment
    env = CryptoTradingEnv(
        df=val_df,
        config=config,
        window_size=config.training.window_size,
        random_start=False,
        max_episode_bars=None,  # Run full data
    )

    # Collect data across episodes
    all_trades = []
    action_counts = Counter()

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_counts[Action(action).name] += 1
            ep_actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Collect trades from this episode
        for trade in env.trades:
            all_trades.append({
                "episode": ep,
                "direction": "LONG" if trade.direction == 1 else "SHORT",
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "exit_reason": trade.exit_reason,
                "pnl_pct": trade.pnl_pct * 100,
                "net_pnl_pct": trade.net_pnl_pct * 100,
                "bars_held": trade.exit_bar - trade.entry_bar if trade.exit_bar else 0,
                "sl_pct": trade.sl_pct * 100,
                "tp_pct": trade.tp_pct * 100,
            })

    if not all_trades:
        print("NO TRADES - Model never opens positions!")
        print(f"Action distribution: {dict(action_counts)}")
        return

    trades_df = pd.DataFrame(all_trades)

    # =====================================================
    # 1. ACTION DISTRIBUTION
    # =====================================================
    print("=" * 60)
    print("1. ACTION DISTRIBUTION")
    print("=" * 60)
    total_actions = sum(action_counts.values())
    for action, count in sorted(action_counts.items()):
        pct = count / total_actions * 100
        print(f"  {action:12s}: {count:6d} ({pct:5.1f}%)")
    print()

    # =====================================================
    # 2. EXIT REASON BREAKDOWN
    # =====================================================
    print("=" * 60)
    print("2. EXIT REASON BREAKDOWN")
    print("=" * 60)
    exit_stats = trades_df.groupby("exit_reason").agg({
        "net_pnl_pct": ["count", "mean", "sum"],
        "bars_held": "mean",
    })
    exit_stats.columns = ["count", "avg_pnl%", "total_pnl%", "avg_bars"]
    exit_stats["pct_of_trades"] = exit_stats["count"] / len(trades_df) * 100
    exit_stats = exit_stats.sort_values("count", ascending=False)

    for reason in exit_stats.index:
        row = exit_stats.loc[reason]
        print(f"  {reason:15s}: {int(row['count']):3d} trades ({row['pct_of_trades']:5.1f}%), "
              f"avg={row['avg_pnl%']:+6.2f}%, total={row['total_pnl%']:+7.2f}%, "
              f"bars={row['avg_bars']:.1f}")
    print()

    # =====================================================
    # 3. DIRECTION ANALYSIS
    # =====================================================
    print("=" * 60)
    print("3. DIRECTION ANALYSIS")
    print("=" * 60)
    for direction in ["LONG", "SHORT"]:
        dir_trades = trades_df[trades_df["direction"] == direction]
        if len(dir_trades) == 0:
            print(f"  {direction}: No trades")
            continue

        wins = dir_trades[dir_trades["net_pnl_pct"] > 0]
        wr = len(wins) / len(dir_trades) * 100
        avg_pnl = dir_trades["net_pnl_pct"].mean()

        print(f"  {direction}: {len(dir_trades):3d} trades, WR={wr:.1f}%, avg={avg_pnl:+.2f}%")
    print()

    # =====================================================
    # 4. WIN/LOSS ANALYSIS
    # =====================================================
    print("=" * 60)
    print("4. WIN/LOSS ANALYSIS")
    print("=" * 60)
    winners = trades_df[trades_df["net_pnl_pct"] > 0]
    losers = trades_df[trades_df["net_pnl_pct"] <= 0]

    print(f"  Winners: {len(winners):3d} ({len(winners)/len(trades_df)*100:.1f}%)")
    if len(winners) > 0:
        print(f"    Avg win:     {winners['net_pnl_pct'].mean():+.2f}%")
        print(f"    Max win:     {winners['net_pnl_pct'].max():+.2f}%")
        print(f"    Avg bars:    {winners['bars_held'].mean():.1f}")

    print(f"  Losers:  {len(losers):3d} ({len(losers)/len(trades_df)*100:.1f}%)")
    if len(losers) > 0:
        print(f"    Avg loss:    {losers['net_pnl_pct'].mean():+.2f}%")
        print(f"    Max loss:    {losers['net_pnl_pct'].min():+.2f}%")
        print(f"    Avg bars:    {losers['bars_held'].mean():.1f}")
    print()

    # =====================================================
    # 5. EXIT REASON BY OUTCOME (KEY INSIGHT)
    # =====================================================
    print("=" * 60)
    print("5. EXIT REASON BY OUTCOME (WHERE ARE LOSSES COMING FROM?)")
    print("=" * 60)

    for reason in trades_df["exit_reason"].unique():
        reason_trades = trades_df[trades_df["exit_reason"] == reason]
        reason_wins = reason_trades[reason_trades["net_pnl_pct"] > 0]
        reason_losses = reason_trades[reason_trades["net_pnl_pct"] <= 0]

        if len(reason_trades) > 0:
            wr = len(reason_wins) / len(reason_trades) * 100
            total_pnl = reason_trades["net_pnl_pct"].sum()
            print(f"  {reason:15s}: WR={wr:5.1f}%, total={total_pnl:+7.2f}%")
            if len(reason_wins) > 0:
                print(f"    Winners: {len(reason_wins):3d}, avg={reason_wins['net_pnl_pct'].mean():+.2f}%")
            if len(reason_losses) > 0:
                print(f"    Losers:  {len(reason_losses):3d}, avg={reason_losses['net_pnl_pct'].mean():+.2f}%")
    print()

    # =====================================================
    # 6. BARS HELD DISTRIBUTION
    # =====================================================
    print("=" * 60)
    print("6. TRADE DURATION (bars held)")
    print("=" * 60)
    print(f"  Min:    {trades_df['bars_held'].min():4d} bars")
    print(f"  Median: {trades_df['bars_held'].median():4.0f} bars")
    print(f"  Mean:   {trades_df['bars_held'].mean():4.1f} bars")
    print(f"  Max:    {trades_df['bars_held'].max():4d} bars")

    # Duration by outcome
    print()
    print("  By outcome:")
    print(f"    Winners avg: {winners['bars_held'].mean():4.1f} bars")
    print(f"    Losers avg:  {losers['bars_held'].mean():4.1f} bars")
    print()

    # =====================================================
    # 7. SUMMARY
    # =====================================================
    print("=" * 60)
    print("7. SUMMARY")
    print("=" * 60)
    total_pnl = trades_df["net_pnl_pct"].sum()
    win_rate = len(winners) / len(trades_df) * 100
    avg_win = winners["net_pnl_pct"].mean() if len(winners) > 0 else 0
    avg_loss = losers["net_pnl_pct"].mean() if len(losers) > 0 else 0

    if avg_loss != 0:
        rr_ratio = abs(avg_win / avg_loss)
    else:
        rr_ratio = float('inf')

    expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)

    print(f"  Total trades:  {len(trades_df)}")
    print(f"  Win rate:      {win_rate:.1f}%")
    print(f"  Avg win:       {avg_win:+.2f}%")
    print(f"  Avg loss:      {avg_loss:+.2f}%")
    print(f"  Risk:Reward:   1:{rr_ratio:.2f}")
    print(f"  Expectancy:    {expectancy:+.3f}%/trade")
    print(f"  Total PnL:     {total_pnl:+.2f}%")
    print()

    # =====================================================
    # 8. RECOMMENDATIONS
    # =====================================================
    print("=" * 60)
    print("8. DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 60)

    # Check for common issues
    issues = []

    if win_rate < 50:
        issues.append(f"❌ Poor entry quality (WR={win_rate:.1f}% < 50%)")

    if rr_ratio < 1.5:
        issues.append(f"❌ Poor R:R ratio (1:{rr_ratio:.2f} < 1:1.5)")

    sl_exits = trades_df[trades_df["exit_reason"] == "SL_HIT"]
    if len(sl_exits) / len(trades_df) > 0.25:
        issues.append(f"❌ Too many SL hits ({len(sl_exits)/len(trades_df)*100:.1f}% > 25%)")

    manual_exits = trades_df[trades_df["exit_reason"] == "MANUAL_CLOSE"]
    if len(manual_exits) / len(trades_df) > 0.50:
        issues.append(f"⚠️ Many MANUAL_CLOSE ({len(manual_exits)/len(trades_df)*100:.1f}% > 50%)")

    tp_exits = trades_df[trades_df["exit_reason"] == "TP_HIT"]
    if len(tp_exits) / len(trades_df) < 0.10:
        issues.append(f"⚠️ Few TP hits ({len(tp_exits)/len(trades_df)*100:.1f}% < 10%)")

    if avg_win < 1.0:  # Less than 1%
        issues.append(f"⚠️ Small winners (avg {avg_win:.2f}% < 1%)")

    if abs(avg_loss) > 2.0:  # More than 2%
        issues.append(f"⚠️ Large losers (avg {avg_loss:.2f}% > 2%)")

    for issue in issues:
        print(f"  {issue}")

    if not issues:
        print("  ✓ No major issues detected")

    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Try to find most recent model
        models_dir = Path("models")
        if models_dir.exists():
            run_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])
            if run_dirs:
                latest = run_dirs[-1]
                model_path = latest / "ppo_crypto_final.zip"
                if model_path.exists():
                    print(f"Using latest model: {model_path}")
                    run_diagnostic(str(model_path))
                else:
                    print(f"No model found in {latest}")
            else:
                print("No model runs found in models/")
        else:
            print("Usage: python scripts/diagnose_model.py <model_path>")
    else:
        run_diagnostic(sys.argv[1])
