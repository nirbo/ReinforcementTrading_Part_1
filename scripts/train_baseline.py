#!/usr/bin/env python
"""Train baseline model without box features for A/B comparison."""

import logging
import os
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from src.crypto.config import load_config
from src.crypto.data_manager import DataManager
from src.crypto.train import train_ppo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting baseline training (no box features)")

    # Load baseline config (default.yaml has box_features.use_box_features=false)
    config = load_config(str(project_root / "configs" / "default.yaml"))

    # Verify box features are disabled
    assert not config.box_features.use_box_features, "Box features should be disabled for baseline"
    logger.info(f"Config verified: box_features.use_box_features = {config.box_features.use_box_features}")
    logger.info(f"Total timesteps: {config.training.total_timesteps:,}")
    logger.info(f"Seed: {config.training.seed}")

    # Load data
    dm = DataManager(config)
    symbol = config.pairs.default_pair
    timeframe = config.timeframes.ltf

    df = dm.load_ohlcv(symbol, timeframe)
    logger.info(f"Loaded {len(df):,} bars of {symbol} {timeframe}")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Add session boundaries for consistency (even if not used by baseline)
    df = DataManager.add_session_boundaries(df)

    # Train/validation split (80/20 temporal)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    logger.info(f"Training bars: {len(train_df):,}")
    logger.info(f"Validation bars: {len(val_df):,}")

    # Train
    output_dir = project_root / "models" / "baseline_no_box"
    model, final_stats = train_ppo(
        train_df=train_df,
        val_df=val_df,
        config=config,
        output_dir=str(output_dir),
    )

    logger.info("=" * 60)
    logger.info("BASELINE TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final stats: {final_stats}")
    logger.info(f"Win rate: {final_stats.get('win_rate', 0):.2%}")
    logger.info(f"Profit factor: {final_stats.get('profit_factor', 0):.4f}")
    logger.info(f"Max drawdown: {final_stats.get('max_drawdown', 0):.2%}")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
