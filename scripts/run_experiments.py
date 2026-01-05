#!/usr/bin/env python3
"""
Hyperparameter experimentation runner for RL trading system.

Usage:
    # Run single experiment with specific parameters
    python scripts/run_experiments.py --sl 2.0 --tp 4.0 --ltf 5m --htf 1h

    # Run predefined experiment grid
    python scripts/run_experiments.py --experiments sl_tp_grid

    # Quick validation mode (1M steps instead of 10M)
    python scripts/run_experiments.py --experiments sl_tp_grid --quick

    # List available experiment grids
    python scripts/run_experiments.py --list-experiments
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.crypto.config import load_config, CryptoConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Predefined experiment grids
EXPERIMENT_GRIDS = {
    "sl_tp_grid": {
        "description": "SL/TP percentage sweep",
        "params": {
            "sl_pct": [0.015, 0.02, 0.03, 0.04],
            "tp_pct": [0.03, 0.04, 0.06, 0.08],
        },
        "constraints": lambda p: p["tp_pct"] >= p["sl_pct"] * 1.5,  # Min 1.5:1 R:R
    },
    "timeframe_grid": {
        "description": "LTF/HTF combinations",
        "params": {
            "ltf": ["3m", "5m", "15m"],
            "htf": ["15m", "30m", "1h", "4h"],
        },
        "constraints": lambda p: _tf_to_minutes(p["htf"]) > _tf_to_minutes(p["ltf"]),
    },
    "rr_grid": {
        "description": "Risk:Reward ratio variations (fixed SL)",
        "params": {
            "sl_pct": [0.03],
            "rr_ratio": [1.5, 2.0, 2.5, 3.0],
        },
        "derived": lambda p: {"tp_pct": p["sl_pct"] * p["rr_ratio"]},
    },
    "indicator_grid": {
        "description": "Indicator length variations",
        "params": {
            "atr_length": [7, 14, 21],
            "rsi_length": [7, 14, 21],
        },
    },
    "quick_validation": {
        "description": "Quick test of key variations",
        "params": {
            "sl_pct": [0.02, 0.03],
            "tp_pct": [0.04, 0.06],
        },
    },
}


def _tf_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes."""
    if tf.endswith("m"):
        return int(tf[:-1])
    elif tf.endswith("h"):
        return int(tf[:-1]) * 60
    elif tf.endswith("d"):
        return int(tf[:-1]) * 1440
    return 0


def generate_experiment_configs(grid_name: str) -> list[dict[str, Any]]:
    """Generate all experiment configurations from a grid."""
    if grid_name not in EXPERIMENT_GRIDS:
        raise ValueError(f"Unknown grid: {grid_name}. Available: {list(EXPERIMENT_GRIDS.keys())}")

    grid = EXPERIMENT_GRIDS[grid_name]
    params = grid["params"]
    constraints = grid.get("constraints", lambda p: True)
    derived = grid.get("derived", lambda p: {})

    # Generate all combinations
    keys = list(params.keys())
    values = [params[k] for k in keys]

    configs = []
    for combo in product(*values):
        config = dict(zip(keys, combo))

        # Apply constraints
        if not constraints(config):
            continue

        # Apply derived values
        config.update(derived(config))

        # Generate experiment name
        name_parts = []
        for k, v in config.items():
            if isinstance(v, float):
                name_parts.append(f"{k}{v:.1%}".replace(".", "p").replace("%", ""))
            else:
                name_parts.append(f"{k}{v}")
        config["name"] = "_".join(name_parts)

        configs.append(config)

    return configs


def create_experiment_config(
    base_config: CryptoConfig,
    **overrides,
) -> CryptoConfig:
    """Create a modified config for an experiment."""
    import copy

    config = copy.deepcopy(base_config)

    # Apply overrides
    for key, value in overrides.items():
        if key == "sl_pct":
            config.risk.default_sl_pct = value
        elif key == "tp_pct":
            config.risk.default_tp_pct = value
        elif key == "ltf":
            config.timeframes.ltf = value
        elif key == "htf":
            config.timeframes.htf = value
        elif key == "atr_length":
            config.indicators.atr_length = value
        elif key == "rsi_length":
            config.indicators.rsi_length = value
        elif key == "hma_length":
            config.indicators.trend_length = value
        elif key == "window_size":
            config.training.window_size = value
        elif key == "ent_coef":
            config.training.ent_coef = value
        elif key == "timesteps":
            config.training.total_timesteps = value

    return config


def run_experiment(
    name: str,
    config: CryptoConfig,
    output_base: Path,
    data_path: Path | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run a single experiment."""
    from src.crypto.train import train_ppo
    from src.crypto.data_manager import DataManager

    output_dir = output_base / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reduce timesteps for quick mode
    if quick:
        config.training.total_timesteps = 1_000_000
        config.training.eval_freq = 50_000
        config.training.checkpoint_freq = 100_000

    # Save experiment config
    config.to_yaml(output_dir / "config.yaml")

    # Load data
    dm = DataManager(config)
    if data_path:
        import pandas as pd
        df = pd.read_parquet(data_path)
    else:
        df = dm.load_ohlcv(config.pairs.default_pair, config.timeframes.ltf)

    if df.empty:
        logger.error(f"No data available for {config.pairs.default_pair}")
        return {"error": "No data"}

    # Split data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    logger.info(f"Starting experiment: {name}")
    logger.info(f"  SL: {config.risk.default_sl_pct:.1%}, TP: {config.risk.default_tp_pct:.1%}")
    logger.info(f"  LTF: {config.timeframes.ltf}, HTF: {config.timeframes.htf}")
    logger.info(f"  Timesteps: {config.training.total_timesteps:,}")

    # Train
    try:
        model, stats = train_ppo(
            train_df=train_df,
            val_df=val_df,
            config=config,
            output_dir=str(output_dir),
        )

        # Save stats
        with open(output_dir / "final_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Experiment {name} complete:")
        logger.info(f"  Win Rate: {stats.get('win_rate', 0):.1%}")
        logger.info(f"  Profit Factor: {stats.get('profit_factor', 0):.2f}")
        logger.info(f"  Final Equity: {stats.get('final_equity', 1):.2%}")

        return stats

    except Exception as e:
        logger.error(f"Experiment {name} failed: {e}")
        return {"error": str(e)}


def run_experiment_grid(
    grid_name: str,
    output_base: Path,
    data_path: Path | None = None,
    quick: bool = False,
) -> list[dict[str, Any]]:
    """Run all experiments in a grid."""
    configs = generate_experiment_configs(grid_name)
    base_config = load_config()

    logger.info(f"Running experiment grid: {grid_name}")
    logger.info(f"Total experiments: {len(configs)}")

    results = []
    for i, exp_config in enumerate(configs, 1):
        name = exp_config.pop("name")
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment {i}/{len(configs)}: {name}")
        logger.info(f"{'='*60}")

        config = create_experiment_config(base_config, **exp_config)
        result = run_experiment(
            name=name,
            config=config,
            output_base=output_base,
            data_path=data_path,
            quick=quick,
        )
        result["name"] = name
        result["params"] = exp_config
        results.append(result)

    return results


def compare_experiments(results: list[dict[str, Any]]) -> str:
    """Generate comparison table for experiments."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("EXPERIMENT COMPARISON")
    lines.append("=" * 80)

    # Header
    header = f"{'Name':<25} {'Win Rate':>10} {'PF':>8} {'R:R':>8} {'Equity':>10} {'Trades':>8}"
    lines.append(header)
    lines.append("-" * 80)

    # Sort by profit factor
    sorted_results = sorted(
        [r for r in results if "error" not in r],
        key=lambda x: x.get("profit_factor", 0),
        reverse=True,
    )

    for r in sorted_results:
        name = r.get("name", "unknown")[:25]
        wr = r.get("win_rate", 0)
        pf = r.get("profit_factor", 0)
        equity = r.get("final_equity", 1) - 1
        trades = r.get("n_trades", 0)

        # Calculate R:R from avg_win/avg_loss
        avg_win = abs(r.get("avg_win", 0.06))
        avg_loss = abs(r.get("avg_loss", 0.03))
        rr = avg_win / avg_loss if avg_loss > 0 else 0

        line = f"{name:<25} {wr:>9.1%} {pf:>8.2f} {rr:>7.2f}:1 {equity:>+9.1%} {trades:>8}"
        lines.append(line)

    # Failed experiments
    failed = [r for r in results if "error" in r]
    if failed:
        lines.append("-" * 80)
        lines.append("FAILED:")
        for r in failed:
            lines.append(f"  {r.get('name', 'unknown')}: {r.get('error', 'unknown error')}")

    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter experiments for RL trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Experiment selection
    parser.add_argument(
        "--experiments",
        type=str,
        help="Predefined experiment grid to run",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiment grids",
    )

    # Single experiment parameters
    parser.add_argument("--sl", type=float, help="Stop loss percentage (e.g., 2.0 for 2%)")
    parser.add_argument("--tp", type=float, help="Take profit percentage (e.g., 4.0 for 4%)")
    parser.add_argument("--ltf", type=str, help="Low timeframe (e.g., 5m)")
    parser.add_argument("--htf", type=str, help="High timeframe (e.g., 1h)")
    parser.add_argument("--name", type=str, help="Experiment name")

    # Common options
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to training data parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/experiments"),
        help="Output directory for experiments",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation mode (1M steps instead of 10M)",
    )

    args = parser.parse_args()

    # List experiments
    if args.list_experiments:
        print("\nAvailable experiment grids:")
        print("-" * 50)
        for name, grid in EXPERIMENT_GRIDS.items():
            configs = generate_experiment_configs(name)
            print(f"\n{name}:")
            print(f"  Description: {grid['description']}")
            print(f"  Parameters: {list(grid['params'].keys())}")
            print(f"  Total experiments: {len(configs)}")
        return

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Run experiment grid
    if args.experiments:
        results = run_experiment_grid(
            grid_name=args.experiments,
            output_base=args.output,
            data_path=args.data,
            quick=args.quick,
        )

        # Print comparison
        comparison = compare_experiments(results)
        print(comparison)

        # Save comparison
        with open(args.output / "comparison.txt", "w") as f:
            f.write(comparison)

        # Save results JSON
        with open(args.output / "all_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        return

    # Run single experiment
    if args.sl or args.tp or args.ltf or args.htf:
        base_config = load_config()

        overrides = {}
        if args.sl:
            overrides["sl_pct"] = args.sl / 100
        if args.tp:
            overrides["tp_pct"] = args.tp / 100
        if args.ltf:
            overrides["ltf"] = args.ltf
        if args.htf:
            overrides["htf"] = args.htf

        config = create_experiment_config(base_config, **overrides)

        name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
        result = run_experiment(
            name=name,
            config=config,
            output_base=args.output,
            data_path=args.data,
            quick=args.quick,
        )

        print(f"\nExperiment complete: {name}")
        print(json.dumps(result, indent=2, default=str))
        return

    # No action specified
    parser.print_help()


if __name__ == "__main__":
    main()
