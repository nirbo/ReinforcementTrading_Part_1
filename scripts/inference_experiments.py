#!/usr/bin/env python3
"""
Fast inference experiments - test existing model with different SL/TP combinations.

Uses VECTORIZED backtesting for 10-50x speedup:
1. Precompute ALL features once
2. Run model on ALL observations in one batched GPU pass
3. Simulate trading with NumPy (no Python loops)

Usage:
    # Test SL/TP matrix (4x4 = 16 combinations) - FAST!
    python scripts/inference_experiments.py --model models/best/ppo_crypto_final.zip \
        --sl 1.5 2.0 2.5 3.0 --tp 3.0 4.0 5.0 6.0

    # Test on specific pair
    python scripts/inference_experiments.py --model models/best/ppo_crypto_final.zip \
        --sl 2.0 3.0 --tp 4.0 6.0 --pair BNB/USDT:USDT

    # Output results to CSV
    python scripts/inference_experiments.py --model models/best/ppo_crypto_final.zip \
        --sl 2.0 3.0 --tp 4.0 6.0 --csv results.csv

    # Use old slow method (for comparison/debugging)
    python scripts/inference_experiments.py --model models/best/ppo_crypto_final.zip \
        --sl 2.0 3.0 --tp 4.0 6.0 --slow
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_inference_experiment(
    model,
    config,
    df: pd.DataFrame,
    sl_pct: float,
    tp_pct: float,
    htf_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run a single backtest with specific SL/TP settings."""
    import copy
    from src.crypto.evaluate import Backtester

    # Create modified config
    exp_config = copy.deepcopy(config)
    exp_config.risk.default_sl_pct = sl_pct
    exp_config.risk.default_tp_pct = tp_pct

    # Run backtest
    backtester = Backtester(model, exp_config)
    result = backtester.run(
        df=df,
        htf_df=htf_df,
        symbol=config.pairs.default_pair,
        timeframe=config.timeframes.ltf,
    )

    # Extract key metrics
    return {
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "rr_ratio": tp_pct / sl_pct if sl_pct > 0 else 0,
        "n_trades": result.n_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "total_pnl": result.total_return_pct,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "avg_win": result.avg_win,
        "avg_loss": result.avg_loss,
        "expectancy": result.expectancy,
    }


def run_fast_sl_tp_matrix(
    model_path: Path,
    sl_values: list[float],
    tp_values: list[float],
    data_path: Path | None = None,
    pair: str | None = None,
    config_path: Path | None = None,
    batch_size: int = 4096,
) -> list[dict[str, Any]]:
    """
    Run FAST vectorized SL × TP matrix.

    Uses batched predictions + NumPy simulation for 10-50x speedup.
    """
    import torch
    import warnings
    from stable_baselines3 import PPO
    from src.crypto.config import load_config, CryptoConfig
    from src.crypto.data_manager import DataManager
    from src.crypto.fast_backtest import fast_sl_tp_matrix

    # Load config
    config = CryptoConfig.from_yaml(config_path) if config_path else load_config()
    if pair:
        config.pairs.default_pair = pair

    # Load model
    logger.info(f"Loading model: {model_path}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*GPU.*MlpPolicy.*")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PPO.load(model_path, device=device)

    # Compile for batched inference
    compile_mode = config.training.torch_compile
    if device == "cuda" and compile_mode and hasattr(torch, "compile"):
        try:
            model.policy = torch.compile(model.policy, mode=compile_mode, fullgraph=False)
            logger.info(f"Policy compiled (mode={compile_mode}, device={device})")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    # Load data
    dm = DataManager(config)
    df = pd.read_parquet(data_path) if data_path else dm.load_ohlcv(config.pairs.default_pair, config.timeframes.ltf)

    if df.empty:
        logger.error(f"No data for {config.pairs.default_pair}")
        return []

    logger.info(f"Data: {len(df)} bars for {config.pairs.default_pair}")

    # Convert percentages to decimals
    sl_decimals = [sl / 100 for sl in sl_values]
    tp_decimals = [tp / 100 for tp in tp_values]

    # Run fast vectorized backtest
    start_time = time.time()
    results = fast_sl_tp_matrix(model, df, config, sl_decimals, tp_decimals, batch_size)
    elapsed = time.time() - start_time

    logger.info(f"Completed {len(results)} experiments in {elapsed:.1f}s ({elapsed/len(results):.2f}s each)")

    return results


def run_sl_tp_matrix(
    model_path: Path,
    sl_values: list[float],
    tp_values: list[float],
    data_path: Path | None = None,
    pair: str | None = None,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run full SL × TP matrix of experiments (SLOW - use run_fast_sl_tp_matrix instead)."""
    from itertools import product
    import torch
    from stable_baselines3 import PPO
    from src.crypto.config import load_config, CryptoConfig
    from src.crypto.data_manager import DataManager

    # Load config
    if config_path:
        config = CryptoConfig.from_yaml(config_path)
    else:
        config = load_config()

    if pair:
        config.pairs.default_pair = pair

    # Load model on GPU with torch.compile for fast inference
    logger.info(f"Loading model: {model_path}")

    import warnings
    # Suppress SB3 warning about MLP on GPU - we know what we're doing with torch.compile
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*GPU.*MlpPolicy.*")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PPO.load(model_path, device=device)

    # Compile for fast inference
    compile_mode = config.training.torch_compile
    if device == "cuda" and compile_mode and hasattr(torch, "compile"):
        try:
            model.policy = torch.compile(
                model.policy,
                mode=compile_mode,
                fullgraph=False,
            )
            logger.info(f"Policy compiled with torch.compile (mode={compile_mode}, device={device})")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    else:
        logger.info(f"Using device: {device}")

    # Load data
    dm = DataManager(config)
    if data_path:
        df = pd.read_parquet(data_path)
    else:
        df = dm.load_ohlcv(config.pairs.default_pair, config.timeframes.ltf)

    htf_df = dm.load_ohlcv(config.pairs.default_pair, config.timeframes.htf)

    if df.empty:
        logger.error(f"No data for {config.pairs.default_pair}")
        return []

    logger.info(f"Data: {len(df)} bars for {config.pairs.default_pair}")

    # Generate all combinations
    combinations = list(product(sl_values, tp_values))
    # Filter: TP must be >= SL (positive R:R)
    combinations = [(sl, tp) for sl, tp in combinations if tp >= sl]

    logger.info(f"Running {len(combinations)} SL×TP combinations...")
    logger.info(f"  SL values: {sl_values}")
    logger.info(f"  TP values: {tp_values}")

    # Run experiments in parallel using thread pool
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    # Use multiple threads to saturate GPU (each thread runs a backtest)
    n_workers = min(len(combinations), os.cpu_count() or 4)
    logger.info(f"  Using {n_workers} parallel workers")

    def run_single(args):
        i, sl, tp = args
        sl_pct = sl / 100
        tp_pct = tp / 100
        try:
            result = run_inference_experiment(
                model=model,
                config=config,
                df=df,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                htf_df=htf_df,
            )
            return (i, sl, tp, result, None)
        except Exception as e:
            return (i, sl, tp, None, str(e))

    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(run_single, (i, sl, tp)): (i, sl, tp)
            for i, (sl, tp) in enumerate(combinations, 1)
        }

        for future in as_completed(futures):
            i, sl, tp, result, error = future.result()
            if error:
                logger.error(f"[{i}/{len(combinations)}] SL={sl:.1f}% TP={tp:.1f}% → Failed: {error}")
                results.append({"sl_pct": sl / 100, "tp_pct": tp / 100, "error": error})
            else:
                logger.info(
                    f"[{i}/{len(combinations)}] SL={sl:.1f}% TP={tp:.1f}% → "
                    f"WR={result['win_rate']:.1%} PF={result['profit_factor']:.2f} "
                    f"PnL={result['total_pnl']:+.1%}"
                )
                results.append(result)

    return results


def print_matrix_table(results: list[dict[str, Any]], metric: str = "profit_factor") -> str:
    """Print results as a 2D matrix table."""
    # Filter out errors
    valid = [r for r in results if "error" not in r]
    if not valid:
        return "No valid results"

    # Get unique SL and TP values
    sl_values = sorted(set(r["sl_pct"] for r in valid))
    tp_values = sorted(set(r["tp_pct"] for r in valid))

    # Build lookup
    lookup = {(r["sl_pct"], r["tp_pct"]): r for r in valid}

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"SL × TP MATRIX - {metric.upper()}")
    lines.append(f"{'='*70}")

    # Header row
    header = f"{'SL \\\\ TP':>8}"
    for tp in tp_values:
        header += f" {tp*100:>6.1f}%"
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    best_val = float("-inf")
    best_coords = None

    for sl in sl_values:
        row = f"{sl*100:>7.1f}%"
        for tp in tp_values:
            key = (sl, tp)
            if key in lookup:
                val = lookup[key].get(metric, 0)
                if metric == "profit_factor":
                    row += f" {val:>6.2f}"
                elif metric in ["win_rate", "total_pnl", "max_drawdown"]:
                    row += f" {val*100:>5.1f}%"
                else:
                    row += f" {val:>6.2f}"

                if val > best_val:
                    best_val = val
                    best_coords = (sl, tp)
            else:
                row += "     -"
        lines.append(row)

    lines.append("-" * len(header))

    # Best result
    if best_coords:
        lines.append(f"Best: SL={best_coords[0]*100:.1f}% TP={best_coords[1]*100:.1f}% → {metric}={best_val:.2f}")

    return "\n".join(lines)


def print_results_table(results: list[dict[str, Any]]) -> str:
    """Print detailed results table."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return "No valid results"

    # Sort by profit factor
    valid.sort(key=lambda x: x.get("profit_factor", 0), reverse=True)

    lines = []
    lines.append(f"\n{'='*90}")
    lines.append("DETAILED RESULTS (sorted by Profit Factor)")
    lines.append(f"{'='*90}")

    header = (
        f"{'SL':>6} {'TP':>6} {'R:R':>6} {'Trades':>7} {'WinRate':>8} "
        f"{'PF':>7} {'PnL':>8} {'MaxDD':>7} {'Sharpe':>7}"
    )
    lines.append(header)
    lines.append("-" * 90)

    for r in valid:
        sl = r["sl_pct"] * 100
        tp = r["tp_pct"] * 100
        rr = r["rr_ratio"]
        trades = r["n_trades"]
        wr = r["win_rate"] * 100
        pf = r["profit_factor"]
        pnl = r["total_pnl"] * 100
        mdd = r["max_drawdown"] * 100
        sharpe = r.get("sharpe_ratio", 0)

        marker = "→" if pf == valid[0]["profit_factor"] else " "
        line = (
            f"{marker}{sl:>5.1f}% {tp:>5.1f}% {rr:>5.1f}:1 {trades:>7} {wr:>7.1f}% "
            f"{pf:>7.2f} {pnl:>+7.1f}% {mdd:>6.1f}% {sharpe:>7.2f}"
        )
        lines.append(line)

    lines.append("=" * 90)

    # Summary
    profitable = [r for r in valid if r.get("profit_factor", 0) > 1]
    lines.append(f"\nProfitable configs: {len(profitable)}/{len(valid)}")

    if profitable:
        best = profitable[0]
        lines.append(f"\nRecommended: SL={best['sl_pct']*100:.1f}% TP={best['tp_pct']*100:.1f}%")
        lines.append(f"  Profit Factor: {best['profit_factor']:.2f}")
        lines.append(f"  Win Rate: {best['win_rate']:.1%}")
        lines.append(f"  Total PnL: {best['total_pnl']:+.1%}")

    return "\n".join(lines)


def export_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    """Export results to CSV."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info(f"Results exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fast inference experiments - test model with different SL/TP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SL/TP matrix (16 combinations)
  python scripts/inference_experiments.py --model models/best/ppo_crypto_final.zip \\
      --sl 1.5 2.0 2.5 3.0 --tp 3.0 4.0 5.0 6.0

  # Fine-grained search around known good values
  python scripts/inference_experiments.py --model models/best/ppo_crypto_final.zip \\
      --sl 2.5 2.75 3.0 3.25 3.5 --tp 5.0 5.5 6.0 6.5 7.0

  # Test on different pair
  python scripts/inference_experiments.py --model models/best/ppo_crypto_final.zip \\
      --sl 2.0 3.0 --tp 4.0 6.0 --pair BNB/USDT:USDT
        """,
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model (.zip)",
    )
    parser.add_argument(
        "--sl",
        type=float,
        nargs="+",
        default=[2.0, 2.5, 3.0, 3.5],
        help="Stop loss percentages to test (e.g., 2.0 2.5 3.0)",
    )
    parser.add_argument(
        "--tp",
        type=float,
        nargs="+",
        default=[4.0, 5.0, 6.0, 7.0],
        help="Take profit percentages to test (e.g., 4.0 5.0 6.0)",
    )
    parser.add_argument(
        "--pair",
        type=str,
        help="Trading pair to test (default: config default)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to data parquet file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Export results to CSV",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Export results to JSON",
    )

    args = parser.parse_args()

    # Validate model exists
    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    # Run experiments
    logger.info("Starting SL×TP inference experiments")
    logger.info(f"Model: {args.model}")

    results = run_fast_sl_tp_matrix(
        model_path=args.model,
        sl_values=args.sl,
        tp_values=args.tp,
        data_path=args.data,
        pair=args.pair,
        config_path=args.config,
    )

    # Print matrix tables
    print(print_matrix_table(results, "profit_factor"))
    print(print_matrix_table(results, "win_rate"))
    print(print_matrix_table(results, "total_pnl"))

    # Print detailed results
    print(print_results_table(results))

    # Export
    if args.csv:
        export_csv(results, args.csv)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results exported to {args.json}")


if __name__ == "__main__":
    main()
