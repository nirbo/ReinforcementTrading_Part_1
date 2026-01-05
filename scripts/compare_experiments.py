#!/usr/bin/env python3
"""
Compare experiment results and generate analysis reports.

Usage:
    # Compare all experiments in a directory
    python scripts/compare_experiments.py models/experiments/

    # Compare specific experiments
    python scripts/compare_experiments.py models/experiments/sl2_tp4 models/experiments/sl3_tp6

    # Generate detailed report
    python scripts/compare_experiments.py models/experiments/ --detailed

    # Export to CSV
    python scripts/compare_experiments.py models/experiments/ --csv results.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_experiment_results(exp_dir: Path) -> dict[str, Any] | None:
    """Load results from an experiment directory."""
    stats_path = exp_dir / "final_stats.json"
    config_path = exp_dir / "config.yaml"

    if not stats_path.exists():
        return None

    with open(stats_path) as f:
        stats = json.load(f)

    # Load config if available
    config = {}
    if config_path.exists():
        from src.crypto.config import CryptoConfig
        try:
            cfg = CryptoConfig.from_yaml(config_path)
            config = {
                "sl_pct": cfg.risk.default_sl_pct,
                "tp_pct": cfg.risk.default_tp_pct,
                "ltf": cfg.timeframes.ltf,
                "htf": cfg.timeframes.htf,
                "timesteps": cfg.training.total_timesteps,
            }
        except Exception:
            pass

    return {
        "name": exp_dir.name,
        "path": str(exp_dir),
        "config": config,
        **stats,
    }


def find_experiments(paths: list[Path]) -> list[dict[str, Any]]:
    """Find and load all experiments from given paths."""
    results = []

    for path in paths:
        if path.is_file() and path.name == "final_stats.json":
            # Direct stats file
            result = load_experiment_results(path.parent)
            if result:
                results.append(result)
        elif path.is_dir():
            # Check if this is an experiment directory
            if (path / "final_stats.json").exists():
                result = load_experiment_results(path)
                if result:
                    results.append(result)
            else:
                # Search subdirectories
                for subdir in path.iterdir():
                    if subdir.is_dir() and (subdir / "final_stats.json").exists():
                        result = load_experiment_results(subdir)
                        if result:
                            results.append(result)

    return results


def calculate_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Calculate derived metrics for comparison."""
    metrics = dict(result)

    # R:R ratio
    avg_win = abs(result.get("avg_win", 0.06))
    avg_loss = abs(result.get("avg_loss", 0.03))
    metrics["rr_ratio"] = avg_win / avg_loss if avg_loss > 0 else 0

    # Expectancy
    win_rate = result.get("win_rate", 0)
    metrics["expectancy"] = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Net return
    metrics["net_return"] = result.get("final_equity", 1) - 1

    # Trades per day (assuming ~288 bars per day for 5m)
    n_trades = result.get("n_trades", 0)
    # This is approximate - would need episode length for accuracy
    metrics["trade_frequency"] = n_trades

    return metrics


def print_comparison_table(results: list[dict[str, Any]], detailed: bool = False) -> str:
    """Generate comparison table."""
    lines = []

    # Calculate metrics for all results
    metrics = [calculate_metrics(r) for r in results]

    # Sort by profit factor
    metrics.sort(key=lambda x: x.get("profit_factor", 0), reverse=True)

    # Header
    lines.append("\n" + "=" * 100)
    lines.append("EXPERIMENT COMPARISON")
    lines.append("=" * 100)

    if detailed:
        header = (
            f"{'Name':<20} {'SL':>6} {'TP':>6} {'LTF':>5} {'HTF':>5} "
            f"{'WR':>7} {'PF':>6} {'R:R':>6} {'Equity':>9} {'Trades':>7}"
        )
    else:
        header = f"{'Name':<30} {'Win Rate':>10} {'PF':>8} {'R:R':>8} {'Equity':>10} {'Trades':>8}"

    lines.append(header)
    lines.append("-" * len(header))

    for m in metrics:
        name = m.get("name", "unknown")
        wr = m.get("win_rate", 0)
        pf = m.get("profit_factor", 0)
        rr = m.get("rr_ratio", 0)
        equity = m.get("net_return", 0)
        trades = m.get("n_trades", 0)

        if detailed:
            cfg = m.get("config", {})
            sl = cfg.get("sl_pct", 0)
            tp = cfg.get("tp_pct", 0)
            ltf = cfg.get("ltf", "?")
            htf = cfg.get("htf", "?")

            line = (
                f"{name[:20]:<20} {sl:>5.1%} {tp:>5.1%} {ltf:>5} {htf:>5} "
                f"{wr:>6.1%} {pf:>6.2f} {rr:>5.2f}:1 {equity:>+8.1%} {trades:>7}"
            )
        else:
            line = f"{name[:30]:<30} {wr:>9.1%} {pf:>8.2f} {rr:>7.2f}:1 {equity:>+9.1%} {trades:>8}"

        # Highlight best result
        if m == metrics[0] and pf > 1:
            line = f"→ {line[2:]}"  # Add arrow for best

        lines.append(line)

    lines.append("=" * len(header))

    # Summary statistics
    if len(metrics) > 1:
        lines.append("\nSUMMARY:")
        profitable = [m for m in metrics if m.get("profit_factor", 0) > 1]
        lines.append(f"  Profitable experiments: {len(profitable)}/{len(metrics)}")

        if profitable:
            best = profitable[0]
            lines.append(f"  Best experiment: {best.get('name')}")
            lines.append(f"    - Profit Factor: {best.get('profit_factor', 0):.2f}")
            lines.append(f"    - Win Rate: {best.get('win_rate', 0):.1%}")
            lines.append(f"    - Net Return: {best.get('net_return', 0):+.1%}")

    return "\n".join(lines)


def export_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    """Export results to CSV."""
    import csv

    metrics = [calculate_metrics(r) for r in results]

    # Flatten config into main dict
    flat_metrics = []
    for m in metrics:
        flat = dict(m)
        cfg = flat.pop("config", {})
        flat.update({f"cfg_{k}": v for k, v in cfg.items()})
        flat_metrics.append(flat)

    # Get all keys
    all_keys = set()
    for m in flat_metrics:
        all_keys.update(m.keys())

    # Remove non-serializable keys
    all_keys.discard("path")
    all_keys = sorted(all_keys)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_metrics)

    print(f"Exported to {output_path}")


def print_recommendations(results: list[dict[str, Any]]) -> str:
    """Generate recommendations based on results."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 60)

    metrics = [calculate_metrics(r) for r in results]
    profitable = [m for m in metrics if m.get("profit_factor", 0) > 1]

    if not profitable:
        lines.append("No profitable experiments found.")
        lines.append("Consider:")
        lines.append("  1. Checking data quality")
        lines.append("  2. Increasing training timesteps")
        lines.append("  3. Trying different SL/TP ratios")
        return "\n".join(lines)

    # Analyze patterns
    best_pf = max(m.get("profit_factor", 0) for m in profitable)
    best = [m for m in profitable if m.get("profit_factor", 0) == best_pf][0]

    lines.append(f"Best configuration: {best.get('name')}")
    lines.append("")

    # SL/TP analysis
    cfg = best.get("config", {})
    if cfg:
        lines.append("Optimal parameters:")
        lines.append(f"  - SL: {cfg.get('sl_pct', 0):.1%}")
        lines.append(f"  - TP: {cfg.get('tp_pct', 0):.1%}")
        lines.append(f"  - LTF: {cfg.get('ltf', '?')}")
        lines.append(f"  - HTF: {cfg.get('htf', '?')}")

    lines.append("")
    lines.append("Next steps:")
    lines.append("  1. Run full training (10M+ steps) with best config")
    lines.append("  2. Test on held-out data period")
    lines.append("  3. Validate on other pairs (BNB, etc.)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Experiment directories or files to compare",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed comparison including config params",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--recommendations",
        action="store_true",
        help="Show recommendations based on results",
    )

    args = parser.parse_args()

    # Load experiments
    results = find_experiments(args.paths)

    if not results:
        print("No experiment results found.")
        print("Checked paths:", [str(p) for p in args.paths])
        sys.exit(1)

    print(f"Found {len(results)} experiments")

    # Print comparison
    comparison = print_comparison_table(results, detailed=args.detailed)
    print(comparison)

    # Export to CSV
    if args.csv:
        export_csv(results, args.csv)

    # Print recommendations
    if args.recommendations:
        recs = print_recommendations(results)
        print(recs)


if __name__ == "__main__":
    main()
