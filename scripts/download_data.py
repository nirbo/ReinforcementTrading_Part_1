#!/usr/bin/env python3
"""
Download historical OHLCV data for all configured trading pairs.

Usage:
    # Download all pairs with default settings (2 years)
    python scripts/download_data.py --days 730

    # Download specific pairs
    python scripts/download_data.py --pairs SOL/USDT:USDT BNB/USDT:USDT --days 365

    # Download with specific timeframes (auto-aggregates non-native like 9m)
    python scripts/download_data.py --timeframes 5m 9m --days 365

    # Include 1m data (needed for custom aggregation)
    python scripts/download_data.py --include-1m --days 365
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.crypto.config import load_config, TimeframeConfig
from src.crypto.data_manager import DataManager
from src.crypto.aggregation import aggregate_from_1m

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_data(
    pairs: list[str] | None = None,
    timeframes: list[str] | None = None,
    days: int = 730,
    include_1m: bool = False,
) -> dict[str, dict[str, int]]:
    """
    Download historical data for specified pairs and timeframes.

    Args:
        pairs: List of trading pairs (e.g., ['SOL/USDT:USDT']). If None, uses config.
        timeframes: List of timeframes. If None, uses config LTF/HTF.
        days: Number of days of history to download.
        include_1m: Always download 1m data (required for custom aggregation).

    Returns:
        Dict mapping symbol -> timeframe -> bar count
    """
    config = load_config()
    dm = DataManager(config)

    # Use config defaults if not specified
    if pairs is None:
        pairs = config.pairs.pairs
    if timeframes is None:
        timeframes = [config.timeframes.ltf, config.timeframes.htf]

    # Separate native and non-native timeframes
    native_tfs = []
    non_native_tfs = []
    for tf in timeframes:
        if TimeframeConfig.is_native(tf):
            native_tfs.append(tf)
        else:
            non_native_tfs.append(tf)
            # Non-native timeframes require 1m base data
            include_1m = True

    # Build final list of timeframes to download from exchange
    exchange_tfs = set(native_tfs)
    if include_1m:
        exchange_tfs.add("1m")

    results = {}
    total_pairs = len(pairs)

    for i, symbol in enumerate(pairs, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol} ({i}/{total_pairs})")
        logger.info(f"{'='*60}")
        results[symbol] = {}

        # Download native timeframes from exchange
        for tf in exchange_tfs:
            logger.info(f"Downloading {symbol} {tf} ({days} days)...")
            try:
                df = dm.collect_historical(symbol, tf, days=days)
                results[symbol][tf] = len(df)
                logger.info(f"  -> {len(df)} bars downloaded")
            except Exception as e:
                logger.error(f"  -> Error: {e}")
                results[symbol][tf] = 0

        # Aggregate non-native timeframes from 1m
        for tf in non_native_tfs:
            logger.info(f"Aggregating {symbol} {tf} from 1m data...")
            try:
                # Parse minutes from timeframe
                if tf.endswith("m"):
                    target_minutes = int(tf[:-1])
                elif tf.endswith("h"):
                    target_minutes = int(tf[:-1]) * 60
                else:
                    logger.error(f"  -> Invalid timeframe format: {tf}")
                    continue

                df = aggregate_from_1m(symbol, target_minutes, config, save=True)
                results[symbol][tf] = len(df)
                logger.info(f"  -> {len(df)} bars aggregated")
            except Exception as e:
                logger.error(f"  -> Error: {e}")
                results[symbol][tf] = 0

    return results


def print_summary(results: dict[str, dict[str, int]]) -> None:
    """Print a summary table of downloaded data."""
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)

    # Collect all timeframes
    all_tfs = set()
    for tfs in results.values():
        all_tfs.update(tfs.keys())
    all_tfs = sorted(all_tfs, key=lambda x: (x[-1], int(x[:-1]) if x[:-1].isdigit() else 0))

    # Print header
    header = f"{'Symbol':<20}" + "".join(f"{tf:>10}" for tf in all_tfs)
    logger.info(header)
    logger.info("-" * len(header))

    # Print rows
    for symbol, tfs in results.items():
        row = f"{symbol:<20}"
        for tf in all_tfs:
            count = tfs.get(tf, 0)
            row += f"{count:>10,}"
        logger.info(row)

    # Print total
    logger.info("-" * len(header))
    total_row = f"{'TOTAL':<20}"
    for tf in all_tfs:
        total = sum(tfs.get(tf, 0) for tfs in results.values())
        total_row += f"{total:>10,}"
    logger.info(total_row)


def main():
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV data for crypto trading pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 2 years of data for all configured pairs
  python scripts/download_data.py --days 730

  # Download specific pairs
  python scripts/download_data.py --pairs SOL/USDT:USDT BNB/USDT:USDT

  # Download with custom timeframes (9m will auto-aggregate from 1m)
  python scripts/download_data.py --timeframes 5m 9m 1h

  # Force include 1m data for future aggregation
  python scripts/download_data.py --include-1m
        """,
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        help="Trading pairs to download (default: all configured pairs)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        help="Timeframes to download (default: config LTF/HTF). Non-native timeframes (e.g., 9m) auto-aggregate from 1m.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="Days of history to download (default: 730 = 2 years)",
    )
    parser.add_argument(
        "--include-1m",
        action="store_true",
        help="Always download 1m data (needed for custom aggregation)",
    )

    args = parser.parse_args()

    logger.info("Crypto Data Downloader")
    logger.info(f"Days: {args.days}")
    logger.info(f"Pairs: {args.pairs or 'all configured'}")
    logger.info(f"Timeframes: {args.timeframes or 'config defaults'}")
    logger.info(f"Include 1m: {args.include_1m}")

    results = download_data(
        pairs=args.pairs,
        timeframes=args.timeframes,
        days=args.days,
        include_1m=args.include_1m,
    )

    print_summary(results)

    # Check for failures
    failures = sum(1 for tfs in results.values() for count in tfs.values() if count == 0)
    if failures:
        logger.warning(f"\n{failures} download(s) failed - check logs above")
        sys.exit(1)

    logger.info("\nDownload complete!")


if __name__ == "__main__":
    main()
