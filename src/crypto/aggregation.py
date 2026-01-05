"""
Timeframe aggregation for crypto OHLCV data.

Allows creating arbitrary timeframes (e.g., 9m) from 1m base data.
This is necessary because exchanges like ByBit don't provide non-standard timeframes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.crypto.config import CryptoConfig, load_config
from src.crypto.data_manager import DataManager

logger = logging.getLogger(__name__)


def aggregate_ohlcv(
    df: pd.DataFrame,
    target_minutes: int,
) -> pd.DataFrame:
    """
    Aggregate OHLCV data to a target timeframe.

    Args:
        df: Source DataFrame with OHLCV data (1m or any base timeframe)
            Must have datetime index and columns: open, high, low, close, volume
        target_minutes: Target timeframe in minutes (e.g., 9 for 9m)

    Returns:
        Aggregated DataFrame with proper OHLCV semantics:
        - Open: first open in window
        - High: max high in window
        - Low: min low in window
        - Close: last close in window
        - Volume: sum of volume in window
    """
    if df.empty:
        return df

    # Validate required columns
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Aggregation rules following OHLCV semantics
    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Resample to target timeframe
    rule = f"{target_minutes}min"
    resampled = df.resample(rule).agg(agg_rules)

    # Drop incomplete bars (bars with NaN values)
    resampled = resampled.dropna()

    logger.debug(
        f"Aggregated {len(df)} bars to {len(resampled)} bars ({target_minutes}m)"
    )

    return resampled


def aggregate_from_1m(
    symbol: str,
    target_minutes: int,
    config: Optional[CryptoConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Load 1m data and aggregate to target timeframe.

    Args:
        symbol: Trading pair (e.g., 'SUI/USDT:USDT')
        target_minutes: Target timeframe in minutes
        config: Configuration object
        save: If True, save aggregated data to parquet

    Returns:
        Aggregated OHLCV DataFrame
    """
    config = config or load_config()
    dm = DataManager(config)

    # Load 1m base data
    df_1m = dm.load_ohlcv(symbol, "1m")

    if df_1m.empty:
        logger.warning(f"No 1m data found for {symbol}")
        return pd.DataFrame()

    logger.info(f"Loaded {len(df_1m)} 1m bars for {symbol}")

    # Aggregate to target timeframe
    df_agg = aggregate_ohlcv(df_1m, target_minutes)

    if df_agg.empty:
        logger.warning(f"Aggregation produced empty result for {symbol}")
        return df_agg

    logger.info(f"Aggregated to {len(df_agg)} {target_minutes}m bars")

    # Save if requested
    if save:
        timeframe_str = f"{target_minutes}m"
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        path = config.data.raw_dir / safe_symbol / timeframe_str / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df_agg.to_parquet(path, compression=config.data.compression)
        logger.info(f"Saved {len(df_agg)} bars to {path}")

    return df_agg


def ensure_timeframe(
    symbol: str,
    timeframe: str,
    config: Optional[CryptoConfig] = None,
) -> pd.DataFrame:
    """
    Ensure data exists for a timeframe, aggregating from 1m if needed.

    This is the main entry point - it will:
    1. Check if the timeframe data already exists
    2. If not, check if 1m data exists
    3. If 1m exists, aggregate to target timeframe
    4. Return the data

    Args:
        symbol: Trading pair
        timeframe: Target timeframe (e.g., '9m', '15m')
        config: Configuration object

    Returns:
        OHLCV DataFrame for the requested timeframe
    """
    config = config or load_config()
    dm = DataManager(config)

    # Standard exchange timeframes that don't need aggregation
    standard_timeframes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
    }

    # Parse target minutes
    if timeframe in standard_timeframes:
        target_minutes = standard_timeframes[timeframe]
    elif timeframe.endswith("m"):
        target_minutes = int(timeframe[:-1])
    elif timeframe.endswith("h"):
        target_minutes = int(timeframe[:-1]) * 60
    else:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

    # Try loading existing data first (auto_aggregate=False to prevent recursion)
    df = dm.load_ohlcv(symbol, timeframe, auto_aggregate=False)
    if not df.empty and len(df) > 100:  # Arbitrary threshold for "enough data"
        logger.debug(f"Found existing {timeframe} data for {symbol}: {len(df)} bars")
        return df

    # Need to aggregate from 1m
    logger.info(f"No {timeframe} data found for {symbol}, aggregating from 1m")
    return aggregate_from_1m(symbol, target_minutes, config, save=True)


def aggregate_all_pairs(
    target_minutes: int,
    config: Optional[CryptoConfig] = None,
) -> dict[str, pd.DataFrame]:
    """
    Aggregate all configured pairs to target timeframe.

    Args:
        target_minutes: Target timeframe in minutes
        config: Configuration object

    Returns:
        Dict mapping symbol to aggregated DataFrame
    """
    config = config or load_config()
    results = {}

    for symbol in config.pairs.pairs:
        try:
            df = aggregate_from_1m(symbol, target_minutes, config, save=True)
            if not df.empty:
                results[symbol] = df
        except Exception as e:
            logger.error(f"Error aggregating {symbol}: {e}")

    return results


def validate_aggregation(
    df_source: pd.DataFrame,
    df_agg: pd.DataFrame,
    target_minutes: int,
) -> dict:
    """
    Validate aggregation correctness.

    Checks:
    1. Bar count ratio is approximately correct
    2. Volume totals match
    3. Price ranges are preserved (agg high >= source highs, etc.)

    Returns:
        Dict with validation results
    """
    if df_source.empty or df_agg.empty:
        return {"valid": False, "error": "Empty dataframe"}

    expected_ratio = target_minutes
    actual_ratio = len(df_source) / len(df_agg)

    # Allow some tolerance for incomplete bars at edges
    ratio_valid = 0.9 * expected_ratio <= actual_ratio <= 1.1 * expected_ratio

    # Volume should roughly match (allowing for edge effects)
    source_vol = df_source["volume"].sum()
    agg_vol = df_agg["volume"].sum()
    vol_ratio = agg_vol / source_vol if source_vol > 0 else 0
    vol_valid = 0.95 <= vol_ratio <= 1.05

    # Price range check - aggregated highs should cover source highs
    source_high = df_source["high"].max()
    source_low = df_source["low"].min()
    agg_high = df_agg["high"].max()
    agg_low = df_agg["low"].min()
    price_valid = agg_high >= source_high * 0.9999 and agg_low <= source_low * 1.0001

    return {
        "valid": ratio_valid and vol_valid and price_valid,
        "bar_ratio": actual_ratio,
        "expected_ratio": expected_ratio,
        "ratio_valid": ratio_valid,
        "volume_ratio": vol_ratio,
        "vol_valid": vol_valid,
        "price_valid": price_valid,
        "source_bars": len(df_source),
        "agg_bars": len(df_agg),
    }


if __name__ == "__main__":
    # Demo: Aggregate all pairs to 9m
    logging.basicConfig(level=logging.INFO)

    config = load_config()

    print("=" * 60)
    print("Aggregating 1m data to 9m for all pairs")
    print("=" * 60)

    results = aggregate_all_pairs(9, config)

    print("\nResults:")
    for symbol, df in results.items():
        print(f"  {symbol}: {len(df)} bars")
        if not df.empty:
            print(f"    Range: {df.index[0]} to {df.index[-1]}")

    # Validate
    print("\nValidation:")
    dm = DataManager(config)
    for symbol in results:
        df_1m = dm.load_ohlcv(symbol, "1m")
        df_9m = results[symbol]
        validation = validate_aggregation(df_1m, df_9m, 9)
        status = "PASS" if validation["valid"] else "FAIL"
        print(f"  {symbol}: {status}")
        print(f"    Bar ratio: {validation['bar_ratio']:.2f} (expected ~9)")
        print(f"    Volume ratio: {validation['volume_ratio']:.4f}")
