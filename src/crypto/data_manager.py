"""
Data management for crypto RL trading system.

Handles:
- CCXT integration for ByBit perpetuals
- OHLCV data fetching (historical + incremental)
- Parquet storage with partitioning
- Dynamic fee fetching and caching
- Multi-pair parallel operations
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.crypto.config import CryptoConfig, load_config

logger = logging.getLogger(__name__)


class DataManager:
    """Manages OHLCV data fetching, storage, and retrieval for crypto perpetuals."""

    # CCXT timeframe to milliseconds mapping
    TIMEFRAME_MS = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "4h": 14_400_000,
        "6h": 21_600_000,
        "12h": 43_200_000,
        "1d": 86_400_000,
    }

    def __init__(self, config: Optional[CryptoConfig] = None):
        self.config = config or load_config()
        self._exchange: Optional[ccxt.Exchange] = None
        self._fee_cache: Dict[str, Tuple[float, datetime]] = {}

    @staticmethod
    def add_session_boundaries(
        df: pd.DataFrame,
        timezone_str: str = "UTC",
    ) -> pd.DataFrame:
        """
        Add session boundary columns to OHLCV DataFrame.

        Detects trading session boundaries based on calendar day changes.
        Used for intraday box strategy features.

        Adds columns:
        - session_id: int, incrementing integer per trading session (0-indexed)
        - session_start: bool, True for first bar of each session
        - session_date: date, the date of the session in specified timezone

        Args:
            df: OHLCV DataFrame with datetime index (must be sorted ascending)
            timezone_str: Timezone for session boundaries (default 'UTC')

        Returns:
            DataFrame with session columns added (original columns preserved)

        Raises:
            ValueError: If DataFrame is empty or index is not datetime

        Note:
            - Data gaps spanning multiple days increment session_id by 1 (not by gap size)
            - First bar is always session_start=True
            - Duplicate timestamps are treated as same session
        """
        if df.empty:
            raise ValueError("Cannot add session boundaries to empty DataFrame")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        # Make a copy to avoid modifying original
        result = df.copy()

        # Ensure index is sorted (critical for session detection)
        if not result.index.is_monotonic_increasing:
            result = result.sort_index()

        # Convert to target timezone if needed
        idx = result.index
        if idx.tz is None:
            # Assume UTC for timezone-naive timestamps
            idx = idx.tz_localize("UTC")

        if timezone_str != "UTC":
            idx = idx.tz_convert(timezone_str)

        # Extract dates for session boundary detection
        dates = idx.date

        # Detect session changes (where date differs from previous)
        # First element: use shift which creates NaT, compared to date gives True
        dates_series = pd.Series(dates, index=result.index)
        session_changes = dates_series != dates_series.shift(1)

        # First bar is always a session start
        session_changes.iloc[0] = True

        # Session ID: cumulative sum of changes, minus 1 for 0-indexing
        session_id = session_changes.cumsum() - 1

        # Add columns
        result["session_id"] = session_id.astype(np.int32)
        result["session_start"] = session_changes
        result["session_date"] = dates_series

        return result

    @property
    def exchange(self) -> ccxt.Exchange:
        """Lazy-load exchange connection."""
        if self._exchange is None:
            exchange_class = getattr(ccxt, self.config.exchange.name)
            options = {
                "defaultType": "swap",  # For perpetuals
                "options": {"defaultType": "swap"},
            }

            if self.config.exchange.testnet:
                options["sandbox"] = True

            if self.config.exchange.api_key:
                options["apiKey"] = self.config.exchange.api_key
                options["secret"] = self.config.exchange.api_secret

            self._exchange = exchange_class(options)
            self._exchange.load_markets()
            logger.info(f"Connected to {self.config.exchange.name} ({'testnet' if self.config.exchange.testnet else 'mainnet'})")

        return self._exchange

    def _get_parquet_path(self, symbol: str, timeframe: str) -> Path:
        """Get parquet file path for symbol/timeframe."""
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        return self.config.data.raw_dir / safe_symbol / timeframe / "data.parquet"

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange.

        Args:
            symbol: Trading pair (e.g., 'SOL/USDT:USDT')
            timeframe: Candle timeframe (e.g., '15m', '1h')
            since: Start timestamp in milliseconds
            limit: Max candles per request (ByBit max: 1000)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=since,
                limit=limit,
                params={"category": "linear"},  # ByBit linear perpetuals
            )

            if not ohlcv:
                return pd.DataFrame()

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")

            # Ensure numeric types
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {symbol}: {e}")
            raise

    def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a date range (handles pagination).

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start: Start datetime (UTC)
            end: End datetime (UTC), defaults to now

        Returns:
            DataFrame with complete OHLCV data
        """
        if end is None:
            end = datetime.now(timezone.utc)

        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        tf_ms = self.TIMEFRAME_MS.get(timeframe, 60_000)

        all_data = []
        current_since = since_ms

        while current_since < end_ms:
            df = self.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)

            if df.empty:
                break

            all_data.append(df)

            # Move to next batch
            last_ts = int(df.index[-1].timestamp() * 1000)
            current_since = last_ts + tf_ms

            # Rate limiting
            if self.config.exchange.rate_limit_per_second > 0:
                import time
                time.sleep(1.0 / self.config.exchange.rate_limit_per_second)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data)
        result = result[~result.index.duplicated(keep="first")]
        result = result.sort_index()

        # Filter to requested range
        result = result[(result.index >= start) & (result.index <= end)]

        return result

    def save_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Save OHLCV data to parquet, merging with existing data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair
            timeframe: Candle timeframe
        """
        if df.empty:
            logger.warning(f"Empty DataFrame, skipping save for {symbol} {timeframe}")
            return

        path = self._get_parquet_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Merge with existing data if present
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

        # Save with compression
        df.to_parquet(path, compression=self.config.data.compression)
        logger.info(f"Saved {len(df)} bars for {symbol} {timeframe}")

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        auto_aggregate: bool = True,
        add_session_info: bool = False,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from parquet.

        For non-native timeframes (e.g., '9m'), will automatically aggregate
        from 1m data if available and auto_aggregate=True.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe (supports custom like '9m')
            start: Optional start filter
            end: Optional end filter
            auto_aggregate: If True, aggregate from 1m for non-native timeframes
            add_session_info: If True, add session boundary columns (session_id,
                              session_start, session_date) for intraday box features

        Returns:
            DataFrame with OHLCV data (and optionally session columns)
        """
        from src.crypto.config import TimeframeConfig

        path = self._get_parquet_path(symbol, timeframe)
        df = pd.DataFrame()

        # Try loading existing data first
        if path.exists():
            df = pd.read_parquet(path)
            if not df.empty:
                # Apply filters
                if start is not None:
                    df = df[df.index >= start]
                if end is not None:
                    df = df[df.index <= end]

        # No existing data - try aggregation for non-native timeframes
        if df.empty and auto_aggregate and not TimeframeConfig.is_native(timeframe):
            # Import here to avoid circular imports
            from src.crypto.aggregation import ensure_timeframe

            logger.info(f"Auto-aggregating {symbol} {timeframe} from 1m data")
            df = ensure_timeframe(symbol, timeframe, self.config)

            if not df.empty:
                # Apply filters
                if start is not None:
                    df = df[df.index >= start]
                if end is not None:
                    df = df[df.index <= end]

        if df.empty:
            logger.warning(f"No data found for {symbol} {timeframe}")
            return pd.DataFrame()

        # Add session boundary info if requested
        if add_session_info:
            timezone_str = self.config.data.session_timezone
            df = self.add_session_boundaries(df, timezone_str)

        return df

    def get_bar_count(self, symbol: str, timeframe: str) -> int:
        """Get number of stored bars for symbol/timeframe."""
        path = self._get_parquet_path(symbol, timeframe)
        if not path.exists():
            return 0
        return len(pd.read_parquet(path))

    def is_data_fresh(
        self,
        symbol: str,
        timeframe: str,
        max_age_bars: int = 10,
    ) -> bool:
        """Check if stored data is recent enough."""
        path = self._get_parquet_path(symbol, timeframe)
        if not path.exists():
            return False

        df = pd.read_parquet(path)
        if df.empty:
            return False

        last_ts = df.index[-1]
        tf_ms = self.TIMEFRAME_MS.get(timeframe, 60_000)
        now = datetime.now(timezone.utc)

        # Data is fresh if last bar is within max_age_bars
        age_ms = (now - last_ts).total_seconds() * 1000
        max_age_ms = max_age_bars * tf_ms

        return age_ms < max_age_ms

    def update_ohlcv(self, symbol: str, timeframe: str) -> int:
        """
        Update stored data with latest candles (incremental).

        Returns:
            Number of new bars added
        """
        path = self._get_parquet_path(symbol, timeframe)

        if path.exists():
            existing = pd.read_parquet(path)
            since = existing.index[-1] if not existing.empty else None
        else:
            since = None

        if since:
            since_ms = int(since.timestamp() * 1000)
        else:
            # Default to last 30 days
            since_ms = int((datetime.now(timezone.utc).timestamp() - 30 * 86400) * 1000)

        new_df = self.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)

        if new_df.empty:
            return 0

        initial_count = self.get_bar_count(symbol, timeframe)
        self.save_ohlcv(new_df, symbol, timeframe)
        final_count = self.get_bar_count(symbol, timeframe)

        return final_count - initial_count

    def fetch_fee(self, symbol: str, use_cache: bool = True) -> float:
        """
        Fetch taker fee for symbol.

        Args:
            symbol: Trading pair
            use_cache: Whether to use cached value

        Returns:
            Taker fee as decimal (e.g., 0.00055 for 0.055%)
        """
        if use_cache and symbol in self._fee_cache:
            cached_fee, cached_time = self._fee_cache[symbol]
            age = (datetime.now(timezone.utc) - cached_time).total_seconds()
            if age < self.config.fees.fee_cache_ttl_seconds:
                return cached_fee

        try:
            markets = self.exchange.fetch_markets()
            for market in markets:
                if market["symbol"] == symbol:
                    taker_fee = market.get("taker", self.config.fees.default_taker_fee)
                    self._fee_cache[symbol] = (taker_fee, datetime.now(timezone.utc))
                    logger.info(f"Fee for {symbol}: {taker_fee:.5f}")
                    return taker_fee

            logger.warning(f"Symbol {symbol} not found, using default fee")
            return self.config.fees.default_taker_fee

        except Exception as e:
            logger.warning(f"Error fetching fee for {symbol}: {e}, using default")
            return self.config.fees.default_taker_fee

    def fetch_multi_timeframe(
        self,
        symbol: str,
        ltf: Optional[str] = None,
        htf: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch both LTF and HTF data for a symbol.

        Returns:
            Tuple of (ltf_df, htf_df)
        """
        ltf = ltf or self.config.timeframes.ltf
        htf = htf or self.config.timeframes.htf

        ltf_df = self.load_ohlcv(symbol, ltf)
        htf_df = self.load_ohlcv(symbol, htf)

        return ltf_df, htf_df

    def fetch_all_pairs(
        self,
        timeframe: str,
        update: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for all configured pairs.

        Args:
            timeframe: Candle timeframe
            update: If True, fetch latest data first

        Returns:
            Dict mapping symbol to DataFrame
        """
        result = {}

        for symbol in self.config.pairs.pairs:
            if update:
                self.update_ohlcv(symbol, timeframe)

            df = self.load_ohlcv(symbol, timeframe)
            if not df.empty:
                result[symbol] = df

        return result

    def collect_historical(
        self,
        symbol: str,
        timeframe: str,
        days: int = 365,
    ) -> pd.DataFrame:
        """
        Collect historical data for training.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days of history

        Returns:
            DataFrame with historical data
        """
        end = datetime.now(timezone.utc)
        start = datetime.fromtimestamp(end.timestamp() - days * 86400, tz=timezone.utc)

        logger.info(f"Collecting {days} days of {symbol} {timeframe} data...")
        df = self.fetch_ohlcv_range(symbol, timeframe, start, end)

        if not df.empty:
            self.save_ohlcv(df, symbol, timeframe)
            logger.info(f"Collected {len(df)} bars for {symbol} {timeframe}")

        return df

    def validate_data(self, symbol: str, timeframe: str) -> Dict[str, any]:
        """
        Validate stored data quality.

        Returns:
            Dict with validation results
        """
        df = self.load_ohlcv(symbol, timeframe)

        if df.empty:
            return {"valid": False, "error": "No data"}

        # Check for gaps
        tf_ms = self.TIMEFRAME_MS.get(timeframe, 60_000)
        expected_delta = pd.Timedelta(milliseconds=tf_ms)
        gaps = df.index.to_series().diff()[1:] > expected_delta * 1.5

        # Check for NaN values
        nan_count = df.isna().sum().sum()

        # Check OHLC validity
        invalid_ohlc = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        ).sum()

        return {
            "valid": gaps.sum() == 0 and nan_count == 0 and invalid_ohlc == 0,
            "bars": len(df),
            "gaps": int(gaps.sum()),
            "nan_count": int(nan_count),
            "invalid_ohlc": int(invalid_ohlc),
            "start": df.index[0].isoformat() if not df.empty else None,
            "end": df.index[-1].isoformat() if not df.empty else None,
        }


def collect_training_data(
    pairs: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    days: int = 365,
    config: Optional[CryptoConfig] = None,
) -> None:
    """
    Convenience function to collect training data for all pairs and timeframes.

    Args:
        pairs: List of symbols (defaults to config)
        timeframes: List of timeframes (defaults to config LTF + HTF)
        days: Days of history
        config: Configuration object
    """
    config = config or load_config()
    dm = DataManager(config)

    pairs = pairs or config.pairs.pairs
    timeframes = timeframes or [config.timeframes.ltf, config.timeframes.htf]

    for symbol in pairs:
        for tf in timeframes:
            try:
                dm.collect_historical(symbol, tf, days)
            except Exception as e:
                logger.error(f"Error collecting {symbol} {tf}: {e}")


if __name__ == "__main__":
    # Demo: collect data for all pairs
    logging.basicConfig(level=logging.INFO)

    config = load_config()
    dm = DataManager(config)

    # Show current status
    for symbol in config.pairs.pairs:
        for tf in [config.timeframes.ltf, config.timeframes.htf]:
            count = dm.get_bar_count(symbol, tf)
            fresh = dm.is_data_fresh(symbol, tf)
            print(f"{symbol} {tf}: {count} bars, fresh={fresh}")
