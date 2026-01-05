#!/usr/bin/env python3
"""
Support Resistance Swing Strategy v7.5.5 - Python Implementation

Converted from Pine Script indicator to Python for live trading signal detection.
Connects to exchange via WebSocket (ccxt pro), prints buy/sell signals,
and stores all alerts in SQLite.

Usage:
    python sr_swing_strategy.py --config config.yaml
    python sr_swing_strategy.py  # Uses default config
"""

import asyncio
import os
import argparse
import csv
import json
import logging
import math
import sqlite3
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Deque

import numpy as np
import yaml

# Try to import ccxt pro, fall back to regular ccxt
try:
    import ccxt.pro as ccxtpro
    HAS_CCXT_PRO = True
except ImportError:
    import ccxt
    HAS_CCXT_PRO = False
    print("Warning: ccxt.pro not available, falling back to REST polling")

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# BACKTEST DB RESET
# =============================================================================
def reset_signal_db(db_path: str, log: bool = True) -> None:
    """Delete the signal DB before a backtest to avoid cross-run contamination."""
    import os

    if not db_path or db_path == ":memory:":
        return

    if os.path.exists(db_path) and os.path.isfile(db_path):
        os.remove(db_path)
        if log:
            logger.info(f"Reset backtest database: {db_path}")

# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================
@dataclass
class Config:
    """All configuration parameters matching Pine Script inputs"""

    # --- Connection ---
    symbol: str = "ETH/USDT"
    timeframe: str = "5m"
    exchange: str = "binance"
    exchange_market: str = "auto"  # "auto", "spot", "swap", "linear", "perp", "futures"

    # --- Calculations ---
    pivot_left_bars: int = 4
    pivot_right_bars: int = 3
    use_high_low: bool = True
    enable_support_entries: bool = True
    enable_resistance_entries: bool = True

    # --- Pivot Strength Filter ---
    enable_pivot_strength: bool = False
    min_touches: int = 2
    touch_tolerance: float = 0.15

    # --- Trade Management ---
    enable_longs: bool = True
    enable_shorts: bool = True
    confirm_bars: int = 1
    exit_confirm_bars: int = 0
    enable_sar: bool = False
    tp_percent: float = 2.0
    use_stop_loss: bool = True
    sl_percent: float = 4.0
    min_profit_percent: float = 1.0
    use_trailing: bool = True
    trailing_activation: float = 0.1
    trail_callback: float = 0.05
    trail_protect_entry: bool = True
    use_time_exit: bool = False
    time_exit_minutes: int = 90
    exchange_fee: float = 0.06

    # --- Trade Management - Trend Aligned ---
    use_trend_mgt: bool = False
    trend_tp_behavior: str = "Unlimited"  # "Original", "Override", "Unlimited"
    trend_tp_override: float = 1.0
    trend_sl_behavior: str = "Override"  # "Original", "Override"
    trend_sl_override: float = 4.0
    trend_exit_on_flip: bool = False

    # --- Trend Filter ---
    trend_method: str = "HMA"  # "Kernel", "EMA", "SMA", "HMA", "ALMA", "Gaussian", "Kalman"
    trend_length: int = 14
    kalman_short_len: int = 18
    kalman_long_len: int = 23
    trend_sigma: float = 12.0
    trend_poles: int = 2
    trend_offset: float = 0.85
    reverse_trend: bool = False
    use_trend_signals: bool = True

    # --- MTF Filter ---
    enable_mtf: bool = True
    mtf_timeframe: str = "7"
    mtf_method: str = "Kalman"
    mtf_length: int = 31
    mtf_kalman_short: int = 17
    mtf_kalman_long: int = 28
    mtf_sigma: float = 6.0
    mtf_alma_offset: float = 0.85
    mtf_strict_mode: bool = True
    mtf_exit_on_flip: bool = True

    # --- Momentum Capture ---
    enable_momentum_mode: bool = True
    enable_breakout: bool = True
    breakout_lookback: int = 40
    breakout_vol_multiplier: float = 1.1
    enable_velocity: bool = True
    velocity_threshold: float = 3.5
    velocity_bars: int = 3
    enable_atr_expansion: bool = False
    atr_multiplier: float = 3.5
    atr_length: int = 14
    momentum_sl_percent: float = 2.0
    momentum_tp_percent: float = 1.0
    enable_reverse_momentum_exit: bool = True

    # --- Momentum Capture - MFI ---
    enable_mfi_momentum: bool = False
    mfi_length: int = 14
    mfi_momentum_threshold: int = 50

    # --- Market Gravity ---
    gravity_mode: str = "Auto (Trend)"  # "Auto (None)", "Auto (Trend)", "Bullish", "Bearish", "Sideways"
    gravity_sl_multiplier: float = 2.0
    gravity_disable_trail: bool = True
    gravity_disable_flip_exit: bool = True
    gravity_tp_behavior: str = "Unlimited"  # "Original", "Unlimited", "2x Target"
    gravity_block_counter: bool = True

    # --- Box Theory ---
    enable_box_theory: bool = False
    box_method: str = "Darvas"  # "Darvas", "High/Low (MTF)"
    box_timeframe: str = "D"
    box_expand: bool = True
    box_lookback: int = 5
    box_sensitivity: float = 0.5
    box_source: str = "High/Low"  # "High/Low", "Close"
    box_trend_filter: bool = False
    box_filter_mode: str = "Strict"  # "Strict", "All"

    # --- Momentum Oscillator Filter ---
    enable_oscillator_filter: bool = False
    oscillator_type: str = "MACD"  # "MACD", "Zero Lag Score"
    macd_fast_len: int = 12
    macd_slow_len: int = 26
    macd_signal_len: int = 9
    macd_entry_mode: str = "Above Zero Line"  # "Above Zero Line", "Below Zero Line", "Signal Cross", "Histogram Positive", "Any"
    macd_exit_on_histogram_peak: bool = False
    macd_exit_on_zero_cross: bool = False
    macd_exit_timeframe: str = "Main"  # "Main", "MTF", "Custom"
    macd_custom_timeframe: str = "15m"
    macd_exit_takes_precedence: bool = True

    # --- Zero Lag Settings ---
    zl_length: int = 50
    zl_volatility_mult: float = 1.5
    zl_loop_start: int = 1
    zl_loop_end: int = 70
    zl_threshold_up: int = 5
    zl_threshold_down: int = -5
    zl_entry_mode: str = "Score + Volatility"  # "Score Only", "Score + Volatility", "ZL Line Direction", "Price vs ZL Line"
    zl_exit_on_trend_flip: bool = False

    # --- RSI Filter ---
    enable_rsi_filter: bool = True
    rsi_length: int = 14
    rsi_oversold: int = 35
    rsi_overbought: int = 65
    rsi_filter_mode: str = "Avoid Extremes"  # "Avoid Extremes", "Momentum Aligned", "Counter-Trend Bounce", "Any"

    # --- Bollinger Bands ---
    enable_bb: bool = False
    bb_length: int = 20
    bb_mult: float = 2.0
    bb_use_filter: bool = True
    enable_bb_squeeze: bool = True
    bb_squeeze_threshold: float = 0.5

    # --- MA Filter ---
    ma_filter_entry: bool = False
    ma_filter_exit: bool = False
    ma_filter_type: str = "SMA"  # "SMA", "EMA"
    ma_filter_period: int = 200
    ma_filter_timeframe: str = "D"

    # --- Database ---
    db_path: str = "signals.db"

    # --- Backtest Data ---
    data_check_months: int = 3

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


def normalize_exchange_id(exchange: str) -> str:
    return (exchange or "").strip().lower()


def normalize_exchange_market(exchange_id: str, market: str, symbol: str) -> str:
    market_hint = (market or "auto").strip().lower()
    symbol_hint = (symbol or "").strip().upper()
    is_tv_perp = symbol_hint.endswith(".P")

    if exchange_id == "bybit":
        if market_hint in ("perp", "linear", "swap", "futures"):
            return "linear"
        if market_hint == "auto":
            if is_tv_perp or ":" in symbol_hint:
                return "linear"
            return "spot"
        return market_hint

    if market_hint == "auto":
        return "spot"
    if market_hint == "perp":
        return "swap"
    return market_hint


def normalize_symbol_for_exchange(symbol: str, exchange_id: str, market_kind: str) -> str:
    sym = (symbol or "").strip()
    if ":" in sym and "/" not in sym:
        prefix, rest = sym.split(":", 1)
        if prefix.isalpha():
            sym = rest
    if sym.upper().endswith(".P"):
        sym = sym[:-2]

    if "/" in sym and ":" in sym:
        base, rest = sym.split("/", 1)
        quote, settle = rest.split(":", 1)
        return f"{base.upper()}/{quote.upper()}:{settle.upper()}"

    base = ""
    quote = ""
    if "/" in sym:
        base, quote = sym.split("/", 1)
    else:
        upper = sym.upper()
        for candidate in ("USDT", "USDC", "USD", "BTC", "ETH"):
            if upper.endswith(candidate) and len(upper) > len(candidate):
                base = upper[:-len(candidate)]
                quote = candidate
                break

    if not base or not quote:
        return sym

    base = base.upper()
    quote = quote.upper()
    if exchange_id == "bybit" and market_kind in ("linear", "swap", "perp", "futures"):
        return f"{base}/{quote}:{quote}"
    return f"{base}/{quote}"


def get_exchange_symbol_config(config: Config) -> tuple[str, str, str]:
    exchange_id = normalize_exchange_id(config.exchange)
    market_kind = normalize_exchange_market(exchange_id, config.exchange_market, config.symbol)
    ccxt_symbol = normalize_symbol_for_exchange(config.symbol, exchange_id, market_kind)
    symbol_slug = ccxt_symbol.replace("/", "_").replace(":", "_")
    data_prefix = f"{exchange_id}_{market_kind}_{symbol_slug}"
    return ccxt_symbol, data_prefix, market_kind


def build_exchange_options(exchange_id: str, market_kind: str) -> dict:
    options = {
        'enableRateLimit': True,
    }
    if exchange_id == "bybit":
        bybit_opts = {}
        if market_kind in ("linear", "swap", "perp", "futures"):
            bybit_opts['defaultType'] = 'swap'
            bybit_opts['defaultSubType'] = 'linear'
        else:
            bybit_opts['defaultType'] = 'spot'
        options['options'] = bybit_opts
    return options


def get_ohlcv_params(exchange_id: str, market_kind: str) -> dict:
    if exchange_id == "bybit" and market_kind in ("linear", "swap", "perp", "futures"):
        return {'category': 'linear'}
    return {}


def adjust_ohlcv_limit(exchange_id: str, limit: int) -> int:
    if exchange_id == "bybit" and limit > 200:
        return 200
    return limit


# =============================================================================
# CANDLE DATA STRUCTURE
# =============================================================================
@dataclass
class Candle:
    """OHLCV candle data"""
    timestamp: int  # Unix timestamp in ms
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_ccxt(cls, data: list) -> 'Candle':
        """Create from ccxt OHLCV format [timestamp, open, high, low, close, volume]"""
        return cls(
            timestamp=int(data[0]),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5])
        )


# =============================================================================
# SQLITE DATABASE
# =============================================================================
class SignalDatabase:
    """SQLite database for storing trading signals"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                candle_time INTEGER,
                symbol TEXT,
                timeframe TEXT,
                signal_type TEXT,
                price REAL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                is_momentum_trade BOOLEAN,
                exit_reason TEXT,
                pnl_percent REAL,
                trend_direction TEXT,
                mtf_direction TEXT,
                rsi_value REAL,
                mfi_value REAL,
                macd_value REAL,
                macd_histogram REAL,
                gravity_mode TEXT,
                box_state TEXT,
                candle_data TEXT,
                filters_snapshot TEXT,
                config_hash TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_signal_id INTEGER,
                exit_signal_id INTEGER,
                symbol TEXT,
                direction INTEGER,
                entry_time INTEGER,
                exit_time INTEGER,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                pnl_percent REAL,
                duration_bars INTEGER,
                is_momentum_trade BOOLEAN,
                exit_reason TEXT,
                FOREIGN KEY (entry_signal_id) REFERENCES signals(id),
                FOREIGN KEY (exit_signal_id) REFERENCES signals(id)
            )
        ''')

        self.conn.commit()

    def store_signal(self, signal_data: Dict[str, Any]) -> int:
        """Store a signal and return its ID"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO signals (
                candle_time, symbol, timeframe, signal_type, price,
                entry_price, stop_loss, take_profit, is_momentum_trade,
                exit_reason, pnl_percent, trend_direction, mtf_direction,
                rsi_value, mfi_value, macd_value, macd_histogram,
                gravity_mode, box_state, candle_data, filters_snapshot, config_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_data.get('candle_time'),
            signal_data.get('symbol'),
            signal_data.get('timeframe'),
            signal_data.get('signal_type'),
            signal_data.get('price'),
            signal_data.get('entry_price'),
            signal_data.get('stop_loss'),
            signal_data.get('take_profit'),
            signal_data.get('is_momentum_trade'),
            signal_data.get('exit_reason'),
            signal_data.get('pnl_percent'),
            signal_data.get('trend_direction'),
            signal_data.get('mtf_direction'),
            signal_data.get('rsi_value'),
            signal_data.get('mfi_value'),
            signal_data.get('macd_value'),
            signal_data.get('macd_histogram'),
            signal_data.get('gravity_mode'),
            signal_data.get('box_state'),
            json.dumps(signal_data.get('candle_data', {})),
            json.dumps(signal_data.get('filters_snapshot', {})),
            signal_data.get('config_hash')
        ))
        self.conn.commit()
        return cursor.lastrowid

    def store_trade(self, trade_data: Dict[str, Any]) -> int:
        """Store a completed trade"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades (
                entry_signal_id, exit_signal_id, symbol, direction,
                entry_time, exit_time, entry_price, exit_price,
                stop_loss, take_profit, pnl_percent, duration_bars,
                is_momentum_trade, exit_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data.get('entry_signal_id'),
            trade_data.get('exit_signal_id'),
            trade_data.get('symbol'),
            trade_data.get('direction'),
            trade_data.get('entry_time'),
            trade_data.get('exit_time'),
            trade_data.get('entry_price'),
            trade_data.get('exit_price'),
            trade_data.get('stop_loss'),
            trade_data.get('take_profit'),
            trade_data.get('pnl_percent'),
            trade_data.get('duration_bars'),
            trade_data.get('is_momentum_trade'),
            trade_data.get('exit_reason')
        ))
        self.conn.commit()
        return cursor.lastrowid

    def close(self):
        """Close database connection"""
        self.conn.close()


def fetch_trades(db: SignalDatabase) -> list:
    """Fetch all trades from the database."""
    cursor = db.conn.cursor()
    cursor.execute('''
        SELECT
            entry_time, exit_time, direction, entry_price, exit_price,
            stop_loss, take_profit, pnl_percent, duration_bars,
            is_momentum_trade, exit_reason
        FROM trades
        ORDER BY entry_time ASC
    ''')
    return cursor.fetchall()


def print_all_trades(db: SignalDatabase) -> None:
    """Print all trades in a readable table."""
    trades = fetch_trades(db)
    if not trades:
        print("\nNo trades to display.\n")
        return

    print("\n" + "=" * 100)
    print(" " * 35 + "TRADE LIST")
    print("=" * 100)
    print(f"{'Entry Time':<20} {'Exit Time':<20} {'Dir':<5} {'Entry':>10} {'Exit':>10} {'PnL%':>8} {'Reason':<15} {'Type':<8}")
    print("-" * 100)

    for t in trades:
        entry_ts, exit_ts, direction, entry_px, exit_px, _, _, pnl, _, is_mom, reason = t
        entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M') if entry_ts else "N/A"
        exit_time = datetime.fromtimestamp(exit_ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M') if exit_ts else "N/A"
        dir_str = "LONG" if direction == 1 else "SHORT"
        trade_type = "MOMENTUM" if is_mom else "PIVOT"
        reason_str = reason or "N/A"
        print(f"{entry_time:<20} {exit_time:<20} {dir_str:<5} {entry_px:>10.4f} {exit_px:>10.4f} {pnl:>+8.2f} {reason_str:<15} {trade_type:<8}")

    print("=" * 100 + "\n")


def export_trades_csv(db: SignalDatabase, path: str) -> None:
    """Export all trades to a CSV file."""
    trades = fetch_trades(db)
    header = [
        "entry_time",
        "exit_time",
        "direction",
        "entry_price",
        "exit_price",
        "stop_loss",
        "take_profit",
        "pnl_percent",
        "duration_bars",
        "is_momentum_trade",
        "exit_reason",
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in trades:
            entry_ts, exit_ts, direction, entry_px, exit_px, sl, tp, pnl, duration, is_mom, reason = t
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat() if entry_ts else ""
            exit_time = datetime.fromtimestamp(exit_ts / 1000, tz=timezone.utc).isoformat() if exit_ts else ""
            dir_str = "LONG" if direction == 1 else "SHORT"
            writer.writerow([
                entry_time,
                exit_time,
                dir_str,
                entry_px,
                exit_px,
                sl,
                tp,
                pnl,
                duration,
                int(bool(is_mom)),
                reason or "",
            ])


# =============================================================================
# TECHNICAL INDICATOR FUNCTIONS
# =============================================================================

def nz(value: Optional[float], default: float = 0.0) -> float:
    """Pine Script nz() - return default if value is None or NaN"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return value


def ema(values: List[float], length: int) -> float:
    """Calculate EMA for the most recent value"""
    if len(values) < length:
        return values[-1] if values else 0.0

    multiplier = 2.0 / (length + 1)
    ema_val = values[0]

    for i in range(1, len(values)):
        ema_val = (values[i] * multiplier) + (ema_val * (1 - multiplier))

    return ema_val


def sma(values: List[float], length: int) -> float:
    """Calculate SMA"""
    if len(values) < length:
        return sum(values) / len(values) if values else 0.0
    return sum(values[-length:]) / length


def wma(values: List[float], length: int) -> float:
    """Calculate WMA (Weighted Moving Average)"""
    if len(values) < length:
        length = len(values)
    if length == 0:
        return 0.0

    recent = values[-length:]
    weights = list(range(1, length + 1))
    weighted_sum = sum(v * w for v, w in zip(recent, weights))
    weight_total = sum(weights)

    return weighted_sum / weight_total


def hma(values: List[float], length: int) -> float:
    """Calculate HMA (Hull Moving Average)"""
    if len(values) < length:
        return values[-1] if values else 0.0

    half_length = max(1, length // 2)
    sqrt_length = max(1, int(math.sqrt(length)))

    # Calculate WMA of half period
    wma_half = []
    for i in range(sqrt_length):
        idx = len(values) - sqrt_length + i
        if idx >= half_length:
            wma_half.append(wma(values[:idx + 1], half_length))
        else:
            wma_half.append(values[idx] if idx >= 0 else values[0])

    # Calculate WMA of full period
    wma_full = []
    for i in range(sqrt_length):
        idx = len(values) - sqrt_length + i
        if idx >= length:
            wma_full.append(wma(values[:idx + 1], length))
        else:
            wma_full.append(values[idx] if idx >= 0 else values[0])

    # Raw HMA values: 2 * WMA(half) - WMA(full)
    raw_hma = [2 * h - f for h, f in zip(wma_half, wma_full)]

    # Final HMA: WMA of raw values
    return wma(raw_hma, sqrt_length)


def alma(values: List[float], length: int, offset: float = 0.85, sigma: float = 6.0) -> float:
    """Calculate ALMA (Arnaud Legoux Moving Average)"""
    if len(values) < length:
        return values[-1] if values else 0.0

    recent = values[-length:]
    m = offset * (length - 1)
    s = length / sigma

    weights = []
    for i in range(length):
        weights.append(math.exp(-((i - m) ** 2) / (2 * s * s)))

    weight_sum = sum(weights)
    if weight_sum == 0:
        return recent[-1]

    return sum(v * w for v, w in zip(recent, weights)) / weight_sum


def stdev(values: List[float], length: int) -> float:
    """Calculate standard deviation"""
    if len(values) < 2:
        return 0.0

    recent = values[-length:] if len(values) >= length else values
    mean = sum(recent) / len(recent)
    variance = sum((x - mean) ** 2 for x in recent) / len(recent)
    return math.sqrt(variance)


def atr(candles: List[Candle], length: int) -> float:
    """Calculate ATR (Average True Range)"""
    if len(candles) < 2:
        return 0.0

    tr_values = []
    for i in range(1, len(candles)):
        high_low = candles[i].high - candles[i].low
        high_close = abs(candles[i].high - candles[i-1].close)
        low_close = abs(candles[i].low - candles[i-1].close)
        tr_values.append(max(high_low, high_close, low_close))

    if not tr_values:
        return 0.0

    # Use RMA (Wilder's smoothing) for ATR
    return rma(tr_values, length)


def rma(values: List[float], length: int) -> float:
    """Calculate RMA (Wilder's smoothed moving average)"""
    if len(values) < length:
        return sum(values) / len(values) if values else 0.0

    alpha = 1.0 / length
    rma_val = sum(values[:length]) / length

    for i in range(length, len(values)):
        rma_val = alpha * values[i] + (1 - alpha) * rma_val

    return rma_val


def rsi(values: List[float], length: int) -> float:
    """Calculate RSI (Relative Strength Index)"""
    if len(values) < length + 1:
        return 50.0

    changes = [values[i] - values[i-1] for i in range(1, len(values))]
    gains = [max(0, c) for c in changes]
    losses = [max(0, -c) for c in changes]

    avg_gain = rma(gains, length)
    avg_loss = rma(losses, length)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def mfi(candles: List[Candle], length: int) -> float:
    """Calculate MFI (Money Flow Index)"""
    if len(candles) < length + 1:
        return 50.0

    typical_prices = [(c.high + c.low + c.close) / 3 for c in candles]
    raw_money_flow = [tp * c.volume for tp, c in zip(typical_prices, candles)]

    positive_flow = 0.0
    negative_flow = 0.0

    for i in range(-length, 0):
        if typical_prices[i] > typical_prices[i-1]:
            positive_flow += raw_money_flow[i]
        elif typical_prices[i] < typical_prices[i-1]:
            negative_flow += raw_money_flow[i]

    if negative_flow == 0:
        return 100.0

    money_ratio = positive_flow / negative_flow
    return 100.0 - (100.0 / (1.0 + money_ratio))


def macd(values: List[float], fast_len: int, slow_len: int, signal_len: int) -> tuple:
    """Calculate MACD line, signal line, and histogram"""
    if len(values) < slow_len:
        return 0.0, 0.0, 0.0

    fast_ema = ema(values, fast_len)
    slow_ema = ema(values, slow_len)
    macd_line = fast_ema - slow_ema

    # For signal line, we need historical MACD values
    macd_values = []
    for i in range(signal_len, len(values) + 1):
        subset = values[:i]
        if len(subset) >= slow_len:
            f = ema(subset, fast_len)
            s = ema(subset, slow_len)
            macd_values.append(f - s)

    if len(macd_values) >= signal_len:
        signal_line = ema(macd_values, signal_len)
    else:
        signal_line = macd_line

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(values: List[float], length: int, mult: float) -> tuple:
    """Calculate Bollinger Bands: (upper, basis, lower, bandwidth)"""
    if len(values) < length:
        basis = values[-1] if values else 0.0
        return basis, basis, basis, 0.0

    basis = sma(values, length)
    dev = stdev(values, length) * mult
    upper = basis + dev
    lower = basis - dev

    bandwidth = (upper - lower) / basis if basis != 0 else 0.0

    return upper, basis, lower, bandwidth


# =============================================================================
# ADVANCED FILTERS
# =============================================================================

class KalmanFilter:
    """Kalman Filter implementation"""

    def __init__(self, length: int, r: float = 0.01, q: float = 0.1):
        self.length = length
        self.r = r
        self.q = q
        self.estimate = None
        self.error_est = 1.0
        self.error_meas = r * length

    def update(self, value: float) -> float:
        """Update filter with new value and return filtered result"""
        if self.estimate is None:
            self.estimate = value
            return value

        # Prediction step
        prediction = self.estimate

        # Update Kalman gain
        kalman_gain = self.error_est / (self.error_est + self.error_meas)

        # Update estimate
        self.estimate = prediction + kalman_gain * (value - prediction)

        # Update error estimate
        self.error_est = (1 - kalman_gain) * self.error_est + self.q / self.length

        return self.estimate

    def reset(self):
        """Reset filter state"""
        self.estimate = None
        self.error_est = 1.0


def gaussian_filter(values: List[float], length: int, poles: int = 2) -> float:
    """
    Gaussian Filter (Ehlers 2-pole approximation)
    """
    if len(values) < 5:
        return values[-1] if values else 0.0

    beta = (1 - math.cos(2 * math.pi / length)) / (math.pow(1.414, 2 / poles) - 1)
    alpha = -beta + math.sqrt(beta ** 2 + 2 * beta)

    # Need at least 5 values for 4-pole filter
    c0 = alpha ** 4
    c1 = 4 * (1 - alpha)
    c2 = -6 * (1 - alpha) ** 2
    c3 = 4 * (1 - alpha) ** 3
    c4 = -(1 - alpha) ** 4

    # Initialize with simple values
    f = list(values[-5:])

    # Apply recursive filter
    result = c0 * values[-1] + c1 * nz(f[-1]) + c2 * nz(f[-2]) + c3 * nz(f[-3]) + c4 * nz(f[-4])

    return result


def rational_quadratic_kernel(values: List[float], lookback: int, relative_weight: float) -> float:
    """
    Rational Quadratic Kernel (Nadaraya-Watson estimator)
    """
    if len(values) < lookback:
        return values[-1] if values else 0.0

    yhat = 0.0
    sum_weights = 0.0

    for i in range(lookback):
        y = values[-(i + 1)]
        w = math.pow(1 + (i ** 2) / ((lookback ** 2) * 2 * relative_weight), -relative_weight)
        yhat += y * w
        sum_weights += w

    if sum_weights == 0:
        return values[-1]

    return yhat / sum_weights


def zero_lag_ema(values: List[float], length: int) -> float:
    """Calculate Zero Lag EMA"""
    if len(values) < length:
        return values[-1] if values else 0.0

    lag = (length - 1) // 2

    # Adjust source for lag compensation
    adjusted = []
    for i in range(len(values)):
        if i >= lag:
            adjusted.append(values[i] + (values[i] - values[i - lag]))
        else:
            adjusted.append(values[i])

    return ema(adjusted, length)


def zero_lag_score(values: List[float], length: int, loop_start: int, loop_end: int) -> float:
    """Calculate Zero Lag momentum score"""
    if len(values) < loop_end:
        return 0.0

    basis = zero_lag_ema(values, length)

    # Calculate score by comparing current basis to historical
    score = 0.0
    zl_values = []

    for i in range(loop_end):
        if len(values) > i:
            subset = values[:len(values) - i]
            if len(subset) >= length:
                zl_values.append(zero_lag_ema(subset, length))

    current_basis = zl_values[0] if zl_values else basis

    for i in range(loop_start - 1, min(loop_end, len(zl_values))):
        if current_basis > zl_values[i]:
            score += 1
        else:
            score -= 1

    return score


# =============================================================================
# TREND CALCULATION
# =============================================================================

def calculate_trend(method: str, values: List[float], length: int,
                   sigma: float, poles: int, offset: float,
                   kalman_short: int = 18, kalman_long: int = 23,
                   kalman_filters: Dict[str, KalmanFilter] = None) -> tuple:
    """
    Calculate trend value and direction
    Returns: (trend_value, is_bullish, is_bearish)
    """
    if not values:
        return 0.0, False, False

    trend_value = 0.0
    is_bullish = False
    is_bearish = False

    if method == "Kalman":
        if kalman_filters is None:
            kalman_filters = {
                'short': KalmanFilter(kalman_short),
                'long': KalmanFilter(kalman_long)
            }

        # Process all values through Kalman filters
        short_val = 0.0
        long_val = 0.0
        for v in values:
            short_val = kalman_filters['short'].update(v)
            long_val = kalman_filters['long'].update(v)

        trend_value = short_val
        is_bullish = short_val > long_val
        is_bearish = short_val < long_val

    elif method == "Gaussian":
        trend_value = gaussian_filter(values, length, poles)
        if len(values) >= 2:
            prev_trend = gaussian_filter(values[:-1], length, poles)
            is_bullish = trend_value > prev_trend
            is_bearish = trend_value < prev_trend

    elif method == "Kernel":
        trend_value = rational_quadratic_kernel(values, length, sigma)
        if len(values) >= 2:
            prev_trend = rational_quadratic_kernel(values[:-1], length, sigma)
            is_bullish = trend_value > prev_trend
            is_bearish = trend_value < prev_trend

    elif method == "EMA":
        trend_value = ema(values, length)
        if len(values) >= 2:
            prev_trend = ema(values[:-1], length)
            is_bullish = trend_value > prev_trend
            is_bearish = trend_value < prev_trend

    elif method == "SMA":
        trend_value = sma(values, length)
        if len(values) >= 2:
            prev_trend = sma(values[:-1], length)
            is_bullish = trend_value > prev_trend
            is_bearish = trend_value < prev_trend

    elif method == "HMA":
        trend_value = hma(values, length)
        if len(values) >= 2:
            prev_trend = hma(values[:-1], length)
            is_bullish = trend_value > prev_trend
            is_bearish = trend_value < prev_trend

    elif method == "ALMA":
        trend_value = alma(values, length, offset, sigma)
        if len(values) >= 2:
            prev_trend = alma(values[:-1], length, offset, sigma)
            is_bullish = trend_value > prev_trend
            is_bearish = trend_value < prev_trend

    return trend_value, is_bullish, is_bearish


# =============================================================================
# PIVOT DETECTION
# =============================================================================

def detect_pivot_high(candles: List[Candle], left_bars: int, right_bars: int, use_high_low: bool) -> Optional[float]:
    """
    Detect pivot high
    Returns the pivot value if found, None otherwise
    """
    if len(candles) < left_bars + right_bars + 1:
        return None

    pivot_idx = len(candles) - right_bars - 1

    if pivot_idx < left_bars:
        return None

    pivot_candle = candles[pivot_idx]
    pivot_value = pivot_candle.high if use_high_low else pivot_candle.close

    # Check left bars
    for i in range(pivot_idx - left_bars, pivot_idx):
        check_value = candles[i].high if use_high_low else candles[i].close
        if check_value >= pivot_value:
            return None

    # Check right bars
    for i in range(pivot_idx + 1, pivot_idx + right_bars + 1):
        check_value = candles[i].high if use_high_low else candles[i].close
        if check_value >= pivot_value:
            return None

    return pivot_value


def detect_pivot_low(candles: List[Candle], left_bars: int, right_bars: int, use_high_low: bool) -> Optional[float]:
    """
    Detect pivot low
    Returns the pivot value if found, None otherwise
    """
    if len(candles) < left_bars + right_bars + 1:
        return None

    pivot_idx = len(candles) - right_bars - 1

    if pivot_idx < left_bars:
        return None

    pivot_candle = candles[pivot_idx]
    pivot_value = pivot_candle.low if use_high_low else pivot_candle.close

    # Check left bars
    for i in range(pivot_idx - left_bars, pivot_idx):
        check_value = candles[i].low if use_high_low else candles[i].close
        if check_value <= pivot_value:
            return None

    # Check right bars
    for i in range(pivot_idx + 1, pivot_idx + right_bars + 1):
        check_value = candles[i].low if use_high_low else candles[i].close
        if check_value <= pivot_value:
            return None

    return pivot_value


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

@dataclass
class TradeState:
    """Trade state management"""
    in_trade: bool = False
    trade_dir: int = 0  # 1 = Long, -1 = Short
    entry_price: Optional[float] = None
    trade_highest: Optional[float] = None
    trade_lowest: Optional[float] = None
    stop_price: Optional[float] = None
    trade_entry_bar: int = 0
    entry_time: int = 0
    entry_signal_id: Optional[int] = None

    # Pending levels
    pending_support: Optional[float] = None
    pending_resistance: Optional[float] = None
    pending_long: bool = False
    pending_short: bool = False
    confirm_count_long: int = 0
    confirm_count_short: int = 0

    # Pivot strength tracking
    support_touches: int = 0
    resistance_touches: int = 0
    last_support_level: Optional[float] = None
    last_resistance_level: Optional[float] = None

    # Momentum/Trend confirmation
    mom_confirm_count_long: int = 0
    mom_confirm_count_short: int = 0
    trend_confirm_count_long: int = 0
    trend_confirm_count_short: int = 0

    # Exit confirmation
    exit_wait_count: int = 0
    pending_exit_reason: str = ""

    # Trade flags
    is_momentum_trade: bool = False
    trailing_was_activated: bool = False

    # P&L tracking
    total_pl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    total_trades: int = 0
    last_trade_pl: Optional[float] = None

    # Extended stats tracking
    first_trade_time: Optional[int] = None
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_win_bars: int = 0
    total_loss_bars: int = 0
    max_win: float = 0.0
    max_loss: float = 0.0
    peak_pl: float = 0.0
    max_drawdown: float = 0.0
    long_trades: int = 0
    long_wins: int = 0
    long_pl: float = 0.0
    short_trades: int = 0
    short_wins: int = 0
    short_pl: float = 0.0
    momentum_trades: int = 0
    momentum_wins: int = 0
    momentum_pl: float = 0.0
    pivot_trades: int = 0
    pivot_wins: int = 0
    pivot_pl: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Box Theory state
    box_top: Optional[float] = None
    box_bottom: Optional[float] = None
    box_state: int = 0  # 0=None, 1=Building Ceiling, 2=Building Floor, 3=Active
    last_box_top: Optional[float] = None
    last_box_bottom: Optional[float] = None
    box_breakout_bull: bool = False
    box_breakout_bear: bool = False


@dataclass
class IndicatorState:
    """Cached indicator values"""
    # Trend
    trend_value: float = 0.0
    trend_is_bullish: bool = False
    trend_is_bearish: bool = False

    # MTF
    htf_rising: bool = False
    htf_falling: bool = False
    mtf_bullish: bool = True
    mtf_bearish: bool = True

    # Oscillators
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    prev_macd_histogram: float = 0.0

    rsi_value: float = 50.0
    mfi_value: float = 50.0

    # Zero Lag
    zl_basis: float = 0.0
    zl_score: float = 0.0
    zl_rising: bool = False
    zl_falling: bool = False

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_basis: float = 0.0
    bb_lower: float = 0.0
    bb_bandwidth: float = 0.0
    bb_is_squeeze: bool = False

    # Volatility
    current_atr: float = 0.0
    avg_atr: float = 0.0

    # Momentum signals
    breakout_bull: bool = False
    breakout_bear: bool = False
    velocity_bull: bool = False
    velocity_bear: bool = False
    atr_exp_bull: bool = False
    atr_exp_bear: bool = False
    mfi_bullish: bool = False
    mfi_bearish: bool = False
    bb_breakout_bull: bool = False
    bb_breakout_bear: bool = False

    # Gravity
    gravity_bull: bool = False
    gravity_bear: bool = False
    current_gravity: str = "Sideways"


# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class SRSwingStrategy:
    """Main strategy engine"""

    def __init__(self, config: Config, db: SignalDatabase):
        self.config = config
        self.db = db

        # Candle buffers
        self.candles: Deque[Candle] = deque(maxlen=500)
        self.htf_candles: Deque[Candle] = deque(maxlen=200)

        # State
        self.trade_state = TradeState()
        self.indicators = IndicatorState()

        # Kalman filter instances (persistent state)
        self.kalman_short = KalmanFilter(config.kalman_short_len)
        self.kalman_long = KalmanFilter(config.kalman_long_len)
        self.mtf_kalman_short = KalmanFilter(config.mtf_kalman_short)
        self.mtf_kalman_long = KalmanFilter(config.mtf_kalman_long)

        # Bar counter
        self.bar_index = 0

        # Previous values for change detection
        self.prev_trend_bullish = False
        self.prev_trend_bearish = False

        # Silent mode (for backtesting without prints)
        self.silent_mode = False

        # JSON output mode
        self.json_output = False

    def process_candle(self, candle: Candle) -> List[Dict[str, Any]]:
        """
        Process a new candle and return any generated signals
        """
        signals = []

        # Add candle to buffer
        self.candles.append(candle)
        self.bar_index += 1

        # Need minimum candles for calculations
        if len(self.candles) < 50:
            return signals

        # Update all indicators
        self._update_indicators()

        # Check exits first (if in trade)
        if self.trade_state.in_trade:
            exit_signals = self._check_exits(candle)
            signals.extend(exit_signals)

        # Check entries (if not in trade or SAR enabled)
        if not self.trade_state.in_trade or self.config.enable_sar:
            entry_signals = self._check_entries(candle)
            signals.extend(entry_signals)

        # Update pending levels
        self._update_pending_levels(candle)

        # Store previous values
        self.prev_trend_bullish = self.indicators.trend_is_bullish
        self.prev_trend_bearish = self.indicators.trend_is_bearish

        return signals

    def process_htf_candle(self, candle: Candle):
        """Process higher timeframe candle for MTF analysis"""
        self.htf_candles.append(candle)
        self._update_mtf_indicators()

    def _update_indicators(self):
        """Update all indicator values"""
        cfg = self.config
        candles = list(self.candles)
        closes = [c.close for c in candles]

        # --- Trend Calculation ---
        if cfg.trend_method == "Kalman":
            # Update Kalman filters
            for c in closes:
                short_val = self.kalman_short.update(c)
                long_val = self.kalman_long.update(c)

            self.indicators.trend_value = short_val
            self.indicators.trend_is_bullish = short_val > long_val
            self.indicators.trend_is_bearish = short_val < long_val
        else:
            trend_val, is_bull, is_bear = calculate_trend(
                cfg.trend_method, closes, cfg.trend_length,
                cfg.trend_sigma, cfg.trend_poles, cfg.trend_offset
            )
            self.indicators.trend_value = trend_val
            self.indicators.trend_is_bullish = is_bull
            self.indicators.trend_is_bearish = is_bear

        # Apply reverse logic if enabled
        if cfg.reverse_trend:
            self.indicators.trend_is_bullish, self.indicators.trend_is_bearish = \
                self.indicators.trend_is_bearish, self.indicators.trend_is_bullish

        # --- Gravity ---
        if cfg.gravity_mode == "Auto (Trend)":
            self.indicators.gravity_bull = self.indicators.trend_is_bullish
            self.indicators.gravity_bear = self.indicators.trend_is_bearish
            self.indicators.current_gravity = "Bullish" if self.indicators.trend_is_bullish else "Bearish" if self.indicators.trend_is_bearish else "Sideways"
        elif cfg.gravity_mode == "Bullish":
            self.indicators.gravity_bull = True
            self.indicators.gravity_bear = False
            self.indicators.current_gravity = "Bullish"
        elif cfg.gravity_mode == "Bearish":
            self.indicators.gravity_bull = False
            self.indicators.gravity_bear = True
            self.indicators.current_gravity = "Bearish"
        else:
            self.indicators.gravity_bull = False
            self.indicators.gravity_bear = False
            self.indicators.current_gravity = cfg.gravity_mode

        # --- RSI ---
        if cfg.enable_rsi_filter:
            self.indicators.rsi_value = rsi(closes, cfg.rsi_length)

        # --- MFI ---
        if cfg.enable_mfi_momentum:
            self.indicators.mfi_value = mfi(candles, cfg.mfi_length)

        # --- MACD ---
        if cfg.enable_oscillator_filter and cfg.oscillator_type == "MACD":
            self.indicators.prev_macd_histogram = self.indicators.macd_histogram
            macd_l, macd_s, macd_h = macd(closes, cfg.macd_fast_len, cfg.macd_slow_len, cfg.macd_signal_len)
            self.indicators.macd_line = macd_l
            self.indicators.macd_signal = macd_s
            self.indicators.macd_histogram = macd_h

        # --- Zero Lag ---
        if cfg.enable_oscillator_filter and cfg.oscillator_type == "Zero Lag Score":
            prev_basis = self.indicators.zl_basis
            self.indicators.zl_basis = zero_lag_ema(closes, cfg.zl_length)
            self.indicators.zl_score = zero_lag_score(closes, cfg.zl_length, cfg.zl_loop_start, cfg.zl_loop_end)
            self.indicators.zl_rising = self.indicators.zl_basis > prev_basis
            self.indicators.zl_falling = self.indicators.zl_basis < prev_basis

        # --- Bollinger Bands ---
        if cfg.enable_bb:
            bb_u, bb_b, bb_l, bb_bw = bollinger_bands(closes, cfg.bb_length, cfg.bb_mult)
            self.indicators.bb_upper = bb_u
            self.indicators.bb_basis = bb_b
            self.indicators.bb_lower = bb_l
            self.indicators.bb_bandwidth = bb_bw

            avg_bw = sma([bb_bw] * 100, 100) if len(closes) >= 100 else bb_bw
            self.indicators.bb_is_squeeze = bb_bw < (avg_bw * cfg.bb_squeeze_threshold)

        # --- ATR ---
        self.indicators.current_atr = atr(candles, cfg.atr_length)
        atr_values = [atr(candles[:i+1], cfg.atr_length) for i in range(max(0, len(candles) - cfg.atr_length), len(candles))]
        self.indicators.avg_atr = sma(atr_values, cfg.atr_length) if atr_values else self.indicators.current_atr

        # --- Momentum Signals ---
        self._update_momentum_signals(candles, closes)

    def _update_momentum_signals(self, candles: List[Candle], closes: List[float]):
        """Update momentum-based signals"""
        cfg = self.config

        if not cfg.enable_momentum_mode:
            return

        # Breakout Detection
        if cfg.enable_breakout and len(candles) >= cfg.breakout_lookback + 1:
            highs = [c.high for c in candles]
            lows = [c.low for c in candles]
            volumes = [c.volume for c in candles]

            recent_high = max(highs[-cfg.breakout_lookback - 1:-1])
            recent_low = min(lows[-cfg.breakout_lookback - 1:-1])
            avg_volume = sma(volumes[:-1], cfg.breakout_lookback)

            self.indicators.breakout_bull = (
                closes[-1] > recent_high and
                volumes[-1] > avg_volume * cfg.breakout_vol_multiplier
            )
            self.indicators.breakout_bear = (
                closes[-1] < recent_low and
                volumes[-1] > avg_volume * cfg.breakout_vol_multiplier
            )

        # Velocity Detection
        if cfg.enable_velocity and len(closes) > cfg.velocity_bars:
            price_change_pct = ((closes[-1] - closes[-cfg.velocity_bars - 1]) / closes[-cfg.velocity_bars - 1]) * 100
            self.indicators.velocity_bull = price_change_pct > cfg.velocity_threshold
            self.indicators.velocity_bear = price_change_pct < -cfg.velocity_threshold

        # ATR Expansion
        if cfg.enable_atr_expansion:
            self.indicators.atr_exp_bull = (
                self.indicators.current_atr > self.indicators.avg_atr * cfg.atr_multiplier and
                self.indicators.trend_is_bullish
            )
            self.indicators.atr_exp_bear = (
                self.indicators.current_atr > self.indicators.avg_atr * cfg.atr_multiplier and
                self.indicators.trend_is_bearish
            )

        # MFI Momentum
        if cfg.enable_mfi_momentum:
            prev_mfi = mfi(candles[:-1], cfg.mfi_length) if len(candles) > cfg.mfi_length + 1 else 50
            self.indicators.mfi_bullish = (
                self.indicators.mfi_value > cfg.mfi_momentum_threshold and
                self.indicators.mfi_value > prev_mfi
            )
            self.indicators.mfi_bearish = (
                self.indicators.mfi_value < cfg.mfi_momentum_threshold and
                self.indicators.mfi_value < prev_mfi
            )

        # BB Squeeze Breakout
        if cfg.enable_bb and cfg.enable_bb_squeeze:
            # Check if was in squeeze recently (last 5 bars)
            was_in_squeeze = self.indicators.bb_is_squeeze  # Simplified
            self.indicators.bb_breakout_bull = was_in_squeeze and closes[-1] > self.indicators.bb_upper
            self.indicators.bb_breakout_bear = was_in_squeeze and closes[-1] < self.indicators.bb_lower

    def _update_mtf_indicators(self):
        """Update MTF (higher timeframe) indicators"""
        if not self.config.enable_mtf or len(self.htf_candles) < 2:
            return

        cfg = self.config
        htf_closes = [c.close for c in self.htf_candles]

        if cfg.mtf_method == "Kalman":
            for c in htf_closes:
                short_val = self.mtf_kalman_short.update(c)
                long_val = self.mtf_kalman_long.update(c)

            self.indicators.htf_rising = short_val > long_val
            self.indicators.htf_falling = short_val < long_val
        else:
            _, is_bull, is_bear = calculate_trend(
                cfg.mtf_method, htf_closes, cfg.mtf_length,
                cfg.mtf_sigma, 2, cfg.mtf_alma_offset
            )
            self.indicators.htf_rising = is_bull
            self.indicators.htf_falling = is_bear

        # Set MTF filter flags
        if cfg.mtf_strict_mode:
            self.indicators.mtf_bullish = self.indicators.htf_rising
            self.indicators.mtf_bearish = self.indicators.htf_falling
        else:
            self.indicators.mtf_bullish = self.indicators.htf_rising or not self.indicators.htf_falling
            self.indicators.mtf_bearish = self.indicators.htf_falling or not self.indicators.htf_rising

    def _check_entry_filters(self, direction: int) -> bool:
        """Check if all entry filters pass for given direction (1=Long, -1=Short)"""
        cfg = self.config
        ind = self.indicators
        is_long = direction == 1

        # MTF Filter
        if cfg.enable_mtf:
            if is_long and not ind.mtf_bullish:
                return False
            if not is_long and not ind.mtf_bearish:
                return False

        # RSI Filter
        if cfg.enable_rsi_filter:
            if cfg.rsi_filter_mode == "Avoid Extremes":
                if is_long and ind.rsi_value >= cfg.rsi_overbought:
                    return False
                if not is_long and ind.rsi_value <= cfg.rsi_oversold:
                    return False
            elif cfg.rsi_filter_mode == "Momentum Aligned":
                if is_long and ind.rsi_value <= 50:
                    return False
                if not is_long and ind.rsi_value >= 50:
                    return False
            elif cfg.rsi_filter_mode == "Counter-Trend Bounce":
                if is_long and ind.rsi_value >= cfg.rsi_oversold:
                    return False
                if not is_long and ind.rsi_value <= cfg.rsi_overbought:
                    return False

        # Oscillator Filter (MACD / Zero Lag)
        if cfg.enable_oscillator_filter:
            if cfg.oscillator_type == "MACD":
                if cfg.macd_entry_mode == "Above Zero Line":
                    if is_long and ind.macd_line <= 0:
                        return False
                    if not is_long and ind.macd_line >= 0:
                        return False
                elif cfg.macd_entry_mode == "Below Zero Line":
                    if is_long and ind.macd_line >= 0:
                        return False
                    if not is_long and ind.macd_line <= 0:
                        return False
                elif cfg.macd_entry_mode == "Histogram Positive":
                    if is_long and ind.macd_histogram <= 0:
                        return False
                    if not is_long and ind.macd_histogram >= 0:
                        return False

            elif cfg.oscillator_type == "Zero Lag Score":
                if cfg.zl_entry_mode == "Score Only":
                    if is_long and ind.zl_score <= cfg.zl_threshold_up:
                        return False
                    if not is_long and ind.zl_score >= cfg.zl_threshold_down:
                        return False
                elif cfg.zl_entry_mode == "ZL Line Direction":
                    if is_long and not ind.zl_rising:
                        return False
                    if not is_long and not ind.zl_falling:
                        return False

        # Bollinger Bands Filter
        if cfg.enable_bb and cfg.bb_use_filter:
            current_close = self.candles[-1].close
            if is_long and current_close > ind.bb_upper:
                return False
            if not is_long and current_close < ind.bb_lower:
                return False

        # Box Theory Filter
        if cfg.enable_box_theory:
            ts = self.trade_state
            if cfg.box_filter_mode == "Strict":
                if is_long and not ts.box_breakout_bull:
                    return False
                if not is_long and not ts.box_breakout_bear:
                    return False

        return True

    def _check_momentum_signal(self, direction: int) -> bool:
        """Check if any momentum signal is active"""
        if not self.config.enable_momentum_mode:
            return False

        ind = self.indicators
        is_long = direction == 1

        if is_long:
            return (ind.breakout_bull or ind.velocity_bull or
                    ind.atr_exp_bull or ind.mfi_bullish or ind.bb_breakout_bull)
        else:
            return (ind.breakout_bear or ind.velocity_bear or
                    ind.atr_exp_bear or ind.mfi_bearish or ind.bb_breakout_bear)

    def _check_exits(self, candle: Candle) -> List[Dict[str, Any]]:
        """Check exit conditions and return exit signals"""
        signals = []
        cfg = self.config
        ts = self.trade_state
        ind = self.indicators

        if not ts.in_trade:
            return signals

        is_long = ts.trade_dir == 1
        current_price = candle.close

        # Update highest/lowest
        if is_long:
            ts.trade_highest = max(ts.trade_highest or current_price, candle.high)
        else:
            ts.trade_lowest = min(ts.trade_lowest or current_price, candle.low)

        # Calculate P&L
        pnl_pct = ((current_price - ts.entry_price) * ts.trade_dir / ts.entry_price) * 100 - cfg.exchange_fee
        profit_ok = pnl_pct >= cfg.min_profit_percent

        # Determine active TP/SL
        active_tp = cfg.momentum_tp_percent if ts.is_momentum_trade else cfg.tp_percent
        active_sl = cfg.momentum_sl_percent if ts.is_momentum_trade else cfg.sl_percent
        is_unlimited_tp = False

        # Trend-Aligned Management
        is_aligned = cfg.use_trend_mgt and ((is_long and ind.trend_is_bullish) or (not is_long and ind.trend_is_bearish))
        if is_aligned:
            if cfg.trend_tp_behavior == "Unlimited":
                is_unlimited_tp = True
            elif cfg.trend_tp_behavior == "Override":
                active_tp = cfg.trend_tp_override
            if cfg.trend_sl_behavior == "Override":
                active_sl = cfg.trend_sl_override

        # Gravity Overrides
        is_gravity_aligned = (ind.current_gravity == "Bullish" and is_long) or (ind.current_gravity == "Bearish" and not is_long)
        if is_gravity_aligned:
            active_sl *= cfg.gravity_sl_multiplier
            if cfg.gravity_tp_behavior == "Unlimited":
                is_unlimited_tp = True
            elif cfg.gravity_tp_behavior == "2x Target":
                active_tp *= 2.0

        # Calculate prices
        tp_price = ts.entry_price * (1 + (1 if is_long else -1) * active_tp / 100)
        sl_price = ts.entry_price * (1 + (-1 if is_long else 1) * active_sl / 100)

        exit_reason = None
        exit_price = current_price

        # Check TP
        if not is_unlimited_tp:
            if (is_long and candle.high >= tp_price) or (not is_long and candle.low <= tp_price):
                exit_reason = "TP"
                exit_price = tp_price

        # Check SL
        if exit_reason is None:
            if (is_long and candle.low <= sl_price) or (not is_long and candle.high >= sl_price):
                exit_reason = "SL"
                exit_price = sl_price

        # Check Trailing Stop
        if exit_reason is None and cfg.use_trailing:
            allow_trailing = not (is_gravity_aligned and cfg.gravity_disable_trail)

            if allow_trailing:
                activation_price = ts.entry_price * (1 + (1 if is_long else -1) * cfg.trailing_activation / 100)
                extreme_price = ts.trade_highest if is_long else ts.trade_lowest
                is_activated = (is_long and extreme_price >= activation_price) or (not is_long and extreme_price <= activation_price)

                if is_activated:
                    trail_stop = extreme_price * (1 + (-1 if is_long else 1) * cfg.trail_callback / 100)

                    if cfg.trail_protect_entry:
                        trail_stop = max(trail_stop, ts.entry_price) if is_long else min(trail_stop, ts.entry_price)

                    if (is_long and candle.low <= trail_stop) or (not is_long and candle.high >= trail_stop):
                        trail_pnl = ((trail_stop - ts.entry_price) * ts.trade_dir / ts.entry_price) * 100 - cfg.exchange_fee
                        if not cfg.trail_protect_entry or trail_pnl >= 0:
                            exit_reason = "Trail SL"
                            exit_price = trail_stop

        # Check MACD Exit
        if exit_reason is None and cfg.enable_oscillator_filter and cfg.oscillator_type == "MACD":
            macd_exit = False

            if cfg.macd_exit_on_histogram_peak:
                if is_long and ind.macd_histogram > 0 and ind.macd_histogram < ind.prev_macd_histogram:
                    macd_exit = True
                if not is_long and ind.macd_histogram < 0 and ind.macd_histogram > ind.prev_macd_histogram:
                    macd_exit = True

            if cfg.macd_exit_on_zero_cross:
                if is_long and ind.macd_line < 0:
                    macd_exit = True
                if not is_long and ind.macd_line > 0:
                    macd_exit = True

            if macd_exit:
                if cfg.macd_exit_takes_precedence or profit_ok:
                    exit_reason = "MACD"

        # Check Trend Flip Exit
        if exit_reason is None and cfg.use_trend_mgt and cfg.trend_exit_on_flip and profit_ok:
            block_exit = is_gravity_aligned and cfg.gravity_disable_flip_exit
            if not block_exit:
                if (is_long and ind.trend_is_bearish) or (not is_long and ind.trend_is_bullish):
                    exit_reason = "Trend Flip"

        # Check MTF Exit
        if exit_reason is None and cfg.enable_mtf and cfg.mtf_exit_on_flip and profit_ok:
            if (is_long and ind.htf_falling) or (not is_long and ind.htf_rising):
                exit_reason = "MTF Flip"

        # Check Reverse Momentum Exit
        if exit_reason is None and cfg.enable_momentum_mode and cfg.enable_reverse_momentum_exit and profit_ok:
            if (is_long and (ind.velocity_bear or ind.breakout_bear)) or \
               (not is_long and (ind.velocity_bull or ind.breakout_bull)):
                exit_reason = "Rev Momentum"

        # Execute Exit
        if exit_reason:
            final_pnl = ((exit_price - ts.entry_price) * ts.trade_dir / ts.entry_price) * 100 - cfg.exchange_fee

            signal_data = self._create_signal_data(
                signal_type=f"EXIT_{'LONG' if is_long else 'SHORT'}",
                candle=candle,
                price=exit_price,
                exit_reason=exit_reason,
                pnl_percent=final_pnl
            )

            # Store signal
            signal_id = self.db.store_signal(signal_data)

            # Store trade record
            trade_data = {
                'entry_signal_id': ts.entry_signal_id,
                'exit_signal_id': signal_id,
                'symbol': cfg.symbol,
                'direction': ts.trade_dir,
                'entry_time': ts.entry_time,
                'exit_time': candle.timestamp,
                'entry_price': ts.entry_price,
                'exit_price': exit_price,
                'stop_loss': sl_price,
                'take_profit': tp_price,
                'pnl_percent': final_pnl,
                'duration_bars': self.bar_index - ts.trade_entry_bar,
                'is_momentum_trade': ts.is_momentum_trade,
                'exit_reason': exit_reason
            }
            self.db.store_trade(trade_data)

            # Update stats
            duration_bars = self.bar_index - ts.trade_entry_bar
            ts.total_pl += final_pnl
            ts.last_trade_pl = final_pnl
            ts.total_trades += 1

            # Track first trade time
            if ts.first_trade_time is None:
                ts.first_trade_time = ts.entry_time

            # Win/Loss tracking
            if final_pnl > 0:
                ts.win_count += 1
                ts.gross_profit += final_pnl
                ts.total_win_bars += duration_bars
                ts.max_win = max(ts.max_win, final_pnl)
                ts.consecutive_wins += 1
                ts.consecutive_losses = 0
                ts.max_consecutive_wins = max(ts.max_consecutive_wins, ts.consecutive_wins)
            else:
                ts.loss_count += 1
                ts.gross_loss += abs(final_pnl)
                ts.total_loss_bars += duration_bars
                ts.max_loss = min(ts.max_loss, final_pnl)
                ts.consecutive_losses += 1
                ts.consecutive_wins = 0
                ts.max_consecutive_losses = max(ts.max_consecutive_losses, ts.consecutive_losses)

            # Drawdown tracking
            ts.peak_pl = max(ts.peak_pl, ts.total_pl)
            current_dd = ts.peak_pl - ts.total_pl
            ts.max_drawdown = max(ts.max_drawdown, current_dd)

            # Direction tracking
            if is_long:
                ts.long_trades += 1
                ts.long_pl += final_pnl
                if final_pnl > 0:
                    ts.long_wins += 1
            else:
                ts.short_trades += 1
                ts.short_pl += final_pnl
                if final_pnl > 0:
                    ts.short_wins += 1

            # Trade type tracking
            if ts.is_momentum_trade:
                ts.momentum_trades += 1
                ts.momentum_pl += final_pnl
                if final_pnl > 0:
                    ts.momentum_wins += 1
            else:
                ts.pivot_trades += 1
                ts.pivot_pl += final_pnl
                if final_pnl > 0:
                    ts.pivot_wins += 1

            # Print signal
            self._print_signal(signal_data, silent=self.silent_mode)

            # Print live stats table after each closed trade (only in live mode)
            if not self.silent_mode:
                self._print_live_stats()

            # Reset trade state
            ts.in_trade = False
            ts.trade_dir = 0
            ts.entry_price = None
            ts.trade_highest = None
            ts.trade_lowest = None
            ts.stop_price = None
            ts.is_momentum_trade = False
            ts.trailing_was_activated = False
            ts.entry_signal_id = None

            signals.append(signal_data)

        return signals

    def _check_entries(self, candle: Candle) -> List[Dict[str, Any]]:
        """Check entry conditions and return entry signals"""
        signals = []
        cfg = self.config
        ts = self.trade_state
        ind = self.indicators

        # Skip if already in trade (unless SAR)
        if ts.in_trade and not cfg.enable_sar:
            return signals

        current_price = candle.close

        # Check Long Entry
        if cfg.enable_longs:
            # Gravity filter
            gravity_ok = not cfg.gravity_block_counter or ind.current_gravity != "Bearish"

            if gravity_ok:
                # Momentum Long
                momentum_signal = self._check_momentum_signal(1)
                trend_signal = cfg.use_trend_signals and ind.trend_is_bullish and not self.prev_trend_bullish

                # Check confirmation
                if momentum_signal or trend_signal:
                    ts.mom_confirm_count_long += 1
                else:
                    ts.mom_confirm_count_long = 0

                # Entry on confirmation threshold
                if ts.mom_confirm_count_long >= cfg.confirm_bars + 1:
                    if self._check_entry_filters(1):
                        # Execute Long Entry
                        signal = self._execute_entry(candle, 1, is_momentum=momentum_signal)
                        if signal:
                            signals.append(signal)
                        ts.mom_confirm_count_long = 0

                # Pivot-Retest Long (if no momentum entry)
                elif not signals and ts.pending_long:
                    if ts.confirm_count_long >= max(cfg.confirm_bars, 1):
                        strength_ok = not cfg.enable_pivot_strength or ts.support_touches >= cfg.min_touches
                        if strength_ok and self._check_entry_filters(1):
                            # Check trend filter for pivot entries
                            if not cfg.use_trend_mgt or ind.trend_is_bullish:
                                signal = self._execute_entry(candle, 1, is_momentum=False)
                                if signal:
                                    signals.append(signal)
                                ts.pending_long = False
                                ts.confirm_count_long = 0

        # Check Short Entry
        if cfg.enable_shorts and not signals:
            # Gravity filter
            gravity_ok = not cfg.gravity_block_counter or ind.current_gravity != "Bullish"

            if gravity_ok:
                # Momentum Short
                momentum_signal = self._check_momentum_signal(-1)
                trend_signal = cfg.use_trend_signals and ind.trend_is_bearish and not self.prev_trend_bearish

                if momentum_signal or trend_signal:
                    ts.mom_confirm_count_short += 1
                else:
                    ts.mom_confirm_count_short = 0

                if ts.mom_confirm_count_short >= cfg.confirm_bars + 1:
                    if self._check_entry_filters(-1):
                        signal = self._execute_entry(candle, -1, is_momentum=momentum_signal)
                        if signal:
                            signals.append(signal)
                        ts.mom_confirm_count_short = 0

                # Pivot-Retest Short
                elif not signals and ts.pending_short:
                    if ts.confirm_count_short >= max(cfg.confirm_bars, 1):
                        strength_ok = not cfg.enable_pivot_strength or ts.resistance_touches >= cfg.min_touches
                        if strength_ok and self._check_entry_filters(-1):
                            if not cfg.use_trend_mgt or ind.trend_is_bearish:
                                signal = self._execute_entry(candle, -1, is_momentum=False)
                                if signal:
                                    signals.append(signal)
                                ts.pending_short = False
                                ts.confirm_count_short = 0

        return signals

    def _execute_entry(self, candle: Candle, direction: int, is_momentum: bool) -> Optional[Dict[str, Any]]:
        """Execute entry and return signal data"""
        cfg = self.config
        ts = self.trade_state

        # If in trade and SAR enabled, close current first
        if ts.in_trade and cfg.enable_sar:
            # Close existing trade (simplified - actual implementation would handle this)
            pass

        entry_price = candle.close

        # Calculate SL
        if is_momentum:
            sl_pct = cfg.momentum_sl_percent
            tp_pct = cfg.momentum_tp_percent
        else:
            sl_pct = cfg.sl_percent
            tp_pct = cfg.tp_percent

        # Apply Trend-Aligned overrides
        is_aligned = cfg.use_trend_mgt and (
            (direction == 1 and self.indicators.trend_is_bullish) or
            (direction == -1 and self.indicators.trend_is_bearish)
        )
        if is_aligned:
            if cfg.trend_sl_behavior == "Override":
                sl_pct = cfg.trend_sl_override
            if cfg.trend_tp_behavior == "Override":
                tp_pct = cfg.trend_tp_override

        sl_price = entry_price * (1 + (-1 if direction == 1 else 1) * sl_pct / 100) if cfg.use_stop_loss else None
        tp_price = entry_price * (1 + (1 if direction == 1 else -1) * tp_pct / 100)

        signal_data = self._create_signal_data(
            signal_type=f"ENTRY_{'LONG' if direction == 1 else 'SHORT'}",
            candle=candle,
            price=entry_price,
            stop_loss=sl_price,
            take_profit=tp_price,
            is_momentum=is_momentum
        )

        # Store signal
        signal_id = self.db.store_signal(signal_data)

        # Update trade state
        ts.in_trade = True
        ts.trade_dir = direction
        ts.entry_price = entry_price
        ts.entry_time = candle.timestamp
        ts.trade_entry_bar = self.bar_index
        ts.entry_signal_id = signal_id
        ts.is_momentum_trade = is_momentum

        # Track first trade time
        if ts.first_trade_time is None:
            ts.first_trade_time = candle.timestamp
        ts.trailing_was_activated = False
        ts.trade_highest = entry_price if direction == 1 else None
        ts.trade_lowest = entry_price if direction == -1 else None
        ts.stop_price = sl_price

        # Print signal
        self._print_signal(signal_data, silent=self.silent_mode)

        return signal_data

    def _update_pending_levels(self, candle: Candle):
        """Update pending support/resistance levels"""
        cfg = self.config
        ts = self.trade_state
        candles = list(self.candles)

        # Detect new pivots
        pivot_high = detect_pivot_high(candles, cfg.pivot_left_bars, cfg.pivot_right_bars, cfg.use_high_low)
        pivot_low = detect_pivot_low(candles, cfg.pivot_left_bars, cfg.pivot_right_bars, cfg.use_high_low)

        # Update pending support
        if cfg.enable_support_entries and pivot_low is not None:
            ts.pending_support = pivot_low
            ts.pending_long = True
            ts.confirm_count_long = 0
            if ts.pending_support != ts.last_support_level:
                ts.support_touches = 0
                ts.last_support_level = ts.pending_support

        # Update pending resistance
        if cfg.enable_resistance_entries and pivot_high is not None:
            ts.pending_resistance = pivot_high
            ts.pending_short = True
            ts.confirm_count_short = 0
            if ts.pending_resistance != ts.last_resistance_level:
                ts.resistance_touches = 0
                ts.last_resistance_level = ts.pending_resistance

        # Maintain pending support (confirmation)
        if ts.pending_long and ts.pending_support:
            if ts.confirm_count_long == 0:
                if candle.low <= ts.pending_support:
                    if candle.close > ts.pending_support:
                        ts.confirm_count_long = 1
                    else:
                        ts.pending_long = False
            elif ts.confirm_count_long > 0:
                if candle.close > ts.pending_support:
                    ts.confirm_count_long += 1
                else:
                    ts.pending_long = False
                    ts.confirm_count_long = 0

            # Track touches
            if cfg.enable_pivot_strength:
                touch_zone = ts.pending_support * cfg.touch_tolerance / 100
                if ts.pending_support - touch_zone <= candle.low <= ts.pending_support + touch_zone:
                    ts.support_touches += 1

        # Maintain pending resistance (confirmation)
        if ts.pending_short and ts.pending_resistance:
            if ts.confirm_count_short == 0:
                if candle.high >= ts.pending_resistance:
                    if candle.close < ts.pending_resistance:
                        ts.confirm_count_short = 1
                    else:
                        ts.pending_short = False
            elif ts.confirm_count_short > 0:
                if candle.close < ts.pending_resistance:
                    ts.confirm_count_short += 1
                else:
                    ts.pending_short = False
                    ts.confirm_count_short = 0

            # Track touches
            if cfg.enable_pivot_strength:
                touch_zone = ts.pending_resistance * cfg.touch_tolerance / 100
                if ts.pending_resistance - touch_zone <= candle.high <= ts.pending_resistance + touch_zone:
                    ts.resistance_touches += 1

    def _create_signal_data(self, signal_type: str, candle: Candle, price: float,
                           stop_loss: float = None, take_profit: float = None,
                           exit_reason: str = None, pnl_percent: float = None,
                           is_momentum: bool = False) -> Dict[str, Any]:
        """Create signal data dictionary"""
        cfg = self.config
        ind = self.indicators
        ts = self.trade_state

        return {
            'candle_time': candle.timestamp,
            'symbol': cfg.symbol,
            'timeframe': cfg.timeframe,
            'signal_type': signal_type,
            'price': price,
            'entry_price': ts.entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'is_momentum_trade': is_momentum,
            'exit_reason': exit_reason,
            'pnl_percent': pnl_percent,
            'trend_direction': 'Bullish' if ind.trend_is_bullish else 'Bearish' if ind.trend_is_bearish else 'Neutral',
            'mtf_direction': 'Rising' if ind.htf_rising else 'Falling' if ind.htf_falling else 'Neutral',
            'rsi_value': ind.rsi_value,
            'mfi_value': ind.mfi_value,
            'macd_value': ind.macd_line,
            'macd_histogram': ind.macd_histogram,
            'gravity_mode': ind.current_gravity,
            'box_state': str(ts.box_state),
            'candle_data': {
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            },
            'filters_snapshot': {
                'mtf_bullish': ind.mtf_bullish,
                'mtf_bearish': ind.mtf_bearish,
                'breakout_bull': ind.breakout_bull,
                'breakout_bear': ind.breakout_bear,
                'velocity_bull': ind.velocity_bull,
                'velocity_bear': ind.velocity_bear,
                'bb_squeeze': ind.bb_is_squeeze,
                'pending_support': ts.pending_support,
                'pending_resistance': ts.pending_resistance
            },
            'config_hash': hash(str(asdict(cfg)))
        }

    def _print_signal(self, signal_data: Dict[str, Any], silent: bool = False):
        """Print signal to console"""
        if silent:
            return

        timestamp = datetime.fromtimestamp(signal_data['candle_time'] / 1000, tz=timezone.utc)
        signal_type = signal_data['signal_type']
        symbol = signal_data['symbol']
        price = signal_data['price']

        if 'ENTRY' in signal_type:
            sl = signal_data.get('stop_loss', 'N/A')
            tp = signal_data.get('take_profit', 'N/A')
            momentum = " (MOMENTUM)" if signal_data.get('is_momentum_trade') else ""

            sl_str = f"{sl:.2f}" if isinstance(sl, (int, float)) else sl
            tp_str = f"{tp:.2f}" if isinstance(tp, (int, float)) else tp

            print(f"\n{'='*60}")
            print(f"[{timestamp}] {signal_type}{momentum}")
            print(f"  Symbol: {symbol} @ {price:.4f}")
            print(f"  SL: {sl_str} | TP: {tp_str}")
            print(f"  Trend: {signal_data['trend_direction']} | Gravity: {signal_data['gravity_mode']}")
            print(f"  RSI: {signal_data['rsi_value']:.1f} | MFI: {signal_data['mfi_value']:.1f}")
            print(f"{'='*60}\n")
        else:
            reason = signal_data.get('exit_reason', 'Unknown')
            pnl = signal_data.get('pnl_percent', 0)
            pnl_str = f"+{pnl:.2f}%" if pnl >= 0 else f"{pnl:.2f}%"

            print(f"\n{'='*60}")
            print(f"[{timestamp}] {signal_type}")
            print(f"  Symbol: {symbol} @ {price:.4f}")
            print(f"  Reason: {reason} | P&L: {pnl_str}")
            print(f"  Total P&L: {self.trade_state.total_pl:.2f}% | Trades: {self.trade_state.total_trades}")
            print(f"{'='*60}\n")

    def _print_live_stats(self):
        """Print comprehensive live stats table after each closed trade"""
        ts = self.trade_state
        cfg = self.config

        # Calculate derived stats
        win_rate = (ts.win_count / ts.total_trades * 100) if ts.total_trades > 0 else 0
        profit_factor = ts.gross_profit / ts.gross_loss if ts.gross_loss > 0 else float('inf')
        avg_win = ts.gross_profit / ts.win_count if ts.win_count > 0 else 0
        avg_loss = ts.gross_loss / ts.loss_count if ts.loss_count > 0 else 0
        avg_trade = ts.total_pl / ts.total_trades if ts.total_trades > 0 else 0
        avg_win_bars = ts.total_win_bars / ts.win_count if ts.win_count > 0 else 0
        avg_loss_bars = ts.total_loss_bars / ts.loss_count if ts.loss_count > 0 else 0
        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)

        # Format first trade date and calculate days
        days_trading = 0
        if ts.first_trade_time:
            first_trade_dt = datetime.fromtimestamp(ts.first_trade_time / 1000, tz=timezone.utc)
            start_date = first_trade_dt.strftime('%Y-%m-%d %H:%M')
            days_trading = (datetime.now(timezone.utc) - first_trade_dt).days
        else:
            start_date = "N/A"

        # Long stats
        long_wr = (ts.long_wins / ts.long_trades * 100) if ts.long_trades > 0 else 0
        # Short stats
        short_wr = (ts.short_wins / ts.short_trades * 100) if ts.short_trades > 0 else 0
        # Momentum stats
        momentum_wr = (ts.momentum_wins / ts.momentum_trades * 100) if ts.momentum_trades > 0 else 0
        # Pivot stats
        pivot_wr = (ts.pivot_wins / ts.pivot_trades * 100) if ts.pivot_trades > 0 else 0

        # Print table
        print("\n" + "=" * 100)
        print(" " * 35 + "LIVE TRADING STATS")
        print("=" * 100)
        print(f"  Symbol: {cfg.symbol} | Timeframe: {cfg.timeframe} | Start: {start_date} | Days: {days_trading}")
        print("-" * 100)

        # Core metrics in columns
        print(f"""
  ┌{'─'*32}┬{'─'*32}┬{'─'*32}┐
  │{'PERFORMANCE':^32}│{'RISK METRICS':^32}│{'TRADE QUALITY':^32}│
  ├{'─'*32}┼{'─'*32}┼{'─'*32}┤
  │ Total Trades:      {ts.total_trades:<11}│ Max Drawdown:    {ts.max_drawdown:>+10.2f}% │ Profit Factor:   {profit_factor:>10.2f}  │
  │ Wins / Losses:   {ts.win_count:>3} / {ts.loss_count:<7}│ Peak P&L:        {ts.peak_pl:>+10.2f}% │ Expectancy:      {expectancy:>+10.2f}% │
  │ Win Rate:        {win_rate:>10.1f}% │ Current DD:      {ts.peak_pl - ts.total_pl:>10.2f}% │ Avg Trade:       {avg_trade:>+10.2f}% │
  │ Total P&L:       {ts.total_pl:>+10.2f}% │ Max Consec Loss: {ts.max_consecutive_losses:>10}  │ Max Consec Win:  {ts.max_consecutive_wins:>10}  │
  ├{'─'*32}┼{'─'*32}┼{'─'*32}┤
  │{'PROFIT / LOSS':^32}│{'DURATION (BARS)':^32}│{'BY DIRECTION':^32}│
  ├{'─'*32}┼{'─'*32}┼{'─'*32}┤
  │ Gross Profit:    {ts.gross_profit:>+10.2f}% │ Avg Win Duration:  {avg_win_bars:>8.1f}  │ Long:  {ts.long_trades:>3} ({long_wr:>5.1f}%) {ts.long_pl:>+7.2f}% │
  │ Gross Loss:      {-ts.gross_loss:>10.2f}% │ Avg Loss Duration: {avg_loss_bars:>8.1f}  │ Short: {ts.short_trades:>3} ({short_wr:>5.1f}%) {ts.short_pl:>+7.2f}% │
  │ Avg Win:         {avg_win:>+10.2f}% │                                │                                │
  │ Avg Loss:        {-avg_loss:>10.2f}% │                                │                                │
  │ Best Trade:      {ts.max_win:>+10.2f}% │{'BY TYPE':^32}│                                │
  │ Worst Trade:     {ts.max_loss:>10.2f}% │ Momentum: {ts.momentum_trades:>3} ({momentum_wr:>5.1f}%) {ts.momentum_pl:>+6.2f}%│                                │
  │                                │ Pivot:    {ts.pivot_trades:>3} ({pivot_wr:>5.1f}%) {ts.pivot_pl:>+6.2f}%│                                │
  └{'─'*32}┴{'─'*32}┴{'─'*32}┘""")

        print("=" * 100 + "\n")

    def get_stats_json(self) -> Dict[str, Any]:
        """Return all stats as JSON-serializable dictionary"""
        ts = self.trade_state
        cfg = self.config

        # Calculate derived stats
        win_rate = (ts.win_count / ts.total_trades * 100) if ts.total_trades > 0 else 0
        profit_factor = ts.gross_profit / ts.gross_loss if ts.gross_loss > 0 else 0
        avg_win = ts.gross_profit / ts.win_count if ts.win_count > 0 else 0
        avg_loss = ts.gross_loss / ts.loss_count if ts.loss_count > 0 else 0
        avg_trade = ts.total_pl / ts.total_trades if ts.total_trades > 0 else 0
        avg_win_bars = ts.total_win_bars / ts.win_count if ts.win_count > 0 else 0
        avg_loss_bars = ts.total_loss_bars / ts.loss_count if ts.loss_count > 0 else 0
        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)

        # Calculate days since first trade
        days_trading = 0
        first_trade_date = None
        if ts.first_trade_time:
            first_trade_dt = datetime.fromtimestamp(ts.first_trade_time / 1000, tz=timezone.utc)
            first_trade_date = first_trade_dt.strftime('%Y-%m-%d %H:%M:%S')
            days_trading = (datetime.now(timezone.utc) - first_trade_dt).days

        # Direction stats
        long_wr = (ts.long_wins / ts.long_trades * 100) if ts.long_trades > 0 else 0
        short_wr = (ts.short_wins / ts.short_trades * 100) if ts.short_trades > 0 else 0
        momentum_wr = (ts.momentum_wins / ts.momentum_trades * 100) if ts.momentum_trades > 0 else 0
        pivot_wr = (ts.pivot_wins / ts.pivot_trades * 100) if ts.pivot_trades > 0 else 0

        # Get trades from database
        cursor = self.db.conn.cursor()
        cursor.execute('''
            SELECT entry_time, exit_time, direction, entry_price, exit_price,
                   pnl_percent, duration_bars, is_momentum_trade, exit_reason
            FROM trades ORDER BY entry_time ASC
        ''')
        trades_raw = cursor.fetchall()

        trades_list = []
        for t in trades_raw:
            trades_list.append({
                'entry_time': datetime.fromtimestamp(t[0] / 1000, tz=timezone.utc).isoformat() if t[0] else None,
                'exit_time': datetime.fromtimestamp(t[1] / 1000, tz=timezone.utc).isoformat() if t[1] else None,
                'direction': 'LONG' if t[2] == 1 else 'SHORT',
                'entry_price': t[3],
                'exit_price': t[4],
                'pnl_percent': round(t[5], 4) if t[5] else 0,
                'duration_bars': t[6],
                'is_momentum_trade': bool(t[7]),
                'exit_reason': t[8]
            })

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades_raw:
            reason = t[8] or 'Unknown'
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'pnl': 0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['pnl'] = round(exit_reasons[reason]['pnl'] + (t[5] or 0), 4)

        # Days since first signal + per-day averages
        ms_per_day = 24 * 60 * 60 * 1000
        days_since_first = 0.0
        trades_per_day = 0.0
        wins_per_day = 0.0
        losses_per_day = 0.0
        if trades_raw:
            first_ts = trades_raw[0][0]
            last_ts = max((t[1] or t[0]) for t in trades_raw)
            if first_ts and last_ts and last_ts >= first_ts:
                days_since_first = (last_ts - first_ts) / ms_per_day
                if days_since_first > 0:
                    trades_per_day = round(ts.total_trades / days_since_first, 4)
                    wins_per_day = round(ts.win_count / days_since_first, 4)
                    losses_per_day = round(ts.loss_count / days_since_first, 4)

        return {
            'symbol': cfg.symbol,
            'timeframe': cfg.timeframe,
            'total_candles': len(self.candles),
            'first_trade_date': first_trade_date,
            'days_trading': days_trading,
            'performance': {
                'total_trades': ts.total_trades,
                'winning_trades': ts.win_count,
                'losing_trades': ts.loss_count,
                'win_rate': round(win_rate, 2),
                'total_pnl': round(ts.total_pl, 4),
                'gross_profit': round(ts.gross_profit, 4),
                'gross_loss': round(-ts.gross_loss, 4),
                'profit_factor': round(profit_factor, 4) if profit_factor != float('inf') else None,
                'expectancy': round(expectancy, 4),
                'avg_trade': round(avg_trade, 4),
                'days_since_first_signal': round(days_since_first, 4) if days_since_first else 0,
                'trades_per_day': trades_per_day,
                'wins_per_day': wins_per_day,
                'losses_per_day': losses_per_day
            },
            'risk_metrics': {
                'max_drawdown': round(ts.max_drawdown, 4),
                'peak_pnl': round(ts.peak_pl, 4),
                'current_drawdown': round(ts.peak_pl - ts.total_pl, 4),
                'max_consecutive_wins': ts.max_consecutive_wins,
                'max_consecutive_losses': ts.max_consecutive_losses
            },
            'trade_details': {
                'avg_win': round(avg_win, 4),
                'avg_loss': round(-avg_loss, 4),
                'best_trade': round(ts.max_win, 4),
                'worst_trade': round(ts.max_loss, 4),
                'avg_win_duration_bars': round(avg_win_bars, 2),
                'avg_loss_duration_bars': round(avg_loss_bars, 2)
            },
            'by_direction': {
                'long': {
                    'trades': ts.long_trades,
                    'wins': ts.long_wins,
                    'win_rate': round(long_wr, 2),
                    'pnl': round(ts.long_pl, 4)
                },
                'short': {
                    'trades': ts.short_trades,
                    'wins': ts.short_wins,
                    'win_rate': round(short_wr, 2),
                    'pnl': round(ts.short_pl, 4)
                }
            },
            'by_type': {
                'momentum': {
                    'trades': ts.momentum_trades,
                    'wins': ts.momentum_wins,
                    'win_rate': round(momentum_wr, 2),
                    'pnl': round(ts.momentum_pl, 4)
                },
                'pivot': {
                    'trades': ts.pivot_trades,
                    'wins': ts.pivot_wins,
                    'win_rate': round(pivot_wr, 2),
                    'pnl': round(ts.pivot_pl, 4)
                }
            },
            'by_exit_reason': exit_reasons,
            'open_position': {
                'in_trade': ts.in_trade,
                'direction': 'LONG' if ts.trade_dir == 1 else 'SHORT' if ts.trade_dir == -1 else None,
                'entry_price': ts.entry_price
            } if ts.in_trade else None,
            'trades': trades_list
        }

    def print_backtest_summary(self):
        """Print comprehensive P&L summary table after backtesting historical data"""
        ts = self.trade_state
        cfg = self.config

        # Fetch all trades from database
        cursor = self.db.conn.cursor()
        cursor.execute('''
            SELECT
                entry_time, exit_time, direction, entry_price, exit_price,
                stop_loss, take_profit, pnl_percent, duration_bars,
                is_momentum_trade, exit_reason
            FROM trades
            ORDER BY entry_time ASC
        ''')
        trades = cursor.fetchall()

        # Calculate statistics
        total_trades = len(trades)
        if total_trades == 0:
            print("\n" + "=" * 80)
            print(" " * 25 + "BACKTEST SUMMARY")
            print("=" * 80)
            print(f"  Symbol: {cfg.symbol} | Timeframe: {cfg.timeframe}")
            print(f"  Historical Candles: {len(self.candles)}")
            print("-" * 80)
            print("  No trades executed during backtest period")
            print("=" * 80 + "\n")
            return

        wins = [t for t in trades if t[7] > 0]
        losses = [t for t in trades if t[7] <= 0]

        total_pnl = sum(t[7] for t in trades)
        gross_profit = sum(t[7] for t in wins) if wins else 0
        gross_loss = abs(sum(t[7] for t in losses)) if losses else 0

        win_rate = (len(wins) / total_trades) * 100
        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        max_win = max(t[7] for t in trades) if trades else 0
        max_loss = min(t[7] for t in trades) if trades else 0

        # Calculate drawdown
        cumulative = 0
        peak = 0
        max_drawdown = 0
        for t in trades:
            cumulative += t[7]
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_drawdown = max(max_drawdown, drawdown)

        # Momentum vs Pivot trades
        momentum_trades = [t for t in trades if t[9]]  # is_momentum_trade
        pivot_trades = [t for t in trades if not t[9]]

        long_trades = [t for t in trades if t[2] == 1]  # direction == 1
        short_trades = [t for t in trades if t[2] == -1]

        # Days since first signal + per-day averages
        ms_per_day = 24 * 60 * 60 * 1000
        first_ts = trades[0][0] if trades else None
        last_ts = max((t[1] or t[0]) for t in trades) if trades else None
        days_since_first = 0.0
        if first_ts and last_ts and last_ts >= first_ts:
            days_since_first = (last_ts - first_ts) / ms_per_day
        trades_per_day = total_trades / days_since_first if days_since_first > 0 else 0
        wins_per_day = len(wins) / days_since_first if days_since_first > 0 else 0
        losses_per_day = len(losses) / days_since_first if days_since_first > 0 else 0

        # Print summary
        print("\n")
        print("=" * 100)
        print(" " * 35 + "BACKTEST P&L SUMMARY")
        print("=" * 100)
        print(f"  Symbol: {cfg.symbol} | Timeframe: {cfg.timeframe} | Candles: {len(self.candles)}")
        print("-" * 100)

        # Overall Statistics
        print("\n  OVERALL PERFORMANCE")
        print("  " + "-" * 40)
        print(f"  {'Total Trades:':<25} {total_trades:>10}")
        print(f"  {'Winning Trades:':<25} {len(wins):>10} ({win_rate:.1f}%)")
        print(f"  {'Losing Trades:':<25} {len(losses):>10} ({100-win_rate:.1f}%)")
        print(f"  {'Total P&L (after fee):':<25} {total_pnl:>+10.2f}%")
        print(f"  {'Gross Profit:':<25} {gross_profit:>+10.2f}%")
        print(f"  {'Gross Loss:':<25} {-gross_loss:>10.2f}%")
        print(f"  {'Profit Factor:':<25} {profit_factor:>10.2f}")
        print(f"  {'Max Drawdown:':<25} {max_drawdown:>10.2f}%")

        print("\n  AVERAGES")
        print("  " + "-" * 40)
        print(f"  {'Avg Win:':<25} {avg_win:>+10.2f}%")
        print(f"  {'Avg Loss:':<25} {-avg_loss:>10.2f}%")
        print(f"  {'Best Trade:':<25} {max_win:>+10.2f}%")
        print(f"  {'Worst Trade:':<25} {max_loss:>10.2f}%")
        print(f"  {'Avg Trade:':<25} {total_pnl/total_trades:>+10.2f}%")

        print("\n  PER-DAY AVERAGES")
        print("  " + "-" * 40)
        print(f"  {'Days Since First Signal:':<25} {days_since_first:>10.2f}")
        print(f"  {'Trades / Day:':<25} {trades_per_day:>10.2f}")
        print(f"  {'Wins / Day:':<25} {wins_per_day:>10.2f}")
        print(f"  {'Losses / Day:':<25} {losses_per_day:>10.2f}")

        # By Direction
        print("\n  BY DIRECTION")
        print("  " + "-" * 40)
        long_pnl = sum(t[7] for t in long_trades)
        short_pnl = sum(t[7] for t in short_trades)
        long_wins = len([t for t in long_trades if t[7] > 0])
        short_wins = len([t for t in short_trades if t[7] > 0])
        print(f"  {'Long Trades:':<25} {len(long_trades):>10} | P&L: {long_pnl:>+8.2f}% | WR: {(long_wins/len(long_trades)*100) if long_trades else 0:.1f}%")
        print(f"  {'Short Trades:':<25} {len(short_trades):>10} | P&L: {short_pnl:>+8.2f}% | WR: {(short_wins/len(short_trades)*100) if short_trades else 0:.1f}%")

        # By Type
        print("\n  BY TRADE TYPE")
        print("  " + "-" * 40)
        momentum_pnl = sum(t[7] for t in momentum_trades)
        pivot_pnl = sum(t[7] for t in pivot_trades)
        momentum_wins = len([t for t in momentum_trades if t[7] > 0])
        pivot_wins = len([t for t in pivot_trades if t[7] > 0])
        print(f"  {'Momentum Trades:':<25} {len(momentum_trades):>10} | P&L: {momentum_pnl:>+8.2f}% | WR: {(momentum_wins/len(momentum_trades)*100) if momentum_trades else 0:.1f}%")
        print(f"  {'Pivot Retest Trades:':<25} {len(pivot_trades):>10} | P&L: {pivot_pnl:>+8.2f}% | WR: {(pivot_wins/len(pivot_trades)*100) if pivot_trades else 0:.1f}%")

        # Exit Reasons
        exit_reasons = {}
        for t in trades:
            reason = t[10] or 'Unknown'
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'pnl': 0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['pnl'] += t[7]

        print("\n  BY EXIT REASON")
        print("  " + "-" * 40)
        for reason, data in sorted(exit_reasons.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"  {reason:<25} {data['count']:>10} trades | P&L: {data['pnl']:>+8.2f}%")

        # Trade Table
        print("\n  TRADE HISTORY")
        print("  " + "-" * 94)
        print(f"  {'#':<4} {'Entry Time':<20} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'P&L':>10} {'Exit Reason':<15} {'Type':<10}")
        print("  " + "-" * 94)

        for i, t in enumerate(trades[-20:], 1):  # Show last 20 trades
            entry_time = datetime.fromtimestamp(t[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            direction = "LONG" if t[2] == 1 else "SHORT"
            entry_price = t[3]
            exit_price = t[4]
            pnl = t[7]
            reason = t[10] or 'Unknown'
            trade_type = "Momentum" if t[9] else "Pivot"

            pnl_str = f"+{pnl:.2f}%" if pnl >= 0 else f"{pnl:.2f}%"
            pnl_color = pnl_str

            print(f"  {i:<4} {entry_time:<20} {direction:<6} {entry_price:>10.4f} {exit_price:>10.4f} {pnl_color:>10} {reason:<15} {trade_type:<10}")

        if len(trades) > 20:
            print(f"  ... showing last 20 of {len(trades)} trades")

        print("  " + "-" * 94)
        print("=" * 100)

        # Current state
        if ts.in_trade:
            print(f"\n  OPEN POSITION: {'LONG' if ts.trade_dir == 1 else 'SHORT'} @ {ts.entry_price:.4f}")
        print("\n")

    def reset_position(self):
        """
        Reset position state after warmup to start fresh for live trading.
        Preserves P&L stats from backtest but clears any open position.
        """
        ts = self.trade_state

        # Clear position state
        ts.in_trade = False
        ts.trade_dir = 0
        ts.entry_price = None
        ts.trade_highest = None
        ts.trade_lowest = None
        ts.stop_price = None
        ts.trade_entry_bar = 0
        ts.entry_time = 0
        ts.entry_signal_id = None

        # Clear pending signals
        ts.pending_support = None
        ts.pending_resistance = None
        ts.pending_long = False
        ts.pending_short = False
        ts.confirm_count_long = 0
        ts.confirm_count_short = 0

        # Clear momentum/trend confirmation
        ts.mom_confirm_count_long = 0
        ts.mom_confirm_count_short = 0
        ts.trend_confirm_count_long = 0
        ts.trend_confirm_count_short = 0

        # Clear exit state
        ts.exit_wait_count = 0
        ts.pending_exit_reason = ""

        # Clear trade flags
        ts.is_momentum_trade = False
        ts.trailing_was_activated = False

        logger.info("Position state reset for live trading - starting fresh")


# =============================================================================
# LIVE DASHBOARD
# =============================================================================

class LiveDashboard:
    """
    Modular live trading dashboard with colors and real-time updates.

    Each section is a separate method for easy customization:
    - render_header(): Symbol, timeframe, timestamp, total P&L
    - render_price_section(): Current + last 3 candles with deltas
    - render_indicators(): All active indicators in grid format
    - render_position(): Current position with live P&L
    - render_signals(): Pending signals and recent alerts
    """

    # ANSI Color Codes - Easy to modify
    class Colors:
        RESET = '\033[0m'
        BOLD = '\033[1m'
        DIM = '\033[2m'

        # Foreground colors
        GREEN = '\033[92m'      # Bullish
        RED = '\033[91m'        # Bearish
        YELLOW = '\033[93m'     # Warning/Neutral
        CYAN = '\033[96m'       # Info
        WHITE = '\033[97m'      # Default text
        MAGENTA = '\033[95m'    # Highlights
        BLUE = '\033[94m'       # Headers

        # Background colors
        BG_GREEN = '\033[42m'
        BG_RED = '\033[41m'
        BG_YELLOW = '\033[43m'
        BG_BLUE = '\033[44m'

    def __init__(self, strategy: 'SRSwingStrategy'):
        self.strategy = strategy
        self.width = 100  # Dashboard width

    def clear_screen(self):
        """Clear terminal screen"""
        print('\033[2J\033[H', end='')

    def colorize(self, text: str, color: str) -> str:
        """Apply color to text"""
        return f"{color}{text}{self.Colors.RESET}"

    def bull_bear_color(self, value: float, text: str = None) -> str:
        """Color text based on positive/negative value"""
        if text is None:
            text = f"{value:+.2f}%"
        if value > 0:
            return self.colorize(text, self.Colors.GREEN)
        elif value < 0:
            return self.colorize(text, self.Colors.RED)
        return self.colorize(text, self.Colors.YELLOW)

    def format_timeframe(self, timeframe: str) -> str:
        """Format timeframe display to include unit for numeric values."""
        tf = timeframe.strip()
        return f"{tf}m" if tf.isdigit() else tf

    def render_header(self, candle: Candle) -> list:
        """Render dashboard header with symbol, time, and total P&L"""
        cfg = self.strategy.config
        ts = self.strategy.trade_state

        from datetime import datetime, timezone
        candle_time = datetime.fromtimestamp(candle.timestamp / 1000, tz=timezone.utc)
        time_str = candle_time.strftime('%Y-%m-%d %H:%M:%S UTC')

        # Total P&L
        total_pl = ts.total_pl
        pl_color = self.Colors.GREEN if total_pl >= 0 else self.Colors.RED
        pl_str = f"{total_pl:+.2f}%"

        # Win rate
        total_trades = ts.win_count + ts.loss_count
        win_rate = (ts.win_count / total_trades * 100) if total_trades > 0 else 0

        lines = []
        lines.append(self.colorize("═" * self.width, self.Colors.BLUE))

        # Title line
        mtf_tf_display = self.format_timeframe(cfg.mtf_timeframe) if cfg.enable_mtf else ""
        title = f"  {cfg.symbol} │ {cfg.timeframe}"
        if cfg.enable_mtf:
            title += f" + {mtf_tf_display}"
        title += f"  │  {time_str}"

        # P&L on right side
        pl_section = f"Total P&L: {self.colorize(pl_str, pl_color)}  │  Trades: {total_trades} ({win_rate:.0f}% WR)"

        lines.append(self.colorize(f"  {cfg.symbol}", self.Colors.BOLD + self.Colors.CYAN) +
                    f" │ {cfg.timeframe}" +
                    (f" + {mtf_tf_display}" if cfg.enable_mtf else "") +
                    f"  │  {time_str}  │  Total P&L: {self.colorize(pl_str, pl_color)}  │  Trades: {total_trades} ({win_rate:.0f}% WR)")

        lines.append(self.colorize("─" * self.width, self.Colors.DIM))

        return lines

    def render_price_section(self, candle: Candle) -> list:
        """
        Render current price and last 3 candles with sequential deltas.

        Delta logic:
        - Current: shows candle change (open→close), delta = change from prev candle
        - Prev -1: shows delta from Prev -1 to Current
        - Prev -2: shows delta from Prev -2 to Prev -1
        - Prev -3: shows close price only (base reference)
        """
        candles = list(self.strategy.candles)
        lines = []

        # Current candle
        current_change = ((candle.close - candle.open) / candle.open) * 100
        direction = "▲" if candle.close >= candle.open else "▼"
        dir_color = self.Colors.GREEN if candle.close >= candle.open else self.Colors.RED

        lines.append(self.colorize("  PRICE ACTION", self.Colors.BOLD + self.Colors.WHITE))
        lines.append("")

        # Current candle (large display)
        price_str = f"{candle.close:.4f}"
        change_str = self.bull_bear_color(current_change)
        lines.append(f"    Current: {self.colorize(direction, dir_color)} {self.colorize(price_str, self.Colors.BOLD + self.Colors.WHITE)}  {change_str}")
        lines.append(f"             O:{candle.open:.4f}  H:{candle.high:.4f}  L:{candle.low:.4f}  C:{candle.close:.4f}")

        # Sequential delta comparison (each candle shows delta to NEXT candle)
        if len(candles) >= 4:
            lines.append("")
            lines.append(f"    {'Candle':<10} {'Close':>12} {'Δ to Next':>15}")
            lines.append(f"    {'-'*10} {'-'*12} {'-'*15}")

            # Get previous candles
            prev1 = candles[-2] if len(candles) >= 2 else None  # Prev -1
            prev2 = candles[-3] if len(candles) >= 3 else None  # Prev -2
            prev3 = candles[-4] if len(candles) >= 4 else None  # Prev -3

            # Prev -3: base reference (no delta - it's the oldest)
            if prev3:
                lines.append(f"    {'Prev -3':<10} {prev3.close:>12.4f} {self.colorize('(base)', self.Colors.DIM):>25}")

            # Prev -2: delta from Prev -2 to Prev -1
            if prev2 and prev1:
                delta_2_to_1 = ((prev1.close - prev2.close) / prev2.close) * 100
                lines.append(f"    {'Prev -2':<10} {prev2.close:>12.4f} {self.bull_bear_color(delta_2_to_1):>25}")

            # Prev -1: delta from Prev -1 to Current
            if prev1:
                delta_1_to_curr = ((candle.close - prev1.close) / prev1.close) * 100
                lines.append(f"    {'Prev -1':<10} {prev1.close:>12.4f} {self.bull_bear_color(delta_1_to_curr):>25}")

            # Current: shows the candle's own change (no forward delta)
            lines.append(f"    {'Current':<10} {candle.close:>12.4f} {self.colorize('───', self.Colors.DIM):>25}")

        lines.append("")
        return lines

    def render_indicators(self, candle: Candle) -> list:
        """Render all active indicators organized by category"""
        ind = self.strategy.indicators
        cfg = self.strategy.config

        lines = []
        lines.append(self.colorize("  INDICATORS", self.Colors.BOLD + self.Colors.WHITE))
        lines.append("")

        # ─── TREND SECTION ───
        lines.append(self.colorize("    ── Trend ──", self.Colors.CYAN))

        # Main Trend
        trend_bull = ind.trend_is_bullish
        trend_bear = ind.trend_is_bearish
        trend_str = "BULLISH" if trend_bull else "BEARISH" if trend_bear else "NEUTRAL"
        trend_color = self.Colors.GREEN if trend_bull else self.Colors.RED if trend_bear else self.Colors.YELLOW
        lines.append(f"    {cfg.trend_method:<12} {self.colorize(trend_str, trend_color):<20} Value: {ind.trend_value:.4f}")

        # MTF
        if cfg.enable_mtf:
            mtf_bull = ind.mtf_bullish and not ind.mtf_bearish
            mtf_bear = ind.mtf_bearish and not ind.mtf_bullish
            mtf_str = "BULLISH" if mtf_bull else "BEARISH" if mtf_bear else "NEUTRAL"
            mtf_color = self.Colors.GREEN if mtf_bull else self.Colors.RED if mtf_bear else self.Colors.YELLOW
            htf_dir = "Rising" if ind.htf_rising else "Falling" if ind.htf_falling else "Flat"
            mtf_tf_display = self.format_timeframe(cfg.mtf_timeframe)
            mtf_label = f"{cfg.mtf_method}({mtf_tf_display})"
            lines.append(f"    MTF {mtf_label:<12} {self.colorize(mtf_str, mtf_color):<20} Direction: {htf_dir}")

        # Gravity
        if cfg.gravity_mode != "Auto (None)":
            grav = ind.current_gravity
            grav_color = self.Colors.GREEN if grav == "Bullish" else self.Colors.RED if grav == "Bearish" else self.Colors.YELLOW
            lines.append(f"    {'Gravity':<12} {self.colorize(grav, grav_color):<20} Mode: {cfg.gravity_mode}")

        lines.append("")

        # ─── OSCILLATORS SECTION ───
        lines.append(self.colorize("    ── Oscillators ──", self.Colors.CYAN))

        # RSI (always show if filter enabled, with visual bar)
        if cfg.enable_rsi_filter:
            rsi = ind.rsi_value
            # Determine zone
            if rsi >= cfg.rsi_overbought:
                rsi_zone = "OVERBOUGHT"
                rsi_color = self.Colors.RED
            elif rsi <= cfg.rsi_oversold:
                rsi_zone = "OVERSOLD"
                rsi_color = self.Colors.GREEN
            else:
                rsi_zone = "NEUTRAL"
                rsi_color = self.Colors.YELLOW

            # Visual bar (0-100 scale, 20 chars wide)
            bar_pos = int(rsi / 5)  # 0-20
            bar = "█" * bar_pos + "░" * (20 - bar_pos)
            lines.append(f"    RSI({cfg.rsi_length})      {self.colorize(f'{rsi:5.1f}', rsi_color)}  [{bar}]  {self.colorize(rsi_zone, rsi_color)}")
            lines.append(f"                   OS: {cfg.rsi_oversold} │ OB: {cfg.rsi_overbought} │ Mode: {cfg.rsi_filter_mode}")

        # MFI (if enabled)
        if cfg.enable_mfi_momentum:
            mfi = ind.mfi_value
            if mfi >= 80:
                mfi_zone = "OVERBOUGHT"
                mfi_color = self.Colors.RED
            elif mfi <= 20:
                mfi_zone = "OVERSOLD"
                mfi_color = self.Colors.GREEN
            else:
                mfi_zone = "NEUTRAL"
                mfi_color = self.Colors.YELLOW

            mfi_signal = "BULL" if ind.mfi_bullish else "BEAR" if ind.mfi_bearish else "---"
            mfi_sig_color = self.Colors.GREEN if ind.mfi_bullish else self.Colors.RED if ind.mfi_bearish else self.Colors.DIM
            bar_pos = int(mfi / 5)
            bar = "█" * bar_pos + "░" * (20 - bar_pos)
            lines.append(f"    MFI({cfg.mfi_length})      {self.colorize(f'{mfi:5.1f}', mfi_color)}  [{bar}]  {self.colorize(mfi_zone, mfi_color)}  Signal: {self.colorize(mfi_signal, mfi_sig_color)}")

        # MACD (if enabled)
        if cfg.enable_oscillator_filter and cfg.oscillator_type == "MACD":
            macd = ind.macd_line
            signal = ind.macd_signal
            hist = ind.macd_histogram
            prev_hist = ind.prev_macd_histogram

            # Histogram direction and momentum
            hist_increasing = hist > prev_hist
            hist_color = self.Colors.GREEN if hist > 0 else self.Colors.RED
            hist_dir = "▲" if hist_increasing else "▼"

            # Cross detection
            cross_state = ""
            if macd > signal and hist > 0:
                cross_state = self.colorize("BULL CROSS", self.Colors.GREEN)
            elif macd < signal and hist < 0:
                cross_state = self.colorize("BEAR CROSS", self.Colors.RED)
            else:
                cross_state = self.colorize("---", self.Colors.DIM)

            lines.append(f"    MACD         Line: {macd:+.6f}  Signal: {signal:+.6f}")
            lines.append(f"                 Hist: {self.colorize(f'{hist:+.6f} {hist_dir}', hist_color)}  {cross_state}")
            lines.append(f"                 Mode: {cfg.macd_entry_mode}")

        lines.append("")

        # ─── MOMENTUM SECTION ───
        if cfg.enable_momentum_mode:
            lines.append(self.colorize("    ── Momentum ──", self.Colors.CYAN))

            mom_active = []

            # Breakout
            if cfg.enable_breakout:
                if ind.breakout_bull:
                    mom_active.append(self.colorize("BREAKOUT▲", self.Colors.GREEN + self.Colors.BOLD))
                elif ind.breakout_bear:
                    mom_active.append(self.colorize("BREAKOUT▼", self.Colors.RED + self.Colors.BOLD))

            # Velocity
            if cfg.enable_velocity:
                if ind.velocity_bull:
                    mom_active.append(self.colorize("VELOCITY▲", self.Colors.GREEN + self.Colors.BOLD))
                elif ind.velocity_bear:
                    mom_active.append(self.colorize("VELOCITY▼", self.Colors.RED + self.Colors.BOLD))

            # ATR Expansion
            if cfg.enable_atr_expansion:
                if ind.atr_exp_bull:
                    mom_active.append(self.colorize("ATR-EXP▲", self.Colors.GREEN))
                elif ind.atr_exp_bear:
                    mom_active.append(self.colorize("ATR-EXP▼", self.Colors.RED))

            if mom_active:
                lines.append(f"    Active:      {' │ '.join(mom_active)}")
            else:
                lines.append(f"    Active:      {self.colorize('None', self.Colors.DIM)}")

            lines.append(f"    ATR:         {ind.current_atr:.6f}  (Avg: {ind.avg_atr:.6f})")
            lines.append("")

        # ─── VOLATILITY SECTION ───
        if cfg.enable_bb:
            lines.append(self.colorize("    ── Volatility ──", self.Colors.CYAN))

            # BB position
            if candle.close > ind.bb_upper:
                bb_pos = "ABOVE UPPER"
                bb_color = self.Colors.RED
            elif candle.close < ind.bb_lower:
                bb_pos = "BELOW LOWER"
                bb_color = self.Colors.GREEN
            else:
                bb_pos = "IN CHANNEL"
                bb_color = self.Colors.YELLOW

            squeeze_str = self.colorize(" [SQUEEZE]", self.Colors.MAGENTA) if ind.bb_is_squeeze else ""
            lines.append(f"    BB({cfg.bb_length},{cfg.bb_mult})    {self.colorize(bb_pos, bb_color)}{squeeze_str}")
            lines.append(f"                 Upper: {ind.bb_upper:.4f}  Mid: {ind.bb_basis:.4f}  Lower: {ind.bb_lower:.4f}")
            lines.append(f"                 Width: {ind.bb_bandwidth:.4f}")
            lines.append("")
        lines.append("")
        return lines

    def render_position(self, candle: Candle) -> list:
        """Render current position with live P&L"""
        ts = self.strategy.trade_state
        cfg = self.strategy.config
        lines = []

        lines.append(self.colorize("  POSITION", self.Colors.BOLD + self.Colors.WHITE))
        lines.append("")

        if ts.in_trade:
            dir_str = "LONG" if ts.trade_dir == 1 else "SHORT"
            dir_color = self.Colors.GREEN if ts.trade_dir == 1 else self.Colors.RED

            # Calculate live P&L
            pnl = ((candle.close - ts.entry_price) / ts.entry_price) * 100 * ts.trade_dir
            pnl_color = self.Colors.GREEN if pnl >= 0 else self.Colors.RED

            # Distance to stop
            stop_dist = None
            if ts.stop_price:
                stop_dist = ((ts.stop_price - candle.close) / candle.close) * 100 * ts.trade_dir

            lines.append(f"    Status: {self.colorize(f'IN {dir_str}', self.Colors.BOLD + dir_color)}")
            lines.append(f"    Entry:  {ts.entry_price:.4f}")
            lines.append(f"    Current: {candle.close:.4f}")
            lines.append(f"    P&L:    {self.colorize(f'{pnl:+.2f}%', pnl_color)}")

            if ts.stop_price:
                stop_color = self.Colors.RED if abs(stop_dist) < 1 else self.Colors.YELLOW
                lines.append(f"    Stop:   {ts.stop_price:.4f} ({self.colorize(f'{stop_dist:+.2f}%', stop_color)} away)")

            if cfg.use_trailing:
                ind = self.strategy.indicators
                is_long = ts.trade_dir == 1
                is_gravity_aligned = (ind.current_gravity == "Bullish" and is_long) or (
                    ind.current_gravity == "Bearish" and not is_long
                )
                allow_trailing = not (is_gravity_aligned and cfg.gravity_disable_trail)

                if allow_trailing and ts.entry_price:
                    activation_price = ts.entry_price * (
                        1 + (1 if is_long else -1) * cfg.trailing_activation / 100
                    )
                    extreme_price = ts.trade_highest if is_long else ts.trade_lowest
                    if extreme_price is None:
                        extreme_price = candle.close
                    activated = (
                        (is_long and extreme_price >= activation_price) or
                        ((not is_long) and extreme_price <= activation_price)
                    )

                    if not activated:
                        remaining = (
                            (activation_price - candle.close) / ts.entry_price * 100
                            if is_long else
                            (candle.close - activation_price) / ts.entry_price * 100
                        )
                        remaining = max(0.0, remaining)
                        remaining_color = self.Colors.YELLOW if remaining <= 0.5 else self.Colors.DIM
                        lines.append(
                            f"    Trail:  Activate @ {activation_price:.4f} "
                            f"({self.colorize(f'{remaining:.2f}%', remaining_color)} away)"
                        )
                    else:
                        trail_stop = extreme_price * (
                            1 + (-1 if is_long else 1) * cfg.trail_callback / 100
                        )
                        if cfg.trail_protect_entry:
                            trail_stop = max(trail_stop, ts.entry_price) if is_long else min(trail_stop, ts.entry_price)
                        distance = (
                            (candle.close - trail_stop) / candle.close * 100
                            if is_long else
                            (trail_stop - candle.close) / candle.close * 100
                        )
                        distance = max(0.0, distance)
                        distance_color = self.Colors.RED if distance <= 0.3 else self.Colors.YELLOW
                        lines.append(
                            f"    Trail:  Active @ {trail_stop:.4f} "
                            f"({self.colorize(f'{distance:.2f}%', distance_color)} to exit)"
                        )
                elif cfg.use_trailing:
                    lines.append(f"    Trail:  {self.colorize('Disabled by Gravity', self.Colors.DIM)}")

            if ts.is_momentum_trade:
                lines.append(f"    Type:   {self.colorize('MOMENTUM', self.Colors.MAGENTA)}")
        else:
            lines.append(f"    Status: {self.colorize('NO POSITION', self.Colors.DIM)}")

            # Show pending signals
            pending = []
            if ts.pending_long:
                lvl = f"@ {ts.pending_support:.4f}" if ts.pending_support else ""
                pending.append(self.colorize(f"LONG {lvl}", self.Colors.GREEN))
            if ts.pending_short:
                lvl = f"@ {ts.pending_resistance:.4f}" if ts.pending_resistance else ""
                pending.append(self.colorize(f"SHORT {lvl}", self.Colors.RED))

            if pending:
                lines.append(f"    Pending: {', '.join(pending)}")

        lines.append("")
        return lines

    def render_footer(self) -> list:
        """Render dashboard footer"""
        lines = []
        lines.append(self.colorize("─" * self.width, self.Colors.DIM))
        lines.append(self.colorize("  Press Ctrl+C to stop", self.Colors.DIM))
        lines.append(self.colorize("═" * self.width, self.Colors.BLUE))
        return lines

    def render(self, candle: Candle):
        """Render full dashboard"""
        self.clear_screen()

        all_lines = []
        all_lines.extend(self.render_header(candle))
        all_lines.extend(self.render_price_section(candle))
        all_lines.extend(self.render_indicators(candle))
        all_lines.extend(self.render_position(candle))
        all_lines.extend(self.render_footer())

        print('\n'.join(all_lines))


# =============================================================================
# WEBSOCKET DATA HANDLER
# =============================================================================

async def run_strategy(config: Config, backtest_only: bool = False,
                       json_output: bool = False, output_file: str = None,
                       show_trades: bool = False, export_trades_path: str = None):
    """
    Main strategy loop with WebSocket data.

    Properly handles multi-timeframe (MTF) data by:
    1. Fetching historical HTF data during warmup
    2. Processing HTF candles with proper synchronization
    3. In live mode, watching HTF candles and calling process_htf_candle() when they complete
    """
    if backtest_only:
        reset_signal_db(config.db_path, log=not json_output)

        # Enforce local data coverage before backtest-only runs
        data_dir = "data"
        req_window_ms = config.data_check_months * 30 * 24 * 60 * 60 * 1000
        now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        req_from_ts = now_ts - req_window_ms if config.data_check_months > 0 else None

        exchange_id = normalize_exchange_id(config.exchange)
        ccxt_symbol, data_prefix, market_kind = get_exchange_symbol_config(config)
        symbol_clean = config.symbol.replace('/', '_')

        def find_local_file(timeframe: str) -> Optional[str]:
            if os.path.exists(data_dir):
                for f in os.listdir(data_dir):
                    if f.startswith(f"{data_prefix}_{timeframe}"):
                        return os.path.join(data_dir, f)
                if exchange_id == "binance" and market_kind == "spot":
                    for f in os.listdir(data_dir):
                        if f.startswith(f"{symbol_clean}_{timeframe}"):
                            return os.path.join(data_dir, f)
            return None

        main_file = find_local_file(config.timeframe)
        if not main_file:
            logger.error(
                f"Missing local data for {config.symbol} {config.timeframe}. "
                "Run: python sr_swing_strategy.py --download-data --from-date <START> --to-date now"
            )
            return

        main_candles = load_local_candles(main_file)
        if config.data_check_months > 0 and not candles_cover_range(main_candles, req_from_ts, now_ts):
            logger.error(
                f"Local {config.timeframe} data does not cover last {config.data_check_months} months. "
                "Run: python sr_swing_strategy.py --download-data --from-date <START> --to-date now"
            )
            return

        if config.enable_mtf and config.mtf_timeframe:
            if config.mtf_timeframe in ("9", "10") and config.timeframe in ("5", "5m"):
                pass
            else:
                htf_file = find_local_file(config.mtf_timeframe)
                if not htf_file:
                    logger.error(
                        f"Missing local HTF data for {config.symbol} {config.mtf_timeframe}. "
                        "Run: python sr_swing_strategy.py --download-data --from-date <START> --to-date now"
                    )
                    return

                htf_candles = load_local_candles(htf_file)
                if config.data_check_months > 0 and not candles_cover_range(htf_candles, req_from_ts, now_ts):
                    logger.error(
                        f"Local {config.mtf_timeframe} data does not cover last {config.data_check_months} months. "
                        "Run: python sr_swing_strategy.py --download-data --from-date <START> --to-date now"
                    )
                    return

    db = SignalDatabase(config.db_path)
    strategy = SRSwingStrategy(config, db)
    strategy.json_output = json_output

    # Determine if we need HTF data
    need_htf = config.enable_mtf and config.mtf_timeframe
    htf_ms = timeframe_to_ms(config.mtf_timeframe) if need_htf else 0

    exchange_id = normalize_exchange_id(config.exchange)
    ccxt_symbol, _, market_kind = get_exchange_symbol_config(config)
    ohlcv_params = get_ohlcv_params(exchange_id, market_kind)
    main_limit = adjust_ohlcv_limit(exchange_id, 500)
    htf_limit = adjust_ohlcv_limit(exchange_id, 200)

    if not json_output:
        logger.info(f"Starting SR Swing Strategy for {config.symbol} on {config.timeframe}")
        logger.info(f"Exchange: {config.exchange} ({market_kind}) | DB: {config.db_path}")
        if ccxt_symbol != config.symbol:
            logger.info(f"Mapped symbol {config.symbol} -> {ccxt_symbol}")
        if need_htf:
            logger.info(f"MTF enabled: HTF={config.mtf_timeframe}")

    if HAS_CCXT_PRO:
        exchange = getattr(ccxtpro, exchange_id)(build_exchange_options(exchange_id, market_kind))
    else:
        exchange = getattr(ccxt, exchange_id)(build_exchange_options(exchange_id, market_kind))

    use_remote_htf = False
    use_resampled_htf = False
    main_tf_ms = timeframe_to_ms(config.timeframe)
    htf_source_timeframe = config.timeframe if need_htf else None
    htf_source_ms = 0
    if need_htf:
        try:
            await exchange.load_markets()
        except Exception as load_err:
            if not json_output:
                logger.warning(f"Failed to load markets for timeframe check: {load_err}")

        use_remote_htf = not should_resample_htf(exchange, config.mtf_timeframe)
        if not use_remote_htf:
            use_resampled_htf = True
            candidate_source = config.timeframe
            if config.mtf_timeframe.strip().isdigit() and config.timeframe.lower() not in ("1m", "1"):
                candidate_source = "1m"

            if candidate_source != config.timeframe:
                timeframes = getattr(exchange, 'timeframes', None)
                if isinstance(timeframes, dict) and timeframes:
                    if candidate_source not in timeframes:
                        candidate_source = config.timeframe

            htf_source_timeframe = candidate_source

            if not json_output:
                logger.warning(
                    f"MTF timeframe {config.mtf_timeframe} not supported by exchange; "
                    f"resampling from {htf_source_timeframe}"
                )
        else:
            htf_source_timeframe = config.mtf_timeframe

        htf_source_ms = timeframe_to_ms(htf_source_timeframe) if htf_source_timeframe else 0

    htf_source_history = []

    try:
        # Fetch historical candles to warm up indicators
        if not json_output:
            logger.info("Fetching historical candles for indicator warmup...")
        historical = await exchange.fetch_ohlcv(
            ccxt_symbol,
            config.timeframe,
            limit=main_limit,
            params=ohlcv_params
        )

        # Fetch historical HTF candles if needed
        historical_htf = []
        htf_source_history = []
        htf_candles_by_time = {}
        if need_htf:
            if use_remote_htf:
                if not json_output:
                    logger.info(f"Fetching historical HTF ({config.mtf_timeframe}) candles...")
                historical_htf = await exchange.fetch_ohlcv(
                    ccxt_symbol,
                    config.mtf_timeframe,
                    limit=htf_limit,
                    params=ohlcv_params
                )
                for htf_ohlcv in historical_htf:
                    htf_candles_by_time[htf_ohlcv[0]] = htf_ohlcv
                if not json_output:
                    logger.info(f"Loaded {len(historical_htf)} HTF candles")
            else:
                if not json_output:
                    logger.info(
                        f"Resampling HTF ({config.mtf_timeframe}) candles from {htf_source_timeframe} data..."
                    )

                if htf_source_timeframe == config.timeframe:
                    htf_source_history = list(historical)
                else:
                    htf_source_limit = 1000
                    if htf_source_ms > 0:
                        target_bars = 200
                        calc_limit = int((htf_ms / htf_source_ms) * target_bars)
                        if calc_limit > 0:
                            htf_source_limit = min(1000, calc_limit)

                    htf_source_history = await exchange.fetch_ohlcv(
                        ccxt_symbol,
                        htf_source_timeframe,
                        limit=adjust_ohlcv_limit(exchange_id, htf_source_limit),
                        params=ohlcv_params
                    )

                if htf_source_history:
                    historical_htf = resample_candles(
                        htf_source_history,
                        htf_ms,
                        htf_source_history[0][0],
                        htf_source_history[-1][0]
                    )
                    for htf_ohlcv in historical_htf:
                        htf_candles_by_time[htf_ohlcv[0]] = htf_ohlcv

                if not json_output:
                    logger.info(f"Loaded {len(historical_htf)} HTF candles (resampled)")

        # Process historical data in silent mode (no signal prints)
        strategy.silent_mode = True
        last_htf_open_time = None

        for ohlcv in historical:
            main_candle_time = ohlcv[0]

            # Process HTF candles that have completed before this main candle
            if need_htf and htf_ms > 0:
                current_htf_open = get_htf_candle_open_time(main_candle_time, htf_ms)

                if last_htf_open_time is not None and current_htf_open > last_htf_open_time:
                    if last_htf_open_time in htf_candles_by_time:
                        htf_ohlcv = htf_candles_by_time[last_htf_open_time]
                        htf_candle = Candle.from_ccxt(htf_ohlcv)
                        strategy.process_htf_candle(htf_candle)

                last_htf_open_time = current_htf_open

            candle = Candle.from_ccxt(ohlcv)
            strategy.process_candle(candle)

        # Disable silent mode for live trading
        strategy.silent_mode = False

        if not json_output:
            logger.info(f"Processed {len(historical)} historical candles")

        # Print P&L summary from backtest
        if json_output:
            results = strategy.get_stats_json()
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {output_file}")
            else:
                print(json.dumps(results, indent=2))
        else:
            strategy.print_backtest_summary()

        if export_trades_path:
            export_trades_csv(db, export_trades_path)
            if not json_output:
                print(f"Trades exported to {export_trades_path}")

        if show_trades:
            if json_output:
                logger.warning("--show-trades ignored when --json is enabled")
            else:
                print_all_trades(db)

        # Exit if backtest only
        if backtest_only:
            if not json_output:
                logger.info("Backtest complete. Exiting (--backtest-only mode)")
            await exchange.close()
            db.close()
            return

        # Reset position state to start fresh for live trading
        # This ensures we don't carry over any open positions from warmup
        strategy.reset_position()

        # Create live dashboard (only for non-JSON output)
        dashboard = LiveDashboard(strategy) if not json_output else None

        if not json_output:
            logger.info("Starting live data stream...")

        # Track last processed HTF candle for live trading
        last_live_htf_open = last_htf_open_time
        htf_source_buffer = []
        last_htf_source_ts = 0
        buffer_window_ms = 0
        if need_htf and use_resampled_htf:
            if htf_source_history:
                htf_source_buffer = list(htf_source_history)
                last_htf_source_ts = htf_source_buffer[-1][0] if htf_source_buffer else 0
            buffer_window_ms = max(htf_ms * 3, htf_source_ms * 20)

        if HAS_CCXT_PRO:
            # WebSocket streaming with MTF support
            while True:
                try:
                    ohlcvs = await exchange.watch_ohlcv(
                        ccxt_symbol,
                        config.timeframe,
                        params=ohlcv_params
                    )

                    if ohlcvs:
                        latest_ohlcv = ohlcvs[-1]
                        main_candle_time = latest_ohlcv[0]

                        if need_htf and use_resampled_htf:
                            if htf_source_timeframe == config.timeframe:
                                htf_source_buffer.append(latest_ohlcv)
                                last_htf_source_ts = latest_ohlcv[0]
                            else:
                                try:
                                    htf_source_data = await exchange.fetch_ohlcv(
                                        ccxt_symbol,
                                        htf_source_timeframe,
                                        limit=20,
                                        params=ohlcv_params
                                    )
                                    for htf_ohlcv in htf_source_data:
                                        if htf_ohlcv[0] > last_htf_source_ts:
                                            htf_source_buffer.append(htf_ohlcv)
                                            last_htf_source_ts = htf_ohlcv[0]
                                except Exception as htf_src_err:
                                    logger.warning(
                                        f"Failed to fetch {htf_source_timeframe} data: {htf_src_err}"
                                    )

                            if buffer_window_ms > 0:
                                cutoff = main_candle_time - buffer_window_ms
                                if cutoff > 0:
                                    htf_source_buffer = [
                                        c for c in htf_source_buffer if c[0] >= cutoff
                                    ]

                        # Check if we need to process a new HTF candle
                        if need_htf and htf_ms > 0:
                            current_htf_open = get_htf_candle_open_time(main_candle_time, htf_ms)

                            if last_live_htf_open is not None and current_htf_open > last_live_htf_open:
                                if use_remote_htf:
                                    # New HTF period - fetch and process the completed HTF candle
                                    try:
                                        htf_data = await exchange.fetch_ohlcv(
                                            ccxt_symbol,
                                            config.mtf_timeframe,
                                            limit=2,
                                            params=ohlcv_params
                                        )
                                        if htf_data and len(htf_data) >= 2:
                                            # The second-to-last candle is the completed one
                                            completed_htf = htf_data[-2]
                                            htf_candle = Candle.from_ccxt(completed_htf)
                                            strategy.process_htf_candle(htf_candle)
                                            if not json_output:
                                                logger.debug(f"Processed HTF candle: {htf_candle.timestamp}")
                                    except Exception as htf_err:
                                        logger.warning(f"Failed to fetch HTF candle: {htf_err}")
                                else:
                                    completed_htf = build_htf_candle_from_main(
                                        htf_source_buffer, last_live_htf_open, htf_ms
                                    )
                                    if completed_htf:
                                        htf_candle = Candle.from_ccxt(completed_htf)
                                        strategy.process_htf_candle(htf_candle)
                                        if not json_output:
                                            logger.debug(f"Processed HTF candle: {htf_candle.timestamp}")
                                    else:
                                        logger.warning(
                                            f"Failed to resample HTF candle for {last_live_htf_open}"
                                        )

                            last_live_htf_open = current_htf_open

                        # Process main candle
                        candle = Candle.from_ccxt(latest_ohlcv)
                        signals = strategy.process_candle(candle)

                        # Render live dashboard
                        if dashboard:
                            dashboard.render(candle)

                            # Show signals below dashboard
                            if signals:
                                for sig in signals:
                                    sig_color = '\033[92m' if 'LONG' in sig.signal_type else '\033[91m'
                                    print(f"{sig_color}  >>> SIGNAL: {sig.signal_type} | Entry: {sig.entry_price:.4f} | SL: {sig.stop_loss:.4f if sig.stop_loss else 'N/A'} | TP: {sig.take_profit:.4f if sig.take_profit else 'N/A'}\033[0m")

                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    await asyncio.sleep(5)
        else:
            # REST polling fallback with MTF support
            last_timestamp = historical[-1][0] if historical else 0

            while True:
                try:
                    ohlcvs = await exchange.fetch_ohlcv(
                        ccxt_symbol,
                        config.timeframe,
                        limit=10,
                        params=ohlcv_params
                    )

                    for ohlcv in ohlcvs:
                        if ohlcv[0] > last_timestamp:
                            main_candle_time = ohlcv[0]

                            if need_htf and use_resampled_htf:
                                if htf_source_timeframe == config.timeframe:
                                    htf_source_buffer.append(ohlcv)
                                    last_htf_source_ts = ohlcv[0]
                                else:
                                    try:
                                        htf_source_data = await exchange.fetch_ohlcv(
                                            ccxt_symbol,
                                            htf_source_timeframe,
                                            limit=20,
                                            params=ohlcv_params
                                        )
                                        for htf_ohlcv in htf_source_data:
                                            if htf_ohlcv[0] > last_htf_source_ts:
                                                htf_source_buffer.append(htf_ohlcv)
                                                last_htf_source_ts = htf_ohlcv[0]
                                    except Exception as htf_src_err:
                                        logger.warning(
                                            f"Failed to fetch {htf_source_timeframe} data: {htf_src_err}"
                                        )

                                if buffer_window_ms > 0:
                                    cutoff = main_candle_time - buffer_window_ms
                                    if cutoff > 0:
                                        htf_source_buffer = [
                                            c for c in htf_source_buffer if c[0] >= cutoff
                                        ]

                            # Check if we need to process a new HTF candle
                            if need_htf and htf_ms > 0:
                                current_htf_open = get_htf_candle_open_time(main_candle_time, htf_ms)

                                if last_live_htf_open is not None and current_htf_open > last_live_htf_open:
                                    if use_remote_htf:
                                        try:
                                            htf_data = await exchange.fetch_ohlcv(
                                                ccxt_symbol,
                                                config.mtf_timeframe,
                                                limit=2,
                                                params=ohlcv_params
                                            )
                                            if htf_data and len(htf_data) >= 2:
                                                completed_htf = htf_data[-2]
                                                htf_candle = Candle.from_ccxt(completed_htf)
                                                strategy.process_htf_candle(htf_candle)
                                        except Exception as htf_err:
                                            logger.warning(f"Failed to fetch HTF candle: {htf_err}")
                                    else:
                                        completed_htf = build_htf_candle_from_main(
                                            htf_source_buffer, last_live_htf_open, htf_ms
                                        )
                                        if completed_htf:
                                            htf_candle = Candle.from_ccxt(completed_htf)
                                            strategy.process_htf_candle(htf_candle)
                                        else:
                                            logger.warning(
                                                f"Failed to resample HTF candle for {last_live_htf_open}"
                                            )

                                last_live_htf_open = current_htf_open

                            candle = Candle.from_ccxt(ohlcv)
                            signals = strategy.process_candle(candle)
                            last_timestamp = ohlcv[0]

                            # Render live dashboard
                            if dashboard:
                                dashboard.render(candle)

                                # Show signals below dashboard
                                if signals:
                                    for sig in signals:
                                        sig_color = '\033[92m' if 'LONG' in sig.signal_type else '\033[91m'
                                        print(f"{sig_color}  >>> SIGNAL: {sig.signal_type} | Entry: {sig.entry_price:.4f} | SL: {sig.stop_loss:.4f if sig.stop_loss else 'N/A'} | TP: {sig.take_profit:.4f if sig.take_profit else 'N/A'}\033[0m")

                    await asyncio.sleep(10)  # Poll every 10 seconds

                except Exception as e:
                    logger.error(f"Polling error: {e}")
                    await asyncio.sleep(5)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await exchange.close()
        db.close()


# =============================================================================
# HISTORICAL DATA DOWNLOAD
# =============================================================================

async def download_historical_data(config: Config, from_date: str, to_date: str, data_dir: str):
    """
    Download historical OHLCV data for backtesting.

    When MTF is enabled in config, this also downloads HTF data automatically.
    """
    import os

    # Create data directory
    os.makedirs(data_dir, exist_ok=True)

    # Parse dates
    from_dt = parse_date(from_date)
    if to_date.lower() == 'now':
        to_dt = datetime.now(timezone.utc)
    else:
        to_dt = parse_date(to_date)

    from_ts = int(from_dt.timestamp() * 1000)
    to_ts = int(to_dt.timestamp() * 1000)

    exchange_id = normalize_exchange_id(config.exchange)
    ccxt_symbol, data_prefix, market_kind = get_exchange_symbol_config(config)
    ohlcv_params = get_ohlcv_params(exchange_id, market_kind)

    # Determine timeframes to download
    timeframes_to_download = [config.timeframe]
    need_htf = config.enable_mtf and config.mtf_timeframe
    if need_htf:
        if config.mtf_timeframe.strip().isdigit():
            if config.timeframe.lower() not in ("1m", "1") and "1m" not in timeframes_to_download:
                timeframes_to_download.append("1m")
        elif config.mtf_timeframe not in timeframes_to_download:
            timeframes_to_download.append(config.mtf_timeframe)

    # Initialize exchange
    if HAS_CCXT_PRO:
        exchange = getattr(ccxtpro, exchange_id)(build_exchange_options(exchange_id, market_kind))
    else:
        exchange = getattr(ccxt, exchange_id)(build_exchange_options(exchange_id, market_kind))

    try:
        try:
            await exchange.load_markets()
        except Exception as load_err:
            logger.warning(f"Failed to load markets for timeframe validation: {load_err}")

        if getattr(exchange, 'timeframes', None):
            supported = set(exchange.timeframes.keys())
            filtered = []
            for tf in timeframes_to_download:
                if tf in supported:
                    filtered.append(tf)
                else:
                    logger.warning(f"Skipping unsupported timeframe download: {tf}")
            timeframes_to_download = filtered

        if not timeframes_to_download:
            logger.error("No supported timeframes to download after filtering")
            return

        if ccxt_symbol != config.symbol:
            logger.info(f"Mapped symbol {config.symbol} -> {ccxt_symbol}")
        logger.info(f"Downloading {config.symbol} data from {from_dt.date()} to {to_dt.date()}")
        logger.info(f"Timeframes: {', '.join(timeframes_to_download)}")

        for timeframe in timeframes_to_download:
            logger.info(f"Downloading {timeframe} data...")

            all_candles = []
            current_ts = from_ts

            while current_ts < to_ts:
                # Fetch batch of candles
                candles = await exchange.fetch_ohlcv(
                    ccxt_symbol,
                    timeframe,
                    since=current_ts,
                    limit=adjust_ohlcv_limit(exchange_id, 1000),
                    params=ohlcv_params
                )

                if not candles:
                    break

                all_candles.extend(candles)

                # Update timestamp
                current_ts = candles[-1][0] + 1

                # Progress
                progress_dt = datetime.fromtimestamp(current_ts / 1000, tz=timezone.utc)
                logger.info(f"  [{timeframe}] Downloaded up to {progress_dt.date()} - Total candles: {len(all_candles)}")

                # Rate limit
                await asyncio.sleep(0.5)

            # Filter to exact date range
            all_candles = [c for c in all_candles if from_ts <= c[0] <= to_ts]

            # Save to file
            filename = f"{data_prefix}_{timeframe}_{from_date.replace('-', '')}_{to_date.replace('-', '')}.json"
            filepath = os.path.join(data_dir, filename)

            with open(filepath, 'w') as f:
                json.dump({
                    'symbol': ccxt_symbol,
                    'input_symbol': config.symbol,
                    'exchange': exchange_id,
                    'market': market_kind,
                    'timeframe': timeframe,
                    'from_date': from_date,
                    'to_date': to_date,
                    'candles': all_candles
                }, f)

            logger.info(f"Saved {len(all_candles)} {timeframe} candles to {filepath}")

    finally:
        await exchange.close()


# =============================================================================
# BACKTEST WITH DATE RANGE
# =============================================================================

async def run_backtest(config: Config, from_date: str, to_date: str,
                       json_output: bool = False, output_file: str = None, data_dir: str = 'data',
                       show_trades: bool = False, export_trades_path: str = None,
                       config_path: Optional[str] = None):
    """
    Run backtest on historical data from date range.

    This function properly handles multi-timeframe (MTF) data by:
    1. Downloading both main timeframe and HTF data (if MTF is enabled)
    2. Synchronizing HTF candle processing with main candles
    3. Calling process_htf_candle() when new HTF candles complete
    """
    import os

    reset_signal_db(config.db_path, log=not json_output)

    # Parse dates
    from_dt = parse_date(from_date)
    if to_date.lower() == 'now':
        to_dt = datetime.now(timezone.utc)
    else:
        to_dt = parse_date(to_date)

    from_ts = int(from_dt.timestamp() * 1000)
    to_ts = int(to_dt.timestamp() * 1000)

    download_cmd = (
        f"python sr_swing_strategy.py --config {config_path} --download-data "
        f"--from-date {from_date} --to-date {to_date}"
    ) if config_path else (
        f"python sr_swing_strategy.py --download-data --from-date {from_date} --to-date {to_date}"
    )

    if not json_output:
        logger.info(f"Running backtest for {config.symbol} {config.timeframe}")
        logger.info(f"Period: {from_dt.date()} to {to_dt.date()}")

    # Determine if we need HTF data
    need_htf = config.enable_mtf and config.mtf_timeframe

    # Calculate timeframe durations in milliseconds
    main_tf_ms = timeframe_to_ms(config.timeframe)
    htf_ms = timeframe_to_ms(config.mtf_timeframe) if need_htf else 0

    if need_htf and not json_output:
        logger.info(f"MTF enabled: Main={config.timeframe} ({main_tf_ms}ms), HTF={config.mtf_timeframe} ({htf_ms}ms)")

    # Check for local data files
    exchange_id = normalize_exchange_id(config.exchange)
    _, data_prefix, market_kind = get_exchange_symbol_config(config)
    symbol_clean = config.symbol.replace('/', '_')

    def find_local_file(timeframe: str) -> Optional[str]:
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.startswith(f"{data_prefix}_{timeframe}"):
                    return os.path.join(data_dir, f)
            if exchange_id == "binance" and market_kind == "spot":
                for f in os.listdir(data_dir):
                    if f.startswith(f"{symbol_clean}_{timeframe}"):
                        return os.path.join(data_dir, f)
        return None

    async def fetch_candles(exchange, timeframe: str, from_ts: int, to_ts: int) -> List:
        """Fetch candles for a specific timeframe"""
        candles = []
        current_ts = from_ts
        while current_ts < to_ts:
            batch = await exchange.fetch_ohlcv(
                config.symbol,
                timeframe,
                since=current_ts,
                limit=1000
            )
            if not batch:
                break
            candles.extend(batch)
            current_ts = batch[-1][0] + 1
            await asyncio.sleep(0.5)
        return [c for c in candles if from_ts <= c[0] <= to_ts]

    # Load main timeframe data (local only for repeatable backtests)
    candles_data = []
    local_file = find_local_file(config.timeframe)
    req_from_ts = required_from_ts(from_ts, to_ts, config.data_check_months)

    if not local_file:
        logger.error(
            f"Missing local data for {config.symbol} {config.timeframe}. "
            f"Run: {download_cmd}"
        )
        return

    if not json_output:
        logger.info(f"Loading main TF data from {local_file}")

    local_candles = load_local_candles(local_file)
    if not candles_cover_range(local_candles, req_from_ts, to_ts):
        logger.error(
            f"Local {config.timeframe} data does not cover required window. "
            f"Run: python sr_swing_strategy.py --download-data --from-date {from_date} --to-date {to_date}"
        )
        return
    if config.data_check_months > 0 and not candles_cover_range(local_candles, from_ts, to_ts):
        logger.warning(
            f"Local {config.timeframe} data does not cover full requested range; "
            f"proceeding with last {config.data_check_months} months."
        )

    candles_data = [c for c in local_candles if from_ts <= c[0] <= to_ts]

    if not candles_data:
        logger.error("No candle data available for the specified date range")
        return

    # Load or download HTF data if needed
    htf_candles_data = []
    if need_htf:
        htf_local_file = find_local_file(config.mtf_timeframe)
        use_1m_source = config.mtf_timeframe.strip().isdigit() and config.timeframe.lower() not in ("1m", "1")

        if use_1m_source:
            one_min_file = find_local_file("1m")
            if not one_min_file:
                logger.error(
                    "Missing local 1m data for HTF resampling. "
                    f"Run: {download_cmd}"
                )
                return

            if not json_output:
                logger.info(f"Loading 1m data from {one_min_file} for HTF resample")

            one_min_candles = load_local_candles(one_min_file)
            if not candles_cover_range(one_min_candles, req_from_ts, to_ts):
                logger.error(
                    "Local 1m data does not cover required window for HTF resampling. "
                    f"Run: {download_cmd}"
                )
                return

            if config.data_check_months > 0 and not candles_cover_range(one_min_candles, from_ts, to_ts):
                logger.warning(
                    "Local 1m data does not cover full requested range; "
                    f"proceeding with last {config.data_check_months} months."
                )

            one_min_slice = [c for c in one_min_candles if from_ts <= c[0] <= to_ts]
            htf_candles_data = resample_candles(
                one_min_slice,
                htf_ms,
                from_ts,
                to_ts,
            )
            if not json_output:
                logger.info(
                    f"Resampled HTF ({config.mtf_timeframe}) from 1m data: {len(htf_candles_data)} candles"
                )

        if not htf_candles_data:
            if htf_local_file:
                if not json_output:
                    logger.info(f"Loading HTF data from {htf_local_file}")
                htf_local_candles = load_local_candles(htf_local_file)
                if not candles_cover_range(htf_local_candles, req_from_ts, to_ts):
                    logger.error(
                        f"Local {config.mtf_timeframe} data does not cover required window. "
                        f"Run: {download_cmd}"
                    )
                    return
                if config.data_check_months > 0 and not candles_cover_range(htf_local_candles, from_ts, to_ts):
                    logger.warning(
                        f"Local {config.mtf_timeframe} data does not cover full requested range; "
                        f"proceeding with last {config.data_check_months} months."
                    )
                htf_candles_data = [c for c in htf_local_candles if from_ts <= c[0] <= to_ts]
            elif config.mtf_timeframe.strip().isdigit():
                if not json_output:
                    logger.info(
                        f"Resampling {config.timeframe} data to {config.mtf_timeframe}m HTF"
                    )
                htf_candles_data = resample_candles(
                    candles_data,
                    htf_ms,
                    from_ts,
                    to_ts,
                )
            else:
                logger.error(
                    f"Missing local HTF data for {config.symbol} {config.mtf_timeframe}. "
                    f"Run: {download_cmd}"
                )
                return

        if not json_output:
            logger.info(f"Loaded {len(htf_candles_data)} HTF candles")

    if not json_output:
        logger.info(f"Processing {len(candles_data)} main candles...")

    # Create index of HTF candles by their open time for fast lookup
    htf_candles_by_time = {}
    for htf_ohlcv in htf_candles_data:
        htf_candles_by_time[htf_ohlcv[0]] = htf_ohlcv

    # Run strategy with proper HTF synchronization
    db = SignalDatabase(config.db_path)
    strategy = SRSwingStrategy(config, db)
    strategy.silent_mode = True
    strategy.json_output = json_output

    # Track which HTF candle we last processed
    last_htf_open_time = None

    for ohlcv in candles_data:
        main_candle_time = ohlcv[0]

        # Process HTF candles that have completed before this main candle
        if need_htf and htf_ms > 0:
            # Find the HTF candle that contains this main candle
            current_htf_open = get_htf_candle_open_time(main_candle_time, htf_ms)

            # Check if we've moved to a new HTF period
            if last_htf_open_time is not None and current_htf_open > last_htf_open_time:
                # The previous HTF candle has completed - process it
                # The completed HTF candle's open time is last_htf_open_time
                if last_htf_open_time in htf_candles_by_time:
                    htf_ohlcv = htf_candles_by_time[last_htf_open_time]
                    htf_candle = Candle.from_ccxt(htf_ohlcv)
                    strategy.process_htf_candle(htf_candle)

            last_htf_open_time = current_htf_open

        # Process main candle
        candle = Candle.from_ccxt(ohlcv)
        strategy.process_candle(candle)

    # Output results
    if json_output:
        results = strategy.get_stats_json()
        results['backtest_info'] = {
            'from_date': from_date,
            'to_date': to_date,
            'total_candles': len(candles_data),
            'htf_candles': len(htf_candles_data) if need_htf else 0,
            'mtf_enabled': need_htf,
            'mtf_timeframe': config.mtf_timeframe if need_htf else None
        }
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        else:
            print(json.dumps(results, indent=2))
    else:
        strategy.print_backtest_summary()

    if export_trades_path:
        export_trades_csv(db, export_trades_path)
        if not json_output:
            print(f"Trades exported to {export_trades_path}")

    if show_trades:
        if json_output:
            logger.warning("--show-trades ignored when --json is enabled")
        else:
            print_all_trades(db)

    db.close()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def parse_date(date_str: str) -> datetime:
    """Parse date string in DD-MM-YYYY format"""
    try:
        return datetime.strptime(date_str, '%d-%m-%Y')
    except ValueError:
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use DD-MM-YYYY or YYYY-MM-DD")


def timeframe_to_ms(timeframe: str) -> int:
    """
    Convert timeframe string to milliseconds.

    Supports formats:
    - Plain number: "7" = 7 minutes, "15" = 15 minutes
    - With suffix: "5m" = 5 minutes, "1h" = 1 hour, "4h" = 4 hours, "1d" = 1 day

    Args:
        timeframe: Timeframe string (e.g., "5m", "7", "1h", "4h", "1d")

    Returns:
        Duration in milliseconds
    """
    tf = timeframe.strip().lower()

    # Multipliers in milliseconds
    minute_ms = 60 * 1000
    hour_ms = 60 * minute_ms
    day_ms = 24 * hour_ms
    week_ms = 7 * day_ms

    # Check for suffix
    if tf.endswith('m'):
        return int(tf[:-1]) * minute_ms
    elif tf.endswith('h'):
        return int(tf[:-1]) * hour_ms
    elif tf.endswith('d'):
        return int(tf[:-1]) * day_ms
    elif tf.endswith('w'):
        return int(tf[:-1]) * week_ms
    else:
        # Plain number = minutes (common convention)
        return int(tf) * minute_ms


def get_htf_candle_open_time(timestamp_ms: int, htf_ms: int) -> int:
    """
    Get the open time of the HTF candle that contains the given timestamp.

    Args:
        timestamp_ms: Timestamp in milliseconds
        htf_ms: HTF candle duration in milliseconds

    Returns:
        Open time of the HTF candle in milliseconds
    """
    return (timestamp_ms // htf_ms) * htf_ms


def resample_candles(candles: List, target_tf_ms: int, from_ts: int, to_ts: int) -> List:
    """
    Resample lower timeframe candles into a higher timeframe series.

    Args:
        candles: List of [timestamp, open, high, low, close, volume]
        target_tf_ms: Target timeframe in milliseconds
        from_ts: Start timestamp (ms)
        to_ts: End timestamp (ms)

    Returns:
        List of resampled candles in OHLCV format
    """
    buckets: dict[int, dict] = {}
    for c in candles:
        ts = c[0]
        if ts < from_ts or ts > to_ts:
            continue
        bucket = (ts // target_tf_ms) * target_tf_ms
        entry = buckets.get(bucket)
        if entry is None:
            buckets[bucket] = {
                "ts": bucket,
                "open": c[1],
                "high": c[2],
                "low": c[3],
                "close": c[4],
                "volume": c[5],
            }
        else:
            entry["high"] = max(entry["high"], c[2])
            entry["low"] = min(entry["low"], c[3])
            entry["close"] = c[4]
            entry["volume"] += c[5]

    resampled = []
    for key in sorted(buckets.keys()):
        b = buckets[key]
        resampled.append([b["ts"], b["open"], b["high"], b["low"], b["close"], b["volume"]])

    return resampled


def should_resample_htf(exchange, timeframe: str) -> bool:
    """Return True when the HTF timeframe should be resampled from main candles."""
    tf = timeframe.strip().lower()
    if tf.isdigit():
        return True

    timeframes = getattr(exchange, 'timeframes', None)
    if isinstance(timeframes, dict) and timeframes:
        return tf not in timeframes

    return False


def build_htf_candle_from_main(candles: List, htf_open: int, htf_ms: int) -> Optional[List]:
    """Aggregate main timeframe candles into a single HTF candle."""
    htf_close = htf_open + htf_ms
    bucket = [c for c in candles if htf_open <= c[0] < htf_close]
    if not bucket:
        return None

    open_price = bucket[0][1]
    close_price = bucket[-1][4]
    high_price = max(c[2] for c in bucket)
    low_price = min(c[3] for c in bucket)
    volume = sum(c[5] for c in bucket)
    return [htf_open, open_price, high_price, low_price, close_price, volume]


def required_from_ts(from_ts: int, to_ts: int, months: int) -> int:
    """Compute coverage window start (last N months, default 30d/month)."""
    if months <= 0:
        return from_ts
    window_ms = months * 30 * 24 * 60 * 60 * 1000
    if to_ts - from_ts > window_ms:
        return to_ts - window_ms
    return from_ts


def candles_cover_range(candles: List, req_from_ts: int, req_to_ts: int) -> bool:
    """Check if candle list covers the required time window."""
    if not candles:
        return False
    min_ts = min(c[0] for c in candles)
    max_ts = max(c[0] for c in candles)
    return min_ts <= req_from_ts and max_ts >= req_to_ts


def load_local_candles(path: str) -> List:
    """Load OHLCV candles from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('candles', [])


def main():
    parser = argparse.ArgumentParser(
        description='SR Swing Strategy - Pine Script to Python Conversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run live trading
  python sr_swing_strategy.py

  # Backtest only (exit after P&L summary)
  python sr_swing_strategy.py --backtest-only

  # Backtest with date range
  python sr_swing_strategy.py --backtest 01-01-2025 15-01-2025

  # Output as JSON
  python sr_swing_strategy.py --backtest-only --json

  # Save results to file
  python sr_swing_strategy.py --backtest-only --json --output results.json

  # Download historical data
  python sr_swing_strategy.py --download-data --from-date 01-01-2024 --to-date 31-12-2024
        """
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--generate-config',
        action='store_true',
        help='Generate default configuration file and exit'
    )
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        help='Override trading symbol (e.g., BTC/USDT)'
    )
    parser.add_argument(
        '--timeframe', '-t',
        type=str,
        help='Override timeframe (e.g., 5m, 15m, 1h)'
    )
    parser.add_argument(
        '--backtest-only',
        action='store_true',
        help='Run backtest on historical data and exit (do not start live trading)'
    )
    parser.add_argument(
        '--backtest',
        nargs=2,
        metavar=('FROM', 'TO'),
        help='Backtest from FROM date to TO date (format: DD-MM-YYYY). Use "now" for current date.'
    )
    parser.add_argument(
        '--show-trades',
        action='store_true',
        help='Print all trades after backtest (ignored with --json)'
    )
    parser.add_argument(
        '--export-trades',
        type=str,
        help='Export all trades to CSV file after backtest'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON instead of formatted table'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save JSON results to specified file'
    )
    parser.add_argument(
        '--download-data',
        action='store_true',
        help='Download historical data for backtesting'
    )
    parser.add_argument(
        '--from-date',
        type=str,
        help='Start date for data download (DD-MM-YYYY)'
    )
    parser.add_argument(
        '--to-date',
        type=str,
        default='now',
        help='End date for data download (DD-MM-YYYY or "now")'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory to store downloaded data (default: data)'
    )

    args = parser.parse_args()

    # Generate default config if requested
    if args.generate_config:
        config = Config()
        config.to_yaml(args.config)
        print(f"Generated default configuration: {args.config}")
        return

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.from_yaml(str(config_path))
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = Config()
        logger.info("Using default configuration")

    # Override from command line
    if args.symbol:
        config.symbol = args.symbol
    if args.timeframe:
        config.timeframe = args.timeframe

    # Download data mode
    if args.download_data:
        if not args.from_date:
            print("Error: --from-date is required with --download-data")
            return
        asyncio.run(download_historical_data(
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            data_dir=args.data_dir
        ))
        return

    # Backtest with date range
    if args.backtest:
        from_date = args.backtest[0]
        to_date = args.backtest[1]
        asyncio.run(run_backtest(
            config=config,
            from_date=from_date,
            to_date=to_date,
            json_output=args.json,
            output_file=args.output,
            data_dir=args.data_dir,
            show_trades=args.show_trades,
            export_trades_path=args.export_trades,
            config_path=str(config_path)
        ))
        return

    # Run strategy (with optional backtest-only flag)
    asyncio.run(run_strategy(
        config=config,
        backtest_only=args.backtest_only,
        json_output=args.json,
        output_file=args.output,
        show_trades=args.show_trades,
        export_trades_path=args.export_trades
    ))


if __name__ == '__main__':
    main()
