# Crypto RL Trading System

A production-grade reinforcement learning system for trading cryptocurrency perpetual futures on ByBit. Built with PyTorch 2.9+, Stable-Baselines3, and Gymnasium.

**Current Best Model:** `models/best/ppo_crypto_final.zip`
- **Win Rate:** 39.1% | **Profit Factor:** 1.19 | **R:R:** 1.86:1 | **Final Equity:** +41.4%

---

## Quick Start Guide

### 1. Install Dependencies

```bash
cd ReinforcementTrading_Part_1
python -m venv venv
source venv/bin/activate
pip install -r Requirements.txt
```

### 2. Download Historical Data

```bash
# Download 2 years of data for all pairs (SOL, BNB, SUI, LINK)
python scripts/download_data.py --days 730

# Or download specific pair
python -c "
from src.crypto.config import load_config
from src.crypto.data_manager import DataManager

config = load_config()
dm = DataManager(config)
dm.collect_historical('SOL/USDT:USDT', '5m', days=730)
dm.collect_historical('SOL/USDT:USDT', '9m', days=730)
"
```

### 3. Train a Model

```bash
# Train with default config (10M steps, ~4-6 hours on GPU)
python -m src.crypto.train --data data/raw/SOL_USDT_USDT/5m/data.parquet --output models/

# Train with custom config
python -m src.crypto.train --config configs/default.yaml --output models/

# Resume training from checkpoint
python -m src.crypto.train --resume models/best/ppo_crypto_final.zip --output models/continued/
```

### 4. Evaluate Model (Backtest)

```bash
# Backtest on trained pair
python -m src.crypto.evaluate --model models/best/ppo_crypto_final.zip --data data/raw/SOL_USDT_USDT/5m/data.parquet

# Backtest on different pair (generalization test)
python -m src.crypto.evaluate --model models/best/ppo_crypto_final.zip --data data/raw/BNB_USDT_USDT/5m/data.parquet --symbol "BNB/USDT:USDT"
```

### 5. Run Diagnostics

```bash
# Analyze model performance by exit type, direction, etc.
python scripts/diagnose_model.py --model models/best/ppo_crypto_final.zip --data data/raw/SOL_USDT_USDT/5m/data.parquet
```

---

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Overview](#overview)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Configuration System](#configuration-system)
6. [Technical Indicators](#technical-indicators)
7. [Feature Extraction](#feature-extraction)
8. [Trading Environment](#trading-environment)
9. [Training Pipeline](#training-pipeline)
10. [Backtesting Engine](#backtesting-engine)
11. [Data Management](#data-management)
12. [Critical Learnings](#critical-learnings)
13. [API Reference](#api-reference)
14. [Testing](#testing)
15. [Examples](#examples)

---

## Overview

### What This System Does

This system trains a PPO (Proximal Policy Optimization) agent to trade cryptocurrency perpetual futures. The agent learns to:

- **Enter positions** (long or short) at optimal times
- **Let SL/TP manage exits** (forced by design - see Critical Learnings)
- **Maintain proper R:R ratio** via 3% SL / 6% TP (2:1 risk/reward)

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Timeframe Analysis** | LTF (5m) for entries, HTF (9m) for trend confirmation |
| **39+ Normalized Features** | Scale-invariant, bounded features for stable learning |
| **3 Discrete Actions** | HOLD, LONG, SHORT (CLOSE removed - see Critical Learnings) |
| **Forced SL/TP Exits** | Positions ONLY exit via stop-loss or take-profit |
| **Dynamic Fees** | Real-time fee fetching from ByBit API |
| **Walk-Forward Validation** | Out-of-sample testing for robustness |

### Trading Pairs

Default configuration targets high-liquidity ByBit perpetuals:

```
SOL/USDT:USDT  (default, best results)
BNB/USDT:USDT  (generalizes well)
SUI/USDT:USDT
LINK/USDT:USDT
```

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  DataManager                                                     │
│  ├── CCXT (ByBit API)                                           │
│  ├── Parquet Storage (data/raw/, data/processed/)              │
│  └── Fee Caching                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INDICATOR LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  indicators.py                                                   │
│  ├── Moving Averages: SMA, EMA, WMA, RMA, HMA, ALMA, ZLEMA     │
│  ├── Filters: Kalman, Gaussian, Rational Quadratic Kernel       │
│  ├── Momentum: RSI, MFI, MACD, Zero-Lag Score                  │
│  ├── Volatility: ATR, Bollinger Bands                          │
│  └── Structure: Pivot Detection, Support/Resistance             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FEATURE LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  feature_extractor.py                                            │
│  ├── 39+ Scale-Invariant Features                               │
│  ├── Bounded Values (clipped to [-1, 1] or [0, 1])             │
│  ├── Position State Features (direction, PnL, time)            │
│  └── HTF Integration (trend alignment)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ENVIRONMENT LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  trading_env.py (Gymnasium-compatible)                           │
│  ├── Observation: (window_size, feature_dim + 3)               │
│  ├── Action Space: Discrete(3) - HOLD, LONG, SHORT             │
│  ├── Reward: Realized PnL (exits via SL/TP only)               │
│  └── SL/TP Management                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  train.py (Stable-Baselines3 PPO)                               │
│  ├── MlpPolicy (fully-connected network)                        │
│  ├── TradingMetricsCallback (custom evaluation)                 │
│  ├── TensorBoard Logging                                         │
│  └── Checkpoint Management                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EVALUATION LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  evaluate.py                                                     │
│  ├── Backtester (single-pass, deterministic)                   │
│  ├── Walk-Forward Validation                                     │
│  ├── Monte Carlo Simulation                                      │
│  └── Benchmark Comparison (Buy & Hold)                          │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
ReinforcementTrading_Part_1/
├── configs/
│   └── default.yaml          # Default configuration
├── data/
│   ├── raw/                  # Raw OHLCV parquet files
│   └── processed/            # Processed feature data
├── models/                   # Trained models (gitignored)
├── src/
│   └── crypto/
│       ├── __init__.py       # Package exports
│       ├── config.py         # Pydantic configuration models
│       ├── data_manager.py   # CCXT + Parquet storage
│       ├── indicators.py     # Technical analysis library
│       ├── feature_extractor.py  # RL feature engineering
│       ├── trading_env.py    # Gymnasium environment
│       ├── train.py          # PPO training pipeline
│       └── evaluate.py       # Backtesting engine
├── tests/
│   ├── conftest.py           # Pytest fixtures
│   ├── test_config.py
│   ├── test_indicators.py
│   ├── test_features.py
│   ├── test_env.py
│   ├── test_train.py
│   └── test_evaluate.py
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- **Python**: 3.11 or 3.12
- **CUDA**: 12.8+ (for RTX 5090/Blackwell GPUs)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)

### Setup

```bash
# Clone repository
git clone <repo-url>
cd ReinforcementTrading_Part_1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For RTX 5090/Blackwell: Install CUDA 12.8+ PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Verify Installation

```bash
# Run tests
python -m pytest tests/ -v

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Quick Start

### 1. Training with Synthetic Data

```python
from src.crypto.train import train_ppo, generate_synthetic_data
from src.crypto.config import load_config

# Generate synthetic data (for testing)
df = generate_synthetic_data(10000, seed=42)

# Split train/validation
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
val_df = df.iloc[split_idx:]

# Train PPO agent
config = load_config()
config.training.total_timesteps = 100_000  # Quick test
model, stats = train_ppo(train_df, val_df, config=config)

print(f"Training complete: {stats}")
```

### 2. Evaluation

```python
from src.crypto.evaluate import Backtester, generate_report

# Create backtester
backtester = Backtester(model, config)

# Run backtest
result = backtester.run(val_df, symbol="SOL/USDT:USDT", timeframe="15m")

# Generate report
report = generate_report(result)
print(report)
```

### 3. Command-Line Training

```bash
# Train with synthetic data
python -m src.crypto.train --synthetic --output models/

# Train with real data
python -m src.crypto.train --data data/raw/SOL_USDT_USDT/15m/data.parquet

# Resume training
python -m src.crypto.train --resume models/20240101_120000/ppo_crypto_final.zip
```

---

## Configuration System

### Configuration Hierarchy

Configuration loads in priority order:

1. **Explicit path** (passed to `load_config(path)`)
2. **Environment variable** (`CRYPTO_CONFIG=/path/to/config.yaml`)
3. **Default file** (`configs/default.yaml`)
4. **Hardcoded defaults** (in Pydantic models)

### Configuration Sections

#### Exchange Configuration

```yaml
exchange:
  name: bybit              # Exchange (only bybit supported)
  testnet: true            # Use testnet (SAFETY DEFAULT)
  rate_limit_per_second: 10.0  # API rate limiting
```

**Environment Variables:**
- `BYBIT_API_KEY`: Your ByBit API key
- `BYBIT_API_SECRET`: Your ByBit API secret

#### Trading Pairs

```yaml
pairs:
  pairs:
    - SUI/USDT:USDT        # ByBit linear perpetual format
    - SOL/USDT:USDT
    - LINK/USDT:USDT
    - BNB/USDT:USDT
  default_pair: SOL/USDT:USDT
```

**Symbol Format**: `BASE/QUOTE:SETTLE` where `:USDT` indicates USDT-margined linear perpetual.

#### Timeframes

```yaml
timeframes:
  ltf: 15m    # Low timeframe (entries)
  htf: 1h     # High timeframe (trend)
```

**Valid Timeframes**: `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `12h`, `1d`

#### Fee Configuration

```yaml
fees:
  use_dynamic_fees: true         # Fetch real fees from API
  default_taker_fee: 0.00055     # 0.055% fallback
  default_maker_fee: 0.0002      # 0.02% fallback
  fee_cache_ttl_seconds: 3600    # Cache fees for 1 hour
```

#### Risk Management

```yaml
risk:
  max_position_pct: 1.0          # 100% equity per trade
  max_leverage: 5                # Maximum 5x leverage
  daily_loss_limit_pct: 0.10     # 10% daily loss limit

  # Standard SL/TP (swing trades)
  default_sl_pct: 0.02           # 2% stop loss
  default_tp_pct: 0.04           # 4% take profit (2:1 R:R)

  # Tight SL/TP (momentum trades)
  tight_sl_pct: 0.01             # 1% tight stop
  tight_tp_pct: 0.015            # 1.5% tight TP
```

#### Indicator Settings

```yaml
indicators:
  # Pivot Detection
  pivot_left_bars: 4             # Bars before pivot
  pivot_right_bars: 3            # Bars after pivot
  use_high_low: true             # Use high/low vs close

  # Trend Filter
  trend_method: HMA              # HMA, Kalman, Gaussian, EMA, SMA, ALMA
  trend_length: 14               # Trend indicator period
  kalman_short_len: 18           # Short Kalman period
  kalman_long_len: 23            # Long Kalman period

  # Momentum
  breakout_lookback: 40          # Breakout detection window
  breakout_vol_multiplier: 1.1   # Volume surge threshold
  velocity_threshold: 3.5        # Velocity signal threshold
  velocity_bars: 3               # Velocity lookback
  atr_length: 14                 # ATR period

  # RSI/MFI
  rsi_length: 14
  mfi_length: 14
  rsi_oversold: 30.0
  rsi_overbought: 70.0

  # MACD
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9

  # Bollinger Bands
  bb_length: 20
  bb_mult: 2.0
```

#### Training Hyperparameters

```yaml
training:
  # PPO Core
  total_timesteps: 1000000       # Total training steps
  learning_rate: 0.0003          # Adam learning rate
  n_steps: 2048                  # Steps per rollout
  batch_size: 64                 # Minibatch size
  n_epochs: 10                   # Epochs per update
  gamma: 0.99                    # Discount factor
  gae_lambda: 0.95               # GAE lambda
  clip_range: 0.2                # PPO clip range
  ent_coef: 0.01                 # Entropy coefficient
  vf_coef: 0.5                   # Value function coefficient
  max_grad_norm: 0.5             # Gradient clipping

  # Environment
  window_size: 30                # Observation window (bars)
  episode_max_steps: 2000        # Max steps per episode

  # Checkpointing
  checkpoint_freq: 50000         # Save every N steps
  eval_freq: 10000               # Evaluate every N steps

  # Reproducibility
  seed: 42
```

#### Data Storage

```yaml
data:
  data_dir: data
  raw_dir: data/raw
  processed_dir: data/processed
  compression: snappy            # Parquet compression
  min_bars: 10000                # Minimum bars for training
```

### Programmatic Configuration

```python
from src.crypto.config import CryptoConfig, load_config

# Load and modify
config = load_config()
config.training.total_timesteps = 500_000
config.risk.default_sl_pct = 0.015

# Save custom config
config.to_yaml("configs/custom.yaml")

# Load custom config
config = CryptoConfig.from_yaml("configs/custom.yaml")
```

---

## Technical Indicators

The system implements a comprehensive technical analysis library in `indicators.py`. All functions are vectorized for efficiency.

### Moving Averages

| Function | Description | Formula |
|----------|-------------|---------|
| `sma(data, length)` | Simple Moving Average | $\frac{1}{n}\sum_{i=0}^{n-1} x_i$ |
| `ema(data, length)` | Exponential Moving Average | $\alpha \cdot x_t + (1-\alpha) \cdot EMA_{t-1}$ |
| `wma(data, length)` | Weighted Moving Average | $\frac{\sum w_i \cdot x_i}{\sum w_i}$ |
| `rma(data, length)` | Wilder's Smoothed MA | $\alpha = \frac{1}{n}$, same as EMA |
| `hma(data, length)` | Hull Moving Average | $WMA(2 \cdot WMA(\frac{n}{2}) - WMA(n), \sqrt{n})$ |
| `alma(data, length, offset, sigma)` | Arnaud Legoux MA | Gaussian-weighted with offset |
| `zero_lag_ema(data, length)` | Zero-Lag EMA | Compensates for EMA lag |

### Advanced Filters

#### Kalman Filter

Adaptive filter that tracks price with minimal lag:

```python
from src.crypto.indicators import KalmanFilter, kalman_trend

# Single filter
kf = KalmanFilter(length=14, r=0.01, q=0.1)
for price in prices:
    estimate = kf.update(price)

# Trend detection (short vs long Kalman)
short_k, long_k, direction = kalman_trend(close, short_len=18, long_len=23)
# direction: 1 (bullish), -1 (bearish), 0 (neutral)
```

#### Gaussian Filter

Ehlers 2-pole Gaussian smoothing:

```python
from src.crypto.indicators import gaussian_filter

smoothed = gaussian_filter(close, length=14, poles=2)
```

#### Rational Quadratic Kernel

Nadaraya-Watson estimator for non-linear smoothing:

```python
from src.crypto.indicators import rational_quadratic_kernel

kernel_estimate = rational_quadratic_kernel(close, lookback=50, relative_weight=1.0)
```

### Momentum Indicators

| Function | Description | Range |
|----------|-------------|-------|
| `rsi(close, length)` | Relative Strength Index | 0-100 |
| `mfi(high, low, close, volume, length)` | Money Flow Index | 0-100 |
| `macd(close, fast, slow, signal)` | MACD | Returns (macd, signal, histogram) |
| `zero_lag_score(close, length)` | Zero-Lag Momentum Score | -length to +length |

### Volatility Indicators

```python
from src.crypto.indicators import atr, bollinger_bands, stdev

# Average True Range
atr_values = atr(high, low, close, length=14)

# Bollinger Bands
upper, middle, lower, bandwidth = bollinger_bands(close, length=20, mult=2.0)
```

### Pivot Detection

```python
from src.crypto.indicators import (
    detect_pivot_high,
    detect_pivot_low,
    support_resistance_levels,
)

# Detect pivots
pivot_highs = detect_pivot_high(high, close, left_bars=4, right_bars=3)
pivot_lows = detect_pivot_low(low, close, left_bars=4, right_bars=3)

# Get current S/R levels
support, resistance = support_resistance_levels(high, low, close)
```

### Breakout & Velocity Signals

```python
from src.crypto.indicators import breakout_signal, velocity_signal, atr_expansion

# Breakout detection
breakout_long, breakout_short = breakout_signal(
    close, high, low, volume,
    lookback=40,
    vol_multiplier=1.1
)

# Velocity signals
velocity_long, velocity_short = velocity_signal(
    close,
    threshold=3.5,
    bars=3,
    atr_series=atr_values
)

# ATR expansion (volatility breakout)
is_expanding = atr_expansion(high, low, close, length=14, multiplier=3.5)
```

### All-in-One Computation

```python
from src.crypto.indicators import compute_all_indicators

# Compute all indicators at once
df_with_indicators = compute_all_indicators(ohlcv_df, config.indicators.model_dump())

# Returns DataFrame with 40+ additional columns:
# sma_20, sma_50, ema_20, atr, trend_value, trend_direction,
# kalman_short, kalman_long, kalman_direction, hma, hma_direction,
# rsi, mfi, macd, macd_signal, macd_histogram,
# bb_upper, bb_basis, bb_lower, bb_width, bb_position,
# pivot_high, pivot_low, support, resistance,
# dist_to_support, dist_to_resistance,
# breakout_long, breakout_short, velocity_long, velocity_short,
# atr_expansion, volume_ratio
```

---

## Feature Extraction

The `FeatureExtractor` class transforms raw OHLCV + indicators into RL-ready observations.

### Design Principles

1. **Scale-Invariant**: All features are ratios or percentages
2. **Bounded**: Values clipped to reasonable ranges ([-1, 1] or [0, 1])
3. **Normalized**: No extreme outliers that could destabilize learning
4. **Position-Aware**: Current position state included in observation

### Feature Categories

#### Price Features (8 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `close_sma20_ratio` | Close / SMA(20) | [0.8, 1.2] |
| `close_sma50_ratio` | Close / SMA(50) | [0.7, 1.3] |
| `atr_close_ratio` | ATR / Close | [0, 0.1] |
| `rsi` | RSI normalized | [0, 1] |
| `rsi_normalized` | (RSI - 50) / 50 | [-1, 1] |
| `volume_ratio` | Volume / Avg(Volume, 20) | [0, 1] |
| `log_return` | log(close / prev_close) | [-0.1, 0.1] |
| `high_low_range` | (High - Low) / Close | [0, 0.1] |

#### Trend Features (6 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `hma_direction` | Sign of HMA change | [-1, 1] |
| `kalman_direction` | Kalman trend direction | [-1, 1] |
| `kalman_ratio` | Short Kalman / Long Kalman | Scaled to [-1, 1] |
| `trend_direction` | Primary trend direction | [-1, 1] |
| `trend_strength` | Deviation from MA | [0, 1] |
| `price_vs_trend` | (Price - Trend) / ATR | [-1, 1] |

#### Pivot/Structure Features (8 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `dist_to_support` | Distance to support / ATR | [-1, 1] |
| `dist_to_resistance` | Distance to resistance / ATR | [-1, 1] |
| `sr_position` | Position between S/R | [0, 1] |
| `pivot_high_detected` | Pivot high at this bar | [0, 1] |
| `pivot_low_detected` | Pivot low at this bar | [0, 1] |
| `recent_high_dist` | Distance to 20-bar high | [0, 1] |
| `recent_low_dist` | Distance to 20-bar low | [0, 1] |
| `sr_touch_count` | Recent S/R touches | [0, 1] |

#### Momentum Features (7 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `breakout_long` | Long breakout signal | [0, 1] |
| `breakout_short` | Short breakout signal | [0, 1] |
| `velocity_long` | Long velocity signal | [0, 1] |
| `velocity_short` | Short velocity signal | [0, 1] |
| `atr_expansion` | Volatility expansion | [0, 1] |
| `mfi` | Money Flow Index | [0, 1] |
| `mfi_normalized` | (MFI - 50) / 50 | [-1, 1] |

#### Filter Features (7 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `rsi_oversold` | RSI < 30 | [0, 1] |
| `rsi_overbought` | RSI > 70 | [0, 1] |
| `macd_bullish` | MACD histogram > 0 | [0, 1] |
| `macd_bearish` | MACD histogram < 0 | [0, 1] |
| `macd_histogram_norm` | Normalized MACD histogram | [-1, 1] |
| `bb_position` | Position within Bollinger Bands | [0, 1] |
| `bb_squeeze` | Bollinger Band squeeze | [0, 1] |

#### HTF Features (2 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `htf_trend` | Higher timeframe trend | [-1, 1] |
| `mtf_aligned` | LTF & HTF trend aligned | [0, 1] |

#### Position State (3 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `position_direction` | Current position | [-1, 1] |
| `unrealized_pnl_pct` | Unrealized PnL | [-1, 1] |
| `time_in_position_norm` | Time in position | [0, 1] |

### Usage

```python
from src.crypto.feature_extractor import FeatureExtractor, PositionState

# Create extractor
extractor = FeatureExtractor(config, window_size=30)

print(f"Feature dimension: {extractor.feature_dim}")  # 39+
print(f"Observation shape: {extractor.observation_shape}")  # (30, 39+)

# Extract features
features_df = extractor.extract_features(ohlcv_df)

# Get observation for RL
position = PositionState(direction=1, unrealized_pnl_pct=0.02, time_in_position=10)
obs = extractor.get_observation(ohlcv_df, current_idx=100, position=position)
# obs.shape: (30, feature_dim)

# Precompute all features
feature_matrix = extractor.precompute_features(ohlcv_df)
# feature_matrix.shape: (n_bars, feature_dim)
```

---

## Trading Environment

The `CryptoTradingEnv` is a Gymnasium-compatible environment for RL training.

### Action Space (3 Discrete Actions)

| Action | Value | Description |
|--------|-------|-------------|
| `HOLD` | 0 | Do nothing |
| `LONG` | 1 | Open long position |
| `SHORT` | 2 | Open short position |

**CLOSE action was removed** - see [Critical Learnings](#critical-learnings).

### Action Semantics

- **SL/TP Only Exits**: Positions exit ONLY via stop-loss or take-profit (no manual close)
- **FLIP Disabled**: Cannot change direction while in position (prevents exit gaming)
- **Standard SL/TP**: 3% SL, 6% TP (2:1 risk/reward)

### Observation Space

```
Shape: (window_size, feature_dim + 3)
       = (30, 39 + 3) = (30, 42)

Features: 39 market features + 3 position features
Window: Rolling window of last 30 bars
Dtype: float32
```

### Reward Function

The reward function combines realized and unrealized components:

```python
reward = 0

# 1. Entry fee (immediate cost)
if opening_position:
    reward -= taker_fee * reward_scale  # -0.055%

# 2. Exit reward (realized PnL)
if closing_position:
    pnl_pct = (exit_price - entry_price) / entry_price * direction
    net_pnl = pnl_pct - 2 * taker_fee  # Entry + exit fees
    reward += net_pnl * reward_scale

# 3. Unrealized PnL shaping (while in position)
if in_position:
    delta_unrealized = current_unrealized - previous_unrealized
    reward += delta_unrealized * unrealized_weight * reward_scale

# 4. Hold penalty (optional, encourages trading)
if flat and not trading:
    reward -= hold_penalty * reward_scale

# Reward scaling: default reward_scale=100 for better learning
```

### Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 30 | Observation window (bars) |
| `taker_fee` | 0.00055 | Taker fee (0.055%) |
| `random_start` | True | Random starting position |
| `min_episode_bars` | 500 | Minimum episode length |
| `max_episode_bars` | 2000 | Maximum episode length |
| `reward_scale` | 100.0 | Reward multiplier |
| `unrealized_pnl_weight` | 0.1 | Unrealized PnL shaping weight |
| `hold_penalty` | 0.0 | Penalty for holding flat |

### SL/TP Mechanics

```python
# Long position SL/TP
sl_price = entry_price * (1 - sl_pct)  # e.g., 100 * 0.98 = 98
tp_price = entry_price * (1 + tp_pct)  # e.g., 100 * 1.04 = 104

# Short position SL/TP
sl_price = entry_price * (1 + sl_pct)  # e.g., 100 * 1.02 = 102
tp_price = entry_price * (1 - tp_pct)  # e.g., 100 * 0.96 = 96

# Checked on each bar using high/low prices
# Conservative: if both SL and TP hit on same bar, assumes SL hit first
```

### Usage

```python
from src.crypto.trading_env import CryptoTradingEnv, Action

env = CryptoTradingEnv(
    df=ohlcv_df,
    config=config,
    window_size=30,
    random_start=True,
)

# Gymnasium interface
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(Action.LONG)

# Get statistics
stats = env.get_trade_stats()
# {'n_trades': 50, 'win_rate': 0.54, 'final_equity': 1.23, 'max_drawdown': 0.08, ...}
```

---

## Training Pipeline

### PPO Configuration

The system uses Stable-Baselines3's PPO implementation:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `policy` | MlpPolicy | Fully-connected network |
| `learning_rate` | 3e-4 | Adam learning rate |
| `n_steps` | 2048 | Steps per rollout |
| `batch_size` | 64 | Minibatch size |
| `n_epochs` | 10 | Update epochs |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | PPO clip range |
| `ent_coef` | 0.01 | Entropy bonus |
| `vf_coef` | 0.5 | Value function coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping |

### TradingMetricsCallback

Custom callback that evaluates trading performance during training:

```python
# Logged to TensorBoard every eval_freq steps:
eval/n_trades        # Number of trades
eval/win_rate        # Win rate percentage
eval/final_equity    # Final equity
eval/max_drawdown    # Maximum drawdown
eval/avg_pnl         # Average PnL per trade
eval/profit_factor   # Gross profit / gross loss
eval/best_equity     # Best equity achieved
```

### Training Function

```python
from src.crypto.train import train_ppo

model, stats = train_ppo(
    train_df=train_data,          # Training OHLCV
    val_df=val_data,              # Validation OHLCV
    config=config,                # Configuration
    output_dir="models",          # Output directory
    htf_train_df=htf_train,       # Optional HTF data
    htf_val_df=htf_val,           # Optional HTF data
    resume_path=None,             # Resume from checkpoint
)

# Returns:
# - model: Trained PPO model
# - stats: Final validation statistics
```

### Output Structure

```
models/20240101_120000/
├── config.yaml              # Training configuration
├── checkpoints/
│   ├── ppo_crypto_50000_steps.zip
│   ├── ppo_crypto_100000_steps.zip
│   └── ...
├── tensorboard/
│   └── PPO_1/
│       └── events.out.tfevents.*
├── ppo_crypto_final.zip     # Final model
└── final_stats.json         # Final validation stats
```

### TensorBoard Monitoring

```bash
tensorboard --logdir models/20240101_120000/tensorboard/
```

### Key Metrics to Monitor

1. **rollout/ep_rew_mean**: Average episode reward (should increase)
2. **train/loss**: PPO loss (should stabilize)
3. **train/entropy_loss**: Entropy (should gradually decrease)
4. **eval/final_equity**: Validation equity (primary metric)
5. **eval/max_drawdown**: Drawdown (should stay low)
6. **eval/win_rate**: Win rate (target: >50%)

---

## Backtesting Engine

### Backtester Class

```python
from src.crypto.evaluate import Backtester, BacktestResult

backtester = Backtester(
    model=trained_model,
    config=config,
    slippage_pct=0.0001,  # 0.01% slippage
)

result = backtester.run(
    df=test_data,
    htf_df=htf_data,
    symbol="SOL/USDT:USDT",
    timeframe="15m",
    deterministic=True,
)
```

### Performance Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| `total_return` | Absolute return | $E_{final} - E_{initial}$ |
| `total_return_pct` | Percentage return | $(E_{final} - E_{initial}) / E_{initial}$ |
| `annualized_return` | Annual return | $(1 + R)^{1/years} - 1$ |
| `sharpe_ratio` | Risk-adjusted return | $\frac{\bar{r} - r_f}{\sigma_r} \times \sqrt{252 \times 24 \times 4}$ |
| `sortino_ratio` | Downside-adjusted return | $\frac{\bar{r} - r_f}{\sigma_{down}} \times \sqrt{252 \times 24 \times 4}$ |
| `calmar_ratio` | Return / max drawdown | $\frac{R_{annual}}{|DD_{max}|}$ |
| `max_drawdown` | Largest peak-to-trough | $\min(E - E_{peak})$ |
| `profit_factor` | Gross profit / loss | $\frac{\sum wins}{\sum losses}$ |
| `expectancy` | Expected value per trade | $W \times \bar{win} + (1-W) \times \bar{loss}$ |

### Walk-Forward Validation

Test model on multiple out-of-sample periods:

```python
results = backtester.walk_forward(
    df=full_data,
    n_splits=5,           # 5 time periods
    train_pct=0.7,        # 70% train, 30% test each split
    symbol="SOL/USDT:USDT",
    timeframe="15m",
)

# Returns list of BacktestResult for each test period
for i, result in enumerate(results):
    print(f"Fold {i+1}: Return={result.total_return_pct:.2%}, Sharpe={result.sharpe_ratio:.2f}")
```

### Monte Carlo Simulation

Shuffle trade order to estimate statistical confidence:

```python
mc_results = backtester.monte_carlo(
    df=test_data,
    n_simulations=100,
    symbol="SOL/USDT:USDT",
)

print(f"Base equity: {mc_results['base_final_equity']:.4f}")
print(f"MC mean:     {mc_results['mc_final_equity_mean']:.4f}")
print(f"MC 5th pct:  {mc_results['mc_final_equity_5th']:.4f}")
print(f"MC 95th pct: {mc_results['mc_final_equity_95th']:.4f}")
```

### Benchmark Comparison

Compare strategy to buy-and-hold:

```python
from src.crypto.evaluate import compare_to_benchmark, generate_report

comparison = compare_to_benchmark(result, test_data)

print(f"Strategy return: {comparison['strategy_return']:.2%}")
print(f"Benchmark return: {comparison['benchmark_return']:.2%}")
print(f"Excess return: {comparison['excess_return']:.2%}")
print(f"Outperformed: {comparison['outperformed']}")

# Full report
report = generate_report(result, comparison)
print(report)
```

### Report Output

```
============================================================
BACKTEST REPORT: SOL/USDT:USDT (15m)
============================================================

PERIOD
----------------------------------------
Start:           2024-01-01 00:00:00
End:             2024-06-01 00:00:00

PERFORMANCE
----------------------------------------
Initial Equity:  1.0000
Final Equity:    1.2500
Total Return:    25.00%
Annualized:      50.00%

RISK METRICS
----------------------------------------
Sharpe Ratio:    1.800
Sortino Ratio:   2.500
Calmar Ratio:    4.000
Max Drawdown:    -6.00%

TRADE STATISTICS
----------------------------------------
Total Trades:    150
Win Rate:        60.00%
Profit Factor:   1.80
Avg Win:         0.008000
Avg Loss:        -0.004000
Avg Trade:       0.001670
Avg Bars Held:   12.5
Largest Win:     0.080000
Largest Loss:    -0.030000
Expectancy:      0.003200

BENCHMARK COMPARISON (Buy & Hold)
----------------------------------------
Strategy Return: 25.00%
Benchmark Return:10.00%
Excess Return:   15.00%
Strategy Sharpe: 1.800
Benchmark Sharpe:0.800
Outperformed:    Yes

============================================================
```

---

## Data Management

### DataManager Class

```python
from src.crypto.data_manager import DataManager

dm = DataManager(config)
```

### Fetching OHLCV Data

```python
# Single fetch (up to 1000 bars)
df = dm.fetch_ohlcv("SOL/USDT:USDT", "15m", limit=1000)

# Range fetch (handles pagination)
from datetime import datetime, timezone, timedelta

end = datetime.now(timezone.utc)
start = end - timedelta(days=30)
df = dm.fetch_ohlcv_range("SOL/USDT:USDT", "15m", start, end)

# Update (incremental, adds new bars)
new_bars = dm.update_ohlcv("SOL/USDT:USDT", "15m")
print(f"Added {new_bars} new bars")
```

### Storage Operations

```python
# Save to parquet (merges with existing)
dm.save_ohlcv(df, "SOL/USDT:USDT", "15m")
# Saved to: data/raw/SOL_USDT_USDT/15m/data.parquet

# Load from parquet
df = dm.load_ohlcv("SOL/USDT:USDT", "15m")

# Load with date filter
df = dm.load_ohlcv("SOL/USDT:USDT", "15m", start=start_date, end=end_date)

# Check bar count
count = dm.get_bar_count("SOL/USDT:USDT", "15m")

# Check if data is fresh
is_fresh = dm.is_data_fresh("SOL/USDT:USDT", "15m", max_age_bars=10)
```

### Multi-Timeframe Operations

```python
# Fetch LTF and HTF together
ltf_df, htf_df = dm.fetch_multi_timeframe("SOL/USDT:USDT")
# Uses config.timeframes.ltf (15m) and htf (1h)

# Fetch all configured pairs
all_data = dm.fetch_all_pairs("15m", update=True)
# Returns: {"SOL/USDT:USDT": df1, "SUI/USDT:USDT": df2, ...}
```

### Historical Collection

```python
# Collect 1 year of history
df = dm.collect_historical("SOL/USDT:USDT", "15m", days=365)

# Collect for all pairs
from src.crypto.data_manager import collect_training_data

collect_training_data(
    pairs=["SOL/USDT:USDT", "SUI/USDT:USDT"],
    timeframes=["15m", "1h"],
    days=365,
    config=config,
)
```

### Data Validation

```python
validation = dm.validate_data("SOL/USDT:USDT", "15m")

print(f"Valid: {validation['valid']}")
print(f"Bars: {validation['bars']}")
print(f"Gaps: {validation['gaps']}")
print(f"NaN count: {validation['nan_count']}")
print(f"Invalid OHLC: {validation['invalid_ohlc']}")
print(f"Date range: {validation['start']} to {validation['end']}")
```

### Fee Fetching

```python
# Get real-time fee (cached for 1 hour)
fee = dm.fetch_fee("SOL/USDT:USDT")
print(f"Taker fee: {fee:.5f}")  # e.g., 0.00055
```

---

## Critical Learnings

These are breakthrough discoveries that dramatically improved model profitability. **Do not revert these changes.**

### 1. Remove CLOSE Action (Critical!)

**Problem**: With CLOSE action available, the model exited 85.7% of trades early via MANUAL_CLOSE:
- Average win: +0.80% (should be +6% at TP)
- R:R ratio: 0.72:1 (inverted from 2:1 target)
- Result: **-328% total loss**

**Solution**: Remove CLOSE action entirely (3 actions: HOLD, LONG, SHORT):
- Forces model to use SL/TP for exits
- R:R ratio restored: 1.86:1
- Result: **+41% profit**

```python
# In trading_env.py
class Action(IntEnum):
    HOLD = 0
    LONG = 1
    SHORT = 2
    # CLOSE removed - model closed winners early, destroying R:R
```

### 2. Disable FLIP (Critical!)

**Problem**: After removing CLOSE, model learned to FLIP (LONG→SHORT) as exit strategy:
- 91.8% of trades were FLIP exits
- Still avoided SL/TP, destroying R:R ratio

**Solution**: Disable FLIP capability - positions can ONLY exit via SL/TP:
```python
# In trading_env.py step():
elif action == Action.LONG:
    if self.position.direction == 0:  # Only if flat
        reward += self._open_position(1, tight=False)
    # FLIP disabled - no position change if already long or short
```

### 3. Win Rate Math

With forced SL/TP exits:
- **SL**: 3% loss
- **TP**: 6% gain
- **R:R**: 2:1

Breakeven win rate = 1 / (1 + R:R) = 1 / 3 = **33.3%**

Current model: **39.1% win rate** → Profitable!

### 4. Exit Type Analysis

Always analyze trades by exit type:
```
TP_HIT: Goal exits - maximize these (target >25%)
SL_HIT: Where losses come from - minimize (<30%)
MANUAL_CLOSE: Should be 0% with CLOSE removed
FLIP: Should be 0% with FLIP disabled
END_OF_DATA: Neutral - just episode end
```

### 5. Generalization

Model trained on SOL generalizes to high-liquidity pairs:
| Pair | Profit Factor | PnL |
|------|---------------|-----|
| SOL (trained) | 1.19 | +46.7% |
| BNB | 1.04 | +6.8% |
| SUI | 0.92 | -27.9% |
| LINK | 0.93 | -21.9% |

**Insight**: Works best on high-volume, low-spread pairs. Consider multi-pair training for better generalization.

---

## Hyperparameter Experimentation

Once you have a working model, systematic experimentation helps find optimal trading parameters.

### Key Parameters to Experiment With

| Category | Parameter | Default | Range to Test |
|----------|-----------|---------|---------------|
| **Timeframes** | LTF | 5m | 1m, 3m, 5m, 15m |
| | HTF | 9m | 15m, 30m, 1h, 4h |
| **Risk** | SL % | 3% | 1.5%, 2%, 3%, 4%, 5% |
| | TP % | 6% | 3%, 4%, 6%, 8%, 10% |
| | R:R ratio | 2:1 | 1.5:1, 2:1, 2.5:1, 3:1 |
| **Indicators** | ATR length | 14 | 7, 14, 21 |
| | RSI length | 14 | 7, 14, 21 |
| | HMA length | 14 | 10, 14, 20, 30 |
| | Pivot bars | 4/3 | 3/2, 4/3, 5/4 |
| **Training** | Window size | 30 | 20, 30, 50 |
| | Entropy coef | 0.02 | 0.01, 0.02, 0.05 |

### Running Experiments

#### Method 1: Manual Config Variation

```bash
# Create experiment config
cp configs/default.yaml configs/exp_sl2_tp4.yaml
# Edit SL/TP values, then train
python -m src.crypto.train --config configs/exp_sl2_tp4.yaml --output models/exp_sl2_tp4/
```

#### Method 2: Training Experiments (Hours)

Train new models with different hyperparameters:

```bash
# Run predefined experiment grid (trains new model per config)
python scripts/run_experiments.py --experiments sl_tp_grid

# Run specific training experiment
python scripts/run_experiments.py --sl 2.0 --tp 4.0 --ltf 5m --htf 1h

# Quick validation mode (1M steps instead of 10M)
python scripts/run_experiments.py --experiments sl_tp_grid --quick

# List available experiment grids
python scripts/run_experiments.py --list-experiments
```

#### Method 3: Inference Experiments (Minutes) - RECOMMENDED

Test existing trained model with different SL/TP - **no retraining needed**:

```bash
# SL × TP matrix (16 combinations, ~10-15 minutes)
python scripts/inference_experiments.py --model models/best/ppo_crypto_final.zip \
    --sl 1.5 2.0 2.5 3.0 --tp 3.0 4.0 5.0 6.0

# Fine-grained search around known good values
python scripts/inference_experiments.py --model models/best/ppo_crypto_final.zip \
    --sl 2.5 2.75 3.0 3.25 3.5 --tp 5.0 5.5 6.0 6.5 7.0

# Test on different trading pair
python scripts/inference_experiments.py --model models/best/ppo_crypto_final.zip \
    --sl 2.0 3.0 --tp 4.0 6.0 --pair BNB/USDT:USDT

# Export results to CSV
python scripts/inference_experiments.py --model models/best/ppo_crypto_final.zip \
    --sl 2.0 3.0 --tp 4.0 6.0 --csv results.csv
```

**Why inference experiments are faster:**
- The model already learned **when to enter** (long/short timing)
- SL/TP is applied at **execution time**, not learned
- So you can test different SL/TP without retraining!

**Output example:**
```
SL × TP MATRIX - PROFIT_FACTOR
==============================
SL \\ TP    3.0%   4.0%   5.0%   6.0%
  1.5%     0.92   1.05   1.12   1.08
  2.0%     0.98   1.11   1.18   1.15
  2.5%     1.02   1.15   1.21   1.19
  3.0%     1.05   1.17   1.22   1.19

Best: SL=2.5% TP=5.0% → profit_factor=1.21
```

#### Method 4: Python API

```python
from scripts.run_experiments import run_experiment, compare_experiments

# Run single experiment
result = run_experiment(
    name="sl2_tp6",
    sl_pct=0.02,
    tp_pct=0.06,
    ltf="5m",
    htf="1h",
    timesteps=1_000_000,  # Quick test
)

# Compare multiple experiments
compare_experiments(["sl2_tp4", "sl2_tp6", "sl3_tp6", "sl3_tp9"])
```

### Experiment Design Guidelines

1. **Change one variable at a time** - Isolate effects
2. **Use same seed** - Reproducibility (`seed: 42` in config)
3. **Quick validation first** - 1M steps before full 10M
4. **Track all metrics** - Not just PnL (win rate, drawdown, trade count)

### Priority Experiments

Based on current model (39% WR, 1.19 PF, 1.86 R:R):

| Priority | Experiment | Hypothesis |
|----------|------------|------------|
| 1 | **SL/TP ratio sweep** | Tighter SL (2%) might catch fewer bad entries |
| 2 | **HTF variations** | 1h or 4h HTF might filter noise better |
| 3 | **LTF variations** | 15m might reduce noise vs 5m |
| 4 | **Multi-pair training** | SOL+BNB combined might generalize better |

### Comparing Results

```bash
# After running experiments, compare results
python scripts/compare_experiments.py models/exp_*/

# Output:
# ┌─────────────┬──────────┬─────────┬───────┬─────────┐
# │ Experiment  │ Win Rate │ PF      │ R:R   │ Equity  │
# ├─────────────┼──────────┼─────────┼───────┼─────────┤
# │ sl2_tp4     │ 42.1%    │ 1.15    │ 1.89  │ +32.4%  │
# │ sl3_tp6     │ 39.1%    │ 1.19    │ 1.86  │ +41.4%  │
# │ sl2_tp6     │ 35.2%    │ 1.21    │ 2.74  │ +28.1%  │
# └─────────────┴──────────┴─────────┴───────┴─────────┘
```

### Metrics to Optimize

| Metric | Target | Why |
|--------|--------|-----|
| **Profit Factor** | > 1.3 | Primary - overall profitability |
| **Win Rate** | > 40% | Psychological sustainability |
| **Max Drawdown** | < 25% | Risk management |
| **Trade Count** | > 50 | Statistical significance |
| **Sharpe Ratio** | > 1.0 | Risk-adjusted returns |

### Saving Experiment Results

All experiments save to `models/experiments/`:
```
models/experiments/
├── sl2_tp4_5m_1h/
│   ├── config.yaml
│   ├── ppo_crypto_final.zip
│   ├── final_stats.json
│   └── tensorboard/
├── sl3_tp6_5m_1h/
│   └── ...
└── comparison_report.md
```

---

## API Reference

### Configuration

```python
from src.crypto.config import (
    CryptoConfig,         # Root configuration
    load_config,          # Load configuration (convenience)
    ExchangeConfig,       # Exchange settings
    TradingPairsConfig,   # Trading pairs
    TimeframeConfig,      # LTF/HTF settings
    FeeConfig,            # Fee model
    RiskConfig,           # Risk management
    IndicatorConfig,      # Indicator parameters
    TrainingConfig,       # PPO hyperparameters
    DataConfig,           # Storage paths
)
```

### Indicators

```python
from src.crypto.indicators import (
    # Moving Averages
    sma, ema, wma, rma, hma, alma, zero_lag_ema,

    # Filters
    KalmanFilter, kalman_filter, kalman_trend,
    gaussian_filter, rational_quadratic_kernel,

    # Momentum
    rsi, mfi, macd, zero_lag_score,

    # Volatility
    atr, stdev, bollinger_bands,

    # Trend
    calculate_trend,

    # Structure
    detect_pivot_high, detect_pivot_low,
    support_resistance_levels,

    # Signals
    breakout_signal, velocity_signal, atr_expansion,

    # Convenience
    compute_all_indicators,
)
```

### Feature Extraction

```python
from src.crypto.feature_extractor import (
    FeatureExtractor,     # Main extractor class
    PositionState,        # Position state dataclass
    FEATURE_SPECS,        # Feature specifications
    create_feature_matrix,  # Convenience function
)
```

### Trading Environment

```python
from src.crypto.trading_env import (
    CryptoTradingEnv,     # Gymnasium environment
    Action,               # Action enum
    Trade,                # Trade record
    make_crypto_env,      # Factory function
)
```

### Training

```python
from src.crypto.train import (
    train_ppo,               # Main training function
    evaluate_model,          # Model evaluation
    create_env,              # Environment factory
    TradingMetricsCallback,  # Custom callback
    generate_synthetic_data, # Test data generation
)
```

### Evaluation

```python
from src.crypto.evaluate import (
    Backtester,              # Backtesting engine
    BacktestResult,          # Result dataclass
    Trade,                   # Trade record
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_drawdown_curve,
    compare_to_benchmark,
    generate_report,
    run_full_evaluation,     # Complete pipeline
)
```

### Data Management

```python
from src.crypto.data_manager import (
    DataManager,             # Main data manager
    collect_training_data,   # Bulk collection
)
```

---

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_indicators.py -v

# Run with coverage
python -m pytest tests/ --cov=src/crypto --cov-report=html

# Run fast (skip slow tests)
python -m pytest tests/ -v -m "not slow"
```

### Test Structure

| File | Coverage |
|------|----------|
| `test_config.py` | Configuration loading, validation |
| `test_indicators.py` | All indicator functions |
| `test_features.py` | Feature extraction, normalization |
| `test_env.py` | Environment Gymnasium compliance |
| `test_train.py` | Training pipeline, callbacks |
| `test_evaluate.py` | Backtesting, metrics |

### Test Fixtures

```python
# In tests/conftest.py

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate 500 bars of synthetic OHLCV data."""

@pytest.fixture
def sample_ohlcv_htf() -> pd.DataFrame:
    """Generate 100 bars of HTF data."""

@pytest.fixture
def config() -> CryptoConfig:
    """Load default configuration."""

@pytest.fixture
def small_config() -> CryptoConfig:
    """Configuration with smaller values for fast testing."""
```

---

## Examples

### Example 1: Complete Training Pipeline

```python
from src.crypto.config import load_config
from src.crypto.data_manager import DataManager
from src.crypto.train import train_ppo
from src.crypto.evaluate import Backtester, generate_report, compare_to_benchmark

# 1. Setup
config = load_config()
dm = DataManager(config)

# 2. Collect data
dm.collect_historical("SOL/USDT:USDT", "15m", days=180)
dm.collect_historical("SOL/USDT:USDT", "1h", days=180)

# 3. Load data
ltf_df, htf_df = dm.fetch_multi_timeframe("SOL/USDT:USDT")

# 4. Train/val split
split = int(len(ltf_df) * 0.8)
train_df, val_df = ltf_df.iloc[:split], ltf_df.iloc[split:]
htf_train, htf_val = htf_df.iloc[:split//4], htf_df.iloc[split//4:]

# 5. Train
model, stats = train_ppo(
    train_df, val_df, config,
    htf_train_df=htf_train,
    htf_val_df=htf_val,
)

# 6. Backtest
backtester = Backtester(model, config)
result = backtester.run(val_df, htf_val, "SOL/USDT:USDT", "15m")

# 7. Report
benchmark = compare_to_benchmark(result, val_df)
print(generate_report(result, benchmark))
```

### Example 2: Custom Training Configuration

```python
from src.crypto.config import CryptoConfig

config = CryptoConfig(
    training=TrainingConfig(
        total_timesteps=2_000_000,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=15,
        ent_coef=0.005,  # Lower entropy for exploitation
    ),
    risk=RiskConfig(
        default_sl_pct=0.015,  # Tighter stops
        default_tp_pct=0.03,
        tight_sl_pct=0.008,
        tight_tp_pct=0.012,
    ),
    indicators=IndicatorConfig(
        trend_method="Kalman",  # Use Kalman instead of HMA
        rsi_oversold=25,
        rsi_overbought=75,
    ),
)

model, stats = train_ppo(train_df, val_df, config=config)
```

### Example 3: Multi-Pair Training

```python
from src.crypto.data_manager import DataManager, collect_training_data
from src.crypto.train import train_ppo
import pandas as pd

# Collect data for all pairs
collect_training_data(days=365)

# Load and combine
dm = DataManager(config)
all_data = dm.fetch_all_pairs("15m")

combined = pd.concat([df.assign(symbol=sym) for sym, df in all_data.items()])
combined = combined.sort_index()

# Train on combined data
split = int(len(combined) * 0.8)
model, stats = train_ppo(combined.iloc[:split], combined.iloc[split:])
```

---

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `python -m pytest tests/ -v`
4. Submit a pull request

## Acknowledgments

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for the PPO implementation
- [CCXT](https://github.com/ccxt/ccxt) for exchange integration
- [Gymnasium](https://gymnasium.farama.org/) for the RL environment interface
