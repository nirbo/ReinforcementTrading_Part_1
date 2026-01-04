# Crypto RL Trading System - Workplan

## Overview

**Goal**: Migrate the existing Forex PPO trading system to support **ByBit Perpetuals crypto trading** using an adapted version of the high win-rate SR Swing Strategy.

**Target Pairs**: SUI/USDT, SOL/USDT, LINK/USDT, BNB/USDT (perpetual futures)

---

## Executive Summary

This project transforms the existing RL trading system from:
- **Forex (EUR/USD)** → **Crypto Perpetuals (ByBit)**
- **Simple indicators** → **Sophisticated SR Swing Strategy**
- **Static fees** → **Dynamic fee fetching from exchange**
- **Single timeframe** → **Multi-timeframe analysis**

---

## Phase 1: Foundation & Data Pipeline

### 1.1 CCXT Integration Module (`crypto_data.py`)

**Objective**: Create a unified data fetching layer for ByBit perpetuals.

| Feature | Implementation |
|---------|---------------|
| OHLCV Fetching | `fetch_ohlcv(symbol, timeframe, since, limit)` |
| Multi-timeframe | Fetch LTF + HTF candles in single call |
| Fee Fetching | `fetch_trading_fee(symbol)` → dynamic taker fee |
| Batch Support | Fetch all 4 pairs in parallel |
| Caching | In-memory + disk cache for historical data |

**Key Components**:
```python
class ByBitDataFetcher:
    async def fetch_ohlcv(self, symbol: str, timeframe: str, since: int = None) -> pd.DataFrame
    async def fetch_multi_timeframe(self, symbol: str, ltf: str, htf: str) -> tuple
    async def fetch_fees(self) -> Dict[str, float]
    async def fetch_market_info(self, symbol: str) -> MarketDetails
```

**ByBit-Specific Requirements**:
- Symbol format: `SUI/USDT:USDT` (linear perpetuals)
- Category param: `{'category': 'linear'}`
- Rate limit: ~100 req/10s
- Max candles per call: 1000-2000

### 1.2 Configuration Management (`config_crypto.py`)

**Objective**: Centralize all configuration with validation.

| Config Category | Parameters |
|-----------------|------------|
| **Exchange** | `api_key`, `api_secret`, `testnet` |
| **Trading Pairs** | List: `['SUI/USDT:USDT', 'SOL/USDT:USDT', ...]` |
| **Timeframes** | `ltf='15m'`, `htf='1h'` or `'4h'` |
| **Fee Model** | `use_dynamic_fees=True`, `default_fee=0.00055` |
| **Risk** | `max_position_pct`, `max_leverage`, `daily_loss_limit` |

### 1.3 Data Storage (`data_manager.py`)

**Objective**: Persist historical data efficiently.

- **Format**: Parquet files (compressed, fast read/write)
- **Structure**: `data/{symbol}/{timeframe}/YYYY-MM.parquet`
- **Metadata**: Store fetch timestamp, source, symbol info
- **Incremental Updates**: Append only new candles since last fetch

---

## Phase 2: Feature Engineering (Strategy Migration)

### 2.1 Indicator Library (`indicators_crypto.py`)

**Objective**: Port SR Swing Strategy indicators to reusable functions.

| Indicator | Source | Notes |
|-----------|--------|-------|
| Pivot Detection | `sr_swing_strategy.py` | `detect_pivot_high/low()` |
| HMA | Custom impl | `hma()` |
| Kalman Filter | Custom impl | `KalmanFilter` class |
| Gaussian Filter | Custom impl | `gaussian_filter()` |
| Rational Quadratic Kernel | Custom impl | `rational_quadratic_kernel()` |
| Zero Lag EMA/Score | Custom impl | `zero_lag_ema()`, `zero_lag_score()` |
| ATR, RSI, MFI, MACD, BB | Standard | Via pandas-ta or custom |

### 2.2 Feature Extractor (`feature_extractor.py`)

**Objective**: Generate RL-ready state vectors from raw candles.

**State Space Design**:
```
Observation Vector (per timestep):
├── Price Features (8)
│   ├── close / sma_20
│   ├── close / sma_50
│   ├── atr / close
│   ├── rsi (14)
│   └── volume / avg_volume
│
├── Trend Features (6)
│   ├── hma_direction (-1, 0, 1)
│   ├── kalman_short / kalman_long
│   ├── htf_trend (HTF direction)
│   └── trend_strength (0-1)
│
├── Pivot/SR Features (8)
│   ├── support_touches
│   ├── resistance_touches
│   ├── distance_to_support
│   ├── distance_to_resistance
│   ├── pivot_high_detected
│   ├── pivot_low_detected
│   ├── sr_strength (0-1)
│   └── recent_high / recent_low
│
├── Momentum Features (6)
│   ├── breakout_long / breakout_short
│   ├── velocity_score
│   ├── atr_expansion
│   └── mfi_value
│
├── Filter Flags (6)
│   ├── rsi_oversold / rsi_overbought
│   ├── macd_bullish / macd_bearish
│   ├── bb_position
│   └── mtf_aligned
│
└── Position State (3)
    ├── position_direction (1, -1, 0)
    ├── unrealized_pnl_pct
    └── time_in_position (bars)
```

**Total State Dimension**: ~37 features × N history window (default: 30)

### 2.3 Multi-Timeframe Integration

**Objective**: Combine LTF (entry) + HTF (trend context).

```
┌─────────────────────────────────────────────────────────────┐
│                    HTF Analysis (1h/4h)                      │
│  • Trend direction (HMA/Kalman/Gaussian)                    │
│  • MTF momentum filter                                      │
│  • SR levels from HTF pivots                                │
│  → Provides: trend_bias, mtf_direction, htf_sr_levels       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    LTF Analysis (5m/15m)                     │
│  • Pivot detection                                          │
│  • Entry signals (support/resistance bounces)               │
│  • Momentum breakout confirmation                           │
│  • RSI/MFI/BB filters                                       │
│  → Provides: entry_signals, momentum_confirmation           │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 3: Trading Environment (`trading_env_crypto.py`)

### 3.1 Environment Specification

**Base**: Custom Gymnasium environment (extends existing `trading_env.py`)

| Component | Forex Version | Crypto Version |
|-----------|---------------|----------------|
| Symbol | EUR/USD | SUI/USDT:USDT (configurable) |
| Fee Model | Fixed spread % | Dynamic taker fee via CCXT |
| Trading Hours | 5-day market | 24/7 continuous |
| SL/TP | 5-120 pips | Percentage-based (%) |
| Position | Spot-like | Perpetual (funding not modeled) |
| Leverage | N/A | 1x-10x (configurable) |

### 3.2 Action Space

**Discrete Multi-Action**:
```
0: HOLD          - No action
1: LONG          - Open long position (if no position)
2: SHORT         - Open short position (if no position)
3: CLOSE         - Close existing position
4: LONG_TIGHT    - Long with tight SL/TP (momentum)
5: SHORT_TIGHT   - Short with tight SL/TP (momentum)
```

### 3.3 Reward Function

**Multi-Component Reward**:
```
reward = reward_pnl + reward_risk + penalty_fees + penalty_inactivity

Where:
- reward_pnl: Realized PnL in quote currency
- reward_risk: Risk-adjusted component (Sharpe-like)
- penalty_fees: -fee * position_size
- penalty_inactivity: Small negative for no trades (encourage action)
```

**Key Design Decisions**:
- Use percentage returns (scale-invariant across assets)
- Dynamic fees pulled from exchange API
- Clip extreme rewards to prevent policy collapse

### 3.4 Position Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max Position | 100% equity | Per trade |
| Max Leverage | 5x | Configurable |
| SL Type | Percentage | 2-5% depending on volatility |
| TP Type | Percentage/Risk:Reward | 2:1 or 3:1 |
| Trailing SL | Optional | Via config |
| Time Exit | Optional | Max bars in trade |

---

## Phase 4: PPO Training Pipeline

### 4.1 Training Configuration (`train_crypto.py`)

**Extends**: `train_agent.py` with crypto-specific enhancements.

| Parameter | Forex Default | Crypto Value |
|-----------|---------------|--------------|
| Total Steps | 600K | 1-2M (higher volatility) |
| Learning Rate | 3e-4 | 1e-4 to 3e-4 |
| Batch Size | 64 | 64-128 |
| N Steps | 2048 | 2048-4096 |
| Entropy Coeff | 0.01 | 0.005-0.02 |
| Clip Range | 0.2 | 0.1-0.2 |
| VF Coeff | 0.5 | 0.5-1.0 |

### 4.2 Curriculum Learning

**Phase 1**: Single pair (SOL/USDT) - highest liquidity
**Phase 2**: Add SUI/USDT - different volatility profile
**Phase 3**: Add LINK/USDT - mean-reverting behavior
**Phase 4**: Add BNB/USDT - lower volatility

### 4.3 Evaluation Framework

- **Out-of-sample**: Last 20% of data (time-based)
- **Metrics**: Sharpe, Sortino, Calmar, Win Rate, Max Drawdown
- **Comparison**: vs Buy & Hold, vs Buy & Hold rebalanced

---

## Phase 5: Testing & Validation

### 5.1 Backtesting Engine

**Purpose**: Validate trained model on historical data.

| Component | Implementation |
|-----------|---------------|
| Data | Held-out test period (not seen during training) |
| Execution | Vectorized backtest (fast) |
| Slippage | Bid/ask spread modeling |
| Fees | Dynamic from exchange API |
| Analysis | Trade list, equity curve, metrics |

### 5.2 Walk-Forward Validation

```
Training Period    Validation Period    Test Period
├─────────────────┼────────────────────┼─────────────────┤
│  2023-07 to     │  2024-01 to        │  2024-07 to     │
│  2024-01        │  2024-07           │  2025-01        │
│  (6 months)     │  (6 months)        │  (6 months)     │
```

### 5.3 Robustness Tests

| Test | Description |
|------|-------------|
| Monte Carlo | Shuffle trade order, resample returns |
| Sensitivity | Vary fee %, SL/TP, position size |
| Regime Change | Test on high/low volatility periods |
| Pair Robustness | Train on A, test on B |

---

## Phase 6: Deployment & Monitoring

### 6.1 Paper Trading Mode

**Purpose**: Validate in real-time without risk.

- Connect to ByBit testnet
- Execute via CCXT (no actual funds)
- Log all signals and simulated PnL

### 6.2 Live Trading Integration

**When ready for production**:

| Component | Implementation |
|-----------|---------------|
| API Connection | CCXT with authenticated ByBit client |
| Order Management | Async order placement, SL/TP, trailing stops |
| Risk Checks | Pre-trade: position limits, daily loss cap |
| Monitoring | Real-time PnL, drawdown alerts, uptime |
| Emergency Stop | Manual kill switch, auto-close positions |

### 6.3 Monitoring Dashboard

**Metrics to track**:
- Training: Episode reward, policy loss, value loss
- Backtest: Sharpe, max drawdown, win rate, profit factor
- Live: Realized PnL, open positions, daily change

---

## Dependencies & Tech Stack

### Core
- `ccxt` - Exchange connectivity
- `stable-baselines3` - PPO implementation
- `gymnasium` - RL environment interface
- `pandas`, `numpy` - Data processing
- `pandas-ta` - Technical indicators

### Optional Enhancements
- `tensorboard` - Training visualization
- `ray` - Hyperparameter optimization (rllib)
- `optuna` - Hyperparameter search

### Development
- `pytest` - Testing
- `black`, `isort` - Code formatting
- `mypy` - Type checking

---

## File Structure

```
/Users/nir/ReinforcementTrading_Part_1/
├── data/                          # Historical data
│   ├── raw/                       # Raw OHLCV (Parquet)
│   ├── processed/                 # Feature matrices
│   └── fees/                      # Fee snapshots
│
├── src/
│   ├── crypto_data.py             # CCXT integration
│   ├── config_crypto.py           # Configuration management
│   ├── data_manager.py            # Data storage/retrieval
│   │
│   ├── indicators_crypto.py       # Indicator library
│   ├── feature_extractor.py       # State vector generation
│   │
│   ├── trading_env_crypto.py      # RL environment
│   ├── train_crypto.py            # Training pipeline
│   ├── evaluate_crypto.py         # Backtesting
│   │
│   └── utils/
│       ├── bybit_utils.py         # ByBit-specific helpers
│       └── metrics.py             # Performance metrics
│
├── configs/
│   ├── default.yaml               # Default configuration
│   ├── pairs/
│   │   ├── sui_usdt.yaml
│   │   ├── sol_usdt.yaml
│   │   ├── link_usdt.yaml
│   │   └── bnb_usdt.yaml
│   └── htf/
│       ├── htf_1h.yaml
│       └── htf_4h.yaml
│
├── models/
│   ├── checkpoints/               # Training snapshots
│   ├── tensorboard/               # TB logs
│   └── final/                     # Best models
│
├── tests/
│   ├── test_indicators.py
│   ├── test_features.py
│   └── test_env.py
│
├── scripts/
│   ├── fetch_data.sh              # Data collection script
│   └── run_training.sh            # Training runner
│
├── docs/
│   ├── crypto-workplan.md         # This file
│   ├── api.md                     # API documentation
│   └── strategy.md                # Strategy reference
│
└── Requirements.txt               # Updated dependencies
```

---

## Success Criteria

### Phase Completion

| Phase | Criteria |
|-------|----------|
| Phase 1 | CCXT fetches all 4 pairs, dynamic fees working |
| Phase 2 | Feature extractor produces valid state vectors |
| Phase 3 | Environment passes Gymnasium checks |
| Phase 4 | PPO trains without errors, converges |
| Phase 5 | Backtest shows positive expectancy |
| Phase 6 | Paper trading validates real-time signals |

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Sharpe Ratio | > 1.0 | Risk-adjusted returns |
| Win Rate | > 55% | Aligned with SR Swing strategy |
| Max Drawdown | < 15% | Per trade / overall |
| Profit Factor | > 1.5 | Gross profits / gross losses |
| Expectancy | > 0 | Positive expected value per trade |

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting | High | Walk-forward validation, cross-pair testing |
| Market regime change | Medium | Curriculum learning, adaptive position sizing |
| Exchange API issues | Medium | Rate limiting, retries, fallbacks |
| Slippage impact | Medium | Conservative slippage estimates in backtest |
| Liquidity (SUI) | Low-Med | Start with SOL, add others gradually |

---

## Timeline & Milestones

| Milestone | Description | Duration |
|-----------|-------------|----------|
| M1 | Data pipeline complete | Week 1 |
| M2 | Feature extractor working | Week 2 |
| M3 | RL environment passes tests | Week 3 |
| M4 | First trained model | Week 4-5 |
| M5 | Backtest validation | Week 6 |
| M6 | Paper trading ready | Week 7-8 |

---

## Open Questions & Decisions

### Technical Decisions Needed

1. **Feature Normalization**: Per-pair or cross-pair normalization?
2. **Episode Length**: Fixed bars (e.g., 1000) or full data with reset?
3. **Position Sizing**: Fixed % or learned via policy?
4. **Fee Incorporation**: Dynamic per-trade or average?

### Strategic Questions

1. **Leverage**: Start with 1x (no leverage) or allow up to 5x?
2. **Single vs Multi-Agent**: One model per pair or unified?
3. **HTF Selection**: Use 1h, 4h, or adaptive based on pair?

---

## References

- **Original Codebase**: `/Users/nir/ReinforcementTrading_Part_1/`
- **Strategy Source**: `/Users/nir/trading-bot-and-signal/sr_swing_strategy.py`
- **CCXT Docs**: https://docs.ccxt.com/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **ByBit API**: https://bybit-exchange.github.io/docs/v5/intro

---

*Document Version*: 1.0
*Created*: 2026-01-04
*Last Updated*: 2026-01-04
