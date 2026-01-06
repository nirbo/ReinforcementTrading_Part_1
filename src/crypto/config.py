"""
Configuration management for crypto RL trading system.

Supports:
- Pydantic validation
- Environment variable overrides
- YAML file configuration
- Sensible defaults for ByBit perpetuals
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ExchangeConfig(BaseModel):
    """Exchange connection settings."""

    name: str = "bybit"
    testnet: bool = True  # Safety default
    api_key: Optional[str] = Field(default=None, description="API key (from env: BYBIT_API_KEY)")
    api_secret: Optional[str] = Field(default=None, description="API secret (from env: BYBIT_API_SECRET)")
    rate_limit_per_second: float = 10.0  # ~100 requests per 10s

    def model_post_init(self, __context) -> None:
        # Override from environment variables
        if self.api_key is None:
            self.api_key = os.getenv("BYBIT_API_KEY")
        if self.api_secret is None:
            self.api_secret = os.getenv("BYBIT_API_SECRET")


class TradingPairsConfig(BaseModel):
    """Trading pairs configuration."""

    pairs: List[str] = Field(
        default=[
            "SUI/USDT:USDT",
            "SOL/USDT:USDT",
            "LINK/USDT:USDT",
            "BNB/USDT:USDT",
        ],
        description="ByBit linear perpetual symbols"
    )
    default_pair: str = "SOL/USDT:USDT"  # Start with highest liquidity

    @field_validator("pairs")
    @classmethod
    def validate_pairs(cls, v: List[str]) -> List[str]:
        for pair in v:
            if not pair.endswith(":USDT"):
                raise ValueError(f"Invalid perpetual symbol format: {pair}. Expected ':USDT' suffix for linear perps")
        return v


class TimeframeConfig(BaseModel):
    """Timeframe settings for multi-timeframe analysis."""

    ltf: str = "15m"  # Low timeframe for entries
    htf: str = "1h"   # High timeframe for trend

    # Standard exchange timeframes (available directly from exchange)
    EXCHANGE_NATIVE: ClassVar[List[str]] = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]

    @field_validator("ltf", "htf")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """
        Validate timeframe format.

        Supports:
        - Exchange-native timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
        - Custom minute-based timeframes: 9m, 7m, etc. (aggregated from 1m)
        - Custom hour-based timeframes: 3h, etc.
        """
        # Exchange-native timeframes
        native = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]
        if v in native:
            return v

        # Custom minute-based (e.g., '9m', '7m')
        if v.endswith("m"):
            try:
                minutes = int(v[:-1])
                if 1 <= minutes <= 60:
                    return v
            except ValueError:
                pass

        # Custom hour-based (e.g., '3h')
        if v.endswith("h"):
            try:
                hours = int(v[:-1])
                if 1 <= hours <= 24:
                    return v
            except ValueError:
                pass

        raise ValueError(
            f"Invalid timeframe: {v}. "
            f"Use standard (1m,5m,15m,1h,...) or custom (Nm for minutes, Nh for hours)"
        )

    @classmethod
    def is_native(cls, timeframe: str) -> bool:
        """Check if timeframe is directly available from exchange."""
        return timeframe in cls.EXCHANGE_NATIVE

    @classmethod
    def to_minutes(cls, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        if timeframe.endswith("m"):
            return int(timeframe[:-1])
        elif timeframe.endswith("h"):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith("d"):
            return int(timeframe[:-1]) * 1440
        else:
            raise ValueError(f"Cannot parse timeframe: {timeframe}")


class FeeConfig(BaseModel):
    """Fee model configuration."""

    use_dynamic_fees: bool = True
    default_taker_fee: float = 0.00055  # 0.055% ByBit taker
    default_maker_fee: float = 0.0002   # 0.02% ByBit maker
    fee_cache_ttl_seconds: int = 3600   # 1 hour cache

    @field_validator("default_taker_fee", "default_maker_fee")
    @classmethod
    def validate_fee(cls, v: float) -> float:
        if not 0 <= v <= 0.01:  # Max 1%
            raise ValueError(f"Fee must be between 0 and 0.01, got {v}")
        return v


class RiskConfig(BaseModel):
    """Risk management settings - Conservative defaults for real trading."""

    # ═══════════════════════════════════════════════════════════════════════════
    # POSITION SIZING - How much capital per trade
    # ═══════════════════════════════════════════════════════════════════════════
    max_position_pct: float = 0.25      # 25% equity per trade (was 100% - too aggressive)
    max_leverage: int = 3               # 3x leverage (conservative)
    apply_leverage: bool = True         # Whether to apply leverage to PnL calculations

    # ═══════════════════════════════════════════════════════════════════════════
    # RISK LIMITS - Circuit breakers to prevent blowups
    # ═══════════════════════════════════════════════════════════════════════════
    daily_loss_limit_pct: float = 0.05  # 5% daily loss = stop trading (was 10%)
    max_concurrent_positions: int = 1   # Only 1 position at a time per pair
    cooldown_bars: int = 3              # Wait 3 bars after closing before new trade
    max_trades_per_day: int = 50        # Prevent overtrading

    # ═══════════════════════════════════════════════════════════════════════════
    # SL/TP CONSTRAINTS - Prevent noise exits and catastrophic losses
    # ═══════════════════════════════════════════════════════════════════════════
    # Default SL/TP (percentage-based)
    default_sl_pct: float = 0.02        # 2% stop loss
    default_tp_pct: float = 0.04        # 4% take profit (2:1 R:R)

    # Tight SL/TP for momentum trades
    tight_sl_pct: float = 0.01          # 1% tight stop
    tight_tp_pct: float = 0.015         # 1.5% tight TP

    # SL/TP bounds (enforced regardless of agent's choice)
    min_sl_pct: float = 0.005           # 0.5% minimum SL (prevent noise exits)
    max_sl_pct: float = 0.05            # 5% maximum SL (prevent catastrophic losses)
    min_tp_pct: float = 0.01            # 1% minimum TP
    max_tp_pct: float = 0.10            # 10% maximum TP

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIPPAGE MODEL - Account for market impact
    # ═══════════════════════════════════════════════════════════════════════════
    base_slippage_pct: float = 0.0001   # 0.01% base slippage
    size_slippage_factor: float = 0.0   # Additional slippage per % of equity (0 = disabled)

    @field_validator("max_leverage")
    @classmethod
    def validate_leverage(cls, v: int) -> int:
        if not 1 <= v <= 100:
            raise ValueError(f"Leverage must be between 1 and 100, got {v}")
        return v

    @field_validator("max_position_pct")
    @classmethod
    def validate_position_pct(cls, v: float) -> float:
        if not 0.01 <= v <= 1.0:
            raise ValueError(f"Position size must be between 1% and 100%, got {v}")
        return v


class IndicatorConfig(BaseModel):
    """SR Swing Strategy indicator settings."""

    # Pivot detection
    pivot_left_bars: int = 4
    pivot_right_bars: int = 3
    use_high_low: bool = True

    # Trend filter
    trend_method: str = "HMA"  # HMA, Kalman, Gaussian, EMA, SMA, ALMA
    trend_length: int = 14
    kalman_short_len: int = 18
    kalman_long_len: int = 23

    # Momentum
    breakout_lookback: int = 40
    breakout_vol_multiplier: float = 1.1
    velocity_threshold: float = 3.5
    velocity_bars: int = 3
    atr_length: int = 14

    # RSI/MFI
    rsi_length: int = 14
    mfi_length: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    mfi_momentum_threshold: float = 50.0  # MFI above this + rising = bullish momentum

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_length: int = 20
    bb_mult: float = 2.0

    # Zero Lag Score (oscillator filter)
    zl_length: int = 32
    zl_loop_start: int = 1
    zl_loop_end: int = 70
    zl_threshold_up: float = 0.5
    zl_threshold_down: float = -0.5


class TrainingConfig(BaseModel):
    """PPO training hyperparameters."""

    # Core PPO
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 512  # Larger batch for GPU utilization
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL ARCHITECTURE - Neural Network Sizing
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # MLP Policy Architecture Formula:
    # ─────────────────────────────────
    # For net_arch = [H1, H2] (two hidden layers):
    #
    #   Policy Network (Actor):
    #     Layer 1: obs_dim × H1 + H1 (bias)
    #     Layer 2: H1 × H2 + H2 (bias)
    #     Output:  H2 × action_dim + action_dim (bias)
    #
    #   Value Network (Critic):
    #     Layer 1: obs_dim × H1 + H1 (bias)
    #     Layer 2: H1 × H2 + H2 (bias)
    #     Output:  H2 × 1 + 1 (bias)
    #
    # Total Parameters ≈ 2 × (obs_dim × H1 + H1 × H2 + H2 × action_dim)
    #
    # Example with obs_dim=180, action_dim=3, net_arch=[256, 256]:
    #   ≈ 2 × (180×256 + 256×256 + 256×3) ≈ 225,000 params
    #
    # Example with obs_dim=180, action_dim=3, net_arch=[512, 512]:
    #   ≈ 2 × (180×512 + 512×512 + 512×3) ≈ 710,000 params
    #
    # Sizing Guidelines:
    # ─────────────────
    # - Small (100-300k params): Good for simple patterns, fast training
    # - Medium (300k-1M params): Balance of capacity and generalization
    # - Large (1M+ params): Risk of overfitting, needs more data
    #
    # For RL trading, smaller is often better to avoid overfitting.
    # ═══════════════════════════════════════════════════════════════════════════

    net_arch: List[int] = Field(
        default=[256, 256],
        description="Hidden layer sizes for policy/value networks"
    )
    n_envs: int = Field(
        default=32,
        description="Number of parallel environments (32 for RTX 5090 with 32GB VRAM)"
    )
    use_gpu: bool = Field(
        default=True,
        description="Use GPU if available"
    )
    torch_compile: Optional[str] = Field(
        default="reduce-overhead",
        description=(
            "torch.compile mode for kernel fusion + Triton optimization. "
            "Options: 'default', 'reduce-overhead', 'max-autotune', "
            "'max-autotune-no-cudagraphs', or null/false to disable"
        )
    )

    # Environment
    window_size: int = 30
    episode_max_steps: int = 2000

    # Checkpointing
    checkpoint_freq: int = 50_000
    eval_freq: int = 10_000

    # Reproducibility
    seed: int = 42


class BoxFeatureConfig(BaseModel):
    """
    Configuration for intraday box strategy features.

    The box strategy identifies session high/low ranges and trades breakouts
    or bounces from these dynamic support/resistance levels.

    Attributes:
        use_box_features: Enable box features in the observation space.
            When False (default), box features are not computed, maintaining
            backward compatibility with existing configs.
        box_warmup_bars: Number of bars required before box features are valid.
            The box needs sufficient price history to establish meaningful
            high/low levels. Minimum 1, maximum 500.
        box_touch_tolerance_pct: Tolerance for detecting price "at level" as a
            percentage of price. E.g., 0.001 = 0.1% tolerance means price within
            0.1% of box high/low is considered "touching" the level.
    """

    use_box_features: bool = False
    box_warmup_bars: int = 30
    box_touch_tolerance_pct: float = 0.001

    @field_validator("box_warmup_bars")
    @classmethod
    def validate_warmup_bars(cls, v: int) -> int:
        if not 1 <= v <= 500:
            raise ValueError(f"box_warmup_bars must be between 1 and 500, got {v}")
        return v

    @field_validator("box_touch_tolerance_pct")
    @classmethod
    def validate_touch_tolerance(cls, v: float) -> float:
        if not 0 < v <= 0.05:
            raise ValueError(
                f"box_touch_tolerance_pct must be > 0 and <= 0.05 (5%), got {v}"
            )
        return v


class DataConfig(BaseModel):
    """Data storage configuration."""

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAIN/EVAL SPLIT - Seed-based holdout for reproducible evaluation
    # ═══════════════════════════════════════════════════════════════════════════
    eval_holdout_pct: float = 0.20      # 20% of data reserved for evaluation
    eval_holdout_seed: int = 12345      # Fixed seed for consistent holdout selection
    use_temporal_split: bool = False    # If True: last N% is eval. If False: random shard

    # Parquet settings
    compression: str = "snappy"

    # Historical data
    min_bars: int = 10000  # Minimum bars required for training

    # ═══════════════════════════════════════════════════════════════════════════
    # SESSION BOUNDARIES - For intraday box strategy features
    # ═══════════════════════════════════════════════════════════════════════════
    session_timezone: str = "UTC"       # Timezone for session boundary detection

    def model_post_init(self, __context) -> None:
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.raw_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class CryptoConfig(BaseModel):
    """Root configuration for crypto RL trading system."""

    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    pairs: TradingPairsConfig = Field(default_factory=TradingPairsConfig)
    timeframes: TimeframeConfig = Field(default_factory=TimeframeConfig)
    fees: FeeConfig = Field(default_factory=FeeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    indicators: IndicatorConfig = Field(default_factory=IndicatorConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    box_features: BoxFeatureConfig = Field(default_factory=BoxFeatureConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CryptoConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data) if data else cls()

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> "CryptoConfig":
        """
        Load configuration with priority:
        1. Provided config_path
        2. CRYPTO_CONFIG env var
        3. configs/default.yaml
        4. Default values
        """
        if config_path:
            return cls.from_yaml(config_path)

        env_path = os.getenv("CRYPTO_CONFIG")
        if env_path and Path(env_path).exists():
            return cls.from_yaml(env_path)

        default_path = Path("configs/default.yaml")
        if default_path.exists():
            return cls.from_yaml(default_path)

        return cls()


# Convenience function
def load_config(path: Optional[str | Path] = None) -> CryptoConfig:
    """Load crypto trading configuration."""
    return CryptoConfig.load(path)


if __name__ == "__main__":
    # Print default config
    config = CryptoConfig()
    print("Default configuration:")
    print(yaml.dump(config.model_dump(), default_flow_style=False))
