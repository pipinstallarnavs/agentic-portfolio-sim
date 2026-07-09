"""Configuration loading.

Configuration is split in two:
  - Strategy/simulation parameters live in a YAML file (config/default.yaml)
    and are loaded into typed Pydantic models here.
  - Secrets and environment-specific values (API keys, provider selection)
    live in environment variables / .env and are read separately by the
    modules that need them (see llm/provider.py in later phases).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class DataConfig(BaseModel):
    start_date: str
    end_date: str
    provider: str = "yfinance"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


class SimulationConfig(BaseModel):
    rebalance_freq: str = "W-FRI"
    initial_capital: float = 1_000_000
    transaction_cost_bps: float = 10.0
    min_history_days: int = 65


class PortfolioConstraintsConfig(BaseModel):
    long_only: bool = True
    max_position_weight: float = 0.20
    max_gross_exposure: float = 1.00
    min_cash_weight: float = 0.00
    max_turnover_per_rebalance: float = 1.00


class MomentumBaselineConfig(BaseModel):
    lookback_days: int = 60
    top_n: int = 5
    vol_lookback_days: int = 20
    cash_weight: float = 0.0


class RiskConfig(BaseModel):
    var_confidence: float = 0.95
    es_confidence: float = 0.95
    var_lookback_days: int = 250
    max_portfolio_volatility: float = 0.30
    max_historical_var_95: float = 0.04
    min_proposal_confidence: float = 0.15


class Config(BaseModel):
    universe: list[str]
    benchmark: str
    data: DataConfig
    simulation: SimulationConfig
    portfolio_constraints: PortfolioConstraintsConfig = Field(default_factory=PortfolioConstraintsConfig)
    momentum_baseline: MomentumBaselineConfig = Field(default_factory=MomentumBaselineConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    random_seed: int = 42

    @property
    def raw_dir(self) -> Path:
        return REPO_ROOT / self.data.raw_dir

    @property
    def processed_dir(self) -> Path:
        return REPO_ROOT / self.data.processed_dir


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the YAML configuration file."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
