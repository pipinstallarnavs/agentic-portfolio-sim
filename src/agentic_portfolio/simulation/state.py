"""Simulation state: the market snapshot agents/strategies see, and the
structured records the engine produces.

The single most important function in this file is ``build_market_state``:
it is the one place responsible for making sure nothing after time ``t``
ever reaches a strategy or agent. Every other module trusts that whatever
DataFrame it's handed has already been cut off correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

import pandas as pd
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class MarketState:
    """Everything a strategy/agent may look at when deciding weights at `as_of`.

    price_panel contains ONLY rows with index <= as_of. There is no way to
    reach into future data through this object.
    """

    as_of: pd.Timestamp
    price_panel: pd.DataFrame  # adjusted close, columns = universe + [benchmark]
    universe: list[str]
    benchmark: str

    def prices(self, ticker: str) -> pd.Series:
        return self.price_panel[ticker].dropna()

    def returns_panel(self, method: str = "simple") -> pd.DataFrame:
        if method == "simple":
            return self.price_panel.pct_change().dropna(how="all")
        import numpy as np

        log_returns = np.log(self.price_panel / self.price_panel.shift(1))
        return log_returns.dropna(how="all")

    @property
    def latest_prices(self) -> dict[str, float]:
        last = self.price_panel.ffill().iloc[-1]
        return {t: float(last[t]) for t in self.price_panel.columns if pd.notna(last[t])}


def build_market_state(
    full_panel: pd.DataFrame,
    as_of: pd.Timestamp,
    universe: list[str],
    benchmark: str,
) -> MarketState:
    """Slice `full_panel` to rows with index <= as_of.

    This is the sole lookahead-prevention boundary in the codebase: every
    other consumer of price data downstream of this call only ever sees
    data that was already available at `as_of`.
    """
    as_of = pd.Timestamp(as_of)
    sliced = full_panel.loc[full_panel.index <= as_of]
    if sliced.empty:
        raise ValueError(f"No data available at or before {as_of}")
    assert sliced.index.max() <= as_of
    return MarketState(as_of=as_of, price_panel=sliced, universe=list(universe), benchmark=benchmark)


class PortfolioProposal(BaseModel):
    """Common output shape for anything that proposes target weights —
    a deterministic baseline today, a Portfolio Agent in later phases.
    This is the interface the simulation engine is written against, so
    swapping in an LLM-based strategy later requires no engine changes.
    """

    target_weights: dict[str, float]
    cash_weight: float = 0.0
    reasoning_summary: str = ""
    confidence: float = 1.0


class Trade(BaseModel):
    ticker: str
    shares_delta: float
    dollar_amount: float  # signed: positive = buy, negative = sell
    price: float


class PortfolioSnapshot(BaseModel):
    """Portfolio state at a point in time."""

    date: date_type
    cash: float
    shares: dict[str, float]
    weights: dict[str, float]
    total_value: float


class RebalanceRecord(BaseModel):
    """Full audit trail for one rebalance event."""

    decision_date: date_type
    execution_date: date_type
    proposed_weights: dict[str, float]
    validation_violations: list[str] = Field(default_factory=list)
    target_weights: dict[str, float]  # after any risk/validation adjustment
    realized_weights: dict[str, float]  # actual weights after execution + costs
    trades: list[Trade] = Field(default_factory=list)
    turnover: float
    transaction_cost: float
    portfolio_value_before: float
    portfolio_value_after: float
    reasoning_summary: str = ""
    confidence: float = 1.0


class SimulationResult(BaseModel):
    """Everything produced by one end-to-end simulation run."""

    strategy_name: str
    equity_curve: dict[str, float]  # ISO date string -> portfolio value
    rebalances: list[RebalanceRecord]
    config_snapshot: dict = Field(default_factory=dict)

    def equity_series(self) -> pd.Series:
        s = pd.Series(self.equity_curve)
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
