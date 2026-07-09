"""Quant Agent.

Per the spec, the Quant Agent never does arithmetic itself — it calls the
deterministic functions in tools/features.py and packages their output into
a structured result. There is no LLM judgment involved in this role even in
the full system: computing momentum/volatility/beta is not a reasoning task.
(A Phase 4 LLM layer could sit *on top* of this to decide which features to
weight more heavily, but the feature computation itself stays exactly here.)
"""

from __future__ import annotations

import math

from agentic_portfolio.agents.base import Agent, AgentContext, AgentResult
from agentic_portfolio.simulation.state import MarketState
from agentic_portfolio.tools.features import (
    calculate_beta,
    calculate_correlation_matrix,
    calculate_momentum,
    calculate_moving_average,
    calculate_realized_volatility,
    calculate_returns,
)

MOMENTUM_WINDOWS = (5, 20, 60)
VOLATILITY_WINDOWS = (20, 60)
MOVING_AVERAGE_WINDOWS = (20, 60)
BETA_WINDOW = 60
CORRELATION_WINDOW = 60


class QuantResult(AgentResult):
    features: dict[str, dict[str, float]] = {}
    signal_scores: dict[str, float] = {}
    correlation_matrix: dict[str, dict[str, float]] = {}


class QuantAgent(Agent):
    name = "quant_agent"

    def run(self, state: MarketState, context: AgentContext, **kwargs) -> QuantResult:
        benchmark_returns = calculate_returns(state.prices(state.benchmark))

        features: dict[str, dict[str, float]] = {}
        for ticker in state.universe:
            prices = state.prices(ticker)
            returns = calculate_returns(prices)
            ticker_features: dict[str, float] = {}

            for w in MOMENTUM_WINDOWS:
                ticker_features[f"momentum_{w}d"] = calculate_momentum(prices, w)
            for w in VOLATILITY_WINDOWS:
                ticker_features[f"volatility_{w}d"] = calculate_realized_volatility(returns, w)
            for w in MOVING_AVERAGE_WINDOWS:
                ticker_features[f"moving_average_{w}d"] = calculate_moving_average(prices, w)
            ticker_features[f"beta_{BETA_WINDOW}d"] = calculate_beta(returns, benchmark_returns, BETA_WINDOW)

            features[ticker] = ticker_features

        returns_panel = state.returns_panel()[list(state.universe)]
        corr = calculate_correlation_matrix(returns_panel, CORRELATION_WINDOW)
        correlation_matrix = {t: {u: float(corr.loc[t, u]) for u in corr.columns} for t in corr.index} if not corr.empty else {}

        signal_scores: dict[str, float] = {}
        for ticker, f in features.items():
            mom = f.get("momentum_60d", float("nan"))
            vol = f.get("volatility_60d", float("nan"))
            if math.isnan(mom) or math.isnan(vol) or vol <= 0:
                continue
            # Simple, explainable composite: reward trailing momentum, penalize volatility.
            signal_scores[ticker] = mom - 0.5 * vol

        return QuantResult(
            agent_name=self.name,
            rationale=f"Computed features for {len(state.universe)} names as of {state.as_of.date()}.",
            confidence=1.0,
            features=features,
            signal_scores=signal_scores,
            correlation_matrix=correlation_matrix,
        )
