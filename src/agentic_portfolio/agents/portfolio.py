"""Portfolio Agent.

Combines the Research Agent's context tags with the Quant Agent's signal
scores into target weights, respecting the same hard constraints every
strategy in this system respects (long-only, per-position cap, gross
exposure cap). The research tilt is a small, fixed, documented adjustment —
not a black box — so its effect is easy to reason about and to ablate later
(does removing the Research Agent change anything?).
"""

from __future__ import annotations

from agentic_portfolio.agents.base import Agent, AgentContext, AgentResult
from agentic_portfolio.agents.quant import QuantResult
from agentic_portfolio.agents.research import ResearchResult
from agentic_portfolio.config import PortfolioConstraintsConfig
from agentic_portfolio.simulation.state import MarketState
from agentic_portfolio.tools.portfolio import cap_and_renormalize_weights

RESEARCH_TILT = 0.05  # +/-5% multiplicative adjustment to a name's signal score per supporting/opposing tag


class PortfolioAgentResult(AgentResult):
    target_weights: dict[str, float] = {}
    cash_weight: float = 0.0


class MockPortfolioAgent(Agent):
    name = "portfolio_agent"

    def __init__(self, constraints: PortfolioConstraintsConfig, top_n: int = 5, cash_weight: float = 0.0):
        self.constraints = constraints
        self.top_n = top_n
        self.cash_weight = cash_weight

    def run(
        self,
        state: MarketState,
        context: AgentContext,
        research: ResearchResult,
        quant: QuantResult,
        **kwargs,
    ) -> PortfolioAgentResult:
        adjusted_scores: dict[str, float] = {}
        for ticker, score in quant.signal_scores.items():
            tags = set(research.context_tags.get(ticker, []))
            tilt = 0.0
            if "outperforming_benchmark" in tags:
                tilt += RESEARCH_TILT
            if "underperforming_benchmark" in tags:
                tilt -= RESEARCH_TILT
            adjusted_scores[ticker] = score * (1.0 + tilt)

        if not adjusted_scores:
            return PortfolioAgentResult(
                agent_name=self.name,
                rationale="No names had valid quant signals; holding 100% cash.",
                confidence=0.0,
                target_weights={},
                cash_weight=1.0,
            )

        ranked = sorted(adjusted_scores.items(), key=lambda kv: kv[1], reverse=True)
        candidates = [t for t, _ in ranked[: self.top_n]]

        vols = {
            t: quant.features[t]["volatility_60d"]
            for t in candidates
            if quant.features.get(t, {}).get("volatility_60d", 0) > 0
        }
        candidates = [t for t in candidates if t in vols]
        if not candidates:
            return PortfolioAgentResult(
                agent_name=self.name,
                rationale="No candidates had valid volatility data; holding 100% cash.",
                confidence=0.0,
                target_weights={},
                cash_weight=1.0,
            )

        inv_vol = {t: 1.0 / vols[t] for t in candidates}
        total_inv_vol = sum(inv_vol.values())
        raw_weights = {t: inv_vol[t] / total_inv_vol for t in candidates}

        target_gross = 1.0 - self.cash_weight
        weights = cap_and_renormalize_weights(raw_weights, self.constraints.max_position_weight, target_gross)
        weights = {t: w for t, w in weights.items() if w > 1e-9}
        cash_weight = 1.0 - sum(weights.values())

        confidence = min(1.0, len(candidates) / self.top_n)
        rationale = (
            f"Selected {len(candidates)} of {len(quant.signal_scores)} scored names "
            f"(momentum - 0.5*vol, research-tilted), inverse-vol weighted."
        )

        return PortfolioAgentResult(
            agent_name=self.name,
            rationale=rationale,
            confidence=confidence,
            target_weights=weights,
            cash_weight=cash_weight,
        )
