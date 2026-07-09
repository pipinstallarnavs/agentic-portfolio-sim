"""Multi-agent orchestration.

``MultiAgentStrategy`` runs Research -> Quant -> Portfolio -> Risk in
sequence and implements the same ``Strategy`` protocol the deterministic
baselines do, so ``WalkForwardSimulator`` can drive it with zero changes.

Every agent's full structured output is kept in ``audit_log`` (one entry per
rebalance) — this is what Phase 8/9 evaluation (agent/system metrics,
ablations) will read from. Only a short rationale and confidence make it
into the ``PortfolioProposal`` the engine itself records, per the "concise
rationale, not chain-of-thought" rule in agents/base.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_portfolio.agents.base import Agent, AgentContext
from agentic_portfolio.agents.portfolio import MockPortfolioAgent, PortfolioAgentResult
from agentic_portfolio.agents.quant import QuantAgent, QuantResult
from agentic_portfolio.agents.research import MockResearchAgent, ResearchResult
from agentic_portfolio.agents.risk import RiskAgent, RiskAgentResult
from agentic_portfolio.config import PortfolioConstraintsConfig, RiskConfig
from agentic_portfolio.simulation.state import MarketState, PortfolioProposal, PortfolioSnapshot


@dataclass
class RebalanceAuditEntry:
    as_of: str
    research: ResearchResult
    quant: QuantResult
    portfolio: PortfolioAgentResult
    risk: RiskAgentResult


class MultiAgentStrategy:
    """The full pipeline: Research + Quant + Portfolio + Risk agents.

    Agents are injected rather than hardcoded so ablations (no Risk Agent,
    no Research Agent, etc. — Phase 9) can be built by swapping in
    no-op/passthrough agent implementations without touching this class.
    """

    name = "multi_agent_mock"

    def __init__(
        self,
        constraints: PortfolioConstraintsConfig,
        risk_config: RiskConfig,
        research_agent: Agent | None = None,
        quant_agent: Agent | None = None,
        portfolio_agent: Agent | None = None,
        risk_agent: Agent | None = None,
        top_n: int = 5,
    ):
        self.constraints = constraints
        self.risk_config = risk_config
        self.research_agent = research_agent or MockResearchAgent()
        self.quant_agent = quant_agent or QuantAgent()
        self.portfolio_agent = portfolio_agent or MockPortfolioAgent(constraints, top_n=top_n)
        self.risk_agent = risk_agent or RiskAgent()
        self.audit_log: list[RebalanceAuditEntry] = []

    def generate_weights(self, state: MarketState, portfolio_state: PortfolioSnapshot) -> PortfolioProposal:
        context = AgentContext(
            portfolio_state=portfolio_state,
            constraints=self.constraints,
            risk_limits=self.risk_config,
        )

        research_result: ResearchResult = self.research_agent.run(state, context)
        quant_result: QuantResult = self.quant_agent.run(state, context)
        portfolio_result: PortfolioAgentResult = self.portfolio_agent.run(
            state, context, research=research_result, quant=quant_result
        )
        risk_result: RiskAgentResult = self.risk_agent.run(state, context, proposal=portfolio_result)

        self.audit_log.append(
            RebalanceAuditEntry(
                as_of=state.as_of.date().isoformat(),
                research=research_result,
                quant=quant_result,
                portfolio=portfolio_result,
                risk=risk_result,
            )
        )

        reasoning = f"{portfolio_result.rationale} | Risk: {risk_result.rationale}"
        return PortfolioProposal(
            target_weights=risk_result.final_weights,
            cash_weight=risk_result.final_cash_weight,
            reasoning_summary=reasoning[:1000],
            confidence=min(portfolio_result.confidence, risk_result.confidence),
        )
