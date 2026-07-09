"""Common agent interface.

Every agent — mock/rule-based today, LLM-backed in Phase 4 — implements the
same shape: it receives the current ``MarketState`` (price data, already cut
off at simulation time t) and a shared ``AgentContext`` (current portfolio
holdings plus the constraint/risk limits every agent must respect), and
returns a structured, machine-readable ``AgentResult``. Agents that consume
another agent's output (Portfolio Agent needs Research + Quant; Risk Agent
needs the Portfolio Agent's proposal) take it as an explicit typed argument
rather than through a mutable shared blackboard — the data flow between
agents should be traceable by reading a function signature, not by chasing
what got written into a shared dict.

Chain-of-thought is deliberately not part of this contract: ``rationale`` is
a short, concise decision summary, never a full reasoning trace. Phase 4's
LLM-backed agents must summarize down to this same shape before returning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from agentic_portfolio.config import PortfolioConstraintsConfig, RiskConfig
from agentic_portfolio.simulation.state import MarketState, PortfolioSnapshot


class AgentResult(BaseModel):
    """Base shape every agent's output extends. Concrete agents add their
    own structured fields on top of this (see research.py, quant.py,
    portfolio.py, risk.py).
    """

    agent_name: str
    rationale: str = ""
    confidence: float = 1.0
    errors: list[str] = Field(default_factory=list)


class AgentContext(BaseModel):
    """Read-only context shared by every agent in one rebalance's pipeline:
    current portfolio holdings, and the constraint/risk limits the system
    must respect. This is the "shared simulation state" agents reason over —
    distinct from MarketState, which is price history, not portfolio state.
    """

    portfolio_state: PortfolioSnapshot
    constraints: PortfolioConstraintsConfig
    risk_limits: RiskConfig

    model_config = {"arbitrary_types_allowed": True}


class Agent(ABC):
    """Abstract base for all agents. Concrete agents may accept additional
    typed keyword arguments beyond (state, context) — e.g. the Portfolio
    Agent also takes `research` and `quant` results — since each agent's
    actual inputs differ by role; this base only fixes the common shape.
    """

    name: str

    @abstractmethod
    def run(self, state: MarketState, context: AgentContext, **kwargs) -> AgentResult: ...
