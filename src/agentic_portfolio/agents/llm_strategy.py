"""Readable LLM research -> proposal -> deterministic risk -> one revision.

Single-agent mode skips the research model call, but sees the same numeric
features, holdings and limits and uses the same final risk gate. Neither
mode lets the model calculate the financial metrics or execute trades.
"""
import json
import math
import time
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field
from agentic_portfolio.agents.base import AgentContext
from agentic_portfolio.agents.portfolio import PortfolioAgentResult
from agentic_portfolio.agents.quant import QuantAgent
from agentic_portfolio.agents.risk import RiskAgent
from agentic_portfolio.simulation.state import PortfolioProposal
from agentic_portfolio.tools.portfolio import validate_weights

FiniteWeight = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class ResearchNotes(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    summary: str = Field(max_length=2000)
    risks: list[str] = Field(max_length=10)


class WeightDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    target_weights: dict[str, FiniteWeight]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    rationale: str = Field(max_length=2000)


def json_safe(value):
    """Missing features become null, rather than nonstandard JSON NaN."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class StructuredCalls:
    """One explicit provider request per call; malformed output fails closed.

    Successful calls retain prompts, raw responses, model and token counts.
    These are new run records, not reconstructed evidence of past experiments.
    """
    def __init__(self, client, max_calls=12):
        self.client, self.max_calls = client, max_calls
        self.attempts = 0
        self.records = []

    def ask(self, schema, payload):
        if self.attempts >= self.max_calls:
            raise RuntimeError('Explicit model-call budget exhausted')
        self.attempts += 1
        system = ('Use only the dated inputs provided; do not use remembered market events. '
                  'Interpret supplied features; do not invent news or market observations. '
                  'Return a JSON object only, with this schema: '+json.dumps(schema.model_json_schema()))
        prompt = json.dumps(json_safe(payload), allow_nan=False)
        started = time.monotonic()
        response = self.client.complete(system, prompt)
        record = dict(system=system, prompt=prompt, response=response.model_dump(),
                      latency_seconds=time.monotonic()-started, schema=schema.__name__)
        self.records.append(record)
        try:
            return schema.model_validate_json(response.content)
        except ValueError:
            record['invalid_response'] = True
            raise ValueError('Model response failed JSON/schema validation; no portfolio accepted') from None


class LLMPortfolioStrategy:
    def __init__(self, calls, constraints, risk_config, multi_agent=True, max_revisions=1):
        if max_revisions not in (0, 1):
            raise ValueError('The teaching implementation allows zero or one revision')
        if not constraints.long_only:
            raise ValueError('The current execution model is long-only')
        self.calls, self.constraints, self.risk_config = calls, constraints, risk_config
        self.multi_agent, self.max_revisions = multi_agent, max_revisions
        self.name = 'multi_agent_llm' if multi_agent else 'single_agent_llm'
        self.audit_log = []

    def generate_weights(self, state, portfolio_state):
        context = AgentContext(portfolio_state=portfolio_state,
                               constraints=self.constraints, risk_limits=self.risk_config)
        quant = QuantAgent().run(state, context)
        payload = dict(as_of=state.as_of.date().isoformat(), universe=state.universe,
                       features=quant.model_dump(mode='json'),
                       holdings=portfolio_state.model_dump(mode='json'),
                       constraints=self.constraints.model_dump(), risk_limits=self.risk_config.model_dump())
        # Both arms start with identical deterministic information. The multi
        # arm additionally lets one model summarize that information first.
        research = None
        if self.multi_agent:
            research = self.calls.ask(ResearchNotes, {**payload, 'task': 'Summarize price-derived context and risks.'})
            payload['research'] = research.model_dump()
        rounds = []
        for revision in range(self.max_revisions+1):
            decision = self.calls.ask(WeightDecision, {**payload, 'task': 'Propose long-only weights; unallocated capital stays in cash.'})
            weights = decision.target_weights
            if set(weights)-set(state.universe) or sum(weights.values()) > 1+1e-9:
                raise ValueError('Model proposed an unknown ticker or borrowed capital')
            proposal = PortfolioAgentResult(agent_name=self.name, target_weights=weights,
                                            cash_weight=1-sum(weights.values()),
                                            confidence=decision.confidence, rationale=decision.rationale)
            # Enforce basic budget/position limits before the risk layer. The
            # risk layer can then reduce turnover/vol/VaR, or hold prior weights.
            validation = validate_weights(weights, max_position_weight=self.constraints.max_position_weight,
                                          max_gross_exposure=self.constraints.max_gross_exposure)
            if not validation.is_valid or proposal.cash_weight < self.constraints.min_cash_weight-1e-9:
                review = dict(decision='invalid', modifications=validation.violations + ['Check cash/position budget'],
                              final_weights=None)
                risk = None
            else:
                risk = RiskAgent().run(state, context, proposal=proposal)
                review = risk.model_dump(mode='json')
            rounds.append(dict(proposal=decision.model_dump(), risk=review))
            if risk is not None and risk.decision == 'approve':
                break
            if revision < self.max_revisions:
                # This is actual feedback: the next call sees both the original
                # proposal and deterministic objections, not just a new prompt.
                payload['previous_proposal'] = decision.model_dump()
                payload['risk_feedback'] = review
        self.audit_log.append(json_safe(dict(as_of=payload['as_of'],
                                             research=research.model_dump() if research else None,
                                             rounds=rounds)))
        if risk is None:
            raise ValueError('No valid portfolio after the bounded revision loop')
        final = validate_weights(risk.final_weights, max_position_weight=self.constraints.max_position_weight,
                                 max_gross_exposure=self.constraints.max_gross_exposure)
        if not final.is_valid or risk.final_cash_weight < self.constraints.min_cash_weight-1e-9:
            raise ValueError('Final risk-reviewed portfolio violates hard limits')
        return PortfolioProposal(target_weights=risk.final_weights, cash_weight=risk.final_cash_weight,
                                 confidence=min(proposal.confidence, risk.confidence),
                                 reasoning_summary=proposal.rationale+' | '+risk.rationale)
