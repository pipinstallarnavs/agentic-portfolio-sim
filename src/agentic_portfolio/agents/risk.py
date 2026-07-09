"""Risk Agent.

Reviews the Portfolio Agent's proposal using the deterministic functions in
tools/risk.py and tools/portfolio.py, and can approve, modify, or reject it.
Every modification is logged with an explicit, human-readable reason — this
is the audit trail the whole "is the Risk Agent worth it" evaluation
question (Phase 8/9) depends on.

Checks, in order:
  1. Confidence floor — reject (hold prior portfolio) if the Portfolio
     Agent's own confidence is too low to act on.
  2. Hard constraints (long-only, per-position cap, gross exposure) — fixed
     by capping and renormalizing, same routine the momentum baseline uses.
  3. Turnover limit — scaled back toward the current portfolio if the
     proposed trade is larger than the configured limit.
  4. Ex-ante portfolio volatility — proposal is scaled down (extra to cash)
     if trailing-covariance-implied annualized volatility exceeds the limit.
  5. Historical VaR — same scaling approach, using the trailing hypothetical
     return series the proposed weights would have produced.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from agentic_portfolio.agents.base import Agent, AgentContext, AgentResult
from agentic_portfolio.agents.portfolio import PortfolioAgentResult
from agentic_portfolio.simulation.state import MarketState
from agentic_portfolio.tools.features import calculate_beta
from agentic_portfolio.tools.portfolio import (
    calculate_portfolio_volatility,
    calculate_turnover,
    cap_and_renormalize_weights,
    validate_weights,
)
from agentic_portfolio.tools.risk import calculate_expected_shortfall, calculate_historical_var


class RiskAgentResult(AgentResult):
    decision: Literal["approve", "modify", "reject"] = "approve"
    final_weights: dict[str, float] = {}
    final_cash_weight: float = 0.0
    risk_metrics: dict[str, float] = {}
    modifications: list[str] = []


class RiskAgent(Agent):
    name = "risk_agent"

    def run(
        self,
        state: MarketState,
        context: AgentContext,
        proposal: PortfolioAgentResult,
        **kwargs,
    ) -> RiskAgentResult:
        constraints = context.constraints
        risk_limits = context.risk_limits
        prior_weights = context.portfolio_state.weights

        if proposal.confidence < risk_limits.min_proposal_confidence:
            cash = 1.0 - sum(prior_weights.values())
            return RiskAgentResult(
                agent_name=self.name,
                rationale=(
                    f"Rejected proposal: confidence {proposal.confidence:.2f} below minimum "
                    f"{risk_limits.min_proposal_confidence:.2f}. Holding prior portfolio."
                ),
                confidence=proposal.confidence,
                decision="reject",
                final_weights=dict(prior_weights),
                final_cash_weight=cash,
                risk_metrics={},
                modifications=["Rejected: confidence below minimum, prior portfolio held unchanged."],
            )

        weights = dict(proposal.target_weights)
        modifications: list[str] = []

        validation = validate_weights(
            weights,
            long_only=constraints.long_only,
            max_position_weight=constraints.max_position_weight,
            max_gross_exposure=constraints.max_gross_exposure,
        )
        if not validation.is_valid:
            target_gross = min(1.0 - proposal.cash_weight, constraints.max_gross_exposure,
                               1.0 - constraints.min_cash_weight)
            weights = {t: max(w, 0.0) for t, w in weights.items()}
            weights = cap_and_renormalize_weights(weights, constraints.max_position_weight, target_gross)
            modifications.append(f"Capped positions to respect constraints: {'; '.join(validation.violations)}")

        # Include the cash leg explicitly, same reasoning as execution.py:
        # a shift into/out of cash is real turnover even though no single
        # equity weight nets against another.
        prior_with_cash = {**prior_weights, "_CASH_": 1.0 - sum(prior_weights.values())}
        proposed_with_cash = {**weights, "_CASH_": 1.0 - sum(weights.values())}
        turnover = calculate_turnover(prior_with_cash, proposed_with_cash)
        if turnover > constraints.max_turnover_per_rebalance + 1e-4 and constraints.max_turnover_per_rebalance > 0:
            scale = constraints.max_turnover_per_rebalance / turnover
            all_tickers = set(weights) | set(prior_weights)
            weights = {
                t: prior_weights.get(t, 0.0) + scale * (weights.get(t, 0.0) - prior_weights.get(t, 0.0))
                for t in all_tickers
            }
            weights = {t: w for t, w in weights.items() if abs(w) > 1e-9}
            modifications.append(
                f"Scaled trade to {scale:.0%} of proposed size to respect turnover limit "
                f"({turnover:.1%} proposed > {constraints.max_turnover_per_rebalance:.1%} limit)."
            )

        returns_panel = state.returns_panel()
        tradable = [t for t in weights if t in returns_panel.columns]
        window = min(risk_limits.var_lookback_days, len(returns_panel))
        trailing_returns = returns_panel[tradable].iloc[-window:] if tradable and window > 1 else pd.DataFrame()

        portfolio_vol = float("nan")
        var_95 = float("nan")
        es_95 = float("nan")
        beta_to_benchmark = float("nan")

        if not trailing_returns.empty:
            cov = trailing_returns.cov()
            portfolio_vol = calculate_portfolio_volatility(weights, cov)
            if portfolio_vol > risk_limits.max_portfolio_volatility + 1e-4 and risk_limits.max_portfolio_volatility > 0:
                scale = risk_limits.max_portfolio_volatility / portfolio_vol
                weights = {t: w * scale for t, w in weights.items()}
                modifications.append(
                    f"Scaled exposure to {scale:.0%} to respect max portfolio volatility "
                    f"({portfolio_vol:.1%} proposed > {risk_limits.max_portfolio_volatility:.1%} limit)."
                )

            weight_series = pd.Series({t: weights.get(t, 0.0) for t in tradable})
            hypothetical_returns = trailing_returns[tradable].mul(weight_series, axis=1).sum(axis=1)
            var_95 = calculate_historical_var(hypothetical_returns, risk_limits.var_confidence)
            if var_95 > risk_limits.max_historical_var_95 + 1e-4 and risk_limits.max_historical_var_95 > 0:
                scale = risk_limits.max_historical_var_95 / var_95
                weights = {t: w * scale for t, w in weights.items()}
                modifications.append(
                    f"Scaled exposure to {scale:.0%} to respect historical VaR limit "
                    f"({var_95:.1%} proposed > {risk_limits.max_historical_var_95:.1%} limit)."
                )
                weight_series = pd.Series({t: weights.get(t, 0.0) for t in tradable})
                hypothetical_returns = trailing_returns[tradable].mul(weight_series, axis=1).sum(axis=1)

            es_95 = calculate_expected_shortfall(hypothetical_returns, risk_limits.es_confidence)

            benchmark_returns = state.returns_panel()[state.benchmark] if state.benchmark in returns_panel.columns else None
            if benchmark_returns is not None:
                betas = {
                    t: calculate_beta(trailing_returns[t], benchmark_returns, window=min(60, len(trailing_returns)))
                    for t in tradable
                }
                beta_to_benchmark = sum(
                    weights.get(t, 0.0) * b for t, b in betas.items() if b == b  # filters NaN
                )

        # Report risk of the FINAL portfolio. Previously the log retained
        # volatility/VaR from before scaling, making a successful reduction
        # look like a continuing breach to anyone reading the audit trail.
        if not trailing_returns.empty:
            portfolio_vol = calculate_portfolio_volatility(weights, cov)
            var_95 = calculate_historical_var(hypothetical_returns, risk_limits.var_confidence)
        final_with_cash = {**weights, "_CASH_": 1.0 - sum(weights.values())}
        turnover = calculate_turnover(prior_with_cash, final_with_cash)
        if turnover > constraints.max_turnover_per_rebalance + 1e-4:
            modifications.append("Risk reduction overrides turnover limit; final turnover reported explicitly.")
        max_position = max((abs(w) for w in weights.values()), default=0.0)
        cash_weight = 1.0 - sum(weights.values())
        decision = "modify" if modifications else "approve"
        rationale = " | ".join(modifications) if modifications else "Proposal approved without modification."

        return RiskAgentResult(
            agent_name=self.name,
            rationale=rationale,
            confidence=proposal.confidence,
            decision=decision,
            final_weights=weights,
            final_cash_weight=cash_weight,
            risk_metrics={
                "portfolio_volatility": portfolio_vol,
                "historical_var_95": var_95,
                "expected_shortfall_95": es_95,
                "max_position": max_position,
                "turnover": turnover,
                "beta_to_benchmark": beta_to_benchmark,
            },
            modifications=modifications,
        )
