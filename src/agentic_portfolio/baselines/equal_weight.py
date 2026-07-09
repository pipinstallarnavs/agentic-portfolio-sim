"""Simplest possible benchmark: equal weight across the full universe,
rebalanced back to equal weight on every rebalance date. No signal, no
ranking — a floor any "smarter" strategy should beat to justify itself.
"""

from __future__ import annotations

from agentic_portfolio.simulation.state import MarketState, PortfolioProposal, PortfolioSnapshot


class EqualWeightBaseline:
    name = "equal_weight_baseline"

    def __init__(self, cash_weight: float = 0.0):
        self.cash_weight = cash_weight

    def generate_weights(self, state: MarketState, portfolio_state: PortfolioSnapshot) -> PortfolioProposal:
        n = len(state.universe)
        gross = 1.0 - self.cash_weight
        weight = gross / n
        weights = {t: weight for t in state.universe}
        return PortfolioProposal(
            target_weights=weights,
            cash_weight=self.cash_weight,
            reasoning_summary=f"Equal weight across all {n} universe names.",
            confidence=1.0,
        )
