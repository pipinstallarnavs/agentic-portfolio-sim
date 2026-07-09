"""Deterministic quant baseline: rank by trailing momentum, take the top N,
inverse-volatility weight them, and respect the same portfolio constraints
every other strategy in this system must respect.

This baseline exists to answer one question the whole project is built
around: is a multi-agent LLM system actually worth its cost and complexity
relative to a few lines of ranking logic?
"""

from __future__ import annotations

import math

from agentic_portfolio.config import MomentumBaselineConfig, PortfolioConstraintsConfig
from agentic_portfolio.simulation.state import MarketState, PortfolioProposal, PortfolioSnapshot
from agentic_portfolio.tools.features import calculate_momentum, calculate_realized_volatility, calculate_returns
from agentic_portfolio.tools.portfolio import cap_and_renormalize_weights


class MomentumBaseline:
    name = "momentum_baseline"

    def __init__(self, config: MomentumBaselineConfig, constraints: PortfolioConstraintsConfig):
        self.cfg = config
        self.constraints = constraints

    def generate_weights(self, state: MarketState, portfolio_state: PortfolioSnapshot) -> PortfolioProposal:
        # portfolio_state is unused: this baseline always proposes fresh weights
        # from the ranking, with no turnover-awareness of current holdings.
        scores: dict[str, float] = {}
        for ticker in state.universe:
            mom = calculate_momentum(state.prices(ticker), self.cfg.lookback_days)
            if not math.isnan(mom):
                scores[ticker] = mom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        candidates = [t for t, _ in ranked[: self.cfg.top_n]]

        vols: dict[str, float] = {}
        for ticker in candidates:
            returns = calculate_returns(state.prices(ticker))
            vol = calculate_realized_volatility(returns, self.cfg.vol_lookback_days)
            if not math.isnan(vol) and vol > 0:
                vols[ticker] = vol
        candidates = [t for t in candidates if t in vols]

        if not candidates:
            return PortfolioProposal(
                target_weights={},
                cash_weight=1.0,
                reasoning_summary="Insufficient history to rank momentum/volatility; holding 100% cash.",
                confidence=0.0,
            )

        inv_vol = {t: 1.0 / vols[t] for t in candidates}
        total_inv_vol = sum(inv_vol.values())
        raw_weights = {t: inv_vol[t] / total_inv_vol for t in candidates}

        target_gross = 1.0 - self.cfg.cash_weight
        weights = cap_and_renormalize_weights(
            raw_weights, self.constraints.max_position_weight, target_gross
        )
        weights = {t: w for t, w in weights.items() if w > 1e-9}
        cash_weight = 1.0 - sum(weights.values())

        reasoning = (
            f"Top {len(candidates)} of {len(state.universe)} names by "
            f"{self.cfg.lookback_days}d momentum, inverse-{self.cfg.vol_lookback_days}d-vol weighted: "
            f"{', '.join(candidates)}."
        )
        return PortfolioProposal(
            target_weights=weights,
            cash_weight=cash_weight,
            reasoning_summary=reasoning,
            confidence=1.0,
        )
