"""Walk-forward simulation engine.

This is the one loop every strategy — deterministic baseline today,
multi-agent system in later phases — runs through. It is written against
the ``Strategy`` protocol (one method: market state in, PortfolioProposal
out) specifically so that plugging in an LLM-driven Portfolio Agent later
requires zero changes here.

Timing convention (also documented in README):
  1. On decision date t, a strategy sees only data with index <= t and
     proposes target weights.
  2. Those weights are executed using prices at t+1 (the next trading day).
  3. The portfolio's value on t+1 already reflects the new holdings, so
     returns from t+1 onward are "in" the new positions.
This is enforced by the loop structure below: proposals are computed on
day t and only ever applied when the loop reaches day t+1.
"""

from __future__ import annotations

import logging
from typing import Protocol

import pandas as pd

from agentic_portfolio.config import Config
from agentic_portfolio.data.provider import DataProvider
from agentic_portfolio.simulation.execution import execute_rebalance
from agentic_portfolio.simulation.state import (
    MarketState,
    PortfolioProposal,
    PortfolioSnapshot,
    RebalanceRecord,
    SimulationResult,
    build_market_state,
)
from agentic_portfolio.tools.portfolio import validate_weights

logger = logging.getLogger(__name__)


class Strategy(Protocol):
    """Anything that can propose target portfolio weights from a MarketState
    plus the portfolio's current holdings.

    Deterministic baselines (Phase 2) and LLM-backed multi-agent
    orchestration (Phase 3+) both implement this same interface, so the
    engine below never needs to know which kind of strategy it's driving.
    ``portfolio_state`` is included (not just market data) because a
    Portfolio Agent needs current holdings to reason about turnover and
    incremental trades, not just to propose weights from scratch each time.
    """

    name: str

    def generate_weights(self, state: MarketState, portfolio_state: PortfolioSnapshot) -> PortfolioProposal: ...


def get_rebalance_dates(trading_days: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    """Pick one trading day per period (e.g. the last trading day of each
    week for freq="W-FRI"), so holidays never produce a missing rebalance.
    """
    marker = pd.Series(trading_days, index=trading_days)
    picked = marker.resample(freq).last().dropna()
    return [pd.Timestamp(d) for d in picked if d in trading_days]


class WalkForwardSimulator:
    def __init__(
        self,
        config: Config,
        data_provider: DataProvider,
        strategy: Strategy,
    ):
        self.config = config
        self.provider = data_provider
        self.strategy = strategy

    def _load_panel(self) -> pd.DataFrame:
        tickers = list(self.config.universe) + [self.config.benchmark]
        panel = self.provider.get_price_panel(
            tickers, self.config.data.start_date, self.config.data.end_date
        )
        panel = panel.dropna(how="any")
        if panel.empty:
            raise ValueError("Price panel is empty after alignment — check data availability")
        return panel

    def run(self) -> SimulationResult:
        cfg = self.config
        universe = list(cfg.universe)
        panel = self._load_panel()
        trading_days = panel.index

        rebalance_dates = get_rebalance_dates(trading_days, cfg.simulation.rebalance_freq)
        min_pos = cfg.simulation.min_history_days
        rebalance_dates = [
            d for d in rebalance_dates
            if trading_days.get_loc(d) >= min_pos and trading_days.get_loc(d) < len(trading_days) - 1
        ]
        decision_dates = set(rebalance_dates)

        cash = float(cfg.simulation.initial_capital)
        shares: dict[str, float] = {t: 0.0 for t in universe}
        equity_curve: dict[str, float] = {}
        rebalances: list[RebalanceRecord] = []
        pending: tuple[pd.Timestamp, PortfolioProposal, list[str]] | None = None

        for day in trading_days:
            if pending is not None:
                decision_date, proposal, violations = pending
                execution_prices = {
                    t: float(panel.loc[day, t]) for t in universe if pd.notna(panel.loc[day, t])
                }
                portfolio_value_before = cash + sum(
                    shares.get(t, 0.0) * p for t, p in execution_prices.items()
                )
                result = execute_rebalance(
                    current_shares=shares,
                    current_cash=cash,
                    target_weights=proposal.target_weights,
                    execution_prices=execution_prices,
                    cost_bps=cfg.simulation.transaction_cost_bps,
                )
                shares = result.new_shares
                cash = result.cash_after
                rebalances.append(
                    RebalanceRecord(
                        decision_date=decision_date.date(),
                        execution_date=day.date(),
                        proposed_weights=proposal.target_weights,
                        validation_violations=violations,
                        target_weights=proposal.target_weights,
                        realized_weights=result.realized_weights,
                        trades=result.trades,
                        turnover=result.turnover,
                        transaction_cost=result.transaction_cost,
                        portfolio_value_before=portfolio_value_before,
                        portfolio_value_after=result.portfolio_value_after,
                        reasoning_summary=proposal.reasoning_summary,
                        confidence=proposal.confidence,
                    )
                )
                pending = None

            if day in decision_dates:
                market_state = build_market_state(panel, day, universe, cfg.benchmark)

                decision_day_prices = {
                    t: float(panel.loc[day, t]) for t in universe if pd.notna(panel.loc[day, t])
                }
                current_value = cash + sum(
                    shares.get(t, 0.0) * p for t, p in decision_day_prices.items()
                )
                current_weights = {
                    t: shares.get(t, 0.0) * p / current_value
                    for t, p in decision_day_prices.items()
                    if shares.get(t, 0.0) != 0.0 and current_value > 0
                }
                portfolio_state = PortfolioSnapshot(
                    date=day.date(),
                    cash=cash,
                    shares=dict(shares),
                    weights=current_weights,
                    total_value=current_value,
                )

                proposal = self.strategy.generate_weights(market_state, portfolio_state)
                validation = validate_weights(
                    proposal.target_weights,
                    long_only=cfg.portfolio_constraints.long_only,
                    max_position_weight=cfg.portfolio_constraints.max_position_weight,
                    max_gross_exposure=cfg.portfolio_constraints.max_gross_exposure,
                )
                if not validation.is_valid:
                    raise ValueError(f"{self.strategy.name}: invalid proposal at {day.date()}: {validation.violations}")
                if set(proposal.target_weights) - set(universe):
                    raise ValueError("Proposal contains a ticker outside the tradable universe")
                pending = (day, proposal, validation.violations)

            day_prices = panel.loc[day, universe]
            value = cash + sum(
                shares.get(t, 0.0) * p for t, p in day_prices.items() if pd.notna(p)
            )
            equity_curve[day.date().isoformat()] = value

        return SimulationResult(
            strategy_name=self.strategy.name,
            equity_curve=equity_curve,
            rebalances=rebalances,
            config_snapshot=cfg.model_dump(),
        )
