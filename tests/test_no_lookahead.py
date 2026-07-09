"""The most important tests in this repo: nothing that happens at
simulation time t may depend on data from t+1 or later.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agentic_portfolio.config import (
    Config,
    DataConfig,
    MomentumBaselineConfig,
    PortfolioConstraintsConfig,
    RiskConfig,
    SimulationConfig,
)
from agentic_portfolio.data.provider import DataProvider
from agentic_portfolio.simulation.engine import WalkForwardSimulator
from agentic_portfolio.simulation.state import MarketState, PortfolioProposal, PortfolioSnapshot, build_market_state
from agentic_portfolio.tools.features import calculate_momentum


def make_panel(n_days: int = 120, shock_day: int | None = None) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    base = pd.Series(range(n_days), index=idx, dtype=float) + 100.0  # steadily rising
    if shock_day is not None:
        # A large future jump that must NOT be visible to any decision made before shock_day.
        base.iloc[shock_day:] += 1000.0
    return pd.DataFrame({"AAPL": base, "MSFT": base * 1.5, "SPY": base * 0.8})


def test_build_market_state_excludes_future_rows():
    panel = make_panel()
    cutoff = panel.index[50]
    state = build_market_state(panel, cutoff, ["AAPL", "MSFT"], "SPY")
    assert state.price_panel.index.max() <= cutoff
    assert cutoff in state.price_panel.index
    assert len(state.price_panel) == 51  # inclusive of cutoff, zero-indexed


def test_build_market_state_raises_if_no_data_before_cutoff():
    panel = make_panel()
    with pytest.raises(ValueError):
        build_market_state(panel, panel.index[0] - pd.Timedelta(days=10), ["AAPL"], "SPY")


def test_momentum_does_not_see_future_shock():
    shock_day = 70
    panel = make_panel(n_days=120, shock_day=shock_day)
    decision_date = panel.index[65]  # strictly before the shock

    state = build_market_state(panel, decision_date, ["AAPL"], "SPY")
    momentum_seen = calculate_momentum(state.prices("AAPL"), lookback=10)

    # If the shock had leaked in, momentum would be enormous (shock is +1000 vs base ~165-175).
    assert momentum_seen < 1.0  # a sane momentum value, not a shock-inflated one

    # Sanity check: the *unsliced* series (simulating a lookahead bug) would see it.
    leaked_momentum = calculate_momentum(panel["AAPL"].loc[:panel.index[75]], lookback=10)
    assert leaked_momentum > momentum_seen + 1.0


class RecordingStrategy:
    """A trivial strategy that records the MarketState it's given at every
    decision date, so tests can assert on exactly what the engine exposed.
    """

    name = "recording_strategy"

    def __init__(self, universe: list[str]):
        self.universe = universe
        self.seen_states: list[MarketState] = []

    def generate_weights(self, state: MarketState, portfolio_state: PortfolioSnapshot) -> PortfolioProposal:
        self.seen_states.append(state)
        weights = {self.universe[0]: 1.0}
        return PortfolioProposal(target_weights=weights, cash_weight=0.0, reasoning_summary="test", confidence=1.0)


class InMemoryProvider(DataProvider):
    def __init__(self, panel: pd.DataFrame):
        self.panel = panel

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_price_panel(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        return self.panel[tickers].loc[start:end]


def _test_config() -> Config:
    return Config(
        universe=["AAPL", "MSFT"],
        benchmark="SPY",
        data=DataConfig(start_date="2024-01-02", end_date="2024-12-31", provider="local"),
        simulation=SimulationConfig(rebalance_freq="W-FRI", initial_capital=1_000_000, transaction_cost_bps=0, min_history_days=10),
        portfolio_constraints=PortfolioConstraintsConfig(max_position_weight=1.0),
        momentum_baseline=MomentumBaselineConfig(),
        risk=RiskConfig(),
    )


def test_engine_never_exposes_future_rows_to_strategy():
    panel = make_panel(n_days=120)
    provider = InMemoryProvider(panel)
    config = _test_config()
    strategy = RecordingStrategy(config.universe)
    simulator = WalkForwardSimulator(config, provider, strategy)

    result = simulator.run()

    assert len(strategy.seen_states) > 0
    for state in strategy.seen_states:
        assert state.price_panel.index.max() <= state.as_of

    # Every rebalance must execute strictly after its decision date, using
    # the *next* trading day's price -- never the decision day's own close
    # being reused as if it were the execution price for a same-day fill.
    for record in result.rebalances:
        assert record.execution_date > record.decision_date


def test_engine_execution_uses_next_day_price_not_decision_day_price():
    # Prices double at index 60 onward; if execution incorrectly used the
    # decision-day price instead of next-day's, share counts computed from
    # a rebalance whose decision date is right at the jump would be wrong.
    idx = pd.bdate_range("2024-01-02", periods=120)
    prices = pd.Series([100.0] * 60 + [200.0] * 60, index=idx)
    panel = pd.DataFrame({"AAPL": prices, "MSFT": prices, "SPY": prices})
    provider = InMemoryProvider(panel)
    config = _test_config()
    strategy = RecordingStrategy(config.universe)
    simulator = WalkForwardSimulator(config, provider, strategy)

    result = simulator.run()

    for record in result.rebalances:
        decision_price = panel.loc[pd.Timestamp(record.decision_date), "AAPL"]
        execution_price = panel.loc[pd.Timestamp(record.execution_date), "AAPL"]
        implied_shares = record.trades[0].shares_delta if record.trades else None
        if implied_shares is not None and decision_price != execution_price:
            trade_price_used = record.trades[0].price
            assert trade_price_used == pytest.approx(execution_price)
            assert trade_price_used != pytest.approx(decision_price)


def test_engine_refuses_a_position_limit_breach():
    import pytest
    panel = make_panel(n_days=120)
    config = _test_config()
    config.portfolio_constraints.max_position_weight = .2
    with pytest.raises(ValueError, match='invalid proposal'):
        WalkForwardSimulator(config, InMemoryProvider(panel), RecordingStrategy(config.universe)).run()
