import numpy as np
import pandas as pd
import pytest

from agentic_portfolio.agents.base import AgentContext
from agentic_portfolio.agents.orchestrator import MultiAgentStrategy
from agentic_portfolio.agents.portfolio import MockPortfolioAgent, PortfolioAgentResult
from agentic_portfolio.agents.quant import QuantAgent
from agentic_portfolio.agents.research import MockResearchAgent
from agentic_portfolio.agents.risk import RiskAgent
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
from agentic_portfolio.simulation.state import PortfolioSnapshot, build_market_state
from agentic_portfolio.tools.features import calculate_momentum, calculate_realized_volatility, calculate_returns


def make_panel(n_days: int = 150, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    tickers = ["A", "B", "C", "D"]
    data = {}
    for i, t in enumerate(tickers):
        drift = 0.0003 + 0.0001 * i
        vol = 0.01 + 0.003 * i
        shocks = rng.normal(drift, vol, size=n_days)
        data[t] = 100.0 * np.exp(np.cumsum(shocks))
    bench_shocks = rng.normal(0.0002, 0.008, size=n_days)
    data["SPY"] = 100.0 * np.exp(np.cumsum(bench_shocks))
    return pd.DataFrame(data, index=idx)


def empty_portfolio_state(as_of) -> PortfolioSnapshot:
    return PortfolioSnapshot(date=as_of.date(), cash=1_000_000.0, shares={}, weights={}, total_value=1_000_000.0)


def make_context(portfolio_state, constraints=None, risk_config=None) -> AgentContext:
    return AgentContext(
        portfolio_state=portfolio_state,
        constraints=constraints or PortfolioConstraintsConfig(),
        risk_limits=risk_config or RiskConfig(),
    )


# --- Research Agent -------------------------------------------------------

def test_research_agent_tags_positive_trend():
    panel = make_panel()
    cutoff = panel.index[100]
    state = build_market_state(panel, cutoff, ["A", "B", "C", "D"], "SPY")
    agent = MockResearchAgent(trend_window=20)
    result = agent.run(state, make_context(empty_portfolio_state(cutoff)))

    assert set(result.context_tags.keys()) == {"A", "B", "C", "D"}
    for ticker in ["A", "B", "C", "D"]:
        trend = calculate_momentum(state.prices(ticker), 20)
        tags = result.context_tags[ticker]
        if trend > 0.02:
            assert "positive_trend" in tags
        elif trend < -0.02:
            assert "negative_trend" in tags


def test_research_agent_no_lookahead_via_market_state():
    panel = make_panel()
    cutoff = panel.index[80]
    state = build_market_state(panel, cutoff, ["A", "B"], "SPY")
    assert state.price_panel.index.max() <= cutoff
    agent = MockResearchAgent()
    result = agent.run(state, make_context(empty_portfolio_state(cutoff)))
    assert result.agent_name == "research_agent"


# --- Quant Agent ------------------------------------------------------------

def test_quant_agent_features_match_tools_directly():
    panel = make_panel()
    cutoff = panel.index[100]
    state = build_market_state(panel, cutoff, ["A", "B", "C", "D"], "SPY")
    agent = QuantAgent()
    result = agent.run(state, make_context(empty_portfolio_state(cutoff)))

    expected_mom_60 = calculate_momentum(state.prices("A"), 60)
    assert result.features["A"]["momentum_60d"] == pytest.approx(expected_mom_60, nan_ok=True)

    expected_vol_20 = calculate_realized_volatility(calculate_returns(state.prices("B")), 20)
    assert result.features["B"]["volatility_20d"] == pytest.approx(expected_vol_20, nan_ok=True)


def test_quant_agent_signal_scores_only_for_valid_data():
    panel = make_panel(n_days=150)
    cutoff = panel.index[10]  # not enough history for 60d momentum/vol
    state = build_market_state(panel, cutoff, ["A", "B", "C", "D"], "SPY")
    agent = QuantAgent()
    result = agent.run(state, make_context(empty_portfolio_state(cutoff)))
    assert result.signal_scores == {}


# --- Portfolio Agent ---------------------------------------------------------

def test_portfolio_agent_respects_max_position_weight():
    panel = make_panel()
    cutoff = panel.index[120]
    universe = ["A", "B", "C", "D"]
    state = build_market_state(panel, cutoff, universe, "SPY")

    research = MockResearchAgent().run(state, make_context(empty_portfolio_state(cutoff)))
    quant = QuantAgent().run(state, make_context(empty_portfolio_state(cutoff)))

    constraints = PortfolioConstraintsConfig(max_position_weight=0.20, max_gross_exposure=1.0)
    agent = MockPortfolioAgent(constraints, top_n=2)  # only 2 names -> would need 50% each without a cap
    result = agent.run(
        state, make_context(empty_portfolio_state(cutoff), constraints=constraints),
        research=research, quant=quant,
    )
    assert all(w <= 0.20 + 1e-9 for w in result.target_weights.values())


def test_portfolio_agent_holds_cash_with_no_signals():
    panel = make_panel()
    cutoff = panel.index[5]
    universe = ["A", "B"]
    state = build_market_state(panel, cutoff, universe, "SPY")
    quant = QuantAgent().run(state, make_context(empty_portfolio_state(cutoff)))
    research = MockResearchAgent().run(state, make_context(empty_portfolio_state(cutoff)))
    agent = MockPortfolioAgent(PortfolioConstraintsConfig())
    result = agent.run(state, make_context(empty_portfolio_state(cutoff)), research=research, quant=quant)
    assert result.target_weights == {}
    assert result.cash_weight == pytest.approx(1.0)
    assert result.confidence == pytest.approx(0.0)


# --- Risk Agent ---------------------------------------------------------

def test_risk_agent_approves_compliant_proposal():
    portfolio_state = PortfolioSnapshot(
        date=pd.Timestamp("2023-06-01").date(), cash=0.0,
        shares={}, weights={"A": 0.15, "B": 0.15}, total_value=1_000_000.0,
    )
    panel = make_panel()
    cutoff = panel.index[120]
    state = build_market_state(panel, cutoff, ["A", "B", "C", "D"], "SPY")
    proposal = PortfolioAgentResult(
        agent_name="portfolio_agent", target_weights={"A": 0.16, "B": 0.14}, cash_weight=0.70, confidence=1.0,
    )
    constraints = PortfolioConstraintsConfig(max_turnover_per_rebalance=1.0)
    risk_limits = RiskConfig(max_portfolio_volatility=5.0, max_historical_var_95=5.0)  # effectively unconstrained
    agent = RiskAgent()
    result = agent.run(
        state, make_context(portfolio_state, constraints, risk_limits), proposal=proposal,
    )
    assert result.decision == "approve"
    assert result.modifications == []
    assert result.final_weights == pytest.approx(proposal.target_weights)


def test_risk_agent_rejects_low_confidence_and_holds_prior():
    prior_weights = {"A": 0.2, "B": 0.2}
    portfolio_state = PortfolioSnapshot(
        date=pd.Timestamp("2023-06-01").date(), cash=0.6, shares={}, weights=prior_weights, total_value=1_000_000.0,
    )
    panel = make_panel()
    cutoff = panel.index[120]
    state = build_market_state(panel, cutoff, ["A", "B", "C", "D"], "SPY")
    proposal = PortfolioAgentResult(
        agent_name="portfolio_agent", target_weights={"C": 0.5}, cash_weight=0.5, confidence=0.05,
    )
    risk_limits = RiskConfig(min_proposal_confidence=0.15)
    agent = RiskAgent()
    result = agent.run(
        state, make_context(portfolio_state, PortfolioConstraintsConfig(), risk_limits), proposal=proposal,
    )
    assert result.decision == "reject"
    assert result.final_weights == prior_weights


def test_risk_agent_scales_down_for_turnover_limit():
    portfolio_state = PortfolioSnapshot(
        date=pd.Timestamp("2023-06-01").date(), cash=1.0, shares={}, weights={}, total_value=1_000_000.0,
    )
    panel = make_panel()
    cutoff = panel.index[120]
    state = build_market_state(panel, cutoff, ["A", "B", "C", "D"], "SPY")
    proposal = PortfolioAgentResult(
        agent_name="portfolio_agent", target_weights={"A": 0.5, "B": 0.5}, cash_weight=0.0, confidence=1.0,
    )
    constraints = PortfolioConstraintsConfig(max_turnover_per_rebalance=0.20, max_position_weight=1.0)
    risk_limits = RiskConfig(max_portfolio_volatility=5.0, max_historical_var_95=5.0)
    agent = RiskAgent()
    result = agent.run(
        state, make_context(portfolio_state, constraints, risk_limits), proposal=proposal,
    )
    assert result.decision == "modify"
    assert any("turnover" in m.lower() for m in result.modifications)
    # Scaled weights should be materially smaller than the full 50/50 proposal.
    assert sum(result.final_weights.values()) < 0.5


def test_risk_agent_caps_constraint_violation():
    portfolio_state = empty_portfolio_state(pd.Timestamp("2023-06-01"))
    panel = make_panel()
    cutoff = panel.index[120]
    state = build_market_state(panel, cutoff, ["A", "B", "C", "D"], "SPY")
    proposal = PortfolioAgentResult(
        agent_name="portfolio_agent", target_weights={"A": 0.6, "B": 0.4}, cash_weight=0.0, confidence=1.0,
    )
    constraints = PortfolioConstraintsConfig(max_position_weight=0.20, max_turnover_per_rebalance=1.0)
    risk_limits = RiskConfig(max_portfolio_volatility=5.0, max_historical_var_95=5.0)
    agent = RiskAgent()
    result = agent.run(
        state, make_context(portfolio_state, constraints, risk_limits), proposal=proposal,
    )
    assert result.decision == "modify"
    assert all(w <= 0.20 + 1e-9 for w in result.final_weights.values())


# --- Full multi-agent orchestration through the engine -----------------------

def _agent_test_config() -> Config:
    return Config(
        universe=["A", "B", "C", "D"],
        benchmark="SPY",
        data=DataConfig(start_date="2023-01-02", end_date="2023-12-31", provider="local"),
        simulation=SimulationConfig(
            rebalance_freq="W-FRI", initial_capital=1_000_000, transaction_cost_bps=5, min_history_days=65
        ),
        portfolio_constraints=PortfolioConstraintsConfig(),
        momentum_baseline=MomentumBaselineConfig(),
        risk=RiskConfig(),
    )


class _InMemoryProvider(DataProvider):
    def __init__(self, panel: pd.DataFrame):
        self.panel = panel

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_price_panel(self, tickers, start, end) -> pd.DataFrame:
        return self.panel[tickers].loc[start:end]


def test_multi_agent_strategy_runs_end_to_end_through_engine():
    panel = make_panel(n_days=150)
    provider = _InMemoryProvider(panel)
    config = _agent_test_config()
    strategy = MultiAgentStrategy(config.portfolio_constraints, config.risk, top_n=2)
    simulator = WalkForwardSimulator(config, provider, strategy)

    result = simulator.run()

    assert len(result.rebalances) > 0
    assert len(strategy.audit_log) == len(result.rebalances)
    for record in result.rebalances:
        assert record.reasoning_summary != ""
        assert 0.0 <= record.confidence <= 1.0

    equity = result.equity_series()
    assert equity.iloc[0] == pytest.approx(config.simulation.initial_capital)
    assert (equity > 0).all()


def test_risk_metrics_describe_final_scaled_weights():
    from agentic_portfolio.tools.portfolio import calculate_portfolio_volatility
    panel = make_panel()
    cutoff = panel.index[120]
    state = build_market_state(panel, cutoff, ['A', 'B', 'C', 'D'], 'SPY')
    context = make_context(empty_portfolio_state(cutoff),
                           PortfolioConstraintsConfig(max_position_weight=1),
                           RiskConfig(max_portfolio_volatility=.03, max_historical_var_95=5))
    proposal = PortfolioAgentResult(agent_name='portfolio_agent', target_weights={'A': 1}, confidence=1)
    result = RiskAgent().run(state, context, proposal=proposal)
    expected = calculate_portfolio_volatility(result.final_weights, state.returns_panel()[['A']].cov())
    assert result.risk_metrics['portfolio_volatility'] == pytest.approx(expected)
    assert expected <= .03001
