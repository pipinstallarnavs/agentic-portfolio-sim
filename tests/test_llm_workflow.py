"""Offline tests verify plumbing, not language-model investment skill."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pandas as pd
import pytest
from agentic_portfolio.agents.llm_strategy import LLMPortfolioStrategy, StructuredCalls, WeightDecision
from agentic_portfolio.config import PortfolioConstraintsConfig, RiskConfig
from agentic_portfolio.llm.anthropic_client import AnthropicLLMClient
from agentic_portfolio.llm.base import LLMResponse
from agentic_portfolio.simulation.state import build_market_state, PortfolioSnapshot


class Responses:
    def __init__(self, values):
        self.values, self.prompts = iter(values), []
    def complete(self, system, prompt):
        self.prompts.append(json.loads(prompt))
        return LLMResponse(content=next(self.values), model='test-double')


def proposal(weights):
    return json.dumps(dict(target_weights=weights, confidence=.8, rationale='Test fixture.'))


def fixture_state():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range('2023-01-02', periods=90)
    panel = pd.DataFrame(100*np.exp(np.cumsum(rng.normal(0, .01, (90, 4)), axis=0)),
                         index=dates, columns=['A','B','C','SPY'])
    cutoff = dates[70]
    state = build_market_state(panel, cutoff, ['A','B','C'], 'SPY')
    holdings = PortfolioSnapshot(date=cutoff.date(), cash=1000, shares={}, weights={}, total_value=1000)
    return state, holdings, panel


def make_strategy(values, multi=False):
    client = Responses(values)
    strategy = LLMPortfolioStrategy(StructuredCalls(client), PortfolioConstraintsConfig(), RiskConfig(), multi_agent=multi)
    return strategy, client


def test_invalid_position_is_revised_using_risk_feedback():
    strategy, client = make_strategy([proposal({'A': .6}), proposal({'A': .1})])
    state, holdings, _ = fixture_state()
    result = strategy.generate_weights(state, holdings)
    assert result.target_weights == {'A': .1}
    assert client.prompts[1]['previous_proposal']['target_weights'] == {'A': .6}
    assert client.prompts[1]['risk_feedback']['decision'] == 'invalid'
    assert len(strategy.audit_log[0]['rounds']) == 2


def test_multi_adds_research_and_single_uses_identical_numeric_inputs():
    state, holdings, _ = fixture_state()
    multi, mclient = make_strategy([json.dumps({'summary':'Test context','risks':[]}), proposal({'A': .1})], True)
    single, sclient = make_strategy([proposal({'A': .1})])
    multi.generate_weights(state, holdings)
    single.generate_weights(state, holdings)
    assert mclient.prompts[1]['features'] == sclient.prompts[0]['features']
    assert 'research' in mclient.prompts[1] and 'research' not in sclient.prompts[0]


@pytest.mark.parametrize('raw', ['not JSON', proposal({'A': float('nan')}), proposal({'A': -.1})])
def test_malformed_responses_fail_without_executing(raw):
    strategy, _ = make_strategy([raw])
    state, holdings, _ = fixture_state()
    with pytest.raises(ValueError, match='JSON/schema'):
        strategy.generate_weights(state, holdings)
    assert strategy.calls.records[0]['invalid_response']


def test_unknown_ticker_fails():
    strategy, _ = make_strategy([proposal({'UNKNOWN': .1})])
    state, holdings, _ = fixture_state()
    with pytest.raises(ValueError, match='unknown ticker'):
        strategy.generate_weights(state, holdings)


def test_failed_revision_is_not_accepted():
    strategy, client = make_strategy([proposal({'A': .6}), proposal({'A': .6})])
    state, holdings, _ = fixture_state()
    with pytest.raises(ValueError, match='No valid portfolio'):
        strategy.generate_weights(state, holdings)
    assert len(client.prompts) == 2


def test_budget_checked_before_provider_request():
    client = Responses([proposal({'A': .1})])
    calls = StructuredCalls(client, max_calls=1)
    calls.ask(WeightDecision, {})
    with pytest.raises(RuntimeError, match='budget'):
        calls.ask(WeightDecision, {})
    assert len(client.prompts) == 1


def test_future_price_change_does_not_change_prompt():
    state, holdings, panel = fixture_state()
    first, c1 = make_strategy([proposal({'A': .1})])
    first.generate_weights(state, holdings)
    panel.loc[panel.index > state.as_of] *= 100
    second, c2 = make_strategy([proposal({'A': .1})])
    second.generate_weights(build_market_state(panel, state.as_of, state.universe, state.benchmark), holdings)
    assert c1.prompts == c2.prompts


def test_provider_adapter_request_and_usage():
    sdk = Mock()
    sdk.messages.create.return_value = SimpleNamespace(stop_reason='end_turn', model='account-model',
        content=[SimpleNamespace(type='text', text='{}')], usage=SimpleNamespace(input_tokens=12, output_tokens=2))
    response = AnthropicLLMClient(model='account-model', sdk_client=sdk).complete('system', 'payload')
    assert response.input_tokens == 12 and response.output_tokens == 2
    assert sdk.messages.create.call_args.kwargs['messages'][0]['content'] == 'payload'


def test_missing_key_does_not_fall_back_to_mock(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    with pytest.raises(ValueError, match='ANTHROPIC_API_KEY'):
        AnthropicLLMClient(model='account-model')


def test_truncated_provider_response_rejected():
    sdk = Mock()
    sdk.messages.create.return_value = SimpleNamespace(stop_reason='max_tokens')
    with pytest.raises(RuntimeError, match='did not finish'):
        AnthropicLLMClient(model='account-model', sdk_client=sdk).complete('system', 'payload')
