#!/usr/bin/env python3
"""Bounded architecture demo, offline by default. --live makes paid API calls.

The default 75-day panel supplies warmup and one rebalance, not enough
observations to draw an investment-performance conclusion.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from learn import synthetic_prices, MemoryPrices
from agentic_portfolio.agents.llm_strategy import LLMPortfolioStrategy, StructuredCalls, json_safe
from agentic_portfolio.baselines.momentum import MomentumBaseline
from agentic_portfolio.baselines.equal_weight import EqualWeightBaseline
from agentic_portfolio.config import load_config
from agentic_portfolio.data.provider import LocalCacheDataProvider
from agentic_portfolio.llm.base import LLMClient, LLMResponse
from agentic_portfolio.llm.provider import get_llm_client
from agentic_portfolio.simulation.engine import WalkForwardSimulator


class ReplayClient(LLMClient):
    """Scripted TEST DOUBLE; never evidence of an actual model decision."""
    def complete(self, system_prompt, user_prompt):
        data = json.loads(user_prompt)
        if 'ResearchNotes' in system_prompt:
            content = dict(summary='Scripted response using supplied features.', risks=['Synthetic demo only.'])
        else:
            # Deliberately exceed a cap first, then fix it when feedback arrives.
            # This exercises the loop; it is not model intelligence.
            weights = {data['universe'][0]: .6}
            if 'risk_feedback' in data:
                weights = {t: .1 for t in data['universe'][:3]}
            content = dict(target_weights=weights, confidence=.8, rationale='Scripted offline proposal.')
        return LLMResponse(content=json.dumps(content), model='offline-replay')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Use Anthropic and incur API charges')
    parser.add_argument('--cached', action='store_true', help='Read existing cache, provenance unverified')
    parser.add_argument('--days', type=int, default=75)
    parser.add_argument('--max-calls', type=int, default=12, help='Shared across both model strategies')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    if args.days < 72 or args.max_calls < 1:
        parser.error('Need at least 72 days and one allowed call')
    cfg = load_config()
    names = cfg.universe+[cfg.benchmark]
    if args.cached:
        panel = LocalCacheDataProvider(cfg.raw_dir).get_price_panel(names, cfg.data.start_date, cfg.data.end_date).dropna()
    else:
        panel = synthetic_prices(names, cfg.random_seed)
    panel = panel.iloc[:args.days]
    if len(panel) < 72:
        parser.error('Too little aligned history')
    cfg.data.start_date, cfg.data.end_date = str(panel.index[0].date()), str(panel.index[-1].date())
    if args.live:
        from dotenv import load_dotenv
        load_dotenv(ROOT/'.env')
        client = get_llm_client('anthropic')
    else:
        client = ReplayClient()
    calls = StructuredCalls(client, max_calls=args.max_calls)
    multi = LLMPortfolioStrategy(calls, cfg.portfolio_constraints, cfg.risk)
    single = LLMPortfolioStrategy(calls, cfg.portfolio_constraints, cfg.risk, multi_agent=False)
    strategies = [multi, single, MomentumBaseline(cfg.momentum_baseline, cfg.portfolio_constraints), EqualWeightBaseline()]
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  execution_mode='live' if args.live else 'offline-replay',
                  data='local cache, provenance unverified' if args.cached else 'synthetic',
                  config=cfg.model_dump(mode='json'), comparisons={})
    output = Path(args.output) if args.output else ROOT/'results'/('llm-live.json' if args.live else 'llm-replay.json')
    try:
        for strategy in strategies:
            result = WalkForwardSimulator(cfg, MemoryPrices(panel), strategy).run()
            report['comparisons'][strategy.name] = dict(
                rebalances=len(result.rebalances),
                total_return=result.equity_series().iloc[-1]/cfg.simulation.initial_capital-1,
                transaction_cost=sum(r.transaction_cost for r in result.rebalances),
                records=[r.model_dump(mode='json') for r in result.rebalances])
        report['status'] = 'completed'
    except Exception as error:
        # Preserve evidence without serializing SDK exceptions/credentials.
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        raise
    finally:
        report.update(calls_attempted=calls.attempts, model_calls=calls.records,
                      multi_agent_audit=multi.audit_log, single_agent_audit=single.audit_log)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False)+'\n')
    print(f"{report['execution_mode']}: {calls.attempts} calls; report: {output}")
    print('Short workflow demonstration; no claim of model investment skill.')


if __name__ == '__main__':
    main()
