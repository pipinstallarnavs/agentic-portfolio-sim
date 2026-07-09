"""Follow one portfolio decision, then compare it with baselines out of sample.

Default data is generated in memory and never overwrites the existing cache.
Use --cached to inspect locally cached prices; this does not verify their source.
There are no paid API calls: all four agents are ordinary Python rules.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent/'src'))
import numpy as np
import pandas as pd
from agentic_portfolio.agents.orchestrator import MultiAgentStrategy
from agentic_portfolio.baselines.equal_weight import EqualWeightBaseline
from agentic_portfolio.baselines.momentum import MomentumBaseline
from agentic_portfolio.config import load_config
from agentic_portfolio.data.provider import DataProvider, LocalCacheDataProvider
from agentic_portfolio.evaluation.metrics import compute_financial_metrics
from agentic_portfolio.simulation.engine import WalkForwardSimulator
from agentic_portfolio.simulation.state import build_market_state, PortfolioSnapshot


class MemoryPrices(DataProvider):
    def __init__(self, panel):
        self.panel = panel

    def get_history(self, ticker, start, end):
        # Only adjusted close is needed by this experiment's shared engine.
        return self.panel.loc[start:end, [ticker]].rename(columns={ticker: 'adj_close'})


def synthetic_prices(tickers, seed):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2023-01-02', periods=320)
    common = rng.normal(.0002, .008, len(dates))
    # Shared factor creates correlation; independent noise creates imperfect
    # diversification. A high-volatility interval makes risk controls visible.
    common[160:200] *= 3
    moves = np.column_stack([common+rng.normal(.00005, .007+i*.0003, len(dates))
                             for i in range(len(tickers))])
    return pd.DataFrame(100*np.exp(np.cumsum(moves, axis=0)), index=dates, columns=tickers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached', action='store_true')
    parser.add_argument('--cost-bps', type=float, default=10)
    parser.add_argument('--risk-vol', type=float, default=.20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.cost_bps < 10000 or args.risk_vol <= 0:
        parser.error('Cost must be in [0,10000) bps; risk volatility must be positive')
    config = load_config()
    config.simulation.transaction_cost_bps = args.cost_bps
    config.risk.max_portfolio_volatility = args.risk_vol
    tickers = config.universe + [config.benchmark]
    if args.cached:
        provider = LocalCacheDataProvider(config.raw_dir)
    else:
        panel = synthetic_prices(tickers, args.seed)
        config.data.start_date = str(panel.index[0].date())
        config.data.end_date = str(panel.index[-1].date())
        provider = MemoryPrices(panel)
    panel = provider.get_price_panel(tickers, config.data.start_date, config.data.end_date).dropna()
    if len(panel) <= config.simulation.min_history_days+1:
        raise ValueError('Insufficient aligned price history')

    # Trace ONE actual decision before looking at a performance table.
    # A state contains only rows through its decision date, so neither the
    # research agent nor the quant agent can inspect tomorrow's price.
    cutoff = panel.index[config.simulation.min_history_days]
    state = build_market_state(panel, cutoff, config.universe, config.benchmark)
    holdings = PortfolioSnapshot(date=cutoff.date(), cash=1e6, shares={}, weights={}, total_value=1e6)
    strategy = MultiAgentStrategy(config.portfolio_constraints, config.risk)
    proposal = strategy.generate_weights(state, holdings)
    audit = strategy.audit_log[-1]
    print('Decision date:', cutoff.date(), '| Last visible date:', state.price_panel.index[-1].date())
    print('Quant scores:', {k: round(v, 4) for k, v in audit.quant.signal_scores.items()})
    print('Before risk:', audit.portfolio.target_weights)
    print('After risk: ', proposal.target_weights)
    print('Risk decision:', audit.risk.rationale)

    # All strategies use the same prices, dates, execution delay and costs.
    # The rule-based multi-agent system adds price context and risk scaling;
    # it does NOT establish that an LLM provides investment skill.
    strategies = [EqualWeightBaseline(),
                  MomentumBaseline(config.momentum_baseline, config.portfolio_constraints),
                  MultiAgentStrategy(config.portfolio_constraints, config.risk)]
    comparisons = {}
    for strategy in strategies:
        result = WalkForwardSimulator(config, provider, strategy).run()
        # Remove shared initial cash-only warmup from performance statistics.
        # Retain the day before the first fill so its fee enters returns.
        first_fill = pd.Timestamp(result.rebalances[0].execution_date)
        start = panel.index[panel.index.get_loc(first_fill)-1]
        result.equity_curve = {d: v for d, v in result.equity_curve.items() if pd.Timestamp(d) >= start}
        comparisons[strategy.name] = compute_financial_metrics(result)
    for name, metrics in comparisons.items():
        print(f"{name:22} return={metrics['total_return']:+.2%} drawdown={metrics['max_drawdown']:.2%} costs={metrics['total_transaction_costs']:.2f}")

    # Save structured outputs, including the original proposal and review.
    # You can point to the exact date/rule that changed a position.
    destination = Path(__file__).parent/'results'/'learning_report.json'
    report = dict(data='local cache (provenance not verified)' if args.cached else 'synthetic',
                  seed=args.seed, cost_bps=args.cost_bps, risk_vol=args.risk_vol,
                  first_decision={name: (value.model_dump(mode='json') if hasattr(value, 'model_dump') else value)
                                  for name, value in vars(audit).items()}, comparisons=comparisons)
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Saved:', destination)


if __name__ == '__main__':
    main()
