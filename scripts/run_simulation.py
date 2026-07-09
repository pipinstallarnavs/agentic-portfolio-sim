#!/usr/bin/env python3
"""Run the deterministic baselines end-to-end over the configured universe
and date range, and write a comparison report to results/.

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --provider local   # fail loudly instead of hitting network
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_portfolio.agents.orchestrator import MultiAgentStrategy
from agentic_portfolio.baselines.equal_weight import EqualWeightBaseline
from agentic_portfolio.baselines.momentum import MomentumBaseline
from agentic_portfolio.config import load_config
from agentic_portfolio.data.provider import get_data_provider
from agentic_portfolio.evaluation.metrics import buy_and_hold_equity_curve, compute_financial_metrics
from agentic_portfolio.evaluation.report import save_results
from agentic_portfolio.simulation.engine import WalkForwardSimulator
from agentic_portfolio.simulation.state import SimulationResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_simulation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--provider", default=None, choices=["yfinance", "local"], help="Override config.data.provider")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    config = load_config(args.config)
    provider_kind = args.provider or config.data.provider
    provider = get_data_provider(provider_kind, config.raw_dir)

    strategies = [
        MomentumBaseline(config.momentum_baseline, config.portfolio_constraints),
        EqualWeightBaseline(cash_weight=0.0),
        MultiAgentStrategy(config.portfolio_constraints, config.risk, top_n=config.momentum_baseline.top_n),
    ]

    results: dict[str, SimulationResult] = {}
    for strategy in strategies:
        logger.info("Running strategy: %s", strategy.name)
        simulator = WalkForwardSimulator(config, provider, strategy)
        result = simulator.run()
        results[strategy.name] = result
        logger.info(
            "%s: %d rebalances, final value %.0f",
            strategy.name, len(result.rebalances), list(result.equity_curve.values())[-1],
        )

    logger.info("Building SPY buy-and-hold benchmark")
    spy_history = provider.get_history(config.benchmark, config.data.start_date, config.data.end_date)
    spy_curve = buy_and_hold_equity_curve(spy_history["adj_close"], config.simulation.initial_capital)
    # Align to the same trading days the strategies used, for an apples-to-apples comparison.
    any_result = next(iter(results.values()))
    strategy_dates = any_result.equity_series().index
    spy_curve = spy_curve.reindex(strategy_dates).ffill()
    results["spy_buy_and_hold"] = SimulationResult(
        strategy_name="spy_buy_and_hold",
        equity_curve={d.date().isoformat(): float(v) for d, v in spy_curve.items()},
        rebalances=[],
        config_snapshot=config.model_dump(),
    )

    report_path = save_results(
        results, Path(args.output_dir), config.risk.var_confidence, config.risk.es_confidence
    )
    logger.info("Report written to %s", report_path)

    print("\n=== Strategy Comparison ===")
    for name, result in results.items():
        metrics = compute_financial_metrics(result, config.risk.var_confidence, config.risk.es_confidence)
        print(
            f"{name:24s} | return {metrics['annualized_return']:+.2%} | "
            f"vol {metrics['annualized_volatility']:.2%} | sharpe {metrics['sharpe_ratio']:.2f} | "
            f"max_dd {metrics['max_drawdown']:.2%}"
        )


if __name__ == "__main__":
    main()
