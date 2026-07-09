"""Download fresh market prices and evaluate existing deterministic strategies.

No synthetic fallback, model calls, parameter search, or broker orders.
Each run saves its own prices, provenance, decisions, and report.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yfinance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_portfolio.agents.orchestrator import MultiAgentStrategy
from agentic_portfolio.baselines.equal_weight import EqualWeightBaseline
from agentic_portfolio.baselines.momentum import MomentumBaseline
from agentic_portfolio.config import load_config
from agentic_portfolio.data.provider import LocalCacheDataProvider, YFinanceDataProvider
from agentic_portfolio.evaluation.metrics import compute_financial_metrics
from agentic_portfolio.simulation.engine import WalkForwardSimulator, get_rebalance_dates


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "results" / f"market-validation-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_config()
    cfg.data.start_date = "2019-09-01"  # History for features before evaluation.
    cfg.data.end_date = "2026-01-01"  # Yahoo's end date is exclusive.
    cfg.data.raw_dir = str(output / "prices")
    downloader = YFinanceDataProvider(cfg.raw_dir)
    provenance = {
        "source": "Fresh Yahoo Finance downloads via yfinance; no synthetic fallback",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": yfinance.__version__,
        "requested_start": cfg.data.start_date,
        "requested_end_exclusive": cfg.data.end_date,
        "series": {},
    }
    for ticker in cfg.universe + [cfg.benchmark]:
        history = downloader.get_history(ticker, cfg.data.start_date, cfg.data.end_date)
        if history.index.has_duplicates or not history.index.is_monotonic_increasing:
            raise ValueError(f"Invalid dates for {ticker}")
        if history["adj_close"].isna().any() or not np.isfinite(history["adj_close"]).all() or (history["adj_close"] <= 0).any():
            raise ValueError(f"Invalid adjusted prices for {ticker}")
        provenance["series"][ticker] = {
            "rows": len(history), "first_date": str(history.index[0].date()),
            "last_date": str(history.index[-1].date()),
            "sha256": hashlib.sha256(downloader.cache.path_for(ticker).read_bytes()).hexdigest(),
        }
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"Downloaded {ticker}: {len(history)} market observations", flush=True)

    provider = LocalCacheDataProvider(cfg.raw_dir)
    panel = provider.get_price_panel(cfg.universe + [cfg.benchmark], cfg.data.start_date, cfg.data.end_date)
    if panel.isna().any().any():
        raise ValueError("Missing dates across instruments; investigate rather than silently dropping rows")
    # Start decisions in 2020 while retaining earlier observations for features.
    cfg.simulation.min_history_days = int(panel.index.searchsorted(pd.Timestamp("2020-01-01")))
    decision_dates = [d for d in get_rebalance_dates(panel.index, cfg.simulation.rebalance_freq)
                      if cfg.simulation.min_history_days <= panel.index.get_loc(d) < len(panel)-1]
    first_decision = decision_dates[0]
    first_fill = panel.index[panel.index.get_loc(first_decision)+1]
    strategies = [EqualWeightBaseline(), MomentumBaseline(cfg.momentum_baseline, cfg.portfolio_constraints),
                  MultiAgentStrategy(cfg.portfolio_constraints, cfg.risk, top_n=cfg.momentum_baseline.top_n)]
    report = {
        "execution_mode": "deterministic; no LLM calls", "data": provenance["source"],
        "config": cfg.model_dump(mode="json"),
        "valuation_start": str(first_decision.date()), "first_fill": str(first_fill.date()),
        "valuation_end": str(panel.index[-1].date()), "comparisons": {}, "annual_returns": {},
        "limitations": [
            "Fixed retrospectively selected ten-stock universe: selection/survivorship bias.",
            "Descriptive historical backtest; no held-out model selection or statistical significance claim.",
            "Baselines share dates, prices, caps and trading costs, but risk overlays differ.",
            "Sharpe assumes a zero risk-free rate; cash earns zero.",
            "Adjusted prices approximate total returns; no market impact, financing or partial fills.",
            "Warmup excluded; valuation begins before the first fill so its transaction fee is included.",
            "No evidence about live LLM performance is produced by this run.",
        ],
    }
    curves = {}
    for strategy in strategies:
        result = WalkForwardSimulator(cfg, provider, strategy).run()
        if pd.Timestamp(result.rebalances[0].execution_date) != first_fill:
            raise AssertionError("Strategies do not share the first fill")
        result.equity_curve = {d: v for d, v in result.equity_curve.items() if pd.Timestamp(d) >= first_decision}
        for trade in result.rebalances:
            next_day = panel.index[panel.index.get_loc(pd.Timestamp(trade.decision_date))+1]
            if pd.Timestamp(trade.execution_date) != next_day:
                raise AssertionError("Execution timing violation")
            if not np.isclose(trade.portfolio_value_before - trade.transaction_cost, trade.portfolio_value_after):
                raise AssertionError("Fee conservation violation")
        equity = result.equity_series()
        metrics = compute_financial_metrics(result)
        metrics.pop("sortino_ratio")  # Inherited definition is not standard Sortino.
        daily = equity.pct_change().dropna()
        if not np.isclose(metrics["sharpe_ratio"], daily.mean()/daily.std(ddof=1)*np.sqrt(252)):
            raise AssertionError("Sharpe cross-check failed")
        if not np.isclose(metrics["total_return"], np.prod(1+daily)-1):
            raise AssertionError("Return cross-check failed")
        report["comparisons"][strategy.name] = metrics
        report["annual_returns"][strategy.name] = {str(year): float(np.prod(1+values)-1)
                                                   for year, values in daily.groupby(daily.index.year)}
        curves[strategy.name] = equity
        (output / f"{strategy.name}.json").write_text(result.model_dump_json(indent=2) + "\n")
        if isinstance(strategy, MultiAgentStrategy):
            report["risk_decisions"] = dict(Counter(a.risk.decision for a in strategy.audit_log))
            audits = [{"as_of": a.as_of, "research": a.research.model_dump(mode="json"),
                       "quant": a.quant.model_dump(mode="json"), "portfolio": a.portfolio.model_dump(mode="json"),
                       "risk": a.risk.model_dump(mode="json")} for a in strategy.audit_log]
            (output / "risk-audit.json").write_text(json.dumps(audits, indent=2, allow_nan=False) + "\n")
        print(f"{strategy.name}: annualized return {metrics['annualized_return']:.2%}, "
              f"Sharpe {metrics['sharpe_ratio']:.3f}, drawdown {metrics['max_drawdown']:.2%}", flush=True)
    pd.DataFrame(curves).to_csv(output / "equity_curves.csv")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Real-market deterministic backtest", "", f"Valuation: {report['valuation_start']} to {report['valuation_end']}; first fill {report['first_fill']}.",
             "", "Fresh Yahoo Finance adjusted-close data. Existing defaults; 10 bps per traded notional. No LLM calls.", "",
             "| Strategy | Total return | Annualized return | Sharpe (zero RF) | Max drawdown | Rebalances |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, m in report["comparisons"].items():
        lines.append(f"| {name} | {m['total_return']:.2%} | {m['annualized_return']:.2%} | {m['sharpe_ratio']:.2f} | {m['max_drawdown']:.2%} | {int(m['n_rebalances'])} |")
    lines += ["", "## Calendar-year returns", "", "2020 begins with the first January fill; annual returns include trading fees.", "",
              "| Year | Equal weight | Momentum | Rule-based multi-agent |", "|---|---:|---:|---:|"]
    for year in report["annual_returns"][strategies[0].name]:
        lines.append("| " + year + " | " + " | ".join(f"{report['annual_returns'][s.name][year]:.2%}" for s in strategies) + " |")
    lines += ["", "## Interpretation", ""] + ["- " + x for x in report["limitations"]]
    lines += ["", "## Validation", "", "Checked positive finite prices, aligned dates, next-trading-day fills, fee conservation, compounded returns and Sharpe arithmetic.",
              "Per-ticker source metadata and hashes are in provenance.json; full decisions and equity curves are saved beside this report."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"REPORT: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
