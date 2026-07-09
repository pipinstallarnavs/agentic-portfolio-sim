"""Turns one or more SimulationResults into comparison tables and a
Markdown report, written to results/.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agentic_portfolio.evaluation.metrics import compute_financial_metrics
from agentic_portfolio.simulation.state import SimulationResult

METRIC_LABELS = {
    "final_value": "Final Value ($)",
    "total_return": "Total Return",
    "annualized_return": "Annualized Return",
    "annualized_volatility": "Annualized Volatility",
    "sharpe_ratio": "Sharpe Ratio",
    "sortino_ratio": "Sortino Ratio",
    "max_drawdown": "Max Drawdown",
    "calmar_ratio": "Calmar Ratio",
    "historical_var_95": "Historical VaR (95%)",
    "expected_shortfall_95": "Expected Shortfall (95%)",
    "n_rebalances": "# Rebalances",
    "mean_turnover": "Mean Turnover",
    "total_transaction_costs": "Total Txn Costs ($)",
    "total_transaction_costs_pct_of_initial": "Txn Costs (% of Initial Capital)",
}

PCT_METRICS = {
    "total_return", "annualized_return", "annualized_volatility", "max_drawdown",
    "historical_var_95", "expected_shortfall_95", "mean_turnover",
    "total_transaction_costs_pct_of_initial",
}


def build_comparison_table(
    results: dict[str, SimulationResult],
    var_confidence: float = 0.95,
    es_confidence: float = 0.95,
) -> pd.DataFrame:
    rows = {
        name: compute_financial_metrics(result, var_confidence, es_confidence)
        for name, result in results.items()
    }
    return pd.DataFrame(rows).T


def _format_cell(metric: str, value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if metric in PCT_METRICS:
        return f"{value:.2%}"
    if metric in ("final_value", "total_transaction_costs"):
        return f"{value:,.0f}"
    return f"{value:.3f}"


def render_markdown_report(
    results: dict[str, SimulationResult],
    table: pd.DataFrame,
    config_summary: dict,
) -> str:
    lines = [
        "# Experiment Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Configuration",
        "",
        f"- Universe: {', '.join(config_summary.get('universe', []))}",
        f"- Benchmark: {config_summary.get('benchmark')}",
        f"- Date range: {config_summary.get('data', {}).get('start_date')} to "
        f"{config_summary.get('data', {}).get('end_date')}",
        f"- Rebalance frequency: {config_summary.get('simulation', {}).get('rebalance_freq')}",
        f"- Transaction cost: {config_summary.get('simulation', {}).get('transaction_cost_bps')} bps",
        "",
        "## Strategy Comparison",
        "",
    ]

    header = "| Metric | " + " | ".join(table.index) + " |"
    sep = "|---" * (len(table.index) + 1) + "|"
    lines += [header, sep]
    for metric in table.columns:
        label = METRIC_LABELS.get(metric, metric)
        row_values = [_format_cell(metric, table.loc[name, metric]) for name in table.index]
        lines.append(f"| {label} | " + " | ".join(row_values) + " |")

    lines += ["", "## Notes", "", "- All metrics computed on the same date range and universe.",
              "- See individual `<strategy>_rebalances.json` files for the full per-rebalance audit trail."]
    return "\n".join(lines)


def save_results(
    results: dict[str, SimulationResult],
    output_dir: Path,
    var_confidence: float = 0.95,
    es_confidence: float = 0.95,
) -> Path:
    """Writes per-strategy equity curves + rebalance logs, a comparison
    CSV/JSON, and a Markdown report to output_dir. Returns the report path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = build_comparison_table(results, var_confidence, es_confidence)
    table.to_csv(output_dir / "comparison_metrics.csv")
    table.to_json(output_dir / "comparison_metrics.json", orient="index", indent=2)

    equity_curves = pd.DataFrame({name: r.equity_series() for name, r in results.items()})
    equity_curves.to_csv(output_dir / "equity_curves.csv")

    for name, result in results.items():
        rebalance_path = output_dir / f"{name}_rebalances.json"
        rebalance_path.write_text(
            json.dumps([r.model_dump(mode="json") for r in result.rebalances], indent=2, default=str)
        )

    first_config = next(iter(results.values())).config_snapshot
    report_md = render_markdown_report(results, table, first_config)
    report_path = output_dir / "report.md"
    report_path.write_text(report_md)

    return report_path
