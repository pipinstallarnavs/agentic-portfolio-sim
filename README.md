![Agentic Portfolio Simulator banner](assets/banner.svg)

# Portfolio research and walk-forward simulation

**LLM extension added:** [workflow and verification status](LLM_WORKFLOW.md). The original deterministic demo below is preserved; statements about its lack of model calls describe that core mode. The extension has live-call code, a bounded revision loop and a single-agent baseline, tested offline; no successful live run has been recorded.

A deterministic research harness with research, quant, portfolio and risk stages. It compares the resulting portfolios with momentum and equal-weight baselines, using a shared dated simulator and transaction-cost model.

Start with [the study guide](STUDY_GUIDE.md) and the heavily commented `learn.py`. The guide covers fee arithmetic, timing examples, validation boundaries, and modification exercises for the complete simulation path.

## Run

```bash
.venv/bin/python learn.py
.venv/bin/python learn.py --risk-vol .10
.venv/bin/python learn.py --cost-bps 25
.venv/bin/python -m pytest -q
# Optional cached run; no network calls:
.venv/bin/python scripts/run_simulation.py --provider local --output-dir results/verified
```

Fresh setup: Python 3.12, an environment named `.venv`, and `pip install -e '.[dev]'`.

The learning demo generates prices in memory, traces one decision and saves `results/learning_report.json`. It does not replace the local data cache. `learn.py --cached` requires that cache to be populated separately; the cache is excluded from Git, and a cached file alone does not establish whether it originated from a provider or the synthetic fallback.

## How the code fits together

- `tools/`: numerical features, weight handling and risk calculations.
- `agents/`: rule-based research tags, computed quant signals, portfolio proposal and one-pass risk review.
- `simulation/state.py`: restricts visible prices to the decision date.
- `simulation/engine.py`: stores the decision and executes it at the next trading day's close.
- `simulation/execution.py`: computes holdings and cash after fees, refusing missing held prices and invalid targets.
- `evaluation/`: reports completed backtests; `config/default.yaml` sets the universe and experiment.
- `llm/`: a mock client for offline runs and an optional Anthropic adapter. The extension supports live model calls and a bounded revision loop, but has no recorded successful live run or performance comparison.

## Execution and validation

Old holdings earn the return through the execution close at t+1; new holdings earn subsequent returns. Weights apply to after-fee wealth, found with a scalar budget equation, so full investment no longer requires negative cash to pay fees. Basic invalid position/gross targets fail before execution. Risk logs recompute volatility, VaR and turnover from final weights.

The tests cover future-data exclusion, next-day prices, deterministic feature/risk calculations, agent decisions, cost conservation, no borrowing on fully invested trades, invalid prices/weights and refusal of over-limit proposals. Risk reduction can override the turnover budget, and this priority is logged. The system is not a joint constrained optimizer.

## Interpret results carefully

All three learning strategies share prices, dates, position caps, execution and costs. The risk overlays differ, so performance differences are not solely attributable to splitting code into agents. There is no single-LLM baseline. Five names at a 20% cap can force equal weights despite the initial inverse-volatility allocation.

The learning comparison removes shared cash-only warmup while retaining the first fill's fee. The inherited large-run report retains warmup and compares SPY buy-and-hold invested from the panel start; it is a descriptive benchmark, not a timing-matched attribution experiment. Earlier saved performance numbers predate the fee correction. Use newly generated reports with their conventions, not stale README numbers.

Sharpe uses a zero risk-free rate and cash earns zero. The inherited Sortino calculation uses the standard deviation of negative observations, rather than standard target downside deviation; do not cite it as standard Sortino. Transaction costs are flat bps with no market impact, partial fills or financing. The universe is small and retrospectively selected. Adjusted-close units are a total-return approximation, not a live trade ledger. Current execution is long-only/unlevered even though the configuration contains a long-only flag.

A live LLM extension would need dated text sources, validated structured responses, error handling, cost/latency accounting and matched ablations. It is intentionally outside this short learning path.
