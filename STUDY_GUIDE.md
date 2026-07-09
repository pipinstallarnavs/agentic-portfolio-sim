# Portfolio research pipeline: a 3½-hour working session

**LLM extension added:** [workflow and verification status](LLM_WORKFLOW.md). The original deterministic demo below is preserved; statements about its lack of model calls describe that core mode. The extension has live-call code, a bounded revision loop and a single-agent baseline, tested offline; no successful live run has been recorded.

**Goal:** trace a portfolio decision, explain why it executes tomorrow, and compare the same trading rules with costs and risk controls. The current agents are Python rules. There is no live LLM, qualitative news research, single-LLM baseline or iterative agent debate.

## Run and follow one decision — 0–20 minutes

From this project folder:

```bash
.venv/bin/python learn.py
.venv/bin/python -m pytest -q
```

On a fresh machine, create a Python 3.12 environment named `.venv`, then install `-e '.[dev]'`. The default lesson needs no network or API key. It creates a synthetic price panel in memory and preserves the existing cache. `--cached` uses cached adjusted closes, but the cache alone does not establish market-data provenance.

Read `learn.py`. It prints a date, scores, proposed weights, reviewed weights and a decision reason. Then it runs equal weight, momentum and the four-agent pipeline through one simulator. The JSON result records inputs and a complete first-decision audit.

## Prices to signals — 20–55 minutes

Read `tools/features.py` and `agents/quant.py` under `src/agentic_portfolio/`.

A simple return is `P_today/P_yesterday - 1`. A 60-day momentum signal is `P_t/P_(t-60) - 1`. Realized volatility is the standard deviation of daily returns multiplied by sqrt(252), under the project's annualization convention. Portfolio volatility uses the full covariance matrix: `sqrt(w'Cov*w)*sqrt(252)`.

Inverse-volatility weighting gives lower-volatility names more weight before caps. It is not a full minimum-variance optimizer and does not equalize risk contributions when correlations differ. There is no need to add an optimizer to explain that tradeoff well.

The research agent uses price-derived tags, such as relative performance; the quant agent computes numerical features; the portfolio agent combines scores and tags. Four classes can provide useful separation of responsibilities even without language models.

**Do:** find one ticker's 60-day momentum in the printed scores/feature trace and compute it from the two endpoint prices. Explain why adjusted closes are used. Cash earns zero in this implementation.

## Risk review — 55–90 minutes

Read `agents/orchestrator.py`, `agents/portfolio.py` and `agents/risk.py`.

The sequence is research → quant → portfolio → risk. Shared context includes holdings and limits; outputs are passed explicitly between functions. This is a single pass, not a conversation in which agents repeatedly debate and revise.

Risk can reject low-confidence proposals, repair a weight proposal, or scale exposure to reduce turnover, forecast volatility or historical VaR. A 40% forecast volatility with a 20% limit suggests scaling risky weights by .5 and moving the remainder to zero-volatility cash, assuming a fixed covariance matrix. The logged risk metrics should describe the **final** weights, not the discarded proposal.

Some limits conflict. Risk reduction here has priority over the turnover budget, and such an override is explicitly logged. It is not a joint constrained optimizer. Baselines share data, position caps, costs and timing, but do not all run the same risk overlays; this comparison cannot isolate an effect of agent architecture by itself.

**Do:** predict the effect of `--risk-vol .10` and rerun. Inspect the actual modified weights. With the default top five and a 20% position cap, selecting exactly five names can force equal weights; the inverse-vol allocation may be fully constrained away. That is a real design tradeoff to discuss.

## Avoid looking into the future — 90–125 minutes

Read `simulation/state.py::build_market_state` and `simulation/engine.py::run`.

At the close of t, compute signals from rows dated at or before t. Store the proposal. At the close of t+1, value the old holdings and execute the new weights. Old holdings earn the t→t+1 return. New holdings start earning returns after that close. Transaction fees are deducted on the fill day.

The no-lookahead tests inject a future shock and inspect the data each strategy sees. Do not assume that slicing eliminates every bias: a handpicked surviving universe still has selection/survivorship bias; retrospective model choices can overfit evaluation data; a present-day LLM may know historical outcomes even when its prompt is date-filtered.

**Do:** draw three dates and label which holdings earn each return. Point to the pending proposal and state cutoff in the code. Explain why executing immediately at a just-observed closing price is optimistic.

## Cash and fees — 125–155 minutes

Read `simulation/execution.py` and the new tests at the end of `tests/test_execution.py`.

The original executor allocated 100% of wealth and then subtracted fees from cash. That creates borrowing in a supposedly unlevered book. The correction defines target weights against **after-fee wealth W**:

`W + fee_rate * sum(abs(weight_i*W - current_value_i)) = current_wealth`

For 1,000 cash and a 10bp purchase fee, W=1000/1.001=999.001. You own that much stock and have zero cash; you do not own 1,000 of stock with negative .999 cash. For a full A→B rotation, charge both the sale and purchase. The scalar equation is monotone under this long-only, gross≤1 model, so simple bisection suffices.

Missing held prices now fail rather than silently value a holding at zero. Invalid weight proposals fail before execution. This execution model intentionally supports long-only, unlevered portfolios; a configuration switch does not implement financing, margin or stock borrowing.

**Do:** reproduce the all-cash fee example by hand. Explain turnover versus traded notional: a complete rotation has one-way turnover 100%, but charges costs on a sale and a purchase.

## Evaluate an experiment — 155–185 minutes

```bash
.venv/bin/python learn.py --cost-bps 0
.venv/bin/python learn.py --cost-bps 25
.venv/bin/python learn.py --risk-vol .10
```

Save each JSON elsewhere before the next run replaces it. Hold the seed fixed for controlled comparisons. The learning report excludes shared cash-only warmup from return statistics while retaining the day before the first fill so its fee counts. Compare return, drawdown, turnover and costs, not only Sharpe. Sharpe here assumes a zero risk-free rate.

The synthetic seed is a controlled experiment, not an estimate of real investment performance. Cached data needs source verification before using historical results in a resume. The existing large-run script reports a longer cached comparison, with additional benchmark/warmup caveats documented in `README.md`.

**Metric caveat:** the inherited Sortino implementation divides by the standard deviation of negative observations. That is not the usual target-downside-deviation definition. It is excluded from the core comparison to discuss; do not cite it as standard Sortino without correcting and validating that definition.

## Own one change — 185–210 minutes

Add a report counting risk approvals/modifications/rejections from `strategy.audit_log`, then print the first date volatility scaling occurred. Confirm that your counts sum to the number of reviews. This is small enough to implement yourself and makes the project demonstrably yours through understanding.

Give a two-minute explanation: measurable research question → deterministic features → dated proposal → risk review → next-day execution with fees → baselines → limitations. Do not describe mock agents as live LLM agents.

## Interview questions and answer outlines

1. **Why agents at all?** Explicit responsibilities and auditability. A comparison must establish whether the added machinery helps; four classes do not automatically create better returns.
2. **Which model did you call?** None in this implemented version. The LLM layer is an interface/mock. The runnable work is a deterministic research harness.
3. **What is actually shared?** Dated market state, holdings and constraints; typed outputs move through explicit function arguments.
4. **What is the risk agent's feedback loop?** It modifies a proposal in one pass. There is no iterative debate; that resume claim needs narrowing.
5. **What prevents lookahead?** State slicing and delayed execution, with future-shock and timing tests. These do not solve universe selection or historical knowledge in a future LLM.
6. **Why an equal-weight baseline?** It tests whether signal/weighting complexity helps beyond a simple diversified allocation under the same execution mechanics.
7. **Why can more agents underperform?** Added heuristics, cash drag, turnover, correlated signals and costs. Report tradeoffs rather than assume an improvement.
8. **Inverse-vol versus risk parity?** Inverse-vol ignores cross-correlations; risk parity uses marginal contributions from covariance.
9. **What happens to fees at 100% target exposure?** Solve for after-fee wealth first. Paying fees after spending all pre-fee wealth implicitly borrows.
10. **Which risk number belongs in the audit?** Recomputed risk of final weights, with proposed metrics separately labelled if retained. Stale pre-scaling metrics mislead reviewers.
11. **Are all constraints jointly enforced?** No. Basic invalid targets fail; risk scaling can override turnover. A true joint optimization would need feasibility handling and explicit priorities.
12. **What would a real LLM add?** Potential interpretation of dated text, using structured proposals checked by deterministic tools. Add provenance, schema validation, failure handling and fair ablations before claiming value.

**Ready-to-discuss check:** explain timing without the README, calculate the fee example, inspect a rejected/modified proposal, and write the intervention counter. These matter more than adding a model API just for the title.
