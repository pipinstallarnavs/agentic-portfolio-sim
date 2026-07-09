# Interview Prep: agentic-portfolio-sim

A Q&A walkthrough of this project, organized so you can read it top to bottom
in about 3 hours and be able to explain (and defend) every design decision.
Each section names the file to have open while you read it.

---

## 1. The pitch (2 min, no file needed)

**Q: What is this project, in one breath?**
A walk-forward backtester for a multi-agent LLM portfolio system, built so I can
measure — not assume — whether coordinating agents (research/quant/portfolio/risk)
beat a plain deterministic rule, under identical data, dates, and constraints.

**Q: Why would anyone care about this over just building a trading bot?**
Because the interesting question isn't "can an LLM propose portfolio weights" —
it obviously can. It's whether the *architecture* (splitting reasoning from
arithmetic, adding a risk-review step) earns its complexity. That's an
evaluation problem, not a demo problem, so most of the engineering effort here
is in the evaluation harness, not the agents.

**Q: What stage is it at?**
Phase 1+2+3: data layer, deterministic tools, the simulator, two rule-based
baselines, and a full four-agent pipeline (Research/Quant/Portfolio/Risk,
orchestrated) all run end-to-end and are tested. Every agent is currently
deterministic/mock — no real LLM call happens yet, that's Phase 4.

---

## 2. Lookahead bias (the load-bearing concept — 30 min)
*Files: `simulation/state.py`, `simulation/engine.py`, `tests/test_no_lookahead.py`*

**Q: What is lookahead bias, concretely, in a backtester?**
Any point where a decision made "as of" date t is allowed to see data from
after t. It's the single most common way a backtest silently lies to you —
your Sharpe ratio looks great because the strategy is quietly using tomorrow's
close to trade today.

**Q: Where in this codebase is lookahead actually prevented, and why there
specifically?**
`build_market_state(panel, as_of, ...)` in `simulation/state.py` — it slices
the full price panel to `index <= as_of` and returns an immutable `MarketState`.
That's the *only* place in the whole codebase that does this slicing. Every
tool function, every strategy, every (future) agent only ever receives a
`MarketState`, never the raw panel. So there's exactly one boundary to get
right and test, instead of trusting every consumer to slice correctly itself.

**Q: Doesn't loading the *entire* price history into memory upfront risk
leaking future data?**
No — loading data isn't the leak; *handing it to a decision* is. The full
panel sits in memory the whole run (that's normal and fine for a backtest),
but the engine only ever calls `build_market_state` with the current
decision date before invoking the strategy. Nothing downstream of that call
holds a reference to the unsliced panel.

**Q: Walk me through the timing convention end to end.**
1. On decision date t, `build_market_state` hands the strategy data through
   t's close only.
2. The proposal is *not* applied on t — it's stored as `pending`.
3. The loop advances to t+1; only then does `execute_rebalance` run, using
   t+1's close prices.
4. t+1's mark-to-market value already reflects the new holdings, so returns
   from t+1 onward belong to the new weights.
This mirrors reality: you can't get a fill at a close price before that close
has actually happened.

**Q: How did you *test* that this actually holds, not just document it?**
`test_no_lookahead.py` does three things: (1) asserts `build_market_state`
never returns a row past the cutoff; (2) builds a synthetic price series with
a deliberate future price shock and shows momentum computed *before* the shock
is unaffected — and contrasts that with the same calculation on the unsliced
series, which *does* see it, proving the slicing is what's doing the work, not
coincidence; (3) runs a full simulation with a "recording" strategy that
snapshots every `MarketState` it's given, then asserts none of them contain a
row after their own decision date, and that every trade's price matches t+1,
never t.

**Q: What's a subtle way lookahead could sneak back in if someone refactored
this carelessly?**
Passing the full panel (instead of a `MarketState`) into a new tool function
"for convenience," or computing a rolling feature over the whole panel once
outside the loop and then slicing the *result* — the second one is wrong
because e.g. a volatility calculation with a centered or backward-looking-but-
cached window can still implicitly depend on how the whole series was framed.
Always slice the input, not the output.

---

## 3. Adjusted vs. unadjusted prices (10 min)
*Files: `data/provider.py`, `data/cache.py`*

**Q: Why does the cache store both `close` and `adj_close`?**
`adj_close` is split/dividend-adjusted and is the only thing used for return
and feature math — using raw `close` across a stock split would show a fake
-50% "crash" that never happened to an actual holder. `close` is kept around
for reference/display but is deliberately never fed into `calculate_returns`
or anything downstream of it.

**Q: What would go wrong if you accidentally used `close` for momentum?**
A stock that did a 4:1 split would show as -75% momentum on the split date
even though nothing happened to the position's value — the ranking-based
momentum baseline would systematically avoid recently-split stocks for no
economic reason.

---

## 4. Data provider abstraction & caching (10 min)
*Files: `data/provider.py`, `data/cache.py`, `scripts/download_data.py`*

**Q: Why is there an abstract `DataProvider` instead of just calling yfinance
directly in the simulator?**
So the simulator never depends on *how* data arrives. `YFinanceDataProvider`
downloads-then-caches; `LocalCacheDataProvider` reads only from disk and
raises if something's missing, guaranteeing a run can't silently hit the
network. Swapping to a paid data vendor later means writing one new class, not
touching the engine.

**Q: How do you avoid depending on live API availability once data is cached?**
`scripts/run_simulation.py --provider local` forces the read-only cache
provider — any cache miss is a loud error, never a silent network call. This
is also what makes `test_no_lookahead.py`'s simulation tests fully offline and
deterministic (they use an in-memory fake provider, not even the cache).

**Q: What's the fallback if I have no network access at all, e.g. in this
sandbox?**
`scripts/download_data.py --synthetic` generates deterministic (seeded) GBM
price series per ticker and writes them through the same cache path — so the
rest of the pipeline can't tell the difference. It's clearly logged as
synthetic so no one mistakes it for real backtest results.

---

## 5. The deterministic tool layer — why agents don't do arithmetic (20 min)
*Files: `tools/features.py`, `tools/portfolio.py`, `tools/risk.py`*

**Q: Why is it a hard rule that agents never compute numbers themselves?**
LLMs are unreliable at arithmetic and, more importantly, non-reproducible —
the same prompt can yield a slightly different Sharpe ratio calculation twice.
If a number matters for a trading decision or an evaluation metric, it must
come from a pure, unit-tested Python function, not from tokens the model
generated. The agent's job is judgment over numbers, not producing them.

**Q: Give me a concrete example of a "pure, tested" function here.**
`calculate_historical_var(returns, confidence)` — empirical VaR via
`np.percentile`. It's deterministic, has no side effects, and is tested
against a known distribution (`test_calculate_historical_var_known_distribution`)
by asserting it matches `np.percentile` directly.

**Q: Walk me through one non-trivial calculation and a real bug you hit.**
`calculate_turnover` inside `execute_rebalance` — first pass computed turnover
using only ticker weights, and a test (`test_execute_rebalance_from_all_cash`)
caught that a move from 100% cash to 100% invested was reporting 0.5 turnover
instead of 1.0. The bug: cash wasn't included as an explicit leg in the
before/after weight dicts passed to `calculate_turnover`, so the "sell cash,
buy equities" side of the trade was invisible to the sum-of-absolute-changes
formula. Fix: add an explicit `_CASH_` weight to both sides before calling it.
This is a good example of why deterministic ≠ obviously-correct — it still
needs tests.

**Q: How do you annualize volatility, and why 252?**
`std(daily_returns) * sqrt(252)` — 252 is the standard US trading-day count
per year. It's defined once (`TRADING_DAYS_PER_YEAR` in `features.py`) and
reused everywhere so annualized numbers across different metrics are
comparable to each other.

---

## 6. Portfolio construction & constraints (15 min)
*Files: `baselines/momentum.py`, `tools/portfolio.py::cap_and_renormalize_weights`, `tools/portfolio.py::validate_weights`*

**Q: How does the momentum baseline turn a ranking into weights?**
Rank the universe by trailing N-day momentum, take the top 5, weight each
inversely to its trailing realized volatility (lower-vol names get more
weight), then cap any name at 20% and redistribute the excess proportionally
among the uncapped names until gross exposure hits target.

**Q: What if capping alone can't reach full exposure — say only 3 names
survive and the cap is 20%?**
`cap_and_renormalize_weights` handles that explicitly: if `n_capped * cap`
is still below the target gross exposure, every capped name is set to exactly
the cap and the remainder is left as cash rather than looping forever or
silently violating the constraint.

**Q: `validate_weights` doesn't fix anything — why not?**
Separation of concerns, on purpose. Validation reports violations; fixing a
violation is a *decision* (sell down the largest position? cut everything
pro-rata? reject the whole proposal?) — that's the Risk Agent's job in Phase
3+, and it needs to be auditable ("modified X because Y"), which a silent
auto-fix inside a generic tool function can't provide.

**Q: Why enforce constraints in the baseline construction *and* re-validate
in the engine?**
Defense in depth, and because Phase 3+ strategies (LLM-driven) won't
necessarily respect constraints by construction the way a hand-written
baseline does — the engine-level validation is the thing that has to catch
that later, so it's built and tested against the baselines now while the
ground truth is easy to reason about.

---

## 7. The simulation engine's control flow (20 min)
*File: `simulation/engine.py`*

**Q: Why is there a `pending` variable instead of executing a proposal
immediately?**
It's the direct implementation of the t → t+1 timing rule: a proposal
computed on day t must not affect that day's portfolio. Storing it as
`pending` and only consuming it when the loop reaches the *next* iteration
is what turns the timing convention from a comment into an enforced
control-flow property.

**Q: How are rebalance dates chosen for "weekly"?**
`get_rebalance_dates` resamples the trading calendar to `W-FRI` and takes the
*last available trading day* in each week — so a holiday-shortened week
(Friday closed) still gets a real rebalance on Thursday instead of silently
skipping that week or erroring.

**Q: Why compute the daily equity curve inside the same loop instead of
after the fact?**
Because share counts change only at execution days; marking to market once
per trading day with whatever `shares`/`cash` are current at that point in
the loop is the simplest way to get a correct daily curve without a second
pass, and it keeps the "what did the portfolio hold on date X" logic in one
place.

**Q: What would you need to change to support a Portfolio Agent instead of
`MomentumBaseline`?**
Nothing in `engine.py`. Both implement the same `Strategy` protocol
(`generate_weights(state) -> PortfolioProposal`) — the engine only calls that
one method. This was a deliberate design constraint from the start, not a
refactor discovered later.

---

## 8. Evaluation & baselines (15 min)
*Files: `evaluation/metrics.py`, `evaluation/report.py`, `baselines/`*

**Q: Why compare against equal-weight *and* SPY buy-and-hold, not just SPY?**
Equal-weight isolates "does the momentum *signal* add anything beyond just
holding the universe," while SPY isolates "does this concentrated 10-name
universe beat the market at all." Without both, a momentum baseline beating
SPY could just be a universe-selection effect (mega-cap tech tilt), not a
signal effect.

**Q: What did the first real run actually show, and is that expected?**
Momentum ~25.3% annualized, equal-weight ~24.5%, SPY buy-and-hold ~14.6%
(2018–2026 cached data). Both beat SPY meaningfully — expected, since the
universe is 10 large/mega-cap names including several of the strongest
performers of that period, so this is largely a universe-selection effect,
not proof the momentum signal itself is strong. Momentum's mean turnover
(~14.5%) is roughly 10x equal-weight's (~1.5%), which is exactly what you'd
expect — equal-weight only trades to correct price drift between rebalances,
while momentum actively rotates its top-5 holdings.

**Q: Sharpe for momentum (1.12) was actually *lower* than equal-weight
(1.14) despite higher returns — is that a bug?**
No — momentum carries higher volatility and, critically, ~13x the transaction
costs (~$434k vs ~$33k on $1M initial capital) from weekly turnover, which
drags risk-adjusted return down even though raw return is higher. This is
exactly the kind of result the project is designed to surface, not something
to explain away.

**Q: Why compute metrics from the equity curve rather than storing them
per-strategy inside the simulator?**
Keeps `simulation/engine.py` from knowing anything about evaluation — it
only produces a `SimulationResult` (equity curve + rebalance audit trail).
`evaluation/metrics.py` is a separate, swappable layer that could add new
metrics without touching the simulator at all.

**Q: Once the mock multi-agent system was added, how did it actually
compare?**
19.79% annualized return, Sharpe 0.99, max drawdown 28.62% — worse raw return
and worse Sharpe than both deterministic baselines (momentum 25.3%/1.12,
equal-weight 24.5%/1.14), but it has the *smallest* max drawdown of every
strategy tested, including SPY (33.7%). That's the Risk Agent's
turnover/volatility/VaR scaling visibly doing its job — trading some upside
for better tail behavior — and it's a genuinely useful result, not a
disappointing one: it's exactly the kind of "multi-agent isn't strictly
better" finding the evaluation harness was built to be capable of producing.

---

## 9. The agent pipeline (Phase 3 — 30 min)
*Files: `agents/base.py`, `agents/research.py`, `agents/quant.py`,
`agents/portfolio.py`, `agents/risk.py`, `agents/orchestrator.py`,
`tests/test_agents.py`*

**Q: Why does `AgentContext` hold portfolio state and constraints, but agent-
to-agent handoffs (Research → Portfolio, Portfolio → Risk) are explicit
function arguments instead of also going through it?**
Deliberate: `AgentContext` is genuinely *shared, read-only* context every
agent needs regardless of role (current holdings, the limits it must
respect). Which agent produced which result and who consumes it is a
data-flow question, and burying that in a mutable shared dict you write into
and read back out of makes the flow something you have to trace at runtime.
Making it an explicit typed parameter (`portfolio_agent.run(state, context,
research=..., quant=...)`) makes the pipeline readable from the function
signature alone — you can see the whole dependency graph in
`orchestrator.py` without running anything.

**Q: The `Strategy` protocol changed from `generate_weights(state)` to
`generate_weights(state, portfolio_state)` to support this — didn't that
risk exactly the "rewrite the simulator" the architecture was supposed to
avoid?**
It touched `engine.py`, but not the invariant that mattered: the walk-forward
loop, the t/t+1 timing, and `build_market_state`'s lookahead boundary are all
untouched. What changed is one method signature gaining a parameter, plus a
few lines in the engine to compute a `PortfolioSnapshot` from data it already
had (`shares`/`cash` at the current decision-day prices) before calling the
strategy. Every existing baseline and test needed a one-line signature update.
That's the right kind of change to make when a real requirement (Portfolio
Agent needs current holdings) surfaces — the alternative (having
`MultiAgentStrategy` silently self-track weights, ignoring risk-agent
modifications and price drift) would have been *more* code and *less*
correct, just to avoid touching `engine.py` on principle.

**Q: Why is the Research Agent explicitly documented as *not* doing real
research?**
Because it isn't, and pretending otherwise would be actively misleading.
There's no news, filings, or fundamentals data source in this MVP —
`MockResearchAgent` computes trend direction, position in the trailing
high/low range, and performance relative to SPY, all from the same price
panel the Quant Agent already sees. It's a legitimate placeholder for what a
real, text-grounded LLM research agent would produce in Phase 4, and it's
honestly labeled as a limitation in the README, not glossed over.

**Q: The Quant Agent's signal score is `momentum_60d - 0.5 * volatility_60d`
— why that formula and not something more sophisticated?**
Because it's the simplest thing that's still explainable in one line, and
sophistication without evidence it helps is exactly the kind of
over-engineering this project is trying to avoid pre-evaluation. The whole
point of Phase 8/9 is to measure whether *any* of this machinery earns its
complexity — starting from something more elaborate would make it harder to
tell whether a result is coming from the signal or from the architecture.

**Q: Walk through the Risk Agent's check ordering and why it's ordered that
way.**
Confidence floor first (cheapest check, and if it fails nothing else
matters — reject outright, hold the prior portfolio). Then hard constraints
(long-only/position cap/gross exposure) via the same `cap_and_renormalize_weights`
the momentum baseline uses — these are non-negotiable, so they're fixed before
anything else is evaluated on top of them. Then turnover, then ex-ante
volatility, then historical VaR — each of the last three scales the *already
partially-fixed* weights down further if needed, so a proposal that breaches
multiple limits gets progressively reined in rather than having each check
overwrite the others' work.

**Q: You mentioned finding a turnover bug in `execution.py` earlier — did the
same bug show up again in the Risk Agent?**
Yes, and that's worth being able to say out loud: the Risk Agent's own
turnover check initially computed `calculate_turnover(prior_weights, weights)`
without an explicit cash leg, which would have under-counted turnover on any
rebalance where the cash weight changed materially (e.g. de-risking into
cash) — silently letting a de-risking move through a turnover limit meant to
catch large trades. I caught this by pattern-matching against the earlier,
already-fixed bug rather than by a failing test, and added the `_CASH_` leg
the same way `execution.py` does. It's a good example of a fix in one place
implying a targeted check everywhere else the same primitive is reused.

**Q: What was the other real issue you hit building the Risk Agent?**
A floating-point boundary false-positive: the very first `turnover >
constraints.max_turnover_per_rebalance` check used a bare `>` comparison, and
against the default (effectively unconstrained) limit of 1.00, a turnover
computed as 1.0000017 due to ordinary floating-point accumulation tripped
the check and logged a "modification" that was actually a no-op (scaling by
99.9998%). Fixed by adding a small epsilon (`+ 1e-4`) to all three
threshold comparisons (turnover, volatility, VaR) in the Risk Agent — the
same category of fix as `validate_weights`' `tolerance` parameter, applied
somewhere it had been missed. It didn't change any actual result, but it
kept the audit trail (which the whole Risk Agent evaluation question depends
on being trustworthy) honest about what actually triggered a modification.

**Q: Why is `MultiAgentStrategy` built with injectable agents
(`research_agent=None, quant_agent=None, ...` defaulting to the mock
implementations) instead of hardcoding the four mock classes?**
So the Phase 9 ablations (no Risk Agent, no Research Agent, single agent)
are a matter of passing in a no-op/passthrough agent at construction time —
e.g. a `PassthroughRiskAgent` that always approves unmodified — with zero
changes to the orchestrator or the engine. This was designed in now,
specifically so ablations don't require new orchestration code later, only
new small agent classes.

---

## 10. Architecture decisions you should be ready to defend

**Q: Why no LangChain/CrewAI/AutoGen?**
The orchestration need here is one interface with one method
(`generate_weights`). A framework adds abstraction, dependency surface, and
debugging indirection to solve a coordination problem this project doesn't
actually have yet. If Phase 3+ orchestration turns out to need real dynamic
tool-calling loops, retries, and multi-turn agent state, that's a decision to
revisit *then*, with evidence, not up front on speculation.

**Q: Why Pydantic for state/results but plain dataclasses for `MarketState`?**
`MarketState` carries pandas DataFrames and is purely internal/in-process —
it's never serialized, so Pydantic's validation/JSON machinery would be pure
overhead. `PortfolioProposal`, `RebalanceRecord`, `SimulationResult` etc. are
either agent I/O (must be structured and validated, not free-form prose) or
get written to disk as JSON — that's exactly what Pydantic is for.

**Q: Why was `agents/` built before `llm/` had a real implementation?**
Because none of the Phase 3 agents needed one — "mock agents" per the spec
means deterministic rule-based logic standing in for what an LLM will do
later, not a fake LLM client with canned responses. Building the
deterministic foundation (data, tools, engine, baselines) *and* the full
mock agent pipeline first, before any real LLM integration, meant every
piece of the orchestration — the `Strategy` protocol change, the
`AgentContext` design, the Risk Agent's decision logic — could be built and
tested without also debugging LLM non-determinism, API failures, or prompt
engineering at the same time. `llm/` still only has the `LLMClient`
interface and a mock client for now; Phase 4 is where it becomes load-bearing.

**Q: What's the biggest current limitation you'd flag to a reviewer
unprompted?**
That the entire premise of the project — does a real LLM-reasoning
multi-agent system beat a single LLM prompt, and does either beat a plain
rule — isn't answerable yet, because no agent in the system does real LLM
reasoning. What exists now is the full measurement harness plus a
deterministic stand-in multi-agent system, which is genuinely useful (it
already produced a real result: better drawdown, worse Sharpe than the
baselines) but is not the experiment the project is ultimately for. I'd
frame it exactly that way rather than implying more progress than there is.

---

## 11. Rapid-fire (last 15 min, do these without looking anything up)

- What does `adj_close` correct for? *Splits and dividends.*
- What's the annualization factor and why? *252 — US trading days/year.*
- What's turnover of 1.0 mean? *A full one-way portfolio replacement (e.g. 100% cash → 100% invested, or completely swapping every holding).*
- Where does transaction cost get subtracted from portfolio value? *`execute_rebalance`: `portfolio_value_after = portfolio_value_before - transaction_cost`.*
- What's the one function you'd point to as "this is where lookahead bias would be introduced if someone got it wrong"? *`build_market_state`.*
- Why is `validate_weights` non-mutating? *Separation between reporting a violation and deciding how to fix it — the latter needs to be an auditable decision, not a side effect.*
- What existing test would fail first if someone reintroduced the turnover-cash-leg bug? *`test_execute_rebalance_from_all_cash` / `test_execute_rebalance_partial_liquidation` in `test_execution.py`.*
- Which agent never involves LLM judgment even in the full future system? *The Quant Agent — computing momentum/vol/beta isn't a reasoning task.*
- What does the Risk Agent do if the Portfolio Agent's confidence is too low? *Rejects the proposal outright and holds the prior portfolio, before any other check runs.*
- In what order does the Risk Agent apply its checks, and why that order? *Confidence floor, then hard constraints (cap+renormalize), then turnover, then volatility, then VaR — cheapest/most decisive check first, each subsequent check scales down whatever the previous one already fixed.*
- What's the single biggest thing left to build? *Real LLM calls behind `LLMClient` (Phase 4) and the single-agent baseline (Phase 5) — everything up to now, including the full mock agent pipeline, is the harness they'll be measured inside.*
