# LLM portfolio workflow

The optional extension adds real Anthropic Messages API code, a research/model-proposal sequence, deterministic risk feedback, at most one revision, and a single-agent baseline. It was implemented after the original resume audit. **Verification so far is offline: there is no successful live API run or live performance result.** No account origin or expired-key history is assumed.

## Start offline

```bash
.venv/bin/python scripts/run_llm_comparison.py
.venv/bin/python -m pytest -q
```

The demo generates synthetic prices in memory, uses a scripted `ReplayClient`, and saves `results/llm-replay.json`. The first scripted proposal intentionally exceeds a position cap; feedback requests a revision; the next scripted proposal fixes it. This tests the mechanism. The report says `offline-replay` and every response identifies its model as `offline-replay`.

The default 75 trading days include feature warmup and one actual rebalance. Its four-way comparison is a workflow smoke test, not a meaningful investment-performance experiment. `--cached` uses the existing price cache, with provenance explicitly marked unverified. `--days` can extend the window, but the shared call budget still stops a run before excess requests.

## Flow to learn in about 45 minutes

1. **10 minutes: adapter and inputs.** Read `llm/anthropic_client.py` and `StructuredCalls` in `agents/llm_strategy.py` under `src/agentic_portfolio`. The API adapter passes a system prompt and JSON payload to the provider and captures model/token usage. It has a 30-second timeout, no hidden retries and a capped response length. Model ID is explicit configuration.
2. **10 minutes: decisions.** Python computes numerical features. In multi-agent mode one model request interprets those features as research context; a second proposes weights. The single-agent arm skips the research request but receives the same numerical state, holdings and limits. Both use deterministic risk review and the same revision allowance. Research is price-derived: no news or filings are fetched.
3. **10 minutes: risk and revision.** Model JSON must match a schema, have finite long-only weights and contain only allowed tickers. Basic position/cash violations generate feedback. The existing RiskAgent computes volatility/VaR/turnover and may modify or reject the proposal. If needed, the portfolio model sees its previous proposal plus the objections and may revise once. A valid final risk adjustment can be used; unresolved hard-invalid proposals fail. Schema errors/unknown tickers abort rather than being silently converted into trades.
4. **10 minutes: evidence.** Open `results/llm-replay.json`. Trace the first proposed weight, risk objection, revised weight and next-day fill. Inspect both arms' identical feature payloads. Logs include timestamps, actual model identifiers, prompts/responses, usage, latency and review rounds. Failed comparisons retain completed call records and an error type.
5. **5 minutes: your change.** Instantiate the strategy with `max_revisions=0`, rerun the deliberately invalid replay, and explain why the run stops. Restore one revision. Predict why multi-agent mode needs one more call than the single-agent baseline.

## Run with a valid API account

Install the existing optional dependency:

```bash
.venv/bin/python -m pip install -e '.[llm]'
```

Set `ANTHROPIC_API_KEY` and `LLM_MODEL` in your local environment or untracked `.env`. Choose a model your account can access. Do not paste keys into interview notes or the repository. The `.env.example` is a template; it does not contain working credentials.

```bash
.venv/bin/python scripts/run_llm_comparison.py --live --max-calls 12
```

This command makes paid API calls, uses synthetic prices by default and saves `results/llm-live.json`. There are no trades sent to a broker. Missing credentials fail; the program does not pretend replay is a live run. The adapter follows the provider's [official Python SDK documentation](https://platform.claude.com/docs/en/api/sdks/python).

A live run may fail JSON validation or propose an invalid allocation. Inspect the recorded failure and improve the request deliberately. Do not rename an offline file as a live result. The run logger is a reproducibility aid, not independently authenticated proof of history.

## Interview questions

- **What makes this genuinely model-backed?** In live mode, the adapter sends Messages API requests; returned text becomes validated research or weight objects. Verification currently covers mocked transport and replay only.
- **What makes the feedback loop real?** A revision request contains the previous proposal and deterministic objections, followed by a fresh risk review. There is an explicit one-revision limit.
- **Why leave quant/risk in Python?** Their numerical outputs must be reproducible and independently tested. The model interprets supplied information and proposes allocations.
- **Is the single-agent comparison fair?** Numeric inputs, constraints, timing and risk checks match; multi-agent has an extra research call. Compare costs and variability too. The short demo cannot establish superiority.
- **Can the model still know future events?** Yes: date-filtering supplied features does not erase training knowledge. Historical LLM backtests need additional controls; asking a model not to use remembered events is not proof that it complied.
- **What can you claim now?** Implementation of the extension and offline-tested orchestration. Successful live operation and model performance require actual recorded runs; implementing it now does not establish a prior professor-provided account or past experiment.

## Verification recorded

Portfolio suite: **72 tests passed**, including 12 new offline LLM tests. The end-to-end replay completed four strategy runs with five scripted calls and two proposal/review rounds in each model arm. No valid-key live request was attempted.
