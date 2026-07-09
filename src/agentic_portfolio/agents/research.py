"""Research Agent.

Per the spec, the Research Agent summarizes market/company context available
at simulation time t and returns structured JSON — never free-form prose a
downstream agent would have to parse.

This MVP has no news, filings, or fundamentals data source, so
``MockResearchAgent`` is honestly limited to *price-derived* context: trend
direction, position relative to the trailing high/low range, and performance
relative to the benchmark. It is a stand-in for what a real (Phase 4)
LLM-backed Research Agent would produce from actual text sources — it is
deterministic and rule-based specifically so the rest of the pipeline can be
built and tested without an LLM or a research data subscription.
"""

from __future__ import annotations

import math

from agentic_portfolio.agents.base import Agent, AgentContext, AgentResult
from agentic_portfolio.simulation.state import MarketState
from agentic_portfolio.tools.features import calculate_momentum

TREND_THRESHOLD = 0.02  # +/-2% over the trend window to call it a real trend, not noise
RELATIVE_THRESHOLD = 0.02  # +/-2% vs benchmark over the same window


class ResearchResult(AgentResult):
    key_points: dict[str, list[str]] = {}
    context_tags: dict[str, list[str]] = {}


class MockResearchAgent(Agent):
    name = "research_agent"

    def __init__(self, trend_window: int = 20, range_window: int = 252):
        self.trend_window = trend_window
        self.range_window = range_window

    def run(self, state: MarketState, context: AgentContext, **kwargs) -> ResearchResult:
        benchmark_prices = state.prices(state.benchmark)
        benchmark_trend = calculate_momentum(benchmark_prices, self.trend_window)

        key_points: dict[str, list[str]] = {}
        context_tags: dict[str, list[str]] = {}

        for ticker in state.universe:
            prices = state.prices(ticker)
            points: list[str] = []
            tags: list[str] = []

            trend = calculate_momentum(prices, self.trend_window)
            if not math.isnan(trend):
                if trend > TREND_THRESHOLD:
                    tags.append("positive_trend")
                    points.append(f"{ticker} is up {trend:.1%} over the trailing {self.trend_window}d.")
                elif trend < -TREND_THRESHOLD:
                    tags.append("negative_trend")
                    points.append(f"{ticker} is down {trend:.1%} over the trailing {self.trend_window}d.")
                else:
                    tags.append("flat_trend")
                    points.append(f"{ticker} is roughly flat ({trend:+.1%}) over the trailing {self.trend_window}d.")

            window = prices.iloc[-self.range_window:] if len(prices) >= 2 else prices
            if len(window) >= 2:
                high, low, last = window.max(), window.min(), window.iloc[-1]
                span = high - low
                position = (last - low) / span if span > 0 else 0.5
                if position >= 0.9:
                    tags.append("near_range_high")
                    points.append(f"{ticker} is near its trailing {len(window)}d high.")
                elif position <= 0.1:
                    tags.append("near_range_low")
                    points.append(f"{ticker} is near its trailing {len(window)}d low.")

            if not math.isnan(trend) and not math.isnan(benchmark_trend):
                relative = trend - benchmark_trend
                if relative > RELATIVE_THRESHOLD:
                    tags.append("outperforming_benchmark")
                    points.append(
                        f"{ticker} is outperforming {state.benchmark} by {relative:.1%} over {self.trend_window}d."
                    )
                elif relative < -RELATIVE_THRESHOLD:
                    tags.append("underperforming_benchmark")
                    points.append(
                        f"{ticker} is underperforming {state.benchmark} by {abs(relative):.1%} over {self.trend_window}d."
                    )

            key_points[ticker] = points
            context_tags[ticker] = tags

        return ResearchResult(
            agent_name=self.name,
            rationale=f"Reviewed price-derived context for {len(state.universe)} names as of {state.as_of.date()}.",
            confidence=1.0,
            key_points=key_points,
            context_tags=context_tags,
        )
