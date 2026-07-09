"""LLM factory: mock for fixtures, Anthropic for explicitly requested live calls. Missing credentials fail; no automatic fallback."""

from __future__ import annotations

import os

from agentic_portfolio.llm.base import LLMClient
from agentic_portfolio.llm.mock import MockLLMClient


def get_llm_client(provider: str | None = None) -> LLMClient:
    kind = provider or os.environ.get("LLM_PROVIDER", "mock")
    if kind == "mock":
        return MockLLMClient()
    if kind == "anthropic":
        from agentic_portfolio.llm.anthropic_client import AnthropicLLMClient
        return AnthropicLLMClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {kind!r}")
