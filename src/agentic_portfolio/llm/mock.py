"""Fixed-response offline test double. Responses from this client are not live model results."""

from __future__ import annotations

from agentic_portfolio.llm.base import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    def __init__(self, fixed_response: str = "{}"):
        self.fixed_response = fixed_response
        self.call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content=self.fixed_response, model="mock", input_tokens=0, output_tokens=0)
