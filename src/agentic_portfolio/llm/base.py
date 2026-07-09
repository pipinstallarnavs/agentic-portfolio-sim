"""Minimal model client interface shared by the live adapter and offline test doubles."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(ABC):
    """Minimal chat-completion interface. Phase 4 adds structured-output
    parsing and tool-calling on top of this, not a change to it.
    """

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse: ...
