"""Small real Messages API adapter. No silent fallback to mock responses."""
import os
from agentic_portfolio.llm.base import LLMClient, LLMResponse


class AnthropicLLMClient(LLMClient):
    def __init__(self, model=None, sdk_client=None):
        self.model = model or os.environ.get('LLM_MODEL')
        if not self.model:
            raise ValueError('Set LLM_MODEL to a model available on your API account')
        if sdk_client is None:
            key = os.environ.get('ANTHROPIC_API_KEY')
            if not key:
                raise ValueError('Set ANTHROPIC_API_KEY locally; do not put it in code or reports')
            from anthropic import Anthropic
            # Disable hidden retries so the run's explicit call budget is exact.
            sdk_client = Anthropic(api_key=key, timeout=30., max_retries=0)
        self.sdk = sdk_client

    def complete(self, system_prompt, user_prompt):
        message = self.sdk.messages.create(
            model=self.model, max_tokens=1500, temperature=0,
            system=system_prompt,
            messages=[{'role': 'user', 'content': user_prompt}],
        )
        if message.stop_reason != 'end_turn':
            raise RuntimeError('Model did not finish its JSON response; no portfolio accepted')
        content = ''.join(block.text for block in message.content if block.type == 'text')
        return LLMResponse(content=content, model=message.model,
                           input_tokens=message.usage.input_tokens,
                           output_tokens=message.usage.output_tokens)
