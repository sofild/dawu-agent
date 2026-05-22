"""Unified token counter supporting multiple providers."""

from __future__ import annotations

from dawu_agent.llm.base import ITokenCounter, Message


class UnifiedTokenCounter(ITokenCounter):
    """Token counter that automatically selects the right encoder per model.

    Falls back to character-count / 4 estimation when no specific encoder is available.
    """

    def __init__(self) -> None:
        self._encoders: dict[str, Any] = {}
        self._fallback_ratio = 4.0  # chars per token fallback

    def _get_encoder(self, model: str) -> Any:
        """Lazy-load encoder for the given model."""
        if model in self._encoders:
            return self._encoders[model]

        encoder = None
        model_lower = model.lower()

        # Try tiktoken for OpenAI models
        if any(prefix in model_lower for prefix in ["gpt-", "text-"]):
            try:
                import tiktoken

                encoder = tiktoken.encoding_for_model(model)
            except (ImportError, KeyError):
                pass

        # Try anthropic tokenizer
        if any(prefix in model_lower for prefix in ["claude-"]):
            try:
                import anthropic

                encoder = anthropic.Anthropic()
            except ImportError:
                pass

        self._encoders[model] = encoder
        return encoder

    def count(self, text: str, model: str) -> int:
        """Count tokens in text for given model."""
        encoder = self._get_encoder(model)

        if encoder is None:
            # Fallback: character count / 4 (rough estimate)
            return max(1, int(len(text) / self._fallback_ratio))

        # OpenAI tiktoken
        if hasattr(encoder, "encode"):
            return len(encoder.encode(text))

        # Anthropic
        if hasattr(encoder, "count_tokens"):
            return encoder.count_tokens(text)

        return max(1, int(len(text) / self._fallback_ratio))

    def count_messages(self, messages: list[Message], model: str) -> int:
        """Count tokens for message list including overhead."""
        total = 0

        for msg in messages:
            # Base overhead per message (~4 tokens for role + delimiters)
            total += 4
            total += self.count(msg.content, model)

            # Tool calls overhead
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total += 4  # tool call overhead
                    total += self.count(tc.name, model)
                    total += self.count(str(tc.arguments), model)

        return total
