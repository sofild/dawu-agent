"""Bridge existing ILLMClient to DSPy's LM interface."""

from __future__ import annotations

from typing import Any

import dspy

from dawu_agent.llm.base import ILLMClient, Message


class DawuLM(dspy.BaseLM):
    """Custom DSPy LM that delegates to our existing ILLMClient.

    This allows DSPy modules to use the same LLM client, API keys,
    and model routing that the agent already uses.
    """

    def __init__(self, llm_client: ILLMClient, model_name: str = "dawu-agent") -> None:
        super().__init__(model=model_name)
        self._client = llm_client
        self.model = model_name

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """DSPy LM interface: convert DSPy messages to our Message format.

        Returns a list of dicts with key ``"text"`` (the canonical DSPy 3.x
        format that ``ChatAdapter`` reads via ``output["text"]``). Returning
        ``"content"`` instead triggers ``KeyError: 'text'`` in the adapter.
        """
        converted = self._convert_messages(prompt, messages)
        if not converted:
            return []

        response = self._client.chat_sync(converted)
        return [{"text": response.content}]

    async def acall(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Async version for DSPy async modules.

        Mirrors the sync path: returns ``[{"text": ...}]`` to match DSPy 3.x
        adapter expectations.
        """
        converted = self._convert_messages(prompt, messages)
        if not converted:
            return []

        response = await self._client.chat(converted)
        return [{"text": response.content}]

    def _convert_messages(
        self,
        prompt: str | None,
        messages: list[dict] | None,
    ) -> list[Message]:
        """Convert DSPy-format messages to our internal Message format."""
        if messages is not None:
            return [
                Message(role=m.get("role", "user"), content=m.get("content", ""))
                for m in messages
            ]
        if prompt is not None:
            return [Message(role="user", content=prompt)]
        return []
