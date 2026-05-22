"""Abstract interfaces for LLM clients, retry policies, and token counters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class Message:
    """Standardized message format across all providers."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: str | None = None  # For tool role: tool_call_id
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass
class ToolCall:
    """Standardized tool call representation."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class StreamChunk:
    """Single chunk in a streaming response."""

    content: str = ""
    tool_call: ToolCall | None = None
    finish_reason: str | None = None  # "stop" | "tool_calls" | "length"


@dataclass
class LLMResponse:
    """Complete standardized response."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""


@dataclass
class ModelCapability:
    """Model capability description."""

    available: bool = False
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = False
    max_context_tokens: int = 0
    max_output_tokens: int = 0


class ILLMClient(ABC):
    """Unified LLM client interface. All provider implementations must inherit this."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send non-streaming chat request, return complete response."""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming chat request, return async iterator."""
        ...

    def chat_sync(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Synchronous chat interface."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.chat(messages, tools)).result()
        except RuntimeError:
            return asyncio.run(self.chat(messages, tools))

    @abstractmethod
    def count_tokens(self, messages: list[Message]) -> int:
        """Estimate token count locally (no API call)."""
        ...

    @abstractmethod
    async def validate_model(self) -> ModelCapability:
        """Validate model availability and capabilities."""
        ...


class IRetryPolicy(ABC):
    """Pluggable retry strategy independent of provider clients."""

    @abstractmethod
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Decide whether to retry based on exception type and attempt count."""
        ...

    @abstractmethod
    def backoff_delay(self, attempt: int) -> float:
        """Calculate delay in seconds for the nth retry attempt."""
        ...

    @property
    @abstractmethod
    def max_retries(self) -> int:
        """Maximum number of retry attempts."""
        ...


class ITokenCounter(ABC):
    """Provider-agnostic token counting interface."""

    @abstractmethod
    def count(self, text: str, model: str) -> int:
        """Estimate token count for given text under specified model."""
        ...

    @abstractmethod
    def count_messages(self, messages: list[Message], model: str) -> int:
        """Estimate token count for message list (including role prefix overhead)."""
        ...
