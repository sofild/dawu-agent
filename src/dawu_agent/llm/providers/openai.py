"""OpenAI / Azure OpenAI client with streaming support."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from dawu_agent.llm.base import (
    ILLMClient,
    IRetryPolicy,
    LLMResponse,
    Message,
    ModelCapability,
    StreamChunk,
    ToolCall,
)
from dawu_agent.llm.token_counter import UnifiedTokenCounter


class OpenAIClient(ILLMClient):
    """OpenAI API client with lazy SDK loading."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        retry_policy: IRetryPolicy | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.retry_policy = retry_policy
        self._client: Any = None
        self._token_counter = UnifiedTokenCounter()

    def _init_client(self) -> None:
        """Lazy-load the OpenAI SDK."""
        if self._client is not None:
            return

        try:
            import openai
        except ImportError as e:
            raise ImportError("openai SDK not installed. Run: pip install openai") from e

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self._client = openai.AsyncOpenAI(**kwargs)

    def _messages_to_openai(self, messages: list[Message]) -> list[dict]:
        """Convert standardized messages to OpenAI format."""
        result = []
        for msg in messages:
            openai_msg: dict[str, Any] = {"role": msg.role, "content": msg.content}

            if msg.role == "assistant" and msg.tool_calls:
                openai_msg["content"] = msg.content or ""
                openai_msg["tool_calls"] = []
                for tc in msg.tool_calls:
                    # tool_call.arguments MUST be a JSON-encoded string per
                    # the OpenAI spec. str(dict) produces single-quoted Python
                    # repr which is rejected by strict third-party
                    # OpenAI-compatible gateways (e.g. autodl).
                    try:
                        arguments_json = (
                            tc.arguments
                            if isinstance(tc.arguments, str)
                            else json.dumps(tc.arguments, ensure_ascii=False)
                        )
                    except (TypeError, ValueError):
                        arguments_json = "{}"

                    openai_msg["tool_calls"].append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": arguments_json,
                        },
                    })

            if msg.role == "tool":
                openai_msg["tool_call_id"] = msg.tool_call_id or ""

            result.append(openai_msg)
        return result

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI response to standardized format."""
        choice = response.choices[0]
        message = choice.message

        content = message.content or ""
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                import json
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        finish_reason = choice.finish_reason or "stop"
        if finish_reason == "tool_calls":
            finish_reason = "tool_calls"

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            finish_reason=finish_reason,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send chat request with retry logic."""
        self._init_client()

        attempt = 0
        while True:
            try:
                return await self._do_chat(messages, tools)
            except Exception as e:
                if self.retry_policy is None or not self.retry_policy.should_retry(e, attempt):
                    raise
                delay = self.retry_policy.backoff_delay(attempt)
                await asyncio.sleep(delay)
                attempt += 1

    async def _do_chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Internal chat implementation."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages_to_openai(messages),
        }

        if tools:
            kwargs["tools"] = tools

        response = await self._client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ):
        """Send streaming chat request."""
        self._init_client()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages_to_openai(messages),
            "stream": True,
        }

        if tools:
            kwargs["tools"] = tools

        stream = await self._client.chat.completions.create(**kwargs)

        current_tool_calls: dict[str, dict[str, Any]] = {}

        async for chunk in stream:
            if not chunk.choices:
                # Some streaming chunks (e.g. final usage-only chunks from
                # third-party OpenAI-compatible APIs) carry no choices.
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                yield StreamChunk(content=delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    tc_id = tc_delta.index
                    if tc_id not in current_tool_calls:
                        current_tool_calls[tc_id] = {"id": "", "name": "", "arguments": ""}

                    if tc_delta.id:
                        current_tool_calls[tc_id]["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        current_tool_calls[tc_id]["name"] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        current_tool_calls[tc_id]["arguments"] += tc_delta.function.arguments

            finish = chunk.choices[0].finish_reason
            if finish:
                # Yield any completed tool calls
                for tc_data in current_tool_calls.values():
                    if tc_data["name"]:
                        import json
                        try:
                            args = json.loads(tc_data["arguments"])
                        except json.JSONDecodeError:
                            args = {}
                        yield StreamChunk(
                            tool_call=ToolCall(
                                id=tc_data["id"],
                                name=tc_data["name"],
                                arguments=args,
                            )
                        )

                reason = "stop"
                if finish == "tool_calls":
                    reason = "tool_calls"
                elif finish == "length":
                    reason = "length"
                yield StreamChunk(finish_reason=reason)

    def count_tokens(self, messages: list[Message]) -> int:
        """Estimate token count."""
        return self._token_counter.count_messages(messages, self.model)

    async def validate_model(self) -> ModelCapability:
        """Validate model capabilities."""
        model_caps = {
            "gpt-4o": (128000, 4096, True, True),
            "gpt-4o-mini": (128000, 4096, True, True),
            "gpt-4-turbo": (128000, 4096, True, True),
            "gpt-4": (8192, 4096, True, False),
            "gpt-3.5-turbo": (16385, 4096, True, False),
        }

        for key, (ctx, out, tools, vision) in model_caps.items():
            if key in self.model:
                return ModelCapability(
                    available=True,
                    supports_tools=tools,
                    supports_vision=vision,
                    supports_streaming=True,
                    max_context_tokens=ctx,
                    max_output_tokens=out,
                )

        return ModelCapability(
            available=True,
            supports_tools=True,
            supports_streaming=True,
            max_context_tokens=128000,
            max_output_tokens=4096,
        )
