"""Anthropic Claude client with streaming and prompt caching support."""

from __future__ import annotations

import asyncio
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


class AnthropicClient(ILLMClient):
    """Anthropic Claude API client with lazy SDK loading."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        base_url: str | None = None,
        retry_policy: IRetryPolicy | None = None,
        enable_caching: bool = True,
        cache_minimum_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.retry_policy = retry_policy
        self.enable_caching = enable_caching
        self.cache_minimum_tokens = cache_minimum_tokens
        self._client: Any = None
        self._token_counter = UnifiedTokenCounter()

    def _init_client(self) -> None:
        """Lazy-load the Anthropic SDK."""
        if self._client is not None:
            return

        try:
            import anthropic
        except ImportError as e:
            raise ImportError("anthropic SDK not installed. Run: pip install anthropic") from e

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self._client = anthropic.AsyncAnthropic(**kwargs)

    def _messages_to_anthropic(self, messages: list[Message]) -> list[dict]:
        """Convert standardized messages to Anthropic format."""
        result = []
        for msg in messages:
            if msg.role == "system":
                # System messages handled separately in Anthropic
                continue

            anthropic_msg: dict[str, Any] = {"role": msg.role, "content": msg.content}

            if msg.role == "assistant" and msg.tool_calls:
                anthropic_msg["content"] = []
                if msg.content:
                    anthropic_msg["content"].append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    anthropic_msg["content"].append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })

            if msg.role == "tool":
                anthropic_msg = {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": msg.content,
                    }],
                }

            result.append(anthropic_msg)
        return result

    def _extract_system_message(self, messages: list[Message]) -> str | None:
        """Extract system message from message list."""
        for msg in messages:
            if msg.role == "system":
                return msg.content
        return None

    def _tools_to_anthropic(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI-style tools to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if "function" in tool:
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object"}),
                })
            else:
                anthropic_tools.append(tool)
        return anthropic_tools

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Anthropic response to standardized format."""
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.input),
                ))

        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"
        elif response.stop_reason == "max_tokens":
            finish_reason = "length"

        usage = {}
        if hasattr(response, "usage"):
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
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
        anthropic_messages = self._messages_to_anthropic(messages)
        system_msg = self._extract_system_message(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": 4096,
        }

        if system_msg:
            kwargs["system"] = system_msg

        if tools:
            kwargs["tools"] = self._tools_to_anthropic(tools)

        response = await self._client.messages.create(**kwargs)
        return self._parse_response(response)

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ):
        """Send streaming chat request."""
        self._init_client()

        anthropic_messages = self._messages_to_anthropic(messages)
        system_msg = self._extract_system_message(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": 4096,
            "stream": True,
        }

        if system_msg:
            kwargs["system"] = system_msg

        if tools:
            kwargs["tools"] = self._tools_to_anthropic(tools)

        async with self._client.messages.stream(**kwargs) as stream:
            current_tool_call: dict[str, Any] | None = None

            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield StreamChunk(content=event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        if current_tool_call is not None:
                            current_tool_call["input_json"] = current_tool_call.get("input_json", "") + event.delta.partial_json

                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool_call = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "input_json": "",
                        }

                elif event.type == "content_block_stop":
                    if current_tool_call is not None:
                        import json
                        try:
                            arguments = json.loads(current_tool_call.get("input_json", "{}"))
                        except json.JSONDecodeError:
                            arguments = {}
                        yield StreamChunk(
                            tool_call=ToolCall(
                                id=current_tool_call["id"],
                                name=current_tool_call["name"],
                                arguments=arguments,
                            )
                        )
                        current_tool_call = None

                elif event.type == "message_stop":
                    yield StreamChunk(finish_reason="stop")

    def count_tokens(self, messages: list[Message]) -> int:
        """Estimate token count."""
        return self._token_counter.count_messages(messages, self.model)

    async def validate_model(self) -> ModelCapability:
        """Validate model capabilities."""
        # Anthropic models generally support tools and streaming
        claude_models = {
            "claude-3-opus": (200000, 4096),
            "claude-3-5-sonnet": (200000, 4096),
            "claude-3-haiku": (200000, 4096),
        }

        model_prefix = self.model.split("-")[0:3]
        model_key = "-".join(model_prefix)

        max_context = 200000
        max_output = 4096

        for key, (ctx, out) in claude_models.items():
            if key in self.model:
                max_context = ctx
                max_output = out
                break

        return ModelCapability(
            available=True,
            supports_tools=True,
            supports_vision="opus" in self.model or "sonnet" in self.model,
            supports_streaming=True,
            max_context_tokens=max_context,
            max_output_tokens=max_output,
        )
