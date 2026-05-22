"""Tests for context compression pipeline."""

import pytest
from dawu_agent.context.compression import CompressionPipeline, CompressionRequest
from dawu_agent.llm.base import ILLMClient, LLMResponse, Message


class MockLLMClient(ILLMClient):
    """Mock LLM client for testing."""

    def __init__(self):
        self.call_count = 0

    def count_tokens(self, messages):
        return sum(len(m.content) // 4 for m in messages)

    async def chat(self, messages, tools=None):
        self.call_count += 1
        return LLMResponse(content="summary")

    async def chat_stream(self, messages, tools=None):
        yield type("Chunk", (), {"content": "", "tool_call": None, "finish_reason": "stop"})()

    async def validate_model(self):
        from dawu_agent.llm.base import ModelCapability
        return ModelCapability(available=True)

    def chat_sync(self, messages, tools=None):
        self.call_count += 1
        return LLMResponse(content="summary")


class TestCompressionPipeline:
    @pytest.fixture
    def pipeline(self):
        llm = MockLLMClient()
        return CompressionPipeline(llm, llm.count_tokens, budget_ratio=0.85)

    def test_should_compress_when_over_budget(self, pipeline):
        messages = [Message(role="user", content="x" * 4000)]
        assert pipeline.should_compress(messages, 1000)

    def test_should_not_compress_when_under_budget(self, pipeline):
        messages = [Message(role="user", content="x" * 100)]
        assert not pipeline.should_compress(messages, 10000)

    def test_snip_strategy(self, pipeline):
        messages = [
            Message(role="system", content="sys"),
            *[Message(role="user", content=f"msg{i}") for i in range(20)],
        ]
        result = pipeline.compact(messages, 1000)
        assert result.level_used >= 1
        assert result.tokens_freed > 0

    def test_emergency_snip(self, pipeline):
        messages = [Message(role="user", content="x" * 10000) for _ in range(50)]
        result = pipeline.compact(messages, 100)
        assert result.level_used == 4
        assert len(result.messages) <= 10
