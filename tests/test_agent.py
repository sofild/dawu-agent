"""Tests for agent core loop."""

import pytest
import asyncio
from dawu_agent.config.loader import Settings
from dawu_agent.observability.telemetry import TelemetryManager
from dawu_agent.core.agent import Agent


class TestAgent:
    @pytest.fixture
    def agent(self):
        settings = Settings()
        telemetry = TelemetryManager(settings)
        telemetry.initialize()
        return Agent(settings=settings, telemetry=telemetry)

    @pytest.mark.asyncio
    async def test_initialize(self, agent):
        """Test agent initialization."""
        await agent.initialize()
        assert agent.state == "idle"
        assert agent._initialized

    @pytest.mark.asyncio
    async def test_run_turn(self, agent):
        """Test basic turn execution."""
        await agent.initialize()
        response = await agent.run_turn("Hello")
        assert "Hello" in response

    @pytest.mark.asyncio
    async def test_max_turns(self, agent):
        """Test max turns limit."""
        await agent.initialize()
        agent._turn_count = 999999
        with pytest.raises(RuntimeError):
            await agent.run_turn("Hello")

    @pytest.mark.asyncio
    async def test_stream_events(self, agent):
        """Test streaming event generation."""
        await agent.initialize()
        events = []
        async for event in agent.run_stream("Test"):
            events.append(event)

        assert len(events) > 0
        assert any(e["type"] == "state_change" for e in events)
