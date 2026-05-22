"""Tests for observability components."""

import pytest
from dawu_agent.config.loader import Settings
from dawu_agent.observability.telemetry import TelemetryManager


class TestTelemetryManager:
    def test_initialization(self):
        """Test telemetry manager initializes without errors."""
        settings = Settings()
        telemetry = TelemetryManager(settings)
        telemetry.initialize()

        assert telemetry._initialized
        assert telemetry.logger is not None

    def test_shutdown(self):
        """Test graceful shutdown."""
        settings = Settings()
        telemetry = TelemetryManager(settings)
        telemetry.initialize()
        telemetry.shutdown()

        # After shutdown, tracer/meter may be None depending on implementation
        assert telemetry._initialized
