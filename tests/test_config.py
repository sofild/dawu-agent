"""Tests for configuration management."""

import pytest
from dawu_agent.config.loader import ConfigLoader, Settings


class TestConfigLoader:
    def test_load_default_settings(self):
        """Test loading settings with defaults."""
        loader = ConfigLoader()
        settings = loader.load(env="development")

        assert settings.env == "development"
        assert settings.llm.provider == "anthropic"
        assert settings.llm.model == "claude-3-5-sonnet-20241022"
        assert settings.agent.max_turns == 50

    def test_settings_validation(self):
        """Test pydantic validation catches invalid values."""
        with pytest.raises(ValueError):
            Settings(llm={"temperature": 3.0})

    def test_source_trace(self):
        """Test configuration source tracking."""
        loader = ConfigLoader()
        loader.load(env="development")
        trace = loader.get_source_trace()
        assert isinstance(trace, dict)
