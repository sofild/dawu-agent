"""LLM client factory with lazy loading, caching, and multi-model profile support."""

from __future__ import annotations

from typing import Any

from dawu_agent.config.loader import LLMConfig, ModelProfile
from dawu_agent.llm.base import ILLMClient
from dawu_agent.llm.retry import ExponentialBackoffRetry


class LLMClientFactory:
    """Factory for creating LLM clients with lazy loading and caching.

    Supports multi-model profiles for scenario-based routing.
    Ensures unused SDKs are never imported, and clients are reused.
    """

    _cache: dict[str, ILLMClient] = {}

    @classmethod
    def create(
        cls,
        config: LLMConfig | ModelProfile,
        profile_name: str | None = None,
        provider_override: str | None = None,
    ) -> ILLMClient:
        """Create or retrieve cached LLM client.

        Args:
            config: LLM configuration or ModelProfile
            profile_name: Optional profile name for scenario-based routing
            provider_override: Optional provider override (for fallback)

        Returns:
            Configured ILLMClient instance
        """
        # Resolve profile
        if isinstance(config, LLMConfig):
            if profile_name:
                profile = config.get_profile(profile_name)
            else:
                profile = config.get_profile("default")
        else:
            profile = config

        provider = (provider_override or profile.api_type).lower()
        cache_key = f"{provider}:{profile.name}"

        if cache_key in cls._cache:
            return cls._cache[cache_key]

        retry_policy = ExponentialBackoffRetry(
            max_retries=profile.max_retries,
            base_delay=profile.retry_backoff_base,
        )

        client = cls._create_client(provider, profile, retry_policy)
        cls._cache[cache_key] = client
        return client

    @classmethod
    def create_for_scenario(
        cls,
        config: LLMConfig,
        scenario: str,
        scenario_map: dict[str, str] | None = None,
    ) -> ILLMClient:
        """Create LLM client for a specific scenario.

        Args:
            config: LLM configuration with profiles
            scenario: Scenario name (e.g., "fast", "complex", "vision")
            scenario_map: Optional mapping of scenario -> profile name

        Returns:
            Configured ILLMClient for the scenario
        """
        # Map scenario to profile name
        profile_name = "default"
        if scenario_map and scenario in scenario_map:
            profile_name = scenario_map[scenario]
        elif scenario in ("fast", "complex", "vision", "local"):
            profile_name = scenario

        return cls.create(config, profile_name=profile_name)

    @classmethod
    def _create_client(
        cls,
        provider: str,
        profile: ModelProfile,
        retry_policy: Any,
    ) -> ILLMClient:
        """Instantiate specific provider client (lazy import)."""

        if provider == "anthropic":
            from dawu_agent.llm.providers.anthropic import AnthropicClient

            return AnthropicClient(
                api_key=profile.api_key,
                model=profile.name,
                base_url=profile.base_url or "https://api.anthropic.com",
                retry_policy=retry_policy,
                enable_caching=profile.prompt_caching,
                cache_minimum_tokens=profile.cache_minimum_tokens,
            )

        elif provider == "openai":
            from dawu_agent.llm.providers.openai import OpenAIClient

            return OpenAIClient(
                api_key=profile.api_key,
                model=profile.name,
                base_url=profile.base_url or "https://api.openai.com/v1",
                retry_policy=retry_policy,
            )

        elif provider == "azure":
            # Azure uses OpenAI SDK with different base URL
            from dawu_agent.llm.providers.openai import OpenAIClient

            return OpenAIClient(
                api_key=profile.api_key,
                model=profile.name,
                base_url=profile.base_url,
                retry_policy=retry_policy,
            )

        elif provider == "local":
            from dawu_agent.llm.providers.openai import OpenAIClient

            return OpenAIClient(
                api_key=profile.api_key or "not-needed",
                model=profile.name,
                base_url=profile.base_url or "http://localhost:11434/v1",
                retry_policy=retry_policy,
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @classmethod
    def clear_cache(cls) -> None:
        """Clear client cache (useful for testing)."""
        cls._cache.clear()
