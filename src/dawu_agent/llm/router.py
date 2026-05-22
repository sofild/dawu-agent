"""Multi-model router for scenario-based LLM selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dawu_agent.config.loader import LLMConfig, ModelProfile
from dawu_agent.llm.base import ILLMClient
from dawu_agent.llm.factory import LLMClientFactory


@dataclass
class RoutingDecision:
    """Result of model routing decision."""

    profile_name: str
    profile: ModelProfile
    reason: str
    estimated_cost: float = 0.0


class ModelRouter:
    """Intelligent router that selects the best model for each task.

    Scenarios:
    - fast: Quick responses, simple tasks (cheap model)
    - complex: Deep reasoning, analysis (powerful model)
    - vision: Image analysis (vision-capable model)
    - local: Privacy-sensitive tasks (local model)
    - default: Balanced choice
    """

    # Cost estimates per 1K tokens (rough USD)
    COST_TABLE: dict[str, float] = {
        "gpt-4o": 0.005,
        "gpt-4o-mini": 0.00015,
        "claude-3-5-sonnet-20241022": 0.003,
        "claude-3-opus-20240229": 0.015,
        "claude-3-haiku-20240307": 0.00025,
        "llama3.1": 0.0,
    }

    def __init__(
        self,
        config: LLMConfig,
        scenario_map: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.scenario_map = scenario_map or {}
        self._clients: dict[str, ILLMClient] = {}

    def route(self, task_description: str, context: dict[str, Any] | None = None) -> RoutingDecision:
        """Determine the best model for a given task.

        Uses keyword matching and context to select scenario.
        """
        context = context or {}
        task_lower = task_description.lower()

        # Vision tasks
        vision_keywords = ["image", "picture", "photo", "vision", "chart", "graph", "plot"]
        if any(kw in task_lower for kw in vision_keywords):
            return self._select_profile("vision", "Task requires vision capabilities")

        # Complex reasoning tasks
        complex_keywords = [
            "analyze", "analysis", "deep", "complex", "reasoning",
            "research", "investigate", "compare", "evaluate",
        ]
        if any(kw in task_lower for kw in complex_keywords):
            return self._select_profile("complex", "Task requires deep reasoning")

        # Simple/quick tasks
        fast_keywords = [
            "quick", "simple", "brief", "short", "summarize",
            "hello", "hi", "greeting", "status",
        ]
        if any(kw in task_lower for kw in fast_keywords):
            return self._select_profile("fast", "Task is simple/quick")

        # Privacy-sensitive
        if context.get("privacy_sensitive") or context.get("local_only"):
            return self._select_profile("local", "Privacy-sensitive task")

        # Default
        return self._select_profile("default", "Default routing")

    def _select_profile(self, scenario: str, reason: str) -> RoutingDecision:
        """Select profile for scenario."""
        profile_name = self.scenario_map.get(scenario, scenario)

        try:
            profile = self.config.get_profile(profile_name)
        except ValueError:
            # Fallback to default
            profile = self.config.get_profile("default")
            profile_name = "default"
            reason += " (requested profile not found, using default)"

        cost = self.COST_TABLE.get(profile.name, 0.003)

        return RoutingDecision(
            profile_name=profile_name,
            profile=profile,
            reason=reason,
            estimated_cost=cost,
        )

    def get_client(self, scenario: str | None = None, task_description: str = "") -> ILLMClient:
        """Get LLM client for scenario or auto-route from task description."""
        if scenario is None and task_description:
            decision = self.route(task_description)
            scenario = decision.profile_name
        elif scenario is None:
            scenario = "default"

        cache_key = scenario
        if cache_key in self._clients:
            return self._clients[cache_key]

        client = LLMClientFactory.create(self.config, profile_name=scenario)
        self._clients[cache_key] = client
        return client

    def get_client_for_profile(self, profile_name: str) -> ILLMClient:
        """Get LLM client for a specific profile name."""
        if profile_name in self._clients:
            return self._clients[profile_name]

        client = LLMClientFactory.create(self.config, profile_name=profile_name)
        self._clients[profile_name] = client
        return client
