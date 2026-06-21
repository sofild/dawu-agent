"""DSPy integration layer for Dawu Agent."""

from __future__ import annotations

import os

# Suppress LiteLLM's remote model-cost-map fetch WARNING. The remote
# endpoint times out from restricted networks; the local backup is
# sufficient for our usage and we never rely on cost reporting.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from dawu_agent.dspy_integration.lm_bridge import DawuLM  # noqa: E402


def configure_dspy(llm_client, model_name: str = "dawu-agent") -> DawuLM:
    """Configure DSPy with our LM bridge.

    Call once during agent initialization to make DSPy modules
    use the same LLM client, API keys, and model routing.
    """
    import dspy

    lm = DawuLM(llm_client, model_name)
    dspy.configure(lm=lm)
    return lm
