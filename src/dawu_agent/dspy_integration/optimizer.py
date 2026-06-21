"""DSPy optimization pipeline for agent modules."""

from __future__ import annotations

import dspy

from dawu_agent.dspy_integration.dataset import get_devset, get_trainset
from dawu_agent.dspy_integration.metrics import composite_metric
from dawu_agent.dspy_integration.modules import AgentReasoner


def optimize_agent_reasoner(
    trainset: list[dspy.Example] | None = None,
    devset: list[dspy.Example] | None = None,
    metric=None,
) -> AgentReasoner:
    """Compile an optimized AgentReasoner using DSPy optimizers.

    Uses BootstrapFewShot to find effective few-shot examples,
    then optionally MIPROv2 for prompt optimization.
    """
    if trainset is None:
        trainset = get_trainset()
    if devset is None:
        devset = get_devset()
    if metric is None:
        metric = composite_metric

    from dspy.teleprompt import BootstrapFewShot

    bootstrap = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
    )

    uncompiled = AgentReasoner()
    compiled = bootstrap.compile(
        uncompiled,
        trainset=trainset,
    )

    return compiled


def save_optimized_module(module: dspy.Module, path: str = "dspy_optimized.json") -> None:
    """Save compiled module for reuse."""
    module.save(path)


def load_optimized_module(path: str = "dspy_optimized.json") -> AgentReasoner:
    """Load a previously compiled module."""
    module = AgentReasoner()
    module.load(path)
    return module
