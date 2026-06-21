"""DSPy evaluation metrics for agent quality."""

from __future__ import annotations

import dspy


def action_correctness(example, prediction, trace=None) -> bool:
    """Metric: Did the agent choose the correct action?"""
    expected = example.action.lower().strip()
    predicted = prediction.action.lower().strip()
    return expected == predicted


def tool_selection_accuracy(example, prediction, trace=None) -> bool:
    """Metric: Did the agent select the correct tool?"""
    if example.action != "use_tool":
        return prediction.action != "use_tool"
    return example.tool_name.lower() == prediction.tool_name.lower()


def response_quality(example, prediction, trace=None) -> bool:
    """Metric: Quality of final response (requires LLM-as-judge)."""
    judge = dspy.ChainOfThought(
        "user_query, agent_response -> quality_score: float"
    )
    result = judge(
        user_query=example.user_query,
        agent_response=prediction.response if hasattr(prediction, "response") else "",
    )
    try:
        return float(result.quality_score) >= 0.7
    except (ValueError, TypeError):
        return False


def composite_metric(example, prediction, trace=None) -> float:
    """Composite metric combining action and tool selection quality."""
    action_score = float(action_correctness(example, prediction, trace))
    tool_score = float(tool_selection_accuracy(example, prediction, trace))
    return (action_score + tool_score) / 2
