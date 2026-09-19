"""Grader functions for Phase 8 Agent evaluation suite.

Each grader takes an :class:`EvalScenario` and a list of :class:`AgentEvent`
instances produced by ``Agent.run_stream()`` and returns a plain dict:

    {"passed": bool, "score": float, "detail": str}

Graders are intentionally deterministic — no LLM-as-judge.  This keeps the
evaluation suite reproducible and cheap to run.

Four graders are provided:

1. **grade_task_completion** — checks that the task was completed (a
   ``FinalResponseEvent`` was emitted).
2. **grade_tool_selection** — checks that correct tools were called and
   forbidden tools were not.
3. **grade_trajectory_quality** — checks for issues like repeated failed
   calls, excessive turns, and loops.
4. **grade_cost** — estimates cost based on turns and tool calls.
"""

from __future__ import annotations

from typing import Any

from dawu_agent.core.events import (
    AgentEvent,
    ErrorEvent,
    FinalResponseEvent,
    StateChangeEvent,
    ToolResultEvent,
    ToolUseEvent,
    TurnStartEvent,
)
from evals.scenarios import EvalScenario

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _extract_tool_names(events: list[AgentEvent]) -> list[str]:
    """Extract tool names from ToolUseEvent instances."""
    return [e.tool_name for e in events if isinstance(e, ToolUseEvent)]


def _extract_final_response(events: list[AgentEvent]) -> str:
    """Extract the text from the last FinalResponseEvent."""
    for event in reversed(events):
        if isinstance(event, FinalResponseEvent):
            return event.text
    return ""


def _count_turns(events: list[AgentEvent]) -> int:
    """Count the number of TurnStartEvent instances."""
    return sum(1 for e in events if isinstance(e, TurnStartEvent))


def _has_error_state(events: list[AgentEvent]) -> bool:
    """Check if any StateChangeEvent transitioned to 'error'."""
    return any(
        isinstance(e, StateChangeEvent) and e.new == "error"
        for e in events
    )


def _extract_error_events(events: list[AgentEvent]) -> list[ErrorEvent]:
    """Extract all ErrorEvent instances."""
    return [e for e in events if isinstance(e, ErrorEvent)]


def _has_tool_errors(events: list[AgentEvent]) -> bool:
    """Check if any ToolResultEvent has is_error=True."""
    return any(
        isinstance(e, ToolResultEvent) and e.is_error
        for e in events
    )


def _result(passed: bool, score: float, detail: str) -> dict[str, Any]:
    """Build a standard grader result dict."""
    return {"passed": passed, "score": round(score, 4), "detail": detail}


# ──────────────────────────────────────────────────────────────
# Grader 1: Task completion
# ──────────────────────────────────────────────────────────────

def grade_task_completion(
    scenario: EvalScenario, events: list[AgentEvent]
) -> dict[str, Any]:
    """Check if the task was completed (FinalResponseEvent emitted).

    Scoring:
        - 1.0 if a non-empty FinalResponseEvent was emitted.
        - 0.5 if a FinalResponseEvent was emitted but text is empty.
        - 0.0 if no FinalResponseEvent at all.
        - Pass if score == 1.0.
    """
    response = _extract_final_response(events)
    has_final = any(isinstance(e, FinalResponseEvent) for e in events)

    if has_final and response.strip():
        return _result(True, 1.0, "task completed with non-empty final response")
    elif has_final:
        return _result(False, 0.5, "final response emitted but empty")
    else:
        return _result(False, 0.0, "no FinalResponseEvent emitted — task incomplete")


# ──────────────────────────────────────────────────────────────
# Grader 2: Tool selection
# ──────────────────────────────────────────────────────────────

def grade_tool_selection(
    scenario: EvalScenario, events: list[AgentEvent]
) -> dict[str, Any]:
    """Check if correct tools were called and forbidden tools were not.

    Scoring:
        - Start at 1.0.
        - For each expected tool NOT called: subtract ``1 / len(expected)``.
        - For each forbidden tool called: subtract 0.5 (capped at 1.0 total).
        - Clamp to [0, 1].
        - Pass if score == 1.0.
    """
    called_tools = _extract_tool_names(events)
    called_set = set(called_tools)

    missing: list[str] = []
    for tool in scenario.expected_tools:
        if tool not in called_set:
            missing.append(tool)

    forbidden_called: list[str] = []
    for tool in scenario.forbidden_tools:
        if tool in called_set:
            forbidden_called.append(tool)

    score = 1.0
    if scenario.expected_tools:
        score -= len(missing) / len(scenario.expected_tools)
    score -= min(len(forbidden_called) * 0.5, 1.0)
    score = max(0.0, min(1.0, score))

    parts: list[str] = []
    if missing:
        parts.append(f"missing expected tools: {missing}")
    if forbidden_called:
        parts.append(f"called forbidden tools: {forbidden_called}")
    if not parts:
        parts.append("all tool selection checks passed")

    return _result(score == 1.0, score, "; ".join(parts))


# ──────────────────────────────────────────────────────────────
# Grader 3: Trajectory quality
# ──────────────────────────────────────────────────────────────

def grade_trajectory_quality(
    scenario: EvalScenario, events: list[AgentEvent]
) -> dict[str, Any]:
    """Check for issues: repeated failed calls, excessive turns, loops.

    Criteria:
        1. Total turns must not exceed ``scenario.max_turns``.
        2. If errors occurred, the agent must eventually produce a
           FinalResponseEvent (not left dead).
        3. Final state must not be ``error`` (unexpected termination).
        4. No more than 3 consecutive ErrorEvents without an intervening
           TurnStartEvent (prevents infinite error loops).

    Scoring:
        - Start at 1.0.
        - -0.3 if turns exceeded max_turns.
        - -0.3 if errors occurred but no final response was produced.
        - -0.2 if final state is ``error``.
        - -0.2 if more than 3 consecutive errors (loop detection).
        - Clamp to [0, 1].
        - Pass if score >= 0.7 (recovery is best-effort).
    """
    turn_count = _count_turns(events)
    error_events = _extract_error_events(events)
    has_tool_errors = _has_tool_errors(events)
    has_errors = bool(error_events) or has_tool_errors
    final_response = _extract_final_response(events)
    has_final_response = bool(final_response.strip())
    final_is_error = _has_error_state(events)

    score = 1.0
    issues: list[str] = []

    # Check 1: turn limit
    if turn_count > scenario.max_turns:
        score -= 0.3
        issues.append(
            f"turns exceeded limit ({turn_count} > {scenario.max_turns})"
        )

    # Check 2: errors without recovery
    if has_errors and not has_final_response:
        score -= 0.3
        issues.append("errors occurred but no final response produced")

    # Check 3: unexpected error state
    if final_is_error:
        score -= 0.2
        issues.append("agent ended in error state")

    # Check 4: consecutive error loop detection
    max_consecutive_errors = 0
    current_consecutive = 0
    for e in events:
        if isinstance(e, ErrorEvent):
            current_consecutive += 1
            max_consecutive_errors = max(max_consecutive_errors, current_consecutive)
        elif isinstance(e, TurnStartEvent):
            current_consecutive = 0

    if max_consecutive_errors > 3:
        score -= 0.2
        issues.append(
            f"too many consecutive errors ({max_consecutive_errors} > 3)"
        )

    score = max(0.0, min(1.0, score))

    if not issues:
        issues.append("no trajectory quality issues detected")

    return _result(score >= 0.7, score, "; ".join(issues))


# ──────────────────────────────────────────────────────────────
# Grader 4: Cost
# ──────────────────────────────────────────────────────────────

def grade_cost(
    scenario: EvalScenario, events: list[AgentEvent]
) -> dict[str, Any]:
    """Estimate cost based on turns and tool calls.

    Cost model (rough heuristic — no real billing data):
        - Each turn: 1 cost unit.
        - Each tool call: 2 cost units.
        - Each error event: 3 cost units (wasted work).
    A "budget" is derived from ``scenario.max_turns``:
        ``budget = max_turns * 4`` (1 turn + ~2 tool calls + small overhead).

    Scoring:
        - score = max(0, 1 - cost / budget)
        - If budget is 0 (max_turns=0), score = 1.0 if cost == 0 else 0.0.
        - Pass if score >= 0.5 (cost within 2x of expected budget).
    """
    turn_count = _count_turns(events)
    tool_call_count = sum(1 for e in events if isinstance(e, ToolUseEvent))
    error_count = len(_extract_error_events(events))

    cost = turn_count * 1 + tool_call_count * 2 + error_count * 3
    budget = max(scenario.max_turns * 4, 1)

    score = max(0.0, 1.0 - cost / budget)

    detail = (
        f"turns={turn_count}, tool_calls={tool_call_count}, "
        f"errors={error_count}, cost={cost}, budget={budget}"
    )

    return _result(score >= 0.5, score, detail)


# ──────────────────────────────────────────────────────────────
# Aggregate grader
# ──────────────────────────────────────────────────────────────

ALL_GRADERS = [
    ("task_completion", grade_task_completion),
    ("tool_selection", grade_tool_selection),
    ("trajectory_quality", grade_trajectory_quality),
    ("cost", grade_cost),
]


def grade_all(
    scenario: EvalScenario, events: list[AgentEvent]
) -> dict[str, dict[str, Any]]:
    """Run all four graders and return a dict of results.

    Returns:
        Dict mapping grader name to result dict.
    """
    return {
        name: grader(scenario, events)
        for name, grader in ALL_GRADERS
    }
