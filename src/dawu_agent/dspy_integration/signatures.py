"""DSPy Signatures for agent reasoning steps."""

from __future__ import annotations

from typing import Literal

import dspy


class ActionDecision(dspy.Signature):
    """Decide the next action for a data analysis agent.

    Given the conversation history and available tools, decide whether to:
    - call a tool (specify which tool and arguments)
    - respond directly to the user
    - stop and provide the final answer
    """

    conversation_history: str = dspy.InputField(
        desc="Full conversation history including tool results"
    )
    user_query: str = dspy.InputField(
        desc="The original user query"
    )
    available_tools: str = dspy.InputField(
        desc="List of available tools with their descriptions"
    )
    current_time: str = dspy.InputField(
        desc="Current date and time for time-aware decisions"
    )
    memory_context: str = dspy.InputField(
        desc="Relevant memories from past interactions"
    )
    action: Literal["use_tool", "respond", "stop"] = dspy.OutputField(
        desc="Next action: use_tool, respond, or stop"
    )
    tool_name: str = dspy.OutputField(
        desc="Name of tool to call (empty if not use_tool)"
    )
    tool_arguments: str = dspy.OutputField(
        desc="JSON arguments for the tool (empty if not use_tool)"
    )
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning for this decision"
    )


class DirectResponse(dspy.Signature):
    """Generate a direct response to the user based on conversation context."""

    conversation_history: str = dspy.InputField(
        desc="Conversation history including tool results"
    )
    user_query: str = dspy.InputField(
        desc="The original user query"
    )
    current_time: str = dspy.InputField(
        desc="Current date and time"
    )
    memory_context: str = dspy.InputField(
        desc="Relevant memories from past interactions"
    )
    response: str = dspy.OutputField(
        desc="Helpful, accurate response to the user"
    )


class ContextSummary(dspy.Signature):
    """Summarize conversation history for context compression."""

    conversation_history: str = dspy.InputField(
        desc="Historical messages to summarize"
    )
    current_task: str = dspy.InputField(
        desc="Current task the user is working on"
    )
    summary: str = dspy.OutputField(
        desc="Structured summary preserving: goals, completed steps, "
        "key decisions, pending tasks, active files, tool context"
    )


class TaskDecomposition(dspy.Signature):
    """Decompose a complex task into independent subtasks."""

    task: str = dspy.InputField(desc="Complex task to decompose")
    available_roles: str = dspy.InputField(
        desc="Available agent roles: data_engineer, analyst, report_writer"
    )
    subtasks: str = dspy.OutputField(
        desc="JSON list of subtasks with role assignments and dependencies"
    )


class ResultSynthesis(dspy.Signature):
    """Synthesize multiple subtask results into a coherent final report."""

    original_task: str = dspy.InputField(desc="The original user task")
    subtask_results: str = dspy.InputField(
        desc="Results from each subtask"
    )
    final_report: str = dspy.OutputField(
        desc="Comprehensive final report combining all subtask results"
    )
