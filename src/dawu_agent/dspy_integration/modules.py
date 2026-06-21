"""DSPy Modules for agent reasoning."""

from __future__ import annotations

import dspy

from dawu_agent.dspy_integration.signatures import (
    ActionDecision,
    ContextSummary,
    DirectResponse,
    ResultSynthesis,
    TaskDecomposition,
)


class AgentReasoner(dspy.Module):
    """Core agent reasoning module.

    Replaces the raw LLM call in the agent loop with a structured
    DSPy pipeline: decide action -> execute -> respond.
    """

    def __init__(self) -> None:
        super().__init__()
        # Phase 1 优化：ChainOfThought → Predict
        # 原因：ChainOfThought 强制 LLM 输出 reasoning 文本字段，与 OpenAI 工具调用协议冲突，
        # 导致 LLM 大量"自言自语"消耗 50-100s/turn，且 reasoning 与 tool_call 互斥。
        # Predict 不强制 reasoning，LLM 可以直接输出结构化 action/tool_name/tool_arguments。
        self.decide_action = dspy.Predict(ActionDecision)
        self.generate_response = dspy.Predict(DirectResponse)

    def forward(
        self,
        conversation_history: str,
        user_query: str,
        available_tools: str,
        current_time: str = "",
        memory_context: str = "",
    ):
        """Decide what to do next based on conversation context."""
        decision = self.decide_action(
            conversation_history=conversation_history,
            user_query=user_query,
            available_tools=available_tools,
            current_time=current_time,
            memory_context=memory_context,
        )
        return decision


class CompressionModule(dspy.Module):
    """DSPy module for context compression (replaces AutocompactStrategy)."""

    def __init__(self) -> None:
        super().__init__()
        self.summarize = dspy.ChainOfThought(ContextSummary)

    def forward(self, conversation_history: str, current_task: str):
        return self.summarize(
            conversation_history=conversation_history,
            current_task=current_task,
        )


class MultiAgentModule(dspy.Module):
    """DSPy module for multi-agent task decomposition and synthesis."""

    def __init__(self) -> None:
        super().__init__()
        self.decompose = dspy.ChainOfThought(TaskDecomposition)
        self.synthesize = dspy.ChainOfThought(ResultSynthesis)

    def forward(
        self,
        task: str,
        available_roles: str = "data_engineer, analyst, report_writer",
    ):
        decomposition = self.decompose(task=task, available_roles=available_roles)
        return decomposition
