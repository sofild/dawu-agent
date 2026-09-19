"""Deterministic verifier for tool execution results (v4 CONTINUE-SITE-8).

The verifier runs after tool execution and checks if the result is acceptable.
On failure, it returns structured feedback that is injected into the next turn's
context, allowing the LLM to self-correct.  After 3 consecutive failures for the
same tool, :meth:`Verifier.should_escalate` returns True to signal that human
intervention or a different strategy is needed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from dawu_agent.tools.base import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Outcome of a single verification check."""

    passed: bool
    failure_detail: str = ""
    suggested_fix: str = ""
    sensor_context: dict[str, Any] = field(default_factory=dict)


class VerifierCheck(ABC):
    """Abstract base class for a single verification check."""

    @abstractmethod
    def check(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_result: ToolResult,
    ) -> VerificationResult:
        """Run the check and return the outcome."""
        ...


class DefaultVerifierCheck(VerifierCheck):
    """Default check: success flag, non-empty data, no error.

    Tools listed in *optional_tools* are allowed to return ``None`` or empty
    data (e.g. existence-check tools whose "not found" is a valid result).
    """

    def __init__(self, optional_tools: set[str] | None = None) -> None:
        self._optional_tools = optional_tools or set()

    def check(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_result: ToolResult,
    ) -> VerificationResult:
        # 1. success must be True
        if not tool_result.success:
            return VerificationResult(
                passed=False,
                failure_detail="tool_result.success is False",
                suggested_fix="Inspect the tool's error field and retry with corrected input.",
                sensor_context={"tool_error": tool_result.error},
            )

        # 2. error must be None
        if tool_result.error is not None:
            return VerificationResult(
                passed=False,
                failure_detail=f"tool_result.error is not None: {tool_result.error}",
                suggested_fix="Retry the tool call; if the error persists, inspect dependencies.",
                sensor_context={"tool_error": tool_result.error},
            )

        # 3. data must not be None/empty for non-optional tools
        if tool_name not in self._optional_tools:
            if tool_result.data is None:
                return VerificationResult(
                    passed=False,
                    failure_detail="tool_result.data is None for a non-optional tool",
                    suggested_fix="Retry the tool call; the tool may have a transient data-fetch issue.",
                    sensor_context={"tool_name": tool_name},
                )
            if isinstance(tool_result.data, (str, list, dict)) and len(tool_result.data) == 0:
                return VerificationResult(
                    passed=False,
                    failure_detail="tool_result.data is empty for a non-optional tool",
                    suggested_fix="Retry the tool call with adjusted parameters.",
                    sensor_context={"tool_name": tool_name},
                )

        return VerificationResult(passed=True)


class Verifier:
    """Deterministic verifier for tool execution results (v4 CONTINUE-SITE-8).

    Runs registered checks in order.  The first failing check short-circuits and
    its :class:`VerificationResult` is returned.  Tracks consecutive failures per
    tool name for escalation decisions.
    """

    _ESCALATE_THRESHOLD = 3

    def __init__(self) -> None:
        self._checks: list[VerifierCheck] = []
        self._consecutive_failures: dict[str, int] = {}

    def register_check(self, check: VerifierCheck) -> None:
        """Register a verification check."""
        self._checks.append(check)

    def verify(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_result: ToolResult,
    ) -> VerificationResult:
        """Run all registered checks. Returns first failure or success."""
        for check in self._checks:
            result = check.check(tool_name, tool_input, tool_result)
            if not result.passed:
                self._consecutive_failures[tool_name] = (
                    self._consecutive_failures.get(tool_name, 0) + 1
                )
                logger.warning(
                    "Verification failed for %s (consecutive=%d): %s",
                    tool_name,
                    self._consecutive_failures[tool_name],
                    result.failure_detail,
                )
                return result

        # Success — reset the counter
        self._consecutive_failures[tool_name] = 0
        return VerificationResult(passed=True)

    def should_escalate(self, tool_name: str) -> bool:
        """Return True after 3 consecutive failures for the same tool."""
        return self._consecutive_failures.get(tool_name, 0) >= self._ESCALATE_THRESHOLD
