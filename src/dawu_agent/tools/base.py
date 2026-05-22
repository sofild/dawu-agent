"""Base tool interfaces and data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolCategory(Enum):
    """Tool classification categories."""

    AGENT = "agent"
    WORKFLOW = "workflow"
    TASK = "task"
    PLAN = "plan"
    ADVANCED = "advanced"
    DATA = "data"


class ConcurrencyMode(Enum):
    """Concurrency safety classification."""

    READ_ONLY = "read_only"  # Safe to execute concurrently
    WRITE = "write"          # Must be serialized
    MIXED = "mixed"          # Depends on arguments (runtime determined)


@dataclass
class ToolResult:
    """Standardized tool execution result."""

    success: bool
    data: Any = None
    error: str | None = None
    truncated: bool = False
    truncated_info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: Any, truncated: bool = False, truncated_info: dict | None = None) -> "ToolResult":
        return cls(
            success=True,
            data=data,
            truncated=truncated,
            truncated_info=truncated_info or {},
        )

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        return cls(success=False, error=message)


@dataclass
class ToolDependency:
    """Tool dependency declaration."""

    tool_name: str
    depends_on: list[str] = field(default_factory=list)
    resolver: str = ""  # Path to extract params from dependency results


class Tool(ABC):
    """Abstract tool interface.

    Each tool is a prompt for the LLM - the description quality directly
    impacts LLM calling accuracy.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed description for the LLM.

        Must include:
        - What the tool does (one sentence)
        - When to use it (and when NOT to use it)
        - Parameter semantics (type, format, constraints)
        - Return value structure
        - Common misuse examples (optional but recommended)
        """
        ...

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        ...

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.TASK

    @property
    def concurrency_mode(self) -> ConcurrencyMode:
        return ConcurrencyMode.READ_ONLY

    @property
    def dependencies(self) -> list[ToolDependency]:
        return []

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute tool logic. Must never raise unhandled exceptions."""
        ...

    def to_llm_definition(self) -> dict[str, Any]:
        """Convert to LLM-compatible tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }
