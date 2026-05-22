"""Agent event types for streaming and session logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentEvent:
    """Base class for all agent events."""

    type: str
    timestamp: float = field(default_factory=lambda: __import__("time").monotonic())


@dataclass
class StateChangeEvent(AgentEvent):
    """Agent state transition event."""

    old: str = ""
    new: str = ""
    reason: str = ""

    def __init__(self, old: str = "", new: str = "", reason: str = "") -> None:
        super().__init__(type="state_change")
        self.old = old
        self.new = new
        self.reason = reason


@dataclass
class TurnStartEvent(AgentEvent):
    """Turn start event."""

    turn: int = 0

    def __init__(self, turn: int = 0) -> None:
        super().__init__(type="turn_start")
        self.turn = turn


@dataclass
class TurnEndEvent(AgentEvent):
    """Turn end event."""

    turn: int = 0
    duration: float = 0.0

    def __init__(self, turn: int = 0, duration: float = 0.0) -> None:
        super().__init__(type="turn_end")
        self.turn = turn
        self.duration = duration


@dataclass
class UserMessageEvent(AgentEvent):
    """User message event."""

    content: str = ""

    def __init__(self, content: str = "") -> None:
        super().__init__(type="user_message")
        self.content = content


@dataclass
class AssistantTextEvent(AgentEvent):
    """Assistant text chunk event."""

    content: str = ""
    is_final: bool = False

    def __init__(self, content: str = "", is_final: bool = False) -> None:
        super().__init__(type="assistant_text")
        self.content = content
        self.is_final = is_final


@dataclass
class ToolUseEvent(AgentEvent):
    """Tool use request event."""

    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_use_id: str = ""

    def __init__(self, tool_name: str = "", tool_input: dict | None = None, tool_use_id: str = "") -> None:
        super().__init__(type="tool_use")
        self.tool_name = tool_name
        self.tool_input = tool_input or {}
        self.tool_use_id = tool_use_id


@dataclass
class ToolResultEvent(AgentEvent):
    """Tool execution result event."""

    tool_use_id: str = ""
    content: str = ""
    is_error: bool = False

    def __init__(self, tool_use_id: str = "", content: str = "", is_error: bool = False) -> None:
        super().__init__(type="tool_result")
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


@dataclass
class ErrorEvent(AgentEvent):
    """Error/recovery event."""

    error_type: str = ""
    turn: int = 0
    action: str = ""
    detail: str = ""

    def __init__(self, error_type: str = "", turn: int = 0, action: str = "", detail: str = "") -> None:
        super().__init__(type="error")
        self.error_type = error_type
        self.turn = turn
        self.action = action
        self.detail = detail


@dataclass
class FinalResponseEvent(AgentEvent):
    """Final response event."""

    text: str = ""

    def __init__(self, text: str = "") -> None:
        super().__init__(type="final_response")
        self.text = text


@dataclass
class CompactionEvent(AgentEvent):
    """Context compaction event."""

    detail: str = ""

    def __init__(self, detail: str = "") -> None:
        super().__init__(type="compaction")
        self.detail = detail
