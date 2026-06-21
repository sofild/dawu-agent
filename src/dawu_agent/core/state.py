"""Agent state machine and runtime state management."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from dawu_agent.llm.base import Message


class AgentState(Enum):
    """Agent lifecycle states."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    EXPIRED = "expired"
    ERROR = "error"


@dataclass
class AgentRunState:
    """Agent runtime state - reconstructed at each continue site.

    Uses pseudo-immutable semantics: replace entire fields rather than mutate.
    """

    status: AgentState = AgentState.IDLE
    messages: list[Message] = field(default_factory=list)

    # Turn control
    turn_number: int = 0
    max_turns: int = 50
    session_started_at: float = 0.0
    session_timeout_seconds: float = 1800.0

    # Recovery flags
    has_attempted_reactive_compact: bool = False

    # Pause signaling
    pause_requested: bool = False
    resume_event: Any = None

    # Error tracking
    last_error_type: str | None = None
    consecutive_errors: int = 0
    max_consecutive_errors: int = 3

    def reset_for_new_session(self) -> None:
        """Reset state for a new session."""
        self.status = AgentState.RUNNING
        self.messages = []
        self.turn_number = 0
        self.session_started_at = time.monotonic()
        self.has_attempted_reactive_compact = False
        self.pause_requested = False
        self.last_error_type = None
        self.consecutive_errors = 0

    def check_expired(self) -> str | None:
        """Check if session has expired. Returns reason or None."""
        if self.turn_number >= self.max_turns:
            return "max_turns_reached"

        elapsed = time.monotonic() - self.session_started_at
        if elapsed > self.session_timeout_seconds:
            return "session_timeout"

        return None

    def check_max_errors(self) -> bool:
        """Check if consecutive errors exceeded limit."""
        return self.consecutive_errors >= self.max_consecutive_errors
