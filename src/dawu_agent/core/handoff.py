"""Cross-session handoff for long-running agent state persistence.

When an agent session is interrupted (budget exhaustion, crash, explicit
pause), a HandoffFile captures the minimal state needed to resume the
conversation in a new process or after a restart.  This is the "long-time
running / cross-session handoff" mechanism described in the v4 architecture.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dawu_agent.core.budget import Budget
from dawu_agent.core.session import SessionEventLog
from dawu_agent.core.state import AgentRunState, AgentState
from dawu_agent.llm.base import Message, ToolCall


@dataclass
class HandoffFile:
    """Serializable snapshot of agent state for cross-session resume.

    Fields are intentionally minimal -- only what is needed to reconstruct
    the conversation context and continue.  Tool-call details and verbose
    outputs are compressed by the session history compressor before being
    stored here.
    """

    session_id: str
    created_at: float
    messages: list[Message]
    turn_number: int
    max_turns: int
    session_started_at: float
    session_timeout_seconds: float
    budget_tokens_used: int = 0
    budget_cost_used: float = 0.0
    budget_turns_used: int = 0
    last_error_type: str | None = None
    consecutive_errors: int = 0
    has_attempted_reactive_compact: bool = False
    # Free-form metadata for debugging / observability
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "messages": [_message_to_dict(m) for m in self.messages],
            "turn_number": self.turn_number,
            "max_turns": self.max_turns,
            "session_started_at": self.session_started_at,
            "session_timeout_seconds": self.session_timeout_seconds,
            "budget_tokens_used": self.budget_tokens_used,
            "budget_cost_used": self.budget_cost_used,
            "budget_turns_used": self.budget_turns_used,
            "last_error_type": self.last_error_type,
            "consecutive_errors": self.consecutive_errors,
            "has_attempted_reactive_compact": self.has_attempted_reactive_compact,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandoffFile:
        """Deserialize from a JSON-compatible dict."""
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            messages=[_message_from_dict(m) for m in data.get("messages", [])],
            turn_number=data.get("turn_number", 0),
            max_turns=data.get("max_turns", 50),
            session_started_at=data.get("session_started_at", 0.0),
            session_timeout_seconds=data.get("session_timeout_seconds", 1800.0),
            budget_tokens_used=data.get("budget_tokens_used", 0),
            budget_cost_used=data.get("budget_cost_used", 0.0),
            budget_turns_used=data.get("budget_turns_used", 0),
            last_error_type=data.get("last_error_type"),
            consecutive_errors=data.get("consecutive_errors", 0),
            has_attempted_reactive_compact=data.get(
                "has_attempted_reactive_compact", False
            ),
            metadata=data.get("metadata", {}),
        )


# ── Message serialization helpers ──────────────────────────────────


def _message_to_dict(msg: Message) -> dict[str, Any]:
    """Serialize a Message to a JSON-compatible dict."""
    return {
        "role": msg.role,
        "content": msg.content,
        "name": msg.name,
        "tool_calls": [
            {
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            }
            for tc in (msg.tool_calls or [])
        ]
        or None,
        "tool_call_id": msg.tool_call_id,
    }


def _message_from_dict(data: dict[str, Any]) -> Message:
    """Deserialize a Message from a JSON-compatible dict."""
    tool_calls = None
    if data.get("tool_calls"):
        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["name"],
                arguments=tc.get("arguments", {}),
            )
            for tc in data["tool_calls"]
        ]
    return Message(
        role=data["role"],
        content=data.get("content", ""),
        name=data.get("name"),
        tool_calls=tool_calls,
        tool_call_id=data.get("tool_call_id"),
    )


# ── HandoffManager ─────────────────────────────────────────────────


class HandoffManager:
    """Manages reading and writing HandoffFile snapshots to disk.

    Files are stored as JSON under ``handoff_dir`` with the naming
    convention ``{session_id}.handoff.json``.  The directory is created
    lazily on first write.
    """

    def __init__(self, handoff_dir: str = "handoffs") -> None:
        self.handoff_dir = Path(handoff_dir)

    def _ensure_dir(self) -> None:
        self.handoff_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        return self.handoff_dir / f"{session_id}.handoff.json"

    def save(self, handoff: HandoffFile) -> Path:
        """Write a handoff snapshot to disk.  Returns the file path."""
        self._ensure_dir()
        path = self._path_for(handoff.session_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(handoff.to_dict(), ensure_ascii=False, indent=2))
            f.flush()
        return path

    def load(self, session_id: str) -> HandoffFile:
        """Read a handoff snapshot from disk.

        Raises:
            FileNotFoundError: if no handoff file exists for the session.
            ValueError: if the file is corrupted or malformed.
        """
        path = self._path_for(session_id)
        if not path.exists():
            raise FileNotFoundError(
                f"No handoff file for session {session_id}: {path}"
            )
        with open(path, encoding="utf-8") as f:
            try:
                data = json.loads(f.read())
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupted handoff file {path}: {exc}") from exc
        return HandoffFile.from_dict(data)

    def list_sessions(self) -> list[str]:
        """Return session IDs that have handoff files, newest first."""
        if not self.handoff_dir.exists():
            return []
        sessions: list[tuple[float, str]] = []
        for path in self.handoff_dir.glob("*.handoff.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.loads(f.read())
                sessions.append(
                    (data.get("created_at", 0.0), data.get("session_id", path.stem))
                )
            except (json.JSONDecodeError, KeyError):
                continue
        sessions.sort(key=lambda x: x[0], reverse=True)
        return [sid for _, sid in sessions]

    def cleanup_old(self, max_age_seconds: float = 86400.0) -> int:
        """Remove handoff files older than ``max_age_seconds``.

        Returns the number of files removed.
        """
        if not self.handoff_dir.exists():
            return 0
        now = time.time()
        removed = 0
        for path in self.handoff_dir.glob("*.handoff.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.loads(f.read())
                created_at = data.get("created_at", 0.0)
            except (json.JSONDecodeError, KeyError):
                # Corrupted file — treat as infinitely old so it gets cleaned
                created_at = 0.0
            if now - created_at > max_age_seconds:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed


# ── Resume helper ──────────────────────────────────────────────────


async def resume_from_handoff(
    agent: Any,
    session_id: str,
    handoff_dir: str = "handoffs",
) -> HandoffFile:
    """Resume an agent session from a handoff file.

    Reads the handoff snapshot, reconstructs the agent's internal state
    (messages, turn counter, budget usage, error flags), and returns the
    HandoffFile so the caller can inspect the resumed context.

    Args:
        agent: An initialized Agent instance (``await agent.initialize()``
            must have been called first).
        session_id: The session ID to resume.
        handoff_dir: Directory containing handoff files.

    Returns:
        The loaded HandoffFile.

    Raises:
        FileNotFoundError: if no handoff file exists for the session.
        ValueError: if the file is corrupted.
    """
    manager = HandoffManager(handoff_dir=handoff_dir)
    handoff = manager.load(session_id)

    # Reconstruct AgentRunState from the handoff
    state: AgentRunState = agent._state
    state.status = AgentState.IDLE
    state.messages = handoff.messages
    state.turn_number = handoff.turn_number
    state.max_turns = handoff.max_turns
    # Reset wall clock for the new process; the original started_at is
    # preserved in the handoff for audit but monotonic time cannot cross
    # process boundaries.
    state.session_started_at = time.monotonic()
    state.session_timeout_seconds = handoff.session_timeout_seconds
    state.has_attempted_reactive_compact = handoff.has_attempted_reactive_compact
    state.last_error_type = handoff.last_error_type
    state.consecutive_errors = handoff.consecutive_errors

    # Reconstruct budget usage so degradation continues monotonically
    budget: Budget | None = getattr(agent, "_budget", None)
    if budget is not None:
        budget._tokens_used = handoff.budget_tokens_used
        budget._cost_used = handoff.budget_cost_used
        budget._turns_used = handoff.budget_turns_used

    # Reconnect session log to the original session ID so events append
    # to the same JSONL file.
    session_log: SessionEventLog | None = getattr(agent, "_session_log", None)
    if session_log is not None:
        session_log.session_id = handoff.session_id
        session_log.log_file = session_log.log_dir / f"{handoff.session_id}.jsonl"
        session_log.detail_log_file = (
            session_log.detail_log_dir / f"{handoff.session_id}.jsonl"
        )

    return handoff
