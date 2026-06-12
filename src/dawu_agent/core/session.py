"""Session WAL (Write-Ahead Log) for immutable event logging."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from dawu_agent.core.events import AgentEvent


class SessionEventLog:
    """Append-only event log for session persistence and replay.

    Design constraints:
    - Events are written to disk before being yielded to callers
    - No delete/update/truncate operations
    - Supports replay to reconstruct AgentState
    """

    def __init__(
        self,
        session_id: str | None = None,
        log_dir: str = "sessions",
        detail_log_dir: str = "logs/running",
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{self.session_id}.jsonl"

        # Secondary detailed log under logs/running/ for human grep; one file
        # per session so every conversation has its own time-ordered JSONL.
        self.detail_log_dir = Path(detail_log_dir)
        self.detail_log_dir.mkdir(parents=True, exist_ok=True)
        self.detail_log_file = self.detail_log_dir / f"{self.session_id}.jsonl"

        self._events: list[dict[str, Any]] = []

    def append(self, event: AgentEvent) -> None:
        """Append event to log (write to disk first, then memory)."""
        event_dict = self._event_to_dict(event)

        # Write to disk first (WAL semantics)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")
            f.flush()

        # Mirror the same event into the per-session detailed log
        with open(self.detail_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")
            f.flush()

        # Then append to memory
        self._events.append(event_dict)

    def _event_to_dict(self, event: AgentEvent) -> dict[str, Any]:
        """Convert event to dictionary."""
        result = {"type": event.type, "timestamp": event.timestamp}

        # Add event-specific fields
        if hasattr(event, "old"):
            result["old"] = event.old
        if hasattr(event, "new"):
            result["new"] = event.new
        if hasattr(event, "reason"):
            result["reason"] = event.reason
        if hasattr(event, "turn"):
            result["turn"] = event.turn
        if hasattr(event, "duration"):
            result["duration"] = event.duration
        if hasattr(event, "content"):
            result["content"] = event.content
        if hasattr(event, "is_final"):
            result["is_final"] = event.is_final
        if hasattr(event, "tool_name"):
            result["tool_name"] = event.tool_name
        if hasattr(event, "tool_input"):
            result["tool_input"] = event.tool_input
        if hasattr(event, "tool_use_id"):
            result["tool_use_id"] = event.tool_use_id
        if hasattr(event, "is_error"):
            result["is_error"] = event.is_error
        if hasattr(event, "error_type"):
            result["error_type"] = event.error_type
        if hasattr(event, "action"):
            result["action"] = event.action
        if hasattr(event, "detail"):
            result["detail"] = event.detail
        if hasattr(event, "text"):
            result["text"] = event.text

        return result

    def replay(self) -> list[dict[str, Any]]:
        """Replay all events from log file."""
        events = []
        if self.log_file.exists():
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        return events

    def get_messages(self) -> list[dict[str, str]]:
        """Extract message history from replayed events."""
        messages = []
        for event in self.replay():
            if event["type"] == "user_message":
                messages.append({"role": "user", "content": event["content"]})
            elif event["type"] == "assistant_text" and event.get("is_final"):
                messages.append({"role": "assistant", "content": event["content"]})
            elif event["type"] == "tool_result" and not event.get("is_error"):
                messages.append({"role": "user", "content": event["content"]})
        return messages
