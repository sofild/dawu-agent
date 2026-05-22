"""Audit logging for security events."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class AuditLogger:
    """Immutable audit log for security events.

    Design: Session = Audit Log. Every security decision is recorded
    in append-only JSONL format.
    """

    def __init__(self, log_dir: str = "logs/audit") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = str(uuid.uuid4())
        self._log_file = self.log_dir / f"audit_{self._session_id}.jsonl"
        self._seq = 0

    def _write(self, event: dict[str, Any]) -> None:
        """Append event to audit log."""
        self._seq += 1
        event["seq"] = self._seq
        event["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        event["session_id"] = self._session_id

        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()

    def log_permission_check(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        decision: str,
        reason: str,
        decision_chain: list[dict[str, Any]] | None = None,
    ) -> None:
        """Log permission check result."""
        self._write({
            "type": "PERMISSION_CHECK",
            "layer": 1,
            "tool_name": tool_name,
            "tool_input_hash": hash(json.dumps(tool_input, sort_keys=True)) & 0xFFFFFFFF,
            "decision": decision,
            "reason": reason,
            "decision_chain": decision_chain or [],
        })

    def log_hook_execution(
        self,
        event_type: str,
        hook_id: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Log hook execution."""
        self._write({
            "type": "HOOK_EXECUTION",
            "layer": 3,
            "event_type": event_type,
            "hook_id": hook_id,
            "success": success,
            "error": error,
        })

    def log_sandbox_decision(
        self,
        decision_type: str,
        allowed: bool,
        path: str | None = None,
        command: str | None = None,
        reason: str = "",
    ) -> None:
        """Log sandbox validation decision."""
        self._write({
            "type": "SANDBOX_DECISION",
            "layer": 4,
            "decision_type": decision_type,
            "allowed": allowed,
            "path": path,
            "command": command,
            "reason": reason,
        })

    def log_tool_call(
        self,
        tool_name: str,
        params_hash: str,
        result_tokens: int,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Log tool execution summary."""
        self._write({
            "type": "TOOL_CALL",
            "tool_name": tool_name,
            "params_hash": params_hash,
            "result_tokens": result_tokens,
            "success": success,
            "duration_ms": duration_ms,
        })

    def log_compression(
        self,
        level: int,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
    ) -> None:
        """Log context compression event."""
        self._write({
            "type": "COMPRESSION",
            "level": level,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_freed": input_tokens - output_tokens,
            "duration_ms": duration_ms,
        })

    def log_security_alert(
        self,
        alert_type: str,
        severity: str,
        details: dict[str, Any],
    ) -> None:
        """Log security alert."""
        self._write({
            "type": "SECURITY_ALERT",
            "alert_type": alert_type,
            "severity": severity,
            "details": details,
        })
