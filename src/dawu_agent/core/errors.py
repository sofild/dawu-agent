"""Unified error classification with retryable semantics (v4).

Every error path in the agent loop must classify its exception into one of
these five categories.  The classification drives retry count, backoff, and
whether the operation is safe to retry at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorKind(Enum):
    """Five canonical error categories (v4).

    Member values are stable identifiers used in logs and telemetry.
    """

    RETRIABLE = "retriable"
    """Transient failure (network blip, 500/502/504).  Retry with backoff."""

    NON_RETRIABLE = "non_retriable"
    """Client error (400/401/403/404/422).  Do NOT retry—payload is wrong."""

    CONTEXT_TOO_LONG = "context_too_long"
    """Prompt exceeds model window.  Compress context, then retry once."""

    RATE_LIMITED = "rate_limited"
    """429 / quota.  Backoff longer; optionally switch to fallback model."""

    VERIFICATION_FAILED = "verification_failed"
    """Tool executed but deterministic verifier rejected the result.
    Inject structured failure feedback and retry; escalate after N fails."""


@dataclass(frozen=True)
class ErrorClassification:
    """Result of classifying an exception."""

    kind: ErrorKind
    original_exception: Exception
    detail: str = ""
    # How many times this specific error kind may be retried before giving up.
    max_retries: int = 3
    # Whether the operation is idempotent (safe to retry without side effects).
    is_idempotent: bool = True
    # Suggested backoff in seconds (0 = retry immediately).
    backoff_seconds: float = 1.0


# ── Per-kind defaults ──────────────────────────────────────────────

_KIND_DEFAULTS: dict[ErrorKind, dict[str, Any]] = {
    ErrorKind.RETRIABLE: {
        "max_retries": 3,
        "is_idempotent": True,
        "backoff_seconds": 2.0,
    },
    ErrorKind.NON_RETRIABLE: {
        "max_retries": 0,
        "is_idempotent": False,
        "backoff_seconds": 0.0,
    },
    ErrorKind.CONTEXT_TOO_LONG: {
        "max_retries": 1,
        "is_idempotent": True,
        "backoff_seconds": 0.0,
    },
    ErrorKind.RATE_LIMITED: {
        "max_retries": 4,
        "is_idempotent": True,
        "backoff_seconds": 5.0,
    },
    ErrorKind.VERIFICATION_FAILED: {
        "max_retries": 3,
        "is_idempotent": False,
        "backoff_seconds": 0.0,
    },
}


def classify_error(exc: Exception) -> ErrorClassification:
    """Classify an exception into an :class:`ErrorKind`.

    Uses substring matching on the stringified exception.  This is intentionally
    heuristic—LLM gateway errors are inconsistent across providers, and we
    prefer over-classifying as retriable (safe) over under-classifying (drops
    recoverable sessions).
    """
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    # ── Non-retriable client errors (4xx) ──
    non_retriable_markers = (
        "param_error", "invalid_request", "400 ", "401 ", "403 ",
        "404 ", "422 ", "400-", "unauthorized", "forbidden",
        "api_key", "authentication",
    )
    if any(m in exc_str for m in non_retriable_markers):
        return _build(ErrorKind.NON_RETRIABLE, exc, exc_str)

    # ── Context too long ──
    context_markers = ("prompt too long", "413", "context_length", "max_tokens")
    if any(m in exc_str for m in context_markers):
        return _build(ErrorKind.CONTEXT_TOO_LONG, exc, exc_str)

    # ── Rate limited ──
    rate_markers = ("429", "rate limit", "quota", "too many requests")
    if any(m in exc_str for m in rate_markers):
        return _build(ErrorKind.RATE_LIMITED, exc, exc_str)

    # ── Model unavailable (503/502/504) → retriable ──
    retriable_markers = ("503", "502", "504", "500", "timeout", "timed out",
                         "connection", "eof", "reset", "unavailable")
    if any(m in exc_str for m in retriable_markers) or "timeout" in exc_type:
        return _build(ErrorKind.RETRIABLE, exc, exc_str)

    # ── Default: treat as retriable (safe default) ──
    return _build(ErrorKind.RETRIABLE, exc, exc_str)


def _build(kind: ErrorKind, exc: Exception, detail: str) -> ErrorClassification:
    defaults = _KIND_DEFAULTS[kind]
    return ErrorClassification(
        kind=kind,
        original_exception=exc,
        detail=detail[:500],
        max_retries=defaults["max_retries"],
        is_idempotent=defaults["is_idempotent"],
        backoff_seconds=defaults["backoff_seconds"],
    )
