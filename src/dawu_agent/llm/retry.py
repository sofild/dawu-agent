"""Retry policy implementations with exponential backoff."""

from __future__ import annotations

import random

from dawu_agent.llm.base import IRetryPolicy


class ExponentialBackoffRetry(IRetryPolicy):
    """Exponential backoff retry with jitter.

    Retries on: 429 (rate limit), 5xx (server errors), network timeouts
    Does NOT retry: 4xx (client errors except 429)
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        self._max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if exception is retryable."""
        if attempt >= self._max_retries:
            return False

        error_str = str(exception).lower()

        # Check for rate limit (429)
        if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
            return True

        # Check for server errors (5xx)
        if any(f"{code}" in error_str for code in [500, 502, 503, 504]):
            return True

        # Check for network/timeout errors
        if any(
            keyword in error_str
            for keyword in ["timeout", "connection", "network", "temporarily unavailable"]
        ):
            return True

        # Client errors (4xx except 429) should not retry
        if "400" in error_str or "401" in error_str or "403" in error_str or "404" in error_str:
            return False

        # Default: retry on unknown errors (conservative)
        return True

    def backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with optional jitter."""
        delay = self.base_delay * (2 ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add ±25% jitter to prevent thundering herd
            delay = delay * (0.75 + random.random() * 0.5)

        return delay
