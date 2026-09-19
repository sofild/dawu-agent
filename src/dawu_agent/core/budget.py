"""Budget and four-level degradation chain (v4).

Tracks token usage and cost per task/session.  When the budget is exceeded,
triggers a four-level degradation chain:

  1. MODEL_DOWNGRADE — switch to a cheaper/faster model
  2. SKIP_NON_CRITICAL — skip optional operations (DSPy observe, memory recall)
  3. REDUCE_CONTEXT — force context compression
  4. ABORT — stop the session gracefully
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class DegradationLevel(Enum):
    """Four-level degradation chain, applied in order."""

    NONE = 0
    MODEL_DOWNGRADE = 1
    SKIP_NON_CRITICAL = 2
    REDUCE_CONTEXT = 3
    ABORT = 4


@dataclass
class BudgetUsage:
    """Snapshot of budget consumption at a point in time."""

    tokens_used: int = 0
    tokens_budget: int = 0
    cost_used: float = 0.0
    cost_budget: float = 0.0
    turns_used: int = 0
    turns_budget: int = 0
    wall_time_seconds: float = 0.0
    wall_time_budget: float = 0.0

    @property
    def token_ratio(self) -> float:
        if self.tokens_budget <= 0:
            return 0.0
        return self.tokens_used / self.tokens_budget

    @property
    def is_exceeded(self) -> bool:
        return (
            (self.tokens_budget > 0 and self.tokens_used >= self.tokens_budget)
            or (self.cost_budget > 0 and self.cost_used >= self.cost_budget)
            or (self.turns_budget > 0 and self.turns_used >= self.turns_budget)
            or (self.wall_time_budget > 0 and self.wall_time_seconds >= self.wall_time_budget)
        )


@dataclass
class BudgetConfig:
    """Configuration for budget enforcement.

    All fields default to 0 (unlimited).  Only set the fields you want enforced.
    """

    max_tokens: int = 0
    max_cost_usd: float = 0.0
    max_turns: int = 0
    max_wall_time_seconds: float = 0.0
    # Ratio at which to start degradation (e.g. 0.8 = 80% of budget).
    degradation_trigger_ratio: float = 0.8


class Budget:
    """Per-session budget tracker with four-level degradation.

    Usage in the agent loop::

        budget = Budget(BudgetConfig(max_tokens=200_000, max_cost_usd=5.0))
        budget.record_tokens(1500)
        budget.record_cost(0.02)
        budget.record_turn()

        level = budget.check()  # → DegradationLevel
        if level == DegradationLevel.ABORT:
            # stop session
        elif level == DegradationLevel.MODEL_DOWNGRADE:
            # switch to cheaper model
    """

    def __init__(self, config: BudgetConfig | None = None) -> None:
        self.config = config or BudgetConfig()
        self._tokens_used = 0
        self._cost_used = 0.0
        self._turns_used = 0
        self._start_time = time.monotonic()
        self._current_level = DegradationLevel.NONE
        self._degraded_once: set[DegradationLevel] = set()

    # ── Recording ────────────────────────────────────────────────

    def record_tokens(self, count: int) -> None:
        self._tokens_used += count

    def record_cost(self, cost: float) -> None:
        self._cost_used += cost

    def record_turn(self) -> None:
        self._turns_used += 1

    # ── Querying ────────────────────────────────────────────────

    @property
    def usage(self) -> BudgetUsage:
        return BudgetUsage(
            tokens_used=self._tokens_used,
            tokens_budget=self.config.max_tokens,
            cost_used=self._cost_used,
            cost_budget=self.config.max_cost_usd,
            turns_used=self._turns_used,
            turns_budget=self.config.max_turns,
            wall_time_seconds=time.monotonic() - self._start_time,
            wall_time_budget=self.config.max_wall_time_seconds,
        )

    @property
    def current_degradation(self) -> DegradationLevel:
        return self._current_level

    @property
    def is_aborted(self) -> bool:
        return self._current_level == DegradationLevel.ABORT

    def check(self) -> DegradationLevel:
        """Check budget and return the current (possibly upgraded) degradation level.

        The level monotonically increases—once degraded to MODEL_DOWNGRADE,
        it never goes back to NONE within the same session.
        """
        usage = self.usage

        # Hard abort: budget fully exceeded
        if usage.is_exceeded:
            self._upgrade_to(DegradationLevel.ABORT)
            return self._current_level

        ratio = self._max_ratio(usage)
        if ratio >= self.config.degradation_trigger_ratio:
            # Progressive degradation
            if self._current_level < DegradationLevel.MODEL_DOWNGRADE:
                self._upgrade_to(DegradationLevel.MODEL_DOWNGRADE)
            elif self._current_level < DegradationLevel.SKIP_NON_CRITICAL:
                self._upgrade_to(DegradationLevel.SKIP_NON_CRITICAL)
            elif self._current_level < DegradationLevel.REDUCE_CONTEXT:
                self._upgrade_to(DegradationLevel.REDUCE_CONTEXT)

        return self._current_level

    def should_skip_non_critical(self) -> bool:
        """True if non-critical operations (DSPy observe, memory recall) should be skipped."""
        return self.check() >= DegradationLevel.SKIP_NON_CRITICAL

    def should_force_compress(self) -> bool:
        """True if context compression should be forced even if under threshold."""
        return self.check() >= DegradationLevel.REDUCE_CONTEXT

    def abort_reason(self) -> str | None:
        """Return the reason for abort, or None if not aborted."""
        if not self.is_aborted:
            return None
        usage = self.usage
        if usage.tokens_budget > 0 and usage.tokens_used >= usage.tokens_budget:
            return f"token budget exceeded ({usage.tokens_used}/{usage.tokens_budget})"
        if usage.cost_budget > 0 and usage.cost_used >= usage.cost_budget:
            return f"cost budget exceeded (${usage.cost_used:.4f}/${usage.cost_budget:.4f})"
        if usage.turns_budget > 0 and usage.turns_used >= usage.turns_budget:
            return f"turn budget exceeded ({usage.turns_used}/{usage.turns_budget})"
        if usage.wall_time_budget > 0 and usage.wall_time_seconds >= usage.wall_time_budget:
            return f"time budget exceeded ({usage.wall_time_seconds:.1f}s/{usage.wall_time_budget:.1f}s)"
        return "budget exceeded"

    # ── Internal ────────────────────────────────────────────────

    def _upgrade_to(self, level: DegradationLevel) -> None:
        if level.value > self._current_level.value:
            self._current_level = level

    @staticmethod
    def _max_ratio(usage: BudgetUsage) -> float:
        ratios = []
        if usage.tokens_budget > 0:
            ratios.append(usage.token_ratio)
        if usage.cost_budget > 0:
            ratios.append(usage.cost_used / usage.cost_budget)
        if usage.turns_budget > 0:
            ratios.append(usage.turns_used / usage.turns_budget)
        if usage.wall_time_budget > 0:
            ratios.append(usage.wall_time_seconds / usage.wall_time_budget)
        return max(ratios) if ratios else 0.0
