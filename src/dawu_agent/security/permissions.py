"""Permission model with 5 modes and 7-level rule hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PermissionMode(Enum):
    """5 permission modes."""

    DEFAULT = "default"
    ACCEPT_EDITS = "accept_edits"
    BYPASS_PERMISSIONS = "bypass_permissions"
    PLAN = "plan"
    ACCEPT_ALL = "accept_all"


class Action(Enum):
    """Permission actions."""

    DENY = "deny"
    ALLOW = "allow"
    ASK = "ask"


@dataclass
class PermissionRule:
    """Single permission rule."""

    pattern: str
    action: Action
    level: int  # 0-6, lower = higher priority
    scope: str  # tool | path | network | process
    target: str | None = None  # e.g., file glob pattern

    def matches(self, tool_name: str, params: dict[str, Any]) -> bool:
        """Check if this rule matches the tool call."""
        import fnmatch

        if self.scope == "tool":
            return fnmatch.fnmatch(tool_name, self.pattern)
        elif self.scope == "path" and "path" in params:
            return fnmatch.fnmatch(params["path"], self.pattern)
        elif self.scope == "network" and "url" in params:
            return fnmatch.fnmatch(params["url"], self.pattern)
        return False

    @property
    def priority(self) -> int:
        return self.level


@dataclass
class PermissionDecision:
    """Result of permission check."""

    action: Action
    matched_rule: PermissionRule | None = None
    reason: str = ""
    decision_chain: list[dict[str, Any]] = field(default_factory=list)


class PermissionManager:
    """Manages 7-level permission rule hierarchy.

    Priority (highest to lowest):
    0. Hardcoded Deny
    1. Settings Deny
    2. Settings Ask
    3. Hook Allow
    4. Hook Deny
    5. YOLO Auto-approve
    6. Default Fallback
    """

    # Hardcoded denies - immutable
    HARDCODED_DENIES = [
        PermissionRule("rm -rf /", Action.DENY, 0, "process"),
        PermissionRule("sudo*", Action.DENY, 0, "process"),
        PermissionRule("chmod 777*", Action.DENY, 0, "process"),
        PermissionRule("mkfs*", Action.DENY, 0, "process"),
        PermissionRule("dd*", Action.DENY, 0, "process"),
        PermissionRule("file_write", Action.DENY, 0, "path", target="config/*"),
        PermissionRule("file_write", Action.DENY, 0, "path", target=".env*"),
        PermissionRule("file_write", Action.DENY, 0, "path", target="*.pem"),
        PermissionRule("file_write", Action.DENY, 0, "path", target="*.key"),
    ]

    def __init__(self, default_mode: PermissionMode = PermissionMode.DEFAULT) -> None:
        self.default_mode = default_mode
        self._rules: list[PermissionRule] = []
        self._yolo_enabled = False

        # Load hardcoded denies
        self._rules.extend(self.HARDCODED_DENIES)

    def add_rule(self, pattern: str, action: Action, level: int, scope: str, target: str | None = None) -> None:
        """Add a user-defined rule (level 1-3 only)."""
        if level <= 0 or level >= 6:
            raise ValueError("User rules must be level 1-3")
        self._rules.append(PermissionRule(pattern, action, level, scope, target))
        # Sort by priority
        self._rules.sort(key=lambda r: r.priority)

    def remove_rule(self, pattern: str) -> None:
        """Remove a user-defined rule."""
        self._rules = [
            r for r in self._rules
            if not (r.pattern == pattern and r.level >= 1)
        ]

    def check_permission(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        session_context: dict[str, Any] | None = None,
    ) -> PermissionDecision:
        """Check permission through 7-level hierarchy."""
        decision_chain = []

        # Special mode handling
        if self.default_mode == PermissionMode.BYPASS_PERMISSIONS:
            return PermissionDecision(
                action=Action.ALLOW,
                reason="bypass_permissions mode",
            )

        if self.default_mode == PermissionMode.PLAN:
            # Plan mode: only allow read operations
            if tool_name not in ("file_read", "file_list", "data_query"):
                return PermissionDecision(
                    action=Action.DENY,
                    reason="plan mode: write/execute operations denied",
                )

        # Traverse rules in priority order
        for rule in self._rules:
            if rule.matches(tool_name, tool_input):
                decision_chain.append({
                    "level": rule.level,
                    "pattern": rule.pattern,
                    "action": rule.action.value,
                    "scope": rule.scope,
                })

                # Hardcoded deny (level 0) - absolute
                if rule.level == 0 and rule.action == Action.DENY:
                    return PermissionDecision(
                        action=Action.DENY,
                        matched_rule=rule,
                        reason=f"Hardcoded deny: {rule.pattern}",
                        decision_chain=decision_chain,
                    )

                # Settings deny (level 1)
                if rule.level == 1 and rule.action == Action.DENY:
                    return PermissionDecision(
                        action=Action.DENY,
                        matched_rule=rule,
                        reason=f"Settings deny: {rule.pattern}",
                        decision_chain=decision_chain,
                    )

                # Settings ask (level 2)
                if rule.level == 2 and rule.action == Action.ASK:
                    return PermissionDecision(
                        action=Action.ASK,
                        matched_rule=rule,
                        reason=f"Settings ask: {rule.pattern}",
                        decision_chain=decision_chain,
                    )

                # Settings allow (level 3)
                if rule.level == 3 and rule.action == Action.ALLOW:
                    return PermissionDecision(
                        action=Action.ALLOW,
                        matched_rule=rule,
                        reason=f"Settings allow: {rule.pattern}",
                        decision_chain=decision_chain,
                    )

        # No rule matched - use default
        if self.default_mode == PermissionMode.ACCEPT_ALL:
            return PermissionDecision(
                action=Action.ALLOW,
                reason="accept_all mode",
                decision_chain=decision_chain,
            )

        if self.default_mode == PermissionMode.ACCEPT_EDITS:
            if tool_name in ("file_write", "file_edit"):
                return PermissionDecision(
                    action=Action.ALLOW,
                    reason="accept_edits mode",
                    decision_chain=decision_chain,
                )

        # Default: ask for write/execute, allow for read
        if tool_name in ("file_write", "file_edit", "bash", "python_execute"):
            return PermissionDecision(
                action=Action.ASK,
                reason="default mode: write/execute requires confirmation",
                decision_chain=decision_chain,
            )

        return PermissionDecision(
            action=Action.ALLOW,
            reason="default mode: read operations allowed",
            decision_chain=decision_chain,
        )

    def get_effective_rules(self) -> list[PermissionRule]:
        """Return all effective rules."""
        return list(self._rules)
