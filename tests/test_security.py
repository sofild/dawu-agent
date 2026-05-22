"""Tests for security system."""

import pytest
from dawu_agent.security.permissions import PermissionManager, PermissionMode, Action
from dawu_agent.security.sandbox import SandboxManager


class TestPermissionManager:
    def test_hardcoded_deny(self):
        pm = PermissionManager()
        decision = pm.check_permission("bash", {"command": "sudo rm -rf /"})
        assert decision.action.value == "deny"
        assert "Hardcoded" in decision.reason

    def test_default_mode_allows_reads(self):
        pm = PermissionManager(PermissionMode.DEFAULT)
        decision = pm.check_permission("file_read", {"path": "./test.txt"})
        assert decision.action.value == "allow"

    def test_default_mode_asks_writes(self):
        pm = PermissionManager(PermissionMode.DEFAULT)
        decision = pm.check_permission("file_write", {"path": "./test.txt"})
        assert decision.action.value == "ask"

    def test_plan_mode_denies_writes(self):
        pm = PermissionManager(PermissionMode.PLAN)
        decision = pm.check_permission("file_write", {"path": "./test.txt"})
        assert decision.action.value == "deny"

    def test_user_rule_override(self):
        pm = PermissionManager()
        pm.add_rule("file_write", Action.ALLOW, 1, "tool")
        decision = pm.check_permission("file_write", {"path": "./test.txt"})
        assert decision.action.value == "allow"

    def test_deny_overrides_allow(self):
        pm = PermissionManager()
        # Hardcoded deny for config files should override user allow
        decision = pm.check_permission("file_write", {"path": "config/settings.yaml"})
        assert decision.action.value == "deny"


class TestSandboxManager:
    def test_validate_path_allows_whitelist(self):
        sm = SandboxManager(allowed_paths=["./workspace"])
        decision = sm.validate_path("./workspace/test.txt")
        assert decision.allowed

    def test_validate_path_denies_outside(self):
        sm = SandboxManager(allowed_paths=["./workspace"])
        decision = sm.validate_path("/etc/passwd")
        assert not decision.allowed

    def test_validate_command_detects_dangerous(self):
        sm = SandboxManager()
        decision = sm.validate_command("sudo rm -rf /")
        assert not decision.allowed
        assert decision.risk_score == 1.0

    def test_validate_command_allows_safe(self):
        sm = SandboxManager()
        decision = sm.validate_command("ls -la")
        assert decision.allowed
        assert decision.risk_score < 0.5
