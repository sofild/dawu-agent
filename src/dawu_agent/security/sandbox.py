"""Sandbox manager with optional Docker-based isolation.

Supports three isolation levels:
- none: No sandbox, direct execution (for local dev without Docker)
- path: Path whitelist validation only
- docker: Full Docker container isolation
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


class PathDecision:
    """Result of path validation."""

    def __init__(self, allowed: bool, resolved_path: str = "", reason: str = "") -> None:
        self.allowed = allowed
        self.resolved_path = resolved_path
        self.reason = reason


class CommandDecision:
    """Result of command validation."""

    def __init__(
        self,
        allowed: bool,
        sanitized_command: str = "",
        risk_score: float = 0.0,
        reason: str = "",
    ) -> None:
        self.allowed = allowed
        self.sanitized_command = sanitized_command
        self.risk_score = risk_score
        self.reason = reason


class SandboxManager:
    """Sandbox manager with path validation and optional Docker isolation.

    Features:
    - Path whitelist validation with symlink resolution
    - Dangerous command pattern detection
    - Docker container execution (optional)
    - Direct subprocess execution (when Docker unavailable)
    - Resource limits (CPU, memory, timeout)
    """

    # Dangerous command patterns
    DANGEROUS_PATTERNS = [
        "rm -rf /",
        "sudo",
        "chmod 777",
        "mkfs",
        "dd if=",
        "> /dev/",
        "curl | sh",
        "wget | sh",
        "bash -c",
        "eval(",
        "exec(",
    ]

    def __init__(
        self,
        isolation_level: str = "path",
        allowed_paths: list[str] | None = None,
        denied_patterns: list[str] | None = None,
        docker_image: str = "dawu-sandbox:latest",
        resource_limits: dict[str, str] | None = None,
    ) -> None:
        self.isolation_level = isolation_level
        self.allowed_paths = allowed_paths or ["./workspace", "./data", "./uploads"]
        self.denied_patterns = denied_patterns or ["*.pem", "*.key", ".env*"]
        self.docker_image = docker_image
        self.resource_limits = resource_limits or {
            "cpu": "1.0",
            "memory": "512m",
            "timeout": "300",
        }
        self._docker_available: bool | None = None
        self._docker_client: Any = None

    def _check_docker_available(self) -> bool:
        """Check if Docker is available on this system."""
        if self._docker_available is not None:
            return self._docker_available

        try:
            import docker
            self._docker_client = docker.from_env()
            self._docker_client.ping()
            self._docker_available = True
        except Exception:
            self._docker_available = False

        return self._docker_available

    def validate_path(self, path: str, operation_type: str = "read") -> PathDecision:
        """Validate path against whitelist.

        Resolves symlinks and .. traversal before checking.
        """
        try:
            resolved = Path(path).resolve()
        except (OSError, ValueError) as e:
            return PathDecision(False, reason=f"Invalid path: {e}")

        # Check denied patterns
        for pattern in self.denied_patterns:
            import fnmatch
            if fnmatch.fnmatch(resolved.name, pattern):
                return PathDecision(
                    False,
                    str(resolved),
                    f"Path matches denied pattern: {pattern}",
                )

        # Check allowed paths
        for allowed in self.allowed_paths:
            allowed_resolved = Path(allowed).resolve()
            try:
                resolved.relative_to(allowed_resolved)
                return PathDecision(True, str(resolved))
            except ValueError:
                continue

        return PathDecision(
            False,
            str(resolved),
            f"Path outside allowed directories: {self.allowed_paths}",
        )

    def validate_command(self, command_string: str) -> CommandDecision:
        """Validate command for dangerous patterns."""
        cmd_lower = command_string.lower()

        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.lower() in cmd_lower:
                return CommandDecision(
                    False,
                    risk_score=1.0,
                    reason=f"Dangerous pattern detected: {pattern}",
                )

        # Check for pipe and redirect (moderate risk)
        risk_score = 0.0
        if "|" in command_string:
            risk_score += 0.3
        if ">" in command_string or "<" in command_string:
            risk_score += 0.3
        if ";" in command_string or "&&" in command_string:
            risk_score += 0.2

        return CommandDecision(
            True,
            command_string,
            risk_score,
            "Command passed basic validation",
        )

    async def execute(
        self,
        command: str,
        workdir: str = "./workspace",
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute command with appropriate isolation level."""
        # Validate command
        cmd_decision = self.validate_command(command)
        if not cmd_decision.allowed:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command denied: {cmd_decision.reason}",
                "exit_code": -1,
            }

        # Choose execution method based on isolation level
        if self.isolation_level == "docker" and self._check_docker_available():
            return await self._execute_docker(command, workdir, env)
        else:
            return await self._execute_subprocess(command, workdir, env)

    async def _execute_docker(
        self,
        command: str,
        workdir: str,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute in Docker container."""
        try:
            container = self._docker_client.containers.run(
                self.docker_image,
                command=shlex.split(command),
                working_dir="/workspace",
                volumes={
                    str(Path(workdir).resolve()): {"bind": "/workspace", "mode": "rw"},
                },
                mem_limit=self.resource_limits.get("memory", "512m"),
                cpu_quota=int(float(self.resource_limits.get("cpu", "1.0")) * 100000),
                network_mode="none",
                detach=True,
            )

            import time
            timeout = int(self.resource_limits.get("timeout", "300"))
            start = time.time()

            while time.time() - start < timeout:
                container.reload()
                if container.status != "running":
                    break
                await __import__("asyncio").sleep(0.1)

            logs = container.logs().decode("utf-8", errors="replace")
            exit_code = container.attrs["State"]["ExitCode"]
            container.remove(force=True)

            return {
                "success": exit_code == 0,
                "stdout": logs,
                "stderr": "",
                "exit_code": exit_code,
            }

        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Docker execution failed: {e}",
                "exit_code": -1,
            }

    async def _execute_subprocess(
        self,
        command: str,
        workdir: str,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute via subprocess (no Docker)."""
        try:
            merged_env = {**os.environ, **(env or {})}
            timeout = int(self.resource_limits.get("timeout", "300"))

            process = await __import__("asyncio").create_subprocess_shell(
                command,
                stdout=__import__("asyncio").subprocess.PIPE,
                stderr=__import__("asyncio").subprocess.PIPE,
                cwd=workdir,
                env=merged_env,
            )

            stdout, stderr = await __import__("asyncio").wait_for(
                process.communicate(), timeout=timeout
            )

            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": process.returncode or 0,
            }

        except __import__("asyncio").TimeoutError:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "exit_code": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Subprocess execution failed: {e}",
                "exit_code": -1,
            }

    async def execute_file_read(self, path: str) -> dict[str, Any]:
        """Execute file read in sandbox context."""
        path_decision = self.validate_path(path, "read")
        if not path_decision.allowed:
            return {
                "success": False,
                "content": "",
                "error": path_decision.reason,
            }

        try:
            content = Path(path_decision.resolved_path).read_text(encoding="utf-8")
            return {
                "success": True,
                "content": content,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "content": "",
                "error": str(e),
            }

    async def execute_file_write(self, path: str, content: str) -> dict[str, Any]:
        """Execute file write in sandbox context."""
        path_decision = self.validate_path(path, "write")
        if not path_decision.allowed:
            return {
                "success": False,
                "error": path_decision.reason,
            }

        try:
            target = Path(path_decision.resolved_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {
                "success": True,
                "path": str(target),
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
