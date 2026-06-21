"""Code execution tool for running Python scripts in a sandboxed subprocess."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from dawu_agent.security.sandbox import SandboxManager
from dawu_agent.tools.base import ConcurrencyMode, Tool, ToolCategory, ToolResult


class PythonExecuteTool(Tool):
    """Execute Python code in a sandboxed subprocess for complex data analysis.

    Writes code to a temporary script in the workspace, executes it via
    subprocess, and returns stdout/stderr. The script is cleaned up after
    execution.

    Use this when data_query's pandas expressions are insufficient, e.g.:
    - Multi-step data transformations
    - Statistical modeling (regression, clustering, etc.)
    - Complex aggregations with custom logic
    - Machine learning inference
    """

    # Default workspace directory for scripts
    WORKSPACE_DIR = "./workspace"

    @property
    def name(self) -> str:
        return "python_execute"

    @property
    def description(self) -> str:
        return (
            "Execute Python code in a sandboxed subprocess for complex data analysis "
            "and statistical computations. Use this when data_query is insufficient "
            "for multi-step transformations, statistical modeling, or custom logic.\n"
            "The code runs in an isolated subprocess with timeout protection.\n"
            "Output via print() will be captured and returned.\n"
            "Parameters:\n"
            "  - code (string, required): Python source code to execute\n"
            "  - timeout (integer, optional): Max execution seconds (default: 60)\n"
            "  - save_result (boolean, optional): Save stdout to a file (default: false)\n"
            "  - result_path (string, optional): Path to save result file\n"
            "Returns: stdout, stderr, and exit_code of the execution"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to execute. Use print() for output.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (default: 60)",
                },
                "save_result": {
                    "type": "boolean",
                    "description": "Whether to save stdout to a file for later use",
                },
                "result_path": {
                    "type": "string",
                    "description": "Path to save the result file (required if save_result is true)",
                },
            },
            "required": ["code"],
        }

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.DATA

    @property
    def concurrency_mode(self) -> ConcurrencyMode:
        # Writing files + executing subprocess — must be serialized
        return ConcurrencyMode.WRITE

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        code = arguments.get("code", "")
        if not code or not code.strip():
            return ToolResult.error("No code provided to execute")

        timeout = arguments.get("timeout", 60)
        save_result = arguments.get("save_result", False)
        result_path = arguments.get("result_path", "")

        if save_result and not result_path:
            return ToolResult.error("result_path is required when save_result is true")

        # Generate a unique script filename
        script_name = f"tmp_{uuid.uuid4().hex[:12]}.py"
        script_path = Path(self.WORKSPACE_DIR) / script_name

        # Ensure workspace directory exists
        script_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize sandbox manager for path validation and execution
        sandbox = SandboxManager(
            isolation_level="path",
            allowed_paths=["./workspace", "./data", "./uploads"],
            resource_limits={"cpu": "1.0", "memory": "512m", "timeout": str(timeout)},
        )

        try:
            # Validate the script path is within allowed directories
            path_decision = sandbox.validate_path(str(script_path), "write")
            if not path_decision.allowed:
                return ToolResult.error(f"Script path denied: {path_decision.reason}")

            # Write the code to the script file
            script_path.write_text(code, encoding="utf-8")

            # Execute via sandbox subprocess
            result = await sandbox.execute(
                command=f"python {script_path}",
                workdir=str(script_path.parent),
            )

            exit_code = result.get("exit_code", -1)
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")

            # Build output
            output_parts = []
            if stdout:
                output_parts.append(f"[stdout]\n{stdout.rstrip()}")
            if stderr:
                output_parts.append(f"[stderr]\n{stderr.rstrip()}")
            if not output_parts:
                output_parts.append("[no output]")

            output = f"Exit code: {exit_code}\n\n" + "\n\n".join(output_parts)

            # Save result to file if requested
            if save_result and exit_code == 0 and stdout:
                try:
                    out_path = Path(result_path)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(stdout, encoding="utf-8")
                    output += f"\n\nResult saved to: {result_path}"
                except Exception as e:
                    output += f"\n\nFailed to save result: {e}"

            return ToolResult.ok(output)

        except Exception as e:
            return ToolResult.error(f"Code execution failed: {e}")

        finally:
            # Clean up the temporary script file
            try:
                if script_path.exists():
                    script_path.unlink()
            except OSError:
                pass
