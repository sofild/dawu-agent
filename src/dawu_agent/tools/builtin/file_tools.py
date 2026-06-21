"""File read/write tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dawu_agent.tools.base import ConcurrencyMode, Tool, ToolCategory, ToolResult


def _resolve_output_path(path: Path, default_dir: str) -> Path:
    """Place bare filenames into a default subdirectory of the project root.

    Absolute paths and explicit relative paths are kept as-is.
    """
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        return Path(default_dir) / path
    return path


class FileReadTool(Tool):
    """Read file contents."""

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. "
            "Use this when you need to examine file contents, source code, data files, or logs. "
            "Do NOT use this for directories - use file_list instead.\n"
            "Parameters:\n"
            "  - path (string, required): Absolute or relative path to the file\n"
            "Returns: File contents as string"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
            },
            "required": ["path"],
        }

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.TASK

    @property
    def concurrency_mode(self) -> ConcurrencyMode:
        return ConcurrencyMode.READ_ONLY

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = Path(arguments["path"])
            if not path.exists():
                return ToolResult.error(f"File not found: {path}")
            if path.is_dir():
                return ToolResult.error(f"Path is a directory, not a file: {path}")

            content = path.read_text(encoding="utf-8")
            return ToolResult.ok(content)
        except Exception as e:
            return ToolResult.error(f"Failed to read file: {e}")


class FileWriteTool(Tool):
    """Write content to a file."""

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return (
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Use this to save analysis results, generated reports, or processed data. "
            "Be careful not to overwrite important files.\n"
            "Parameters:\n"
            "  - path (string, required): Path to write to. If only a filename is given, "
            "it will be saved under workspace/.\n"
            "  - content (string, required): Content to write\n"
            "Returns: Success confirmation with file path"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to write the file to",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        }

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.TASK

    @property
    def concurrency_mode(self) -> ConcurrencyMode:
        return ConcurrencyMode.WRITE

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = _resolve_output_path(Path(arguments["path"]), "workspace")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(arguments["content"], encoding="utf-8")
            return ToolResult.ok(f"File written successfully: {path}")
        except Exception as e:
            return ToolResult.error(f"Failed to write file: {e}")


class FileListTool(Tool):
    """List directory contents."""

    @property
    def name(self) -> str:
        return "file_list"

    @property
    def description(self) -> str:
        return (
            "List files and directories in a given path. "
            "Use this to explore the workspace or find files.\n"
            "Parameters:\n"
            "  - path (string, required): Directory path to list\n"
            "Returns: List of files and directories"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list",
                },
            },
            "required": ["path"],
        }

    @property
    def concurrency_mode(self) -> ConcurrencyMode:
        return ConcurrencyMode.READ_ONLY

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = Path(arguments["path"])
            if not path.exists():
                return ToolResult.error(f"Directory not found: {path}")
            if not path.is_dir():
                return ToolResult.error(f"Path is not a directory: {path}")

            items = []
            for item in sorted(path.iterdir()):
                item_type = "dir" if item.is_dir() else "file"
                items.append(f"[{item_type}] {item.name}")

            return ToolResult.ok("\n".join(items))
        except Exception as e:
            return ToolResult.error(f"Failed to list directory: {e}")
