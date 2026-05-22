"""MCP adapter for connecting to external MCP servers."""

from __future__ import annotations

from typing import Any

from dawu_agent.tools.base import Tool, ToolResult


class MCPAdapter:
    """Adapter for Model Context Protocol servers.

    Wraps external MCP tools as internal Tool instances.
    """

    def __init__(self, server_config: dict[str, Any]) -> None:
        self.server_config = server_config
        self.name = server_config.get("name", "unnamed")
        self.transport = server_config.get("transport", "stdio")
        self.command = server_config.get("command", "")
        self._client: Any = None
        self._tools: list[dict[str, Any]] = []

    async def connect(self) -> None:
        """Establish connection to MCP server."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as e:
            raise ImportError("mcp SDK not installed. Run: pip install mcp") from e

        if self.transport == "stdio":
            params = StdioServerParameters(command=self.command, args=[])
            self._client = stdio_client(params)
            await self._client.__aenter__()

            # List available tools
            session = ClientSession(self._client)
            await session.initialize()
            tools_response = await session.list_tools()
            self._tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.inputSchema,
                }
                for tool in tools_response.tools
            ]
        else:
            raise ValueError(f"Unsupported MCP transport: {self.transport}")

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None

    def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from MCP server."""
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Call a tool on the MCP server."""
        if not self._client:
            return ToolResult.error("MCP client not connected")

        try:
            from mcp import ClientSession
            session = ClientSession(self._client)
            result = await session.call_tool(name, arguments)

            # Convert MCP result to ToolResult
            content = "\n".join(
                item.text if hasattr(item, "text") else str(item)
                for item in result.content
            )
            return ToolResult.ok(content)
        except Exception as e:
            return ToolResult.error(f"MCP tool call failed: {e}")

    def wrap_tools(self) -> list[Tool]:
        """Wrap MCP tools as internal Tool instances."""
        wrapped = []
        for tool_def in self._tools:
            wrapped.append(MCPToolWrapper(self, tool_def))
        return wrapped


class MCPToolWrapper(Tool):
    """Wrapper that exposes an MCP remote tool as an internal Tool."""

    def __init__(self, adapter: MCPAdapter, tool_def: dict[str, Any]) -> None:
        self._adapter = adapter
        self._tool_def = tool_def

    @property
    def name(self) -> str:
        return self._tool_def["name"]

    @property
    def description(self) -> str:
        return self._tool_def.get("description", f"MCP tool: {self.name}")

    @property
    def input_schema(self) -> dict[str, Any]:
        return self._tool_def.get("schema", {"type": "object", "properties": {}})

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return await self._adapter.call_tool(self.name, arguments)
