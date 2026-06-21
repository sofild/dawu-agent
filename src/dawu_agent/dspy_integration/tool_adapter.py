"""Adapt existing Tool instances to DSPy Tool format."""

from __future__ import annotations

import dspy

from dawu_agent.tools.base import Tool


def adapt_tool(tool: Tool) -> dspy.Tool:
    """Convert a dawu_agent Tool to a DSPy Tool.

    DSPy Tool expects a plain Python function with type hints and docstring.
    We create a wrapper that delegates to the tool's execute method.
    """

    async def tool_wrapper(**kwargs) -> str:
        result = await tool.execute(kwargs)
        if result.success:
            return str(result.data) if result.data else ""
        return f"Error: {result.error}"

    tool_wrapper.__name__ = tool.name
    tool_wrapper.__doc__ = tool.description

    return dspy.Tool(tool_wrapper)


def adapt_registry_tools(tool_registry) -> list[dspy.Tool]:
    """Convert all tools in a ToolRegistry to DSPy Tools."""
    import contextlib

    dspy_tools: list[dspy.Tool] = []
    for tool in tool_registry.all_tools():
        with contextlib.suppress(Exception):
            dspy_tools.append(adapt_tool(tool))
    return dspy_tools
