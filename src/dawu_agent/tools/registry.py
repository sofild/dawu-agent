"""Tool registry with lazy loading, partitioning, and batch execution."""

from __future__ import annotations

import asyncio
from typing import Any

from dawu_agent.tools.base import ConcurrencyMode, Tool, ToolResult


class ToolRegistry:
    """Central registry for tool management and execution.

    Features:
    - Dynamic registration/deregistration
    - Lazy loading for LLM context (send only relevant tools)
    - Concurrency partitioning (read-only parallel, write serial)
    - Result truncation for large outputs
    """

    def __init__(self, max_concurrent: int = 5, max_result_chars: int = 8000) -> None:
        self._tools: dict[str, Tool] = {}
        self._core_tools: set[str] = set()
        self._max_concurrent = max_concurrent
        self._max_result_chars = max_result_chars
        self._recently_used: list[str] = []
        self._usage_count: dict[str, int] = {}
        # Per-tool trigger keywords (used to force-include a tool when the user
        # prompt contains any of its declared triggers, regardless of how LLM
        # would otherwise bias the selection).
        self._triggers: dict[str, set[str]] = {}

    def register(self, tool: Tool, core: bool = False) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        if core:
            self._core_tools.add(tool.name)

    def register_triggers(self, tool_name: str, triggers: list[str]) -> None:
        """Register a set of trigger keywords for context-aware tool selection."""
        if not triggers:
            return
        self._triggers.setdefault(tool_name, set()).update(t.strip() for t in triggers if t.strip())

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        self._tools.pop(name, None)
        self._core_tools.discard(name)

    def get(self, name: str) -> Tool | None:
        """Get tool by name."""
        return self._tools.get(name)

    def list_all(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_for_llm(self, context_hint: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Get tool definitions for LLM context.

        Strategy:
        1. Always include core tools
        2. Include recently used tools
        3. If context_hint provided, include relevant tools by keyword matching
        4. Respect limit
        """
        selected = set(self._core_tools)

        # Add recently used
        selected.update(self._recently_used[-5:])

        # Context-based selection
        if context_hint:
            hint_lower = context_hint.lower()
            for name, tool in self._tools.items():
                desc_lower = tool.description.lower()
                if any(keyword in hint_lower for keyword in desc_lower.split()[:10]):
                    selected.add(name)

            # Trigger-based force-include (used by Skill tools). If the user
            # prompt contains any trigger word registered for a tool, that
            # tool is always made available to the LLM, bypassing heuristics.
            for name, triggers in self._triggers.items():
                if any(t and t in context_hint for t in triggers):
                    selected.add(name)

        # Sort by usage frequency (LRU-like)
        sorted_tools = sorted(
            selected,
            key=lambda n: self._usage_count.get(n, 0),
            reverse=True,
        )

        # Apply limit
        limited = sorted_tools[:limit]

        return [self._tools[name].to_llm_definition() for name in limited if name in self._tools]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a single tool."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(f"Tool '{name}' not found")

        # Track usage
        self._recently_used.append(name)
        self._usage_count[name] = self._usage_count.get(name, 0) + 1

        try:
            result = await tool.execute(arguments)
            # Apply truncation if needed
            if isinstance(result.data, str) and len(result.data) > self._max_result_chars:
                result = self._truncate_result(result)
            return result
        except Exception as e:
            return ToolResult.error(f"Tool execution failed: {e}")

    async def execute_batch(self, calls: list[dict[str, Any]]) -> list[ToolResult]:
        """Execute multiple tools with partitioning.

        Partitioning algorithm:
        1. Separate read-only and write tools
        2. Execute read-only concurrently (up to max_concurrent)
        3. Execute write tools serially
        """
        reads = []
        writes = []

        for call in calls:
            name = call.get("name", "")
            tool = self._tools.get(name)
            if tool is None:
                continue

            if tool.concurrency_mode == ConcurrencyMode.READ_ONLY:
                reads.append(call)
            else:
                writes.append(call)

        results = {}

        # Phase 1: Execute reads concurrently in batches
        for i in range(0, len(reads), self._max_concurrent):
            batch = reads[i : i + self._max_concurrent]
            batch_tasks = [
                self._execute_single(call["name"], call.get("arguments", {}))
                for call in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for call, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results[call["name"]] = ToolResult.error(str(result))
                else:
                    results[call["name"]] = result

        # Phase 2: Execute writes serially
        for call in writes:
            name = call["name"]
            results[name] = await self._execute_single(name, call.get("arguments", {}))

        # Return in original order
        return [results.get(call["name"], ToolResult.error("Missing result")) for call in calls]

    async def _execute_single(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute single tool with error handling."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(f"Tool '{name}' not found")

        self._recently_used.append(name)
        self._usage_count[name] = self._usage_count.get(name, 0) + 1

        try:
            result = await tool.execute(arguments)
            if isinstance(result.data, str) and len(result.data) > self._max_result_chars:
                result = self._truncate_result(result)
            return result
        except Exception as e:
            return ToolResult.error(f"Execution failed: {e}")

    def _truncate_result(self, result: ToolResult) -> ToolResult:
        """Truncate long string results: keep head + tail, summarize middle."""
        if not isinstance(result.data, str):
            return result

        text = result.data
        max_chars = self._max_result_chars

        head_len = int(max_chars * 0.4)
        tail_len = int(max_chars * 0.4)

        head = text[:head_len]
        tail = text[-tail_len:]
        omitted = len(text) - head_len - tail_len

        summary = f"\n... [Omitted {omitted} characters] ...\n"

        truncated_data = head + summary + tail

        return ToolResult(
            success=result.success,
            data=truncated_data,
            error=result.error,
            truncated=True,
            truncated_info={
                "original_length": len(text),
                "truncated_length": len(truncated_data),
                "omitted": omitted,
            },
        )
