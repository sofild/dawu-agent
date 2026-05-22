"""Hook system with event bus for pre/post tool interception."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

HookCallback = Callable[..., Coroutine[Any, Any, Any]]


@dataclass
class HookHandle:
    """Handle for registered hook."""

    event_type: str
    callback: HookCallback
    priority: int
    id: str = field(default_factory=lambda: __import__("uuid").uuid4().hex[:8])


class HookSystem:
    """Event-driven hook system for agent lifecycle interception.

    Events: PreToolUse, PostToolUse, UserPromptSubmit, Stop, SessionStart
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[HookHandle]] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        event_type: str,
        callback: HookCallback,
        priority: int = 50,
    ) -> HookHandle:
        """Register a hook. Lower priority = earlier execution."""
        handle = HookHandle(event_type, callback, priority)

        async with self._lock:
            if event_type not in self._hooks:
                self._hooks[event_type] = []
            self._hooks[event_type].append(handle)
            self._hooks[event_type].sort(key=lambda h: h.priority)

        return handle

    async def unregister(self, handle: HookHandle) -> None:
        """Unregister a hook."""
        async with self._lock:
            if handle.event_type in self._hooks:
                self._hooks[handle.event_type] = [
                    h for h in self._hooks[handle.event_type] if h.id != handle.id
                ]

    async def execute_pre(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute PreToolUse hooks in priority order.

        Each hook receives the current input and returns modified input.
        Hook exceptions are isolated - one failing hook doesn't break others.
        """
        event_type = "PreToolUse"
        current_input = dict(tool_input)

        hooks = self._hooks.get(event_type, [])
        for handle in hooks:
            try:
                result = await handle.callback(
                    tool_name=tool_name,
                    tool_input=current_input,
                    context=context or {},
                )
                if isinstance(result, dict):
                    current_input = result
            except Exception as e:
                # Log but don't break the chain
                print(f"Hook {handle.id} failed: {e}")
                continue

        return current_input

    async def execute_post(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute PostToolUse hooks in priority order."""
        event_type = "PostToolUse"
        current_output = tool_output

        hooks = self._hooks.get(event_type, [])
        for handle in hooks:
            try:
                result = await handle.callback(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output=current_output,
                    context=context or {},
                )
                if result is not None:
                    current_output = result
            except Exception as e:
                print(f"Hook {handle.id} failed: {e}")
                continue

        return current_output

    async def execute_user_prompt(
        self,
        prompt_text: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Execute UserPromptSubmit hooks."""
        event_type = "UserPromptSubmit"
        current_text = prompt_text

        hooks = self._hooks.get(event_type, [])
        for handle in hooks:
            try:
                result = await handle.callback(
                    prompt_text=current_text,
                    context=context or {},
                )
                if isinstance(result, str):
                    current_text = result
            except Exception as e:
                print(f"Hook {handle.id} failed: {e}")
                continue

        return current_text

    async def execute_stop(
        self,
        reason: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Execute Stop hooks."""
        event_type = "Stop"

        hooks = self._hooks.get(event_type, [])
        for handle in hooks:
            try:
                await handle.callback(
                    reason=reason,
                    context=context or {},
                )
            except Exception as e:
                print(f"Hook {handle.id} failed: {e}")
                continue

    def list_hooks(self, event_type: str | None = None) -> list[HookHandle]:
        """List registered hooks."""
        if event_type:
            return list(self._hooks.get(event_type, []))
        return [h for hooks in self._hooks.values() for h in hooks]
