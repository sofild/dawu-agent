"""Enterprise Agent with state machine, 7 continue sites, and streaming."""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncGenerator

from dawu_agent.config.loader import Settings
from dawu_agent.context.compression import CompressionPipeline
from dawu_agent.context.memory import MemoryManager
from dawu_agent.core.events import (
    AgentEvent,
    AssistantTextEvent,
    CompactionEvent,
    ErrorEvent,
    FinalResponseEvent,
    StateChangeEvent,
    ToolResultEvent,
    ToolUseEvent,
    TurnEndEvent,
    TurnStartEvent,
    UserMessageEvent,
)
from dawu_agent.core.session import SessionEventLog
from dawu_agent.core.state import AgentRunState, AgentState
from dawu_agent.llm.base import ILLMClient, Message, ToolCall
from dawu_agent.llm.factory import LLMClientFactory
from dawu_agent.multi.coordinator import Coordinator
from dawu_agent.multi.sub_agent import SubAgent, SubAgentConfig
from dawu_agent.observability.telemetry import TelemetryManager
from dawu_agent.security.audit import AuditLogger
from dawu_agent.security.hooks import HookSystem
from dawu_agent.security.permissions import PermissionManager, PermissionMode
from dawu_agent.security.sandbox import SandboxManager
from dawu_agent.tools.base import ToolResult
from dawu_agent.tools.builtin.data_tools import (
    DataQueryTool,
    DataVisualizeTool,
    ReportGenerateTool,
)
from dawu_agent.tools.builtin.search_tools import (
    BaiduSearchTool,
    BingSearchTool,
    TavilySearchTool,
)
from dawu_agent.tools.builtin.file_tools import (
    FileListTool,
    FileReadTool,
    FileWriteTool,
)
from dawu_agent.tools.mcp.adapter import MCPAdapter
from dawu_agent.tools.registry import ToolRegistry


class Agent:
    """Enterprise Agent implementing the core reasoning loop.

    Architecture:
    - State machine with explicit states (idle, running, paused, expired, error)
    - 7 continue sites for controlled flow interruption/resumption
    - Async generator for streaming events
    - Session WAL for immutable event logging
    - Multi-agent coordination via Coordinator
    - 6-layer security defense
    """

    def __init__(self, settings: Settings, telemetry: TelemetryManager) -> None:
        self.settings = settings
        self.telemetry = telemetry
        self._state = AgentRunState(max_turns=settings.agent.max_turns)
        self._session_log: SessionEventLog | None = None
        self._llm_client: ILLMClient | None = None
        self._tool_registry = ToolRegistry(
            max_concurrent=settings.tools.max_concurrent_tools
            if hasattr(settings, "tools") else 5,
        )
        # Phase 5: Context management
        self._compression_pipeline: CompressionPipeline | None = None
        self._memory_manager: MemoryManager | None = None
        # Phase 6: Security
        self._permission_manager = PermissionManager(
            default_mode=PermissionMode(settings.permissions.default_mode)
            if hasattr(settings, "permissions") else PermissionMode.DEFAULT
        )
        self._hook_system = HookSystem()
        self._sandbox_manager: SandboxManager | None = None
        self._audit_logger: AuditLogger | None = None
        # Phase 9: Multi-agent
        self._coordinator: Coordinator | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize agent subsystems."""
        if self._initialized:
            return

        self.telemetry.logger.info("agent.initializing")

        # Initialize LLM client
        self._llm_client = LLMClientFactory.create(self.settings.llm)

        # Initialize session log
        self._session_log = SessionEventLog()

        # Initialize compression pipeline (Phase 5)
        if self._llm_client:
            self._compression_pipeline = CompressionPipeline(
                llm_client=self._llm_client,
                token_counter=self._llm_client.count_tokens,
            )

        # Initialize memory manager (Phase 5)
        if self.settings.enable_vector_memory:
            self._memory_manager = MemoryManager()
            await self._memory_manager.initialize()

        # Initialize security (Phase 6)
        if self.settings.enable_sandbox:
            self._sandbox_manager = SandboxManager()
        self._audit_logger = AuditLogger()

        # Load permission rules from config
        if hasattr(self.settings, "permissions") and hasattr(self.settings.permissions, "rules"):
            for rule in self.settings.permissions.rules.get("settings", []):
                from dawu_agent.security.permissions import Action
                self._permission_manager.add_rule(
                    pattern=rule.get("pattern", ""),
                    action=Action(rule.get("action", "ask")),
                    level=1,
                    scope=rule.get("scope", "tool"),
                    target=rule.get("target"),
                )

        # Initialize multi-agent coordinator (Phase 9)
        if self.settings.enable_multi_agent and self._llm_client:
            self._coordinator = Coordinator(
                llm_client=self._llm_client,
                max_sub_agents=self.settings.multi_agent.max_sub_agents
                if hasattr(self.settings, "multi_agent") else 5,
            )

        # Register built-in tools
        self._register_builtin_tools()

        # Register time-aware hook for search tools
        from dawu_agent.tools.builtin.time_aware_hook import search_time_aware_hook

        await self._hook_system.register(
            "PreToolUse",
            search_time_aware_hook,
            priority=30,
        )

        # Connect MCP servers if configured
        if hasattr(self.settings, "tools") and hasattr(self.settings.tools, "mcp_servers"):
            await self._connect_mcp_servers()

        self._initialized = True
        self._state.status = AgentState.IDLE
        self.telemetry.logger.info(
            "agent.initialized",
            session_id=self._session_log.session_id,
        )

    def _register_builtin_tools(self) -> None:
        """Register all built-in tools."""
        core_tools = [
            FileReadTool(),
            FileWriteTool(),
            FileListTool(),
            DataQueryTool(),
            DataVisualizeTool(),
            ReportGenerateTool(),
            BingSearchTool(),
            BaiduSearchTool(),
            TavilySearchTool(),
        ]
        for tool in core_tools:
            self._tool_registry.register(tool, core=True)

    async def _connect_mcp_servers(self) -> None:
        """Connect to configured MCP servers."""
        for server_config in self.settings.tools.mcp_servers:
            if not server_config.get("enabled", True):
                continue
            try:
                adapter = MCPAdapter(server_config)
                await adapter.connect()
                mcp_tools = adapter.wrap_tools()
                for tool in mcp_tools:
                    self._tool_registry.register(tool)
                self.telemetry.logger.info(
                    "mcp.connected",
                    server=server_config.get("name"),
                    tools_count=len(mcp_tools),
                )
            except Exception as e:
                self.telemetry.logger.error(
                    "mcp.connection_failed",
                    server=server_config.get("name"),
                    error=str(e),
                )

    async def shutdown(self) -> None:
        """Gracefully shutdown agent."""
        self.telemetry.logger.info("agent.shutting_down")

        # Trigger auto-dream before shutdown
        if self._memory_manager:
            try:
                stats = await self._memory_manager.auto_dream(self._llm_client)
                self.telemetry.logger.info("memory.auto_dream", stats=stats)
            except Exception as e:
                self.telemetry.logger.error("memory.auto_dream_failed", error=str(e))

        self._state.status = AgentState.IDLE

    async def run_turn(self, user_input: str) -> str:
        """Execute a single agent turn (synchronous wrapper)."""
        events = []
        async for event in self.run_stream(user_input):
            if isinstance(event, AssistantTextEvent):
                events.append(event.content)
            elif isinstance(event, FinalResponseEvent):
                events.append(event.text)
            elif isinstance(event, ErrorEvent):
                raise RuntimeError(f"{event.error_type}: {event.detail}")
        return "".join(events)

    async def run_stream(self, user_input: str) -> AsyncGenerator[AgentEvent, None]:
        """Execute agent turn with streaming events and 7 continue sites."""
        if not self._initialized or self._llm_client is None or self._session_log is None:
            yield ErrorEvent(error_type="not_initialized", action="abort")
            return

        # Entry state validation
        if self._state.status == AgentState.RUNNING:
            yield ErrorEvent(error_type="already_running", action="abort")
            return

        # Initialize new session
        self._state.reset_for_new_session()
        yield StateChangeEvent(old="idle", new="running")
        self._session_log.append(StateChangeEvent(old="idle", new="running"))

        # Record user message
        user_msg = UserMessageEvent(content=user_input)
        self._session_log.append(user_msg)
        yield user_msg

        # Inject relevant memories (Phase 5)
        memory_context = ""
        if self._memory_manager:
            memories = await self._memory_manager.get_relevant_memories(
                messages=[Message(role="user", content=user_input)],
                current_task=user_input,
            )
            memory_context = self._memory_manager.format_for_context(memories)

        # Build system prompt with memory context and current time
        from datetime import datetime
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Asia/Shanghai")
        current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        weekday = datetime.now(tz).strftime("%A")

        system_content = (
            f"You are Dawu Agent, an enterprise data analysis assistant. "
            f"The current date and time is {current_time} ({weekday}). "
            f"When the user mentions time-related words (today, yesterday, this week, etc.) "
            f"and you need to search, always pass the appropriate time filter parameter."
        )
        if memory_context:
            system_content += f"\n\n{memory_context}"

        self._state.messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_input),
        ]

        # =====================================================================
        # NEVER-EXITING MAIN LOOP
        # =====================================================================
        while True:
            # Check pause
            if self._state.pause_requested:
                self._state.status = AgentState.PAUSED
                yield StateChangeEvent(old="running", new="paused")
                await self._wait_for_resume()
                self._state.status = AgentState.RUNNING
                yield StateChangeEvent(old="paused", new="running")

            # Check expiration
            expire_reason = self._state.check_expired()
            if expire_reason:
                self._state.status = AgentState.EXPIRED
                yield StateChangeEvent(old="running", new="expired", reason=expire_reason)
                self._session_log.append(
                    StateChangeEvent(old="running", new="expired", reason=expire_reason)
                )
                break

            # Turn start
            self._state.turn_number += 1
            turn_start = time.monotonic()
            turn_start_event = TurnStartEvent(turn=self._state.turn_number)
            self._session_log.append(turn_start_event)
            yield turn_start_event

            try:
                # ---------------------------------------------------------------
                # CONTINUE SITE 1: Proactive Compaction (Phase 5)
                # ---------------------------------------------------------------
                token_count = self._llm_client.count_tokens(self._state.messages)
                max_tokens = self.settings.agent.max_tokens_per_session
                if self._compression_pipeline and self._compression_pipeline.should_compress(
                    self._state.messages, max_tokens
                ):
                    yield CompactionEvent(detail=f"Token count {token_count} exceeds threshold")
                    result = self._compression_pipeline.compact(
                        self._state.messages, max_tokens, reason="proactive"
                    )
                    self._state.messages = result.messages
                    if self._audit_logger:
                        self._audit_logger.log_compression(
                            level=result.level_used,
                            input_tokens=token_count,
                            output_tokens=self._llm_client.count_tokens(result.messages),
                            duration_ms=0,
                        )

                # ---------------------------------------------------------------
                # Call LLM with streaming
                # ---------------------------------------------------------------
                tools = self._tool_registry.get_for_llm(context_hint=user_input)
                accumulated_text = []
                tool_calls_buffer: list[ToolCall] = []

                async for chunk in self._llm_client.chat_stream(
                    messages=self._state.messages,
                    tools=tools if tools else None,
                ):
                    if chunk.content:
                        accumulated_text.append(chunk.content)
                        text_event = AssistantTextEvent(content=chunk.content)
                        self._session_log.append(text_event)
                        yield text_event

                    if chunk.tool_call:
                        tool_calls_buffer.append(chunk.tool_call)
                        tool_event = ToolUseEvent(
                            tool_name=chunk.tool_call.name,
                            tool_input=chunk.tool_call.arguments,
                            tool_use_id=chunk.tool_call.id,
                        )
                        self._session_log.append(tool_event)
                        yield tool_event

                    if chunk.finish_reason:
                        final_text = "".join(accumulated_text)
                        if final_text:
                            final_event = AssistantTextEvent(content=final_text, is_final=True)
                            self._session_log.append(final_event)

                # ---------------------------------------------------------------
                # Handle tool calls (CONTINUE SITE 7)
                # ---------------------------------------------------------------
                if tool_calls_buffer:
                    for tc in tool_calls_buffer:
                        # Phase 6: Permission check
                        perm_decision = self._permission_manager.check_permission(
                            tc.name, tc.arguments
                        )
                        if self._audit_logger:
                            self._audit_logger.log_permission_check(
                                tool_name=tc.name,
                                tool_input=tc.arguments,
                                decision=perm_decision.action.value,
                                reason=perm_decision.reason,
                                decision_chain=perm_decision.decision_chain,
                            )

                        if perm_decision.action.value == "deny":
                            result = ToolResult.error(
                                f"Permission denied: {perm_decision.reason}"
                            )
                        else:
                            # Phase 6: Hook pre-execution
                            modified_input = await self._hook_system.execute_pre(
                                tc.name, tc.arguments, context={"user_input": user_input}
                            )
                            result = await self._tool_registry.execute(tc.name, modified_input)
                            # Phase 6: Hook post-execution
                            result = await self._hook_system.execute_post(
                                tc.name, modified_input, result
                            )

                        result_event = ToolResultEvent(
                            tool_use_id=tc.id,
                            content=str(result.data) if result.data else result.error or "",
                            is_error=not result.success,
                        )
                        self._session_log.append(result_event)
                        yield result_event

                        # Append tool result to messages (pseudo-immutable)
                        self._state.messages = [
                            *self._state.messages,
                            Message(
                                role="tool",
                                content=str(result.data) if result.data else result.error or "",
                                tool_call_id=tc.id,
                            ),
                        ]

                    # Reset consecutive errors on successful tool execution
                    self._state.consecutive_errors = 0
                    self._emit_turn_end(turn_start)
                    continue  # Back to while True

                # ---------------------------------------------------------------
                # No tool calls - final response
                # ---------------------------------------------------------------
                final_text = "".join(accumulated_text)
                if final_text:
                    self._state.messages = [
                        *self._state.messages,
                        Message(role="assistant", content=final_text),
                    ]

                self._state.status = AgentState.IDLE
                yield StateChangeEvent(old="running", new="idle", reason="complete")
                yield FinalResponseEvent(text=final_text)
                self._emit_turn_end(turn_start)
                break

            # ================================================================
            # ERROR RECOVERY: 6 Continue Sites
            # ================================================================
            except Exception as e:
                error_str = str(e).lower()

                # CONTINUE SITE 2: Prompt Too Long
                if "prompt too long" in error_str or "413" in error_str:
                    if self._state.has_attempted_reactive_compact:
                        # Already tried - do aggressive snip
                        self._state.messages = self._emergency_snip(self._state.messages)
                        self._state.has_attempted_reactive_compact = False
                    else:
                        if self._compression_pipeline:
                            result = self._compression_pipeline.compact(
                                self._state.messages, max_tokens, reason="reactive"
                            )
                            self._state.messages = result.messages
                        else:
                            self._state.messages = self._reactive_compact(self._state.messages)
                        self._state.has_attempted_reactive_compact = True

                    self._state.consecutive_errors += 1
                    yield ErrorEvent(
                        error_type="prompt_too_long",
                        turn=self._state.turn_number,
                        action="compact",
                    )
                    continue

                # CONTINUE SITE 3: Max Output Tokens
                if "max_tokens" in error_str or "length" in error_str:
                    self._state.messages = [
                        *self._state.messages,
                        Message(role="user", content="Please continue from where you left off."),
                    ]
                    self._state.consecutive_errors += 1
                    yield ErrorEvent(
                        error_type="max_output_tokens",
                        turn=self._state.turn_number,
                        action="continue_prompt",
                    )
                    continue

                # CONTINUE SITE 4: Model Unavailable / Fallback
                if any(code in error_str for code in ["503", "429", "rate limit"]):
                    if self.settings.llm.fallback_provider:
                        self._llm_client = LLMClientFactory.create(
                            self.settings.llm,
                            provider_override=self.settings.llm.fallback_provider,
                        )
                        yield ErrorEvent(
                            error_type="model_unavailable",
                            turn=self._state.turn_number,
                            action="fallback",
                            detail=f"Switched to {self.settings.llm.fallback_provider}",
                        )
                    else:
                        await asyncio.sleep(min(2 ** self._state.consecutive_errors, 60))

                    self._state.consecutive_errors += 1
                    continue

                # Generic retriable error
                if self._state.check_max_errors():
                    self._state.status = AgentState.ERROR
                    self._state.last_error_type = type(e).__name__
                    yield StateChangeEvent(
                        old="running",
                        new="error",
                        reason=f"max_consecutive_errors: {e}",
                    )
                    break

                self._state.consecutive_errors += 1
                yield ErrorEvent(
                    error_type="retriable",
                    turn=self._state.turn_number,
                    action="retry",
                    detail=str(e),
                )
                await asyncio.sleep(min(2 ** self._state.consecutive_errors, 30))
                continue

    async def run_multi_agent(self, task: str) -> str:
        """Execute task using multi-agent coordination."""
        if not self._coordinator:
            return "Multi-agent coordination not enabled"

        # Decompose task
        subtasks = await self._coordinator.decompose(task)

        # Execute in parallel
        results = await self._coordinator.execute_parallel(
            subtasks,
            sub_agent_factory=self._create_sub_agent,
        )

        # Synthesize results
        return await self._coordinator.synthesize(results, task)

    def _create_sub_agent(self, role: str) -> SubAgent:
        """Factory for creating sub-agents."""
        config = SubAgentConfig(
            role=role,
            model=self.settings.llm.model,
            provider=self.settings.llm.provider,
        )
        return SubAgent(config)

    def _emit_turn_end(self, turn_start: float) -> None:
        """Emit turn end event and reset turn-level flags."""
        elapsed = time.monotonic() - turn_start
        event = TurnEndEvent(turn=self._state.turn_number, duration=elapsed)
        if self._session_log:
            self._session_log.append(event)

    def _reactive_compact(self, messages: list[Message]) -> list[Message]:
        """Reactive compaction: remove oldest non-system messages."""
        if len(messages) <= 4:
            return messages
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]
        return [*system_msgs, *other_msgs[-6:]]

    def _emergency_snip(self, messages: list[Message]) -> list[Message]:
        """Emergency snip: aggressive truncation."""
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]
        return [*system_msgs, *other_msgs[-4:]]

    async def _wait_for_resume(self) -> None:
        """Wait for external resume signal."""
        while self._state.pause_requested:
            await asyncio.sleep(0.1)

    @property
    def state(self) -> AgentState:
        return self._state.status
