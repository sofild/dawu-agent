"""Enterprise Agent with state machine, 7 continue sites, and streaming."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator


def _short_args(arguments: dict[str, Any], max_len: int = 80) -> str:
    """Format a tool-call arguments dict as a short human-readable string.

    Used by _compress_session_history to produce compact entries like
    `query_school(school="武汉理工", keyword="图书馆")` instead of dumping
    raw JSON.
    """
    if not arguments:
        return ""
    parts: list[str] = []
    for k, v in arguments.items():
        if isinstance(v, str):
            sv = v if len(v) <= 30 else v[:27] + "..."
            parts.append(f'{k}="{sv}"')
        elif isinstance(v, (int, float, bool)):
            parts.append(f"{k}={v}")
        elif v is None:
            parts.append(f"{k}=None")
        else:
            # dict / list — just give a length hint
            try:
                parts.append(f"{k}=<{type(v).__name__}·{len(v)}>")
            except Exception:
                parts.append(f"{k}=<{type(v).__name__}>")
    s = ", ".join(parts)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _classify_outcome(content: str) -> dict[str, Any]:
    """Classify a tool result into a short outcome label.

    Returns:
        dict with keys:
          - kind: "success" | "empty" | "error" | "wrong_param" | "unknown"
          - summary: short Chinese phrase suitable for log/summary
          - hint: optional short data sample (for successful results)
    """
    if not content:
        return {"kind": "empty", "summary": "空结果", "hint": None}

    low = content.lower()
    # Wrong parameter / schema rejection
    if "does not accept parameter" in low or "unexpected keyword argument" in low:
        return {"kind": "wrong_param", "summary": "参数错误（不要再用同样的参数）", "hint": None}
    if "is not in list" in low or "not allowed" in low:
        return {"kind": "wrong_param", "summary": "参数不被允许", "hint": None}
    if "missing" in low and "argument" in low:
        return {"kind": "error", "summary": "缺少必填参数", "hint": None}

    # Explicit error markers
    is_error = (
        content.startswith("Error")
        or "Traceback" in content
        or '"_status": "error"' in content
        or '"success": false' in content
        or "失败" in content
        or "错误" in content
        or "异常" in content
    )
    if is_error:
        # Take first 80 chars as the failure detail
        head = content.replace("\n", " ")[:80]
        return {"kind": "error", "summary": f"失败：{head}", "hint": None}

    # Empty / no data
    stripped = content.strip()
    if stripped in ("[]", "{}", "null", "None", '""', ""):
        return {"kind": "empty", "summary": "无数据", "hint": None}
    if "没有数据" in content or "未找到" in content or "not found" in low:
        return {"kind": "empty", "summary": "无数据", "hint": None}

    # Success — try to extract a short data sample
    hint: str | None = None
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            for key in ("data", "rows", "result", "items"):
                if key in data and data[key]:
                    hint = f"{key}: {json.dumps(data[key], ensure_ascii=False)[:120]}"
                    break
            if not hint:
                hint = json.dumps(data, ensure_ascii=False)[:120]
        elif isinstance(data, list) and data:
            hint = json.dumps(data[:1], ensure_ascii=False)[:120]
    except (json.JSONDecodeError, TypeError, ValueError):
        # Not JSON — just take a textual head
        hint = stripped[:120]

    return {"kind": "success", "summary": "成功", "hint": hint}


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
from dawu_agent.tools.builtin.code_execute_tool import PythonExecuteTool
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
        # DSPy integration
        self._use_dspy: bool = getattr(settings, "use_dspy", True)
        self._agent_reasoner: Any = None
        self._dspy_lm: Any = None
        # Holds background DSPy observation tasks with a strong reference;
        # otherwise Python GC may drop them mid-flight and produce
        # "Task was destroyed but it is pending!" warnings.
        self._dspy_pending_tasks: set[asyncio.Task] = set()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize agent subsystems."""
        if self._initialized:
            return

        self.telemetry.logger.info("agent.initializing")

        # Initialize LLM client
        self._llm_client = LLMClientFactory.create(self.settings.llm)
        self._primary_llm_client = self._llm_client  # keep ref for recovery

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

        # Configure DSPy integration
        if self._use_dspy and self._llm_client:
            try:
                from dawu_agent.dspy_integration import configure_dspy
                self._dspy_lm = configure_dspy(
                    self._llm_client, self.settings.llm.model
                )

                # Load optimized module if available, otherwise use default
                import os
                optimized_path = os.path.join(
                    os.path.dirname(__file__), "..", "dspy_optimized.json"
                )
                if os.path.exists(optimized_path):
                    from dawu_agent.dspy_integration.optimizer import load_optimized_module
                    self._agent_reasoner = load_optimized_module(optimized_path)
                else:
                    from dawu_agent.dspy_integration.modules import AgentReasoner
                    self._agent_reasoner = AgentReasoner()

                self.telemetry.logger.info("dspy.configured")
            except Exception as e:
                self._use_dspy = False
                self.telemetry.logger.warning("dspy.config_failed", error=str(e))

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
            PythonExecuteTool(),
            BingSearchTool(),
            BaiduSearchTool(),
            TavilySearchTool(),
        ]
        for tool in core_tools:
            self._tool_registry.register(tool, core=True)

        # Auto-discover SKILL.md-based skills from ./skills and register each
        # as a single tool. Trigger keywords are stored on the registry so
        # that `get_for_llm(context_hint=...)` will force-include a skill
        # tool when the user prompt contains any of its trigger words.
        try:
            from dawu_agent.tools.skill_loader import SkillLoader
            loaded = SkillLoader(skills_dir="skills").register_all(self._tool_registry)
            self.telemetry.logger.info("skills.loaded", count=loaded)
        except Exception as e:
            self.telemetry.logger.error("skills.load_failed", error=str(e))

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

    async def run_stream(
        self,
        user_input: str,
        *,
        continue_session: bool = False,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute agent turn with streaming events and 7 continue sites.

        Args:
            user_input: The user's message.
            continue_session: If True, append the user message to the
                existing conversation history instead of starting a new
                session.  This allows follow-up queries (e.g. "继续")
                to retain full context from the previous interaction.
        """
        if not self._initialized or self._llm_client is None or self._session_log is None:
            yield ErrorEvent(error_type="not_initialized", action="abort")
            return

        # Entry state validation
        if self._state.status == AgentState.RUNNING:
            yield ErrorEvent(error_type="already_running", action="abort")
            return

        # Ensure state is reset to IDLE if the generator is closed early
        # (e.g., consumer raised on an ErrorEvent). Otherwise consecutive
        # turns would be rejected with `already_running`.
        try:
            if continue_session and self._state.messages:
                # Continue the existing conversation.  The previous session
                # may have produced 20+ tool calls (especially when the LLM
                # was guessing model_name / cycling through wrong params).
                # Carrying them over verbatim would (a) blow up the LLM
                # context and (b) cause the LLM to re-try the same failed
                # calls.  Compress the history into a single summary
                # message: "已试过 X，结果 Y" — keeping only facts, not
                # verbose tool output.
                self._state.status = AgentState.RUNNING
                self._state.turn_number = 0
                self._state.consecutive_errors = 0
                self._state.has_attempted_reactive_compact = False
                self._state.session_started_at = time.monotonic()
                yield StateChangeEvent(old="idle", new="running")
                self._session_log.append(StateChangeEvent(old="idle", new="running"))

                # Record user message
                user_msg = UserMessageEvent(content=user_input)
                self._session_log.append(user_msg)
                yield user_msg

                # Compress previous history (returns a new list — state.messages
                # is pseudo-immutable, so we replace it).
                compressed = self._compress_session_history(
                    self._state.messages, user_input
                )
                self._state.messages = [
                    *compressed,
                    Message(role="user", content=user_input),
                ]
            else:
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
                # Phase 3 优化（skill 优先硬性指令）：当用户问题命中任何已注册
                # skill 的 trigger 关键词时，在 system prompt 顶部插入一段
                # 强约束——告诉 LLM：1) 必须优先调用 skill 工具；2) 严禁
                # 退化到 web_search。这是与 registry.py 中"工具列表硬剔除"
                # 配对的兜底（即使有模型硬要把 web_search 描述里的字面
                # "优先调用 skill" 忽略，这段 system 指令也会重复强化）。
                skill_priority_block = self._build_skill_priority_block(user_input)
                if skill_priority_block:
                    system_content = f"{skill_priority_block}\n\n{system_content}"
                if memory_context:
                    system_content += f"\n\n{memory_context}"

                self._state.messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content=user_input),
                ]

            # =================================================================
            # NEVER-EXITING MAIN LOOP
            # =================================================================
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
                    # Yield a FinalResponseEvent so the caller knows why the
                    # session ended instead of silently returning an empty
                    # string.
                    yield FinalResponseEvent(
                        text=f'会话已结束（{expire_reason}）。您可以输入"继续"让我接着完成。'
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
                    # Call LLM with streaming (DSPy or original path)
                    # ---------------------------------------------------------------
                    tools = self._tool_registry.get_for_llm(context_hint=user_input)
                    # Phase 3 优化：消费并 telemetry 记录本轮 web_search 抑制事件
                    # （仅在 skill trigger 命中时才有事件，正常 query 不产生日志）
                    self._log_skill_priority_event()

                    # Phase 2 优化（改 #6 + #10）：注入 failed_actions_block 到 system message
                    # 目的：让 LLM 看到自己之前的失败模式，避免重复同样错误
                    # 不硬编码——failed action 模式从 tool_result 历史动态检测
                    self._inject_failed_actions_block()

                    accumulated_text: list[str] = []
                    tool_calls_buffer: list[ToolCall] = []
                    last_finish_reason: str | None = None

                    if self._use_dspy and self._agent_reasoner:
                        # ========================================================================
                        # Phase 1 优化：DSPy 路径改为"决策可观测 hook"
                        # ========================================================================
                        # 改前：DSPy 走同步调用、强制 reasoning 字段、串行化 tool_call；失败时 fallback
                        #       到原始 LLM 路径（用户已等 60s+）。见 .trae/documents/dawu-agent单数据查询优化方案.md §2.1。
                        # 改后：DSPy 不再阻塞主流程——主流程始终走原始 chat_stream + tools= 协议；
                        #       DSPy 在后台异步 fire-and-forget 记录"如果用 DSPy 会选什么 action"，
                        #       仅用于后续 A/B 对比和离线优化训练数据生成。
                        #
                        # 默认关闭：观察 hook 每轮额外消耗 30-40s LLM 调用，且
                        # `chat_sync` 跨线程新建事件循环极易触发 "Event loop is closed"，
                        # 实际收益（离线 A/B / 训练数据）未在主流程验证。开启方式：
                        #   AGENT_DSPY_OBSERVE=true（环境变量）或 settings.agent.dspy_observe
                        # ========================================================================
                        if getattr(self.settings.agent, "dspy_observe", False):
                            try:
                                # 后台异步记录 DSPy 决策 trace，不 await——不阻塞主流程
                                # 注意：asyncio 已在文件顶部 import，无须重复 import
                                task = asyncio.create_task(
                                    self._dspy_observe_decision(user_input, tools)
                                )
                                # 持有强引用，否则 GC 会把未完成的 task 静默吞掉
                                self._dspy_pending_tasks.add(task)
                                task.add_done_callback(self._dspy_pending_tasks.discard)
                            except Exception as dspy_err:
                                # DSPy 观察失败时只记日志，绝不阻塞主流程
                                self.telemetry.logger.warning(
                                    "dspy.observe_skipped", error=str(dspy_err)
                                )

                        # 始终走原始 LLM 工具调用路径
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
                                last_finish_reason = chunk.finish_reason
                                ft = "".join(accumulated_text)
                                if ft:
                                    fe = AssistantTextEvent(content=ft, is_final=True)
                                    self._session_log.append(fe)
                    else:
                        # Original LLM call path
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
                                last_finish_reason = chunk.finish_reason
                                final_text = "".join(accumulated_text)
                                if final_text:
                                    final_event = AssistantTextEvent(content=final_text, is_final=True)
                                    self._session_log.append(final_event)

                    # ---------------------------------------------------------------
                    # Per-turn tool call cap. If the LLM emits too many tool calls
                    # in one assistant message, the session is effectively
                    # looping (or the LLM is hallucinating a multi-step plan). We
                    # truncate to the cap and inject a user nudge to force a
                    # synthesis on the data we *do* have.
                    # ---------------------------------------------------------------
                    max_tool_calls = getattr(
                        self.settings.agent, "max_tool_calls_per_turn", 8
                    )
                    if len(tool_calls_buffer) > max_tool_calls:
                        truncated = tool_calls_buffer[max_tool_calls:]
                        tool_calls_buffer = tool_calls_buffer[:max_tool_calls]
                        dropped_names = ",".join(
                            sorted({tc.name for tc in truncated})
                        ) or "unknown"
                        yield ErrorEvent(
                            error_type="tool_call_limit",
                            turn=self._state.turn_number,
                            action="truncate",
                            detail=(
                                f"单轮工具调用数 {len(tool_calls_buffer) + len(truncated)} "
                                f"超过上限 {max_tool_calls}，已丢弃 {len(truncated)} 个 "
                                f"({dropped_names})。请基于已有工具结果总结输出。"
                            ),
                        )
                        # Persist the dropped tool calls as text-only reasoning
                        # so the LLM can see what was skipped, then nudge it to
                        # synthesize before next turn.
                        self._state.messages = [
                            *self._state.messages,
                            Message(
                                role="assistant",
                                content=final_text or "",
                                tool_calls=list(tool_calls_buffer),
                            ),
                        ]
                        # Process the kept tool calls normally below; the nudge
                        # is appended after we collect their results.
                        dropped_nudge_pending = True
                    else:
                        dropped_nudge_pending = False

                    # ---------------------------------------------------------------
                    # CONTINUE SITE 8: Max Output Tokens (finish_reason="length")
                    # The LLM returned a truncated response without throwing an
                    # exception.  We must append the partial text and prompt the
                    # model to continue, rather than treating it as final.
                    # ---------------------------------------------------------------
                    if last_finish_reason == "length" and not tool_calls_buffer:
                        partial_text = "".join(accumulated_text)
                        if partial_text:
                            self._state.messages = [
                                *self._state.messages,
                                Message(role="assistant", content=partial_text),
                            ]
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

                    # ---------------------------------------------------------------
                    # CONTINUE SITE 9: Stream interrupted / empty response
                    # If the stream ended without a finish_reason (connection
                    # drop) or the LLM returned an empty response after tool
                    # results, we retry with a continuation prompt instead of
                    # silently ending the session.
                    # ---------------------------------------------------------------
                    final_text = "".join(accumulated_text)
                    has_tool_results = any(
                        m.role == "tool" for m in self._state.messages
                    )

                    if not tool_calls_buffer and (
                        last_finish_reason is None
                        or (not final_text and has_tool_results)
                    ):
                        if self._state.check_max_errors():
                            # Too many retries — give up gracefully
                            self._state.status = AgentState.ERROR
                            yield StateChangeEvent(
                                old="running", new="error",
                                reason="max_consecutive_errors: empty_response",
                            )
                            yield FinalResponseEvent(
                                text='模型多次返回空回复，会话已终止。您可以输入"继续"让我重试。'
                            )
                            break

                        reason = "stream_interrupted" if last_finish_reason is None else "empty_response"
                        self._state.messages = [
                            *self._state.messages,
                            Message(
                                role="user",
                                content="Please continue. Provide your analysis based on the tool results above.",
                            ),
                        ]
                        self._state.consecutive_errors += 1
                        yield ErrorEvent(
                            error_type=reason,
                            turn=self._state.turn_number,
                            action="continue_prompt",
                        )
                        continue

                    # ---------------------------------------------------------------
                    # Handle tool calls (CONTINUE SITE 7)
                    # ---------------------------------------------------------------
                    if tool_calls_buffer:
                        # The OpenAI protocol requires that every `tool`
                        # message is preceded by an `assistant` message
                        # whose `tool_calls` list contains the matching
                        # `tool_call.id`. Strict third-party gateways
                        # (e.g. autodl) reject the request with HTTP 400
                        # PARAM_ERROR when this is missing.
                        self._state.messages = [
                            *self._state.messages,
                            Message(
                                role="assistant",
                                content=final_text or "",
                                tool_calls=list(tool_calls_buffer),
                            ),
                        ]

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
                        # Recover to primary LLM client if we were on fallback
                        if self._llm_client is not self._primary_llm_client:
                            self._llm_client = self._primary_llm_client
                        self._emit_turn_end(turn_start)
                        # If we hit the tool-call cap earlier, force a synthesis
                        # turn by appending a user reminder. The cap is a
                        # deadlock-avoidance mechanism, not just a warning.
                        if dropped_nudge_pending:
                            self._state.messages = [
                                *self._state.messages,
                                Message(
                                    role="user",
                                    content=(
                                        "本轮已超过单次最大工具调用次数；"
                                        "请基于已收集到的工具结果直接给出总结回答，"
                                        "不要再调用任何工具。"
                                    ),
                                ),
                            ]
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
                    # Recover to primary LLM client if we were on fallback
                    if self._llm_client is not self._primary_llm_client:
                        self._llm_client = self._primary_llm_client
                    yield StateChangeEvent(old="running", new="idle", reason="complete")
                    yield FinalResponseEvent(text=final_text)
                    self._emit_turn_end(turn_start)
                    break

                # ================================================================
                # ERROR RECOVERY: 6 Continue Sites
                # ================================================================
                except Exception as e:
                    error_str = str(e).lower()

                    # CONTINUE SITE 1b: Non-retriable client error (4xx).
                    # Bad-request / param-error / auth-error mean the LLM gateway
                    # rejected our payload. Retrying with the same payload (or a
                    # micro-variant) burns tokens and stalls the session. The
                    # pragmatic fix is to terminate gracefully so the user can
                    # see the error and try a different approach.
                    if any(
                        marker in error_str
                        for marker in ("param_error", "invalid_request", "400 ", "401 ",
                                       "403 ", "404 ", "422 ", "400-")
                    ):
                        self._state.status = AgentState.ERROR
                        self._state.last_error_type = type(e).__name__
                        yield StateChangeEvent(
                            old="running",
                            new="error",
                            reason=f"client_error: {e}",
                        )
                        yield FinalResponseEvent(
                            text=f'LLM 网关拒绝请求（{type(e).__name__}: {str(e)[:240]}）。'
                                 f'会话已终止，请检查模型/参数配置或更换网关后输入"继续"重试。'
                        )
                        break

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
                        fallback_profile = self.settings.llm.get_fallback()
                        if fallback_profile:
                            # Use the dedicated fallback profile (correct API
                            # key + model name) instead of overriding the
                            # provider on the primary profile.
                            self._llm_client = LLMClientFactory.create(fallback_profile)
                            yield ErrorEvent(
                                error_type="model_unavailable",
                                turn=self._state.turn_number,
                                action="fallback",
                                detail=f"Switched to fallback model: {fallback_profile.name}",
                            )
                        else:
                            # No fallback configured — exponential backoff
                            await asyncio.sleep(min(2 ** self._state.consecutive_errors, 60))

                        self._state.consecutive_errors += 1
                        # Backoff even when fallback succeeded, to avoid
                        # hammering the fallback endpoint on tight loops.
                        await asyncio.sleep(min(2 ** (self._state.consecutive_errors - 1), 30))
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
                        yield FinalResponseEvent(
                            text=f'连续错误次数过多，会话已终止（{type(e).__name__}: {e}）。您可以输入"继续"让我重试。'
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
        finally:
            if self._state.status == AgentState.RUNNING:
                self._state.status = AgentState.IDLE

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

    def _compress_session_history(
        self, messages: list[Message], new_user_input: str
    ) -> list[Message]:
        """Compress the previous session's history when continuing.

        This addresses the DSPy regression where "继续"-style continuation
        re-feeds the entire 30+ turn history to the LLM, causing:
          - 50-100s reasoning time per turn
          - Repeated calls to tools that already failed
          - 1.5+ hour total runtimes

        Strategy (lossy but fact-preserving):
          1. Keep the FIRST system message intact (still needed for prompt)
          2. Keep the FIRST user message intact (the original task)
          3. Walk through all subsequent messages and build a summary
             with these facts:
             - For each (assistant tool_call, tool result) pair, extract
               the tool name + key args + outcome (success/failure/empty).
             - Format as a single "已试过 X，结果 Y" list.
             - Cap each entry at ~120 chars to keep summary small.
          4. Inject the summary as a user-role message (so the LLM treats
             it as context, not as a new instruction).
          5. Optionally preserve the LAST assistant text (if it had
             content) so the LLM knows where it left off.

        The returned list does NOT include the new user input — the caller
        appends that.
        """
        if not messages:
            return []

        # 1) Keep system prompt (first one)
        system_msgs = [m for m in messages if m.role == "system"]
        system_msg = system_msgs[0] if system_msgs else None

        # 2) Find first user message (the original task)
        first_user = next((m for m in messages if m.role == "user"), None)

        # 3) Walk the rest and build "已试过 X，结果 Y" entries.
        # Group: when we see assistant with tool_calls, remember them;
        # when we see tool result(s) following, pair them up.
        actions: list[str] = []  # formatted entries
        seen_action_keys: set[str] = set()  # dedup identical actions
        last_assistant_text: str = ""
        successful_data_samples: list[str] = []  # keep up to 3 short data hints

        pending_calls: dict[str, dict[str, Any]] = {}
        for msg in messages:
            if msg.role == "system" or msg is first_user:
                continue

            if msg.tool_calls:
                # Multiple tool calls can be issued in one assistant turn
                for tc in msg.tool_calls:
                    pending_calls[tc.id] = {
                        "name": tc.name,
                        "args": tc.arguments,
                        "args_str": _short_args(tc.arguments),
                    }

            elif msg.role == "tool" and msg.tool_call_id:
                pc = pending_calls.pop(msg.tool_call_id, None)
                if pc is None:
                    continue
                outcome = _classify_outcome(msg.content)
                key = f"{pc['name']}|{pc['args_str']}|{outcome['kind']}"
                if key in seen_action_keys:
                    continue
                seen_action_keys.add(key)
                entry = (
                    f"- 试过 {pc['name']}({pc['args_str']}) → {outcome['summary']}"
                )
                actions.append(entry[:240])  # cap per-entry size
                # If outcome includes a small data hint, capture it
                if outcome.get("hint") and len(successful_data_samples) < 3:
                    successful_data_samples.append(outcome["hint"][:160])

            elif msg.role == "assistant" and msg.content:
                last_assistant_text = msg.content

        # 4) Build summary
        summary_parts: list[str] = []
        summary_parts.append(
            "[上一轮压缩摘要 — 不要再重复这些调用]"
        )
        if actions:
            summary_parts.append("已试过的工具调用及结果：")
            summary_parts.extend(actions[-12:])  # cap to last 12 actions
        else:
            summary_parts.append("(无工具调用)")
        if successful_data_samples:
            summary_parts.append("已获取的关键数据：")
            summary_parts.extend(f"- {s}" for s in successful_data_samples)
        if last_assistant_text:
            tail = last_assistant_text[:200].replace("\n", " ")
            summary_parts.append(f"上一轮最后的助手输出: {tail}")

        summary_text = "\n".join(summary_parts)

        # 5) Assemble
        result: list[Message] = []
        if system_msg is not None:
            result.append(system_msg)
        if first_user is not None:
            result.append(first_user)
        result.append(Message(role="user", content=summary_text))
        return result

    async def _wait_for_resume(self) -> None:
        """Wait for external resume signal."""
        while self._state.pause_requested:
            await asyncio.sleep(0.1)

    @property
    def state(self) -> AgentState:
        return self._state.status

    # =================================================================
    # DSPy helper methods
    # =================================================================

    async def _dspy_decide_action(self, user_input: str, tools: list[dict]) -> Any:
        """Use DSPy module to decide next action.

        Improvement over the naive version (addresses DSPy regression where
        every-turn LLM cost is 50-100s and 5+ wrong tool calls happen):

        1. Truncate tools_desc to a concise per-tool summary (name + description
           truncated to 200 chars + a 1-line parameter hint). Full schemas are
           typically 3-5KB each — too much for an LLM to read in 60s.
        2. Prepend a "DECISION RULES" block that explicitly tells the LLM:
           - When to use_tool vs respond
           - That `query_school(keyword=...)` should be the FIRST call for any
             "查某校某业务" question (this is the SKILL.md rule inlined into
             the system prompt)
           - That "Action X does not accept parameter(s): [Y]" must NOT be
             re-tried with the same wrong params
        3. Use a stable sorted order for tools so caching is reliable.
        """
        import json

        history_text = self._format_messages_for_dspy()
        # Compress tool list: keep name + first 200 chars of desc + parameter names
        compact_tools: list[dict] = []
        for t in tools or []:
            fn = t.get("function", t)
            params = fn.get("parameters", {})
            prop_names = list(params.get("properties", {}).keys())
            compact_tools.append({
                "name": fn.get("name", ""),
                "description": (fn.get("description") or "")[:200],
                "parameters": prop_names,
            })
        # Stable sort by name for cache stability
        compact_tools.sort(key=lambda x: x.get("name", ""))
        tools_desc = json.dumps(compact_tools, ensure_ascii=False, indent=1)

        from datetime import datetime
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Asia/Shanghai")
        current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        memory_context = ""
        if self._memory_manager:
            memories = await self._memory_manager.get_relevant_memories(
                messages=self._state.messages, current_task=user_input,
            )
            memory_context = self._memory_manager.format_for_context(memories)

        # Inject decision rules as a system-style block at the top of user_query
        # (DSPy forwards user_query as the actual prompt user message).
        decision_rules = (
            "DECISION RULES (highest priority):\n"
            "1. If the user asks about a school's indicator/collection-point data,\n"
            "   the FIRST tool call MUST be query_school(school_name, model_name=None,\n"
            "   cycle=None, keyword=<extracted keyword>). Do NOT pre-explore with\n"
            "   list_models/search_tables/list_data_sets first — they waste 30s each.\n"
            "2. If a previous tool result said 'does not accept parameter(s): [X]',\n"
            "   do NOT retry with the same wrong parameter names. Use the 'Allowed'\n"
            "   parameter list from that error.\n"
            "3. If query_school returns _status != 'ok', it includes models_available\n"
            "   / candidates / hint fields — use those to recover, don't guess again.\n"
            "4. When the data is sufficient to answer the user, set action='respond'\n"
            "   and write a clear Chinese answer. Don't keep probing.\n"
        )
        augmented_query = f"{decision_rules}\n\nUSER QUERY: {user_input}"

        return self._agent_reasoner(
            conversation_history=history_text,
            user_query=augmented_query,
            available_tools=tools_desc,
            current_time=current_time,
            memory_context=memory_context,
        )

    async def _dspy_observe_decision(self, user_input: str, tools: list[dict]) -> None:
        """DSPy 决策可观测 hook（Phase 1 新增）。

        与 _dspy_decide_action 的区别：
        - _dspy_decide_action 是阻塞调用，结果会驱动 LLM 工具选择（已废弃）
        - _dspy_observe_decision 是 fire-and-forget：调用 DSPy 但不 await 结果，
          仅把"如果用 DSPy 会选什么 action"记到 session log
        - 用于离线 A/B 对比和训练数据生成

        任何异常都会被吞掉，绝不阻塞主流程。
        """
        import time
        start = time.time()
        try:
            decision = await self._dspy_decide_action(user_input, tools)
            duration_ms = int((time.time() - start) * 1000)
            action = (
                decision.action.lower().strip()
                if getattr(decision, "action", None)
                else "unknown"
            )
            tool_name = (
                decision.tool_name
                if getattr(decision, "tool_name", None)
                else None
            )
            self.telemetry.logger.info(
                "dspy.observed_decision",
                action=action,
                tool_name=tool_name,
                duration_ms=duration_ms,
                has_reasoning=bool(getattr(decision, "reasoning", None)),
            )
        except Exception as e:
            # 观察失败时只记日志，绝不向上抛
            self.telemetry.logger.warning(
                "dspy.observe_failed",
                error=str(e),
                duration_ms=int((time.time() - start) * 1000),
            )

    # =================================================================
    # Phase 2: failed_actions_block injection (decision reversibility)
    # =================================================================

    _FAILED_RESULT_INDICATORS = (
        "does not accept parameter",
        "wrong_param",
        "[learned]",
        "no_data_for_model",
        "no_match_for_keyword",
        "model_not_found",
        "session expired",
        "timeout",
        "error:",
    )

    def _detect_failed_actions(self) -> list[dict]:
        """扫描 messages 历史，识别 (tool_name, args) 失败模式。

        判定"失败"基于 tool_result 内容特征：
        - 含 [learned] tag（来自 skill_tool.py 软错误）
        - 含"does not accept parameter"
        - 含特定 _status 字段标记（model_not_found / no_data_for_model 等）
        - 含"timeout" / "error:"

        同一 (tool_name, args) 模式聚合去重，按出现次数排序。
        """
        messages = self._state.messages
        # tool_call_id -> (tool_name, args)
        tool_calls: dict[str, tuple[str, str]] = {}
        # tool_call_id -> result content
        tool_results: dict[str, str] = {}
        for msg in messages:
            if msg.role == "assistant":
                # Assistant message with tool_calls
                tcs = getattr(msg, "tool_calls", None) or []
                for tc in tcs:
                    tid = getattr(tc, "id", None)
                    if not tid:
                        continue
                    name = getattr(tc, "name", "") or (
                        getattr(tc, "function", {}) or {}
                    ).get("name", "")
                    args = getattr(tc, "arguments", None) or (
                        getattr(tc, "function", {}) or {}
                    ).get("arguments", "")
                    tool_calls[tid] = (name, args)
            elif msg.role == "tool":
                tid = getattr(msg, "tool_call_id", None)
                if tid:
                    content = msg.content or ""
                    tool_results[tid] = content

        failed: list[dict] = []
        for tid, (name, args) in tool_calls.items():
            result = tool_results.get(tid)
            if not result:
                continue
            result_lower = result.lower()
            if any(
                indicator in result_lower
                for indicator in self._FAILED_RESULT_INDICATORS
            ):
                # args 可能是 dict（LLM 直接给的对象）或 str（序列化 JSON）。
                # 统一转成字符串，避免 Counter key unhashable、以及后续
                # _build_failed_actions_block 里的 len()/切片操作炸掉。
                args_str = json.dumps(args, ensure_ascii=False, default=str) \
                    if isinstance(args, dict) else (str(args) if args else "")
                failed.append({"tool_name": name, "args": args_str, "result": result})

        # 聚合：相同 (tool_name, args_str) 模式统计次数
        from collections import Counter
        keys = [(f["tool_name"], f["args"]) for f in failed]
        counter = Counter(keys)
        aggregated: list[dict] = []
        for (name, args), count in counter.most_common():
            aggregated.append({
                "tool_name": name,
                "args": args,
                "count": count,
            })
        return aggregated

    def _build_failed_actions_block(self, failed: list[dict]) -> str:
        """把 failed actions 渲染成 system prompt 顶部 block。

        限制 block 大小为 500 字符；只列最近 3 个失败模式。
        不硬编码任何业务领域——只描述通用失败模式。
        """
        if not failed:
            return ""
        top = failed[:3]
        lines = ["[DO-NOT-RETRY] 之前轮次中已失败的工具调用："]
        for item in top:
            name = item["tool_name"]
            args = item["args"]
            count = item["count"]
            # 截短 args 避免 prompt 爆炸
            args_short = (args[:80] + "…") if len(args) > 80 else args
            lines.append(
                f"- {name} args={args_short} 已失败 {count} 次。"
                f"请改用其他 action 或参数，不要再用相同调用。"
            )
        block = "\n".join(lines)
        if len(block) > 500:
            block = block[:500] + "…"
        return block

    def _inject_failed_actions_block(self) -> None:
        """把 failed_actions_block 注入 system message（Phase 2 改 #10）。

        注入位置：system message 的最前面。
        原因：LLM 对 system 指令的服从度最高，注入到 user_query 或 tool description
        容易被截断（8K 上限）或被忽略。
        """
        failed = self._detect_failed_actions()
        block = self._build_failed_actions_block(failed)
        if not block:
            return
        messages = self._state.messages
        if not messages:
            return
        # 找到 system message 并前缀注入
        for i, msg in enumerate(messages):
            if msg.role == "system":
                content = msg.content or ""
                # 避免重复注入（如果 block 已在 content 头部）
                if content.startswith("[DO-NOT-RETRY]"):
                    return
                new_content = f"{block}\n\n{content}"
                # Message 可能是 pydantic / dataclass——用 replace 或重新构造
                try:
                    messages[i] = msg.model_copy(update={"content": new_content})
                except AttributeError:
                    try:
                        messages[i] = msg.copy(update={"content": new_content})
                    except AttributeError:
                        # 兜底：直接修改属性
                        msg.content = new_content
                return

    # =================================================================
    # Phase 3: SKILL_PRIORITY_BLOCK (skill 优先硬性指令)
    # =================================================================

    def _build_skill_priority_block(self, user_input: str) -> str:
        """根据用户 query 命中的 skill trigger，构造 system prompt 顶部的
        skill 优先硬性指令。

        调用时机：run_stream 启动时构造 system_content 之前。
        不硬编码业务领域——skill 名称、trigger 关键词完全从 _tool_registry
        动态派生。

        返回空字符串 = 没有 skill 命中 = 不注入（与 _inject_failed_actions_block
        行为一致）。

        配套的"工具列表硬剔除"逻辑在 ToolRegistry.get_for_llm 中——本 block
        只是兜底的 system 指令，LLM 即使看到 web_search 描述里有"优先调 skill"
        字样，本 block 也会再次强化。
        """
        try:
            matched = self._tool_registry.skill_triggers_match(user_input)
        except Exception:
            return ""

        if not matched:
            return ""

        # 取已注册 skill 工具的真实 description（保留 trigger 段），让 LLM
        # 看到具体可用 skill 的触发关键词和函数签名。
        skill_lines: list[str] = []
        for tool_name in sorted(matched):
            tool = self._tool_registry.get(tool_name)
            if tool is None:
                continue
            desc = tool.description or ""
            # description 本身已经含 trigger 关键词，截前 400 字避免 prompt 爆炸
            skill_lines.append(f"- 工具 `{tool_name}`: {desc[:400]}")

        if not skill_lines:
            return ""

        block = (
            "[SKILL-PRIORITY 硬性指令 — 系统级约束，优先级高于工具描述]\n"
            "用户问题命中了本系统已注册 skill 的 trigger 关键词。**必须遵守**：\n"
            "1. **第一步必须调用下述 skill 工具**——不要先调用任何 web_search "
            "（bing_search / baidu_search / tavily_search）。\n"
            "2. skill 内置了准确的本地数据源（如数据库查询），其结果比公网搜索"
            "更可靠、更新更及时。\n"
            "3. **即使 skill 返回空数据或结果不理想，也绝对不要退化到 web_search**——"
            "如果 skill 没数据，就直接告诉用户「该业务暂无数据」，不要拿公网结果"
            "拼凑。\n"
            "4. 同一轮内允许在调 skill 之后再调用其他工具（file_read / "
            "data_query 等），但严禁调用任何 web_search 工具。\n\n"
            "本轮已为你激活的 skill 工具：\n"
            + "\n".join(skill_lines)
        )
        return block

    def _log_skill_priority_event(self) -> None:
        """从 ToolRegistry 消费并 telemetry 记录最近的 web_search 抑制事件。

        配套 ToolRegistry.get_for_llm 写的 _last_web_search_suppression。
        每次 get_for_llm 后调用一次（consume 模式，避免重复日志）。
        """
        try:
            ev = self._tool_registry.consume_last_web_search_suppression()
        except Exception:
            return
        if not ev:
            return
        try:
            self.telemetry.logger.info(
                "tools.skill_priority_suppressed_web_search",
                matched_skills=ev.get("matched_skills", []),
                suppressed_tools=ev.get("suppressed_tools", []),
            )
        except Exception:
            # telemetry 失败绝不能阻塞主流程
            pass

    async def _dspy_respond(self, user_input: str) -> Any:
        """Use DSPy module to generate direct response."""
        from dawu_agent.dspy_integration.modules import AgentReasoner

        responder = AgentReasoner().generate_response
        history_text = self._format_messages_for_dspy()
        from datetime import datetime
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Asia/Shanghai")
        current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        return responder(
            conversation_history=history_text,
            user_query=user_input,
            current_time=current_time,
            memory_context="",
        )

    def _format_messages_for_dspy(self) -> str:
        """Format message history as text for DSPy input.

        Phase 2 优化（改 #6）：
        - 移除 8K 硬限制（由 LLM client 的 max_tokens 决定）
        - 新增 candidates 注入：当工具结果含 _status ∈ {model_not_found,
          no_data_for_model, no_match_for_keyword} 时，把相关候选项
          （models_available / candidates / suggestion）额外写入 summary
        - 保留 per-role tier cap（system 2000 / user 1500 / assistant 800 /
          tool 400）以避免单条 message 占用过多 token
        - 保留去重 + learned summary

        注：本函数目前仅被 DSPy 观察 hook 调用（_dspy_decide_action /
        _dspy_respond），不影响主路径（chat_stream 直接用 messages）。
        """
        TIER_CAPS = {
            "system": 2000,
            "user": 1500,
            "assistant": 800,
            "tool": 400,
        }
        parts: list[str] = []
        seen_tool_results: dict[str, int] = {}
        failures: list[str] = []
        candidate_hints: list[str] = []  # Phase 2 新增：候选可用 model/table

        for msg in self._state.messages:
            role = msg.role
            cap = TIER_CAPS.get(role, 500)
            content = msg.content[:cap]

            if msg.tool_calls:
                tc_str = ", ".join(
                    f"{tc.name}({tc.arguments})" for tc in msg.tool_calls
                )
                parts.append(f"[{role}] {content} [Tool Calls: {tc_str}]")
            elif msg.tool_call_id:
                # Deduplicate identical tool results
                key = content[:120]
                seen_tool_results[key] = seen_tool_results.get(key, 0) + 1
                if seen_tool_results[key] > 1:
                    if seen_tool_results[key] == 2:
                        parts.append(f"[tool result] (与上条相同，已省略) {content}")
                else:
                    parts.append(f"[tool result] {content}")
                # Detect "Error" markers → add to failure list
                low = content.lower()
                if "error" in low or "未知" in content or "失败" in content:
                    failures.append(content[:200])

                # Phase 2 优化：candidates 注入
                # 当工具返回 _status ∈ {model_not_found, no_data_for_model,
                # no_match_for_keyword} 时，把候选项（models_available /
                # candidates / suggestion）提取出来供 LLM 下次使用
                if '"_status": "model_not_found"' in content:
                    candidate_hints.append(
                        self._extract_candidate_hint(content, "model_not_found")
                    )
                elif '"_status": "no_data_for_model"' in content:
                    candidate_hints.append(
                        self._extract_candidate_hint(content, "no_data_for_model")
                    )
                elif '"_status": "no_match_for_keyword"' in content:
                    candidate_hints.append(
                        self._extract_candidate_hint(content, "no_match_for_keyword")
                    )
            else:
                parts.append(f"[{role}] {content}")

        # 拼接 learned + candidates 总结
        summary_block: list[str] = []
        if seen_tool_results:
            dup_count = sum(1 for c in seen_tool_results.values() if c > 1)
            if dup_count:
                summary_block.append(
                    f"[learned] 本会话内有 {dup_count} 个工具结果出现过 ≥2 次（已去重）"
                )
        if failures:
            uniq = list(dict.fromkeys(failures))[-5:]
            summary_block.append(
                "[learned] 已观察到以下失败 (不要再重复相同的参数):\n - "
                + "\n - ".join(uniq)
            )
        # Phase 2 新增：candidates 注入
        uniq_hints = [h for h in dict.fromkeys(candidate_hints) if h]
        if uniq_hints:
            summary_block.append(
                "[candidates] 以下 tool 返回了可用的候选项，下一次调用请优先使用：\n - "
                + "\n - ".join(uniq_hints)
            )
        if summary_block:
            parts.append("\n" + "\n".join(summary_block))

        return "\n".join(parts)

    @staticmethod
    def _extract_candidate_hint(content: str, status: str) -> str:
        """从 tool_result content 中提取可用的候选项提示。

        不硬编码业务字段名——通过正则从 JSON 文本中提取
        models_available / candidates / suggestion 字段。
        """
        import re

        # 查找 "models_available": [...] / "candidates": [...] / "suggestion": "..."
        patterns = {
            "models_available": r'"models_available"\s*:\s*\[([^\]]+)\]',
            "candidates": r'"candidates"\s*:\s*\[([^\]]+)\]',
            "suggestion": r'"suggestion"\s*:\s*"([^"]+)"',
        }
        fragments = []
        for field, pat in patterns.items():
            m = re.search(pat, content)
            if m:
                # 截短避免 prompt 爆炸
                val = m.group(1)[:200]
                fragments.append(f"{field}={val}")
        if not fragments:
            return f"{status}: (no candidates found in result)"
        return f"{status}: " + ", ".join(fragments)

    def _parse_dspy_tool_call(self, decision) -> list[ToolCall]:
        """Parse DSPy decision output into ToolCall objects."""
        import json
        import uuid

        tool_calls: list[ToolCall] = []
        if decision.tool_name:
            try:
                args = json.loads(decision.tool_arguments) if decision.tool_arguments else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=decision.tool_name,
                arguments=args,
            ))
        return tool_calls
