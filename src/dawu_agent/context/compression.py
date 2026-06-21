"""Four-level compression pipeline for context management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from dawu_agent.llm.base import ILLMClient, Message


@dataclass
class CompressionRequest:
    """Request to compress message history."""

    messages: list[Message]
    token_budget: int
    current_tokens: int
    reason: str = ""


@dataclass
class CompressionResult:
    """Result of compression operation."""

    messages: list[Message]
    tokens_freed: int
    level_used: int
    recovery_hints: list[str] = field(default_factory=list)


class CompressionStrategy:
    """Base class for compression strategies."""

    def compress(self, request: CompressionRequest) -> CompressionResult | None:
        """Attempt to compress. Return None if cannot handle."""
        raise NotImplementedError


class SnipStrategy(CompressionStrategy):
    """Level 1: Remove oldest messages with summary injection."""

    def __init__(self, token_counter: Callable[[list[Message]], int]) -> None:
        self.token_counter = token_counter

    def compress(self, request: CompressionRequest) -> CompressionResult | None:
        messages = request.messages
        target_free = request.current_tokens - request.token_budget

        if len(messages) < 10:
            return None

        # Accumulate from oldest to find what to snip
        accumulated = 0
        snip_count = 0
        for msg in messages:
            accumulated += self.token_counter([msg])
            snip_count += 1
            if accumulated >= target_free:
                break

        if snip_count < 5:
            return None

        oldest_block = messages[:snip_count]
        remainder = messages[snip_count:]

        # Generate summary of snipped content
        summary_parts = []
        for msg in oldest_block:
            preview = msg.content[:80].replace("\n", " ")
            summary_parts.append(f"[{msg.role}] {preview}...")

        injection = Message(
            role="system",
            content=(
                "── 上下文已剪断（Snip）──\n"
                f"已跳过 {snip_count} 条历史消息，摘要如下：\n"
                + "\n".join(summary_parts)
            ),
        )

        result_messages = [injection, *remainder]
        tokens_freed = accumulated

        return CompressionResult(
            messages=result_messages,
            tokens_freed=tokens_freed,
            level_used=1,
            recovery_hints=["recent_files", "active_plan"],
        )


class MicrocompactStrategy(CompressionStrategy):
    """Level 2: Truncate long tool results (head + tail)."""

    def __init__(self, token_counter: Callable[[list[Message]], int]) -> None:
        self.token_counter = token_counter
        self.threshold = 4000  # tokens
        self.head_tokens = 1500
        self.tail_tokens = 500

    def compress(self, request: CompressionRequest) -> CompressionResult:
        messages = list(request.messages)  # Shallow copy
        freed = 0

        for i, msg in enumerate(messages):
            if msg.role != "tool":
                continue

            msg_tokens = self.token_counter([msg])
            if msg_tokens <= self.threshold:
                continue

            # Head + tail truncation
            content = msg.content
            head_len = int(len(content) * (self.head_tokens / max(msg_tokens, 1)))
            tail_len = int(len(content) * (self.tail_tokens / max(msg_tokens, 1)))

            head = content[:head_len]
            tail = content[-tail_len:]
            omitted = len(content) - head_len - tail_len

            separator = f"\n... [中间 {omitted} 字符已截断] ...\n"
            new_content = head + separator + tail

            messages[i] = Message(
                role=msg.role,
                content=new_content,
                name=msg.name,
                tool_calls=msg.tool_calls,
                tool_call_id=msg.tool_call_id,
            )
            freed += omitted // 4  # Rough token estimate

        if freed == 0:
            return None

        return CompressionResult(
            messages=messages,
            tokens_freed=freed,
            level_used=2,
            recovery_hints=["tool_outputs_truncated"],
        )


class CollapseStrategy(CompressionStrategy):
    """Level 3: Fold consecutive non-critical messages into summary."""

    def __init__(self, token_counter: Callable[[list[Message]], int]) -> None:
        self.token_counter = token_counter
        self.collapse_window = 20

    def compress(self, request: CompressionRequest) -> CompressionResult | None:
        messages = request.messages

        if len(messages) < self.collapse_window:
            return None

        # Find collapsible span (non-critical messages)
        # Critical: tool_calls, system messages
        # Non-critical: plain assistant/user text
        collapsible_start = None
        collapsible_end = None

        for i, msg in enumerate(messages):
            is_critical = (
                msg.role == "system"
                or (msg.role == "assistant" and msg.tool_calls)
                or msg.role == "tool"
            )
            if not is_critical:
                if collapsible_start is None:
                    collapsible_start = i
                collapsible_end = i
            else:
                if collapsible_start is not None and (collapsible_end - collapsible_start) >= 10:
                    break
                collapsible_start = None

        if collapsible_start is None or collapsible_end is None:
            return None

        span = messages[collapsible_start : collapsible_end + 1]
        if len(span) < 10:
            return None

        # Generate template summary (no LLM call)
        summary_parts = []
        total_tokens = 0
        for msg in span:
            tokens = self.token_counter([msg])
            total_tokens += tokens
            preview = msg.content[:200].replace("\n", " ")
            summary_parts.append(f"[{msg.role}] {preview}... ({tokens} tokens)")

        summary = "\n".join(summary_parts)

        collapsed_msg = Message(
            role="system",
            content=(
                f"── 折叠上下文（Collapse）──\n"
                f"已折叠 {len(span)} 条消息，原共 {total_tokens} tokens：\n"
                f"{summary}"
            ),
        )

        result_messages = (
            messages[:collapsible_start]
            + [collapsed_msg]
            + messages[collapsible_end + 1 :]
        )

        freed = total_tokens - self.token_counter([collapsed_msg])

        return CompressionResult(
            messages=result_messages,
            tokens_freed=max(0, freed),
            level_used=3,
            recovery_hints=["collapsed_conversation_history"],
        )


class AutocompactStrategy(CompressionStrategy):
    """Level 4: LLM-based semantic summarization (DSPy-enhanced)."""

    def __init__(
        self,
        llm_client: ILLMClient,
        token_counter: Callable[[list[Message]], int],
    ) -> None:
        self.llm_client = llm_client
        self.token_counter = token_counter
        self._compression_module: Any = None
        # Try to load DSPy CompressionModule
        try:
            from dawu_agent.dspy_integration.modules import CompressionModule
            self._compression_module = CompressionModule()
        except Exception:
            self._compression_module = None

    def compress(self, request: CompressionRequest) -> CompressionResult | None:
        messages = request.messages
        target_free = request.current_tokens - request.token_budget

        # Find compression range from tail to head
        accumulated = 0
        compression_start = 0
        for i in range(len(messages) - 1, -1, -1):
            accumulated += self.token_counter([messages[i]])
            if accumulated >= target_free:
                compression_start = i
                break

        # Keep last 10 messages uncompressed
        if len(messages) - compression_start < 10:
            compression_start = max(0, len(messages) - 10)

        if compression_start == 0:
            return None

        historical = messages[:compression_start]
        recent = messages[compression_start:]

        # Build compact prompt
        compact_prompt = """请对以下对话历史进行结构化摘要，保留关键信息：
1. 用户的核心目标（当前任务是什么）
2. 已完成的步骤（按顺序列出）
3. 关键决策点（触发了哪些工具，为什么要触发）
4. 未解决的问题/待处理的任务
5. 当前工作的文件清单
6. 正在使用的工具上下文

格式：简洁，使用要点列表。不要包含无关的闲聊内容。

历史消息：
"""

        history_text = "\n".join(
            f"[{m.role}] {m.content[:500]}" for m in historical
        )

        summary = ""
        # Try DSPy CompressionModule first
        if self._compression_module is not None:
            try:
                result = self._compression_module(
                    conversation_history=history_text,
                    current_task=request.reason or "ongoing analysis",
                )
                summary = result.summary
            except Exception:
                summary = ""

        # Fallback to raw LLM call if DSPy failed or not available
        if not summary:
            try:
                summary_response = self.llm_client.chat_sync(
                    messages=[
                        Message(role="user", content=compact_prompt + history_text),
                    ]
                )
                summary = summary_response.content
            except Exception:
                return None

        recovery_message = Message(
            role="system",
            content=(
                "── 上下文压缩点（Autocompact）──\n"
                "以下历史已被压缩为摘要，但关键状态已保留：\n\n"
                f"{summary}\n\n"
                "── 恢复指令 ──\n"
                "1. 继续执行未完成的任务\n"
                "2. 必要时重新读取当前工作文件\n"
                "3. 正在使用的工具上下文已在前置消息中恢复"
            ),
        )

        result_messages = [recovery_message, *recent]
        historical_tokens = self.token_counter(historical)
        summary_tokens = self.token_counter([recovery_message])
        freed = historical_tokens - summary_tokens

        return CompressionResult(
            messages=result_messages,
            tokens_freed=max(0, freed),
            level_used=4,
            recovery_hints=[
                "recent_files",
                "active_plan",
                "active_skill",
                "pending_tasks",
            ],
        )


class CompressionPipeline:
    """Orchestrates four-level compression: Snip → Microcompact → Collapse → Autocompact."""

    def __init__(
        self,
        llm_client: ILLMClient,
        token_counter: Callable[[list[Message]], int],
        budget_ratio: float = 0.85,
    ) -> None:
        self.token_counter = token_counter
        self.budget_ratio = budget_ratio
        self.strategies = [
            SnipStrategy(token_counter),
            MicrocompactStrategy(token_counter),
            CollapseStrategy(token_counter),
            AutocompactStrategy(llm_client, token_counter),
        ]

    def should_compress(self, messages: list[Message], max_tokens: int) -> bool:
        """Check if compression is needed."""
        current = self.token_counter(messages)
        return current > max_tokens * self.budget_ratio

    def compact(
        self,
        messages: list[Message],
        max_tokens: int,
        reason: str = "",
    ) -> CompressionResult:
        """Run compression pipeline until budget is met or all strategies exhausted."""
        current_tokens = self.token_counter(messages)
        budget = int(max_tokens * self.budget_ratio)

        if current_tokens <= budget:
            return CompressionResult(
                messages=list(messages),
                tokens_freed=0,
                level_used=0,
            )

        request = CompressionRequest(
            messages=list(messages),
            token_budget=budget,
            current_tokens=current_tokens,
            reason=reason,
        )

        # Try each strategy in order
        for strategy in self.strategies:
            result = strategy.compress(request)
            if result is not None:
                # Check if budget is now met
                new_tokens = self.token_counter(result.messages)
                if new_tokens <= budget:
                    return result
                # Otherwise continue with next strategy
                request = CompressionRequest(
                    messages=result.messages,
                    token_budget=budget,
                    current_tokens=new_tokens,
                    reason=reason,
                )

        # All strategies exhausted - emergency snip
        return self._emergency_snip(request)

    def _emergency_snip(self, request: CompressionRequest) -> CompressionResult:
        """Emergency: keep only system messages + last 4 messages."""
        messages = request.messages
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]
        kept = system_msgs + other_msgs[-4:]

        freed = request.current_tokens - self.token_counter(kept)

        return CompressionResult(
            messages=kept,
            tokens_freed=max(0, freed),
            level_used=4,
            recovery_hints=["emergency_truncation"],
        )
