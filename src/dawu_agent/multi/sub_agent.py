"""Sub-agent with isolated context for multi-agent coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dawu_agent.llm.base import ILLMClient, Message, ToolCall
from dawu_agent.llm.factory import LLMClientFactory
from dawu_agent.multi.coordinator import TaskResult
from dawu_agent.tools.builtin.data_tools import (
    DataQueryTool,
    DataVisualizeTool,
    ReportGenerateTool,
)
from dawu_agent.tools.builtin.file_tools import FileReadTool, FileWriteTool
from dawu_agent.tools.registry import ToolRegistry


@dataclass
class SubAgentConfig:
    """Configuration for a sub-agent."""

    role: str = "analyst"
    system_prompt: str = ""
    max_turns: int = 10
    max_tokens: int = 8000
    model: str = "claude-3-5-sonnet-20241022"
    provider: str = "anthropic"


class SubAgent:
    """Isolated sub-agent with blank context.

    Key design: Sub-agent starts with EMPTY message list, not parent's history.
    Only receives the subtask description. Returns structured summary, not raw output.
    """

    ROLE_PROMPTS = {
        "data_engineer": """你是一个数据工程师子代理。你的职责是数据提取、转换和加载（ETL）。

任务完成后，请按以下结构化格式返回结果：
{{
  "summary": "任务执行摘要（200字以内）",
  "findings": ["发现1", "发现2"],
  "files_modified": ["修改的文件路径"],
  "errors": [],
  "suggestions": ["建议1"]
}}

规则：
- 始终检查数据模式后再转换
- 记录每个转换步骤
- 验证输出数据质量
- 使用高效操作
- 绝不修改原始源文件""",
        "analyst": """你是一个统计分析子代理。你的职责是从清洁数据中提取洞察。

任务完成后，请按以下结构化格式返回结果：
{{
  "summary": "分析摘要（200字以内）",
  "findings": ["发现1", "发现2"],
  "files_modified": [],
  "errors": [],
  "suggestions": ["建议1"]
}}

规则：
- 始终声明统计测试的假设
- 报告置信区间和p值
- 避免将噪声过度解读为信号
- 使用适当的可视化类型
- 用业务友好的语言总结发现""",
        "report_writer": """你是一个报告撰写子代理。你的职责是将分析结果合成为专业交付物。

任务完成后，请按以下结构化格式返回结果：
{{
  "summary": "报告摘要（200字以内）",
  "findings": ["关键发现1", "关键发现2"],
  "files_modified": ["生成的报告路径"],
  "errors": [],
  "suggestions": []
}}

规则：
- 最重要的发现放在前面（倒金字塔）
- 包含方法论部分以确保可复现性
- 引用数据来源和分析时间戳
- 使用清晰的标题、要点和表格
- 确保所有可视化都有标题、标签和图例""",
    }

    def __init__(self, config: SubAgentConfig) -> None:
        self.config = config
        self._llm_client: ILLMClient | None = None
        self._tool_registry = ToolRegistry()
        self._register_tools()

    def _register_tools(self) -> None:
        """Register tools based on role."""
        tools_by_role = {
            "data_engineer": [FileReadTool(), FileWriteTool(), DataQueryTool()],
            "analyst": [DataQueryTool(), DataVisualizeTool()],
            "report_writer": [ReportGenerateTool(), FileWriteTool(), DataVisualizeTool()],
        }

        for tool in tools_by_role.get(self.config.role, []):
            self._tool_registry.register(tool)

    async def _init_llm(self) -> None:
        """Lazy-init LLM client."""
        if self._llm_client is None:
            from dawu_agent.config.loader import LLMConfig
            config = LLMConfig(
                provider=self.config.provider,
                model=self.config.model,
            )
            self._llm_client = LLMClientFactory.create(config)

    async def execute(self, subtask_description: str) -> TaskResult:
        """Execute subtask with isolated context.

        Starts with blank message list, only receives subtask description.
        Returns structured summary, not raw conversation.
        """
        await self._init_llm()
        if self._llm_client is None:
            return TaskResult(
                subtask_id="unknown",
                success=False,
                errors=["LLM client initialization failed"],
            )

        # Build isolated message list (blank start + system prompt + subtask)
        system_prompt = self.config.system_prompt or self.ROLE_PROMPTS.get(
            self.config.role, "你是一个数据分析助手。"
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=subtask_description),
        ]

        turn_count = 0
        accumulated_response = []

        while turn_count < self.config.max_turns:
            turn_count += 1

            try:
                tools = self._tool_registry.get_for_llm()
                response = await self._llm_client.chat(
                    messages=messages,
                    tools=[t.to_llm_definition() for t in self._tool_registry.list_all()] if tools else None,
                )

                accumulated_response.append(response.content)

                # Handle tool calls
                if response.tool_calls:
                    messages.append(Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=[
                            ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                            for tc in response.tool_calls
                        ],
                    ))

                    for tc in response.tool_calls:
                        result = await self._tool_registry.execute(tc.name, tc.arguments)
                        messages.append(Message(
                            role="tool",
                            content=str(result.data) if result.data else result.error or "",
                            tool_call_id=tc.id,
                        ))
                    continue

                # No tool calls - parse final result
                final_text = "\n".join(accumulated_response)
                return self._parse_result(final_text)

            except Exception as e:
                return TaskResult(
                    subtask_id="unknown",
                    success=False,
                    errors=[f"Execution error: {e}"],
                    raw_output="\n".join(accumulated_response),
                )

        # Max turns reached
        return TaskResult(
            subtask_id="unknown",
            success=False,
            errors=["Max turns reached"],
            raw_output="\n".join(accumulated_response),
        )

    def _parse_result(self, text: str) -> TaskResult:
        """Parse structured result from agent output."""
        import json
        import re

        # Try to find JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return TaskResult(
                    subtask_id="unknown",
                    success=True,
                    summary=data.get("summary", text[:200]),
                    findings=data.get("findings", []),
                    files_modified=data.get("files_modified", []),
                    errors=data.get("errors", []),
                    suggestions=data.get("suggestions", []),
                    raw_output=text,
                )
            except json.JSONDecodeError:
                pass

        # Fallback: treat entire text as summary
        return TaskResult(
            subtask_id="unknown",
            success=True,
            summary=text[:500],
            raw_output=text,
        )
