"""Coordinator for multi-agent task decomposition and synthesis."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from dawu_agent.llm.base import ILLMClient, Message


@dataclass
class SubTask:
    """A decomposed sub-task for a sub-agent."""

    id: str
    description: str
    role: str  # data_engineer | analyst | report_writer
    dependencies: list[str] = field(default_factory=list)
    expected_output: str = ""
    max_tokens: int = 8000


@dataclass
class TaskResult:
    """Result from a sub-agent execution."""

    subtask_id: str
    success: bool
    summary: str = ""
    findings: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    raw_output: str = ""


class Coordinator:
    """Central coordinator for multi-agent task decomposition.

    Responsibilities:
    1. Decompose complex tasks into independent sub-tasks
    2. Allocate sub-tasks to specialized sub-agents
    3. Collect and synthesize results
    """

    def __init__(
        self,
        llm_client: ILLMClient,
        max_sub_agents: int = 5,
        task_timeout: int = 300,
    ) -> None:
        self.llm_client = llm_client
        self.max_sub_agents = max_sub_agents
        self.task_timeout = task_timeout
        # Try to load DSPy MultiAgentModule
        self._multi_agent_module: Any = None
        try:
            from dawu_agent.dspy_integration.modules import MultiAgentModule
            self._multi_agent_module = MultiAgentModule()
        except Exception:
            self._multi_agent_module = None

    async def decompose(self, task: str, context: str = "") -> list[SubTask]:
        """Decompose a complex task into sub-tasks using LLM (DSPy-enhanced)."""
        # Try DSPy MultiAgentModule first
        if self._multi_agent_module is not None:
            try:
                import json
                result = self._multi_agent_module(
                    task=f"{task}\n上下文：{context}" if context else task,
                    available_roles="data_engineer, analyst, report_writer",
                )
                data = json.loads(result.subtasks)
                subtasks = []
                for item in data.get("subtasks", []):
                    subtasks.append(SubTask(
                        id=item["id"],
                        description=item["description"],
                        role=item["role"],
                        dependencies=item.get("dependencies", []),
                        expected_output=item.get("expected_output", ""),
                    ))
                return subtasks[:self.max_sub_agents]
            except Exception:
                pass  # Fall through to original LLM path

        # Original LLM-based decomposition
        prompt = f"""请将以下复杂数据分析任务分解为独立的子任务。

任务：{task}
上下文：{context}

要求：
1. 每个子任务应该是独立的，可以并行执行
2. 子任务之间如果有依赖关系，请明确标注
3. 为每个子任务指定最合适的角色（data_engineer/analyst/report_writer）
4. 每个子任务应该有明确的输出预期

请按以下JSON格式输出：
{{
  "subtasks": [
    {{
      "id": "task-1",
      "description": "子任务描述",
      "role": "data_engineer",
      "dependencies": [],
      "expected_output": "预期输出"
    }}
  ]
}}

只输出JSON，不要其他解释。"""

        try:
            response = self.llm_client.chat_sync(
                messages=[Message(role="user", content=prompt)]
            )
            import json
            data = json.loads(response.content)
            subtasks = []
            for item in data.get("subtasks", []):
                subtasks.append(SubTask(
                    id=item["id"],
                    description=item["description"],
                    role=item["role"],
                    dependencies=item.get("dependencies", []),
                    expected_output=item.get("expected_output", ""),
                ))
            return subtasks[:self.max_sub_agents]
        except Exception:
            # Fallback: single task
            return [SubTask(
                id="task-1",
                description=task,
                role="analyst",
                dependencies=[],
            )]

    async def execute_parallel(
        self,
        subtasks: list[SubTask],
        sub_agent_factory,
    ) -> list[TaskResult]:
        """Execute sub-tasks in parallel where possible.

        Respects dependency ordering: tasks with dependencies wait
        for their dependencies to complete first.
        """
        completed: dict[str, TaskResult] = {}
        pending = list(subtasks)

        while pending:
            # Find tasks with no unmet dependencies
            ready = [
                t for t in pending
                if all(dep in completed for dep in t.dependencies)
            ]

            if not ready:
                # Circular dependency or missing dependency
                break

            # Execute ready tasks in parallel
            tasks = [
                self._execute_single(t, sub_agent_factory, completed)
                for t in ready
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for subtask, result in zip(ready, results):
                pending.remove(subtask)
                if isinstance(result, Exception):
                    completed[subtask.id] = TaskResult(
                        subtask_id=subtask.id,
                        success=False,
                        errors=[str(result)],
                    )
                else:
                    completed[subtask.id] = result

        return list(completed.values())

    async def _execute_single(
        self,
        subtask: SubTask,
        sub_agent_factory,
        completed_results: dict[str, TaskResult],
    ) -> TaskResult:
        """Execute a single sub-task."""
        # Build dependency context
        dep_context = ""
        for dep_id in subtask.dependencies:
            if dep_id in completed_results:
                dep_result = completed_results[dep_id]
                dep_context += f"\n依赖任务 {dep_id} 的结果：\n{dep_result.summary}\n"

        full_description = subtask.description + dep_context

        # Create and run sub-agent
        agent = sub_agent_factory(role=subtask.role)
        try:
            result = await asyncio.wait_for(
                agent.execute(full_description),
                timeout=self.task_timeout,
            )
            return result
        except asyncio.TimeoutError:
            return TaskResult(
                subtask_id=subtask.id,
                success=False,
                errors=["Task timeout"],
            )

    async def synthesize(self, results: list[TaskResult], original_task: str) -> str:
        """Synthesize sub-agent results into final output (DSPy-enhanced)."""
        # Build summaries for both paths
        summaries = []
        for r in results:
            status = "成功" if r.success else "失败"
            summaries.append(
                f"【子任务 {r.subtask_id} - {status}】\n"
                f"摘要：{r.summary}\n"
                f"发现：{'; '.join(r.findings)}\n"
                f"错误：{'; '.join(r.errors)}\n"
            )
        summaries_text = "\n".join(summaries)

        # Try DSPy MultiAgentModule.synthesize first
        if self._multi_agent_module is not None:
            try:
                result = self._multi_agent_module.synthesize(
                    original_task=original_task,
                    subtask_results=summaries_text,
                )
                return result.final_report
            except Exception:
                pass  # Fall through to original LLM path

        # Original LLM-based synthesis
        prompt = f"""请综合以下子任务的结果，生成最终的分析报告。

原始任务：{original_task}

子任务结果：
{summaries_text}

请生成一份结构化的综合报告，包含：
1. 执行摘要
2. 各子任务的关键发现
3. 整体结论和建议
4. 存在的问题（如有）
"""

        try:
            response = self.llm_client.chat_sync(
                messages=[Message(role="user", content=prompt)]
            )
            return response.content
        except Exception as e:
            return f"综合报告生成失败：{e}\n\n原始结果：\n{summaries_text}"
