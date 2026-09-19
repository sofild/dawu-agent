"""Evaluation scenarios for Phase 8 Agent evaluation suite.

Each scenario is a dataclass describing a test case: the user input, which
tools should / should not be called, and what keywords the final response
must contain.  Scenarios are grouped by *category* so the grader can produce
per-category pass rates.

Usage::

    from evals.scenarios import ALL_SCENARIOS
    for scenario in ALL_SCENARIOS:
        events = run_agent(scenario.user_input)
        result = grade_all(scenario, events)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalScenario:
    """A single evaluation scenario."""

    id: str
    name: str
    user_input: str
    expected_tools: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    max_turns: int = 10
    category: str = "general"  # general | tool_selection | error_recovery | context_management


# ──────────────────────────────────────────────────────────────
# Tool selection scenarios
# ──────────────────────────────────────────────────────────────

TS_FILE_READ = EvalScenario(
    id="ts-001",
    name="read_local_file",
    user_input="请读取 README.md 文件的内容",
    expected_tools=["file_read"],
    expected_keywords=["readme"],
    forbidden_tools=["bing_search", "baidu_search", "tavily_search"],
    max_turns=5,
    category="tool_selection",
)

TS_DATA_QUERY = EvalScenario(
    id="ts-002",
    name="query_dataset",
    user_input="用 pandas 读取 data.csv 并统计各列的缺失值数量",
    expected_tools=["data_query"],
    expected_keywords=["缺失", "nan"],
    forbidden_tools=["bing_search", "baidu_search", "tavily_search"],
    max_turns=8,
    category="tool_selection",
)

TS_FILE_LIST = EvalScenario(
    id="ts-003",
    name="list_directory",
    user_input="列出当前目录下的所有文件",
    expected_tools=["file_list"],
    expected_keywords=["文件"],
    forbidden_tools=["bing_search", "baidu_search"],
    max_turns=5,
    category="tool_selection",
)

# ──────────────────────────────────────────────────────────────
# Skill priority scenarios
# ──────────────────────────────────────────────────────────────

SKILL_PRIORITY = EvalScenario(
    id="skill-001",
    name="skill_triggers_suppress_web_search",
    user_input="请查询武汉大学图书馆的开放时间",
    expected_tools=["data_query"],
    expected_keywords=["图书馆"],
    forbidden_tools=["bing_search", "baidu_search", "tavily_search"],
    max_turns=8,
    category="tool_selection",
)

# ──────────────────────────────────────────────────────────────
# Error recovery scenarios
# ──────────────────────────────────────────────────────────────

ERR_WRONG_PARAM = EvalScenario(
    id="err-001",
    name="wrong_params_then_recover",
    user_input="请读取不存在的文件路径 /nonexistent/file.txt 的内容",
    expected_tools=["file_read"],
    expected_keywords=["不存在", "找不到", "not found", "错误"],
    forbidden_tools=["bing_search", "baidu_search"],
    max_turns=6,
    category="error_recovery",
)

ERR_MODEL_UNAVAILABLE = EvalScenario(
    id="err-002",
    name="model_unavailable_retry",
    user_input="请分析 sales.xlsx 中的销售趋势",
    expected_tools=["data_query"],
    expected_keywords=["销售", "趋势"],
    forbidden_tools=["bing_search"],
    max_turns=12,
    category="error_recovery",
)

ERR_EMPTY_RESPONSE = EvalScenario(
    id="err-003",
    name="empty_response_recovery",
    user_input="请读取空文件 empty.txt 并总结内容",
    expected_tools=["file_read"],
    expected_keywords=["空"],
    forbidden_tools=["bing_search", "baidu_search"],
    max_turns=8,
    category="error_recovery",
)

ERR_NON_RETRIABLE = EvalScenario(
    id="err-004",
    name="non_retriable_terminate_gracefully",
    user_input="请用无效的 API key 连接服务并处理结果",
    expected_tools=[],
    expected_keywords=["终止", "失败", "错误"],
    forbidden_tools=["bing_search"],
    max_turns=6,
    category="error_recovery",
)

# ──────────────────────────────────────────────────────────────
# Context management scenarios
# ──────────────────────────────────────────────────────────────

CTX_LONG_CONVERSATION = EvalScenario(
    id="ctx-001",
    name="long_conversation_compression",
    user_input="请先读取 config.yaml，再读取 data.csv，然后读取 report.md，"
    "最后总结这三个文件的内容",
    expected_tools=["file_read"],
    expected_keywords=["config", "data", "report"],
    forbidden_tools=["bing_search", "baidu_search"],
    max_turns=12,
    category="context_management",
)

CTX_CONTINUATION = EvalScenario(
    id="ctx-002",
    name="multi_turn_continuation",
    user_input="继续",
    expected_tools=[],
    expected_keywords=["继续"],
    forbidden_tools=["bing_search"],
    max_turns=5,
    category="context_management",
)

# ──────────────────────────────────────────────────────────────
# Budget enforcement scenario
# ──────────────────────────────────────────────────────────────

BUDGET_ENFORCEMENT = EvalScenario(
    id="budget-001",
    name="token_turn_limit_enforcement",
    user_input="请执行一个非常复杂的数据分析任务，读取多个大文件并做交叉验证",
    expected_tools=["data_query", "file_read"],
    expected_keywords=[],
    forbidden_tools=["bing_search"],
    max_turns=15,
    category="general",
)

# ──────────────────────────────────────────────────────────────
# All scenarios
# ──────────────────────────────────────────────────────────────

ALL_SCENARIOS: list[EvalScenario] = [
    TS_FILE_READ,
    TS_DATA_QUERY,
    TS_FILE_LIST,
    SKILL_PRIORITY,
    ERR_WRONG_PARAM,
    ERR_MODEL_UNAVAILABLE,
    ERR_EMPTY_RESPONSE,
    ERR_NON_RETRIABLE,
    CTX_LONG_CONVERSATION,
    CTX_CONTINUATION,
    BUDGET_ENFORCEMENT,
]

# Quick lookup by id
SCENARIOS_BY_ID: dict[str, EvalScenario] = {s.id: s for s in ALL_SCENARIOS}

# Categories present in the scenario set
CATEGORIES: list[str] = sorted({s.category for s in ALL_SCENARIOS})
