"""Training and evaluation datasets for DSPy optimization."""

from __future__ import annotations

import dspy

# Action decision training examples
ACTION_TRAINSET = [
    dspy.Example(
        conversation_history="[user] 查询2024年高校评估数据",
        user_query="查询2024年高校评估数据",
        available_tools="data_query: Query data using Pandas/SQL; bing_search: Search the web",
        current_time="2024-12-15 10:00:00 CST",
        memory_context="",
        action="use_tool",
        tool_name="data_query",
        tool_arguments='{"query": "SELECT * FROM evaluation WHERE year=2024"}',
        reasoning="User wants data, should use data_query tool",
    ).with_inputs(
        "conversation_history", "user_query", "available_tools",
        "current_time", "memory_context",
    ),
    dspy.Example(
        conversation_history="[user] 你好 [assistant] 你好！有什么可以帮你的？ [user] 谢谢",
        user_query="谢谢",
        available_tools="data_query, file_read, bing_search",
        current_time="2024-12-15 10:01:00 CST",
        memory_context="",
        action="stop",
        tool_name="",
        tool_arguments="",
        reasoning="User is thanking, conversation is complete",
    ).with_inputs(
        "conversation_history", "user_query", "available_tools",
        "current_time", "memory_context",
    ),
    dspy.Example(
        conversation_history="[user] 帮我搜索一下最新的AI技术趋势",
        user_query="帮我搜索一下最新的AI技术趋势",
        available_tools=(
            "data_query: Query data; bing_search: Search the web; "
            "baidu_search: AI search"
        ),
        current_time="2024-12-15 10:02:00 CST",
        memory_context="",
        action="use_tool",
        tool_name="baidu_search",
        tool_arguments='{"query": "最新AI技术趋势 2024"}',
        reasoning="User wants to search for latest AI trends, should use search tool",
    ).with_inputs(
        "conversation_history", "user_query", "available_tools",
        "current_time", "memory_context",
    ),
    dspy.Example(
        conversation_history=(
            "[user] 查询销售数据 [assistant] 好的 [tool result] "
            "显示2024年Q1-Q4销售数据 [assistant] 数据已获取 "
            "[user] 帮我生成一个图表"
        ),
        user_query="帮我生成一个图表",
        available_tools="data_query, data_visualize, report_generate",
        current_time="2024-12-15 10:05:00 CST",
        memory_context="",
        action="use_tool",
        tool_name="data_visualize",
        tool_arguments='{"chart_type": "bar", "data_source": "sales_data"}',
        reasoning="User wants a chart, should use data_visualize tool",
    ).with_inputs(
        "conversation_history", "user_query", "available_tools",
        "current_time", "memory_context",
    ),
    dspy.Example(
        conversation_history=(
            "[user] 什么是数据分析？ [assistant] 数据分析是通过统计和逻辑方法..."
        ),
        user_query="什么是数据分析？",
        available_tools="data_query, file_read, bing_search",
        current_time="2024-12-15 10:06:00 CST",
        memory_context="",
        action="respond",
        tool_name="",
        tool_arguments="",
        reasoning="User asks a general knowledge question, respond directly",
    ).with_inputs(
        "conversation_history", "user_query", "available_tools",
        "current_time", "memory_context",
    ),
    dspy.Example(
        conversation_history=(
            "[user] 读取data.csv文件 [assistant] 好的 [tool result] "
            "文件内容：name,age,score... [user] 帮我分析这些数据的分布"
        ),
        user_query="帮我分析这些数据的分布",
        available_tools="data_query, data_visualize, report_generate",
        current_time="2024-12-15 10:08:00 CST",
        memory_context="",
        action="use_tool",
        tool_name="data_query",
        tool_arguments='{"query": "DESCRIBE data; SELECT * FROM data LIMIT 100"}',
        reasoning="User wants data analysis, should use data_query tool",
    ).with_inputs(
        "conversation_history", "user_query", "available_tools",
        "current_time", "memory_context",
    ),
]


def get_trainset() -> list[dspy.Example]:
    """Return the training dataset."""
    return ACTION_TRAINSET


def get_devset() -> list[dspy.Example]:
    """Return the development dataset for evaluation."""
    return ACTION_TRAINSET[:3]
