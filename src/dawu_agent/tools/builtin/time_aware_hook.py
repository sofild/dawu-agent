"""Time-aware hook for search tools — automatically injects time filters."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

# Default timezone for all date/time operations
DEFAULT_TZ = ZoneInfo("Asia/Shanghai")

# Time keyword mapping
TIME_KEYWORDS: dict[str, list[str]] = {
    "day": ["今天", "今日", "昨天", "昨日", "最近一天", "近一天", "24小时", "24h"],
    "week": ["最近一周", "近一周", "本周", "这周", "过去一周", "7天", "七天"],
    "month": ["最近一个月", "近一个月", "本月", "这个月", "过去一个月", "30天", "三十天"],
    "year": ["最近一年", "近一年", "今年", "过去一年", "365天"],
}

# Bing filters parameter mapping
BING_TIME_FILTERS: dict[str, str] = {
    "day": 'ex1:"ez5_1827"',
    "week": 'ex1:"ez5_1828"',
    "month": 'ex1:"ez5_1829"',
    "year": 'ex1:"ez5_1830"',
}


def detect_time_range(text: str) -> str | None:
    """Detect time range from user query."""
    for range_key, keywords in TIME_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return range_key
    return None


def get_date_for_query(time_range: str) -> str:
    """Get a date string to append to query for engines without native time filter."""
    now = datetime.now(DEFAULT_TZ)
    if time_range == "day":
        return now.strftime("%Y-%m-%d")
    elif time_range == "week":
        start = now - timedelta(days=6)
        return f"{start.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}"
    elif time_range == "month":
        start = now - timedelta(days=30)
        return f"{start.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}"
    elif time_range == "year":
        start = now - timedelta(days=364)
        return f"{start.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}"
    return ""


async def search_time_aware_hook(
    tool_name: str,
    tool_input: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """PreToolUse hook: inject time filters into search tools.

    This hook intercepts search tool calls and automatically injects
    time-related parameters when the user's query contains time keywords
    (e.g., today, yesterday, this week). It only injects parameters that
    the LLM did not already provide, respecting the LLM's explicit choices.
    """
    if tool_name not in ("bing_search", "baidu_search", "tavily_search"):
        return tool_input

    user_query = context.get("user_input", "")
    if not user_query:
        return tool_input

    time_range = detect_time_range(user_query)
    if not time_range:
        return tool_input

    modified = dict(tool_input)

    if tool_name == "bing_search":
        if "time_range" not in modified:
            modified["time_range"] = time_range
        query = modified.get("query", "")
        date_str = get_date_for_query(time_range)
        if date_str and date_str not in query:
            modified["query"] = f"{query} {date_str}"

    elif tool_name == "baidu_search":
        freshness_map = {
            "day": "pd",
            "week": "pw",
            "month": "pm",
            "year": "py",
        }
        if "freshness" not in modified:
            modified["freshness"] = freshness_map.get(time_range, "pd")

    elif tool_name == "tavily_search":
        if "topic" not in modified and time_range in ("day", "week"):
            modified["topic"] = "news"
        if "search_depth" not in modified:
            modified["search_depth"] = "advanced"

    return modified
