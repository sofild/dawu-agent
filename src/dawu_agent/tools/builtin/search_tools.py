"""Search engine tools for external data retrieval."""

from __future__ import annotations

import os
from typing import Any

import requests

from dawu_agent.tools.base import ConcurrencyMode, Tool, ToolCategory, ToolResult


class BingSearchTool(Tool):
    """Bing search engine (no API key required)."""

    @property
    def name(self) -> str:
        return "bing_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using Bing to retrieve publicly available information. "
            "Use this when the user asks about current events, facts, or data not "
            "available in the local context. "
            "Do NOT use this for local file operations or data analysis - use file_read or data_query instead.\n"
            "When the user mentions time (today, yesterday, this week, this month, etc.), "
            "pass the time_range parameter to filter results by time.\n"
            "Parameters:\n"
            "  - query (string, required): Search keywords\n"
            "  - max_results (integer, optional): Maximum results to return (default 10, max 20)\n"
            "  - time_range (string, optional): Time filter - 'day', 'week', 'month', 'year'\n"
            "Returns: List of search results with title, url, and snippet"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 20,
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "Time filter: day=last 24h, week=last 7d, month=last 30d, year=last 365d",
                },
            },
            "required": ["query"],
        }

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.DATA

    @property
    def concurrency_mode(self) -> ConcurrencyMode:
        return ConcurrencyMode.READ_ONLY

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            query = arguments["query"]
            max_results = min(int(arguments.get("max_results", 10)), 20)
            time_range = arguments.get("time_range")

            from urllib.parse import quote
            from lxml import html

            search_url = f"https://www.bing.com/search?q={quote(query)}"

            # Apply Bing time filter if specified
            if time_range:
                bing_filters = {
                    "day": 'ex1:"ez5_1827"',
                    "week": 'ex1:"ez5_1828"',
                    "month": 'ex1:"ez5_1829"',
                    "year": 'ex1:"ez5_1830"',
                }
                if time_range in bing_filters:
                    search_url += f"&filters={bing_filters[time_range]}"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }

            response = requests.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()

            tree = html.fromstring(response.content)
            items = tree.xpath("//li[contains(@class, 'b_algo')]")

            results = []
            for item in items[:max_results]:
                try:
                    title_elems = item.xpath(".//h2//a//text()")
                    href_elems = item.xpath(".//h2//a/@href")
                    body_elems = item.xpath(".//div[contains(@class, 'b_caption')]//p//text()")

                    if title_elems and href_elems:
                        title = "".join(title_elems).strip()
                        href = "".join(href_elems).strip()
                        body = "".join(body_elems).strip() if body_elems else ""

                        if title and href:
                            results.append({
                                "title": title,
                                "url": href,
                                "snippet": body,
                            })
                except Exception:
                    continue

            if not results:
                return ToolResult.ok("No results found.")

            output_lines = [f"Bing search results for '{query}':\n"]
            for i, r in enumerate(results, 1):
                output_lines.append(f"[{i}] {r['title']}")
                output_lines.append(f"    URL: {r['url']}")
                if r["snippet"]:
                    output_lines.append(f"    {r['snippet']}")
                output_lines.append("")

            return ToolResult.ok("\n".join(output_lines))

        except ImportError as e:
            return ToolResult.error(f"Required library not installed: {e}")
        except Exception as e:
            return ToolResult.error(f"Bing search failed: {e}")


class BaiduSearchTool(Tool):
    """Baidu Qianfan AI Search."""

    @property
    def name(self) -> str:
        return "baidu_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using Baidu Qianfan AI Search for Chinese content or "
            "when Bing results are insufficient. "
            "Requires BAIDU_API_KEY environment variable. "
            "Use this for Chinese queries, local Chinese news, or China-specific information.\n"
            "When the user mentions time (today, yesterday, this week, this month, etc.), "
            "pass the freshness parameter to filter results by time. "
            "Example: user asks '今天的新闻' -> pass freshness='pd'.\n"
            "Parameters:\n"
            "  - query (string, required): Search keywords\n"
            "  - max_results (integer, optional): Maximum results (default 10, max 50)\n"
            "  - freshness (string, optional): Time filter - 'pd'(day), 'pw'(week), 'pm'(month), 'py'(year)\n"
            "Returns: List of search results with title, url, and content"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                },
                "freshness": {
                    "type": "string",
                    "enum": ["pd", "pw", "pm", "py"],
                    "description": "Time filter: pd=day, pw=week, pm=month, py=year",
                },
            },
            "required": ["query"],
        }

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.DATA

    @property
    def concurrency_mode(self) -> ConcurrencyMode:
        return ConcurrencyMode.READ_ONLY

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            api_key = os.getenv("BAIDU_API_KEY") or os.getenv("BAIDU_QIANFAN_API_KEY")
            if not api_key:
                return ToolResult.error(
                    "BAIDU_API_KEY not configured. Please set it in the .env file."
                )

            query = arguments["query"]
            count = min(int(arguments.get("max_results", 10)), 50)
            freshness = arguments.get("freshness")

            from datetime import datetime, timedelta
            from zoneinfo import ZoneInfo

            # Build time filter using Asia/Shanghai timezone
            search_filter = {}
            if freshness:
                tz = ZoneInfo("Asia/Shanghai")
                current_time = datetime.now(tz)
                end_date = (current_time + timedelta(days=1)).strftime("%Y-%m-%d")
                if freshness == "pd":
                    start_date = (current_time - timedelta(days=1)).strftime("%Y-%m-%d")
                elif freshness == "pw":
                    start_date = (current_time - timedelta(days=6)).strftime("%Y-%m-%d")
                elif freshness == "pm":
                    start_date = (current_time - timedelta(days=30)).strftime("%Y-%m-%d")
                elif freshness == "py":
                    start_date = (current_time - timedelta(days=364)).strftime("%Y-%m-%d")
                else:
                    start_date = (current_time - timedelta(days=1)).strftime("%Y-%m-%d")
                search_filter = {
                    "range": {"page_time": {"gte": start_date, "lt": end_date}}
                }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            request_body = {
                "messages": [{"content": query, "role": "user"}],
                "search_source": "baidu_search_v2",
                "resource_type_filter": [{"type": "web", "top_k": count}],
            }
            if search_filter:
                request_body["search_filter"] = search_filter

            response = requests.post(
                "https://qianfan.baidubce.com/v2/ai_search/web_search",
                json=request_body,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and data.get("code"):
                return ToolResult.error(f"Baidu API error: {data.get('message', 'Unknown error')}")

            references = data.get("references", []) if isinstance(data, dict) else []

            if not references:
                return ToolResult.ok("No results found.")

            output_lines = [f"Baidu search results for '{query}':\n"]
            for i, item in enumerate(references[:count], 1):
                title = item.get("title", "N/A")
                url = item.get("url", "N/A")
                content = item.get("content", "")
                output_lines.append(f"[{i}] {title}")
                output_lines.append(f"    URL: {url}")
                if content:
                    snippet = content[:200] + "..." if len(content) > 200 else content
                    output_lines.append(f"    {snippet}")
                output_lines.append("")

            return ToolResult.ok("\n".join(output_lines))

        except Exception as e:
            return ToolResult.error(f"Baidu search failed: {e}")


class TavilySearchTool(Tool):
    """Tavily AI Search - high-quality web search with optional AI-generated answers."""

    @property
    def name(self) -> str:
        return "tavily_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using Tavily AI for high-quality, curated results. "
            "Supports advanced search depth and topic filtering. "
            "Requires TAVILY_API_KEY environment variable. "
            "Use this for research tasks requiring comprehensive or recent information.\n"
            "When the user asks about recent events (today, yesterday, this week, etc.), "
            "set topic='news' and search_depth='advanced' for better results.\n"
            "Parameters:\n"
            "  - query (string, required): Search keywords\n"
            "  - max_results (integer, optional): Maximum results (default 5, max 20)\n"
            "  - search_depth (string, optional): 'basic', 'advanced', or 'fast'\n"
            "  - topic (string, optional): 'general', 'news', or 'finance'\n"
            "  - include_answer (boolean, optional): Include AI-generated summary (default false)\n"
            "Returns: Search results with title, url, content, and optional AI answer"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced", "fast"],
                    "description": "Search depth level",
                },
                "topic": {
                    "type": "string",
                    "enum": ["general", "news", "finance"],
                    "description": "Search topic category",
                },
                "include_answer": {
                    "type": "boolean",
                    "description": "Include AI-generated answer summary",
                    "default": False,
                },
            },
            "required": ["query"],
        }

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.DATA

    @property
    def concurrency_mode(self) -> ConcurrencyMode:
        return ConcurrencyMode.READ_ONLY

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                return ToolResult.error(
                    "TAVILY_API_KEY not configured. Please set it in the .env file."
                )

            query = arguments["query"]
            max_results = min(int(arguments.get("max_results", 5)), 20)
            search_depth = arguments.get("search_depth")
            topic = arguments.get("topic")
            include_answer = arguments.get("include_answer", False)

            try:
                from tavily import TavilyClient
            except ImportError:
                return ToolResult.error(
                    "tavily-python not installed. Run: pip install tavily-python"
                )

            client = TavilyClient(api_key=api_key)

            kwargs = {"query": query, "max_results": max_results}
            if search_depth:
                kwargs["search_depth"] = search_depth
            if topic:
                kwargs["topic"] = topic
            if include_answer:
                kwargs["include_answer"] = True

            result = client.search(**kwargs)

            output_lines = [f"Tavily search results for '{query}':\n"]

            if result.get("answer"):
                output_lines.append(f"AI Summary: {result['answer']}\n")

            results = result.get("results", [])
            if not results:
                return ToolResult.ok("No results found.")

            for i, item in enumerate(results, 1):
                title = item.get("title", "N/A")
                url = item.get("url", "N/A")
                content = item.get("content", "")
                output_lines.append(f"[{i}] {title}")
                output_lines.append(f"    URL: {url}")
                if content:
                    snippet = content[:200] + "..." if len(content) > 200 else content
                    output_lines.append(f"    {snippet}")
                output_lines.append("")

            return ToolResult.ok("\n".join(output_lines))

        except Exception as e:
            return ToolResult.error(f"Tavily search failed: {e}")
