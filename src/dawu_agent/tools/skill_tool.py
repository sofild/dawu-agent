"""Generic SkillTool wrapper for SKILL.md-based skills.

A Skill is a directory under `skills/<name>/` containing a SKILL.md file
(with YAML front-matter) and an optional `scripts/` directory with the actual
Python implementation. The SkillTool exposes the whole skill as a single LLM
call: the model provides an `action` (function name in scripts/) and an `args`
dict which is forwarded to that function.

This is intentionally generic so adding a new skill requires zero changes to
the agent core.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from dawu_agent.tools.base import ConcurrencyMode, Tool, ToolCategory, ToolResult


# ---------------------------------------------------------------------------
# Per-skill action whitelist. The LLM may only invoke the actions listed here;
# each value is the ordered list of parameter names the function accepts.
# ---------------------------------------------------------------------------
ACTION_SCHEMAS: dict[str, list[str]] = {
    "query": ["sql", "db", "params"],
    "list_schools": ["limit"],
    "get_school": ["name", "dept_code"],
    "search_schools": ["keyword", "limit"],
    "list_models": [],
    "get_model_info": ["model_id"],
    "get_model_config": ["model_id"],
    "get_all_norms_for_model": ["model_id"],
    "get_school_model_task": ["school_id", "model_id"],
    "get_all_school_tasks": ["school_id"],
    "get_indicator_scores": ["task_id", "norm_ids"],
    "get_data_points": ["task_id"],
    "analyze_school_indicators": ["school_name", "model_id"],
    "search_tables": ["keyword", "limit"],
    "fetch_school_data": ["table_code", "dept_code", "limit"],
    "get_table_schema": ["table_code"],
    "list_db2_tables": ["limit"],
    "search_and_fetch_school_data": ["keyword", "school_name", "table_limit"],
    "decode_enum": ["table_code", "field_code", "value"],
    "get_field_enum_dict": ["table_code", "field_code"],
    "list_data_sets": [],
    "get_data_set_versions": ["data_set_id"],
    "to_json": ["data", "indent"],
}


def _slugify(name: str) -> str:
    """`gxpg-db-query` -> `gxpg_db_query`."""
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() and ch.isascii()) else "_")
    slug = "".join(out).strip("_").lower()
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def _load_module_from_path(py_path: Path, module_alias: str):
    """Import a python file as a uniquely-named module and return it."""
    spec = importlib.util.spec_from_file_location(module_alias, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = module
    spec.loader.exec_module(module)
    return module


class SkillTool(Tool):
    """Generic wrapper for SKILL.md-based skills.

    The tool name (LLM-facing) is derived from the skill directory name.
    Parameters: `action` (one of ACTION_SCHEMAS keys) and `args` (kwargs dict).
    """

    def __init__(
        self,
        skill_name: str,
        description: str,
        scripts_dir: Path,
        triggers: list[str] | None = None,
    ) -> None:
        self._skill_name = skill_name
        self._description = description
        self._scripts_dir = scripts_dir
        self._triggers = triggers or []
        self._module_cache: dict[str, Any] = {}

    # --- Tool interface ----------------------------------------------------

    @property
    def name(self) -> str:
        return _slugify(self._skill_name)

    @property
    def description(self) -> str:
        if self._triggers:
            trig = "、".join(self._triggers[:12])
            return (
                f"{self._description}\n\n"
                f"**触发关键词**：{trig}\n"
                f"当用户问题命中以上任意关键词或提到 '{self._skill_name}' 时，"
                f"**必须**优先调用本工具（不要用 web_search / bing_search）。\n"
                f"参数 action 取自 action 列表；args 为该函数的命名参数 dict。"
            )
        return (
            f"{self._description}\n"
            f"当用户问题与本 skill 主题相关时，**优先**调用本工具，"
            f"不要退回到通用 web_search。参数 action 取自 action 列表；"
            f"args 为该函数的命名参数 dict。"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        actions_enum = sorted(ACTION_SCHEMAS.keys())
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": actions_enum,
                    "description": (
                        f"要执行的函数名。可用值：{', '.join(actions_enum)}"
                    ),
                },
                "args": {
                    "type": "object",
                    "description": (
                        "传给该函数的命名参数 dict。未列出的键会被忽略；"
                        "未知 action 会被拒绝。"
                    ),
                },
            },
            "required": ["action"],
        }

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.DATA

    @property
    def concurrency_mode(self) -> ConcurrencyMode:
        return ConcurrencyMode.READ_ONLY

    @property
    def triggers(self) -> list[str]:
        return list(self._triggers)

    # --- Execution ---------------------------------------------------------

    def _resolve_module(self, script_name: str):
        """Return the loaded module for `script_name` (caches per script)."""
        if script_name in self._module_cache:
            return self._module_cache[script_name]

        py_path = self._scripts_dir / f"{script_name}.py"
        if not py_path.exists():
            raise FileNotFoundError(f"Skill script not found: {py_path}")
        alias = f"_skill_{_slugify(self._skill_name)}_{script_name}"
        module = _load_module_from_path(py_path, alias)
        self._module_cache[script_name] = module
        return module

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            action = arguments.get("action")
            if not action:
                return ToolResult.error("Missing required parameter: action")
            if action not in ACTION_SCHEMAS:
                return ToolResult.error(
                    f"Unknown action '{action}'. "
                    f"Allowed: {sorted(ACTION_SCHEMAS.keys())}"
                )

            allowed_params = ACTION_SCHEMAS[action]
            user_args = arguments.get("args") or {}
            if not isinstance(user_args, dict):
                return ToolResult.error("`args` must be a JSON object")

            # Filter to allowed params only
            kwargs: dict[str, Any] = {}
            for key in allowed_params:
                if key in user_args:
                    kwargs[key] = user_args[key]

            # Resolve the function. Convention: scripts/<script>.py, default to "db_query".
            script_name = "db_query"
            module = self._resolve_module(script_name)
            func = getattr(module, action, None)
            if func is None or not callable(func):
                return ToolResult.error(
                    f"Function '{action}' not found in {script_name}.py"
                )

            # Many of the db_query functions are sync; if so run in a thread.
            if inspect.iscoroutinefunction(func):
                raw = await func(**kwargs)
            else:
                import asyncio
                raw = await asyncio.to_thread(func, **kwargs)

            return ToolResult.ok(self._serialize(raw))

        except Exception as e:
            return ToolResult.error(f"Skill execution failed: {type(e).__name__}: {e}")

    @staticmethod
    def _serialize(data: Any) -> str:
        """Render the function result as a compact, LLM-friendly string."""
        try:
            if isinstance(data, str):
                return data
            return json.dumps(data, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(data)
