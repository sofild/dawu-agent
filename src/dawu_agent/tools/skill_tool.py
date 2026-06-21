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
# Base action whitelist. Each value is the ordered list of parameter names
# the function accepts.
#
# Resolution order (per-skill, at registration time):
#   1. If SKILL.md declares `exposed_actions` -> use that (manual override)
#   2. Else, auto-generate from db_query.py public functions via inspect
#   3. Fallback to this base whitelist if auto-generation fails
#
# This base dict is also kept as a safety net for legacy skills and as the
# default schema for skills whose scripts/ directory can't be introspected.
# ---------------------------------------------------------------------------
BASE_ACTION_SCHEMAS: dict[str, list[str]] = {
    "query": ["sql", "db", "params"],
    "list_schools": ["limit"],
    "get_school": ["name", "dept_code"],
    "list_models": [],
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

# Functions whose names look risky to expose even if defined in db_query.py.
# Belt-and-suspenders: auto-generation already skips names starting with `_`,
# this is a second guard against accidental exposure of helpers.
_DISALLOWED_ACTION_NAMES: set[str] = {
    "main", "setup", "teardown", "run", "init", "__init__",
    "os", "sys", "subprocess", "shutil", "pathlib",
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


def auto_generate_action_schemas(
    module: Any,
    extra_disallowed: set[str] | None = None,
) -> dict[str, list[str]]:
    """Introspect a loaded module and return an {action: [param_names]} dict.

    Rules:
    - Skip names starting with `_`
    - Skip names matching a built-in module (os, sys, ...) or unsafe name
    - Only pick top-level functions defined in the module itself (not re-imports)
    - Parameter order follows the function signature
    """
    disallowed = _DISALLOWED_ACTION_NAMES | (extra_disallowed or set())
    schemas: dict[str, list[str]] = {}
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        if name in disallowed:
            continue
        if not callable(func):
            continue
        # Only pick functions actually defined in this module (not imports).
        try:
            if getattr(func, "__module__", None) != module.__name__:
                continue
        except Exception:
            continue
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            continue
        params = [
            p.name
            for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        # Include zero-arg functions too
        schemas[name] = params
    return schemas


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
        action_schemas: dict[str, list[str]] | None = None,
    ) -> None:
        self._skill_name = skill_name
        self._description = description
        self._scripts_dir = scripts_dir
        self._triggers = triggers or []
        # If caller didn't pass explicit schemas, defer resolution to first use
        # so we pick up the latest script content (and respect SKILL.md
        # `exposed_actions` overrides at the loader level).
        self._explicit_schemas = action_schemas
        self._resolved_schemas: dict[str, list[str]] | None = action_schemas
        # (script_name -> (mtime_ns, module)) cache for hot-reload detection
        self._module_cache: dict[str, tuple[int, Any]] = {}
        # Resolve default script path lazily
        self._default_script = "db_query"

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
        actions_enum = sorted(self._get_action_schemas().keys())
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

    # --- Schema resolution -------------------------------------------------

    def _get_action_schemas(self) -> dict[str, list[str]]:
        """Return the active action whitelist, resolving it on first use.

        Resolution order (per-skill, at registration time):
          1. Auto-generate from the skill's default script (db_query.py) via
             inspect → produces the full set of module-level functions.
          2. If SKILL.md declares `exposed_actions`, **merge** it on top of
             the auto-generated set:
               - keys present in both: exposed_actions wins (parameter list
                 is the source of truth from SKILL.md).
               - keys only in exposed_actions: added (this is how we expose
                 functions imported from other modules that aren't in the
                 default script's namespace).
               - keys only in auto-generated: kept (so adding new functions
                 to db_query.py automatically becomes available).
          3. If the script is missing or unparsable, fall back to
             BASE_ACTION_SCHEMAS so the tool remains usable.

        On every call (after the first), this also checks whether the script
        file's mtime has changed since the cached module was loaded. If so,
        the module is reloaded and the schema is regenerated. This makes
        editing `db_query.py` automatically visible to the LLM without
        restarting the agent.
        """
        if self._resolved_schemas is not None:
            # Hot-reload check: if the script was modified, drop the cache
            # and fall through to regeneration.
            if self._explicit_schemas is None:
                if self._script_mtime_changed(self._default_script):
                    self._resolved_schemas = None
                    self._module_cache.pop(self._default_script, None)
                else:
                    return self._resolved_schemas
            else:
                return self._resolved_schemas

        # 1. Auto-generate from the script
        auto_schemas: dict[str, list[str]] = {}
        script_path = self._scripts_dir / f"{self._default_script}.py"
        if script_path.exists():
            try:
                module = self._resolve_module(self._default_script)
                auto_schemas = auto_generate_action_schemas(module)
            except Exception:
                auto_schemas = {}

        if not auto_schemas:
            auto_schemas = dict(BASE_ACTION_SCHEMAS)

        # 2. Merge explicit `exposed_actions` on top
        if self._explicit_schemas:
            merged: dict[str, list[str]] = dict(auto_schemas)
            for k, v in self._explicit_schemas.items():
                # exposed_actions is authoritative for that key
                merged[k] = list(v)
            schemas = merged
        else:
            schemas = auto_schemas

        self._resolved_schemas = schemas
        return schemas

    def _script_mtime_changed(self, script_name: str) -> bool:
        """Return True if the on-disk mtime differs from the cached one."""
        cached = self._module_cache.get(script_name)
        if cached is None:
            return False  # nothing cached yet, not a "change"
        cached_mtime, _ = cached
        try:
            current_mtime = (self._scripts_dir / f"{script_name}.py").stat().st_mtime_ns
        except OSError:
            return False
        return current_mtime != cached_mtime

    def refresh_schemas(self) -> dict[str, list[str]]:
        """Force re-resolution of the action whitelist on next access.

        Useful when the underlying script has been edited and you want the
        LLM-visible action list to update without restarting the agent.
        """
        self._resolved_schemas = None
        return self._get_action_schemas()

    # --- Execution ---------------------------------------------------------

    def clear_cache(self) -> None:
        """清空本 skill 的模块缓存，下次执行时强制重新加载代码."""
        for script_name, (_, module) in list(self._module_cache.items()):
            full_alias = getattr(module, "__name__", None) or (
                f"_skill_{_slugify(self._skill_name)}_{script_name}"
            )
            if full_alias in sys.modules:
                del sys.modules[full_alias]
        self._module_cache.clear()
        # Force re-resolution of auto-generated schemas on next call.
        self._resolved_schemas = None

    def _resolve_module(self, script_name: str):
        """Return the loaded module for `script_name`, hot-reloading on change.

        Compares the file's mtime_ns against the cached entry. If it changed
        (or no cache entry exists), the module is reloaded and the cache is
        updated. The active action whitelist is also invalidated so it picks
        up any new/removed functions.
        """
        py_path = self._scripts_dir / f"{script_name}.py"
        if not py_path.exists():
            raise FileNotFoundError(f"Skill script not found: {py_path}")

        try:
            current_mtime = py_path.stat().st_mtime_ns
        except OSError:
            current_mtime = -1

        cached = self._module_cache.get(script_name)
        if cached is not None:
            cached_mtime, cached_module = cached
            if cached_mtime == current_mtime:
                return cached_module
            # mtime changed -> evict and reload
            full_alias = getattr(cached_module, "__name__", None)
            if full_alias and full_alias in sys.modules:
                del sys.modules[full_alias]
            self._module_cache.pop(script_name, None)
            # Invalidate schema so it's regenerated from the new module.
            if self._explicit_schemas is None:
                self._resolved_schemas = None

        alias = f"_skill_{_slugify(self._skill_name)}_{script_name}"
        module = _load_module_from_path(py_path, alias)
        self._module_cache[script_name] = (current_mtime, module)
        return module

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            action_schemas = self._get_action_schemas()
            action = arguments.get("action")
            if not action:
                return ToolResult.error("Missing required parameter: action")
            if action not in action_schemas:
                return ToolResult.error(
                    f"Unknown action '{action}'. "
                    f"Allowed: {sorted(action_schemas.keys())}"
                )

            allowed_params = action_schemas[action]
            user_args = arguments.get("args") or {}
            if not isinstance(user_args, dict):
                return ToolResult.error("`args` must be a JSON object")

            # Filter to allowed params only — but capture any silently-dropped
            # keys so the LLM gets explicit feedback (was the #1 source of
            # wasted tool calls in the 武汉理工 session; see
            # reports/会话耗时分析_武汉理工图书馆.md P0-3).
            kwargs: dict[str, Any] = {}
            ignored: list[str] = []
            for key, value in user_args.items():
                if key in allowed_params:
                    kwargs[key] = value
                else:
                    ignored.append(key)
            if ignored:
                # Phase 2 优化（改 #7 + #11）：
                # - 加 [learned] tag 让 agent.py 提取并加入 failed_actions_block
                # - 加"相邻 action 提示"：列出该 skill 的所有 action 及其参数，
                #   让 LLM 知道下一步该用哪个 action
                # 不硬编码任何业务领域——相邻 action 从 action_schemas 动态派生
                nearby_hint = self._build_nearby_actions_hint(action, ignored)
                error_msg = (
                    f"[learned] Action '{action}' does not accept parameter(s): "
                    f"{sorted(ignored)}. "
                    f"Allowed: {sorted(allowed_params)}. "
                    f"Re-call with the correct parameter names.\n"
                    f"{nearby_hint}"
                )
                return ToolResult.error(error_msg)

            # Resolve the function. Convention: scripts/<script>.py, default to "db_query".
            script_name = self._default_script
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

    def _build_nearby_actions_hint(
        self, current_action: str, ignored: list[str]
    ) -> str:
        """构建"相邻 action 提示"，列出该 skill 的所有 action 及其参数。

        用途：当 LLM 调错 action 或参数时，让它知道"也许你想用的是 X action"。

        不硬编码业务领域——hint 从 `_get_action_schemas()` 动态派生。

        Args:
            current_action: 当前 LLM 调用的 action
            ignored: 被拒绝的参数列表

        Returns:
            多行提示字符串。如果 skill 中没有其他 action 则返回空字符串。
        """
        try:
            action_schemas = self._get_action_schemas()
        except Exception:
            return ""

        other_actions = {
            k: v for k, v in action_schemas.items() if k != current_action
        }
        if not other_actions:
            return ""

        # 按名称相似度排序（共享子串越多越靠前）
        def name_similarity(name: str) -> int:
            score = 0
            for ig in ignored:
                # 如果被拒绝的参数名出现在 action 名中，加分
                if ig in name:
                    score += 3
            # 简单的字符交集
            common = set(current_action) & set(name)
            score += len(common)
            return score

        sorted_actions = sorted(
            other_actions.items(),
            key=lambda kv: name_similarity(kv[0]),
            reverse=True,
        )
        # 取 top 3
        top = sorted_actions[:3]

        lines = ["[hint] Other actions in this skill (you may have wanted one of these):"]
        for name, params in top:
            param_str = ", ".join(params) if params else "(no params)"
            lines.append(f"  - {name}({param_str})")
        return "\n".join(lines)
