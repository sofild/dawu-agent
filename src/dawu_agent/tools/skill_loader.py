"""Auto-discover SKILL.md-based skills from a directory and register them as Tools."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dawu_agent.tools.base import Tool
from dawu_agent.tools.registry import ToolRegistry
from dawu_agent.tools.skill_tool import SkillTool


# Regex for YAML front-matter delimited by '---' on the first line.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class SkillDefinition:
    """A discovered skill (pre-registration)."""

    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    scripts_dir: Path | None = None
    skill_dir: Path | None = None
    tool_name: str = ""

    def to_tool(self) -> Tool:
        assert self.scripts_dir is not None
        return SkillTool(
            skill_name=self.name,
            description=self.description,
            scripts_dir=self.scripts_dir,
            triggers=self.triggers,
        )


class SkillLoader:
    """Scan `skills/` and register each subdirectory as a SkillTool."""

    def __init__(self, skills_dir: str | Path = "skills") -> None:
        self._skills_dir = Path(skills_dir)

    def discover(self) -> list[SkillDefinition]:
        if not self._skills_dir.exists():
            return []

        results: list[SkillDefinition] = []
        for child in sorted(self._skills_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name.startswith(("_", ".")):
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                defn = self._parse_skill_md(skill_md, child)
            except Exception:
                # Skip malformed skills rather than failing the whole agent.
                continue
            results.append(defn)
        return results

    def register_all(self, registry: ToolRegistry) -> int:
        """Register every discovered skill into the registry. Returns count."""
        count = 0
        for defn in self.discover():
            try:
                tool = defn.to_tool()
            except Exception:
                continue
            try:
                registry.register(tool, core=False)
            except Exception:
                continue
            # Stash triggers for context-aware selection
            if hasattr(registry, "register_triggers"):
                try:
                    registry.register_triggers(tool.name, defn.triggers)
                except Exception:
                    pass
            count += 1
        return count

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_skill_md(path: Path, skill_dir: Path) -> SkillDefinition:
        raw = path.read_text(encoding="utf-8")
        meta: dict[str, Any] = {}
        body = raw
        m = _FRONTMATTER_RE.match(raw)
        if m:
            meta = _parse_simple_yaml(m.group(1))
            body = raw[m.end():]

        name = str(meta.get("name") or skill_dir.name)
        description = str(
            meta.get("description")
            or _extract_first_paragraph(body)
            or skill_dir.name
        )

        triggers_raw = meta.get("triggers") or meta.get("keywords") or []
        if isinstance(triggers_raw, str):
            triggers = [t.strip() for t in re.split(r"[,，、\s]+", triggers_raw) if t.strip()]
        elif isinstance(triggers_raw, list):
            triggers = [str(t).strip() for t in triggers_raw if str(t).strip()]
        else:
            triggers = []

        # Augment triggers from description text (best-effort keyword boost)
        triggers.extend(_extract_zh_keywords(description, max_keywords=12))
        # Dedupe preserving order
        seen: set[str] = set()
        triggers_dedup: list[str] = []
        for t in triggers:
            if t not in seen:
                seen.add(t)
                triggers_dedup.append(t)

        scripts_dir = skill_dir / "scripts"
        if not scripts_dir.exists():
            # Even without scripts/, register so LLM can be informed.
            scripts_dir = skill_dir

        from dawu_agent.tools.skill_tool import _slugify

        return SkillDefinition(
            name=name,
            description=description.strip(),
            triggers=triggers_dedup,
            scripts_dir=scripts_dir,
            skill_dir=skill_dir,
            tool_name=_slugify(name),
        )


def _parse_simple_yaml(block: str) -> dict[str, Any]:
    """Tiny YAML subset parser: `key: value` per line, and `key: [a, b, c]` lists."""
    out: dict[str, Any] = {}
    for line in block.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1]
            out[key] = [s.strip().strip('"\'') for s in inner.split(",") if s.strip()]
        elif value.startswith('"') and value.endswith('"'):
            out[key] = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            out[key] = value[1:-1]
        else:
            out[key] = value
    return out


def _extract_first_paragraph(body: str) -> str:
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if para and not para.startswith("#"):
            return para
    return ""


_ZH_KEYWORDS = {
    "高校", "评估", "指标", "采集点", "学校", "数据库", "项目",
    "学生", "教师", "教学", "科研", "学科", "专业", "图书馆",
    "实验室", "经费", "管理", "课程", "招生", "就业", "国际",
    "学术", "建设", "发展", "质量", "绩效", "统计", "分析",
}


def _extract_zh_keywords(text: str, max_keywords: int = 12) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for kw in _ZH_KEYWORDS:
        if kw in text and kw not in seen:
            found.append(kw)
            seen.add(kw)
        if len(found) >= max_keywords:
            break
    return found
