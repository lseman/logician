"""PromptBuilder: composable system prompt assembly pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from ..config import Config
from .state import TurnState


@runtime_checkable
class PromptComponent(Protocol):
    def render(self, state: TurnState, config: Config) -> str | None: ...


class PromptBuilder:
    def __init__(self, components: list[PromptComponent]) -> None:
        self.components = components

    def build(self, state: TurnState, config: Config) -> str:
        parts = [rendered for comp in self.components if (rendered := comp.render(state, config))]
        return "\n\n".join(parts)


def discover_context_files(base_dir: Path) -> list[Path]:
    """Find all AGENTS.md and CLAUDE.md from cwd up to git root, like Pi does."""
    files: list[Path] = []
    seen = set()
    current = base_dir.resolve()
    # Try to find git root
    git_root = current
    while len(str(git_root)) > 1:
        if (git_root / ".git").exists():
            break
        git_root = git_root.parent
    while current != current.parent and current != git_root.parent:
        for name in ("AGENTS.md", "CLAUDE.md", "AGENTS.MD", "CLAUDE.MD"):
            p = current / name
            if p.exists() and p not in seen:
                files.append(p)
                seen.add(p)
        current = current.parent
    return files


class IdentityComponent:
    def __init__(
        self,
        base_prompt_fn: Callable[[], str] | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self._cached: str | None = None
        self._base_prompt_fn = base_prompt_fn
        self._base_dir = base_dir or Path.cwd()

    def render(self, state: TurnState, config: Config) -> str | None:
        if self._base_prompt_fn is not None:
            prompt = str(self._base_prompt_fn() or "").strip()
            if prompt:
                return prompt
        if self._cached is None:
            # Primary: SOUL.md (project-level)
            soul = Path(__file__).parent.parent.parent / "SOUL.md"
            parts: list[str] = []

            # Discover AGENTS.md / CLAUDE.md up the tree (like Pi)
            context_files = discover_context_files(self._base_dir)

            if soul.exists():
                parts.append(f"# SOUL\n\n{soul.read_text(encoding='utf-8')}")
            for p in context_files:
                parts.append(f"# {p.name}\n\n{p.read_text(encoding='utf-8')}")

            if parts:
                self._cached = "\n\n".join(parts)
            else:
                self._cached = "You are a capable coding agent."
        return self._cached


class CoreToolSchemasComponent:
    def __init__(self, tool_schema_fn: Callable[[], str]) -> None:
        self._fn = tool_schema_fn

    def render(self, state: TurnState, config: Config) -> str | None:
        schema = self._fn()
        if not schema.strip():
            return None
        return f"## Available Tools\n\n{schema}"


class DomainToolsComponent:
    def __init__(self, domain_schema_fn: Callable[[set[str]], str]) -> None:
        self._fn = domain_schema_fn

    def render(self, state: TurnState, config: Config) -> str | None:
        if not state.domain_groups_activated:
            return None
        schema = self._fn(state.domain_groups_activated)
        if not schema.strip():
            return None
        return f"## Domain Tools\n\n{schema}"


class SkillPlaybookComponent:
    """Pi-style: static skill list with 1-line descriptions in system prompt.

    Replaces the expensive dynamic skill routing (~1,493 tokens/turn) with
    a compact, static skill catalog (~600 tokens). The model discovers
    skills by name from the list and reads SKILL.md on demand.
    """

    def __init__(
        self,
        routing_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._fn = routing_fn  # kept for backward compat / complex queries
        self._cached: str | None = None

    # Essential skill names (normalized) — always shown.
    # These cover ~80% of tasks and are the "core tools" of skills.
    ESSENTIAL_NAMES = frozenset({
        "think", "todo", "scratch", "orchestrator",
        "explore", "edit_block", "multi_edit", "search_replace", "patch", "quality",
        "shell", "git", "repl", "web",
        "test_driven_development", "systematic_debugging",
        "verification_before_completion", "writing_plans",
        "memory_management",
        "plan_mode", "subagent_driven_development", "executing_plans",
        "requests",
    })

    def _build_skill_list(self) -> str:
        """Build compact skill list: name — short description (Pi-style).

        Strategy: show only essential skills with ~30-char descriptions.
        Pi shows 7 tools with 1-line descriptions. This shows ~20 skills.
        """
        try:
            from src.tools import ToolRegistry
            registry = ToolRegistry()
            skills = registry.list_skills()
        except Exception:
            return ""

        # Filter to essential skills only
        essential = []
        for s in skills:
            normalized = s.name.lower().replace(" ", "_")
            if normalized in self.ESSENTIAL_NAMES or normalized.startswith("explore") or normalized.startswith("todo"):
                essential.append(s)

        # Also match by ID prefix
        id_prefixes = {"think", "todo", "scratch", "shell", "git", "repl", "web",
                       "edit_block", "multi_edit", "search_replace", "patch", "quality",
                       "test_driven", "systematic_debugging", "verification", "writing_plan",
                       "memory_management", "plan_mode", "subagent", "executing", "requests"}
        for s in skills:
            if s.id.lower().split("__")[0] in id_prefixes:
                if s not in essential:
                    essential.append(s)

        # Deduplicate
        seen = set()
        essential = [s for s in essential if s.id not in seen and not seen.add(s.id)]

        lines: list[str] = ["## Available Skills"]
        lines.append("Use these skills when they match your task. Read SKILL.md for full instructions.")
        lines.append("")

        for s in sorted(essential, key=lambda x: x.name.lower()):
            desc = (s.summary or s.description or "").strip()
            # Truncate to ~40 chars — short enough for compact display
            if len(desc) > 40:
                desc = desc[:37] + "..."
            lines.append(f"- **{s.name}**: {desc}")

        lines.append("")
        lines.append(
            "## Skill Loading\n"
            "When a skill above matches your task, read its SKILL.md for full instructions. "
            "Skills live under `skills/`. Example: `read_file path=skills/global/think/SKILL.md`"
        )

        return "\n".join(lines)

    def render(self, state: TurnState, config: Config) -> str | None:
        if not getattr(config, "enable_skill_routing", False):
            return None
        if state.classified_as in {"social", "informational"}:
            return None

        if self._cached is None:
            self._cached = self._build_skill_list()

        if not self._cached.strip():
            return None

        # For complex queries, also append routing context
        if self._fn is not None and len(self._cached) > 200:
            query = state.user_query or state.classified_as
            cache_key = (state.turn_id, query)
            # We use a simple heuristic: if the query is complex (long, has multiple concepts),
            # append routing results
            if len(str(query or "")) > 100:
                routing = self._fn(query)
                return f"{self._cached}\n\n## Skill Routing (complex query)\n{routing}"

        return self._cached


class TurnContextComponent:
    def render(self, state: TurnState, config: Config) -> str | None:
        if not state.files_written:
            return None
        lines = [f"Files written this turn: {', '.join(state.files_written)}"]
        lines.append("Verify with tests or a linter before finishing.")
        return "## Turn Context\n\n" + "\n".join(lines)


class RuntimeContextComponent:
    def __init__(self, runtime_context_fn: Callable[[], str]) -> None:
        self._fn = runtime_context_fn

    def render(self, state: TurnState, config: Config) -> str | None:
        summary = str(self._fn() or "").strip()
        if not summary:
            return None
        return f"## Runtime Context\n\n{summary}"


class RetrievalContextComponent:
    def __init__(self, retrieval_context_fn: Callable[[TurnState], str]) -> None:
        self._fn = retrieval_context_fn
        self._cache_key: tuple[str, str] | None = None
        self._cached: str = ""

    def render(self, state: TurnState, config: Config) -> str | None:
        if not getattr(config, "prompt_rag_context_enabled", True):
            return None
        if state.classified_as in {"social", "informational"}:
            return None
        query = str(state.user_query or "").strip()
        if not query:
            return None
        cache_key = (state.turn_id, query)
        if self._cache_key != cache_key:
            self._cache_key = cache_key
            self._cached = str(self._fn(state) or "").strip()
        if not self._cached:
            return None
        return f"## Retrieval Context\n\n{self._cached}"


class PythonEditingPreferenceComponent:
    def render(self, state: TurnState, config: Config) -> str | None:
        if not getattr(config, "python_structural_editing_preference", True):
            return None

        available = set(state.available_tool_names or set())
        structural = {
            "edit_file_libcst",
            "replace_function_body",
            "replace_docstring",
            "find_function_by_name",
        }
        if not available.intersection(structural):
            return None

        query = str(state.user_query or "").lower()
        looks_python = (
            ".py" in query
            or "python" in query
            or "function" in query
            or "class" in query
            or any(str(path).endswith(".py") for path in state.files_written)
            or any(str(path).endswith(".py") for path in state.files_read)
        )
        if not looks_python:
            return None

        return (
            "## Python Editing Preference\n\n"
            "For Python changes, prefer structural LibCST/symbol-aware tools over raw text edits when possible.\n"
            "Use `find_function_by_name` / `find_class_by_name` to inspect symbols first, then prefer "
            "`replace_function_body`, `replace_docstring`, or `edit_file_libcst` instead of raw `edit_file`."
        )


def default_prompt_builder(
    base_prompt_fn: Callable[[], str] | None,
    tool_schema_fn: Callable[[], str],
    routing_fn: Callable[[str], str],
    runtime_context_fn: Callable[[], str] | None = None,
    retrieval_context_fn: Callable[[TurnState], str] | None = None,
    domain_schema_fn: Callable[[set[str]], str] | None = None,
) -> PromptBuilder:
    components: list[PromptComponent] = [
        IdentityComponent(base_prompt_fn),
        CoreToolSchemasComponent(tool_schema_fn),
    ]
    if domain_schema_fn is not None:
        components.append(DomainToolsComponent(domain_schema_fn))
    components.append(SkillPlaybookComponent(routing_fn))
    components.append(PythonEditingPreferenceComponent())
    if retrieval_context_fn is not None:
        components.append(RetrievalContextComponent(retrieval_context_fn))
    if runtime_context_fn is not None:
        components.append(RuntimeContextComponent(runtime_context_fn))
    components.append(TurnContextComponent())
    return PromptBuilder(components)
