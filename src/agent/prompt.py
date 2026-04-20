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


class IdentityComponent:
    def __init__(self, base_prompt_fn: Callable[[], str] | None = None) -> None:
        self._cached: str | None = None
        self._base_prompt_fn = base_prompt_fn

    def render(self, state: TurnState, config: Config) -> str | None:
        if self._base_prompt_fn is not None:
            prompt = str(self._base_prompt_fn() or "").strip()
            if prompt:
                return prompt
        if self._cached is None:
            candidates = [
                Path(__file__).parent.parent.parent / "SOUL.md",
            ]
            for p in candidates:
                if p.exists():
                    self._cached = p.read_text(encoding="utf-8")
                    break
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
    def __init__(self, routing_fn: Callable[[str], str]) -> None:
        self._fn = routing_fn
        self._cache_key: tuple[str, str] | None = None
        self._cached: str = ""

    def render(self, state: TurnState, config: Config) -> str | None:
        if not getattr(config, "enable_skill_routing", False):
            return None
        if state.classified_as in {"social", "informational"}:
            return None
        query = state.user_query or state.classified_as
        # Cache routing result for the duration of a turn — the query doesn't
        # change between iterations and routing involves an index scan.
        cache_key = (state.turn_id, query)
        if self._cache_key != cache_key:
            self._cache_key = cache_key
            self._cached = self._fn(query)
        if not self._cached.strip():
            return None
        return f"## Active Skill\n\n{self._cached}"


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
