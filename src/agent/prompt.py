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
        parts = [
            rendered
            for comp in self.components
            if (rendered := comp.render(state, config))
        ]
        return "\n\n".join(parts)


class IdentityComponent:
    def __init__(self) -> None:
        self._cached: str | None = None

    def render(self, state: TurnState, config: Config) -> str | None:
        if self._cached is None:
            candidates = [
                Path(__file__).parent.parent.parent / "SOUL.md",  # repo root
                Path.cwd() / "SOUL.md",
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


def default_prompt_builder(
    tool_schema_fn: Callable[[], str],
    routing_fn: Callable[[str], str],
    domain_schema_fn: Callable[[set[str]], str] | None = None,
) -> PromptBuilder:
    components: list[PromptComponent] = [
        IdentityComponent(),
        CoreToolSchemasComponent(tool_schema_fn),
    ]
    if domain_schema_fn is not None:
        components.append(DomainToolsComponent(domain_schema_fn))
    components.append(SkillPlaybookComponent(routing_fn))
    components.append(TurnContextComponent())
    return PromptBuilder(components)
