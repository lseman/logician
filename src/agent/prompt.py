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
    _cached: str | None = None

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

    def render(self, state: TurnState, config: Config) -> str | None:
        if not getattr(config, "enable_skill_routing", False):
            return None
        playbook = self._fn(state.classified_as)
        if not playbook.strip():
            return None
        return f"## Active Skill\n\n{playbook}"


class TurnContextComponent:
    def render(self, state: TurnState, config: Config) -> str | None:
        if state.iteration == 0 and not state.files_written:
            return None
        lines = [f"Iteration: {state.iteration}"]
        if state.files_written:
            lines.append(f"Files written this turn: {', '.join(state.files_written)}")
            lines.append("Remember to verify your changes with tests or a linter.")
        return "## Turn Context\n\n" + "\n".join(lines)


def default_prompt_builder(
    tool_schema_fn: Callable[[], str],
    domain_schema_fn: Callable[[set[str]], str],
    routing_fn: Callable[[str], str],
) -> PromptBuilder:
    return PromptBuilder([
        IdentityComponent(),
        CoreToolSchemasComponent(tool_schema_fn),
        DomainToolsComponent(domain_schema_fn),
        SkillPlaybookComponent(routing_fn),
        TurnContextComponent(),
    ])
