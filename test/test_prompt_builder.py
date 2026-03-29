"""Tests for PromptBuilder and its components."""
from __future__ import annotations

import pytest

from src.agent.prompt import (
    CoreToolSchemasComponent,
    DomainToolsComponent,
    IdentityComponent,
    PromptBuilder,
    RetrievalContextComponent,
    RuntimeContextComponent,
    SkillPlaybookComponent,
    TurnContextComponent,
    default_prompt_builder,
)
from src.agent.state import TurnState
from src.config import Config


def make_state(**kwargs) -> TurnState:
    defaults = dict(turn_id="test-turn-1")
    defaults.update(kwargs)
    return TurnState(**defaults)


def make_config(**kwargs) -> Config:
    cfg = Config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# PromptBuilder core behaviour
# ---------------------------------------------------------------------------

def test_empty_builder_returns_empty_string():
    builder = PromptBuilder([])
    state = make_state()
    config = make_config()
    assert builder.build(state, config) == ""


def test_component_returning_none_is_skipped():
    class NullComp:
        def render(self, state, config):
            return None

    class TextComp:
        def render(self, state, config):
            return "hello"

    builder = PromptBuilder([NullComp(), TextComp(), NullComp()])
    assert builder.build(make_state(), make_config()) == "hello"


def test_components_joined_with_double_newline():
    class A:
        def render(self, state, config):
            return "part-a"

    class B:
        def render(self, state, config):
            return "part-b"

    builder = PromptBuilder([A(), B()])
    result = builder.build(make_state(), make_config())
    assert result == "part-a\n\npart-b"


# ---------------------------------------------------------------------------
# IdentityComponent
# ---------------------------------------------------------------------------

def test_identity_returns_non_empty():
    comp = IdentityComponent()
    # Reset cache to avoid cross-test pollution
    comp._cached = None
    result = comp.render(make_state(), make_config())
    assert result is not None
    assert len(result) > 0


def test_identity_fallback_when_soul_missing(monkeypatch):
    import src.agent.prompt as prompt_mod
    from pathlib import Path

    comp = IdentityComponent()
    comp._cached = None

    # Patch IdentityComponent.render's candidates by monkeypatching Path.exists
    # to return False for any SOUL.md path, and Path.cwd to a non-existent location.
    original_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self.name == "SOUL.md":
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    comp._cached = None
    result = comp.render(make_state(), make_config())
    assert result == "You are a capable coding agent."


def test_identity_prefers_supplied_base_prompt():
    comp = IdentityComponent(lambda: "Custom system prompt")
    result = comp.render(make_state(), make_config())
    assert result == "Custom system prompt"


# ---------------------------------------------------------------------------
# CoreToolSchemasComponent
# ---------------------------------------------------------------------------

def test_core_tools_returns_none_when_empty():
    comp = CoreToolSchemasComponent(lambda: "")
    assert comp.render(make_state(), make_config()) is None


def test_core_tools_returns_wrapped_schema():
    comp = CoreToolSchemasComponent(lambda: "tool_a: does stuff")
    result = comp.render(make_state(), make_config())
    assert result is not None
    assert result.startswith("## Available Tools")
    assert "tool_a: does stuff" in result


def test_core_tools_returns_none_for_whitespace_only():
    comp = CoreToolSchemasComponent(lambda: "   \n  ")
    assert comp.render(make_state(), make_config()) is None


# ---------------------------------------------------------------------------
# DomainToolsComponent
# ---------------------------------------------------------------------------

def test_domain_tools_returns_none_when_no_groups():
    comp = DomainToolsComponent(lambda groups: "domain schema")
    state = make_state()
    assert comp.render(state, make_config()) is None


def test_domain_tools_returns_content_when_groups_present():
    comp = DomainToolsComponent(lambda groups: f"tools for {sorted(groups)}")
    state = make_state()
    state.domain_groups_activated = {"timeseries"}
    result = comp.render(state, make_config())
    assert result is not None
    assert result.startswith("## Domain Tools")
    assert "timeseries" in result


def test_domain_tools_returns_none_when_fn_returns_empty():
    comp = DomainToolsComponent(lambda groups: "")
    state = make_state()
    state.domain_groups_activated = {"timeseries"}
    assert comp.render(state, make_config()) is None


# ---------------------------------------------------------------------------
# SkillPlaybookComponent
# ---------------------------------------------------------------------------

def test_skill_playbook_returns_none_when_routing_disabled():
    comp = SkillPlaybookComponent(lambda q: "some playbook")
    config = make_config(enable_skill_routing=False)
    assert comp.render(make_state(), config) is None


def test_skill_playbook_returns_content_when_routing_enabled():
    comp = SkillPlaybookComponent(lambda q: "use edit_block skill")
    config = make_config(enable_skill_routing=True)
    result = comp.render(make_state(), config)
    assert result is not None
    assert result.startswith("## Active Skill")
    assert "edit_block" in result


def test_skill_playbook_returns_none_when_fn_returns_empty():
    comp = SkillPlaybookComponent(lambda q: "")
    config = make_config(enable_skill_routing=True)
    assert comp.render(make_state(), config) is None


# ---------------------------------------------------------------------------
# RetrievalContextComponent
# ---------------------------------------------------------------------------

def test_retrieval_context_returns_none_when_disabled():
    comp = RetrievalContextComponent(lambda state: "retrieved context")
    config = make_config(prompt_rag_context_enabled=False)
    assert comp.render(make_state(user_query="find foo"), config) is None


def test_retrieval_context_wraps_summary():
    comp = RetrievalContextComponent(lambda state: "repo:file.py - context")
    result = comp.render(make_state(user_query="find foo"), make_config())
    assert result is not None
    assert result.startswith("## Retrieval Context")
    assert "file.py" in result


def test_retrieval_context_caches_per_turn():
    calls = {"count": 0}

    def _render(state):
        calls["count"] += 1
        return f"context for {state.user_query}"

    comp = RetrievalContextComponent(_render)
    state = make_state(user_query="find foo")
    config = make_config()
    first = comp.render(state, config)
    second = comp.render(state, config)
    assert first == second
    assert calls["count"] == 1


# ---------------------------------------------------------------------------
# RuntimeContextComponent
# ---------------------------------------------------------------------------

def test_runtime_context_returns_none_when_empty():
    comp = RuntimeContextComponent(lambda: "")
    assert comp.render(make_state(), make_config()) is None


def test_runtime_context_wraps_summary():
    comp = RuntimeContextComponent(lambda: "Mounted paths available:\n- /repo")
    result = comp.render(make_state(), make_config())
    assert result is not None
    assert result.startswith("## Runtime Context")
    assert "/repo" in result


# ---------------------------------------------------------------------------
# TurnContextComponent
# ---------------------------------------------------------------------------

def test_turn_context_returns_none_at_iteration_zero_no_files():
    comp = TurnContextComponent()
    state = make_state()
    assert state.iteration == 0
    assert state.files_written == []
    assert comp.render(state, make_config()) is None


def test_turn_context_returns_content_when_files_written():
    comp = TurnContextComponent()
    state = make_state()
    state.record_write("/tmp/foo.py")
    result = comp.render(state, make_config())
    assert result is not None
    assert "foo.py" in result
    assert "## Turn Context" in result


def test_turn_context_returns_none_when_no_files_regardless_of_iteration():
    comp = TurnContextComponent()
    state = make_state()
    state.iteration = 3
    result = comp.render(state, make_config())
    assert result is None


def test_turn_context_includes_verify_reminder_when_files_written():
    comp = TurnContextComponent()
    state = make_state()
    state.iteration = 1
    state.record_write("/src/agent/foo.py")
    result = comp.render(state, make_config())
    assert result is not None
    assert "verify" in result.lower() or "linter" in result.lower()


# ---------------------------------------------------------------------------
# default_prompt_builder factory
# ---------------------------------------------------------------------------

def test_default_prompt_builder_returns_prompt_builder():
    builder = default_prompt_builder(
        base_prompt_fn=lambda: "base",
        tool_schema_fn=lambda: "",
        routing_fn=lambda q: "",
    )
    assert isinstance(builder, PromptBuilder)
    assert len(builder.components) == 5


def test_default_prompt_builder_includes_retrieval_component_when_supplied():
    builder = default_prompt_builder(
        base_prompt_fn=lambda: "base",
        tool_schema_fn=lambda: "",
        routing_fn=lambda q: "",
        retrieval_context_fn=lambda state: "retrieved context",
        runtime_context_fn=lambda: "runtime context",
    )
    assert isinstance(builder, PromptBuilder)
    assert len(builder.components) == 7
