"""Tests for ToolDispatcher."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.agent.dispatcher import DispatchResult, ToolDispatcher, _READ_ONLY_TOOLS
from src.agent.state import TurnState
from src.tools.runtime import ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeRegistry:
    """Minimal ToolRegistry substitute that returns a fixed string."""

    def __init__(self, return_value: str = "ok", raise_on: set[str] | None = None) -> None:
        self._return_value = return_value
        self._raise_on: set[str] = raise_on or set()
        self.calls: list[ToolCall] = []

    def execute(self, call: ToolCall, use_toon: bool = True) -> str:
        self.calls.append(call)
        if call.name in self._raise_on:
            raise RuntimeError(f"simulated failure for {call.name}")
        return self._return_value


def make_state(turn_id: str = "test-turn") -> TurnState:
    return TurnState(turn_id=turn_id)


def make_call(
    name: str,
    arguments: dict[str, Any] | None = None,
    call_id: str | None = None,
) -> ToolCall:
    return ToolCall(
        id=call_id or f"call-{name}",
        name=name,
        arguments=arguments or {},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_read_only_tool_returns_dispatch_result():
    reg = FakeRegistry(return_value="file contents")
    dispatcher = ToolDispatcher(reg)
    call = make_call("read_file", {"path": "/tmp/foo.py"})
    state = make_state()

    results = asyncio.run(dispatcher.dispatch([call], state))

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, DispatchResult)
    assert r.tool_name == "read_file"
    assert r.output == "file contents"
    assert r.error is None


def test_write_tool_updates_state_files_written():
    reg = FakeRegistry(return_value="written")
    dispatcher = ToolDispatcher(reg)
    call = make_call("write_file", {"path": "/src/agent/foo.py"})
    state = make_state()

    asyncio.run(dispatcher.dispatch([call], state))

    assert "/src/agent/foo.py" in state.files_written


def test_write_tool_with_file_path_key_updates_state():
    reg = FakeRegistry(return_value="written")
    dispatcher = ToolDispatcher(reg)
    call = make_call("edit_file", {"file_path": "/src/agent/bar.py"})
    state = make_state()

    asyncio.run(dispatcher.dispatch([call], state))

    assert "/src/agent/bar.py" in state.files_written


def test_state_tool_calls_updated_after_dispatch():
    reg = FakeRegistry()
    dispatcher = ToolDispatcher(reg)
    call = make_call("read_file", {"path": "/tmp/x.py"})
    state = make_state()

    asyncio.run(dispatcher.dispatch([call], state))

    assert len(state.tool_calls) == 1
    assert state.tool_calls[0].name == "read_file"


def test_consecutive_tool_count_incremented():
    reg = FakeRegistry()
    dispatcher = ToolDispatcher(reg)
    calls = [
        make_call("read_file", {"path": "/a.py"}, call_id="c1"),
        make_call("read_file", {"path": "/b.py"}, call_id="c2"),
    ]
    state = make_state()
    assert state.consecutive_tool_count == 0

    asyncio.run(dispatcher.dispatch(calls, state))

    assert state.consecutive_tool_count == 2


def test_failed_tool_returns_error_in_result_does_not_raise():
    reg = FakeRegistry(raise_on={"bad_tool"})
    dispatcher = ToolDispatcher(reg)
    call = make_call("bad_tool", {})
    state = make_state()

    results = asyncio.run(dispatcher.dispatch([call], state))

    assert len(results) == 1
    r = results[0]
    assert r.error is not None
    assert "simulated failure" in r.error
    assert r.output == ""


def test_multiple_reads_execute_and_return_all_results():
    reg = FakeRegistry(return_value="data")
    dispatcher = ToolDispatcher(reg)
    calls = [
        make_call("read_file", {"path": "/a.py"}, call_id="r1"),
        make_call("grep", {"pattern": "foo"}, call_id="r2"),
        make_call("glob", {"pattern": "*.py"}, call_id="r3"),
    ]
    state = make_state()

    results = asyncio.run(dispatcher.dispatch(calls, state))

    assert len(results) == 3
    names = {r.tool_name for r in results}
    assert names == {"read_file", "grep", "glob"}
    assert all(r.error is None for r in results)


def test_read_only_tools_frozenset_contents():
    assert "read_file" in _READ_ONLY_TOOLS
    assert "glob" in _READ_ONLY_TOOLS
    assert "grep" in _READ_ONLY_TOOLS
    assert "think" in _READ_ONLY_TOOLS
    assert "write_file" not in _READ_ONLY_TOOLS


def test_dispatch_result_duration_ms_non_negative():
    reg = FakeRegistry()
    dispatcher = ToolDispatcher(reg)
    call = make_call("think", {})
    state = make_state()

    results = asyncio.run(dispatcher.dispatch([call], state))

    assert results[0].duration_ms >= 0


def test_mixed_read_and_write_calls():
    reg = FakeRegistry(return_value="done")
    dispatcher = ToolDispatcher(reg)
    calls = [
        make_call("read_file", {"path": "/r.py"}, call_id="r1"),
        make_call("write_file", {"path": "/w.py"}, call_id="w1"),
    ]
    state = make_state()

    results = asyncio.run(dispatcher.dispatch(calls, state))

    assert len(results) == 2
    assert state.consecutive_tool_count == 2
    assert "/w.py" in state.files_written
    assert "/r.py" not in state.files_written
