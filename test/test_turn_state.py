"""Tests for src/agent/state.py — TurnState class."""

from __future__ import annotations

import pytest

from src.agent.state import TurnState
from src.tools.runtime import ToolCall


class TestTurnState:
    """Test TurnState dataclass."""

    def test_turn_state_initialization(self) -> None:
        """TurnState initializes with turn_id."""
        state = TurnState(turn_id="turn_123")
        assert state.turn_id == "turn_123"
        assert state.iteration == 0
        assert state.consecutive_tool_count == 0
        assert state.tool_calls == []
        assert state.seen_signatures == {}
        assert state.files_written == []
        assert state.domain_groups_activated == set()
        assert state.guardrail_nudges == {}
        assert state.classified_as == "execution"
        assert state.final_response is None
        assert state.trace == []

    def test_tool_signature_basic(self) -> None:
        """tool_signature produces consistent hash for same arguments."""
        state = TurnState(turn_id="turn_1")
        call = ToolCall(id="call_1", name="test_tool", arguments={"key": "value"})
        sig1 = state.tool_signature(call)
        sig2 = state.tool_signature(call)
        assert sig1 == sig2

    def test_tool_signature_different_args(self) -> None:
        """tool_signature produces different hash for different arguments."""
        state = TurnState(turn_id="turn_1")
        call1 = ToolCall(id="call_1", name="test_tool", arguments={"key": "value1"})
        call2 = ToolCall(id="call_2", name="test_tool", arguments={"key": "value2"})
        sig1 = state.tool_signature(call1)
        sig2 = state.tool_signature(call2)
        assert sig1 != sig2

    def test_tool_signature_different_tool_names(self) -> None:
        """tool_signature includes tool name in hash."""
        state = TurnState(turn_id="turn_1")
        call1 = ToolCall(id="call_1", name="tool_a", arguments={"key": "value"})
        call2 = ToolCall(id="call_2", name="tool_b", arguments={"key": "value"})
        sig1 = state.tool_signature(call1)
        sig2 = state.tool_signature(call2)
        assert sig1 != sig2

    def test_tool_signature_format(self) -> None:
        """tool_signature returns string in format 'tool_name:hash'."""
        state = TurnState(turn_id="turn_1")
        call = ToolCall(id="call_1", name="my_tool", arguments={})
        sig = state.tool_signature(call)
        assert ":" in sig
        parts = sig.split(":")
        assert len(parts) == 2
        assert parts[0] == "my_tool"
        assert len(parts[1]) == 16  # SHA256 truncated to 16 chars

    def test_tool_signature_stable_across_calls(self) -> None:
        """tool_signature is stable across multiple TurnState instances."""
        state1 = TurnState(turn_id="turn_1")
        state2 = TurnState(turn_id="turn_2")
        call = ToolCall(id="call_1", name="test_tool", arguments={"x": 42})
        sig1 = state1.tool_signature(call)
        sig2 = state2.tool_signature(call)
        assert sig1 == sig2

    def test_record_call_appends_to_tool_calls(self) -> None:
        """record_call appends to tool_calls list."""
        state = TurnState(turn_id="turn_1")
        call1 = ToolCall(id="1", name="tool_a", arguments={"x": 1})
        call2 = ToolCall(id="2", name="tool_b", arguments={"y": 2})

        state.record_call(call1)
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0] == call1

        state.record_call(call2)
        assert len(state.tool_calls) == 2
        assert state.tool_calls[1] == call2

    def test_record_call_updates_seen_signatures(self) -> None:
        """record_call updates seen_signatures counter."""
        state = TurnState(turn_id="turn_1")
        call = ToolCall(id="1", name="test_tool", arguments={"key": "value"})

        state.record_call(call)
        sig = state.tool_signature(call)
        assert sig in state.seen_signatures
        assert state.seen_signatures[sig] == 1

    def test_record_call_counts_duplicates(self) -> None:
        """record_call increments count for duplicate signatures."""
        state = TurnState(turn_id="turn_1")
        call = ToolCall(id="1", name="test_tool", arguments={"key": "value"})

        state.record_call(call)
        state.record_call(call)
        state.record_call(call)

        sig = state.tool_signature(call)
        assert state.seen_signatures[sig] == 3

    def test_record_call_different_signatures(self) -> None:
        """record_call tracks different signatures separately."""
        state = TurnState(turn_id="turn_1")
        call1 = ToolCall(id="1", name="tool_a", arguments={"x": 1})
        call2 = ToolCall(id="2", name="tool_a", arguments={"x": 2})

        state.record_call(call1)
        state.record_call(call2)

        sig1 = state.tool_signature(call1)
        sig2 = state.tool_signature(call2)
        assert state.seen_signatures[sig1] == 1
        assert state.seen_signatures[sig2] == 1

    def test_record_write_appends_path(self) -> None:
        """record_write appends path to files_written."""
        state = TurnState(turn_id="turn_1")
        state.record_write("/path/to/file1.py")
        assert state.files_written == ["/path/to/file1.py"]

        state.record_write("/path/to/file2.py")
        assert state.files_written == ["/path/to/file1.py", "/path/to/file2.py"]

    def test_record_write_no_duplicates(self) -> None:
        """record_write skips duplicate paths."""
        state = TurnState(turn_id="turn_1")
        state.record_write("/path/to/file.py")
        state.record_write("/path/to/file.py")
        state.record_write("/path/to/file.py")
        assert state.files_written == ["/path/to/file.py"]

    def test_record_write_preserves_order(self) -> None:
        """record_write preserves order of first occurrence."""
        state = TurnState(turn_id="turn_1")
        state.record_write("/a.py")
        state.record_write("/b.py")
        state.record_write("/c.py")
        state.record_write("/b.py")  # duplicate, should not appear again
        assert state.files_written == ["/a.py", "/b.py", "/c.py"]

    def test_last_write_index_returns_minus_one_when_no_writes(self) -> None:
        """last_write_index returns -1 when no write tool calls exist."""
        state = TurnState(turn_id="turn_1")
        call1 = ToolCall(id="1", name="read_file", arguments={})
        call2 = ToolCall(id="2", name="search_code", arguments={})

        state.record_call(call1)
        state.record_call(call2)

        assert state.last_write_index() == -1

    def test_last_write_index_finds_write_file(self) -> None:
        """last_write_index finds write_file calls."""
        state = TurnState(turn_id="turn_1")
        call1 = ToolCall(id="1", name="read_file", arguments={})
        call2 = ToolCall(id="2", name="write_file", arguments={"path": "/file.py"})
        call3 = ToolCall(id="3", name="read_file", arguments={})

        state.record_call(call1)
        state.record_call(call2)
        state.record_call(call3)

        assert state.last_write_index() == 1

    def test_last_write_index_finds_edit_file(self) -> None:
        """last_write_index finds edit_file calls."""
        state = TurnState(turn_id="turn_1")
        call1 = ToolCall(id="1", name="read_file", arguments={})
        call2 = ToolCall(id="2", name="edit_file", arguments={"path": "/file.py"})

        state.record_call(call1)
        state.record_call(call2)

        assert state.last_write_index() == 1

    def test_last_write_index_finds_apply_edit_block(self) -> None:
        """last_write_index finds apply_edit_block calls."""
        state = TurnState(turn_id="turn_1")
        call1 = ToolCall(id="1", name="read_file", arguments={})
        call2 = ToolCall(id="2", name="apply_edit_block", arguments={})

        state.record_call(call1)
        state.record_call(call2)

        assert state.last_write_index() == 1

    def test_last_write_index_most_recent(self) -> None:
        """last_write_index returns most recent write, not first."""
        state = TurnState(turn_id="turn_1")
        call1 = ToolCall(id="1", name="write_file", arguments={})
        call2 = ToolCall(id="2", name="read_file", arguments={})
        call3 = ToolCall(id="3", name="write_file", arguments={})
        call4 = ToolCall(id="4", name="read_file", arguments={})

        state.record_call(call1)
        state.record_call(call2)
        state.record_call(call3)
        state.record_call(call4)

        assert state.last_write_index() == 2

    def test_last_write_index_with_multiple_write_types(self) -> None:
        """last_write_index finds most recent among all write types."""
        state = TurnState(turn_id="turn_1")
        call1 = ToolCall(id="1", name="write_file", arguments={})
        call2 = ToolCall(id="2", name="read_file", arguments={})
        call3 = ToolCall(id="3", name="edit_file", arguments={})
        call4 = ToolCall(id="4", name="apply_edit_block", arguments={})
        call5 = ToolCall(id="5", name="read_file", arguments={})

        state.record_call(call1)
        state.record_call(call2)
        state.record_call(call3)
        state.record_call(call4)
        state.record_call(call5)

        assert state.last_write_index() == 3

    def test_turn_state_custom_fields(self) -> None:
        """TurnState supports custom field assignments."""
        state = TurnState(turn_id="turn_1")
        state.iteration = 5
        state.consecutive_tool_count = 3
        state.classified_as = "design"
        state.final_response = "Here's the solution"

        assert state.iteration == 5
        assert state.consecutive_tool_count == 3
        assert state.classified_as == "design"
        assert state.final_response == "Here's the solution"

    def test_turn_state_domain_groups_activated(self) -> None:
        """TurnState tracks activated domain groups."""
        state = TurnState(turn_id="turn_1")
        state.domain_groups_activated.add("timeseries")
        state.domain_groups_activated.add("academic")

        assert "timeseries" in state.domain_groups_activated
        assert "academic" in state.domain_groups_activated

    def test_turn_state_guardrail_nudges(self) -> None:
        """TurnState tracks guardrail nudges."""
        state = TurnState(turn_id="turn_1")
        state.guardrail_nudges["duplicate_tool"] = 2
        state.guardrail_nudges["max_iterations"] = 1

        assert state.guardrail_nudges["duplicate_tool"] == 2
        assert state.guardrail_nudges["max_iterations"] == 1

    def test_turn_state_trace(self) -> None:
        """TurnState maintains a trace of execution."""
        state = TurnState(turn_id="turn_1")
        state.trace.append({"step": 1, "action": "init"})
        state.trace.append({"step": 2, "action": "call_tool"})

        assert len(state.trace) == 2
        assert state.trace[0]["action"] == "init"
        assert state.trace[1]["action"] == "call_tool"

    def test_record_call_with_complex_arguments(self) -> None:
        """record_call works with complex nested arguments."""
        state = TurnState(turn_id="turn_1")
        call = ToolCall(
            id="1",
            name="complex_tool",
            arguments={
                "nested": {"key": "value", "list": [1, 2, 3]},
                "string": "test",
                "number": 42,
            },
        )

        state.record_call(call)
        sig = state.tool_signature(call)

        assert len(state.tool_calls) == 1
        assert sig in state.seen_signatures
        assert state.seen_signatures[sig] == 1
