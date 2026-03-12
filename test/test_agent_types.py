"""Tests for src/agent/types.py — TurnResult type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.agent.types import TurnResult
from src.messages import Message, MessageRole
from src.tools.runtime import ToolCall


@dataclass
class _StubState:
    """Minimal stub for TurnState — real implementation in Task 3."""

    final_response: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class TestTurnResult:
    """Test TurnResult type."""

    def test_turn_result_final_response(self) -> None:
        """TurnResult.final_response returns state.final_response."""
        state = _StubState(final_response="Hello, world!")
        messages: list[Message] = []
        result = TurnResult(state, messages)

        assert result.final_response == "Hello, world!"

    def test_turn_result_final_response_none(self) -> None:
        """TurnResult.final_response returns None when state has no response."""
        state = _StubState(final_response=None)
        messages: list[Message] = []
        result = TurnResult(state, messages)

        assert result.final_response is None

    def test_turn_result_tool_calls(self) -> None:
        """TurnResult.tool_calls returns state.tool_calls."""
        tool_call = ToolCall(id="1", name="test_tool", arguments={"key": "value"})
        state = _StubState(tool_calls=[tool_call])
        messages: list[Message] = []
        result = TurnResult(state, messages)

        assert result.tool_calls == [tool_call]
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "1"
        assert result.tool_calls[0].name == "test_tool"

    def test_turn_result_tool_calls_empty(self) -> None:
        """TurnResult.tool_calls returns empty list when state has no calls."""
        state = _StubState(tool_calls=[])
        messages: list[Message] = []
        result = TurnResult(state, messages)

        assert result.tool_calls == []

    def test_turn_result_state(self) -> None:
        """TurnResult.state returns the passed state."""
        state = _StubState(final_response="test")
        messages: list[Message] = []
        result = TurnResult(state, messages)

        assert result.state is state

    def test_turn_result_messages(self) -> None:
        """TurnResult.messages is the passed messages list."""
        state = _StubState()
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi"),
        ]
        result = TurnResult(state, messages)

        assert result.messages == messages
        assert len(result.messages) == 2
        assert result.messages[0].content == "Hello"
        assert result.messages[1].content == "Hi"

    def test_turn_result_messages_empty(self) -> None:
        """TurnResult.messages is empty when passed empty list."""
        state = _StubState()
        messages: list[Message] = []
        result = TurnResult(state, messages)

        assert result.messages == []

    def test_turn_result_multiple_tool_calls(self) -> None:
        """TurnResult.tool_calls handles multiple tool calls."""
        tool_calls = [
            ToolCall(id="1", name="tool_a", arguments={"x": 1}),
            ToolCall(id="2", name="tool_b", arguments={"y": 2}),
            ToolCall(id="3", name="tool_c", arguments={"z": 3}),
        ]
        state = _StubState(tool_calls=tool_calls)
        messages: list[Message] = []
        result = TurnResult(state, messages)

        assert result.tool_calls == tool_calls
        assert len(result.tool_calls) == 3

    def test_turn_result_combined_state_and_messages(self) -> None:
        """TurnResult stores both state and messages together."""
        tool_call = ToolCall(id="42", name="analyze", arguments={"data": [1, 2, 3]})
        state = _StubState(
            final_response="Analysis complete",
            tool_calls=[tool_call],
        )
        messages = [
            Message(role=MessageRole.USER, content="Analyze this data"),
            Message(role=MessageRole.TOOL, content="Data analyzed", tool_call_id="42"),
        ]
        result = TurnResult(state, messages)

        assert result.final_response == "Analysis complete"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "analyze"
        assert len(result.messages) == 2
        assert result.state is state
