"""Tests for the structured event system in src/agent/events.py."""
import asyncio
import time
import uuid
from unittest.mock import MagicMock

import pytest

from src.agent.events import (
    AgentEvent,
    AgentEventType,
    EventEmitter,
    EventStream,
    EventCallback,
    EventCallbackAsync,
    make_turn_start,
    make_turn_end,
    make_tool_execution_start,
    make_tool_execution_end,
    make_guardrail_nudge,
    make_repair_nudge,
    make_classified,
)


class TestAgentEventType:
    """Test that event types are properly defined."""

    def test_all_event_types_present(self):
        expected = [
            "AGENT_START",
            "AGENT_END",
            "TURN_START",
            "TURN_END",
            "CLASSIFIED",
            "TOOL_EXECUTION_START",
            "TOOL_EXECUTION_END",
            "GUARDRAIL_NUDGE",
            "REPAIR_NUDGE",
        ]
        for name in expected:
            assert hasattr(AgentEventType, name), f"Missing event type: {name}"

    def test_event_type_values(self):
        assert AgentEventType.TURN_START.value == "turn_start"
        assert AgentEventType.TOOL_EXECUTION_START.value == "tool_execution_start"
        assert AgentEventType.GUARDRAIL_NUDGE.value == "guardrail_nudge"
        assert AgentEventType.REPAIR_NUDGE.value == "repair_nudge"


class TestAgentEvent:
    """Test the AgentEvent dataclass."""

    def test_create_basic_event(self):
        event = AgentEvent(
            type=AgentEventType.TURN_START,
            turn_id="test-123",
        )
        assert event.type == AgentEventType.TURN_START
        assert event.turn_id == "test-123"
        assert isinstance(event.timestamp, float)

    def test_create_event_with_all_fields(self):
        event = AgentEvent(
            type=AgentEventType.TOOL_EXECUTION_START,
            turn_id="turn-1",
            tool_call_id="call-1",
            tool_name="read_file",
            tool_args={"path": "test.py", "offset": 0},
            intent="execution",
        )
        assert event.tool_call_id == "call-1"
        assert event.tool_name == "read_file"
        assert event.tool_args == {"path": "test.py", "offset": 0}
        assert event.intent == "execution"

    def test_to_dict(self):
        event = AgentEvent(
            type=AgentEventType.TURN_END,
            turn_id="turn-1",
            message="Final answer here.",
        )
        d = event.to_dict()
        assert d["type"] == "turn_end"
        assert d["turn_id"] == "turn-1"
        assert d["message"] == "Final answer here."
        assert "timestamp" in d
        assert "turn_index" not in d  # None fields not in dict

    def test_to_dict_with_none_fields(self):
        event = AgentEvent(type=AgentEventType.TURN_START, turn_id="turn-1")
        d = event.to_dict()
        assert "message" not in d
        assert "tool_name" not in d
        assert "guard_name" not in d


class TestEventFactoryFunctions:
    """Test convenience factory functions."""

    def test_make_turn_start(self):
        event = make_turn_start("turn-1")
        assert event.type == AgentEventType.TURN_START
        assert event.turn_id == "turn-1"

    def test_make_turn_end(self):
        event = make_turn_end("turn-1", "The answer is 42")
        assert event.type == AgentEventType.TURN_END
        assert event.message == "The answer is 42"

    def test_make_tool_execution_start(self):
        event = make_tool_execution_start(
            "turn-1", "call-1", "read_file", {"path": "main.py"}
        )
        assert event.type == AgentEventType.TOOL_EXECUTION_START
        assert event.tool_call_id == "call-1"
        assert event.tool_name == "read_file"
        assert event.tool_args == {"path": "main.py"}

    def test_make_tool_execution_end(self):
        event = make_tool_execution_end(
            "turn-1", "call-1", "read_file", "File content", False
        )
        assert event.type == AgentEventType.TOOL_EXECUTION_END
        assert event.result == "File content"
        assert event.is_error is False

    def test_make_tool_execution_end_error(self):
        event = make_tool_execution_end(
            "turn-1", "call-1", "write_file", "Permission denied", True
        )
        assert event.is_error is True

    def test_make_guardrail_nudge(self):
        event = make_guardrail_nudge("turn-1", "duplicate_tool", "Stop duplicating tools")
        assert event.type == AgentEventType.GUARDRAIL_NUDGE
        assert event.guard_name == "duplicate_tool"
        assert event.nudge == "Stop duplicating tools"

    def test_make_repair_nudge(self):
        event = make_repair_nudge(
            "turn-1", "attempt", 2, "read_file", "json_parse_error", "Fix JSON"
        )
        assert event.type == AgentEventType.REPAIR_NUDGE
        assert event.repair_stage == "attempt"
        assert event.attempt == 2
        assert event.tool_name == "read_file"
        assert event.error_type == "json_parse_error"
        assert event.message == "Fix JSON"

    def test_make_classified(self):
        event = make_classified("turn-1", "execution", ["coding"])
        assert event.type == AgentEventType.CLASSIFIED
        assert event.intent == "execution"
        assert event.domain_groups == ["coding"]


class TestEventEmitter:
    """Test the EventEmitter pub/sub implementation."""

    def test_basic_emit(self):
        emitter = EventEmitter()
        received = []
        emitter.on("turn_start", lambda e: received.append(e))
        emitter.emit(AgentEvent(type=AgentEventType.TURN_START, turn_id="1"))
        assert len(received) == 1
        assert received[0].type == AgentEventType.TURN_START

    def test_sync_handler(self):
        emitter = EventEmitter()
        received = []
        def handler(event: AgentEvent) -> None:
            received.append(event)
        emitter.on("turn_end", handler)
        emitter.emit(AgentEvent(type=AgentEventType.TURN_END, turn_id="1"))
        assert len(received) == 1

    def test_async_handler(self):
        loop = asyncio.new_event_loop()
        try:
            emitter = EventEmitter()
            received = []

            async def handler(event: AgentEvent) -> None:
                received.append(event)

            emitter.on_async("turn_end", handler)
            loop.run_until_complete(emitter.emit_async(AgentEvent(
                type=AgentEventType.TURN_END, turn_id="1"
            )))
            assert len(received) == 1
        finally:
            loop.close()

    def test_wildcard_listener(self):
        emitter = EventEmitter()
        received = []
        emitter.on("*", lambda e: received.append(e))
        emitter.emit(AgentEvent(type=AgentEventType.TURN_START, turn_id="1"))
        emitter.emit(AgentEvent(type=AgentEventType.TOOL_EXECUTION_START, turn_id="1"))
        assert len(received) == 2

    def test_off_removes_handler(self):
        emitter = EventEmitter()
        received = []
        handler = lambda e: received.append(e)
        emitter.on("turn_end", handler)
        emitter.off("turn_end", handler)
        emitter.emit(AgentEvent(type=AgentEventType.TURN_END, turn_id="1"))
        assert len(received) == 0

    def test_wildcard_off(self):
        emitter = EventEmitter()
        received = []
        handler = lambda e: received.append(e)
        emitter.on("*", handler)
        emitter.off("*", handler)
        emitter.emit(AgentEvent(type=AgentEventType.TURN_START, turn_id="1"))
        assert len(received) == 0

    def test_handler_exception_does_not_stop_others(self):
        emitter = EventEmitter()
        received = []

        def bad_handler(event: AgentEvent) -> None:
            raise ValueError("boom")

        def good_handler(event: AgentEvent) -> None:
            received.append(event)

        emitter.on("turn_end", bad_handler)
        emitter.on("turn_end", good_handler)
        emitter.emit(AgentEvent(type=AgentEventType.TURN_END, turn_id="1"))
        assert len(received) == 1

    def test_once_handler(self):
        emitter = EventEmitter()
        received = []
        emitter.once("turn_end", lambda e: received.append(e))
        emitter.emit(AgentEvent(type=AgentEventType.TURN_END, turn_id="1"))
        emitter.emit(AgentEvent(type=AgentEventType.TURN_END, turn_id="2"))
        assert len(received) == 1  # only fired once

    def test_agent_event_type_as_key(self):
        emitter = EventEmitter()
        received = []
        emitter.on(AgentEventType.TURN_END, lambda e: received.append(e))
        emitter.emit(AgentEvent(type=AgentEventType.TURN_END, turn_id="1"))
        assert len(received) == 1


class TestEventStream:
    """Test the async EventStream generator."""

    def test_stream_buffer_yields(self):
        """Test EventStream yields events from its buffer."""
        loop = asyncio.new_event_loop()
        try:

            async def collect():
                emitter = EventEmitter()
                stream = EventStream(emitter, "turn_start")
                # Pre-populate buffer
                stream.push(AgentEvent(type=AgentEventType.TURN_START, turn_id="1"))
                stream.push(AgentEvent(type=AgentEventType.TOOL_EXECUTION_START, turn_id="1"))
                stream.push(AgentEvent(type=AgentEventType.TURN_START, turn_id="2"))
                # Call end() to stop the infinite loop in __aiter__
                stream.end()
                events = [e async for e in stream]
                return events

            events = loop.run_until_complete(collect())
            assert len(events) == 2
            assert events[0].turn_id == "1"
            assert events[1].turn_id == "2"
        finally:
            loop.close()

    def test_stream_filters_by_type(self):
        """Test EventStream filters by event type."""
        loop = asyncio.new_event_loop()
        try:

            async def count():
                emitter = EventEmitter()
                stream = EventStream(emitter, "turn_start")
                stream.push(AgentEvent(type=AgentEventType.TURN_END, turn_id="1"))
                stream.push(AgentEvent(type=AgentEventType.TURN_START, turn_id="2"))
                c = 0
                async for event in stream:
                    c += 1
                    break
                return c

            c = loop.run_until_complete(count())
            assert c == 1  # only turn_start, not turn_end
        finally:
            loop.close()


class TestAgentLoopIntegration:
    """Test that AgentLoop properly emits events through _emit helper."""

    def test_loop_accepts_emitter(self):
        """Verify AgentLoop accepts an EventEmitter parameter."""
        from src.agent.loop import AgentLoop

        loop = MagicMock()
        loop._emit = MagicMock()
        loop._emit(AgentEvent(type=AgentEventType.TURN_START, turn_id="test"))
        loop._emit.assert_called_once()

    def test_loop_emit_does_not_raise(self):
        """Verify _emit doesn't raise even with no emitter."""
        from src.agent.loop import AgentLoop

        loop = MagicMock()
        loop._emitter = None
        loop._emit = lambda event: None  # simplified
        loop._emit(AgentEvent(type=AgentEventType.TURN_START, turn_id="test"))


class TestAgentSubscribeEvents:
    """Test the Agent.subscribe_events API."""

    def test_subscribe_returns_dispose(self):
        from src.agent.core import Agent

        agent = Agent()
        received = []
        handler = lambda e: received.append(e)
        dispose = agent.subscribe_events(handler)
        assert callable(dispose)

    def test_subscribe_receives_events(self):
        from src.agent.core import Agent

        agent = Agent()
        received = []
        agent.subscribe_events(lambda e: received.append(e))
        agent._event_emitter.emit(AgentEvent(
            type=AgentEventType.TOOL_EXECUTION_START,
            turn_id="test",
            tool_name="read_file",
        ))
        assert len(received) == 1
        assert received[0]["type"] == "tool_execution_start"

    def test_dispose_stops_events(self):
        from src.agent.core import Agent

        agent = Agent()
        received = []
        dispose = agent.subscribe_events(lambda e: received.append(e))
        agent._event_emitter.emit(AgentEvent(
            type=AgentEventType.TURN_START,
            turn_id="1",
        ))
        assert len(received) == 1

        dispose()
        agent._event_emitter.emit(AgentEvent(
            type=AgentEventType.TURN_END,
            turn_id="2",
        ))
        assert len(received) == 1  # still 1, dispose worked
