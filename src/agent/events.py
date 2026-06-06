"""Structured event system for the agent loop.

Inspired by Pi's AgentEvent architecture. Events are emitted at key lifecycle
points (agent start/turn start/turn end/tool execution/message streaming) so
that the bridge/TUI can react to granular state changes without tight coupling.

Usage:
    emitter = EventEmitter()
    emitter.on("tool_execution_start", lambda e: print(f"Tool: {e.tool_name}"))
    await loop.run(emitter=emitter, ...)  # loop calls emitter.emit()
    # Or: for subscription, use emitter.subscribe(handler) which returns a dispose fn.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class AgentEventType(str, Enum):
    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    # Turn lifecycle
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    # Message lifecycle
    MESSAGE_START = "message_start"
    MESSAGE_UPDATE = "message_update"
    MESSAGE_END = "message_end"
    # Tool execution lifecycle
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_UPDATE = "tool_execution_update"
    TOOL_EXECUTION_END = "tool_execution_end"
    # Guardrail events
    GUARDRAIL_NUDGE = "guardrail_nudge"
    GUARDRAIL_HARD_STOP = "guardrail_hard_stop"
    # Repair events
    REPAIR_NUDGE = "repair_nudge"
    # Turn classification
    CLASSIFIED = "classified"


@dataclass
class AgentEvent:
    """Single structured event emitted by the agent loop."""
    type: AgentEventType
    turn_id: str | None = None
    timestamp: float = field(default_factory=lambda: __import__("time").time())

    # Message fields
    message: str | None = None
    role: str | None = None  # "user", "assistant", "tool"

    # Tool execution fields
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    result: str | None = None
    is_error: bool | None = None

    # Guardrail/repair
    guard_name: str | None = None
    nudge: str | None = None
    hard_stop: bool | None = None
    repair_stage: str | None = None  # "nudge", "invalid"
    attempt: int | None = None
    error_type: str | None = None

    # Classification
    intent: str | None = None
    domain_groups: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        d = {}
        for k, v in asdict(self).items():
            if v is not None and v != []:
                d[k] = v
        d["type"] = self.type.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ---------------------------------------------------------------------------
# Event emitter
# ---------------------------------------------------------------------------

EventCallback = Callable[[AgentEvent], None]
EventCallbackAsync = Callable[[AgentEvent], None]  # simplified: async callbacks are awaited internally

class EventEmitter:
    """Pub/sub event emitter.

    Supports both sync and async handlers. Async handlers are run in the
    current event loop via asyncio.ensure_future.

    Usage:
        emitter = EventEmitter()
        emitter.on("tool_execution_start", handler)
        emitter.emit(AgentEvent(type=AgentEventType.TOOL_EXECUTION_START, ...))

        # Async-safe:
        emitter.emit_sync(AgentEvent(...))  # runs sync handlers in thread pool
    """

    def __init__(self) -> None:
        self._listeners: dict[str, list[EventCallback]] = {}
        self._async_listeners: dict[str, list[EventCallbackAsync]] = {}
        self._wildcard_sync: list[EventCallback] = []
        self._wildcard_async: list[EventCallbackAsync] = []

    def on(self, event_type: str | AgentEventType, handler: EventCallback) -> None:
        """Register a sync handler for an event type. Use '*' for all events."""
        key = event_type.value if isinstance(event_type, AgentEventType) else event_type
        if key == "*":
            self._wildcard_sync.append(handler)
        else:
            self._listeners.setdefault(key, []).append(handler)

    def once(self, event_type: str | AgentEventType, handler: EventCallback) -> None:
        """Register a one-shot handler."""
        key = event_type.value if isinstance(event_type, AgentEventType) else event_type
        def _once(event: AgentEvent) -> None:
            try:
                handler(event)
            finally:
                self._listeners[key] = [h for h in self._listeners[key] if h is not _once]
        self._listeners.setdefault(key, []).append(_once)

    def off(self, event_type: str | AgentEventType, handler: EventCallback) -> None:
        """Remove a specific handler. Use '*' to remove wildcard handler."""
        key = event_type.value if isinstance(event_type, AgentEventType) else event_type
        if key == "*":
            if hasattr(self, '_wildcard_sync'):
                self._wildcard_sync = [h for h in self._wildcard_sync if h is not handler]
        elif key in self._listeners:
            self._listeners[key] = [h for h in self._listeners[key] if h is not handler]

    def emit(self, event: AgentEvent) -> None:
        """Emit an event synchronously. Runs all sync handlers including wildcards."""
        key = event.type.value
        # Wildcard handlers run first (broadcast to all)
        for handler in list(self._wildcard_sync):
            try:
                handler(event)
            except Exception:
                pass
        # Specific handlers
        for handler in list(self._listeners.get(key, [])):
            try:
                handler(event)
            except Exception:
                pass  # handler failure should not stop other handlers

    async def emit_async(self, event: AgentEvent) -> None:
        """Emit an event, running async handlers concurrently."""
        key = event.type.value
        sync_handlers = list(self._listeners.get(key, []))
        async_handlers = list(self._async_listeners.get(key, []))

        # Run sync handlers
        for handler in sync_handlers:
            try:
                handler(event)
            except Exception:
                pass

        # Run async handlers concurrently
        if async_handlers:
            await asyncio.gather(
                *(self._run_async_handler(h, event) for h in async_handlers),
                return_exceptions=True,
            )

    async def _run_async_handler(self, handler: EventCallbackAsync, event: AgentEvent) -> None:
        try:
            await handler(event)
        except Exception:
            pass

    def on_async(self, event_type: str | AgentEventType, handler: EventCallbackAsync) -> None:
        """Register an async handler for an event type."""
        key = event_type.value if isinstance(event_type, AgentEventType) else event_type
        self._async_listeners.setdefault(key, []).append(handler)


# ---------------------------------------------------------------------------
# Event stream (like Pi's EventStream)
# ---------------------------------------------------------------------------

class EventStream:
    """Async generator that yields events from an EventEmitter.

    Usage:
        stream = EventStream(emitter, "tool_execution_end")
        async for event in stream:
            print(event)
    """

    def __init__(self, emitter: EventEmitter, event_type: str) -> None:
        self._emitter = emitter
        self._event_type = event_type
        self._buffer: list[AgentEvent] = []
        self._done = False
        self._event: AgentEvent | None = None

    def push(self, event: AgentEvent) -> None:
        self._buffer.append(event)
        if self._event_type == event.type.value:
            self._event = event

    async def __aiter__(self) -> "EventStream":
        for event in self._buffer:
            if self._event_type == event.type.value:
                yield event
        self._buffer.clear()
        while not self._done:
            await asyncio.sleep(0.01)
            if self._event is not None:
                yield self._event
                self._event = None

    def end(self) -> None:
        self._done = True


# ---------------------------------------------------------------------------
# Helper: build events for the bridge
# ---------------------------------------------------------------------------

def make_turn_start(turn_id: str) -> AgentEvent:
    return AgentEvent(type=AgentEventType.TURN_START, turn_id=turn_id)

def make_turn_end(turn_id: str, final_response: str) -> AgentEvent:
    return AgentEvent(type=AgentEventType.TURN_END, turn_id=turn_id, message=final_response)

def make_tool_execution_start(
    turn_id: str, tool_call_id: str, tool_name: str, tool_args: dict[str, Any]
) -> AgentEvent:
    return AgentEvent(
        type=AgentEventType.TOOL_EXECUTION_START,
        turn_id=turn_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_args=tool_args,
    )

def make_tool_execution_end(
    turn_id: str, tool_call_id: str, tool_name: str, result: str, is_error: bool
) -> AgentEvent:
    return AgentEvent(
        type=AgentEventType.TOOL_EXECUTION_END,
        turn_id=turn_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        result=result,
        is_error=is_error,
    )

def make_guardrail_nudge(
    turn_id: str, guard_name: str, nudge: str
) -> AgentEvent:
    return AgentEvent(
        type=AgentEventType.GUARDRAIL_NUDGE,
        turn_id=turn_id,
        guard_name=guard_name,
        nudge=nudge,
    )

def make_repair_nudge(
    turn_id: str, stage: str, attempt: int, tool: str, error_type: str, message: str
) -> AgentEvent:
    return AgentEvent(
        type=AgentEventType.REPAIR_NUDGE,
        turn_id=turn_id,
        repair_stage=stage,
        attempt=attempt,
        tool_name=tool,
        error_type=error_type,
        message=message,
    )

def make_classified(turn_id: str, intent: str, domain_groups: list[str] | None = None) -> AgentEvent:
    return AgentEvent(
        type=AgentEventType.CLASSIFIED,
        turn_id=turn_id,
        intent=intent,
        domain_groups=domain_groups,
    )
