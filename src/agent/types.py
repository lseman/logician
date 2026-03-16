"""Shared agent types. Message and ToolCall live in their existing modules;
this module adds TurnResult which depends on TurnState (imported lazily to
avoid circular imports)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..messages import Message, MessageRole  # noqa: F401
from ..tools.runtime import ToolCall

if TYPE_CHECKING:
    from .state import TurnState


class TurnResult:
    """The result of a single agent turn."""

    def __init__(self, state: TurnState, messages: list[Message]) -> None:
        self._state = state
        self.messages = messages

    @property
    def final_response(self) -> str | None:
        return self._state.final_response

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self._state.tool_calls

    @property
    def thinking_log(self) -> list[str]:
        """All thinking/planning content collected during this turn."""
        return self._state.thinking_log

    @property
    def state(self) -> TurnState:
        return self._state


__all__ = ["TurnResult", "Message", "MessageRole", "ToolCall"]
