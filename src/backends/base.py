"""LLMBackend protocol — formalises the interface both backends implement."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ..messages import Message


@runtime_checkable
class LLMBackend(Protocol):
    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        grammar: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = True,
        on_token: Callable[[str], None] | None = None,
    ) -> str: ...

    def count_tokens(self, text: str) -> int: ...
