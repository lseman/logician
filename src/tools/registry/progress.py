# -*- coding: utf-8 -*-
"""
Thread-local progress callback system for tool execution.

Provides a simple progress event system that tools and the registry can use
to report execution phases to UI consumers (e.g. TUI, web UI).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

__all__ = [
    "ProgressEvent",
    "ProgressCallback",
    "set_progress_callback",
    "get_progress_callback",
    "emit_progress",
]


@dataclass(frozen=True)
class ProgressEvent:
    tool_name: str
    phase: str
    message: str
    progress: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[ProgressEvent], None]

_progress_local = threading.local()


def set_progress_callback(callback: ProgressCallback | None) -> None:
    _progress_local.callback = callback


def get_progress_callback() -> ProgressCallback | None:
    return getattr(_progress_local, "callback", None)


def emit_progress(tool_name: str, phase: str, message: str, **kwargs: Any) -> None:
    """Emit a progress event if a callback is registered."""
    cb = get_progress_callback()
    if cb is None:
        return
    try:
        event = ProgressEvent(
            tool_name=tool_name,
            phase=phase,
            message=message,
            **kwargs,
        )
        cb(event)
    except Exception:
        # Never let progress callbacks crash tool execution
        pass
