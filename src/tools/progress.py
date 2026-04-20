"""Typed progress tracking for long-running tool operations.

OpenClaude-inspired pattern: structured progress types for better UX
during long-running operations (cargo build, web search, MCP operations, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# =============================================================================
# PROGRESS TYPES (Typed hierarchy for type-safe progress tracking)
# =============================================================================

@dataclass
class BashProgress:
    """Progress for bash command execution."""
    command: str
    phase: str = "running"
    duration_seconds: float = 0.0
    exit_code: int | None = None


@dataclass
class WebSearchProgress:
    """Progress for web search operations."""
    query: str
    phase: str = "fetching"
    results_count: int = 0
    current_result: int = 0


@dataclass
class FileOperationProgress:
    """Progress for file operations (read/write/scan)."""
    path: str
    operation: str = "reading"
    lines_read: int = 0
    total_lines: int | None = None
    bytes_read: int = 0
    bytes_total: int | None = None


@dataclass
class MCPProgress:
    """Progress for MCP tool operations."""
    tool_name: str
    phase: str = "invoking"
    result: Any | None = None
    error: str | None = None


# Union type for any progress type
ToolProgressData = BashProgress | WebSearchProgress | FileOperationProgress | MCPProgress


# =============================================================================
# PROGRESS UTILITIES
# =============================================================================

def format_progress(progress: ToolProgressData) -> str:
    """Format progress data as a string for NDJSON output."""
    if isinstance(progress, BashProgress):
        return f"bash:{progress.command}:{progress.phase}:{progress.duration_seconds:.1f}s"
    if isinstance(progress, WebSearchProgress):
        return f"web:{progress.query}:{progress.phase}:{progress.results_count}:{progress.current_result}"
    if isinstance(progress, FileOperationProgress):
        return f"file:{progress.path}:{progress.operation}:{progress.lines_read}/{progress.total_lines or '?'}:{progress.bytes_read}/{progress.bytes_total or '?'}B"
    if isinstance(progress, MCPProgress):
        return f"mcp:{progress.tool_name}:{progress.phase}:{progress.error or 'ok'}"
    return "unknown"


def build_progress_message(
    progress: ToolProgressData,
    tool_name: str,
    tool_use_id: str,
) -> dict[str, Any]:
    """Build a progress message for tool use."""
    return {
        "tool_use_id": tool_use_id,
        "tool_name": tool_name,
        "progress": format_progress(progress),
        "data": {
            "type": type(progress).__name__,
            "phase": progress.phase,
            "duration": getattr(progress, "duration_seconds", None),
        },
    }


# =============================================================================
# PROGRESS HANDLER
# =============================================================================

class ProgressHandler:
    """Manages progress tracking for tool operations."""

    def __init__(self):
        self._current_progress: ToolProgressData | None = None
        self._history: list[dict[str, Any]] = []

    def start(
        self,
        progress: ToolProgressData,
        tool_name: str,
        tool_use_id: str,
    ) -> None:
        """Start tracking progress for a tool operation."""
        self._current_progress = progress

    def update(
        self,
        phase: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Update progress for the current operation."""
        if self._current_progress is None:
            return

        # Update the progress object's phase
        if isinstance(self._current_progress, BashProgress):
            self._current_progress.phase = phase
        elif isinstance(self._current_progress, WebSearchProgress):
            self._current_progress.phase = phase
            if data and "results_count" in data:
                self._current_progress.results_count = data["results_count"]
            if data and "current_result" in data:
                self._current_progress.current_result = data["current_result"]
        elif isinstance(self._current_progress, FileOperationProgress):
            self._current_progress.phase = phase
            if data and "lines_read" in data:
                self._current_progress.lines_read = data["lines_read"]
            if data and "total_lines" in data:
                self._current_progress.total_lines = data["total_lines"]
            if data and "bytes_read" in data:
                self._current_progress.bytes_read = data["bytes_read"]
            if data and "bytes_total" in data:
                self._current_progress.bytes_total = data["bytes_total"]
        elif isinstance(self._current_progress, MCPProgress):
            self._current_progress.phase = phase

    def finish(
        self,
        result: Any | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Finish tracking progress and return summary."""
        if self._current_progress is None:
            return {"status": "error", "error": "No active progress"}

        progress_type = type(self._current_progress).__name__
        summary = {
            "tool_use_id": self._current_progress.__dict__.get("command") or "unknown",
            "progress_type": progress_type,
            "final_phase": self._current_progress.phase,
            "duration_seconds": getattr(self._current_progress, "duration_seconds", 0),
        }
        if error:
            summary["error"] = error
        if result is not None:
            summary["result"] = str(result)[:1000] if isinstance(result, str) else result

        self._history.append(summary)
        self._current_progress = None

        return summary

    def get_current(self) -> ToolProgressData | None:
        """Get the current progress object."""
        return self._current_progress

    def get_history(self) -> list[dict[str, Any]]:
        """Get the progress history."""
        return self._history


# Global progress handler instance
_progress_handler: ProgressHandler | None = None


def get_progress_handler() -> ProgressHandler:
    """Get or create the global progress handler."""
    global _progress_handler
    if _progress_handler is None:
        _progress_handler = ProgressHandler()
    return _progress_handler


def set_progress_handler(handler: ProgressHandler) -> None:
    """Set a custom progress handler."""
    global _progress_handler
    _progress_handler = handler
