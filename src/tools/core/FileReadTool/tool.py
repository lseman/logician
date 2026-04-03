"""Read-only file access tools."""

from __future__ import annotations

from typing import Any

from ..filesystem import DEFAULT_FILESYSTEM
from ..SearchTool.inspection import read_edit_context as inspection_read_edit_context
from .state import current_file_snapshot, record_file_snapshot, resolve_tool_path

__all__ = [
    "read_file",
    "read_edit_context",
    "list_dir",
]


def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, Any]:
    """Read a file, optionally restricted to a line range."""
    try:
        resolved = resolve_tool_path(path)
        snap = current_file_snapshot(globals().get("ctx"), resolved)
        if snap and resolved.exists():
            current_mtime = resolved.stat().st_mtime_ns
            snap_mtime = snap.get("mtime_ns")
            if snap_mtime is not None and int(current_mtime) == int(snap_mtime):
                req_start = start_line or 1
                snap_start = snap.get("start_line") or 1
                snap_end = snap.get("end_line") or snap.get("total_lines")
                range_covered = snap.get("full_read") or (
                    req_start >= snap_start
                    and (end_line is None or (snap_end is not None and end_line <= snap_end))
                )
                if range_covered:
                    summary = {
                        "path": snap.get("path"),
                        "full_read": snap.get("full_read"),
                        "start_line": snap.get("start_line"),
                        "end_line": snap.get("end_line"),
                        "total_lines": snap.get("total_lines"),
                        "truncated": snap.get("truncated"),
                    }
                    return {
                        "status": "ok",
                        "path": str(resolved),
                        "file_type": "file_unchanged",
                        "unchanged": True,
                        "message": "File unchanged since last read — returning cached content.",
                        "content": snap.get("content", ""),
                        "total_lines": snap.get("total_lines"),
                        "snapshot": summary,
                    }
    except (OSError, ValueError):
        pass

    result = DEFAULT_FILESYSTEM.read_file(path, start_line=start_line, end_line=end_line)
    if result.get("status") != "ok" or result.get("file_type") != "text":
        return result

    try:
        resolved = resolve_tool_path(path)
        stat_result = resolved.stat()
    except (OSError, ValueError):
        return result

    total_lines = int(result.get("total_lines") or 0)
    effective_start = max(1, start_line or 1)
    effective_end = total_lines if not end_line or end_line <= 0 else min(total_lines, end_line)
    truncated = bool(result.get("truncated"))
    full_read = not truncated and effective_start == 1 and effective_end >= total_lines
    snapshot = record_file_snapshot(
        globals().get("ctx"),
        resolved,
        content=str(result.get("content") or ""),
        mtime_ns=stat_result.st_mtime_ns,
        size_bytes=stat_result.st_size,
        full_read=full_read,
        start_line=effective_start,
        end_line=effective_end,
        total_lines=total_lines,
        truncated=truncated,
        source="read_file",
    )
    result["snapshot"] = {
        "path": snapshot["path"],
        "full_read": snapshot["full_read"],
        "start_line": snapshot["start_line"],
        "end_line": snapshot["end_line"],
        "total_lines": snapshot["total_lines"],
        "truncated": snapshot["truncated"],
    }
    return result


def read_edit_context(
    path: str,
    needle: str,
    context_lines: int = 3,
    max_scan_bytes: int = 10 * 1024 * 1024,
) -> dict[str, Any]:
    """Read a bounded slice of a file around a matching needle."""
    return inspection_read_edit_context(
        path,
        needle,
        context_lines=context_lines,
        max_scan_bytes=max_scan_bytes,
    )


def list_dir(path: str = ".", glob_pattern: str = "*") -> dict[str, Any]:
    """List entries in a directory."""
    return DEFAULT_FILESYSTEM.list_dir(path, glob_pattern=glob_pattern)
