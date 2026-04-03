"""Shared snapshot state for file-oriented tools."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from ...runtime import Context
from ..filesystem import DEFAULT_FILESYSTEM

_LOCAL_TOOL_CTX = Context()


def get_file_state_context(runtime_ctx: Any | None) -> Context:
    ctx = runtime_ctx if isinstance(runtime_ctx, Context) else None
    if ctx is None:
        return _LOCAL_TOOL_CTX
    if not hasattr(ctx, "file_snapshots"):
        ctx.file_snapshots = {}
    return ctx


def resolve_tool_path(path: str | Path) -> Path:
    return DEFAULT_FILESYSTEM.resolve_path(path)


def current_file_snapshot(
    runtime_ctx: Any | None,
    path: str | Path,
) -> dict[str, Any] | None:
    ctx = get_file_state_context(runtime_ctx)
    resolved = resolve_tool_path(path)
    payload = getattr(ctx, "file_snapshots", {})
    return dict(payload.get(str(resolved), {})) or None


def clear_file_snapshot(runtime_ctx: Any | None, path: str | Path) -> None:
    ctx = get_file_state_context(runtime_ctx)
    resolved = resolve_tool_path(path)
    payload = getattr(ctx, "file_snapshots", {})
    payload.pop(str(resolved), None)


def build_snapshot(
    path: Path,
    *,
    content: str,
    mtime_ns: int | None,
    size_bytes: int | None,
    full_read: bool,
    start_line: int | None,
    end_line: int | None,
    total_lines: int | None,
    truncated: bool,
    source: str,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "content": content,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "mtime_ns": int(mtime_ns) if mtime_ns is not None else None,
        "size_bytes": int(size_bytes) if size_bytes is not None else None,
        "full_read": bool(full_read),
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total_lines,
        "truncated": bool(truncated),
        "source": source,
    }


def record_file_snapshot(
    runtime_ctx: Any | None,
    path: str | Path,
    *,
    content: str,
    mtime_ns: int | None,
    size_bytes: int | None,
    full_read: bool,
    start_line: int | None = None,
    end_line: int | None = None,
    total_lines: int | None = None,
    truncated: bool = False,
    source: str = "read_file",
) -> dict[str, Any]:
    ctx = get_file_state_context(runtime_ctx)
    resolved = resolve_tool_path(path)
    snapshot = build_snapshot(
        resolved,
        content=content,
        mtime_ns=mtime_ns,
        size_bytes=size_bytes,
        full_read=full_read,
        start_line=start_line,
        end_line=end_line,
        total_lines=total_lines,
        truncated=truncated,
        source=source,
    )
    payload = getattr(ctx, "file_snapshots", {})
    payload[str(resolved)] = snapshot
    return snapshot


def record_symbol_snapshot(
    runtime_ctx: Any | None,
    path: str | Path,
    *,
    content: str,
    mtime_ns: int | None,
    size_bytes: int | None,
    source: str,
) -> dict[str, Any]:
    return record_file_snapshot(
        runtime_ctx,
        path,
        content=content,
        mtime_ns=mtime_ns,
        size_bytes=size_bytes,
        full_read=False,
        truncated=False,
        source=source,
    )


def refresh_snapshot_after_write(
    runtime_ctx: Any | None,
    path: str | Path,
    *,
    content: str,
) -> dict[str, Any]:
    resolved = resolve_tool_path(path)
    stat_result = resolved.stat()
    total_lines = len(content.splitlines())
    return record_file_snapshot(
        runtime_ctx,
        resolved,
        content=content,
        mtime_ns=stat_result.st_mtime_ns,
        size_bytes=stat_result.st_size,
        full_read=True,
        start_line=1,
        end_line=total_lines,
        total_lines=total_lines,
        truncated=False,
        source="post_write",
    )


def ensure_snapshot_allows_existing_file_write(
    runtime_ctx: Any | None,
    path: str | Path,
    *,
    operation: str,
    allow_partial: bool = False,
) -> tuple[Path, dict[str, Any], str, os.stat_result] | dict[str, Any]:
    try:
        resolved = resolve_tool_path(path)
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "path": str(path)}

    snapshot = current_file_snapshot(runtime_ctx, resolved)
    if not snapshot:
        return {
            "status": "error",
            "error": (
                f"File has not been read yet. Read `{resolved}` before attempting to {operation} it."
            ),
            "path": str(resolved),
            "reason": "file_not_read",
            "requires_read": True,
        }
    if not allow_partial and not bool(snapshot.get("full_read")):
        return {
            "status": "error",
            "error": (
                f"File was only partially inspected. Read the full file `{resolved}` before attempting to {operation} it."
            ),
            "path": str(resolved),
            "reason": "partial_read",
            "requires_read": True,
            "snapshot": _snapshot_summary(snapshot),
        }

    try:
        stat_result = resolved.stat()
        current_content = DEFAULT_FILESYSTEM.read_text(
            resolved,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return {
            "status": "error",
            "error": f"Cannot read current file contents: {exc}",
            "path": str(resolved),
            "reason": "current_read_failed",
        }

    snapshot_content = str(snapshot.get("content") or "")
    snapshot_mtime_ns = snapshot.get("mtime_ns")
    if (
        snapshot_mtime_ns is not None
        and int(stat_result.st_mtime_ns) != int(snapshot_mtime_ns)
        and current_content != snapshot_content
    ):
        return {
            "status": "error",
            "error": (
                "File has changed since it was read. Read it again before applying this write."
            ),
            "path": str(resolved),
            "reason": "stale_snapshot",
            "requires_read": True,
            "snapshot": _snapshot_summary(snapshot),
            "current": {
                "mtime_ns": int(stat_result.st_mtime_ns),
                "size_bytes": int(stat_result.st_size),
            },
        }

    return resolved, snapshot, current_content, stat_result


def parse_structured_patch(diff_text: str) -> list[dict[str, Any]]:
    if not diff_text.strip():
        return []

    hunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    header_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    for raw_line in diff_text.splitlines():
        if raw_line.startswith(("--- ", "+++ ", "diff --git ", "index ")):
            continue
        match = header_re.match(raw_line)
        if match:
            current = {
                "old_start": int(match.group(1)),
                "old_lines": int(match.group(2) or "1"),
                "new_start": int(match.group(3)),
                "new_lines": int(match.group(4) or "1"),
                "lines": [],
            }
            hunks.append(current)
            continue
        if current is not None:
            current["lines"].append(raw_line)
    return hunks


def _snapshot_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": snapshot.get("path"),
        "mtime_ns": snapshot.get("mtime_ns"),
        "size_bytes": snapshot.get("size_bytes"),
        "full_read": bool(snapshot.get("full_read")),
        "start_line": snapshot.get("start_line"),
        "end_line": snapshot.get("end_line"),
        "total_lines": snapshot.get("total_lines"),
        "truncated": bool(snapshot.get("truncated")),
        "source": snapshot.get("source"),
    }


__all__ = [
    "build_snapshot",
    "clear_file_snapshot",
    "current_file_snapshot",
    "ensure_snapshot_allows_existing_file_write",
    "get_file_state_context",
    "parse_structured_patch",
    "record_file_snapshot",
    "record_symbol_snapshot",
    "refresh_snapshot_after_write",
    "resolve_tool_path",
]
