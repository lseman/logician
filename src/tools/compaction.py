"""
Tool result compaction with ContentReplacementState (OpenClaude-style).

This module provides file-based storage for oversized tool results, with
preview path references that LLMs can use without seeing the full content.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Maximum size for inline results (100KB)
DEFAULT_MAX_INLINE_CHARS = 100_000

# Maximum number of compacted files to keep
MAX_COMPACTED_FILES = 100


@dataclass
class ContentReplacementState:
    """
    Tracks tool results that have been compacted to disk.

    This is a per-session state that survives conversation compaction.
    When a tool result exceeds max_result_size_chars, it's saved to disk
    and the LLM receives a preview with a path reference.

    Fields:
        replacements: Maps result ID to (file_path, preview_text)
        metadata: Additional metadata about each replacement
        total_compacted_bytes: Total bytes stored in compacted files
    """

    replacements: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_compacted_bytes: int = 0

    def register_compaction(
        self,
        result_id: str,
        file_path: str,
        preview_text: str,
        original_size: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a compacted result."""
        self.replacements[result_id] = {
            "file_path": file_path,
            "preview_text": preview_text,
            "original_size": original_size,
        }
        if metadata:
            self.metadata[result_id] = metadata

        # Update total bytes
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        self.total_compacted_bytes += file_size

    def get_replacement(self, result_id: str) -> dict[str, Any] | None:
        """Get the replacement for a result ID."""
        return self.replacements.get(result_id)

    def has_replacement(self, result_id: str) -> bool:
        """Check if a result has a replacement."""
        return result_id in self.replacements

    def get_preview_text(self, result_id: str) -> str | None:
        """Get preview text for a result."""
        replacement = self.replacements.get(result_id)
        if not replacement:
            return None
        return replacement.get("preview_text")

    def clear(self, result_id: str | None = None) -> None:
        """Clear all replacements or one specific result."""
        if result_id is not None:
            self.replacements.pop(result_id, None)
            self.metadata.pop(result_id, None)
        else:
            self.replacements.clear()
            self.metadata.clear()

    def clear_old_files(self, max_age_hours: int = 24) -> None:
        """
        Clean up compacted files older than max_age_hours.

        This prevents disk space from growing unbounded.
        """
        for result_id, data in list(self.replacements.items()):
            file_path = data.get("file_path")
            if not file_path or not os.path.exists(file_path):
                continue

            try:
                import time

                stat = os.stat(file_path)
                age_seconds = time.time() - stat.st_mtime
                age_hours = age_seconds / 3600

                if age_hours > max_age_hours:
                    # Clean up
                    os.remove(file_path)
                    self.replacements.pop(result_id, None)
                    self.metadata.pop(result_id, None)
                    self.total_compacted_bytes -= stat.st_size
            except OSError:
                pass


class FileBasedContentReplacementState(ContentReplacementState):
    """
    Persistent version of ContentReplacementState that saves to disk.

    This is useful for long-running sessions that need to survive
    restarts with their compaction state intact.
    """

    def __init__(self, storage_path: str | None = None):
        super().__init__()
        self.storage_path = Path(storage_path) if storage_path else None
        self._loaded = False

    def load(self) -> None:
        """Load state from disk."""
        if self.storage_path and self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self.replacements = data.get("replacements", {})
                    self.metadata = data.get("metadata", {})
                    self.total_compacted_bytes = data.get("total_compacted_bytes", 0)
                    self._loaded = True
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        """Save state to disk."""
        if not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "replacements": self.replacements,
            "metadata": self.metadata,
            "total_compacted_bytes": self.total_compacted_bytes,
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2, default=str)


def compact_result(
    result: Any,
    max_chars: int = DEFAULT_MAX_INLINE_CHARS,
    state: ContentReplacementState | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    """
    Compact a result that exceeds max_chars to a file.

    Args:
        result: The result to check (string, bytes, dict, etc.)
        max_chars: Maximum characters for inline output
        state: ContentReplacementState to register compacted results

    Returns:
        Tuple of (compact_result, metadata) where metadata is None for
        inline results and contains file_path + preview for compacted results.
    """
    # Serialize to string for size check, but preserve the original type for
    # non-compacted results so callers do not lose structure.
    if isinstance(result, str):
        serialized = result
    else:
        serialized = json.dumps(result, ensure_ascii=False, default=str)

    if len(serialized) <= max_chars:
        return result, None

    # Need to compact
    if state is None:
        state = ContentReplacementState()

    # Create unique file path
    import uuid

    unique_id = uuid.uuid4().hex[:8]
    storage_dir = Path(tempfile.gettempdir()) / "logician_tool_compaction"
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old directory if needed
    if storage_dir.exists() and len(list(storage_dir.glob("*"))) > MAX_COMPACTED_FILES:
        for old_file in storage_dir.glob("*.json"):
            if old_file.stat().st_mtime < time.time() - 86400:  # 24h
                old_file.unlink()

    file_path = storage_dir / f"{unique_id}.json"

    # Write full content
    with open(file_path, "w") as f:
        f.write(serialized)

    # Create preview (first 5000 chars)
    preview = serialized[:5000]

    # Register with state
    state.register_compaction(
        result_id=unique_id,
        file_path=str(file_path),
        preview_text=preview,
        original_size=len(serialized),
        metadata={"max_chars": max_chars},
    )

    return f"[COMPACTED: {file_path}] (Full content in file)", {
        "file_path": str(file_path),
        "preview": preview,
        "original_size": len(serialized),
    }
