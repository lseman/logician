"""Compatibility facade for core file tools.

The public tool names stay anchored here for backwards imports, while the
implementation is split across focused modules:
- ``filesystem.py`` for path resolution and read-only filesystem metadata
- ``editing.py`` for write/edit flows
- ``inspection.py`` for code navigation helpers
- ``git_tools.py`` for git-aware inspection
"""

from __future__ import annotations

from typing import Any

from .editing import apply_edit_block, edit_file, preview_edit, smart_edit, write_file
from .filesystem import DEFAULT_FILESYSTEM
from .git_tools import get_git_diff, get_git_status
from .inspection import find_imports, get_symbol_info, read_line, search_file
from .search import grep_files

__all__ = [
    "read_file",
    "write_file",
    "edit_file",
    "search_file",
    "list_dir",
    "apply_edit_block",
    "preview_edit",
    "smart_edit",
    "get_git_status",
    "get_git_diff",
    "get_symbol_info",
    "read_line",
    "find_imports",
    "grep_files",
]


def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, Any]:
    """Read a file, optionally restricted to a line range."""
    return DEFAULT_FILESYSTEM.read_file(path, start_line=start_line, end_line=end_line)


def list_dir(path: str = ".", glob_pattern: str = "*") -> dict[str, Any]:
    """List entries in a directory."""
    return DEFAULT_FILESYSTEM.list_dir(path, glob_pattern=glob_pattern)

