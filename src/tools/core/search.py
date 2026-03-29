"""Core search tools backed by the shared filesystem backend."""

from __future__ import annotations

from typing import Any

from .filesystem import DEFAULT_FILESYSTEM


def glob_files(pattern: str, path: str = ".") -> dict[str, Any]:
    """Find files matching a glob pattern under a directory.

    Returns structured metadata for each match instead of a plain newline string.

    Args:
        pattern: Glob pattern such as ``"**/*.py"`` or ``"*.md"``.
        path: Base directory to search.
    """
    return DEFAULT_FILESYSTEM.glob(pattern=pattern, path=path)


def grep_files(
    pattern: str,
    path: str = ".",
    glob: str = "**/*",
    literal: bool = True,
    case_sensitive: bool = True,
    context_lines: int = 1,
    max_matches: int = 50,
) -> dict[str, Any]:
    """Search for a pattern across files in a directory tree."""
    return DEFAULT_FILESYSTEM.grep(
        pattern,
        path=path,
        glob=glob,
        literal=literal,
        case_sensitive=case_sensitive,
        context_lines=context_lines,
        max_matches=max_matches,
    )

