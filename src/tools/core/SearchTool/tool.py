"""Core search tools backed by the shared filesystem backend."""

from __future__ import annotations

from typing import Any

from ..filesystem import DEFAULT_FILESYSTEM
from .inspection import search_code as inspection_search_code


def glob_files(
    pattern: str,
    path: str = ".",
    include_hidden: bool = False,
    offset: int = 0,
    max_results: int | None = None,
    output_mode: str = "entries",
) -> dict[str, Any]:
    """Find files matching a glob pattern under a directory.

    Returns structured metadata for each match instead of a plain newline string.

    Args:
        pattern: Glob pattern such as ``"**/*.py"`` or ``"*.md"``.
        path: Base directory to search.
    """
    return DEFAULT_FILESYSTEM.glob(
        pattern=pattern,
        path=path,
        include_hidden=include_hidden,
        offset=offset,
        max_results=max_results,
        output_mode=output_mode,
    )


def grep_files(
    pattern: str,
    path: str = ".",
    glob: str = "**/*",
    literal: bool = True,
    case_sensitive: bool = True,
    context_lines: int = 1,
    max_matches: int = 50,
    include_hidden: bool = False,
    offset: int = 0,
    output_mode: str = "matches",
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
        include_hidden=include_hidden,
        offset=offset,
        output_mode=output_mode,
    )


def search_code(
    query: str,
    path: str = ".",
    glob: str = "**/*",
    mode: str = "literal",
    case_sensitive: bool = True,
    context_lines: int = 2,
    max_results: int = 50,
    include_hidden: bool = False,
    offset: int = 0,
) -> dict[str, Any]:
    """Search code using multiline text or Python symbol matching."""
    return inspection_search_code(
        query,
        path=path,
        glob=glob,
        mode=mode,
        case_sensitive=case_sensitive,
        context_lines=context_lines,
        max_results=max_results,
        include_hidden=include_hidden,
        offset=offset,
    )
