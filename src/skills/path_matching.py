# -*- coding: utf-8 -*-
"""
Path matching for conditional skill activation.

Uses gitignore-style glob patterns (same approach as openclaude) to match
file paths against skill ``paths`` filters defined in SKILL.md frontmatter.

Supported patterns:
- Literal prefix: ``src/`` matches ``src/foo.py``
- Glob patterns: ``**/*.py`` matches any ``.py`` file at any depth
- Extension-only: ``*.py`` matches any file with that extension
- Negation: ``!*.md`` is handled by the caller (filter out negated patterns first)
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable


def matches_path_filters(
    file_paths: Iterable[str | Path],
    cwd: str | Path,
    filters: list[str],
) -> bool:
    """
    Check if any file_path matches any of the path filters.

    Parameters
    ----------
    file_paths :
        Absolute or relative file paths to check (e.g. files being edited).
    cwd :
        Current working directory for resolving relative paths.
    filters :
        Path filter patterns from a skill's ``paths`` frontmatter field.

    Returns
    -------
    True if at least one filter matches at least one file path.
    """
    if not filters:
        return False

    cwd = Path(cwd).resolve()

    for file_str in file_paths:
        file_path = Path(file_str).resolve()
        try:
            rel = file_path.relative_to(cwd)
        except ValueError:
            # file not under cwd — skip
            continue

        rel_str = str(rel)
        for f in filters:
            if _match_glob(rel_str, f):
                return True

    return False


def _match_glob(rel_path: str, pattern: str) -> bool:
    """
    Match a relative file path against a gitignore-style glob pattern.

    Supports:
    - ``**`` — matches everything
    - ``**/foo.py`` — matches any foo.py at any depth
    - ``**/*.py`` — matches any .py file at any depth
    - ``src/`` — matches anything under src/
    - ``*.py`` — matches any .py file
    - Literal file names: ``README.md``
    """
    # Exact match
    if fnmatch.fnmatch(rel_path, pattern):
        return True

    # "**" matches everything
    if pattern == "**":
        return True

    # "**/something" — match at any depth
    if pattern.startswith("**/"):
        suffix = pattern[3:]
        # rel_path matches suffix directly or anywhere under it
        if fnmatch.fnmatch(rel_path, suffix):
            return True
        if fnmatch.fnmatch(rel_path, "**/" + suffix):
            return True
        return False

    # Extension-only patterns like "*.py" — match against filename only
    if "/" not in pattern:
        filename = Path(rel_path).name
        if fnmatch.fnmatch(filename, pattern):
            return True
        # Also try matching against the relative path (for flat directories)
        if fnmatch.fnmatch(rel_path, pattern):
            return True

    # Directory prefix patterns like "src/" — match anything under it
    if pattern.endswith("/"):
        dir_prefix = pattern[:-1]
        if rel_path.startswith(dir_prefix + "/") or fnmatch.fnmatch(rel_path, dir_prefix + "/**"):
            return True
        if fnmatch.fnmatch(rel_path, pattern + "**"):
            return True

    # General glob: try with "**/" prefix for directory patterns like "src/"
    if fnmatch.fnmatch(rel_path, "**/" + pattern):
        return True

    return False
