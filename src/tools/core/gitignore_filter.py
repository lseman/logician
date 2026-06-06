"""Gitignore-aware path filtering.

Provides utilities to parse .gitignore patterns from a directory tree
and filter file paths according to those patterns. Uses the `pathspec`
library for spec-compliant matching.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

try:
    from pathspec import PathSpec
    from pathspec.patterns.gitwildmatch import GitWildMatchPattern
    _HAS_PATHSPEC = True
except ImportError:
    _HAS_PATHSPEC = False

try:
    from ...logging_utils import get_logger
except ImportError:
    get_logger = lambda name: __import__("logging").getLogger(name)

_log = get_logger("gitignore_filter")


def _find_gitignore_files(root: Path) -> list[Path]:
    """Find all .gitignore files in the directory tree up to root.

    Returns gitignore files from innermost to outermost (so outer patterns
    can override inner ones when combined).
    """
    gitignores: list[Path] = []
    # Walk from root upward to find all .gitignore files
    # Start from root and go up to the filesystem root
    current = root.resolve()
    seen = set()
    while current != current.parent and str(current) not in seen:
        seen.add(str(current))
        gi = current / ".gitignore"
        if gi.is_file():
            gitignores.append(gi)
        current = current.parent
    # Also check for .git directory's ignore patterns
    git_dir = root / ".git"
    if git_dir.is_dir():
        info_exclude = git_dir / "info" / "exclude"
        if info_exclude.is_file():
            gitignores.append(info_exclude)
    return gitignores


def _parse_gitignore(gitignore_path: Path) -> list[str]:
    """Parse a .gitignore file and return its patterns."""
    patterns: list[str] = []
    try:
        content = gitignore_path.read_text(encoding="utf-8", errors="replace")
        for line in content.splitlines():
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Handle negation
            patterns.append(line)
    except OSError:
        pass
    return patterns


def get_gitignore_spec(root: Path) -> PathSpec | None:
    """Build a combined PathSpec from all .gitignore files under root.

    Returns None if pathspec is not available or no .gitignore files found.
    """
    if not _HAS_PATHSPEC:
        return None
    try:
        gitignore_files = _find_gitignore_files(root)
        if not gitignore_files:
            return None
        all_patterns: list[str] = []
        for gi in reversed(gitignore_files):  # Outer first, inner overrides
            all_patterns.extend(_parse_gitignore(gi))
        if not all_patterns:
            return None
        return PathSpec.from_lines(GitWildMatchPattern, all_patterns)
    except Exception as exc:
        _log.debug("Failed to build gitignore spec: %s", exc)
        return None


def is_gitignored(rel_path: str, root: Path, spec: PathSpec | None = None) -> bool:
    """Check if a relative path matches any .gitignore pattern.

    Args:
        rel_path: Relative path from root (e.g., "src/main.py").
        root: Root directory the path is relative to.
        spec: Optional pre-computed PathSpec (avoids re-parsing .gitignore).

    Returns:
        True if the path should be ignored.
    """
    if spec is None:
        spec = get_gitignore_spec(root)
    if spec is None:
        return False
    # PathSpec.match_file expects a relative path
    try:
        return spec.match_file(rel_path)
    except Exception:
        return False


def filter_gitignored(paths: list[str], root: Path) -> list[str]:
    """Filter out paths that match .gitignore patterns.

    Args:
        paths: List of absolute or relative file paths.
        root: Root directory for .gitignore resolution.

    Returns:
        Filtered list with gitignored paths removed.
    """
    if not paths:
        return []
    spec = get_gitignore_spec(root)
    if spec is None:
        return paths
    resolved_root = root.resolve()
    result: list[str] = []
    for p in paths:
        try:
            abs_p = Path(p)
            if not abs_p.is_absolute():
                abs_p = resolved_root / abs_p
            rel = str(abs_p.relative_to(resolved_root))
            if not is_gitignored(rel, root, spec):
                result.append(p)
        except (ValueError, OSError):
            # Can't resolve relative to root — keep the path
            result.append(p)
    return result


def filter_gitignored_with_spec(
    paths: list[str],
    root: Path,
    spec: PathSpec,
) -> list[str]:
    """Filter paths using a pre-computed PathSpec (no re-parsing).

    Useful when filtering multiple batches against the same root.
    """
    resolved_root = root.resolve()
    result: list[str] = []
    for p in paths:
        try:
            abs_p = Path(p)
            if not abs_p.is_absolute():
                abs_p = resolved_root / abs_p
            rel = str(abs_p.relative_to(resolved_root))
            if not spec.match_file(rel):
                result.append(p)
        except (ValueError, OSError):
            result.append(p)
    return result


__all__ = [
    "get_gitignore_spec",
    "is_gitignored",
    "filter_gitignored",
    "filter_gitignored_with_spec",
]
