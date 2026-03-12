"""Core search tools: glob (file finding) and grep (content search)."""
from __future__ import annotations

import re
from pathlib import Path


def glob_files(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern under a directory.

    Returns matching paths, one per line, sorted by modification time (newest first).

    Args:
        pattern: Glob pattern (e.g. "**/*.py", "*.txt").
        path: Base directory to search (default ".").

    Returns:
        One file path per line, or error/no-matches message.

    Example:
        glob_files("**/*.py", "src/")
    """
    base = Path(path).resolve()
    if not base.exists():
        return f"Error: directory not found: {path}"
    try:
        matches = sorted(
            base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not matches:
            return f"No files matching {pattern!r} under {path}"
        return "\n".join(str(m.relative_to(base)) for m in matches[:500])
    except Exception as e:
        return f"Error: {e}"


def grep_files(
    pattern: str,
    path: str = ".",
    glob: str = "*",
    output_mode: str = "files_with_matches",
) -> str:
    """Search file contents for a regex pattern.

    Args:
        pattern: Regular expression pattern to search for.
        path: Directory to search (default ".").
        glob: Glob pattern to filter files (default "*").
        output_mode: How to format results:
            - "files_with_matches": list files containing the pattern (default)
            - "content": show matching lines with file:line context
            - "count": show match counts per file

    Returns:
        Formatted search results or error message.

    Example:
        grep_files("class Agent", "src/", "*.py", "content")
    """
    base = Path(path).resolve()
    if not base.exists():
        return f"Error: path not found: {path}"

    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"

    try:
        if base.is_file():
            files = [base]
        else:
            files = [f for f in base.rglob(glob) if f.is_file()]
    except Exception as e:
        return f"Error: {e}"

    results: list[str] = []

    for f in sorted(files)[:1000]:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines = text.splitlines()
        matches = [(i + 1, line) for i, line in enumerate(lines) if rx.search(line)]
        if not matches:
            continue

        rel = str(f.relative_to(base) if f.is_relative_to(base) else f)

        if output_mode == "files_with_matches":
            results.append(rel)
        elif output_mode == "content":
            for lineno, line in matches[:50]:
                results.append(f"{rel}:{lineno}: {line}")
        elif output_mode == "count":
            results.append(f"{rel}: {len(matches)}")

    if not results:
        return f"No matches for {pattern!r} in {path}"
    return "\n".join(results[:2000])
