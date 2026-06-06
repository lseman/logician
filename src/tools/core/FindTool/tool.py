"""Find files by glob pattern. Inspired by Pi's find tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..filesystem import DEFAULT_FILESYSTEM
from ..gitignore_filter import get_gitignore_spec, filter_gitignored
from ..FileReadTool.state import resolve_tool_path

DEFAULT_MAX_RESULTS = 1000
DEFAULT_MAX_BYTES = 64 * 1024  # 64KB output limit


def find_files(
    pattern: str,
    path: str = ".",
    limit: int = DEFAULT_MAX_RESULTS,
    include_hidden: bool = False,
) -> dict[str, Any]:
    """Find files matching a glob pattern. Respects .gitignore.

    Args:
        pattern: Glob pattern (e.g. '*.ts', '**/*.json', 'src/**/*.spec.ts')
        path: Directory to search in (default: current directory)
        limit: Maximum number of results (default: 1000)
        include_hidden: Whether to include hidden files/directories

    Returns:
        Dict with 'files' list and metadata. Output truncated to 64KB.
    """
    try:
        base = resolve_tool_path(path)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}

    if not base.exists():
        return {"status": "error", "error": f"Path not found: {path}"}

    is_dir = base.is_dir() if not include_hidden else True

    # Get gitignore spec if in a git repo
    gitignore_spec = None
    try:
        git_dir = base
        while git_dir != git_dir.parent:
            if (git_dir / ".git").exists():
                gitignore_spec = get_gitignore_spec(git_dir)
                break
            git_dir = git_dir.parent
    except Exception:
        pass

    results: list[str] = []
    base_str = str(base.resolve())

    try:
        # Use Path.glob for pattern matching
        # Handle both simple and nested globs
        if "/" in pattern or "**" in pattern:
            # Nested glob - use rglob or custom walk
            if "**/" in pattern:
                # Extract base pattern and match recursively
                glob_part = pattern.replace("**/", "").replace("**", "")
                if glob_part:
                    for p in base.rglob(glob_part):
                        if p.is_file():
                            rel = str(p.relative_to(base))
                            if _should_include(rel, include_hidden, gitignore_spec):
                                results.append(rel)
                                if len(results) >= limit:
                                    break
                else:
                    # Pattern like "**" or "**/"
                    for p in base.rglob("*"):
                        if p.is_file():
                            rel = str(p.relative_to(base))
                            if _should_include(rel, include_hidden, gitignore_spec):
                                results.append(rel)
                                if len(results) >= limit:
                                    break
            else:
                # Pattern like "src/*.py" - walk src/ and match
                dir_part = pattern.rsplit("/", 1)[0]
                file_part = pattern.rsplit("/", 1)[1]
                search_dir = base / dir_part if dir_part else base
                if search_dir.exists():
                    for p in search_dir.glob(file_part):
                        if p.is_file():
                            rel = str(p.relative_to(base))
                            if _should_include(rel, include_hidden, gitignore_spec):
                                results.append(rel)
                                if len(results) >= limit:
                                    break
        else:
            # Simple glob like "*.py"
            for p in base.glob(pattern):
                if p.is_file():
                    rel = str(p.relative_to(base))
                    if _should_include(rel, include_hidden, gitignore_spec):
                        results.append(rel)
                        if len(results) >= limit:
                            break
    except Exception as exc:
        return {"status": "error", "error": f"Glob error: {exc}"}

    if not results:
        return {
            "status": "ok",
            "files": [],
            "count": 0,
            "total": 0,
        }

    # Build output
    output = "\n".join(results[:limit])
    truncated = len(results) > limit

    # Apply byte limit
    if len(output.encode("utf-8")) > DEFAULT_MAX_BYTES:
        output = output[:DEFAULT_MAX_BYTES]
        truncated = True

    result: dict[str, Any] = {
        "status": "ok",
        "files": results[:limit],
        "count": min(len(results), limit),
        "total": len(results),
    }

    if truncated:
        result["truncated"] = True
        result["notice"] = f"{limit} results limit reached. Use a more specific pattern or increase limit."

    return result


def _should_include(
    rel_path: str,
    include_hidden: bool,
    gitignore_spec: Any | None,
) -> bool:
    """Check if a path should be included based on filters."""
    # Skip hidden files/dirs unless requested
    if not include_hidden:
        parts = Path(rel_path).parts
        if any(part.startswith(".") for part in parts):
            return False

    # Apply gitignore
    if gitignore_spec is not None:
        return filter_gitignored([rel_path], gitignore_spec)

    return True
