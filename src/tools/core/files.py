"""Core file operations: read, write, edit, apply_edit_block."""
from __future__ import annotations

import re
from pathlib import Path


def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """Read a file. Optionally limit to a line range (1-indexed, inclusive).

    Returns file contents as a string with line numbers prepended (cat -n style).

    Args:
        path: File path (absolute or relative).
        start_line: First line to return, 1-indexed (default: 1).
        end_line: Last line to return, inclusive (default: EOF).

    Returns:
        File contents with line numbers, or error message.
    """
    p = Path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return f"Error reading {path}: {e}"

    if start_line is not None or end_line is not None:
        s = max(0, (start_line or 1) - 1)
        e = end_line if end_line is not None else len(lines)
        lines = lines[s:e]
        offset = s
    else:
        offset = 0

    return "\n".join(f"{i + offset + 1:6}\t{line}" for i, line in enumerate(lines))


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed.

    Args:
        path: File path (absolute or relative).
        content: Text content to write.

    Returns:
        Success message or error.
    """
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written: {path} ({len(content)} bytes)"
    except Exception as e:
        return f"Error writing {path}: {e}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace the first occurrence of old_string with new_string in a file.

    Args:
        path: File path to edit.
        old_string: Exact text to find (must be unique in the file).
        new_string: Replacement text.

    Returns:
        Success message or error. Returns error if old_string is not found
        or appears more than once.
    """
    p = Path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    try:
        content = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {path}: {e}"

    count = content.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {path}"
    if count > 1:
        return f"Error: old_string found {count} times in {path} — must be unique"

    new_content = content.replace(old_string, new_string, 1)
    try:
        p.write_text(new_content, encoding="utf-8")
        return f"Edited: {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


def apply_edit_block(path: str, blocks: str) -> str:
    """Apply one or more SEARCH/REPLACE edit blocks to a file.

    Blocks are in the format:
        <<<<<<< SEARCH
        old content
        =======
        new content
        >>>>>>> REPLACE

    Multiple blocks can be provided in sequence.

    Args:
        path: File path to edit.
        blocks: String containing one or more SEARCH/REPLACE blocks.

    Returns:
        Success message with count of blocks applied, or error message.
    """
    p = Path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    try:
        content = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {path}: {e}"

    # Parse all SEARCH/REPLACE blocks
    pattern = re.compile(
        r"<{7}\s*SEARCH\s*\n(.*?)\n={7}\s*\n(.*?)\n>{7}\s*REPLACE",
        re.DOTALL,
    )
    matches = list(pattern.finditer(blocks))
    if not matches:
        return f"Error: no valid SEARCH/REPLACE blocks found in blocks argument"

    result = content
    applied = 0
    for m in matches:
        search_text = m.group(1)
        replace_text = m.group(2)
        if search_text not in result:
            return f"Error: SEARCH text not found in {path}:\n{search_text[:200]}"
        result = result.replace(search_text, replace_text, 1)
        applied += 1

    try:
        p.write_text(result, encoding="utf-8")
        return f"Applied {applied} edit block(s) to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"
