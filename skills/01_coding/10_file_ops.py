from __future__ import annotations

if "llm" not in globals():

    class _NoOpLLM:
        def tool(self, func=None, *, name=None, description=None):
            return func if func is not None else (lambda f: f)

    llm = _NoOpLLM()

import fnmatch
from pathlib import Path


def _resolve_path(path: str) -> Path:
    """Resolve path against default_cwd (if set) so relative paths work correctly."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        base = (
            (_coding_config.get("default_cwd") or None)
            if "_coding_config" in globals()
            else None
        )
        if base:
            p = Path(base) / p
    return p.resolve()


@llm.tool(
    description="Read the contents of a file, optionally restricted to a line range."
)
def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
    """Use when: Read a source file or text file to inspect its contents.

    Triggers: read file, show file, view code, open file, print file, cat file, inspect source.
    Avoid when: You need to execute the file or modify it.
    Inputs:
      path (str, required): Absolute or relative path to the file.
      start_line (int, optional): First line to return, 1-based (default 1).
      end_line (int, optional): Last line to return, inclusive (0 = until EOF).
    Returns: JSON with file content and metadata.
    Side effects: Reads from filesystem; no writes.
    """
    try:
        p = _resolve_path(path)
        if not p.exists():
            return _safe_json({"status": "error", "error": f"File not found: {path}"})
        if not p.is_file():
            return _safe_json({"status": "error", "error": f"Not a file: {path}"})

        lines = p.read_text(encoding="utf-8", errors="replace").splitlines(
            keepends=True
        )
        total = len(lines)
        s = max(1, start_line) - 1  # convert to 0-based
        e = total if end_line <= 0 else min(end_line, total)
        selected = lines[s:e]
        content = "".join(selected)
        _MAX = 16_000
        truncated = False
        if len(content) > _MAX:
            content = content[:_MAX] + "\n...[truncated]"
            truncated = True
        return _safe_json(
            {
                "status": "ok",
                "path": str(p),
                "total_lines": total,
                "returned_lines": f"{s + 1}-{s + len(selected)}",
                "content": content,
                "truncated": truncated,
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


@llm.tool(
    description="Write (create or overwrite/append) a file with the given content."
)
def write_file(path: str, content: str, mode: str = "w") -> str:
    """Use when: Create a new file or overwrite/append an existing one.

    Triggers: write file, create file, save file, overwrite file, append to file.
    Avoid when: You only need to patch a specific string — use edit_file_replace instead.
    Inputs:
      path (str, required): Absolute or relative path to the destination file.
      content (str, required): Text content to write.
      mode (str, optional): "w" to overwrite (default), "a" to append.
    Returns: JSON with status and bytes written.
    Side effects: Creates or modifies a file on disk.
    """
    if mode not in ("w", "a"):
        return _safe_json({"status": "error", "error": "mode must be 'w' or 'a'"})
    try:
        p = _resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open(mode, encoding="utf-8") as fh:
            fh.write(content)
        return _safe_json(
            {
                "status": "ok",
                "path": str(p),
                "bytes_written": len(content.encode("utf-8")),
                "mode": mode,
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


@llm.tool(
    description="Replace the first occurrence of an exact string in a file (in-place patch)."
)
def edit_file_replace(path: str, old_string: str, new_string: str) -> str:
    """Use when: Apply a targeted patch to a source file by replacing a specific code block.

    Triggers: edit file, patch file, fix code, replace in file, change line, update function.
    Avoid when: You need to rewrite the entire file — use write_file instead.
    Inputs:
      path (str, required): Path to the file to edit.
      old_string (str, required): Exact text to find (must be unique in the file).
      new_string (str, required): Replacement text.
    Returns: JSON with status; includes diff summary.
    Side effects: Modifies the file in-place; no backup is created.
    """
    try:
        p = _resolve_path(path)
        if not p.is_file():
            return _safe_json({"status": "error", "error": f"File not found: {path}"})

        original = p.read_text(encoding="utf-8")
        count = original.count(old_string)
        if count == 0:
            return _safe_json(
                {"status": "error", "error": "old_string not found in file"}
            )
        if count > 1:
            return _safe_json(
                {
                    "status": "error",
                    "error": f"old_string matches {count} locations — be more specific",
                }
            )

        patched = original.replace(old_string, new_string, 1)
        p.write_text(patched, encoding="utf-8")
        old_lines = len(old_string.splitlines())
        new_lines = len(new_string.splitlines())
        return _safe_json(
            {
                "status": "ok",
                "path": str(p),
                "lines_removed": old_lines,
                "lines_added": new_lines,
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


@llm.tool(description="List files and subdirectories inside a directory.")
def list_directory(path: str = ".", glob_pattern: str = "*") -> str:
    """Use when: Explore a directory structure to understand project layout.

    Triggers: list files, show directory, ls, what files, directory contents, explore folder.
    Avoid when: You need to search file contents — use search_in_files instead.
    Inputs:
      path (str, optional): Directory to list (default ".").
      glob_pattern (str, optional): Glob filter (default "*", e.g. "*.py").
    Returns: JSON with sorted list of entries (files and directories).
    Side effects: Read-only.
    """
    try:
        p = _resolve_path(path)
        if not p.is_dir():
            return _safe_json({"status": "error", "error": f"Not a directory: {path}"})
        entries = []
        for child in sorted(p.iterdir()):
            if not fnmatch.fnmatch(child.name, glob_pattern):
                continue
            entry = {
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
            }
            if child.is_file():
                entry["size_bytes"] = child.stat().st_size
            entries.append(entry)

        return _safe_json(
            {
                "status": "ok",
                "path": str(p),
                "count": len(entries),
                "entries": entries,
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})
