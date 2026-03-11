"""CC-parity file exploration and editing tools.

Provides six tools that mirror the behaviour of the built-in Claude Code
tool-set so the agent can perform precise file operations with the same
semantics users expect.

Tool inventory
--------------
cc_glob       -- find files by glob pattern, sorted newest-first
cc_grep       -- search file contents (ripgrep with pure-Python fallback)
cc_read       -- read a file in cat-n format with line numbers
cc_edit       -- surgical string replacement (unique-match enforced)
cc_write      -- write / overwrite a file, creating parent dirs
cc_multi_edit -- apply multiple sequential edits to a single file
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from skills.coding.bootstrap.runtime_access import tool


__skill__ = {
    "name": "cc_tools",
    "description": (
        "Precise Claude Code-style file exploration and surgical editing tools. "
        "Use for finding files, reading narrow ranges, and applying exact string edits."
    ),
    "aliases": [
        "find files",
        "read file",
        "edit file",
        "search code",
        "navigate codebase",
        "glob",
        "grep",
    ],
    "triggers": [
        "look for",
        "find where",
        "read the",
        "edit",
        "change",
        "fix",
        "modify",
        "search for",
    ],
    "preferred_tools": [
        "cc_glob",
        "cc_grep",
        "cc_read",
        "cc_edit",
        "cc_multi_edit",
    ],
    "example_queries": [
        "find all Python files in src/",
        "where is this function defined",
        "read lines 40 to 80 of config.py",
        "fix the bug in the parse function",
    ],
    "when_not_to_use": [
        "shell execution",
        "git operations",
        "running tests",
    ],
    "next_skills": ["shell", "git", "quality"],
    "workflow": [
        "Find candidate files with cc_glob or cc_grep.",
        "Read only the relevant slice with cc_read.",
        "Use cc_edit for one surgical change and cc_multi_edit for multiple changes in one file.",
        "Use cc_write only for new files or full rewrites.",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# cc_glob
# ---------------------------------------------------------------------------

@tool
def cc_glob(pattern: str, path: str = ".", head_limit: int = 0) -> str:
    """Find files matching a glob pattern, sorted by modification time (newest first).

    Use when: searching for files by name pattern (e.g. **/*.py, src/**/config.*).
    """
    base = Path(path)
    matches = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        return "(no matches)"
    lines = [str(m) for m in matches]
    if head_limit and head_limit > 0:
        lines = lines[:head_limit]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# cc_grep
# ---------------------------------------------------------------------------

def _grep_pure_python(
    pattern: str,
    path: str,
    glob_filter: str,
    output_mode: str,
    context: int,
    case_insensitive: bool,
    head_limit: int,
    offset: int,
    multiline: bool,
) -> str:
    """Pure-Python fallback for cc_grep."""
    flags = re.IGNORECASE if case_insensitive else 0
    if multiline:
        flags |= re.DOTALL

    base = Path(path)
    if glob_filter:
        candidates = list(base.rglob(glob_filter))
    else:
        candidates = [p for p in base.rglob("*") if p.is_file()]

    results: list[str] = []

    if output_mode == "files_with_matches":
        for fp in candidates:
            try:
                text = _read_text(fp)
            except Exception:
                continue
            if re.search(pattern, text, flags):
                results.append(str(fp))

    elif output_mode == "count":
        for fp in candidates:
            try:
                lines = _read_text(fp).splitlines()
            except Exception:
                continue
            count = sum(1 for ln in lines if re.search(pattern, ln, flags))
            if count:
                results.append(f"{fp}:{count}")

    else:  # content
        for fp in candidates:
            try:
                lines = _read_text(fp).splitlines()
            except Exception:
                continue
            for i, ln in enumerate(lines):
                if re.search(pattern, ln, flags):
                    lo = max(0, i - context)
                    hi = min(len(lines), i + context + 1)
                    for j in range(lo, hi):
                        results.append(f"{fp}:{j + 1}:{lines[j]}")

    if offset:
        results = results[offset:]
    if head_limit and head_limit > 0:
        results = results[:head_limit]

    return "\n".join(results) if results else "(no matches)"

@tool
def cc_grep(
    pattern: str,
    path: str = ".",
    glob: str = "",
    type: str = "",
    output_mode: str = "files_with_matches",
    context: int = 0,
    case_insensitive: bool = False,
    head_limit: int = 0,
    offset: int = 0,
    multiline: bool = False,
) -> str:
    """Search file contents using ripgrep (falls back to pure-Python grep).

    Use when: finding which files contain a pattern, or viewing matching lines.
    """
    if shutil.which("rg") is None:
        return _grep_pure_python(
            pattern, path, glob, output_mode, context,
            case_insensitive, head_limit, offset, multiline,
        )

    cmd: list[str] = ["rg"]

    if output_mode == "files_with_matches":
        cmd.append("--files-with-matches")
    elif output_mode == "count":
        cmd.append("--count")
    # content mode: default rg output

    if context and output_mode == "content":
        cmd += ["-C", str(context)]

    if case_insensitive:
        cmd.append("-i")

    if multiline:
        cmd += ["-U", "--multiline-dotall"]

    if glob:
        cmd += ["--glob", glob]

    if type:
        cmd += ["--type", type]

    cmd += ["--", pattern, path]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        raw = proc.stdout.strip()
    except Exception as exc:
        return _grep_pure_python(
            pattern, path, glob, output_mode, context,
            case_insensitive, head_limit, offset, multiline,
        )

    if not raw:
        return "(no matches)"

    lines = raw.splitlines()
    if offset:
        lines = lines[offset:]
    if head_limit and head_limit > 0:
        lines = lines[:head_limit]

    return "\n".join(lines) if lines else "(no matches)"


# ---------------------------------------------------------------------------
# cc_read
# ---------------------------------------------------------------------------

@tool
def cc_read(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file, returning lines in cat-n format with line numbers.

    Use when: reading a file or a specific section of a file.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not p.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {file_path}")

    all_lines = _read_text(p).splitlines()
    selected = all_lines[offset: offset + limit]

    out_lines: list[str] = []
    for i, line in enumerate(selected, start=offset + 1):
        truncated = line[:2000]
        out_lines.append(f"{i:>6}\t{truncated}")

    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# cc_edit
# ---------------------------------------------------------------------------

@tool
def cc_edit(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Replace an exact string in a file. Fails if string not found or not unique.

    Use when: making a single surgical edit to an existing file.
    IMPORTANT: old_string must be unique in the file. Include enough surrounding
    context (blank lines, neighbouring statements) to make it unique.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = _read_text(p)
    count = content.count(old_string)

    if count == 0:
        raise ValueError(f"old_string not found in {file_path!r}")

    if count > 1 and not replace_all:
        raise ValueError(
            f"old_string is not unique in {file_path!r}: found {count} matches. "
            "Provide more context to make it unique, or set replace_all=True."
        )

    new_content = content.replace(old_string, new_string)
    _write_text(p, new_content)
    return f"Replaced {count if replace_all else 1} occurrence(s) in {file_path}"


# ---------------------------------------------------------------------------
# cc_write
# ---------------------------------------------------------------------------

@tool
def cc_write(file_path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed.

    Use when: creating a new file from scratch. For existing files, prefer cc_edit.
    """
    p = Path(file_path)
    _write_text(p, content)
    return f"Written {len(content)} bytes to {file_path}"


# ---------------------------------------------------------------------------
# cc_multi_edit
# ---------------------------------------------------------------------------

@tool
def cc_multi_edit(file_path: str, edits: list[dict[str, Any]]) -> str:
    """Apply multiple string replacements to a file in a single call.

    Use when: making >=2 edits to the same file. Each edit sees the result of
    the previous one (sequential, not parallel).
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = _read_text(p)

    for idx, edit in enumerate(edits):
        old_string: str = edit["old_string"]
        new_string: str = edit["new_string"]
        replace_all: bool = edit.get("replace_all", False)

        count = content.count(old_string)
        if count == 0:
            raise ValueError(
                f"Edit #{idx}: old_string not found in {file_path!r}"
            )
        if count > 1 and not replace_all:
            raise ValueError(
                f"Edit #{idx}: old_string is not unique in {file_path!r}: "
                f"found {count} matches. Set replace_all=True or use more context."
            )

        content = content.replace(old_string, new_string)

    _write_text(p, content)
    return f"Applied {len(edits)} edit(s) to {file_path}"


# ---------------------------------------------------------------------------
# GBNF Grammars
# ---------------------------------------------------------------------------

_CC_EDIT_GRAMMAR = r"""root      ::= tool-call
tool-call ::= "{\"tool_call\": {\"name\": \"cc_edit\", \"arguments\": " args "}}"
args      ::= "{\"file_path\": " string ", \"old_string\": " string ", \"new_string\": " string opt-replace "}"
opt-replace ::= "" | ", \"replace_all\": " bool
bool      ::= "true" | "false"
string    ::= "\"" char* "\""
char      ::= [^"\\] | "\\" escape
escape    ::= ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
"""

_CC_MULTI_EDIT_GRAMMAR = r"""root      ::= tool-call
tool-call ::= "{\"tool_call\": {\"name\": \"cc_multi_edit\", \"arguments\": " args "}}"
args      ::= "{\"file_path\": " string ", \"edits\": [" edit ("," edit)* "]}"
edit      ::= "{\"old_string\": " string ", \"new_string\": " string opt-replace "}"
opt-replace ::= "" | ", \"replace_all\": " bool
bool      ::= "true" | "false"
string    ::= "\"" char* "\""
char      ::= [^"\\] | "\\" escape
escape    ::= ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
"""

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__tools__ = [cc_glob, cc_grep, cc_read, cc_edit, cc_write, cc_multi_edit]

__grammars__: dict[str, str] = {
    "cc_edit": _CC_EDIT_GRAMMAR,
    "cc_multi_edit": _CC_MULTI_EDIT_GRAMMAR,
}
