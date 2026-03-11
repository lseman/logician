# cc_tools Skill Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `skills/coding/cc_tools/` skill with 6 Claude Code-parity tools (cc_glob, cc_grep, cc_read, cc_edit, cc_write, cc_multi_edit) plus llama.cpp grammar enforcement for edit tools.

**Architecture:** New skill folder with `SKILL.md` + `scripts/tools.py`. Grammar collection added to the registry loading path via `__grammars__` module export. Agent core selects grammar before LLM call when predicted tool has a registered grammar.

**Tech Stack:** Python 3.10+, pathlib, subprocess (rg), `src/tools/__init__.py`, `src/tools/registry/loading.py`, `src/tools/registry/introspection.py`, `src/agent/core.py`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `skills/coding/cc_tools/SKILL.md` | Create | Skill metadata + workflow guide |
| `skills/coding/cc_tools/scripts/tools.py` | Create | 6 tool implementations + `__grammars__` |
| `src/tools/__init__.py` | Modify (line ~115) | Add `self._grammars: dict[str, str] = {}` |
| `src/tools/registry/loading.py` | Modify (line ~531) | Collect `__grammars__` after module exec |
| `src/tools/registry/introspection.py` | Modify (line ~100) | Add `get_grammar(tool_name)` method; register `cc_tools` in `_CODING_SKILL_IDS` |
| `src/agent/core.py` | Modify (line ~3082) | Select grammar before LLM call |
| `test/test_cc_tools.py` | Create | Tests for all 6 tools + grammar hook |

---

## Chunk 1: Skill Files

### Task 1: Create `SKILL.md`

**Files:**
- Create: `skills/coding/cc_tools/SKILL.md`

- [ ] **Step 1: Write SKILL.md**

```markdown
---
name: cc_tools
description: >
  Precise file exploration and editing for coding tasks. Use for finding files,
  reading code with line ranges, and making surgical targeted edits. Preferred
  over explore/file_ops for any new coding work.
aliases:
  - find files
  - read file
  - edit file
  - search code
  - navigate codebase
  - glob
  - grep
triggers:
  - look for
  - find where
  - read the
  - edit
  - change
  - fix
  - modify
  - search for
preferred_tools:
  - cc_glob
  - cc_grep
  - cc_read
  - cc_edit
  - cc_multi_edit
when_not_to_use:
  - shell execution (use shell skill)
  - git operations (use git skill)
  - running tests (use quality skill)
next_skills:
  - shell
  - git
  - quality
---

## Workflow

Always follow this sequence — it mirrors how Claude Code works:

1. **Find** — use `cc_glob` for file patterns (`**/*.py`), `cc_grep output_mode=files_with_matches` to find which files contain a pattern
2. **Read** — use `cc_read offset=N limit=50` to read only the relevant lines; never read an entire large file
3. **Edit** — use `cc_edit` for one surgical change; `cc_multi_edit` for ≥2 changes in the same file; `cc_write` only for new files

## Rules

- Never edit a file you have not read in the current session
- `cc_edit` requires `old_string` to be unique in the file — include enough surrounding lines to make it unique
- For multiple edits in one file, always use `cc_multi_edit` (batched = fewer LLM round-trips)
- Use `cc_grep output_mode=files_with_matches` first; switch to `content` only when you need to see matching lines
- Use `cc_read` with `offset` and `limit` when you know approximately where in the file the relevant code lives

## Tool Quick Reference

| Tool | Use for |
|------|---------|
| `cc_glob` | Find files by name pattern (e.g. `**/*.py`, `src/**/config.*`) |
| `cc_grep` | Find files or lines matching a regex (content search) |
| `cc_read` | Read a file, optionally starting at line N for M lines |
| `cc_edit` | Replace one exact string in a file (surgical edit) |
| `cc_write` | Write a new file from scratch (or full overwrite) |
| `cc_multi_edit` | Apply multiple replacements to one file in a single call |
```

- [ ] **Step 2: Verify file exists**

```bash
ls -la skills/coding/cc_tools/SKILL.md
```

Expected: file present, non-empty.

---

### Task 2: Write `tools.py`

**Files:**
- Create: `skills/coding/cc_tools/scripts/tools.py`

- [ ] **Step 1: Write failing test first**

Create `test/test_cc_tools.py`:

```python
"""Tests for cc_tools skill — cc_glob, cc_grep, cc_read, cc_edit, cc_write, cc_multi_edit."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers to load skill module directly (no registry needed)
# ---------------------------------------------------------------------------
SKILL_MODULE = Path(__file__).resolve().parents[1] / "skills/coding/cc_tools/scripts/tools.py"


def _load_tools_module() -> dict:
    """Execute tools.py in an isolated namespace and return its globals."""
    ns: dict = {}
    exec(SKILL_MODULE.read_text(), ns, ns)  # noqa: S102
    return ns


@pytest.fixture(scope="module")
def tools_ns() -> dict:
    return _load_tools_module()


# ---------------------------------------------------------------------------
# cc_glob
# ---------------------------------------------------------------------------

def test_cc_glob_returns_matching_files(tools_ns, tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    result = tools_ns["cc_glob"]("*.py", path=str(tmp_path))
    assert "a.py" in result
    assert "b.txt" not in result


def test_cc_glob_head_limit(tools_ns, tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("")
    result = tools_ns["cc_glob"]("*.py", path=str(tmp_path), head_limit=2)
    assert result.count(".py") == 2


def test_cc_glob_no_match(tools_ns, tmp_path):
    result = tools_ns["cc_glob"]("*.xyz", path=str(tmp_path))
    assert "(no matches)" in result


# ---------------------------------------------------------------------------
# cc_grep
# ---------------------------------------------------------------------------

def test_cc_grep_files_with_matches(tools_ns, tmp_path):
    (tmp_path / "yes.py").write_text("hello world\n")
    (tmp_path / "no.py").write_text("nothing here\n")
    result = tools_ns["cc_grep"]("hello", path=str(tmp_path), output_mode="files_with_matches")
    assert "yes.py" in result
    assert "no.py" not in result


def test_cc_grep_content_mode(tools_ns, tmp_path):
    (tmp_path / "f.py").write_text("line1\nhello\nline3\n")
    result = tools_ns["cc_grep"]("hello", path=str(tmp_path), output_mode="content")
    assert "hello" in result


def test_cc_grep_count_mode(tools_ns, tmp_path):
    (tmp_path / "f.py").write_text("x\nx\nx\n")
    result = tools_ns["cc_grep"]("x", path=str(tmp_path), output_mode="count")
    assert "3" in result


def test_cc_grep_no_match_returns_empty(tools_ns, tmp_path):
    (tmp_path / "f.py").write_text("nothing\n")
    result = tools_ns["cc_grep"]("zzznomatch", path=str(tmp_path))
    assert "(no matches)" in result or result.strip() == ""


# ---------------------------------------------------------------------------
# cc_read
# ---------------------------------------------------------------------------

def test_cc_read_full_file(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("line1\nline2\nline3\n")
    result = tools_ns["cc_read"](str(f))
    assert "     1\tline1" in result
    assert "     3\tline3" in result


def test_cc_read_with_offset_and_limit(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("\n".join(f"L{i}" for i in range(10)))
    result = tools_ns["cc_read"](str(f), offset=2, limit=3)
    assert "L2" in result
    assert "L5" not in result


def test_cc_read_truncates_long_lines(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("A" * 3000 + "\n")
    result = tools_ns["cc_read"](str(f))
    # Should be at most 2000 chars for that line
    line_content = result.split("\t", 1)[1]
    assert len(line_content) <= 2000


# ---------------------------------------------------------------------------
# cc_edit
# ---------------------------------------------------------------------------

def test_cc_edit_basic_replacement(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("hello world\n")
    tools_ns["cc_edit"](str(f), "hello", "goodbye")
    assert f.read_text() == "goodbye world\n"


def test_cc_edit_not_found_raises(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("hello world\n")
    with pytest.raises(Exception, match="not found"):
        tools_ns["cc_edit"](str(f), "zzzmissing", "x")


def test_cc_edit_not_unique_raises(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("x\nx\nx\n")
    with pytest.raises(Exception, match="not unique|matches"):
        tools_ns["cc_edit"](str(f), "x", "y")


def test_cc_edit_replace_all(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("x\nx\nx\n")
    tools_ns["cc_edit"](str(f), "x", "y", replace_all=True)
    assert f.read_text() == "y\ny\ny\n"


# ---------------------------------------------------------------------------
# cc_write
# ---------------------------------------------------------------------------

def test_cc_write_creates_file(tools_ns, tmp_path):
    f = tmp_path / "new.py"
    tools_ns["cc_write"](str(f), "print('hello')\n")
    assert f.read_text() == "print('hello')\n"


def test_cc_write_creates_parent_dirs(tools_ns, tmp_path):
    f = tmp_path / "sub" / "dir" / "new.py"
    tools_ns["cc_write"](str(f), "x\n")
    assert f.read_text() == "x\n"


# ---------------------------------------------------------------------------
# cc_multi_edit
# ---------------------------------------------------------------------------

def test_cc_multi_edit_sequential(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("alpha beta gamma\n")
    tools_ns["cc_multi_edit"](str(f), [
        {"old_string": "alpha", "new_string": "A"},
        {"old_string": "beta", "new_string": "B"},
    ])
    assert f.read_text() == "A B gamma\n"


def test_cc_multi_edit_second_sees_first_result(tools_ns, tmp_path):
    """Each edit sees the result of the previous — sequential, not parallel."""
    f = tmp_path / "f.py"
    f.write_text("foo\n")
    tools_ns["cc_multi_edit"](str(f), [
        {"old_string": "foo", "new_string": "bar"},
        {"old_string": "bar", "new_string": "baz"},
    ])
    assert f.read_text() == "baz\n"


# ---------------------------------------------------------------------------
# __grammars__ export
# ---------------------------------------------------------------------------

def test_grammars_exported(tools_ns):
    grammars = tools_ns.get("__grammars__", {})
    assert isinstance(grammars, dict)
    assert "cc_edit" in grammars
    assert "cc_multi_edit" in grammars
    for v in grammars.values():
        assert isinstance(v, str) and len(v) > 20


# ---------------------------------------------------------------------------
# __tools__ export
# ---------------------------------------------------------------------------

def test_tools_exported(tools_ns):
    tools = tools_ns.get("__tools__", [])
    names = [getattr(t, "__name__", None) for t in tools]
    assert "cc_glob" in names
    assert "cc_grep" in names
    assert "cc_read" in names
    assert "cc_edit" in names
    assert "cc_write" in names
    assert "cc_multi_edit" in names
```

- [ ] **Step 2: Run test to confirm all fail**

```bash
pytest test/test_cc_tools.py -v 2>&1 | head -40
```

Expected: All tests fail with `FileNotFoundError` or `ModuleNotFoundError`.

- [ ] **Step 3: Write `tools.py`**

Create `skills/coding/cc_tools/scripts/tools.py`:

```python
"""cc_tools — Claude Code-parity file exploration and editing tools.

Six tools mirroring Claude Code's exact contracts:
  cc_glob       — pattern-based file finder, sorted by mtime
  cc_grep       — ripgrep content search with output_mode dispatch
  cc_read       — line-range file reader in cat-n format
  cc_edit       — uniqueness-gated string replacement
  cc_write      — full file write (new files only)
  cc_multi_edit — sequential multi-replacement for one file
"""
from __future__ import annotations

import json as _json
import shutil
import subprocess
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# GBNF grammars for llama.cpp constrained decoding
# ---------------------------------------------------------------------------

_CC_EDIT_GRAMMAR = r"""
root      ::= tool-call
tool-call ::= "{\"tool_call\": {\"name\": \"cc_edit\", \"arguments\": " args "}}"
args      ::= "{\"file_path\": " string ", \"old_string\": " string ", \"new_string\": " string opt-replace "}"
opt-replace ::= "" | ", \"replace_all\": " bool
bool      ::= "true" | "false"
string    ::= "\"" char* "\""
char      ::= [^"\\] | "\\" escape
escape    ::= ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
"""

_CC_MULTI_EDIT_GRAMMAR = r"""
root      ::= tool-call
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
# Tool implementations
# ---------------------------------------------------------------------------

class _ToolError(RuntimeError):
    """Raised when a tool contract is violated."""


def cc_glob(pattern: str, path: str = ".", head_limit: int = 0) -> str:
    """Find files matching a glob pattern, sorted by modification time (newest first).

    Use when: searching for files by name pattern (e.g. **/*.py, src/**/config.*).

    Parameters:
        pattern: Glob pattern relative to path (e.g. '**/*.py').
        path: Root directory to search from (default: current directory).
        head_limit: Maximum number of results to return (0 = unlimited).
    """
    root = Path(path).expanduser().resolve()
    try:
        matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception as exc:
        raise _ToolError(f"cc_glob error: {exc}") from exc
    if head_limit and head_limit > 0:
        matches = matches[:head_limit]
    if not matches:
        return "(no matches)"
    return "\n".join(str(p) for p in matches)


def cc_grep(
    pattern: str,
    path: str = ".",
    glob: str = "",
    type: str = "",
    output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
    context: int = 0,
    case_insensitive: bool = False,
    head_limit: int = 0,
    offset: int = 0,
    multiline: bool = False,
) -> str:
    """Search file contents using ripgrep (falls back to pure-Python grep).

    Use when: finding which files contain a pattern, or viewing matching lines.

    Parameters:
        pattern: Regular expression to search for.
        path: File or directory to search (default: current directory).
        glob: File glob filter (e.g. '*.py', '**/*.ts').
        type: Ripgrep file type filter (e.g. 'py', 'rust', 'js').
        output_mode: 'files_with_matches' (default) | 'content' | 'count'.
        context: Lines of context to show before and after each match (content mode).
        case_insensitive: Case-insensitive search.
        head_limit: Maximum results to return (0 = unlimited).
        offset: Skip first N results before applying head_limit.
        multiline: Enable multiline matching (. matches newlines).
    """
    if shutil.which("rg"):
        return _cc_grep_rg(
            pattern=pattern,
            path=path,
            glob=glob,
            file_type=type,
            output_mode=output_mode,
            context=context,
            case_insensitive=case_insensitive,
            head_limit=head_limit,
            offset=offset,
            multiline=multiline,
        )
    return _cc_grep_fallback(
        pattern=pattern,
        path=path,
        glob=glob,
        output_mode=output_mode,
        case_insensitive=case_insensitive,
        head_limit=head_limit,
        offset=offset,
    )


def _cc_grep_rg(
    *,
    pattern: str,
    path: str,
    glob: str,
    file_type: str,
    output_mode: str,
    context: int,
    case_insensitive: bool,
    head_limit: int,
    offset: int,
    multiline: bool,
) -> str:
    argv = ["rg"]
    if output_mode == "files_with_matches":
        argv.append("-l")
    elif output_mode == "count":
        argv.append("-c")
    else:
        if context > 0:
            argv += ["-C", str(int(context))]
        argv.append("-n")
    if case_insensitive:
        argv.append("-i")
    if multiline:
        argv += ["-U", "--multiline-dotall"]
    if glob:
        argv += ["--glob", glob]
    if file_type:
        argv += ["--type", file_type]
    argv += [pattern, path]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        raw = proc.stdout
    except Exception as exc:
        raise _ToolError(f"rg error: {exc}") from exc
    return _apply_head_offset(raw.strip(), head_limit=head_limit, offset=offset) or "(no matches)"


def _cc_grep_fallback(
    *,
    pattern: str,
    path: str,
    glob: str,
    output_mode: str,
    case_insensitive: bool,
    head_limit: int,
    offset: int,
) -> str:
    import re

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        raise _ToolError(f"invalid regex: {exc}") from exc

    root = Path(path)
    file_glob = glob or "**/*"
    candidates = [p for p in root.glob(file_glob) if p.is_file()]

    results: list[str] = []
    for candidate in candidates:
        try:
            text = candidate.read_text(errors="replace")
        except Exception:
            continue
        if output_mode == "files_with_matches":
            if rx.search(text):
                results.append(str(candidate))
        elif output_mode == "count":
            count = len(rx.findall(text))
            if count:
                results.append(f"{candidate}:{count}")
        else:
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    results.append(f"{candidate}:{i}:{line}")

    return _apply_head_offset("\n".join(results), head_limit=head_limit, offset=offset) or "(no matches)"


def _apply_head_offset(text: str, *, head_limit: int, offset: int) -> str:
    if not text:
        return text
    lines = text.splitlines()
    if offset:
        lines = lines[offset:]
    if head_limit and head_limit > 0:
        lines = lines[:head_limit]
    return "\n".join(lines)


def cc_read(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file, returning lines in cat-n format with line numbers.

    Use when: reading a file or a specific section of a file.

    Parameters:
        file_path: Absolute or relative path to the file.
        offset: 0-indexed line number to start reading from (default: 0).
        limit: Maximum number of lines to return (default: 2000).
    """
    p = Path(file_path).expanduser().resolve()
    try:
        text = p.read_text(errors="replace")
    except FileNotFoundError:
        raise _ToolError(f"File not found: {file_path}")
    except Exception as exc:
        raise _ToolError(f"cc_read error: {exc}") from exc

    lines = text.splitlines()
    window = lines[offset: offset + max(1, int(limit))]
    return "\n".join(
        f"{offset + i + 1:6}\t{line[:2000]}"
        for i, line in enumerate(window)
    )


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

    Parameters:
        file_path: Path to the file to edit.
        old_string: Exact string to find and replace. Must be unique in the file.
        new_string: Replacement string.
        replace_all: If True, replace all occurrences (skips uniqueness check).
    """
    p = Path(file_path).expanduser().resolve()
    try:
        content = p.read_text()
    except FileNotFoundError:
        raise _ToolError(f"File not found: {file_path}")
    except Exception as exc:
        raise _ToolError(f"cc_edit read error: {exc}") from exc

    count = content.count(old_string)
    if count == 0:
        raise _ToolError(
            f"cc_edit: old_string not found in {file_path}. "
            "Check for whitespace differences or use cc_read to inspect the file."
        )
    if count > 1 and not replace_all:
        raise _ToolError(
            f"cc_edit: old_string matches {count} locations in {file_path}. "
            "Add more surrounding context to make it unique, or use replace_all=true."
        )

    if replace_all:
        new_content = content.replace(old_string, new_string)
        replaced = count
    else:
        new_content = content.replace(old_string, new_string, 1)
        replaced = 1

    try:
        p.write_text(new_content)
    except Exception as exc:
        raise _ToolError(f"cc_edit write error: {exc}") from exc

    noun = "replacement" if replaced == 1 else "replacements"
    return f"Edited {file_path} ({replaced} {noun})"


def cc_write(file_path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed.

    Use when: creating a new file from scratch. For existing files, prefer cc_edit.

    Parameters:
        file_path: Path to write to (will be created or overwritten).
        content: Full file content to write.
    """
    p = Path(file_path).expanduser().resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    except Exception as exc:
        raise _ToolError(f"cc_write error: {exc}") from exc
    lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"Wrote {file_path} ({lines} lines)"


def cc_multi_edit(file_path: str, edits: list[dict]) -> str:
    """Apply multiple string replacements to a file in a single call.

    Use when: making ≥2 edits to the same file. Each edit sees the result of
    the previous one (sequential, not parallel).

    Parameters:
        file_path: Path to the file to edit.
        edits: List of {old_string, new_string, replace_all?} dicts applied in order.
    """
    if not edits:
        raise _ToolError("cc_multi_edit: edits list is empty")
    results: list[str] = []
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise _ToolError(f"cc_multi_edit: edit[{i}] must be a dict")
        old = edit.get("old_string")
        new = edit.get("new_string")
        if old is None or new is None:
            raise _ToolError(f"cc_multi_edit: edit[{i}] missing old_string or new_string")
        ra = bool(edit.get("replace_all", False))
        results.append(cc_edit(file_path, str(old), str(new), replace_all=ra))
    return "\n".join(results)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__tools__ = [cc_glob, cc_grep, cc_read, cc_edit, cc_write, cc_multi_edit]

__grammars__: dict[str, str] = {
    "cc_edit": _CC_EDIT_GRAMMAR,
    "cc_multi_edit": _CC_MULTI_EDIT_GRAMMAR,
}

__skill__ = {
    "name": "cc_tools",
    "description": (
        "Precise file exploration and editing. Use for finding files, reading code, "
        "and making targeted edits. Preferred over legacy explore/file_ops."
    ),
    "aliases": ["find files", "read file", "edit file", "search code", "glob", "grep"],
    "triggers": ["find", "read the", "edit", "change", "fix", "modify", "search for"],
    "preferred_tools": ["cc_glob", "cc_grep", "cc_read", "cc_edit", "cc_multi_edit"],
    "when_not_to_use": ["shell execution", "git operations", "running tests"],
    "next_skills": ["shell", "git", "quality"],
}
```

- [ ] **Step 4: Run tests — expect most to pass**

```bash
pytest test/test_cc_tools.py -v
```

Expected: All tests PASS. If `rg` is not installed, grep tests using rg will fall back to Python — still pass.

- [ ] **Step 5: Commit**

```bash
git add skills/coding/cc_tools/ test/test_cc_tools.py
git commit -m "feat: add cc_tools skill with 6 Claude Code-parity tools"
```

---

## Chunk 2: Registry Grammar Hook

### Task 3: Add `self._grammars` to `ToolRegistry.__init__`

**Files:**
- Modify: `src/tools/__init__.py` around line 115

- [ ] **Step 1: Write failing test**

Add to `test/test_cc_tools.py`:

```python
# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_registry_collects_grammars():
    """ToolRegistry collects __grammars__ from skill modules."""
    from src.tools import ToolRegistry
    registry = ToolRegistry(auto_load_from_skills=True)
    grammar = registry.get_grammar("cc_edit")
    assert grammar is not None
    assert "cc_edit" in grammar or "tool-call" in grammar


def test_registry_get_grammar_unknown_returns_none():
    from src.tools import ToolRegistry
    registry = ToolRegistry(auto_load_from_skills=False)
    assert registry.get_grammar("nonexistent_tool_xyz") is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest test/test_cc_tools.py::test_registry_collects_grammars test/test_cc_tools.py::test_registry_get_grammar_unknown_returns_none -v
```

Expected: `AttributeError: 'ToolRegistry' object has no attribute 'get_grammar'`

- [ ] **Step 3: Add `self._grammars` to `ToolRegistry.__init__`**

In `src/tools/__init__.py`, find the block initializing instance variables (around line 110-118). Add one line after `self._tool_exec_stats`:

```python
# Find this line:
        self._tool_exec_stats: ToolExecutionStats = {}
# Add immediately after:
        self._grammars: dict[str, str] = {}
```

- [ ] **Step 4: Collect `__grammars__` in `_register_tools_from_python_module`**

In `src/tools/registry/loading.py`, find the end of `_register_tools_from_python_module` (around line 531-534). After `self._merge_python_module_globals(execution_globals)`, add:

```python
        # Collect GBNF grammars exported by skill modules (for llama.cpp constrained decoding)
        grammars_raw = execution_globals.get("__grammars__", {})
        if isinstance(grammars_raw, dict):
            for tool_name, grammar in grammars_raw.items():
                if isinstance(tool_name, str) and isinstance(grammar, str):
                    self._grammars[tool_name] = grammar
```

The full block should look like:

```python
        self._merge_python_module_globals(execution_globals)
        # Collect GBNF grammars exported by skill modules (for llama.cpp constrained decoding)
        grammars_raw = execution_globals.get("__grammars__", {})
        if isinstance(grammars_raw, dict):
            for tool_name, grammar in grammars_raw.items():
                if isinstance(tool_name, str) and isinstance(grammar, str):
                    self._grammars[tool_name] = grammar
        if registered > 0:
            self._invalidate_skill_resolution_cache()
        return registered
```

- [ ] **Step 5: Add `get_grammar` to `RegistryIntrospectionMixin`**

In `src/tools/registry/introspection.py`, add after the `_skills_health_tool` method (around line 652) or just before `_coding_capability_audit`:

```python
    def get_grammar(self, tool_name: str) -> str | None:
        """Return the registered GBNF grammar for a tool, or None if not registered.

        Used by Agent.run() to enable llama.cpp constrained decoding for tools
        that export __grammars__ (e.g. cc_edit, cc_multi_edit).
        """
        return self._grammars.get(str(tool_name or "").strip())
```

- [ ] **Step 6: Run registry tests**

```bash
pytest test/test_cc_tools.py::test_registry_collects_grammars test/test_cc_tools.py::test_registry_get_grammar_unknown_returns_none -v
```

Expected: Both PASS.

- [ ] **Step 7: Register `cc_tools` in `_CODING_SKILL_IDS`**

In `src/tools/registry/introspection.py`, find `_CODING_SKILL_IDS` (around line 78). Add `"cc_tools"`:

```python
_CODING_SKILL_IDS = {
    "cc_tools",          # <-- add this line
    "file_ops",
    "multi_edit",
    # ... rest unchanged
}
```

- [ ] **Step 8: Add cc_tools to `_CODING_CAPABILITY_GROUPS` nice_to_have**

In `src/tools/registry/introspection.py`, in `_CODING_CAPABILITY_GROUPS`:

```python
# Under "discovery" → "nice_to_have", add:
    "cc_glob",
    "cc_grep",
    "cc_read",

# Under "editing" → "nice_to_have", add:
    "cc_edit",
    "cc_write",
    "cc_multi_edit",
```

- [ ] **Step 9: Commit**

```bash
git add src/tools/__init__.py src/tools/registry/loading.py src/tools/registry/introspection.py
git commit -m "feat: add grammar registry hook (__grammars__ collection + get_grammar)"
```

---

## Chunk 3: Agent Core Grammar Injection

### Task 4: Use grammar in `Agent.run()` LLM call

**Files:**
- Modify: `src/agent/core.py` around line 3080-3110

- [ ] **Step 1: Write failing test**

Add to `test/test_cc_tools.py`:

```python
def test_agent_selects_grammar_for_cc_edit(tmp_path):
    """Agent._select_grammar_for_tool returns grammar when tool has one registered."""
    from src.tools import ToolRegistry
    registry = ToolRegistry(auto_load_from_skills=True)
    grammar = registry.get_grammar("cc_edit")
    assert grammar is not None, "cc_edit grammar must be registered after skill load"
    assert len(grammar) > 50, "grammar should be a non-trivial GBNF string"
```

- [ ] **Step 2: Run to confirm it passes** (this one should already pass after Task 3)

```bash
pytest test/test_cc_tools.py::test_agent_selects_grammar_for_cc_edit -v
```

Expected: PASS — no agent code change needed for this test.

- [ ] **Step 3: Inject grammar into `_llm_generate` call in `Agent.run()`**

In `src/agent/core.py`, find the block around line 3077-3111 where `_tools_payload` is assembled and `_llm_generate` is called.

Find this pattern:
```python
            _tools_payload: list | None = None
            if (
                bool(getattr(self.config, "constrained_decoding", False))
                and self.config.use_chat_api
            ):
```

Add grammar selection immediately before `_llm_generate`:

```python
            # Grammar-constrained decoding: if the last tool call was a grammar-registered
            # tool (e.g. cc_edit), force the grammar on the NEXT generation so the model
            # produces a well-formed tool call. Only used when NOT in chat-api/tools mode.
            _grammar: str | None = None
            if (
                not _tools_payload  # grammar and tools payload are mutually exclusive
                and not self.config.use_chat_api
            ):
                _last_tool = getattr(self, "_last_tool_name", None)
                if _last_tool:
                    _grammar = self.tools.get_grammar(_last_tool)

            gen_start = time.perf_counter()
            try:
                text = self._llm_generate(
                    llm_convo,
                    temperature=temp,
                    max_tokens=n_tok,
                    stream=_do_stream,
                    on_token=stream_callback,
                    tools=_tools_payload,
                    grammar=_grammar,      # <-- new kwarg
                )
```

- [ ] **Step 4: Track `_last_tool_name` after each tool execution**

In `src/agent/core.py`, find where tool execution results are processed (search for `tool_result` or `execute(tool_call)`). After a successful tool execution, add:

```python
                self._last_tool_name = tool_call.name
```

Reset at the start of each `run()` call (before the main loop begins):

```python
            self._last_tool_name: str | None = None
```

- [ ] **Step 5: Verify `_llm_generate` accepts `grammar` kwarg**

Check the signature of `_llm_generate` in `src/agent/core.py` (search `def _llm_generate`). If it doesn't accept `grammar`, add it:

```python
    def _llm_generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
        tools: list | None = None,
        grammar: str | None = None,   # <-- add if missing
    ) -> str:
```

And forward `grammar` to the backend call. The llama.cpp backend already accepts a `grammar` field in its `/completion` endpoint request body — confirm this in `src/backends/llama_cpp.py` if needed.

- [ ] **Step 6: Run full test suite to check for regressions**

```bash
pytest test/ -x -q 2>&1 | tail -20
```

Expected: All existing tests pass; new cc_tools tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/agent/core.py
git commit -m "feat: inject GBNF grammar into llm call for grammar-registered tools (cc_edit, cc_multi_edit)"
```

---

## Chunk 4: Final Verification

### Task 5: Smoke test the full skill loading

- [ ] **Step 1: Run skills_health check**

```python
from src.tools import ToolRegistry
r = ToolRegistry(auto_load_from_skills=True)
import json
print(json.dumps(json.loads(r._skills_health_tool()), indent=2))
```

Expected output should include:
- `cc_tools` in `coding_skill_ids`
- `cc_glob`, `cc_grep`, `cc_read`, `cc_edit`, `cc_write`, `cc_multi_edit` in tool list

- [ ] **Step 2: Verify grammar is present**

```python
print(r.get_grammar("cc_edit"))
print(r.get_grammar("cc_multi_edit"))
```

Expected: Non-None GBNF strings.

- [ ] **Step 3: Run full test suite**

```bash
pytest test/ -q 2>&1 | tail -10
```

Expected: All pass.

- [ ] **Step 4: Final commit**

```bash
git add -p  # review any remaining unstaged changes
git commit -m "feat: cc_tools skill complete — 6 CC-parity tools with grammar enforcement"
```
