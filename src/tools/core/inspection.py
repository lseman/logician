"""Read-only code inspection helpers for core tools."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from .filesystem import DEFAULT_FILESYSTEM


def _err(message: str) -> dict[str, Any]:
    return {"status": "error", "error": message}


def _read_text_preserve_newlines(path: Path) -> str:
    return DEFAULT_FILESYSTEM.read_text(path, encoding="utf-8", errors="replace")


def search_file(
    path: str,
    pattern: str,
    literal: bool = True,
    case_sensitive: bool = True,
    context_lines: int = 2,
) -> dict[str, Any]:
    """Search for text or regex matches in a single file."""
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    try:
        text = _read_text_preserve_newlines(p)
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")

    lines = text.splitlines()
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        rx = re.compile(re.escape(pattern) if literal else pattern, flags | re.MULTILINE)
    except re.error as exc:
        return _err(f"Invalid regex: {exc}")

    matches: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not rx.search(line):
            continue
        ctx_start = max(0, index - context_lines)
        ctx_end = min(len(lines), index + context_lines + 1)
        matches.append(
            {
                "line_number": index + 1,
                "line_content": line,
                "context_before": lines[ctx_start:index],
                "context_after": lines[index + 1 : ctx_end],
            }
        )
        if len(matches) >= 50:
            break

    return {
        "status": "ok",
        "path": str(p),
        "total_matches": len(matches),
        "matches": matches,
    }


def get_symbol_info(path: str, symbol: str) -> dict[str, Any]:
    """Find a function or class definition inside a file."""
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    try:
        content = _read_text_preserve_newlines(p)
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")

    lines = content.splitlines()
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == symbol
            ):
                start = node.lineno
                end = node.end_lineno or start
                definition = lines[start - 1] if 0 < start <= len(lines) else ""
                indent = len(definition) - len(definition.lstrip())
                return {
                    "status": "ok",
                    "path": str(p),
                    "symbol": symbol,
                    "type": "class" if isinstance(node, ast.ClassDef) else "function",
                    "location": {"start_line": start, "end_line": end},
                    "indentation": indent,
                    "definition": definition,
                }
    except SyntaxError:
        pass

    rx = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b")
    for index, line in enumerate(lines, 1):
        if not rx.match(line):
            continue
        indent = len(line) - len(line.lstrip())
        return {
            "status": "ok",
            "path": str(p),
            "symbol": symbol,
            "type": "function" if "def" in line else "class",
            "location": {"start_line": index, "end_line": index},
            "indentation": indent,
            "definition": line,
        }

    return _err(f"Symbol '{symbol}' not found in {path}")


def read_line(path: str, line_number: int) -> dict[str, Any]:
    """Read one specific line with small metadata."""
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    try:
        lines = _read_text_preserve_newlines(p).splitlines(keepends=True)
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")

    if line_number < 1 or line_number > len(lines):
        return _err(f"Line {line_number} out of range (1-{len(lines)})")

    line = lines[line_number - 1]
    stripped = line.lstrip()
    return {
        "status": "ok",
        "path": str(p),
        "line_number": line_number,
        "total_lines": len(lines),
        "content": line.rstrip("\r\n"),
        "indentation": len(line) - len(stripped),
        "is_empty": not stripped.strip(),
        "is_comment": stripped.startswith(("#", "//", "<!--", "*")),
    }


def find_imports(path: str) -> dict[str, Any]:
    """List import statements in a Python-like file."""
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    try:
        lines = _read_text_preserve_newlines(p).splitlines()
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")

    imports = []
    for index, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped.startswith(("import ", "from ")):
            continue
        imports.append(
            {
                "line": index,
                "raw": line.rstrip(),
                "type": "from" if stripped.startswith("from ") else "import",
            }
        )

    return {"status": "ok", "path": str(p), "imports": imports, "total": len(imports)}

