"""Read-only code inspection helpers for core tools."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from ..FileReadTool.state import record_file_snapshot, resolve_tool_path
from ..filesystem import DEFAULT_FILESYSTEM

DEFAULT_EDIT_CONTEXT_SCAN_BYTES = 10 * 1024 * 1024
DEFAULT_CODE_SEARCH_MAX_RESULTS = 50


def _err(message: str) -> dict[str, Any]:
    return {"status": "error", "error": message}


def _read_text_preserve_newlines(path: Path) -> str:
    return DEFAULT_FILESYSTEM.read_text(path, encoding="utf-8", errors="replace")


def search_code(
    query: str,
    path: str = ".",
    glob: str = "**/*",
    mode: str = "literal",
    case_sensitive: bool = True,
    context_lines: int = 2,
    max_results: int = DEFAULT_CODE_SEARCH_MAX_RESULTS,
    include_hidden: bool = False,
    offset: int = 0,
) -> dict[str, Any]:
    """Search code across a tree using multiline text or Python symbol matching."""
    if not query:
        return _err("query must not be empty")
    if context_lines < 0:
        return _err("context_lines must be >= 0")
    if max_results <= 0:
        return _err("max_results must be >= 1")
    if offset < 0:
        return _err("offset must be >= 0")

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"literal", "regex", "symbol"}:
        return _err("mode must be one of: literal, regex, symbol")

    try:
        root = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))
    if not root.exists():
        return _err(f"Path not found: {path}")

    if normalized_mode == "symbol":
        return _search_python_symbols(
            query,
            root=root,
            glob=glob,
            case_sensitive=case_sensitive,
            max_results=max_results,
            include_hidden=include_hidden,
            offset=offset,
        )

    return _search_multiline_code(
        query,
        root=root,
        glob=glob,
        literal=normalized_mode == "literal",
        case_sensitive=case_sensitive,
        context_lines=context_lines,
        max_results=max_results,
        include_hidden=include_hidden,
        offset=offset,
    )


def read_edit_context(
    path: str,
    needle: str,
    context_lines: int = 3,
    max_scan_bytes: int = DEFAULT_EDIT_CONTEXT_SCAN_BYTES,
) -> dict[str, Any]:
    """Return a bounded context window around a matching needle."""
    if not needle:
        return _err("needle must not be empty")
    if context_lines < 0:
        return _err("context_lines must be >= 0")
    if max_scan_bytes <= 0:
        return _err("max_scan_bytes must be > 0")

    try:
        p = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    try:
        stat_result = p.stat()
    except OSError as exc:
        return _err(f"Cannot stat file: {exc}")

    scan_limit = max(max_scan_bytes, len(needle.encode("utf-8", errors="ignore")) + 1)
    try:
        with open(p, "rb") as fh:
            raw = fh.read(scan_limit + 1)
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")

    truncated_scan = len(raw) > scan_limit
    payload = raw[:scan_limit] if truncated_scan else raw
    if DEFAULT_FILESYSTEM._is_probably_binary(payload):
        return _err(f"Binary files are not supported for read_edit_context: {path}")

    text = payload.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"):
        text = text[1:]
    normalized_text = _normalize_newlines(text)
    normalized_needle = _normalize_newlines(needle)
    match_offset = normalized_text.find(normalized_needle)

    if match_offset < 0:
        return {
            "status": "ok",
            "path": str(p),
            "found": False,
            "needle": needle,
            "content": "",
            "line_offset": 1,
            "match_start_line": None,
            "match_end_line": None,
            "truncated": truncated_scan,
            "scanned_bytes": len(payload),
            "size_bytes": int(stat_result.st_size),
        }

    lines = normalized_text.splitlines(keepends=True)
    total_lines_scanned = len(lines)
    match_start_line = normalized_text.count("\n", 0, match_offset) + 1
    match_end_line = match_start_line + normalized_needle.count("\n")
    start_line = max(1, match_start_line - context_lines)
    end_line = min(total_lines_scanned, match_end_line + context_lines)
    content = "".join(lines[start_line - 1 : end_line])
    last_newline = normalized_text.rfind("\n", 0, match_offset)
    column = match_offset + 1 if last_newline == -1 else match_offset - last_newline
    context_truncated = truncated_scan and end_line >= total_lines_scanned

    snapshot = record_file_snapshot(
        globals().get("ctx"),
        p,
        content=content,
        mtime_ns=stat_result.st_mtime_ns,
        size_bytes=stat_result.st_size,
        full_read=False,
        start_line=start_line,
        end_line=end_line,
        total_lines=None if truncated_scan else total_lines_scanned,
        truncated=context_truncated,
        source="read_edit_context",
    )

    return {
        "status": "ok",
        "path": str(p),
        "found": True,
        "needle": needle,
        "content": content,
        "line_offset": start_line,
        "match_start_line": match_start_line,
        "match_end_line": match_end_line,
        "match_start_column": column,
        "context_before": lines[start_line - 1 : match_start_line - 1],
        "context_after": lines[match_end_line:end_line],
        "truncated": context_truncated,
        "scanned_bytes": len(payload),
        "size_bytes": int(stat_result.st_size),
        "snapshot": {
            "path": snapshot["path"],
            "full_read": snapshot["full_read"],
            "start_line": snapshot["start_line"],
            "end_line": snapshot["end_line"],
            "total_lines": snapshot["total_lines"],
            "truncated": snapshot["truncated"],
        },
    }


def _iter_search_paths(root: Path, glob: str, include_hidden: bool) -> list[Path]:
    base = root.parent if root.is_file() else root
    candidates = [root] if root.is_file() else sorted(root.glob(glob))
    return [
        candidate
        for candidate in candidates
        if candidate.is_file()
        and (include_hidden or not DEFAULT_FILESYSTEM._path_is_hidden(candidate, base))
    ]


def _relative_search_path(path: Path, root: Path) -> str:
    base = root.parent if root.is_file() else root
    try:
        return str(path.relative_to(base))
    except ValueError:
        return path.name


def _offset_to_line_column(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    last_newline = text.rfind("\n", 0, offset)
    column = offset + 1 if last_newline == -1 else offset - last_newline
    return line, column


def _context_window_from_offsets(
    text: str,
    start_offset: int,
    end_offset: int,
    context_lines: int,
) -> dict[str, Any]:
    lines = text.splitlines(keepends=True)
    match_start_line, match_start_column = _offset_to_line_column(text, start_offset)
    end_anchor = max(start_offset, end_offset - 1)
    match_end_line, match_end_column = _offset_to_line_column(text, end_anchor)
    context_start = max(1, match_start_line - context_lines)
    context_end = min(len(lines), match_end_line + context_lines)
    return {
        "content": "".join(lines[context_start - 1 : context_end]),
        "line_offset": context_start,
        "match_start_line": match_start_line,
        "match_end_line": match_end_line,
        "match_start_column": match_start_column,
        "match_end_column": match_end_column,
        "context_before": lines[context_start - 1 : match_start_line - 1],
        "context_after": lines[match_end_line:context_end],
    }


def _search_multiline_code(
    query: str,
    *,
    root: Path,
    glob: str,
    literal: bool,
    case_sensitive: bool,
    context_lines: int,
    max_results: int,
    include_hidden: bool,
    offset: int,
) -> dict[str, Any]:
    flags = re.MULTILINE | re.DOTALL
    if not case_sensitive:
        flags |= re.IGNORECASE
    try:
        rx = re.compile(re.escape(query) if literal else query, flags)
    except re.error as exc:
        return _err(f"Invalid regex: {exc}")

    page_limit = max_results
    raw_limit = offset + page_limit
    matches: list[dict[str, Any]] = []
    files_with_matches: set[str] = set()
    files_searched = 0
    skipped_binary = 0
    skipped_large = 0

    for candidate in _iter_search_paths(root, glob, include_hidden):
        try:
            stat_result = candidate.stat()
        except OSError:
            continue
        if stat_result.st_size > DEFAULT_FILESYSTEM.max_file_bytes:
            skipped_large += 1
            continue
        try:
            raw = DEFAULT_FILESYSTEM.read_bytes(candidate)
        except OSError:
            continue
        if DEFAULT_FILESYSTEM._is_probably_binary(raw):
            skipped_binary += 1
            continue
        text = raw.decode("utf-8", errors="replace")
        if text.startswith("\ufeff"):
            text = text[1:]
        normalized = _normalize_newlines(text)
        file_rel = _relative_search_path(candidate, root)
        files_searched += 1

        for match in rx.finditer(normalized):
            window = _context_window_from_offsets(
                normalized,
                match.start(),
                match.end(),
                context_lines,
            )
            files_with_matches.add(file_rel)
            matches.append(
                {
                    "file": file_rel,
                    "match": match.group(0),
                    **window,
                }
            )
            if len(matches) >= raw_limit:
                break
        if len(matches) >= raw_limit:
            break

    page = matches[offset : offset + page_limit]
    truncated = len(matches) >= raw_limit
    return {
        "status": "ok",
        "mode": "literal" if literal else "regex",
        "query": query,
        "path": str(root),
        "glob": glob,
        "case_sensitive": case_sensitive,
        "context_lines": context_lines,
        "include_hidden": include_hidden,
        "file_count": files_searched,
        "files_with_matches": len(files_with_matches),
        "files_skipped_binary": skipped_binary,
        "files_skipped_too_large": skipped_large,
        "max_results": max_results,
        "offset": offset,
        "returned_count": len(page),
        "total_matches": len(matches),
        "truncated": truncated,
        "next_offset": (offset + len(page)) if truncated else None,
        "matches": page,
    }


def _search_python_symbols(
    query: str,
    *,
    root: Path,
    glob: str,
    case_sensitive: bool,
    max_results: int,
    include_hidden: bool,
    offset: int,
) -> dict[str, Any]:
    needle = query if case_sensitive else query.lower()
    page_limit = max_results
    raw_limit = offset + page_limit
    matches: list[dict[str, Any]] = []
    files_with_matches: set[str] = set()
    files_searched = 0

    for candidate in _iter_search_paths(root, glob, include_hidden):
        if candidate.suffix not in {".py", ".pyi"}:
            continue
        try:
            text = _read_text_preserve_newlines(candidate)
        except OSError:
            continue
        files_searched += 1
        file_rel = _relative_search_path(candidate, root)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            name = node.name
            haystack = name if case_sensitive else name.lower()
            if needle not in haystack:
                continue
            start_line = node.lineno
            end_line = node.end_lineno or start_line
            definition = lines[start_line - 1] if 0 < start_line <= len(lines) else ""
            matches.append(
                {
                    "file": file_rel,
                    "symbol": name,
                    "kind": (
                        "class"
                        if isinstance(node, ast.ClassDef)
                        else "async_function"
                        if isinstance(node, ast.AsyncFunctionDef)
                        else "function"
                    ),
                    "line_number": start_line,
                    "end_line": end_line,
                    "definition": definition,
                }
            )
            files_with_matches.add(file_rel)
            if len(matches) >= raw_limit:
                break
        if len(matches) >= raw_limit:
            break

    page = matches[offset : offset + page_limit]
    truncated = len(matches) >= raw_limit
    return {
        "status": "ok",
        "mode": "symbol",
        "query": query,
        "path": str(root),
        "glob": glob,
        "case_sensitive": case_sensitive,
        "include_hidden": include_hidden,
        "file_count": files_searched,
        "files_with_matches": len(files_with_matches),
        "max_results": max_results,
        "offset": offset,
        "returned_count": len(page),
        "total_matches": len(matches),
        "truncated": truncated,
        "next_offset": (offset + len(page)) if truncated else None,
        "matches": page,
    }


def get_symbol_info(path: str, symbol: str) -> dict[str, Any]:
    """Find a function or class definition inside a file."""
    try:
        p = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    try:
        content = _read_text_preserve_newlines(p)
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")
    stat_result = p.stat()
    record_file_snapshot(
        globals().get("ctx"),
        p,
        content=content,
        mtime_ns=stat_result.st_mtime_ns,
        size_bytes=stat_result.st_size,
        full_read=True,
        start_line=1,
        end_line=len(content.splitlines()),
        total_lines=len(content.splitlines()),
        truncated=False,
        source="get_symbol_info",
    )

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
    try:
        p = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    try:
        text = _read_text_preserve_newlines(p)
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")
    stat_result = p.stat()
    lines = text.splitlines(keepends=True)
    record_file_snapshot(
        globals().get("ctx"),
        p,
        content=text,
        mtime_ns=stat_result.st_mtime_ns,
        size_bytes=stat_result.st_size,
        full_read=True,
        start_line=1,
        end_line=len(lines),
        total_lines=len(lines),
        truncated=False,
        source="read_line",
    )

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
    try:
        p = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    try:
        text = _read_text_preserve_newlines(p)
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")
    stat_result = p.stat()
    lines = text.splitlines()
    record_file_snapshot(
        globals().get("ctx"),
        p,
        content=text,
        mtime_ns=stat_result.st_mtime_ns,
        size_bytes=stat_result.st_size,
        full_read=True,
        start_line=1,
        end_line=len(lines),
        total_lines=len(lines),
        truncated=False,
        source="find_imports",
    )

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


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")
