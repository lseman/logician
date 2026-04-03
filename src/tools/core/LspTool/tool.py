"""Lightweight source navigation tool with LSP-style operations.

This is a Python-first implementation that provides a useful subset of common
LSP queries without requiring an external language-server process.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path
from typing import Any

from ..FileReadTool.state import record_file_snapshot, resolve_tool_path
from ..filesystem import DEFAULT_FILESYSTEM

_PYTHON_EXTENSIONS = {".py", ".pyi"}
_OPERATION_ALIASES = {
    "document_symbols": "document_symbols",
    "documentsymbols": "document_symbols",
    "documentsymbol": "document_symbols",
    "workspace_symbols": "workspace_symbols",
    "workspacesymbols": "workspace_symbols",
    "workspace_symbol": "workspace_symbols",
    "workspaceSymbol": "workspace_symbols",
    "go_to_definition": "go_to_definition",
    "gotodefinition": "go_to_definition",
    "goToDefinition": "go_to_definition",
    "definition": "go_to_definition",
    "go_to_implementation": "go_to_definition",
    "gotoimplementation": "go_to_definition",
    "goToImplementation": "go_to_definition",
    "find_references": "find_references",
    "findreferences": "find_references",
    "findReferences": "find_references",
    "references": "find_references",
    "hover": "hover",
}


def _err(message: str, **extra: Any) -> dict[str, Any]:
    payload = {"status": "error", "error": message}
    if extra:
        payload.update(extra)
    return payload


class _SymbolCollector(ast.NodeVisitor):
    def __init__(self, *, path: Path, lines: list[str]) -> None:
        self.path = path
        self.lines = lines
        self.stack: list[str] = []
        self.symbols: list[dict[str, Any]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._record(node, kind="class")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._record(node, kind="function")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._record(node, kind="async_function")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()
        return node

    def _record(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        kind: str,
    ) -> None:
        line_number = int(getattr(node, "lineno", 0) or 0)
        end_line = int(getattr(node, "end_lineno", line_number) or line_number)
        col = int(getattr(node, "col_offset", 0) or 0) + 1
        end_col = int(getattr(node, "end_col_offset", col - 1) or (col - 1)) + 1
        definition = self.lines[line_number - 1] if 0 < line_number <= len(self.lines) else ""
        self.symbols.append(
            {
                "name": node.name,
                "kind": kind,
                "path": str(self.path),
                "line": line_number,
                "end_line": end_line,
                "column": col,
                "end_column": end_col,
                "container": self.stack[-1] if self.stack else None,
                "definition": definition,
                "docstring": ast.get_docstring(node, clean=False),
            }
        )


def lsp_tool(
    operation: str,
    path: str,
    *,
    line: int | None = None,
    character: int | None = None,
    query: str | None = None,
    glob: str = "**/*.py",
    include_hidden: bool = False,
    max_results: int = 50,
) -> dict[str, Any]:
    """Perform source-navigation queries with an LSP-like interface."""
    normalized = _normalize_operation(operation)
    if normalized is None:
        return _err(
            "Unsupported operation. Use one of: document_symbols, workspace_symbols, go_to_definition, find_references, hover"
        )
    if max_results <= 0:
        return _err("max_results must be >= 1")

    try:
        resolved = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))
    if not resolved.exists():
        return _err(f"Path not found: {path}")

    if normalized == "workspace_symbols":
        symbol_query = str(query or "").strip()
        if not symbol_query:
            return _err("query is required for workspace_symbols")
        results = _workspace_symbols(
            symbol_query,
            root=resolved,
            glob=glob,
            include_hidden=include_hidden,
            max_results=max_results,
        )
        return {
            "status": "ok",
            "operation": normalized,
            "query": symbol_query,
            "path": str(resolved),
            "result_count": len(results),
            "results": results,
        }

    if not resolved.is_file():
        return _err(f"Operation '{normalized}' requires a file path")
    if resolved.suffix not in _PYTHON_EXTENSIONS:
        return _err(f"lsp_tool currently supports Python files only: {path}")

    source = _read_python_source(resolved, source="lsp_tool")
    if isinstance(source, dict):
        return source
    content, tree, lines = source
    symbols = _collect_symbols(resolved, tree, lines)

    if normalized == "document_symbols":
        return {
            "status": "ok",
            "operation": normalized,
            "path": str(resolved),
            "result_count": min(len(symbols), max_results),
            "results": symbols[:max_results],
            "truncated": len(symbols) > max_results,
        }

    symbol_name = str(query or "").strip() or _symbol_at_position(
        content, line=line, character=character
    )
    if not symbol_name:
        return _err(
            "Could not determine symbol at the requested position. Provide query explicitly or pass a valid line/character on an identifier."
        )

    if normalized == "go_to_definition":
        results = _definition_results(
            symbol_name,
            current_file_symbols=symbols,
            workspace_root=resolved.parent,
            glob=glob,
            include_hidden=include_hidden,
            max_results=max_results,
        )
        return {
            "status": "ok",
            "operation": normalized,
            "symbol": symbol_name,
            "path": str(resolved),
            "result_count": len(results),
            "results": results,
        }

    if normalized == "hover":
        results = _definition_results(
            symbol_name,
            current_file_symbols=symbols,
            workspace_root=resolved.parent,
            glob=glob,
            include_hidden=include_hidden,
            max_results=1,
        )
        hover = None
        if results:
            first = dict(results[0])
            hover = {
                "symbol": first.get("name"),
                "kind": first.get("kind"),
                "path": first.get("path"),
                "line": first.get("line"),
                "definition": first.get("definition"),
                "docstring": first.get("docstring") or "",
            }
        return {
            "status": "ok",
            "operation": normalized,
            "symbol": symbol_name,
            "path": str(resolved),
            "result_count": 1 if hover else 0,
            "results": [hover] if hover else [],
        }

    references = _find_references(
        symbol_name,
        root=resolved.parent,
        glob=glob,
        include_hidden=include_hidden,
        max_results=max_results,
    )
    return {
        "status": "ok",
        "operation": normalized,
        "symbol": symbol_name,
        "path": str(resolved),
        "result_count": len(references),
        "results": references,
    }


def _normalize_operation(operation: str) -> str | None:
    raw = str(operation or "").strip()
    if not raw:
        return None
    return _OPERATION_ALIASES.get(raw, _OPERATION_ALIASES.get(raw.lower()))


def _read_python_source(
    path: Path, *, source: str
) -> tuple[str, ast.AST, list[str]] | dict[str, Any]:
    try:
        content = DEFAULT_FILESYSTEM.read_text(path, encoding="utf-8", errors="replace")
    except OSError as exc:
        return _err(f"Cannot read file: {exc}", path=str(path))
    try:
        tree = ast.parse(content, filename=str(path))
    except SyntaxError as exc:
        return _err(f"Cannot parse Python file: {exc}", path=str(path))
    lines = content.splitlines()
    stat_result = path.stat()
    record_file_snapshot(
        globals().get("ctx"),
        path,
        content=content,
        mtime_ns=stat_result.st_mtime_ns,
        size_bytes=stat_result.st_size,
        full_read=True,
        start_line=1,
        end_line=len(lines),
        total_lines=len(lines),
        truncated=False,
        source=source,
    )
    return content, tree, lines


def _collect_symbols(path: Path, tree: ast.AST, lines: list[str]) -> list[dict[str, Any]]:
    collector = _SymbolCollector(path=path, lines=lines)
    collector.visit(tree)
    return collector.symbols


def _symbol_at_position(content: str, *, line: int | None, character: int | None) -> str:
    if line is None or character is None or line <= 0 or character <= 0:
        return ""
    lines = content.splitlines()
    if line > len(lines):
        return ""
    line_text = lines[line - 1]
    if not line_text:
        return ""
    index = min(max(character - 1, 0), max(len(line_text) - 1, 0))
    if not (line_text[index].isalnum() or line_text[index] == "_"):
        if index > 0 and (line_text[index - 1].isalnum() or line_text[index - 1] == "_"):
            index -= 1
        else:
            return ""
    start = index
    end = index + 1
    while start > 0 and (line_text[start - 1].isalnum() or line_text[start - 1] == "_"):
        start -= 1
    while end < len(line_text) and (line_text[end].isalnum() or line_text[end] == "_"):
        end += 1
    return line_text[start:end]


def _iter_workspace_python_files(root: Path, glob: str, include_hidden: bool) -> list[Path]:
    search_root = root.parent if root.is_file() else root
    candidates = [root] if root.is_file() else sorted(search_root.glob(glob))
    out: list[Path] = []
    for candidate in candidates:
        try:
            if not candidate.is_file() or candidate.suffix not in _PYTHON_EXTENSIONS:
                continue
        except OSError:
            continue
        if not include_hidden and DEFAULT_FILESYSTEM._path_is_hidden(candidate, search_root):
            continue
        out.append(candidate)
    return out


def _workspace_symbols(
    query: str,
    *,
    root: Path,
    glob: str,
    include_hidden: bool,
    max_results: int,
) -> list[dict[str, Any]]:
    needle = query.lower()
    results: list[dict[str, Any]] = []
    for candidate in _iter_workspace_python_files(root, glob, include_hidden):
        source = _read_python_source(candidate, source="lsp_tool.workspace_symbols")
        if isinstance(source, dict):
            continue
        _, tree, lines = source
        for symbol in _collect_symbols(candidate, tree, lines):
            if needle not in str(symbol.get("name") or "").lower():
                continue
            results.append(symbol)
            if len(results) >= max_results:
                return results
    return results


def _definition_results(
    symbol_name: str,
    *,
    current_file_symbols: list[dict[str, Any]],
    workspace_root: Path,
    glob: str,
    include_hidden: bool,
    max_results: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()

    def append_symbol(symbol: dict[str, Any]) -> None:
        key = (
            str(symbol.get("path") or ""),
            int(symbol.get("line") or 0),
            str(symbol.get("name") or ""),
        )
        if key in seen:
            return
        seen.add(key)
        results.append(symbol)

    for symbol in current_file_symbols:
        if symbol.get("name") == symbol_name:
            append_symbol(symbol)
            if len(results) >= max_results:
                return results

    for candidate in _iter_workspace_python_files(workspace_root, glob, include_hidden):
        if results and str(results[0].get("path") or "") == str(candidate):
            continue
        source = _read_python_source(candidate, source="lsp_tool.definition")
        if isinstance(source, dict):
            continue
        _, tree, lines = source
        for symbol in _collect_symbols(candidate, tree, lines):
            if symbol.get("name") != symbol_name:
                continue
            append_symbol(symbol)
            if len(results) >= max_results:
                return results

    return results


def _find_references(
    symbol_name: str,
    *,
    root: Path,
    glob: str,
    include_hidden: bool,
    max_results: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for candidate in _iter_workspace_python_files(root, glob, include_hidden):
        source = _read_python_source(candidate, source="lsp_tool.references")
        if isinstance(source, dict):
            continue
        content, _, lines = source
        try:
            tokens = tokenize.generate_tokens(io.StringIO(content).readline)
        except tokenize.TokenError:
            continue
        for token in tokens:
            if token.type != tokenize.NAME or token.string != symbol_name:
                continue
            line_text = lines[token.start[0] - 1] if 0 < token.start[0] <= len(lines) else ""
            results.append(
                {
                    "name": symbol_name,
                    "path": str(candidate),
                    "line": token.start[0],
                    "end_line": token.end[0],
                    "column": token.start[1] + 1,
                    "end_column": token.end[1],
                    "line_content": line_text,
                }
            )
            if len(results) >= max_results:
                return results
    return results


__all__ = ["lsp_tool"]
