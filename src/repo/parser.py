from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from code_review_graph.parser import CodeParser, EdgeInfo, NodeInfo

    _CODE_PARSER = CodeParser()
    _TS_ENABLED = True
except ImportError:
    _CODE_PARSER = None
    EdgeInfo = None  # type: ignore[assignment]
    NodeInfo = None  # type: ignore[assignment]
    _TS_ENABLED = False


TREE_SITTER_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".cs",
    ".rb",
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".h",
    ".hpp",
    ".kt",
    ".swift",
    ".php",
    ".scala",
    ".sol",
    ".vue",
    ".dart",
    ".r",
    ".mjs",
    ".astro",
    ".pl",
    ".pm",
    ".t",
    ".xs",
    ".lua",
    ".ipynb",
}


def tree_sitter_available() -> bool:
    return _TS_ENABLED and _CODE_PARSER is not None


def parse_file_symbols_imports(
    path: Path, source: bytes
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse file symbols and imports using the Tree-sitter code parser."""
    if not tree_sitter_available():
        return [], []

    nodes, edges = _CODE_PARSER.parse_bytes(path, source)
    symbols: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []

    for node in nodes:
        if node.kind in {"Class", "Function", "Type", "Test"} and node.name:
            symbol_kind = node.kind.lower()
            if symbol_kind == "file":
                continue
            symbols.append(
                {
                    "id": f"symbol:{node.file_path}:{node.name}:{node.line_start}",
                    "kind": "symbol",
                    "symbol_kind": symbol_kind,
                    "name": node.name,
                    "line": int(node.line_start or 0),
                    "match_names": [node.name],
                }
            )

    for edge in edges:
        if edge.kind == "IMPORTS_FROM":
            imports.append(
                {
                    "source": edge.source,
                    "module": edge.target,
                    "symbol": "",
                    "local_symbol": "",
                    "line": int(edge.line or 0),
                }
            )

    return symbols, imports
