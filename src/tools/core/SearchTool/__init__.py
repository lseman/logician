"""Package-style wrapper for search and inspection primitives."""

from .inspection import find_imports, get_symbol_info, read_line
from .tool import (
    fd_find,
    find_references,
    glob_files,
    grep_files,
    rg_search,
    search_code,
    search_file,
    search_symbols,
)

__all__ = [
    "fd_find",
    "find_imports",
    "find_references",
    "get_symbol_info",
    "glob_files",
    "grep_files",
    "read_line",
    "rg_search",
    "search_code",
    "search_file",
    "search_symbols",
]
