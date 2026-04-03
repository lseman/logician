"""Package-style wrapper for search and inspection primitives."""

from .inspection import find_imports, get_symbol_info, read_line, search_file
from .tool import glob_files, grep_files, search_code

__all__ = [
    "find_imports",
    "get_symbol_info",
    "glob_files",
    "grep_files",
    "read_line",
    "search_code",
    "search_file",
]
