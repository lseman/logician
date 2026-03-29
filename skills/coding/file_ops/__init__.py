"""File Ops skill — re-exports skill metadata only.

All tool implementations are always-on core tools from src/tools/core/files.py.
"""
from .file_ops import __skill__, __grammars__

__all__ = ["__skill__", "__grammars__"]
