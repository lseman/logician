"""File Ops skill — re-exports skill metadata only.

All tool implementations are always-on core tools from src/tools/core/*Tool/.
"""

from .file_ops import __grammars__, __skill__

__all__ = ["__skill__", "__grammars__"]
