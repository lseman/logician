"""Package-style wrapper for file mutation tools."""

from .tool import apply_edit_block, edit_file, preview_edit, smart_edit, write_file

__all__ = [
    "apply_edit_block",
    "edit_file",
    "preview_edit",
    "smart_edit",
    "write_file",
]
