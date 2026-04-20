"""Package-style wrapper for structured filesystem mutation tools."""

from .tool import delete_path, mkdir, move_path

__all__ = ["delete_path", "mkdir", "move_path"]
