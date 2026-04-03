"""Package-style wrapper for git inspection tools."""

from .tool import get_git_diff, get_git_status

__all__ = ["get_git_diff", "get_git_status"]
