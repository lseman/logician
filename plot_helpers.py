"""
Backward-compatible wrapper.

Use `apps.plotting.embedding_map` for the canonical module location.
"""

from apps.plotting.embedding_map import plot_session_embedding_map

__all__ = ["plot_session_embedding_map"]
