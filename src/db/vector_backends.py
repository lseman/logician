from __future__ import annotations

# Backward-compatibility re-exports.
from .backends import (
    _ChromaDBCollection,
    _HNSWCollection,
    _USEARCHCollection,
    create_vector_collection,
)

__all__ = [
    "_ChromaDBCollection",
    "_HNSWCollection",
    "_USEARCHCollection",
    "create_vector_collection",
]
