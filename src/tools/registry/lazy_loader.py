# -*- coding: utf-8 -*-
"""
Tool cache with TTL support for cacheable tools.

Provides an LRU-style cache with configurable entry count and time-to-live,
used by tools marked as cacheable in ToolRuntimeMetadata.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ToolCache", "CacheEntry"]


@dataclass(frozen=True)
class CacheEntry:
    result: Any
    expires_at: float


class ToolCache:
    """Thread-safe LIFO cache with TTL for cacheable tools."""

    def __init__(
        self,
        max_entries: int = 100,
        ttl_seconds: float = 300.0,
    ) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._ttl = ttl_seconds

    def _make_key(self, tool_name: str, args: dict[str, Any]) -> str:
        raw = f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, tool_name: str, args: dict[str, Any]) -> Any | None:
        key = self._make_key(tool_name, args)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if time.time() > entry.expires_at:
                del self._cache[key]
                return None
            return entry.result

    def put(self, tool_name: str, args: dict[str, Any], result: Any) -> None:
        key = self._make_key(tool_name, args)
        with self._lock:
            if len(self._cache) >= self._max_entries:
                oldest_key = min(
                    self._cache,
                    key=lambda k: self._cache[k].expires_at,
                )
                del self._cache[oldest_key]
            self._cache[key] = CacheEntry(
                result=result,
                expires_at=time.time() + self._ttl,
            )

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
