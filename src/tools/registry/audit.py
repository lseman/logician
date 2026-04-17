# -*- coding: utf-8 -*-
"""
Audit logging for tool execution.

Records every tool call with timestamp, arguments, result status, and duration.
Useful for debugging, monitoring, and compliance.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["AuditEntry", "AuditLogger"]


@dataclass
class AuditEntry:
    timestamp: float
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    duration_s: float
    error: str = ""
    skill_id: str = ""
    version: str = "1.0.0"


class AuditLogger:
    """In-memory audit log with sliding window."""

    def __init__(self, max_entries: int = 10000) -> None:
        self._entries: list[AuditEntry] = []
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def log(self, entry: AuditEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]

    def get_entries(
        self,
        tool_name: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        with self._lock:
            entries = list(self._entries)
            if tool_name:
                entries = [e for e in entries if e.tool_name == tool_name]
            if since is not None:
                entries = [e for e in entries if e.timestamp >= since]
            return list(entries[-limit:])

    def summary(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._entries)
            errors = sum(1 for e in self._entries if not e.success)
            return {
                "total_calls": total,
                "total_errors": errors,
                "error_rate": errors / max(1, total),
            }
