"""ToolDispatcher: parallel read tools, serial write tools."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ..tools import ToolRegistry
from ..tools.runtime import ToolCall
from .state import TurnState


@dataclass
class DispatchResult:
    tool_name: str
    call_id: str
    output: str
    error: str | None = None
    duration_ms: int = 0


# Read-only tools — safe to run in parallel. Hardcoded; not configurable.
_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "read_file", "glob", "grep", "think",
})


class ToolDispatcher:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def dispatch(
        self,
        calls: list[ToolCall],
        state: TurnState,
    ) -> list[DispatchResult]:
        """
        Split calls into read batch (parallel) then write calls (serial).
        Updates state: records calls, updates seen_signatures, files_written,
        and increments consecutive_tool_count.
        """
        reads = [c for c in calls if c.name in _READ_ONLY_TOOLS]
        writes = [c for c in calls if c.name not in _READ_ONLY_TOOLS]

        results: list[DispatchResult] = []

        # Parallel reads
        if reads:
            read_results = await asyncio.gather(
                *[self._execute_one(call) for call in reads]
            )
            results.extend(read_results)

        # Serial writes
        for call in writes:
            result = await self._execute_one(call)
            results.append(result)
            # Track file writes for VerificationGuard
            path = self._extract_write_path(call)
            if path:
                state.record_write(path)

        # Update state
        for call in calls:
            state.record_call(call)
        state.consecutive_tool_count += len(calls)

        return results

    async def _execute_one(self, call: ToolCall) -> DispatchResult:
        t0 = time.monotonic()
        try:
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None,
                lambda: self._registry.execute(call),
            )
            return DispatchResult(
                tool_name=call.name,
                call_id=call.id,
                output=str(output),
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
        except Exception as exc:
            return DispatchResult(
                tool_name=call.name,
                call_id=call.id,
                output="",
                error=str(exc),
                duration_ms=int((time.monotonic() - t0) * 1000),
            )

    def _extract_write_path(self, call: ToolCall) -> str | None:
        """Extract the file path from a write tool call, if present."""
        return call.arguments.get("path") or call.arguments.get("file_path")
