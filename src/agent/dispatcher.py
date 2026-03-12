"""ToolDispatcher: parallel read tools, serial write tools."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
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
        # Tool result cache: maps signature → (timestamp, result)
        self._cache: dict[str, tuple[float, str]] = {}

    async def dispatch(
        self,
        calls: list[ToolCall],
        state: TurnState,
        config: Config | None = None,
    ) -> list[DispatchResult]:
        """
        Split calls into read batch (parallel) then write calls (serial).
        Updates state: records calls, updates seen_signatures, files_written,
        and increments consecutive_tool_count.

        For multi-tool batches, checks if all calls are safe to execute together
        (no conflicting writes). Falls back to serial execution if unsafe.
        """
        reads = [c for c in calls if c.name in _READ_ONLY_TOOLS]
        writes = [c for c in calls if c.name not in _READ_ONLY_TOOLS]

        results: list[DispatchResult] = []

        # Warn if batch is unsafe — execute serially instead
        if len(calls) > 1 and not self._is_safe_batch(calls):
            # Fall back to serial execution for unsafe batch
            for call in calls:
                result = await self._execute_one(call, config)
                results.append(result)
                if call.name not in _READ_ONLY_TOOLS:
                    path = self._extract_write_path(call)
                    if path:
                        state.record_write(path)
                state.record_call(call)
                state.consecutive_tool_count += 1
            return results

        # Parallel reads
        if reads:
            read_results = await asyncio.gather(
                *[self._execute_one(call, config) for call in reads]
            )
            results.extend(read_results)

        # Serial writes
        for call in writes:
            result = await self._execute_one(call, config)
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

    def _cache_key(self, call: ToolCall) -> str:
        """Compute a stable cache key for a tool call."""
        raw = json.dumps(
            {"n": call.name, "a": call.arguments},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"{call.name}:{digest}"

    async def _execute_one(self, call: ToolCall, config: Config | None = None) -> DispatchResult:
        # Check cache for read-only tools
        if call.name in _READ_ONLY_TOOLS and config is not None:
            if getattr(config, "tool_cache_enabled", True):
                key = self._cache_key(call)
                ttl = getattr(config, "tool_cache_ttl", 3600)
                cached = self._cache.get(key)
                if cached and (time.time() - cached[0]) < ttl:
                    return DispatchResult(
                        tool_name=call.name,
                        call_id=call.id,
                        output=cached[1],
                        duration_ms=0,
                    )

        t0 = time.monotonic()
        try:
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None,
                lambda: self._registry.execute(call),
            )
            output_str = str(output)
            duration_ms = int((time.monotonic() - t0) * 1000)

            # Store in cache if read-only and succeeded
            if call.name in _READ_ONLY_TOOLS and config is not None:
                if getattr(config, "tool_cache_enabled", True):
                    key = self._cache_key(call)
                    self._cache[key] = (time.time(), output_str)

            return DispatchResult(
                tool_name=call.name,
                call_id=call.id,
                output=output_str,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            return DispatchResult(
                tool_name=call.name,
                call_id=call.id,
                output="",
                error=str(exc),
                duration_ms=int((time.monotonic() - t0) * 1000),
            )

    def _is_safe_batch(self, calls: list[ToolCall]) -> bool:
        """Returns True if all calls can safely execute in a batch.

        A batch is safe when:
        - All calls are read-only, OR
        - At most one write call and all others are reads, OR
        - All write calls target different files
        """
        if len(calls) <= 1:
            return True

        writes = [c for c in calls if c.name not in _READ_ONLY_TOOLS]
        if not writes:
            return True  # all reads, safe

        if len(writes) > 1:
            # Multiple writes — only safe if they target different paths
            paths = {
                c.arguments.get("path") or c.arguments.get("file_path", "")
                for c in writes
            }
            return len(paths) == len(writes) and "" not in paths

        return True  # single write with some reads

    def _extract_write_path(self, call: ToolCall) -> str | None:
        """Extract the file path from a write tool call, if present."""
        return call.arguments.get("path") or call.arguments.get("file_path")
