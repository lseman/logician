"""ToolDispatcher: parallel read tools, serial write tools."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

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
    cache_hit: bool = False


# Read-only tools — safe to run in parallel. Hardcoded; not configurable.
# Includes both core names and common skill-based aliases.
_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    # Core tools (registered by Agent.__init__)
    "read_file", "glob_files", "grep_files", "think",
    # Short/alias names that may appear after normalization
    "glob", "grep",
    # Common skill-based read tools
    "fd_find", "rg_search", "read_file_smart", "list_directory",
    "rg_multiline", "rg_replace",  # rg_replace is content-only, no disk writes
    "get_file_outline", "find_in_file", "count_in_file",
    "find_symbol", "find_references", "find_path",
    "git_diff", "git_log", "git_blame", "git_status",
    "scratch_read", "scratch_list",
    "search_tools", "describe_tool",
    "think_recall",
})

# Tools that actually write to disk — only these trigger VerificationGuard.
_WRITE_TOOLS: frozenset[str] = frozenset({
    "write_file", "edit_file", "apply_edit_block",
    "edit_file_replace", "sed_replace", "regex_replace",
    "multi_edit", "multi_patch", "apply_unified_diff",
    "scratch_write", "scratch_delete",
})


class ToolDispatcher:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        # Tool result cache: maps signature → (timestamp, result)
        self._cache: dict[str, tuple[float, str]] = {}
        self._cache_writes_since_evict: int = 0

    def _evict_expired(self, ttl: float) -> None:
        """Remove all cache entries older than ttl seconds."""
        now = time.time()
        expired = [k for k, (ts, _) in self._cache.items() if (now - ts) >= ttl]
        for k in expired:
            del self._cache[k]
        self._cache_writes_since_evict = 0

    async def dispatch(
        self,
        calls: list[ToolCall],
        state: TurnState,
        config: Config | None = None,
        tool_callback: Callable[[str, dict[str, Any], dict[str, Any]], None] | None = None,
    ) -> list[DispatchResult]:
        """
        Split calls into read batch (parallel) then write calls (serial).
        Updates state: records calls, updates seen_signatures, files_written,
        and increments consecutive_tool_count.

        For multi-tool batches, checks if all calls are safe to execute together
        (no conflicting writes). Falls back to serial execution if unsafe.
        """
        indexed_calls = list(enumerate(calls, start=1))
        reads = [(idx, c) for idx, c in indexed_calls if c.name in _READ_ONLY_TOOLS]
        writes = [(idx, c) for idx, c in indexed_calls if c.name not in _READ_ONLY_TOOLS]

        results: list[DispatchResult] = []

        # Warn if batch is unsafe — execute serially instead
        if len(calls) > 1 and not self._is_safe_batch(calls):
            # Fall back to serial execution for unsafe batch
            for idx, call in indexed_calls:
                result = await self._execute_one(
                    call,
                    config,
                    tool_callback=tool_callback,
                    sequence=idx,
                )
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
                *[
                    self._execute_one(
                        call,
                        config,
                        tool_callback=tool_callback,
                        sequence=idx,
                    )
                    for idx, call in reads
                ]
            )
            results.extend(read_results)

        # Serial writes
        for idx, call in writes:
            result = await self._execute_one(
                call,
                config,
                tool_callback=tool_callback,
                sequence=idx,
            )
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

    async def _execute_one(
        self,
        call: ToolCall,
        config: Config | None = None,
        *,
        tool_callback: Callable[[str, dict[str, Any], dict[str, Any]], None] | None = None,
        sequence: int = 0,
    ) -> DispatchResult:
        def _emit(stage: str, **meta: Any) -> None:
            if tool_callback is None:
                return
            try:
                tool_callback(
                    call.name,
                    dict(call.arguments or {}),
                    {
                        "stage": stage,
                        "sequence": sequence,
                        **meta,
                    },
                )
            except TypeError:
                tool_callback(call.name, dict(call.arguments or {}))
            except Exception:
                pass

        _emit("start")

        # Check cache for read-only tools
        if call.name in _READ_ONLY_TOOLS and config is not None:
            if getattr(config, "tool_cache_enabled", True):
                key = self._cache_key(call)
                ttl = getattr(config, "tool_cache_ttl", 3600)
                cached = self._cache.get(key)
                if cached and (time.time() - cached[0]) < ttl:
                    _emit("end", status="ok", duration_ms=0, cache_hit=True)
                    return DispatchResult(
                        tool_name=call.name,
                        call_id=call.id,
                        output=cached[1],
                        duration_ms=0,
                        cache_hit=True,
                    )

        t0 = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(
                None,
                lambda: self._registry.execute(call),
            )
            output_str = str(output)
            duration_ms = int((time.monotonic() - t0) * 1000)

            # Truncate long outputs before caching or returning
            max_chars = getattr(config, "tool_result_max_chars", 0) if config else 0
            if max_chars and len(output_str) > max_chars:
                output_str = (
                    output_str[:max_chars]
                    + f"\n\n[...output truncated at {max_chars} chars]"
                )

            # Store in cache if read-only and succeeded
            if call.name in _READ_ONLY_TOOLS and config is not None:
                if getattr(config, "tool_cache_enabled", True):
                    key = self._cache_key(call)
                    self._cache[key] = (time.time(), output_str)
                    self._cache_writes_since_evict += 1
                    ttl = getattr(config, "tool_cache_ttl", 3600)
                    if self._cache_writes_since_evict >= 50:
                        self._evict_expired(ttl)

            # First 160 chars of output for display in the TUI tool_end event
            result_preview = output_str[:160].replace("\n", " ").strip()
            _emit(
                "end",
                status="ok",
                duration_ms=duration_ms,
                cache_hit=False,
                result_preview=result_preview,
            )
            return DispatchResult(
                tool_name=call.name,
                call_id=call.id,
                output=output_str,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            _emit(
                "end",
                status="error",
                duration_ms=duration_ms,
                cache_hit=False,
                error=str(exc),
            )
            return DispatchResult(
                tool_name=call.name,
                call_id=call.id,
                output="",
                error=str(exc),
                duration_ms=duration_ms,
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
        """Extract the file path from a write tool call.

        Only returns a path for tools that actually write to disk so that
        VerificationGuard is not triggered by read or shell operations.
        """
        if call.name not in _WRITE_TOOLS:
            return None
        return call.arguments.get("path") or call.arguments.get("file_path")
