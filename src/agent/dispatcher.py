"""ToolDispatcher: parallel read tools, serial write tools."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..config import Config
from ..tools import ToolRegistry
from ..tools.runtime import Tool, ToolCall
from .state import ToolResultRecord, TurnState


@dataclass
class DispatchResult:
    tool_name: str
    call_id: str
    output: str
    error: str | None = None
    duration_ms: int = 0
    cache_hit: bool = False
    status: str = "ok"
    read_only: bool = False
    writes_files: bool = False
    verifier: bool = False
    has_content: bool = False


# Read-only tool-name fallbacks for legacy aliases or tests that construct
# bare Tool objects without runtime metadata. Prefer declared runtime metadata.
_READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        # Core tools retained as fallback for bare dispatch/tests without registry metadata
        "read_file",
        "read_edit_context",
        "search_file",
        "glob_files",
        "grep_files",
        "search_code",
        "think",
        # Short/alias names that may appear after normalization
        "glob",
        "grep",
        # Common skill-based read tools
        "fd_find",
        "rg_search",
        "read_file_smart",
        "list_directory",
        "rg_multiline",
        "rg_replace",  # rg_replace is content-only, no disk writes
        "get_file_outline",
        "find_in_file",
        "count_in_file",
        "find_symbol",
        "find_references",
        "find_path",
        "git_diff",
        "git_log",
        "git_blame",
        "git_status",
        "scratch_read",
        "scratch_list",
        "search_tools",
        "describe_tool",
        "think_recall",
    }
)

# Tools that actually write to disk — only these trigger VerificationGuard.
_WRITE_TOOLS: frozenset[str] = frozenset(
    {
        # Core tools retained as fallback for bare dispatch/tests without registry metadata
        "write_file",
        "edit_file",
        "apply_edit_block",
        "edit_file_replace",
        "sed_replace",
        "regex_replace",
        "multi_edit",
        "multi_patch",
        "apply_unified_diff",
        "scratch_write",
        "scratch_delete",
    }
)

_VERIFICATION_NAME_PARTS: tuple[str, ...] = (
    "test",
    "pytest",
    "ruff",
    "lint",
    "check",
    "verify",
    "mypy",
)

_CONTENT_READ_TOOLS: frozenset[str] = frozenset(
    {
        # Core tools retained as fallback for bare dispatch/tests without registry metadata
        "read_file",
        "read_edit_context",
        "search_file",
        "read_line",
        "read_file_smart",
        "find_function_by_name",
        "find_class_by_name",
        "get_symbol_info",
        "find_imports",
    }
)


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

    def _tool_def(self, name: str) -> Tool | None:
        try:
            return self._registry.get(name)
        except Exception:
            return None

    def _tool_runtime_value(self, name: str, field: str) -> Any | None:
        tool = self._tool_def(name)
        if tool is None:
            return None

        runtime_value = getattr(tool.runtime, field, None)
        if runtime_value is not None:
            return runtime_value

        meta = dict(getattr(getattr(tool, "function", None), "__llm_tool_meta__", {}) or {})
        runtime_meta = (
            dict(meta.get("runtime") or {}) if isinstance(meta.get("runtime"), dict) else {}
        )
        if field in runtime_meta:
            return runtime_meta.get(field)
        return meta.get(field)

    def _is_read_only_tool(self, name: str) -> bool:
        read_only = self._tool_runtime_value(name, "read_only")
        if read_only is not None:
            return bool(read_only)
        writes_files = self._tool_runtime_value(name, "writes_files")
        if writes_files is not None:
            return not bool(writes_files)
        return name in _READ_ONLY_TOOLS

    def _writes_files_tool(self, name: str) -> bool:
        writes_files = self._tool_runtime_value(name, "writes_files")
        if writes_files is not None:
            return bool(writes_files)
        return name in _WRITE_TOOLS

    def _is_verifier_tool(self, name: str) -> bool:
        verifier = self._tool_runtime_value(name, "verifier")
        if verifier is not None:
            return bool(verifier)
        lower = name.lower()
        return any(part in lower for part in _VERIFICATION_NAME_PARTS)

    def _is_cacheable_tool(self, name: str) -> bool:
        cacheable = self._tool_runtime_value(name, "cacheable")
        if cacheable is not None:
            return bool(cacheable)
        return self._is_read_only_tool(name)

    def _result_has_content(self, output: str, error: str | None) -> bool:
        if error or not str(output or "").strip():
            return False
        text = str(output).strip()
        try:
            payload = json.loads(text)
        except Exception:
            return bool(text)

        if isinstance(payload, list):
            return len(payload) > 0

        if not isinstance(payload, dict):
            return bool(payload)

        status = str(payload.get("status", "")).strip().lower()
        if status in {"error", "failed", "fail"}:
            return False
        if payload.get("ok") is False or payload.get("success") is False:
            return False
        if payload.get("count") == 0 or payload.get("matches_found") == 0:
            return False

        meaningful_keys = (
            "content",
            "contents",
            "result",
            "results",
            "matches",
            "items",
            "files",
            "paths",
            "stdout",
            "diff",
            "preview",
            "structured_patch",
            "value",
        )
        for key in meaningful_keys:
            value = payload.get(key)
            if value not in (None, "", [], {}, False):
                return True

        return len(payload) > 1 or status == "ok"

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
        reads = [(idx, c) for idx, c in indexed_calls if self._is_read_only_tool(c.name)]
        writes = [(idx, c) for idx, c in indexed_calls if not self._is_read_only_tool(c.name)]

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
            self._update_state_from_dispatch(state, indexed_calls, results)
            return results

        # Read tools currently run serially here. Wrapping executor-backed tool
        # calls in gathered tasks can hang loop shutdown under Python 3.13.
        if reads:
            read_results: list[DispatchResult] = []
            for idx, call in reads:
                read_results.append(
                    await self._execute_one(
                        call,
                        config,
                        tool_callback=tool_callback,
                        sequence=idx,
                    )
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
        self._update_state_from_dispatch(state, indexed_calls, results)

        return results

    def _update_state_from_dispatch(
        self,
        state: TurnState,
        indexed_calls: list[tuple[int, ToolCall]],
        results: list[DispatchResult],
    ) -> None:
        result_by_call_id = {result.call_id: result for result in results}
        for _, call in indexed_calls:
            state.record_call(call)
            result = result_by_call_id.get(call.id)
            if result is not None and result.status == "ok" and self._writes_files_tool(call.name):
                path = self._extract_write_path(call)
                if path:
                    state.record_write(path)
            if result is not None and result.status == "ok" and self._is_content_read_tool(call):
                path = self._extract_read_path(call)
                if path:
                    state.record_read(path)
            if result is not None:
                state.record_tool_result(
                    ToolResultRecord(
                        call_id=result.call_id,
                        tool_name=result.tool_name,
                        status=result.status,
                        output=result.output,
                        error=result.error,
                        read_only=result.read_only,
                        writes_files=result.writes_files,
                        verifier=result.verifier,
                        cache_hit=result.cache_hit,
                        has_content=result.has_content,
                    )
                )
        state.consecutive_tool_count += len(indexed_calls)

    def _is_content_read_tool(self, call: ToolCall) -> bool:
        content_reader = self._tool_runtime_value(call.name, "content_reader")
        if content_reader is not None:
            return bool(content_reader)
        return str(call.name or "").strip() in _CONTENT_READ_TOOLS

    def _extract_read_path(self, call: ToolCall) -> str | None:
        args = dict(call.arguments or {})
        for key in ("path", "file_path", "filename"):
            value = str(args.get(key, "") or "").strip()
            if value:
                return value
        return None

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

    def prepare_call(self, call: ToolCall) -> tuple[ToolCall | None, str | None]:
        prepare = getattr(self._registry, "prepare_call", None)
        if not callable(prepare):
            return call, None
        return prepare(call)

    def available_tool_names(self) -> set[str]:
        list_tools = getattr(self._registry, "list_tools", None)
        if not callable(list_tools):
            return set()
        try:
            return {
                str(getattr(tool, "name", "") or "").strip()
                for tool in list_tools()
                if str(getattr(tool, "name", "") or "").strip()
            }
        except Exception:
            return set()

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
        is_read_only = self._is_read_only_tool(call.name)
        writes_files = self._writes_files_tool(call.name)
        verifier = self._is_verifier_tool(call.name)
        cache_enabled = getattr(config, "tool_cache_enabled", True) if config else True
        cache_ttl = getattr(config, "tool_cache_ttl", 3600) if config else 3600
        max_chars = getattr(config, "tool_result_max_chars", 0) if config else 0

        if is_read_only and cache_enabled and self._is_cacheable_tool(call.name):
            key = self._cache_key(call)
            cached = self._cache.get(key)
            if cached and (time.time() - cached[0]) < cache_ttl:
                _emit(
                    "end",
                    status="ok",
                    duration_ms=0,
                    cache_hit=True,
                    result_output=cached[1] if writes_files else "",
                )
                return DispatchResult(
                    tool_name=call.name,
                    call_id=call.id,
                    output=cached[1],
                    duration_ms=0,
                    cache_hit=True,
                    status="ok",
                    read_only=is_read_only,
                    writes_files=writes_files,
                    verifier=verifier,
                    has_content=self._result_has_content(cached[1], None),
                )

        t0 = time.monotonic()
        try:
            output = self._registry.execute(call)
            output_str = str(output)
            duration_ms = int((time.monotonic() - t0) * 1000)

            # Truncate long outputs before caching or returning
            if max_chars and len(output_str) > max_chars:
                output_str = (
                    output_str[:max_chars] + f"\n\n[...output truncated at {max_chars} chars]"
                )

            # Store in cache if read-only and succeeded
            if is_read_only and cache_enabled and self._is_cacheable_tool(call.name):
                key = self._cache_key(call)
                self._cache[key] = (time.time(), output_str)
                self._cache_writes_since_evict += 1
                if self._cache_writes_since_evict >= 50:
                    self._evict_expired(cache_ttl)

            # First 160 chars of output for display in the TUI tool_end event
            result_preview = output_str[:160].replace("\n", " ").strip()
            status = (
                "error"
                if getattr(self._registry, "_result_indicates_error", lambda _v: False)(output_str)
                else "ok"
            )
            _emit(
                "end",
                status=status,
                duration_ms=duration_ms,
                cache_hit=False,
                result_preview=result_preview,
                result_output=output_str if writes_files else "",
            )
            return DispatchResult(
                tool_name=call.name,
                call_id=call.id,
                output=output_str,
                duration_ms=duration_ms,
                status=status,
                read_only=is_read_only,
                writes_files=writes_files,
                verifier=verifier,
                has_content=self._result_has_content(
                    output_str,
                    "tool_result_error" if status == "error" else None,
                ),
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
                status="error",
                read_only=is_read_only,
                writes_files=writes_files,
                verifier=verifier,
                has_content=False,
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

        writes = [c for c in calls if not self._is_read_only_tool(c.name)]
        if not writes:
            return True  # all reads, safe

        if len(writes) > 1:
            # Multiple writes — only safe if they target different paths
            paths = {c.arguments.get("path") or c.arguments.get("file_path", "") for c in writes}
            return len(paths) == len(writes) and "" not in paths

        return True  # single write with some reads

    def _extract_write_path(self, call: ToolCall) -> str | None:
        """Extract the file path from a write tool call.

        Only returns a path for tools that actually write to disk so that
        VerificationGuard is not triggered by read or shell operations.
        """
        if not self._writes_files_tool(call.name):
            return None
        return call.arguments.get("path") or call.arguments.get("file_path")
