from __future__ import annotations

import json
import time
from typing import Any

from ..runtime import HAS_TOON, Tool, ToolCall, encode


class RegistryExecutionMixin:
    """ToolRegistry mixin."""

    def _prepare_execute_tool(self, call: ToolCall) -> tuple[Tool | None, str | None]:
        tool = self.get(call.name)
        if not tool:
            self._log.error("Tool not found: %s", call.name)
            suggestions = self._suggest_tool_names(call.name, max_items=5)
            self._record_tool_execution(
                tool_name=call.name,
                duration_s=0.0,
                ok=False,
                error="tool not found",
            )
            return None, self._tool_error_payload(
                call.name,
                f"Tool '{call.name}' is not registered.",
                error_type="tool_not_found",
                suggestions=suggestions,
            )

        if self._execution_globals.get("ctx", None) is None:
            self._record_tool_execution(
                tool_name=call.name,
                duration_s=0.0,
                ok=False,
                error="context missing",
            )
            return None, self._tool_error_payload(
                call.name,
                "ctx is None (Context not injected into ToolRegistry._execution_globals).",
                error_type="missing_context",
            )
        return tool, None

    def _run_tool_execution(
        self,
        *,
        call: ToolCall,
        tool: Tool,
        prepared_args: dict[str, Any] | None,
    ) -> tuple[Any | None, float, str | None]:
        exec_start = time.perf_counter()
        try:
            result = tool.function(**(prepared_args or {}))
            exec_dur = time.perf_counter() - exec_start
            return result, exec_dur, None
        except Exception as exc:
            exec_dur = time.perf_counter() - exec_start
            self._log.exception("Tool execution failed: %s", call.name)
            self._record_tool_execution(
                tool_name=call.name,
                duration_s=exec_dur,
                ok=False,
                error=str(exc),
            )
            return (
                None,
                exec_dur,
                self._tool_error_payload(
                    call.name,
                    str(exc),
                    error_type="tool_exception",
                ),
            )

    @staticmethod
    def _serialize_execution_result(result: Any, use_toon: bool) -> str:
        if use_toon and HAS_TOON and isinstance(result, (dict, list)):
            assert encode is not None
            return encode(result)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def execute(self, call: ToolCall, use_toon: bool = True) -> str:
        tool, preflight_error = self._prepare_execute_tool(call)
        if preflight_error is not None:
            return preflight_error
        assert tool is not None

        prepared_args, validation_error = self._prepare_tool_arguments(
            tool, call.arguments
        )
        if validation_error is not None:
            self._record_tool_execution(
                tool_name=call.name,
                duration_s=0.0,
                ok=False,
                error="argument validation failed",
            )
            return validation_error

        result, exec_dur, execution_error = self._run_tool_execution(
            call=call,
            tool=tool,
            prepared_args=prepared_args,
        )
        if execution_error is not None:
            return execution_error

        is_error_result = self._result_indicates_error(result)
        self._record_tool_execution(
            tool_name=call.name,
            duration_s=exec_dur,
            ok=not is_error_result,
            error="result payload indicates failure"
            if is_error_result
            else "",
        )
        if not is_error_result:
            skill_id = str(getattr(tool, "skill_id", "") or "").strip()
            if skill_id:
                try:
                    self._catalog.note_skill_usage(skill_id)
                except Exception:
                    self._log.debug(
                        "Failed to record skill usage for skill_id=%s", skill_id
                    )
        return self._serialize_execution_result(result, use_toon)
