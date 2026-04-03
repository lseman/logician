from __future__ import annotations

import fnmatch
import json
import time
from typing import Any

from ..runtime import HAS_TOON, Tool, ToolCall, encode


class RegistryExecutionMixin:
    """ToolRegistry mixin."""

    def _prepare_execute_tool(
        self,
        call: ToolCall,
        *,
        record_stats: bool = True,
    ) -> tuple[Tool | None, str | None]:
        tool = self.get(call.name)
        if not tool:
            self._log.error("Tool not found: %s", call.name)
            suggestions = self._suggest_tool_names(call.name, max_items=5)
            if record_stats:
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
            if record_stats:
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

    def prepare_call(self, call: ToolCall) -> tuple[ToolCall | None, str | None]:
        tool, preflight_error = self._prepare_execute_tool(call, record_stats=False)
        if preflight_error is not None:
            return None, preflight_error
        assert tool is not None

        prepared_args, validation_error = self._prepare_tool_arguments(tool, call.arguments)
        if validation_error is not None:
            return None, validation_error

        return (
            ToolCall(
                id=call.id,
                name=call.name,
                arguments=dict(prepared_args or {}),
            ),
            None,
        )

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

    # Tools blocked when plan_mode is active.
    _PLAN_MODE_BLOCKED_TOOLS = frozenset(
        {
            "write_file",
            "edit_file",
            "apply_edit_block",
            "smart_edit",
            "preview_edit",
            "bash",
            "patch",
        }
    )

    @staticmethod
    def _default_permission_mode(tool: Tool, ctx: Any | None) -> str:
        runtime_mode = str(getattr(tool.runtime, "permission_mode", "") or "").strip().lower()
        if runtime_mode in {"plan", "auto", "bypass"}:
            return runtime_mode
        default_mode = str(getattr(ctx, "permission_default_mode", "") or "").strip().lower()
        if default_mode in {"plan", "auto", "bypass"}:
            return default_mode
        if bool(getattr(tool.runtime, "writes_files", False)):
            return "plan"
        return "auto"

    @staticmethod
    def _resolve_permission_mode(
        tool: Tool, tool_name: str, ctx: Any | None
    ) -> tuple[str, str | None]:
        rules = getattr(ctx, "permission_rules", []) if ctx is not None else []
        matched_pattern: str | None = None
        matched_mode: str | None = None
        for item in list(rules or []):
            if isinstance(item, dict):
                pattern = str(item.get("pattern") or "").strip()
                mode = str(item.get("mode") or "").strip().lower()
            else:
                pattern = str(getattr(item, "pattern", "") or "").strip()
                mode = str(getattr(item, "mode", "") or "").strip().lower()
            if not pattern or mode not in {"plan", "auto", "bypass"}:
                continue
            if fnmatch.fnmatchcase(tool_name, pattern) or fnmatch.fnmatch(
                tool_name.lower(), pattern.lower()
            ):
                matched_pattern = pattern
                matched_mode = mode
        if matched_mode is not None:
            return matched_mode, matched_pattern
        return RegistryExecutionMixin._default_permission_mode(tool, ctx), None

    def execute(self, call: ToolCall, use_toon: bool = True) -> str:
        tool, preflight_error = self._prepare_execute_tool(call)
        if preflight_error is not None:
            return preflight_error
        assert tool is not None

        ctx = self._execution_globals.get("ctx")
        permission_mode, matched_rule = self._resolve_permission_mode(tool, call.name, ctx)
        if permission_mode == "plan":
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        f"Tool '{call.name}' is blocked by the current permission policy. "
                        "Switch the matching rule to auto or bypass to execute it."
                    ),
                    "error_type": "permission_blocked",
                    "permission_mode": permission_mode,
                    "matched_rule": matched_rule,
                },
                ensure_ascii=False,
            )

        if ctx is not None and getattr(ctx, "plan_mode", False) and permission_mode != "bypass":
            if call.name in self._PLAN_MODE_BLOCKED_TOOLS or bool(
                getattr(tool.runtime, "writes_files", False)
            ):
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            f"Tool '{call.name}' is blocked while plan mode is active. "
                            "Use exit_plan_mode first or grant bypass permission for this tool."
                        ),
                        "error_type": "plan_mode_blocked",
                        "permission_mode": permission_mode,
                        "matched_rule": matched_rule,
                    },
                    ensure_ascii=False,
                )

        prepared_args, validation_error = self._prepare_tool_arguments(tool, call.arguments)
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
            error="result payload indicates failure" if is_error_result else "",
        )
        if not is_error_result:
            skill_id = str(getattr(tool, "skill_id", "") or "").strip()
            if skill_id:
                try:
                    self._catalog.note_skill_usage(skill_id)
                except Exception:
                    self._log.debug("Failed to record skill usage for skill_id=%s", skill_id)
            # Increment session tool call counter.
            _ctx = self._execution_globals.get("ctx")
            if _ctx is not None:
                try:
                    _ctx.tool_call_count = getattr(_ctx, "tool_call_count", 0) + 1
                except Exception:
                    pass
        return self._serialize_execution_result(result, use_toon)
