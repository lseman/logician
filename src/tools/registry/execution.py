from __future__ import annotations

import fnmatch
import inspect
import json
import time
from typing import Any

from ..compaction import ContentReplacementState, compact_result
from ..runtime import HAS_TOON, Tool, ToolCall, encode
from .validation import ValidationResult


class RegistryExecutionMixin:
    """ToolRegistry mixin."""

    @staticmethod
    def _is_validation_result_payload(payload: Any) -> bool:
        if isinstance(payload, ValidationResult):
            return True
        return hasattr(payload, "to_json") and (
            hasattr(payload, "ok") or hasattr(payload, "result")
        )

    @staticmethod
    def _validation_result_ok(payload: Any) -> bool:
        if hasattr(payload, "ok"):
            return bool(getattr(payload, "ok"))
        return bool(getattr(payload, "result", False))

    def _prepare_execute_tool(
        self,
        call: ToolCall,
        *,
        record_stats: bool = True,
    ) -> tuple[Tool | None, str | None]:
        if hasattr(self, "_ensure_catalog_configuration"):
            self._ensure_catalog_configuration()
        if hasattr(self, "_register_builtin_tools") and call.name not in self._tools:
            self._register_builtin_tools()
        tool = self.get(call.name)
        if not tool and hasattr(self, "_find_skill_ids_by_tool_name"):
            self._log.info(
                "Tool '%s' not found; searching skills for matching tool definitions", call.name
            )
            for skill_id in self._find_skill_ids_by_tool_name(call.name):
                if hasattr(self, "_load_tool_modules_for_skill_id"):
                    self._load_tool_modules_for_skill_id(skill_id)
                    tool = self.get(call.name)
                    if tool:
                        break

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
    def _hook_message(payload: Any, default: str) -> str:
        if isinstance(payload, ValidationResult):
            issues = payload.issues or []
            if issues:
                return issues[0].message
            return default
        if hasattr(payload, "message"):
            value = getattr(payload, "message")
            if isinstance(value, str) and value.strip():
                return value.strip()
        if isinstance(payload, dict):
            for key in ("message", "error", "reason", "detail"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(payload, str) and payload.strip():
            return payload.strip()
        return default

    @staticmethod
    def _invoke_tool_hook(
        callback: Any,
        *,
        prepared_args: dict[str, Any],
        ctx: Any | None,
        tool: Tool,
        call: ToolCall,
    ) -> Any:
        if not callable(callback):
            return None

        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            signature = None

        if signature is None:
            return callback(prepared_args)

        params = list(signature.parameters.values())
        if not params:
            return callback()

        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        mapping = {
            "input": prepared_args,
            "arguments": prepared_args,
            "args": prepared_args,
            "payload": prepared_args,
            "tool_input": prepared_args,
            "context": ctx,
            "ctx": ctx,
            "runtime_ctx": ctx,
            "tool": tool,
            "call": call,
            "tool_name": tool.name,
        }

        if has_var_kwargs:
            filtered_kwargs = {k: v for k, v in mapping.items() if v is not None}
            return callback(**filtered_kwargs)

        positional_args: list[Any] = []
        unresolved_named = False
        for index, param in enumerate(params):
            if param.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                continue
            if param.name in mapping:
                positional_args.append(mapping[param.name])
                continue
            unresolved_named = True
            break

        if not unresolved_named:
            return callback(*positional_args)

        positional_count = len(
            [
                p
                for p in params
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        )
        if positional_count <= 1:
            return callback(prepared_args)
        if positional_count == 2:
            return callback(prepared_args, ctx)
        return callback(prepared_args, ctx, tool)

    def _validate_tool_input(
        self,
        *,
        tool: Tool,
        call: ToolCall,
        prepared_args: dict[str, Any],
        ctx: Any | None,
    ) -> str | None:
        if not callable(tool.validate_input):
            return None

        try:
            result = self._invoke_tool_hook(
                tool.validate_input,
                prepared_args=prepared_args,
                ctx=ctx,
                tool=tool,
                call=call,
            )
        except Exception as exc:
            return self._tool_error_payload(
                call.name,
                f"validate_input hook failed: {exc}",
                error_type="execution_error",
            )

        if self._is_validation_result_payload(result):
            return None if self._validation_result_ok(result) else result.to_json()
        if result in (None, True):
            return None
        if result is False:
            return self._tool_error_payload(
                call.name,
                "Tool input was rejected by validate_input.",
                error_type="invalid_arguments",
            )
        if isinstance(result, dict):
            behavior = str(result.get("behavior") or "").strip().lower()
            explicit_success = (
                result.get("result") is True
                or result.get("ok") is True
                or result.get("valid") is True
                or result.get("success") is True
                or behavior in {"allow", "allowed", "approve", "approved", "passthrough", "ok"}
            )
            explicit_failure = (
                result.get("result") is False
                or result.get("ok") is False
                or result.get("valid") is False
                or result.get("success") is False
                or behavior in {"deny", "block", "blocked", "reject", "rejected", "ask"}
            )
            if explicit_success and not explicit_failure:
                return None
            if explicit_failure or any(k in result for k in ("message", "error", "reason")):
                return self._tool_error_payload(
                    call.name,
                    self._hook_message(result, "Tool input was rejected by validate_input."),
                    error_type="invalid_arguments",
                    validation_result=result,
                )
            return None
        if isinstance(result, str):
            return self._tool_error_payload(
                call.name,
                result,
                error_type="invalid_arguments",
            )
        if not result:
            return self._tool_error_payload(
                call.name,
                "Tool input was rejected by validate_input.",
                error_type="invalid_arguments",
            )
        return None

    def _check_tool_permissions(
        self,
        *,
        tool: Tool,
        call: ToolCall,
        prepared_args: dict[str, Any],
        ctx: Any | None,
    ) -> str | None:
        if not callable(tool.check_permissions):
            return None

        try:
            result = self._invoke_tool_hook(
                tool.check_permissions,
                prepared_args=prepared_args,
                ctx=ctx,
                tool=tool,
                call=call,
            )
        except Exception as exc:
            return self._tool_error_payload(
                call.name,
                f"check_permissions hook failed: {exc}",
                error_type="execution_error",
            )

        if self._is_validation_result_payload(result):
            return None if self._validation_result_ok(result) else result.to_json()
        if result in (None, True):
            return None
        if result is False:
            return self._tool_error_payload(
                call.name,
                "Tool execution was blocked by check_permissions.",
                error_type="permission_blocked",
            )
        if isinstance(result, dict):
            behavior = str(result.get("behavior") or "").strip().lower()
            explicit_allow = (
                result.get("result") is True
                or result.get("ok") is True
                or result.get("allowed") is True
                or result.get("success") is True
                or behavior in {"allow", "allowed", "approve", "approved", "passthrough", "ok"}
            )
            explicit_block = (
                result.get("result") is False
                or result.get("ok") is False
                or result.get("allowed") is False
                or result.get("success") is False
                or behavior in {"deny", "block", "blocked", "reject", "rejected", "ask"}
            )
            if explicit_allow and not explicit_block:
                return None
            if explicit_block or any(k in result for k in ("message", "error", "reason")):
                return self._tool_error_payload(
                    call.name,
                    self._hook_message(result, "Tool execution was blocked by check_permissions."),
                    error_type="permission_blocked",
                    permission_result=result,
                )
            return None
        if isinstance(result, str):
            return self._tool_error_payload(
                call.name,
                result,
                error_type="permission_blocked",
            )
        if not result:
            return self._tool_error_payload(
                call.name,
                "Tool execution was blocked by check_permissions.",
                error_type="permission_blocked",
            )
        return None

    @staticmethod
    def _get_content_replacement_state(ctx: Any | None) -> ContentReplacementState:
        state = getattr(ctx, "content_replacement_state", None) if ctx is not None else None
        if isinstance(state, ContentReplacementState):
            return state
        state = ContentReplacementState()
        if ctx is not None:
            setattr(ctx, "content_replacement_state", state)
        return state

    def _compact_execution_result(self, tool: Tool, result: Any, ctx: Any | None) -> Any:
        max_chars = int(getattr(tool, "max_result_size_chars", 100_000) or 0)
        if max_chars <= 0:
            return result

        compacted_result, metadata = compact_result(
            result,
            max_chars=max_chars,
            state=self._get_content_replacement_state(ctx),
        )
        if metadata is None:
            return result

        payload: dict[str, Any] = {
            "status": "error" if self._result_indicates_error(result) else "ok",
            "tool": tool.name,
            "compacted": True,
            "message": f"Tool result exceeded {max_chars} chars and was compacted to disk.",
            "result_path": metadata.get("file_path"),
            "preview": metadata.get("preview", ""),
            "original_size": metadata.get("original_size"),
        }
        if isinstance(result, dict):
            error_type = result.get("error_type")
            if error_type:
                payload["error_type"] = error_type
        if isinstance(compacted_result, str):
            payload["reference"] = compacted_result
        payload["original_result_type"] = type(result).__name__
        return payload

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

        input_validation_error = self._validate_tool_input(
            tool=tool,
            call=call,
            prepared_args=dict(prepared_args or {}),
            ctx=ctx,
        )
        if input_validation_error is not None:
            self._record_tool_execution(
                tool_name=call.name,
                duration_s=0.0,
                ok=False,
                error="tool validate_input rejected arguments",
            )
            return input_validation_error

        permission_check_error = self._check_tool_permissions(
            tool=tool,
            call=call,
            prepared_args=dict(prepared_args or {}),
            ctx=ctx,
        )
        if permission_check_error is not None:
            self._record_tool_execution(
                tool_name=call.name,
                duration_s=0.0,
                ok=False,
                error="tool check_permissions blocked execution",
            )
            return permission_check_error

        result, exec_dur, execution_error = self._run_tool_execution(
            call=call,
            tool=tool,
            prepared_args=prepared_args,
        )
        if execution_error is not None:
            # If the result is already a JSON-formatted string, return it.
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
        compacted_result = self._compact_execution_result(tool, result, ctx)
        return self._serialize_execution_result(compacted_result, use_toon)
