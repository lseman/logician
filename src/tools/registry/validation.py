# -*- coding: utf-8 -*-
"""
Structured validation results — Zod-compatible typed error reporting.

Replaces raw _tool_error_payload dicts with a typed ValidationResult class
that callers (and the LLM) can inspect programmatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Sequence

from ..runtime import Tool, ToolParameter


class ValidationErrorCode(str, Enum):
    TOOL_NOT_FOUND = "tool_not_found"
    MISSING_CONTEXT = "missing_context"
    PERMISSION_BLOCKED = "permission_blocked"
    PLAN_MODE_BLOCKED = "plan_mode_blocked"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    SCHEMA_TYPE_VALIDATION_FAILED = "schema_type_validation_failed"
    INVALID_ARGUMENTS = "invalid_arguments"
    TOOL_EXCEPTION = "tool_exception"
    TOOL_NOT_REGISTERED = "tool_not_registered"
    DEPRECATED_TOOL = "deprecated_tool"
    EXECUTION_ERROR = "execution_error"


@dataclass(frozen=True)
class ZodIssue:
    code: ValidationErrorCode
    path: list[str]
    message: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    ok: bool
    issues: list[ZodIssue] = field(default_factory=list)
    tool_name: str = ""
    # Extended fields for schema validation
    missing_required: list[str] = field(default_factory=list)
    unknown_arguments: list[str] = field(default_factory=list)
    type_errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def error_type(self) -> str:
        if self.ok:
            return ""
        if self.issues:
            return self.issues[0].code.value
        return "execution_error"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "ok" if self.ok else "error",
            "error_type": self.error_type,
            "tool": self.tool_name,
            "issues": [
                {
                    "code": i.code.value,
                    "path": i.path,
                    "message": i.message,
                    "params": i.params,
                }
                for i in self.issues
            ],
        }
        if self.missing_required:
            payload["missing_required"] = self.missing_required
        if self.unknown_arguments:
            payload["unknown_arguments"] = self.unknown_arguments
        if self.type_errors:
            payload["type_errors"] = self.type_errors
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False)

    @classmethod
    def ok_result(cls, tool_name: str = "") -> "ValidationResult":
        return cls(ok=True, tool_name=tool_name)

    @classmethod
    def error_result(
        cls,
        tool_name: str,
        code: ValidationErrorCode,
        message: str,
        *,
        path: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> "ValidationResult":
        return cls(
            ok=False,
            issues=[
                ZodIssue(
                    code=code,
                    path=path or [],
                    message=message,
                    params=params or {},
                )
            ],
            tool_name=tool_name,
        )

    @classmethod
    def multi_error_result(
        cls,
        tool_name: str,
        issues: list[ZodIssue],
    ) -> "ValidationResult":
        return cls(ok=False, issues=issues, tool_name=tool_name)


# ---------------------------------------------------------------------------
# Standalone validate_tool_arguments for test_tools_runtime_validation.py
# ---------------------------------------------------------------------------

_JSON_TYPE_MAP: dict[str, str] = {
    "string": "string",
    "str": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
    "map": "object",
}

_COMMON_PARAM_ALIASES: dict[str, set[str]] = {
    "filepath": {"path", "file", "filename"},
}


def _json_schema_type(param_type: str) -> str:
    return _JSON_TYPE_MAP.get(str(param_type or "").strip().lower(), "string")


def _resolve_argument_alias(
    unknown_name: str,
    allowed_names: Sequence[str],
    common_aliases: dict[str, set[str]] | None = None,
) -> str | None:
    unknown_raw = str(unknown_name or "").strip()
    if not unknown_raw:
        return None
    unknown = unknown_raw.lower()
    if unknown in allowed_names:
        return unknown

    # Check common aliases
    if common_aliases:
        for canonical, aliases in common_aliases.items():
            if canonical not in allowed_names:
                continue
            if unknown in {a.lower() for a in aliases}:
                return canonical

    # Fuzzy match
    best_name, best_ratio = "", 0.0
    l = unknown_raw.strip().lower()
    for allowed in allowed_names:
        r = str(allowed or "").strip().lower()
        if not r:
            continue
        ratio = SequenceMatcher(None, l, r).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = allowed
    if best_name and best_ratio >= 0.86:
        return best_name
    return None


def _coerce_value(param_type: str, value: Any) -> Any:
    json_type = _json_schema_type(param_type)
    if value is None:
        return value
    if json_type == "string":
        return value if isinstance(value, str) else str(value)
    if json_type == "integer":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and float(value).is_integer():
            return int(value)
        if isinstance(value, str):
            raw = value.strip()
            raw_num = raw[1:] if raw[:1] in {"+", "-"} else raw
            if raw_num.isdigit():
                return int(raw)
    if json_type == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
    if json_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes", "on"}
        if isinstance(value, (int, float)):
            return bool(value)
    return value


def validate_tool_arguments(
    tool: Tool,
    arguments: dict[str, Any] | None,
    *,
    common_aliases: dict[str, set[str]] | None = None,
    strict_names: bool = True,
) -> tuple[dict[str, Any] | None, ValidationResult | None]:
    """Validate and prepare tool arguments.

    Returns (prepared_args, error). If error is None, arguments are valid.
    """
    if arguments is not None and not isinstance(arguments, dict):
        return None, ValidationResult.error_result(
            tool_name=tool.name,
            code=ValidationErrorCode.INVALID_ARGUMENTS,
            message="Tool arguments must be a JSON object/dict.",
        )

    raw = dict(arguments or {})
    if not tool.parameters:
        return raw, None

    param_by_name = {p.name: p for p in tool.parameters}
    allowed_names = list(param_by_name.keys())
    required_names = [p.name for p in tool.parameters if p.required]

    # Remap aliases
    prepared: dict[str, Any] = {}
    for key, value in raw.items():
        resolved = _resolve_argument_alias(key, allowed_names, common_aliases)
        if resolved:
            prepared[resolved] = value
        else:
            prepared[key] = value

    # Check required
    missing = sorted(set(n for n in required_names if n not in prepared))
    # Check unknown: keys that resolved to something not in allowed_names
    unknown_keys = sorted(k for k in prepared if k not in allowed_names) if strict_names else []

    if missing:
        err = ValidationResult.error_result(
            tool_name=tool.name,
            code=ValidationErrorCode.SCHEMA_VALIDATION_FAILED,
            message=f"Missing required parameters: {', '.join(missing)}",
            params={"missing_required": missing},
        )
        err.missing_required = missing
        if unknown_keys:
            err.unknown_arguments = unknown_keys
        return None, err

    if unknown_keys:
        err = ValidationResult.error_result(
            tool_name=tool.name,
            code=ValidationErrorCode.SCHEMA_VALIDATION_FAILED,
            message=f"Unknown parameters: {', '.join(unknown_keys)}",
            params={"unknown_arguments": unknown_keys},
        )
        err.unknown_arguments = unknown_keys
        return None, err

    # Coerce and validate types/enum
    type_errors: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    for name, value in prepared.items():
        if name not in param_by_name:
            continue
        param = param_by_name[name]
        coerced = _coerce_value(param.type, value)
        # Validate enum
        if param.enum and coerced not in param.enum:
            type_errors.append({
                "parameter": name,
                "actual": coerced,
                "expected": param.type,
                "allowed_values": param.enum,
            })
        final[name] = coerced

    if type_errors:
        err = ValidationResult.error_result(
            tool_name=tool.name,
            code=ValidationErrorCode.SCHEMA_TYPE_VALIDATION_FAILED,
            message=f"Type validation failed for {len(type_errors)} parameter(s)",
            params={"type_errors": type_errors},
        )
        err.type_errors = type_errors
        return None, err

    return final, None
