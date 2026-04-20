from __future__ import annotations

import fnmatch
import json
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

from ..runtime import SkillCard, Tool, ToolParameter, _safe_json_fallback
from .types import _COMMON_PARAM_ALIASES, ToolExecutionStats

_CODING_CAPABILITY_GROUPS: dict[str, dict[str, list[str]]] = {
    "discovery": {
        "required": [
            "list_directory",
            "read_file",
            "rg_search",
            "fd_find",
            "get_project_map",
            "get_file_outline",
        ],
        "nice_to_have": [
            "find_symbol",
            "find_references",
            "find_path",
            "find_in_file",
            "read_file_smart",
            "sed_read",
        ],
    },
    "editing": {
        "required": [
            "write_file",
            "edit_file_replace",
            "multi_edit",
            "apply_unified_diff",
            "apply_edit_block",
        ],
        "nice_to_have": [
            "multi_patch",
            "regex_replace",
            "rg_replace",
            "sed_replace",
            "show_diff",
            "count_in_file",
        ],
    },
    "execution": {
        "required": ["run_shell", "run_python"],
        "nice_to_have": [
            "start_background_process",
            "get_process_output",
            "send_input_to_process",
            "kill_process",
            "set_working_directory",
            "set_venv",
        ],
    },
    "quality": {
        "required": ["run_quality_check", "run_pytest", "run_ruff", "run_mypy"],
        "nice_to_have": ["auto_format", "smart_quality_gate", "parse_traceback"],
    },
    "version_control": {
        "required": ["git_status", "git_diff", "git_log"],
        "nice_to_have": ["git_checkpoint", "git_restore_checkpoint", "git_commit"],
    },
    "docs_context": {
        "required": ["web_search", "fetch_url", "rag_search"],
        "nice_to_have": ["github_read_file", "pypi_info", "docling_add_file"],
    },
    "meta": {
        "required": ["invoke_skill", "describe_tool", "search_tools", "skills_health"],
        "nice_to_have": ["think", "todo"],
    },
}

_DOCS_CONTEXT_CANONICAL_TOOL_NAMES = (
    "fetch_url",
    "web_search",
    "pypi_info",
    "github_read_file",
)

_SHELL_GIT_REVIEW_SPECS: dict[str, dict[str, Any]] = {
    "shell": {
        "skill_tools": ("run_shell",),
        "core_aliases": {
            "run_shell": ("bash",),
        },
        "promoted_to_core": (
            "set_venv",
            "set_working_directory",
            "install_packages",
            "show_coding_config",
            "start_background_process",
            "send_input_to_process",
            "get_process_output",
            "kill_process",
            "list_processes",
            "run_python",
            "check_imports",
            "list_installed_packages",
        ),
        "requires_shared_core_state": (),
        "promotion_note": (
            "Shell execution state now lives in a shared core runtime, so the environment/config/background-process helpers are always-on core tools. The coding shell skill is reduced to the higher-level run_shell workflow wrapper plus routing guidance."
        ),
    },
    "git": {
        "skill_tools": (
            "git_status",
            "git_diff",
            "git_log",
            "git_commit",
            "git_checkpoint",
            "git_restore_checkpoint",
            "git_blame",
        ),
        "core_aliases": {
            "git_status": ("get_git_status",),
            "git_diff": ("get_git_diff",),
        },
        "core_candidates_now": (),
        "requires_shared_core_state": (),
        "promotion_note": (
            "Keep git skill-owned for now: there are useful read-only overlaps with core, but the current git family is still mostly a higher-level workflow layer rather than a missing always-on primitive set."
        ),
    },
}

_CAPABILITY_TOOL_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "list_directory": ("list_dir",),
    "rg_search": ("grep_files",),
    "fd_find": ("glob_files",),
    "read_file_smart": ("read_file",),
    "run_shell": ("bash",),
    "edit_file_replace": ("edit_file",),
    "git_status": ("get_git_status",),
    "git_diff": ("get_git_diff",),
}

_CODING_SKILL_IDS = {
    "file_ops",
    "multi_edit",
    "web",
    "shell",
    "git",
    "quality",
    "repl",
    "patch",
    "edit_block",
    "rag",
    "explore",
    "search_replace",
    "meta_skills",
    "docling_context",
    "firecrawl",
    "think",
    "scratch",
    "todo",
}


class RegistryIntrospectionMixin:
    """ToolRegistry mixin."""

    @staticmethod
    def _matching_capability_tool_name(
        required_name: str,
        available_names: set[str],
    ) -> str | None:
        name = str(required_name or "").strip()
        if not name:
            return None
        candidates = (name, *_CAPABILITY_TOOL_EQUIVALENTS.get(name, ()))
        for candidate in candidates:
            if candidate in available_names:
                return candidate
        return None

    @staticmethod
    def _schema_validation_hint(tool_name: str) -> str | None:
        normalized = str(tool_name or "").strip()
        if normalized == "write_file":
            return (
                "write_file expects arguments like "
                '{"path":"/tmp/file.py","content":"full file text here"}. '
                "Pass all code in the single `content` field and do not add extra keys."
            )
        if normalized == "edit_file":
            return (
                "edit_file expects arguments like "
                '{"path":"src/app.py","old_string":"exact old text","new_string":"replacement text"}. '
                "Put the full match in `old_string` and the full replacement in `new_string`."
            )
        if normalized == "run_python":
            return (
                "run_python expects arguments like "
                '{"code":"print(\'hello\')\\n"}. '
                "Pass Python source code in the single `code` field, using real newlines or escaped `\\n`."
            )
        return None

    def _tool_error_payload(
        self,
        tool_name: str,
        error: str,
        *,
        error_type: str = "execution_error",
        **extra: Any,
    ) -> str:
        sj = self._execution_globals.get("_safe_json", _safe_json_fallback)
        payload: dict[str, Any] = {
            "status": "error",
            "error_type": error_type,
            "tool": tool_name,
            "error": error,
        }
        if extra:
            payload.update(extra)
        return sj(payload)

    @staticmethod
    def _fuzzy_ratio(left: str, right: str) -> float:
        l = str(left or "").strip().lower()
        r = str(right or "").strip().lower()
        if not l or not r:
            return 0.0
        return SequenceMatcher(None, l, r).ratio()

    @staticmethod
    def _normalize_name_key(name: str) -> str:
        return "".join(ch for ch in str(name or "").lower() if ch.isalnum())

    def _resolve_argument_alias(
        self,
        *,
        unknown_name: str,
        allowed_names: Sequence[str],
    ) -> str | None:
        unknown_raw = str(unknown_name or "").strip()
        if not unknown_raw:
            return None
        unknown = unknown_raw.lower()
        if unknown in allowed_names:
            return unknown

        unknown_norm = self._normalize_name_key(unknown_raw)
        if not unknown_norm:
            return None

        for allowed in allowed_names:
            if self._normalize_name_key(allowed) == unknown_norm:
                return allowed

        for canonical, aliases in _COMMON_PARAM_ALIASES.items():
            if canonical not in allowed_names:
                continue
            alias_norms = {self._normalize_name_key(alias) for alias in aliases}
            if unknown_norm in alias_norms:
                return canonical

        best_name = ""
        best_ratio = 0.0
        for allowed in allowed_names:
            ratio = self._fuzzy_ratio(unknown, allowed)
            if ratio > best_ratio:
                best_ratio = ratio
                best_name = allowed
        if best_name and best_ratio >= 0.86:
            return best_name
        return None

    @staticmethod
    def _fast_tokens(text: str) -> set[str]:
        out: set[str] = set()
        buf: list[str] = []
        for ch in str(text or "").lower():
            if ch.isalnum() or ch == "_":
                buf.append(ch)
                continue
            if len(buf) >= 2:
                out.add("".join(buf))
            buf.clear()
        if len(buf) >= 2:
            out.add("".join(buf))
        return out

    def _suggest_tool_names(self, query: str, *, max_items: int = 3) -> list[str]:
        q = str(query or "").strip().lower()
        if not q:
            return []
        scored: list[tuple[float, str]] = []
        for tool in self._tools.values():
            tname = tool.name.lower()
            text = f"{tool.name} {tool.description} {tool.skill_id or ''}".strip().lower()
            ratio = self._fuzzy_ratio(q, text)
            if tname == q:
                ratio = 1.0
            elif q in tname or tname in q:
                ratio = max(ratio, 0.88)
            if ratio >= 0.38:
                scored.append((ratio, tool.name))
        scored.sort(key=lambda item: item[0], reverse=True)
        out: list[str] = []
        for _, name in scored:
            if name not in out:
                out.append(name)
            if len(out) >= max(1, int(max_items)):
                break
        return out

    def _suggest_parameter_names(
        self,
        unknown_name: str,
        allowed_names: Sequence[str],
        *,
        max_items: int = 2,
    ) -> list[str]:
        unknown = str(unknown_name or "").strip().lower()
        if not unknown:
            return []
        scored: list[tuple[float, str]] = []
        for name in allowed_names:
            n = str(name or "").strip().lower()
            if not n:
                continue
            ratio = self._fuzzy_ratio(unknown, n)
            if unknown in n or n in unknown:
                ratio = max(ratio, 0.86)
            if ratio >= 0.45:
                scored.append((ratio, str(name)))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [name for _, name in scored[: max(1, int(max_items))]]

    def _coerce_param_value(self, ptype: str, value: Any) -> Any:
        json_type = self._json_schema_type(ptype)
        if value is None:
            return None

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
            raise ValueError(f"expected integer, got {type(value).__name__}")

        if json_type == "number":
            if isinstance(value, bool):
                return float(int(value))
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                raw = value.strip()
                try:
                    return float(raw)
                except ValueError:
                    pass
            raise ValueError(f"expected number, got {type(value).__name__}")

        if json_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str):
                raw = value.strip().lower()
                if raw in {"true", "1", "yes", "y", "on"}:
                    return True
                if raw in {"false", "0", "no", "n", "off"}:
                    return False
            raise ValueError(f"expected boolean, got {type(value).__name__}")

        if json_type == "array":
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, str):
                raw = value.strip()
                if not raw:
                    return []
                if raw.startswith("["):
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        return parsed
                    raise ValueError("expected JSON array")
                return [item.strip() for item in raw.split(",") if item.strip()]
            raise ValueError(f"expected array, got {type(value).__name__}")

        if json_type == "object":
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                raw = value.strip()
                if raw.startswith("{"):
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        return parsed
                    raise ValueError("expected JSON object")
            raise ValueError(f"expected object, got {type(value).__name__}")

        return value

    def _prepare_tool_arguments(
        self, tool: Tool, arguments: dict[str, Any] | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        raw, payload_error = self._normalize_tool_argument_payload(tool, arguments)
        if payload_error is not None:
            return None, payload_error

        # When metadata is missing, pass arguments through untouched.
        if not tool.parameters:
            return raw, None

        param_by_name = {p.name: p for p in tool.parameters}
        allowed_names = list(param_by_name.keys())
        required_names = [p.name for p in tool.parameters if p.required]
        normalized_raw, remapped_arguments = self._remap_tool_argument_aliases(
            raw,
            param_by_name=param_by_name,
            allowed_names=allowed_names,
        )

        schema_error = self._validate_tool_argument_names(
            tool=tool,
            normalized_raw=normalized_raw,
            param_by_name=param_by_name,
            required_names=required_names,
            allowed_names=allowed_names,
            remapped_arguments=remapped_arguments,
        )
        if schema_error is not None:
            return None, schema_error

        return self._coerce_and_validate_tool_argument_values(
            tool=tool,
            normalized_raw=normalized_raw,
            param_by_name=param_by_name,
        )

    def _normalize_tool_argument_payload(
        self,
        tool: Tool,
        arguments: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], str | None]:
        if arguments is not None and not isinstance(arguments, dict):
            err = self._tool_error_payload(
                tool.name,
                "Tool arguments must be a JSON object/dict.",
                error_type="invalid_arguments",
                received_type=type(arguments).__name__,
            )
            return {}, err
        raw = arguments if isinstance(arguments, dict) else {}
        return dict(raw), None

    def _remap_tool_argument_aliases(
        self,
        raw: dict[str, Any],
        *,
        param_by_name: dict[str, ToolParameter],
        allowed_names: Sequence[str],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        normalized_raw = dict(raw)
        remapped_arguments: dict[str, str] = {}
        for name in list(raw.keys()):
            if name in param_by_name:
                continue
            target = self._resolve_argument_alias(
                unknown_name=name,
                allowed_names=allowed_names,
            )
            if not target or target in normalized_raw:
                continue
            normalized_raw[target] = normalized_raw.pop(name)
            remapped_arguments[name] = target
        return normalized_raw, remapped_arguments

    def _validate_tool_argument_names(
        self,
        *,
        tool: Tool,
        normalized_raw: dict[str, Any],
        param_by_name: dict[str, ToolParameter],
        required_names: Sequence[str],
        allowed_names: Sequence[str],
        remapped_arguments: dict[str, str],
    ) -> str | None:
        unknown = [name for name in normalized_raw.keys() if name not in param_by_name]
        missing = [name for name in required_names if name not in normalized_raw]
        if not self._strict_tool_arg_validation or (not unknown and not missing):
            return None

        unknown_suggestions = {
            name: self._suggest_parameter_names(name, allowed_names) for name in unknown
        }
        hint = self._schema_validation_hint(tool.name)
        return self._tool_error_payload(
            tool.name,
            "Tool arguments failed schema validation.",
            error_type="schema_validation_failed",
            missing_required=missing,
            unknown_arguments=unknown,
            unknown_argument_suggestions=unknown_suggestions,
            allowed_arguments=list(allowed_names),
            remapped_arguments=remapped_arguments,
            usage_hint=hint,
        )

    def _coerce_and_validate_tool_argument_values(
        self,
        *,
        tool: Tool,
        normalized_raw: dict[str, Any],
        param_by_name: dict[str, ToolParameter],
    ) -> tuple[dict[str, Any] | None, str | None]:
        coerced: dict[str, Any] = {}
        type_errors: list[dict[str, Any]] = []
        for name, value in normalized_raw.items():
            if name not in param_by_name:
                continue
            param_meta = param_by_name[name]
            if not self._coerce_tool_argument_types:
                coerced_value = value
            else:
                try:
                    coerced_value = self._coerce_param_value(param_meta.type, value)
                except Exception as exc:
                    type_errors.append(
                        {
                            "name": name,
                            "expected_type": self._json_schema_type(param_meta.type),
                            "error": str(exc),
                        }
                    )
                    continue

            enum_values = param_meta.enum or []
            if enum_values and coerced_value not in enum_values:
                type_errors.append(
                    {
                        "name": name,
                        "expected_type": self._json_schema_type(param_meta.type),
                        "allowed_values": list(enum_values),
                        "error": f"expected one of {enum_values!r}",
                    }
                )
                continue

            coerced[name] = coerced_value

        if type_errors:
            err = self._tool_error_payload(
                tool.name,
                "Tool arguments have invalid types.",
                error_type="schema_type_validation_failed",
                type_errors=type_errors,
            )
            return None, err
        return coerced, None

    def _result_indicates_error(self, result: Any) -> bool:
        if isinstance(result, dict):
            status = str(result.get("status", "")).strip().lower()
            if status in {"error", "failed", "fail"}:
                return True
            if result.get("ok") is False or result.get("success") is False:
                return True
            return False

        if isinstance(result, str):
            text = result.strip()
            if text.lower().startswith("error:"):
                return True
            try:
                payload = json.loads(text)
            except Exception:
                return False
            if isinstance(payload, dict):
                status = str(payload.get("status", "")).strip().lower()
                if status in {"error", "failed", "fail"}:
                    return True
                if payload.get("ok") is False or payload.get("success") is False:
                    return True
        return False

    def _record_tool_execution(
        self,
        *,
        tool_name: str,
        duration_s: float,
        ok: bool,
        error: str = "",
    ) -> None:
        stats = self._tool_exec_stats.setdefault(
            tool_name,
            {
                "calls": 0,
                "errors": 0,
                "successes": 0,
                "avg_duration_s": 0.0,
                "last_duration_s": 0.0,
                "last_error": "",
                "last_called_at": 0.0,
            },
        )
        calls = int(stats.get("calls", 0)) + 1
        prev_avg = float(stats.get("avg_duration_s", 0.0))
        avg = prev_avg + ((float(duration_s) - prev_avg) / max(1, calls))
        stats["calls"] = calls
        stats["avg_duration_s"] = round(avg, 6)
        stats["last_duration_s"] = round(float(duration_s), 6)
        stats["last_called_at"] = round(time.time(), 6)
        if ok:
            stats["successes"] = int(stats.get("successes", 0)) + 1
        else:
            stats["errors"] = int(stats.get("errors", 0)) + 1
            stats["last_error"] = str(error or "")

    def tool_execution_stats(self) -> ToolExecutionStats:
        return {name: dict(stats) for name, stats in self._tool_exec_stats.items()}

    def get_grammar(self, tool_name: str) -> str | None:
        """Return the GBNF grammar for a tool, or None if not registered.

        Used by Agent.run() to enable llama.cpp constrained decoding for
        grammar-aware tools (e.g. write_file, edit_file, multi_edit).
        """
        return self._grammars.get(str(tool_name or "").strip())

    def _tool_public_metadata(self, tool: Tool) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "skill_id": tool.skill_id,
            "source_path": tool.source_path,
            "parameters": [
                {
                    "name": p.name,
                    "type": self._json_schema_type(p.type),
                    "required": bool(p.required),
                    "description": p.description,
                    "enum": list(p.enum) if p.enum else None,
                }
                for p in tool.parameters
            ],
            "runtime": tool.runtime.model_dump(),
            "stats": dict(self._tool_exec_stats.get(tool.name, {})),
        }

    def _describe_tool_tool(self, name: str) -> str:
        tool_name = str(name or "").strip()
        if not tool_name:
            return self._tool_error_payload(
                "describe_tool",
                "Parameter 'name' is required.",
                error_type="invalid_arguments",
            )

        tool = self.get(tool_name)
        if tool is None:
            suggestions = self._suggest_tool_names(tool_name, max_items=5)
            return self._tool_error_payload(
                "describe_tool",
                f"Tool '{tool_name}' is not registered.",
                error_type="tool_not_found",
                suggestions=suggestions,
            )

        return json.dumps(
            {
                "status": "ok",
                "tool": self._tool_public_metadata(tool),
            },
            ensure_ascii=False,
        )

    def _search_tools_tool(self, query: str, top_k: int = 8) -> str:
        q = str(query or "").strip()
        if not q:
            return self._tool_error_payload(
                "search_tools",
                "Parameter 'query' is required.",
                error_type="invalid_arguments",
            )
        k = max(1, min(20, int(top_k or 8)))
        q_l = q.lower()
        q_tokens = self._fast_tokens(q_l)

        scored: list[tuple[float, Tool]] = []
        for tool in self._tools.values():
            text = f"{tool.name} {tool.description} {tool.skill_id or ''}".strip().lower()
            ratio = self._fuzzy_ratio(q_l, text)
            tool_tokens = self._fast_tokens(text)
            overlap = 0.0
            if q_tokens and tool_tokens:
                overlap = len(q_tokens.intersection(tool_tokens)) / max(1, len(q_tokens))
            score = (ratio * 0.7) + (overlap * 0.3)
            if q_l in tool.name.lower():
                score += 0.25
            if score >= 0.15:
                scored.append((score, tool))

        scored.sort(key=lambda item: item[0], reverse=True)
        matches = []
        for score, tool in scored[:k]:
            md = self._tool_public_metadata(tool)
            md["score"] = round(float(score), 4)
            matches.append(md)

        return json.dumps(
            {
                "status": "ok",
                "query": q,
                "count": len(matches),
                "matches": matches,
            },
            ensure_ascii=False,
        )

    def _skills_health_tool(
        self,
        include_sources: bool = False,
        max_items: int = 25,
    ) -> str:
        limit = max(1, min(200, int(max_items or 25)))
        self._sync_catalog_with_registered_tools()
        sources = self._catalog.iter_skills_sources()
        readable, unreadable = self._scan_readable_skill_sources(sources)
        cards = list(self._catalog.skills.values())
        tool_backed_cards = [card for card in cards if card.tool_names]
        missing_tools_by_skill = self._missing_tools_by_skill(tool_backed_cards)
        discovered_superpowers = self._discovered_superpowers_sources(sources)
        payload = self._skills_health_payload(
            sources=sources,
            readable=readable,
            unreadable=unreadable,
            cards=cards,
            tool_backed_cards=tool_backed_cards,
            missing_tools_by_skill=missing_tools_by_skill,
            discovered_superpowers=discovered_superpowers,
        )
        self._append_skills_health_payload_details(
            payload=payload,
            include_sources=include_sources,
            limit=limit,
            sources=sources,
            readable=readable,
            unreadable=unreadable,
            discovered_superpowers=discovered_superpowers,
            missing_tools_by_skill=missing_tools_by_skill,
        )
        return json.dumps(payload, ensure_ascii=False)

    def _tool_permissions_tool(
        self,
        command: str = "view",
        pattern: str = "",
        mode: str = "",
        tool_name: str = "",
    ) -> str:
        ctx = self._execution_globals.get("ctx")
        if ctx is None:
            return self._tool_error_payload(
                "tool_permissions",
                "ctx is None (AppState not injected into ToolRegistry).",
                error_type="missing_context",
            )

        cmd = str(command or "view").strip().lower() or "view"
        if cmd not in {"view", "set", "remove", "clear"}:
            return self._tool_error_payload(
                "tool_permissions",
                "Invalid command. Use one of: view, set, remove, clear.",
                error_type="invalid_arguments",
            )

        current_rules = list(getattr(ctx, "permission_rules", []) or [])

        def _dump_rules() -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for item in list(getattr(ctx, "permission_rules", []) or []):
                if isinstance(item, dict):
                    row = {
                        "pattern": str(item.get("pattern") or "").strip(),
                        "mode": str(item.get("mode") or "").strip().lower(),
                    }
                else:
                    row = {
                        "pattern": str(getattr(item, "pattern", "") or "").strip(),
                        "mode": str(getattr(item, "mode", "") or "").strip().lower(),
                    }
                if row["pattern"] and row["mode"]:
                    rows.append(row)
            return rows

        if cmd == "set":
            rule_pattern = str(pattern or "").strip()
            rule_mode = str(mode or "").strip().lower()
            if not rule_pattern or rule_mode not in {"plan", "auto", "bypass"}:
                return self._tool_error_payload(
                    "tool_permissions",
                    "`set` requires both `pattern` and `mode` (plan|auto|bypass).",
                    error_type="invalid_arguments",
                )
            updated = []
            replaced = False
            for item in current_rules:
                item_pattern = (
                    str(item.get("pattern") or "").strip()
                    if isinstance(item, dict)
                    else str(getattr(item, "pattern", "") or "").strip()
                )
                if item_pattern == rule_pattern:
                    updated.append({"pattern": rule_pattern, "mode": rule_mode})
                    replaced = True
                else:
                    updated.append(item)
            if not replaced:
                updated.append({"pattern": rule_pattern, "mode": rule_mode})
            ctx.permission_rules = updated

        elif cmd == "remove":
            rule_pattern = str(pattern or "").strip()
            if not rule_pattern:
                return self._tool_error_payload(
                    "tool_permissions",
                    "`remove` requires `pattern`.",
                    error_type="invalid_arguments",
                )
            ctx.permission_rules = [
                item
                for item in current_rules
                if (
                    str(item.get("pattern") or "").strip()
                    if isinstance(item, dict)
                    else str(getattr(item, "pattern", "") or "").strip()
                )
                != rule_pattern
            ]

        elif cmd == "clear":
            ctx.permission_rules = []

        payload: dict[str, Any] = {
            "status": "ok",
            "command": cmd,
            "default_mode": str(getattr(ctx, "permission_default_mode", "auto") or "auto"),
            "rules": _dump_rules(),
        }
        preview_name = str(tool_name or "").strip()
        if preview_name:
            tool = self.get(preview_name)
            if tool is None:
                payload["tool_preview"] = {
                    "name": preview_name,
                    "status": "missing",
                    "suggestions": self._suggest_tool_names(preview_name, max_items=5),
                }
            else:
                runtime_mode = (
                    str(getattr(tool.runtime, "permission_mode", "") or "").strip().lower()
                )
                effective_mode = (
                    runtime_mode
                    if runtime_mode in {"plan", "auto", "bypass"}
                    else ("plan" if bool(getattr(tool.runtime, "writes_files", False)) else "auto")
                )
                matched_rule = None
                for row in payload["rules"]:
                    pattern_value = str(row.get("pattern") or "")
                    if pattern_value and (
                        fnmatch.fnmatchcase(preview_name, pattern_value)
                        or fnmatch.fnmatch(preview_name.lower(), pattern_value.lower())
                    ):
                        matched_rule = row
                payload["tool_preview"] = {
                    "name": preview_name,
                    "matched_rule": matched_rule,
                    "effective_mode": matched_rule.get("mode") if matched_rule else effective_mode,
                }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _scan_readable_skill_sources(
        sources: Sequence[Path],
    ) -> tuple[list[str], list[dict[str, str]]]:
        readable: list[str] = []
        unreadable: list[dict[str, str]] = []
        for src in sources:
            try:
                src.read_text(encoding="utf-8")
                readable.append(str(src))
            except Exception as exc:
                unreadable.append({"path": str(src), "error": str(exc)})
        return readable, unreadable

    def _missing_tools_by_skill(
        self, tool_backed_cards: Sequence[SkillCard]
    ) -> list[dict[str, Any]]:
        missing_tools_by_skill: list[dict[str, Any]] = []
        for card in tool_backed_cards:
            missing = [name for name in card.tool_names if name not in self._tools]
            if not missing:
                continue
            missing_tools_by_skill.append(
                {
                    "skill_id": card.id,
                    "skill_name": card.name,
                    "missing_tools": missing,
                }
            )
        return missing_tools_by_skill

    @staticmethod
    def _discovered_superpowers_sources(sources: Sequence[Path]) -> list[str]:
        return [
            str(src)
            for src in sources
            if src.name.upper() == "SKILL.MD" and "10_superpowers" in str(src).replace("\\", "/")
        ]

    @staticmethod
    def _normalized_path_text(path: str | Path) -> str:
        return str(path).replace("\\", "/")

    @staticmethod
    def _is_bootstrap_only_module(module_path: Path) -> bool:
        name = module_path.name.lower()
        if name in {"00_bootstrap.py", "bootstrap.py"}:
            return True
        try:
            text = module_path.read_text(encoding="utf-8")
        except Exception:
            return False
        return "_CODING_BOOTSTRAP_ONLY = True" in text or "BOOTSTRAP_ONLY = True" in text

    @staticmethod
    def _is_metadata_only_module(module_path: Path) -> bool:
        try:
            text = module_path.read_text(encoding="utf-8")
        except Exception:
            return False
        markers = (
            "_CODING_METADATA_ONLY = True",
            "_QOL_METADATA_ONLY = True",
            "metadata-only",
            "routing metadata",
        )
        return any(marker in text for marker in markers)

    def _tool_provider_info(self, tool_name: str) -> dict[str, Any] | None:
        tool = self._tools.get(tool_name)
        if tool is None:
            return None
        skill_id = str(tool.skill_id or "").strip().lower()
        source_path = str(tool.source_path or "")
        provider = "builtin" if source_path == "<builtin>" else skill_id or "unknown"
        return {
            "tool_name": tool.name,
            "skill_id": skill_id or None,
            "source_path": source_path or None,
            "provider": provider,
        }

    def _shell_git_overlap_review(self) -> dict[str, Any]:
        core_tool_names = {
            tool.name
            for tool in self._tools.values()
            if str(tool.skill_id or "").strip().lower() == "core"
        }
        review: dict[str, Any] = {}
        for family, spec in _SHELL_GIT_REVIEW_SPECS.items():
            skill_tools = [name for name in spec.get("skill_tools", ()) if name in self._tools]
            promoted_to_core = [
                name
                for name in spec.get("promoted_to_core", ())
                if (self._tool_provider_info(name) or {}).get("skill_id") == "core"
            ]
            requires_shared_core_state = [
                name for name in spec.get("requires_shared_core_state", ()) if name in self._tools
            ]
            exact_conflicts = sorted(set(skill_tools).intersection(core_tool_names))
            semantic_alias_overlaps = {
                name: [alias for alias in aliases if alias in core_tool_names]
                for name, aliases in dict(spec.get("core_aliases", {}) or {}).items()
                if any(alias in core_tool_names for alias in aliases)
            }
            distinct_skill_only_tools = sorted(
                name
                for name in skill_tools
                if name not in semantic_alias_overlaps and name not in exact_conflicts
            )
            review[family] = {
                "should_promote_to_core": False,
                "promoted_to_core": promoted_to_core,
                "requires_shared_core_state": requires_shared_core_state,
                "exact_core_name_conflicts": exact_conflicts,
                "semantic_alias_overlaps": semantic_alias_overlaps,
                "distinct_skill_only_tools": distinct_skill_only_tools,
                "promotion_note": str(spec.get("promotion_note") or "").strip(),
            }
        return review

    def _coding_capability_audit(self, cards: Sequence[SkillCard]) -> dict[str, Any]:
        tool_names = set(self._tools.keys())
        groups: dict[str, Any] = {}
        missing_required_tools: list[str] = []
        missing_tool_groups: dict[str, list[str]] = {}
        required_total = 0
        required_present_total = 0

        for group_name, spec in _CODING_CAPABILITY_GROUPS.items():
            required = list(dict.fromkeys(spec.get("required", [])))
            optional = list(dict.fromkeys(spec.get("nice_to_have", [])))
            required_present: list[str] = []
            required_missing: list[str] = []
            alias_resolutions: dict[str, str] = {}
            for name in required:
                matched = self._matching_capability_tool_name(name, tool_names)
                if matched is None:
                    required_missing.append(name)
                    continue
                required_present.append(name)
                if matched != name:
                    alias_resolutions[name] = matched
            optional_present = [name for name in optional if name in tool_names]

            required_total += len(required)
            required_present_total += len(required_present)
            for name in required_missing:
                missing_required_tools.append(name)
                missing_tool_groups.setdefault(name, []).append(group_name)

            coverage = (len(required_present) / max(1, len(required))) * 100.0
            groups[group_name] = {
                "required_total": len(required),
                "required_present": len(required_present),
                "required_missing_count": len(required_missing),
                "required_coverage_pct": round(coverage, 2),
                "required_missing_tools": required_missing,
                "required_alias_resolutions": alias_resolutions,
                "optional_total": len(optional),
                "optional_present": len(optional_present),
            }

            if group_name == "docs_context":
                provider_details = {
                    name: self._tool_provider_info(name)
                    for name in _DOCS_CONTEXT_CANONICAL_TOOL_NAMES
                    if self._tool_provider_info(name) is not None
                }
                canonical_core = all(
                    (provider_details.get(name) or {}).get("skill_id") == "core"
                    for name in ("fetch_url", "web_search")
                    if name in provider_details
                )
                groups[group_name].update(
                    {
                        "canonical_provider": "core" if canonical_core else "mixed",
                        "provider_details": provider_details,
                    }
                )

        coding_tools = [
            tool
            for tool in self._tools.values()
            if str(tool.skill_id or "").strip().lower() in _CODING_SKILL_IDS
        ]
        coding_skill_ids = sorted(
            {
                str(tool.skill_id or "").strip().lower()
                for tool in coding_tools
                if str(tool.skill_id or "").strip()
            }
        )
        coding_cards = [
            card
            for card in cards
            if (
                self._normalized_path_text(card.source_path).find("/coding/") >= 0
                or card.id in _CODING_SKILL_IDS
            )
        ]

        missing_required_unique = list(dict.fromkeys(missing_required_tools))
        required_coverage = (required_present_total / max(1, required_total)) * 100.0
        if required_coverage >= 97.0 and not missing_required_unique:
            maturity = "state_of_the_art"
        elif required_coverage >= 88.0:
            maturity = "strong"
        elif required_coverage >= 72.0:
            maturity = "good"
        else:
            maturity = "baseline"

        recommendations: list[str] = []
        for tool_name in missing_required_unique[:12]:
            groups_for_tool = ", ".join(missing_tool_groups.get(tool_name, []))
            recommendations.append(
                f"Add/fix `{tool_name}` ({groups_for_tool or 'core capability'})."
            )

        return {
            "status": "ok",
            "maturity": maturity,
            "required_tools_total": required_total,
            "required_tools_present": required_present_total,
            "required_coverage_pct": round(required_coverage, 2),
            "missing_required_tools": missing_required_unique,
            "missing_required_count": len(missing_required_unique),
            "coding_tools_total": len(coding_tools),
            "coding_skill_ids": coding_skill_ids,
            "coding_cards_total": len(coding_cards),
            "groups": groups,
            "overlap_review": self._shell_git_overlap_review(),
            "recommendations": recommendations,
        }

    def _coding_organization_audit(self) -> dict[str, Any]:
        coding_dir = self.skills_dir_path / "coding"
        if not coding_dir.is_dir():
            return {
                "status": "missing",
                "coding_dir": str(coding_dir),
                "coding_dir_exists": False,
                "coding_modules_count": 0,
                "issues_count": 1,
                "issues": [f"Missing coding skills directory: {coding_dir}"],
            }

        modules = sorted(
            p
            for p in coding_dir.rglob("*.py")
            if p.is_file()
            and p.name != "__init__.py"
            and not p.name.startswith("_")
            and "__pycache__" not in p.parts
        )
        module_names = [str(p.relative_to(coding_dir)).replace("\\", "/") for p in modules]

        no_prefix: list[str] = []
        duplicate_prefixes: list[int] = []
        large_prefix_gaps: list[list[int]] = []

        source_paths: set[str] = set()
        for tool in self._tools.values():
            source = str(tool.source_path or "").strip()
            if not source or not source.endswith(".py"):
                continue
            source_paths.add(self._normalized_path_text(source))
            try:
                source_paths.add(self._normalized_path_text(Path(source).resolve()))
            except Exception:
                pass

        modules_without_registered_tools: list[str] = []
        bootstrap_only_modules: list[str] = []
        metadata_only_modules: list[str] = []
        for module in modules:
            raw = self._normalized_path_text(module)
            resolved = self._normalized_path_text(module.resolve())
            if raw in source_paths or resolved in source_paths:
                continue
            if self._is_bootstrap_only_module(module):
                bootstrap_only_modules.append(module.name)
                continue
            if self._is_metadata_only_module(module):
                metadata_only_modules.append(module.name)
                continue
            modules_without_registered_tools.append(module.name)

        issues: list[str] = []
        if modules_without_registered_tools:
            issues.append(
                "Module(s) with no registered tool: "
                + ", ".join(modules_without_registered_tools[:8])
            )

        return {
            "status": "ok" if not issues else "needs_attention",
            "coding_dir": str(coding_dir),
            "coding_dir_exists": True,
            "coding_module_root": str(coding_dir),
            "coding_modules_count": len(module_names),
            "coding_modules": module_names,
            "bootstrap_only_modules": bootstrap_only_modules,
            "metadata_only_modules": metadata_only_modules,
            "modules_without_numeric_prefix": no_prefix,
            "duplicate_numeric_prefixes": duplicate_prefixes,
            "large_numeric_gaps": large_prefix_gaps,
            "modules_without_registered_tools": modules_without_registered_tools,
            "issues_count": len(issues),
            "issues": issues,
        }

    def _skills_health_payload(
        self,
        *,
        sources: Sequence[Path],
        readable: Sequence[str],
        unreadable: Sequence[dict[str, str]],
        cards: Sequence[SkillCard],
        tool_backed_cards: Sequence[SkillCard],
        missing_tools_by_skill: Sequence[dict[str, Any]],
        discovered_superpowers: Sequence[str],
    ) -> dict[str, Any]:
        guidance_only_cards = [card for card in cards if not card.tool_names]
        superpower_cards = [
            card for card in cards if Path(card.source_path).name.upper() == "SKILL.MD"
        ]
        coding = self._coding_capability_audit(cards)
        organization = self._coding_organization_audit()
        return {
            "status": "ok",
            "paths": {
                "skills_md_path": str(self.skills_md_path),
                "skills_dir_path": str(self.skills_dir_path),
                "skills_md_exists": self.skills_md_path.exists(),
                "skills_dir_exists": self.skills_dir_path.is_dir(),
            },
            "discovery": {
                "source_count": len(sources),
                "readable_count": len(readable),
                "unreadable_count": len(unreadable),
                "superpowers_skill_md_count": len(discovered_superpowers),
            },
            "catalog": {
                "skills_total": len(cards),
                "guidance_only_skills": len(guidance_only_cards),
                "tool_backed_skills": len(tool_backed_cards),
                "superpowers_skills": len(superpower_cards),
                "routing": {
                    "min_score": float(getattr(self._catalog, "_routing_min_score", 0.0) or 0.0),
                    "recall_k": int(getattr(self._catalog, "_routing_recall_k", 0) or 0),
                    "dense_enabled": bool(getattr(self._catalog, "_dense_enabled", False)),
                    "dense_model": str(getattr(self._catalog, "_dense_model_name", "") or ""),
                    "dense_available": bool(
                        getattr(self._catalog, "_dense_model", None) is not None
                    ),
                    "weights": dict(getattr(self._catalog, "_routing_weights", {}) or {}),
                },
            },
            "coding": coding,
            "organization": organization,
            "registry": {
                "tools_total": len(self._tools),
                "python_skill_tools": len(
                    [
                        tool
                        for tool in self._tools.values()
                        if str(tool.source_path or "").endswith(".py")
                    ]
                ),
                "builtin_tools": len(
                    [
                        tool
                        for tool in self._tools.values()
                        if str(tool.source_path or "") == "<builtin>"
                    ]
                ),
            },
            "checks": {
                "brainstorming_present": any(card.id == "brainstorming" for card in cards),
                "missing_tools_by_skill_count": len(missing_tools_by_skill),
                "coding_required_coverage_pct": float(
                    coding.get("required_coverage_pct", 0.0) or 0.0
                ),
                "coding_missing_required_count": int(coding.get("missing_required_count", 0) or 0),
                "coding_maturity": str(coding.get("maturity", "unknown") or "unknown"),
                "organization_issues_count": int(organization.get("issues_count", 0) or 0),
                "organization_status": str(organization.get("status", "unknown") or "unknown"),
            },
        }

    @staticmethod
    def _append_skills_health_payload_details(
        *,
        payload: dict[str, Any],
        include_sources: bool,
        limit: int,
        sources: Sequence[Path],
        readable: Sequence[str],
        unreadable: Sequence[dict[str, str]],
        discovered_superpowers: Sequence[str],
        missing_tools_by_skill: Sequence[dict[str, Any]],
    ) -> None:
        coding = payload.get("coding", {}) or {}
        recommendations = coding.get("recommendations", []) or []
        if recommendations:
            coding["recommendations"] = list(recommendations)[:limit]
        groups = coding.get("groups", {}) or {}
        if isinstance(groups, dict):
            for group_name, group_payload in groups.items():
                if not isinstance(group_payload, dict):
                    continue
                missing_tools = group_payload.get("required_missing_tools", []) or []
                group_payload["required_missing_tools"] = list(missing_tools)[:limit]

        organization = payload.get("organization", {}) or {}
        modules_without_tools = organization.get("modules_without_registered_tools", []) or []
        if modules_without_tools:
            organization["modules_without_registered_tools"] = list(modules_without_tools)[:limit]
        issues = organization.get("issues", []) or []
        if issues:
            organization["issues"] = list(issues)[:limit]

        if missing_tools_by_skill:
            payload["checks"]["missing_tools_by_skill"] = list(missing_tools_by_skill)[:limit]
        if unreadable:
            payload["discovery"]["unreadable"] = list(unreadable)[:limit]
        if not include_sources:
            return

        payload["discovery"]["sources"] = [str(src) for src in list(sources)[:limit]]
        payload["discovery"]["readable_sources"] = list(readable)[:limit]
        payload["discovery"]["superpowers_skill_md"] = list(discovered_superpowers)[:limit]
