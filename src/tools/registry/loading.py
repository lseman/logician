from __future__ import annotations

import ast
import importlib.machinery
import inspect
import json
import re
import sys
import types
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, get_args, get_origin

from ...mcp.client import MCPClient, MCPToolDef
from ..runtime import (
    ToolParameter,
    ToolRuntimeMetadata,
    _safe_json_fallback,
    materialize_tool,
)
from .catalog import ToolSection
from .types import _BuiltinToolDefinition


class RegistryLoadingMixin:
    """ToolRegistry mixin."""

    def load_tools_from_skills(self) -> None:
        python_count = self._load_python_skill_modules()
        builtin_count = self._register_builtin_tools()
        # Defer skill catalog building (SKILL.md parsing + routing index) to first
        # routing query via ensure_skill_catalog(). Building it here at startup adds
        # ~100-200ms of YAML parsing and tokenization with no benefit — the catalog
        # is only needed when the agent routes a query for the first time.
        if python_count > 0 or builtin_count > 0:
            self._version += 1
            self._invalidate_skill_resolution_cache()

    def _register_builtin_tools(self) -> int:
        """Register internal meta-tools that are always available."""
        registered = 0
        for definition in self._builtin_tool_definitions():
            if self._register_builtin_tool(definition):
                registered += 1

        if registered > 0:
            self._invalidate_skill_resolution_cache()
        return registered

    def _builtin_tool_definitions(self) -> list[_BuiltinToolDefinition]:
        return [
            {
                "name": "invoke_skill",
                "description": (
                    "Force a specific skill into the next routing pass, or optionally execute its primary tool directly. "
                    "Use when user asks to explicitly apply a named skill, or when you know exactly what arguments to pass."
                ),
                "parameters": [
                    ToolParameter(
                        name="skill",
                        type="string",
                        description="Skill id/name/query to force (e.g. 'brainstorming').",
                        required=True,
                    ),
                    ToolParameter(
                        name="reason",
                        type="string",
                        description="Optional rationale to store with the forced skill.",
                        required=False,
                    ),
                    ToolParameter(
                        name="top_k",
                        type="integer",
                        description="How many matched skills to force (default 1, max 3).",
                        required=False,
                    ),
                    ToolParameter(
                        name="args",
                        type="string",
                        description="Optional JSON dictionary string. If provided, the primary tool of the matched skill will be directly executed with these arguments.",
                        required=False,
                    ),
                ],
                "function": self._invoke_skill_tool,
                "doc": (
                    "**Description:** Force a skill into the next routing pass or execute it immediately.\n\n"
                    "**Parameters:**\n"
                    "- skill (string, required): Skill id, name, alias, or intent text.\n"
                    "- reason (string, optional): Why the skill is being forced.\n"
                    "- top_k (integer, optional): Number of matched skills to force.\n"
                    "- args (string, optional): JSON string of arguments to execute the primary tool directly."
                ),
            },
            {
                "name": "describe_tool",
                "description": "Return a tool's contract, parameter schema, and runtime stats.",
                "parameters": [
                    ToolParameter(
                        name="name",
                        type="string",
                        description="Registered tool name.",
                        required=True,
                    )
                ],
                "function": self._describe_tool_tool,
                "doc": (
                    "**Description:** Inspect a tool schema before calling it.\n\n"
                    "**Parameters:**\n"
                    "- name (string, required): Tool name to inspect."
                ),
            },
            {
                "name": "search_tools",
                "description": (
                    "Search tools by intent across names, descriptions, and skill ids."
                ),
                "parameters": [
                    ToolParameter(
                        name="query",
                        type="string",
                        description="Intent query (e.g. 'edit file', 'run tests').",
                        required=True,
                    ),
                    ToolParameter(
                        name="top_k",
                        type="integer",
                        description="Maximum matches to return (default 8, max 20).",
                        required=False,
                    ),
                ],
                "function": self._search_tools_tool,
                "doc": (
                    "**Description:** Find the best matching tools for an intent.\n\n"
                    "**Parameters:**\n"
                    "- query (string, required): Search phrase.\n"
                    "- top_k (integer, optional): Number of matches to return."
                ),
            },
            {
                "name": "skills_health",
                "description": (
                    "Inspect skill source discovery, catalog hydration, and key loading diagnostics."
                ),
                "parameters": [
                    ToolParameter(
                        name="include_sources",
                        type="boolean",
                        description="Include discovered/readable source file samples in the payload.",
                        required=False,
                    ),
                    ToolParameter(
                        name="max_items",
                        type="integer",
                        description="Maximum number of file entries to include per list (default 25, max 200).",
                        required=False,
                    ),
                ],
                "function": self._skills_health_tool,
                "doc": (
                    "**Description:** Diagnose why skills are or are not loading.\n\n"
                    "**Parameters:**\n"
                    "- include_sources (boolean, optional): Include source path samples.\n"
                    "- max_items (integer, optional): Max items per source list."
                ),
            },
            {
                "name": "tool_permissions",
                "description": (
                    "View or update wildcard-based tool permission rules with modes plan, auto, and bypass."
                ),
                "parameters": [
                    ToolParameter(
                        name="command",
                        type="string",
                        description="Command: view, set, remove, or clear.",
                        required=False,
                        enum=["view", "set", "remove", "clear"],
                    ),
                    ToolParameter(
                        name="pattern",
                        type="string",
                        description="Wildcard tool-name pattern such as `write_*`, `*git*`, or `bash`.",
                        required=False,
                    ),
                    ToolParameter(
                        name="mode",
                        type="string",
                        description="Permission mode for `set`: plan, auto, or bypass.",
                        required=False,
                        enum=["plan", "auto", "bypass"],
                    ),
                    ToolParameter(
                        name="tool_name",
                        type="string",
                        description="Optional tool name to preview its effective mode in `view` output.",
                        required=False,
                    ),
                ],
                "function": self._tool_permissions_tool,
                "doc": (
                    "**Description:** Inspect or change wildcard-based tool permission rules.\n\n"
                    "**Parameters:**\n"
                    "- command (string, optional): `view`, `set`, `remove`, or `clear`. Defaults to `view`.\n"
                    "- pattern (string, optional): Wildcard pattern for tool names.\n"
                    "- mode (string, optional): `plan`, `auto`, or `bypass` for `set`.\n"
                    "- tool_name (string, optional): Show the effective permission mode for a specific tool when viewing."
                ),
            },
        ]

    def _register_builtin_tool(self, definition: _BuiltinToolDefinition) -> bool:
        name = definition["name"]
        if name in self._tools:
            return False
        self._tools[name] = materialize_tool(
            definition["function"],
            name=name,
            description=definition["description"],
            parameters=definition["parameters"],
            doc=definition["doc"],
            skill_id="meta_skills",
            source_path="<builtin>",
        )
        self._catalog.tool_docs[name] = definition["doc"]
        self._log.info("✓ Loaded builtin tool: %s", name)
        return True

    def _load_python_skill_modules(self) -> int:
        if not self.skills_dir_path.is_dir():
            return 0

        import os as _os

        py_skill_files: list[Path] = []
        seen: set[str] = set()
        for root, dirs, files in _os.walk(str(self.skills_dir_path), followlinks=True):
            root_path = Path(root)
            try:
                rel_parts = root_path.relative_to(self.skills_dir_path).parts
            except Exception:
                rel_parts = ()

            if rel_parts:
                top_level = rel_parts[0]
                if self._is_lazy_skill_group_dir_name(
                    top_level
                ) and not self._is_lazy_skill_group_active(top_level):
                    dirs[:] = []
                    continue
            else:
                dirs[:] = [
                    d
                    for d in dirs
                    if not self._is_lazy_skill_group_dir_name(d)
                    or self._is_lazy_skill_group_active(d)
                ]

            # When a skill folder defines scripts/, only modules inside that tree
            # should be auto-loaded as executable tools.
            if "scripts" not in rel_parts and (root_path / "scripts").is_dir():
                dirs[:] = [d for d in dirs if d == "scripts"]
                continue

            for fname in files:
                if not fname.endswith(".py") or fname == "__init__.py" or fname.startswith("_"):
                    continue
                module_path = root_path / fname
                key = str(module_path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                py_skill_files.append(module_path)
        py_skill_files = sorted(py_skill_files)
        if not py_skill_files:
            return 0

        total_registered = 0
        for module_path in py_skill_files:
            total_registered += self._register_tools_from_python_module(module_path)

        self._log.info(
            "Loaded %d tool(s) from %d Python skill module(s)",
            total_registered,
            len(py_skill_files),
        )
        return total_registered

    @staticmethod
    def _safe_module_segment(name: str) -> str:
        segment = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name or "")).strip("_")
        if not segment:
            segment = "module"
        if segment[0].isdigit():
            segment = f"m_{segment}"
        return segment

    def _ensure_transient_package(
        self,
        package_name: str,
        package_path: Path | None,
    ) -> None:
        if package_name in sys.modules:
            return
        pkg = types.ModuleType(package_name)
        pkg.__package__ = package_name
        pkg.__path__ = [str(package_path)] if package_path is not None else []  # type: ignore[attr-defined]
        pkg.__spec__ = importlib.machinery.ModuleSpec(
            package_name,
            loader=None,
            is_package=True,
        )
        sys.modules[package_name] = pkg

    def _python_module_import_names(
        self, module_path: Path
    ) -> tuple[list[str], list[str], str, str]:
        rel_module_path = module_path.relative_to(self.skills_dir_path).with_suffix("")
        raw_parts = list(rel_module_path.parts)
        safe_parts = [self._safe_module_segment(part) for part in raw_parts]
        if not safe_parts:
            safe_parts = ["module"]

        root_package = "skills_runtime"
        package_parts = [root_package, *safe_parts[:-1]]
        package_name = ".".join(package_parts)
        module_name = ".".join([*package_parts, safe_parts[-1]])
        return raw_parts, safe_parts, package_name, module_name

    def _ensure_transient_module_packages(
        self, raw_parts: Sequence[str], safe_parts: Sequence[str]
    ) -> None:
        root_package = "skills_runtime"
        self._ensure_transient_package(root_package, self.skills_dir_path)
        current_package = root_package
        current_path = self.skills_dir_path
        for raw_part, safe_part in zip(raw_parts[:-1], safe_parts[:-1]):
            current_path = current_path / raw_part
            current_package = f"{current_package}.{safe_part}"
            self._ensure_transient_package(
                current_package,
                current_path if current_path.is_dir() else None,
            )

    @staticmethod
    def _create_transient_module(
        module_name: str,
        package_name: str,
        module_path: Path,
    ) -> types.ModuleType:
        transient_module = types.ModuleType(module_name)
        transient_module.__file__ = str(module_path)
        transient_module.__package__ = package_name
        transient_module.__spec__ = importlib.machinery.ModuleSpec(
            module_name,
            loader=None,
            is_package=False,
        )
        sys.modules[module_name] = transient_module
        return transient_module

    @staticmethod
    def _populate_module_execution_globals(
        *,
        execution_globals: dict[str, Any],
        module_path: Path,
        module_name: str,
        package_name: str,
        module_spec: importlib.machinery.ModuleSpec | None,
    ) -> None:
        execution_globals["__file__"] = str(module_path)
        execution_globals["__name__"] = module_name
        execution_globals["__package__"] = package_name
        execution_globals["__spec__"] = module_spec

    def _register_collected_python_tools(
        self,
        *,
        tool_entries: Sequence[tuple[Callable[..., Any], dict[str, Any]]],
        module_path: Path,
        skill_id: str,
        skill_meta: dict[str, Any] | None,
    ) -> int:
        registered = 0
        for tool_fn, meta in tool_entries:
            tool_name = str(meta.get("name") or getattr(tool_fn, "__name__", "")).strip()
            if not tool_name:
                continue
            if tool_name in self._tools:
                self._log.info(
                    "Skipping Python tool '%s' from %s because name is already registered",
                    tool_name,
                    module_path,
                )
                continue

            description = str(
                meta.get("description") or self._python_tool_doc_summary(tool_fn) or ""
            ).strip()
            try:
                tool_fn.__doc__ = self._compose_tool_docstring(
                    tool_fn,
                    base_description=description,
                )
            except (AttributeError, TypeError):
                self._log.warning(
                    "Unable to set __doc__ for tool %s from %s: %s",
                    tool_name,
                    module_path,
                    type(tool_fn).__name__,
                )
            params = self._parameters_from_signature(
                tool_fn,
                parameter_overrides=meta.get("parameters"),
            )
            wrapped = self._wrap_tool_function(tool_fn, tool_name)
            self._tools[tool_name] = materialize_tool(
                wrapped,
                name=tool_name,
                description=description,
                parameters=params,
                runtime=ToolRuntimeMetadata.from_tool_meta(meta),
                doc=str(meta.get("doc") or "").strip() or None,
                skill_id=skill_id,
                source_path=str(module_path),
                skill_meta=dict(skill_meta or {}),
            )
            doc_override = str(meta.get("doc") or "").strip()
            self._catalog.tool_docs[tool_name] = (
                doc_override or inspect.getdoc(tool_fn) or description
            )
            self._log.info("✓ Loaded Python tool: %s (%s)", tool_name, module_path)
            registered += 1
        return registered

    @staticmethod
    def _python_tool_doc_summary(tool_fn: Callable[..., Any]) -> str:
        doc = inspect.getdoc(tool_fn) or ""
        if not doc:
            return ""
        first_block = doc.split("\n\n", 1)[0].strip()
        first_line = next((line.strip() for line in first_block.splitlines() if line.strip()), "")
        if first_line.lower().startswith("use when:"):
            first_line = first_line[len("use when:") :].strip()
        return first_line

    def _python_tool_entries_from_module(
        self,
        *,
        execution_globals: dict[str, Any],
        module_path: Path,
        module_name: str,
    ) -> list[tuple[Callable[..., Any], dict[str, Any]]]:
        """Resolve exported tools for a Python skill module.

        First checks for __tools__ export. If not present, automatically discovers
        all public callable functions in the module (excluding special/internal names).
        """
        entries: list[tuple[Callable[..., Any], dict[str, Any]]] = []
        seen: set[int] = set()
        exports = execution_globals.get("__tools__")
        tool_meta_raw = execution_globals.get("__tool_meta__", {})
        tool_meta = tool_meta_raw if isinstance(tool_meta_raw, dict) else {}

        if exports is not None:
            # __tools__ export is present - use it
            if isinstance(exports, (str, bytes)) or not isinstance(exports, Sequence):
                self._log.warning(
                    "Ignoring invalid __tools__ export in %s; expected a sequence of callables or function names",
                    module_path,
                )
                return exports

            for export in exports:
                tool_fn: Callable[..., Any] | None = None
                if callable(export):
                    tool_fn = export
                elif isinstance(export, str):
                    candidate = execution_globals.get(export)
                    if callable(candidate):
                        tool_fn = candidate
                if tool_fn is None:
                    self._log.warning(
                        "Ignoring invalid __tools__ entry %r in %s",
                        export,
                        module_path,
                    )
                    continue
                if id(tool_fn) in seen:
                    continue
                meta = getattr(tool_fn, "__llm_tool_meta__", {})
                explicit_meta = tool_meta.get(getattr(tool_fn, "__name__", ""), {})
                if isinstance(explicit_meta, dict):
                    meta = {**meta, **explicit_meta}
                entries.append((tool_fn, meta))
                seen.add(id(tool_fn))
        else:
            # No __tools__ export: automatically discover all public callable functions
            self._log.debug(
                "No __tools__ export found in %s; auto-discovering public callables",
                module_path,
            )
            for name, obj in execution_globals.items():
                # Skip internal/protected names and non-callables
                if name.startswith("_"):
                    continue
                if not inspect.isfunction(obj):
                    continue
                if getattr(obj, "__module__", None) != module_name:
                    continue
                # Skip common non-tool objects
                if name in {
                    "__name__",
                    "__file__",
                    "__package__",
                    "__spec__",
                    "__builtins__",
                    "json",
                    "np",
                    "pd",
                    "ctx",
                    "call_tool",
                    "_safe_json",
                }:
                    continue
                if id(obj) in seen:
                    continue
                meta = getattr(obj, "__llm_tool_meta__", {})
                explicit_meta = tool_meta.get(name, {})
                if isinstance(explicit_meta, dict):
                    meta = {**meta, **explicit_meta}
                entries.append((obj, meta))
                seen.add(id(obj))

        return entries

    def _merge_python_module_globals(self, execution_globals: dict[str, Any]) -> None:
        for key, value in execution_globals.items():
            if key in {"__name__", "__file__", "__skill__", "__tools__", "__tool_meta__"}:
                continue
            self._execution_globals[key] = value
        self._execution_globals["call_tool"] = self.call_tool
        if "ctx" not in self._execution_globals:
            self._execution_globals["ctx"] = None
        if "_safe_json" not in self._execution_globals:
            self._execution_globals["_safe_json"] = _safe_json_fallback

    def _register_tools_from_python_module(self, module_path: Path) -> int:
        code = module_path.read_text(encoding="utf-8")
        execution_globals = dict(self._execution_globals)
        execution_globals.pop("__skill__", None)
        execution_globals.pop("__tools__", None)
        execution_globals.pop("__tool_meta__", None)
        raw_parts, safe_parts, package_name, module_name = self._python_module_import_names(
            module_path
        )
        self._ensure_transient_module_packages(raw_parts, safe_parts)
        transient_module = self._create_transient_module(
            module_name=module_name,
            package_name=package_name,
            module_path=module_path,
        )
        self._populate_module_execution_globals(
            execution_globals=execution_globals,
            module_path=module_path,
            module_name=module_name,
            package_name=package_name,
            module_spec=transient_module.__spec__,
        )

        try:
            exec(code, execution_globals, execution_globals)
        except Exception as exc:
            self._log.warning(
                "Skipping Python skill module %s due to import/runtime error: %s",
                module_path,
                exc,
            )
            return 0
        transient_module.__dict__.update(execution_globals)
        skill_id = self._catalog._skill_id_from_source(module_path)
        skill_meta_raw = execution_globals.get("__skill__", {})
        skill_meta = dict(skill_meta_raw) if isinstance(skill_meta_raw, dict) else None
        if skill_meta:
            self._catalog.register_python_skill_metadata(
                skill_id=skill_id,
                source_path=str(module_path),
                skill_meta=skill_meta,
            )
        tool_entries = self._python_tool_entries_from_module(
            execution_globals=execution_globals,
            module_path=module_path,
            module_name=module_name,
        )
        registered = self._register_collected_python_tools(
            tool_entries=tool_entries,
            module_path=module_path,
            skill_id=skill_id,
            skill_meta=skill_meta,
        )
        self._merge_python_module_globals(execution_globals)
        # Collect GBNF grammars exported by skill modules (llama.cpp constrained decoding)
        grammars_raw = execution_globals.get("__grammars__", {})
        if isinstance(grammars_raw, dict):
            for tool_name, grammar in grammars_raw.items():
                if isinstance(tool_name, str) and isinstance(grammar, str):
                    self._grammars[tool_name] = grammar
        if registered > 0:
            self._invalidate_skill_resolution_cache()
        return registered

    def _default_parameter_description(
        self,
        *,
        name: str,
        ptype: str,
        required: bool,
        default: Any = inspect._empty,
    ) -> str:
        label = str(name or "value").replace("_", " ").strip()
        if not label:
            label = "value"
        json_type = self._json_schema_type(ptype)
        if required:
            return f"Required {label} ({json_type})."
        if default is inspect._empty:
            return f"Optional {label} ({json_type})."
        return f"Optional {label} ({json_type}, default={default!r})."

    def _parameters_from_signature(
        self,
        func: Callable[..., Any],
        *,
        parameter_overrides: dict[str, Any] | None = None,
    ) -> list[ToolParameter]:
        sig = inspect.signature(func)
        out: list[ToolParameter] = []
        overrides = parameter_overrides if isinstance(parameter_overrides, dict) else {}
        for p in sig.parameters.values():
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            ann = p.annotation
            ptype = "string"
            enum_values: list[Any] | None = None
            if ann is not inspect._empty:
                ptype = self._annotation_to_type_name(ann)
                enum_values = self._annotation_literal_values(ann)
            elif p.default is not inspect._empty and p.default is not None:
                ptype = self._annotation_to_type_name(type(p.default))
            required = p.default is inspect._empty
            override = overrides.get(p.name)
            if isinstance(override, dict) and str(override.get("description", "")).strip():
                description = str(override.get("description")).strip()
            elif isinstance(override, str) and override.strip():
                description = override.strip()
            else:
                description = self._default_parameter_description(
                    name=p.name,
                    ptype=ptype,
                    required=required,
                    default=p.default,
                )
            out.append(
                ToolParameter(
                    name=p.name,
                    type=ptype,
                    description=description,
                    required=required,
                    enum=enum_values,
                )
            )
        return out

    @staticmethod
    def _normalize_enum_values(values: Sequence[Any] | None) -> list[Any] | None:
        if not values:
            return None
        out: list[Any] = []
        seen: set[str] = set()
        for value in values:
            item_method = getattr(value, "item", None)
            module_name = type(value).__module__
            if callable(item_method) and module_name.startswith("numpy"):
                try:
                    norm = item_method()
                except Exception:
                    norm = value
            else:
                norm = value
            if norm is not None and not isinstance(norm, (str, int, float, bool)):
                continue
            key = json.dumps(norm, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            out.append(norm)
        return out or None

    def _literal_values_from_string_annotation(self, annotation: str) -> list[Any] | None:
        text = str(annotation or "").strip()
        lower = text.lower()
        if not lower:
            return None
        if lower.startswith("typing.literal["):
            inner = text[text.find("[") + 1 : -1]
        elif lower.startswith("literal[") and text.endswith("]"):
            inner = text[text.find("[") + 1 : -1]
        else:
            return None

        try:
            parsed = ast.parse(f"[{inner}]", mode="eval").body
        except Exception:
            return None
        if not isinstance(parsed, ast.List):
            return None
        values: list[Any] = []
        for element in parsed.elts:
            if not isinstance(element, ast.Constant):
                return None
            values.append(element.value)
        return self._normalize_enum_values(values)

    def _annotation_literal_values(self, annotation: Any) -> list[Any] | None:
        if isinstance(annotation, str):
            return self._literal_values_from_string_annotation(annotation)
        if get_origin(annotation) is Literal:
            return self._normalize_enum_values(list(get_args(annotation)))
        return None

    def _annotation_to_type_name(self, annotation: Any) -> str:
        if isinstance(annotation, str):
            text = annotation.strip().lower()
            if not text:
                return "string"
            literal_values = self._literal_values_from_string_annotation(annotation)
            if literal_values:
                first_non_null = next((v for v in literal_values if v is not None), None)
                if first_non_null is not None:
                    return self._annotation_to_type_name(type(first_non_null))
            # Handle stringified unions, e.g. "int | None" from postponed annotations.
            if "|" in text:
                parts = [part.strip() for part in text.split("|")]
                for part in parts:
                    if part in {"none", "nonetype"}:
                        continue
                    return self._annotation_to_type_name(part)
                return "string"
            if text.startswith("optional[") and text.endswith("]"):
                inner = text[len("optional[") : -1].strip()
                return self._annotation_to_type_name(inner)
            if text.startswith("list[") or text.startswith("typing.list["):
                return "list"
            if text.startswith("dict[") or text.startswith("typing.dict["):
                return "dict"
            if text in {"int", "builtins.int", "integer"}:
                return "int"
            if text in {"float", "builtins.float", "number"}:
                return "float"
            if text in {"bool", "builtins.bool", "boolean"}:
                return "bool"
            if text in {"list", "builtins.list", "array"}:
                return "list"
            if text in {"dict", "builtins.dict", "object", "map"}:
                return "dict"
            if text in {"str", "builtins.str", "string"}:
                return "string"
            return "string"

        origin = get_origin(annotation)
        if origin is list:
            return "list"
        if origin is dict:
            return "dict"
        if origin is Literal:
            args = get_args(annotation)
            if args:
                return self._annotation_to_type_name(type(args[0]))
            return "string"
        if origin in {types.UnionType}:
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if args:
                return self._annotation_to_type_name(args[0])
            return "string"
        if str(origin) in {"typing.Union", "types.UnionType"}:
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if args:
                return self._annotation_to_type_name(args[0])
            return "string"
        if annotation in {int}:
            return "int"
        if annotation in {float}:
            return "float"
        if annotation in {bool}:
            return "bool"
        if annotation in {list}:
            return "list"
        if annotation in {dict}:
            return "dict"
        if annotation in {str}:
            return "string"
        return "string"

    @staticmethod
    def _docstring_triggers(tool_fn: Callable[..., Any]) -> list[str]:
        return [f"user asks to {getattr(tool_fn, '__name__', 'run this tool').replace('_', ' ')}"]

    @staticmethod
    def _docstring_avoid_when() -> str:
        return "another specialized tool is a clearer match"

    def _docstring_inputs_text(
        self,
        *,
        tool_fn: Callable[..., Any],
    ) -> str:
        sig = inspect.signature(tool_fn)
        args: list[str] = []
        for p in sig.parameters.values():
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if p.default is inspect._empty:
                args.append(f"{p.name} (required)")
            else:
                args.append(f"{p.name} (optional, default={p.default!r})")
        return ", ".join(args) if args else "none"

    @staticmethod
    def _docstring_returns_text() -> str:
        return "JSON string payload with status, results, and/or error details."

    @staticmethod
    def _docstring_side_effects_text() -> str:
        return "May read/update shared tool context depending on implementation."

    def _compose_tool_docstring(
        self,
        tool_fn: Callable[..., Any],
        *,
        base_description: str,
    ) -> str:
        existing_doc = inspect.getdoc(tool_fn) or ""
        if existing_doc:
            return existing_doc

        description = (
            base_description.strip() or "Run this tool when it best fits the user request."
        )
        triggers = self._docstring_triggers(tool_fn)
        avoid_when = self._docstring_avoid_when()
        inputs_text = self._docstring_inputs_text(
            tool_fn=tool_fn,
        )
        returns_text = self._docstring_returns_text()
        side_effects = self._docstring_side_effects_text()

        return (
            f"Use when: {description}\n\n"
            f"Triggers: {', '.join(triggers)}.\n"
            f"Avoid when: {avoid_when}.\n"
            f"Inputs: {inputs_text}.\n"
            f"Returns: {returns_text}.\n"
            f"Side effects: {side_effects}"
        )

    def reload_skills(self) -> int:
        """Reload all skill files from disk, re-registering every tool.

        Returns the number of tools registered after the reload.
        Preserves the execution globals (ctx, etc.) and bootstrap state.
        """
        self._log.info("Reloading skills from disk…")
        # Clear registered tools and catalog state so everything is re-parsed.
        self._tools.clear()
        self._catalog._skills.clear()
        self._catalog._tool_docs.clear()
        self._catalog._all_tool_sections.clear()
        self._catalog.set_active_lazy_skill_groups(self._active_lazy_skill_groups)
        # Allow bootstrap to re-run so any helper globals are refreshed.
        self._bootstrapped = False
        self._invalidate_skill_resolution_cache()
        self.load_tools_from_skills()
        n = len(self._tools)
        self._log.info("Reload complete — %d tools registered", n)
        return n

    def _mcp_registry_tool_name(self, *, client_name: str, safe_name: str) -> str:
        if safe_name not in self._tools:
            return safe_name
        return f"{client_name}__{safe_name}"

    @staticmethod
    def _mcp_tool_parameters(tdef: MCPToolDef) -> list[ToolParameter]:
        from ..runtime import ToolParameter as _TP

        return [
            _TP(
                name=p.name,
                type=p.type,
                description=p.description,
                required=p.required,
            )
            for p in tdef.parameters
        ]

    @staticmethod
    def _mcp_tool_caller(client: MCPClient, mcp_tool_name: str) -> Callable[..., str]:
        def _call(**kwargs: Any) -> str:
            try:
                result = client.call_tool(mcp_tool_name, kwargs)
                if isinstance(result, str):
                    return result
                return json.dumps(result, ensure_ascii=False, indent=2)
            except Exception as exc:
                return json.dumps(
                    {"status": "error", "error": str(exc), "tool": mcp_tool_name},
                    ensure_ascii=False,
                )

        _call.__name__ = mcp_tool_name
        _call.__doc__ = f"MCP tool '{mcp_tool_name}' via server '{client.name}'."
        return _call

    def _register_mcp_tool(self, *, client: MCPClient, tdef: MCPToolDef) -> str:
        reg_name = self._mcp_registry_tool_name(
            client_name=client.name,
            safe_name=tdef.safe_name,
        )
        self._tools[reg_name] = materialize_tool(
            self._mcp_tool_caller(client, tdef.name),
            name=reg_name,
            description=tdef.description,
            parameters=self._mcp_tool_parameters(tdef),
            skill_id=f"mcp__{client.name}",
            source_path=client.url,
        )
        self._log.info(
            "✓ MCP tool registered: %s (server=%s mcp_name=%s)",
            reg_name,
            client.name,
            tdef.name,
        )
        return reg_name

    def load_from_mcp_server(self, client: MCPClient) -> int:
        """Connect to an MCP server, discover its tools, and register each one.

        The tool name in the registry uses the safe (Python-identifier) form
        so the LLM can always invoke it.  The actual MCP tool name (which may
        contain hyphens) is stored in the closure and used verbatim when
        calling the server.

        Returns the number of newly registered tools.
        """
        try:
            tool_defs: list[MCPToolDef] = client.list_tools()
        except Exception as exc:
            self._log.error("MCP '%s': failed to list tools: %s", client.name, exc)
            return 0

        registered = 0
        for tdef in tool_defs:
            self._register_mcp_tool(client=client, tdef=tdef)
            registered += 1

        self._version += 1
        if registered > 0:
            self._invalidate_skill_resolution_cache()
        self._log.info("MCP '%s': %d tool(s) registered", client.name, registered)
        return registered

    def _run_bootstrap_blocks(self, skill_contents: list[tuple[Path, str]]) -> None:
        if self._bootstrapped:
            return

        blocks = []
        for _, content in skill_contents:
            blocks.extend(self._catalog.parse_bootstrap_sections(content))

        if not blocks:
            self._bootstrapped = True
            return

        self._log.info("Found %d bootstrap section(s) in skills sources", len(blocks))
        for b in blocks:
            name = b.get("name", "bootstrap")
            code = b.get("code", "")
            if not code.strip():
                self._log.warning("Bootstrap '%s': no python code fence found", name)
                continue
            try:
                exec(code, self._execution_globals, self._execution_globals)
                self._log.info("✓ Ran bootstrap: %s", name)
            except Exception as e:
                self._log.error("Bootstrap '%s' failed: %s", name, e)

        self._execution_globals["call_tool"] = self.call_tool
        if "ctx" not in self._execution_globals:
            self._execution_globals["ctx"] = None
        if "_safe_json" not in self._execution_globals:
            self._execution_globals["_safe_json"] = _safe_json_fallback
        self._bootstrapped = True

    def _resolve_tool_callable_from_scope(
        self,
        *,
        expected_name: str,
        local_scope: dict[str, Any],
    ) -> Callable[..., Any] | None:
        func = local_scope.get(expected_name)
        if func is not None:
            return func

        callables = [v for v in local_scope.values() if callable(v)]
        if callables:
            func = callables[0]
            self._log.warning(
                "Tool '%s': using function '%s' (expected name mismatch)",
                expected_name,
                getattr(func, "__name__", "unknown"),
            )
            return func
        self._log.error("No callable found for tool: %s", expected_name)
        return None

    def _register_materialized_tool(
        self,
        *,
        name: str,
        description: str,
        parameters: list[ToolParameter],
        wrapped: Callable[..., Any],
        skill_id: str,
        source_path: str,
        doc_content: str,
    ) -> None:
        self._tools[name] = materialize_tool(
            wrapped,
            name=name,
            description=description,
            parameters=parameters,
            doc=doc_content,
            skill_id=skill_id or None,
            source_path=source_path or None,
        )
        self._catalog.tool_docs[name] = self._catalog.tool_doc_from_section(doc_content)
        self._log.info("✓ Loaded tool: %s", name)
        self._invalidate_skill_resolution_cache()

    def _register_tool_from_section(self, tool_info: ToolSection) -> None:
        name = tool_info["name"]
        description = tool_info.get("description", "")
        parameters = tool_info.get("parameters", [])
        code = tool_info.get("code", "")
        source_path = tool_info.get("source_path", "")
        skill_id = tool_info.get("skill_id", "")

        if not code:
            self._log.warning("No code found for tool: %s", name)
            return

        local_scope: dict[str, Any] = {}
        exec(code, self._execution_globals, local_scope)

        func = self._resolve_tool_callable_from_scope(
            expected_name=name,
            local_scope=local_scope,
        )
        if func is None:
            return

        wrapped = self._wrap_tool_function(func, name)
        self._register_materialized_tool(
            name=name,
            description=description,
            parameters=parameters,
            wrapped=wrapped,
            skill_id=skill_id,
            source_path=source_path,
            doc_content=tool_info["content"],
        )

    def _wrap_tool_function(self, func: Callable, tool_name: str) -> Callable:
        def wrapped(**kwargs):
            try:
                func_globals = getattr(func, "__globals__", None)
                if isinstance(func_globals, dict):
                    func_globals["ctx"] = self._execution_globals.get("ctx")
                    func_globals["_safe_json"] = self._execution_globals.get(
                        "_safe_json", _safe_json_fallback
                    )
                    func_globals["call_tool"] = self.call_tool
                return func(**kwargs)
            except Exception as e:
                self._log.exception("Error in tool '%s'", tool_name)
                sj = self._execution_globals.get("_safe_json", _safe_json_fallback)
                try:
                    return sj({"status": "error", "error": str(e), "tool": tool_name})
                except Exception:
                    return json.dumps(
                        {"status": "error", "error": str(e), "tool": tool_name},
                        ensure_ascii=False,
                    )

        wrapped.__name__ = getattr(func, "__name__", tool_name)
        wrapped.__doc__ = getattr(func, "__doc__", None)
        return wrapped
