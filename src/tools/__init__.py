# -*- coding: utf-8 -*-
"""
Unified tools package with skill-aware loading and prompt rendering.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypedDict

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..mcp.client import MCPClient, MCPToolDef
from .catalog import SkillCatalog, ToolSection
from .parser import parse_tool_call_strict, parse_tool_calls
from .runtime import (
    HAS_TOON,
    Context,
    SkillCard,
    SkillSelection,
    Tool,
    ToolCall,
    ToolParameter,
    _safe_json_fallback,
    check_optional_deps,
    encode,
)


class _OpenAIToolFunction(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class _OpenAIToolSchema(TypedDict):
    type: str
    function: _OpenAIToolFunction


class ToolRegistry:
    """
    Registry for managing and executing tools loaded from SKILLS sources.
    """

    def __init__(self, auto_load_from_skills: bool = True) -> None:
        self._tools: dict[str, Tool] = {}
        self._log = get_logger("agent.tools")
        self.skills_md_path = Path(__file__).resolve().parents[2] / "SKILLS.md"
        self.skills_dir_path = Path(__file__).resolve().parents[2] / "skills"
        self._catalog = SkillCatalog(
            skills_md_path=self.skills_md_path,
            skills_dir_path=self.skills_dir_path,
            log=self._log,
        )

        self._execution_globals: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "json": json,
            "np": np,
            "pd": pd,
            "ctx": None,
        }
        self._execution_globals["call_tool"] = self.call_tool
        self._execution_globals["_safe_json"] = _safe_json_fallback
        self._legacy_py_tool_metadata_by_module: Optional[
            dict[str, dict[str, dict[str, Any]]]
        ] = None

        self._bootstrapped: bool = False
        self._version: int = 0

        if auto_load_from_skills and self._catalog.iter_skills_sources():
            self._log.info(
                "Auto-loading tools from %d source(s)",
                len(self._catalog.iter_skills_sources()),
            )
            self.load_tools_from_skills()

    def __repr__(self) -> str:
        return (
            f"ToolRegistry(tools={len(self._tools)}, "
            f"skills_md_path={str(self.skills_md_path)!r}, "
            f"skills_dir_path={str(self.skills_dir_path)!r}, "
            f"bootstrapped={self._bootstrapped})"
        )

    def __str__(self) -> str:
        names = ", ".join(sorted(self._tools.keys()))
        return (
            f"ToolRegistry[{len(self._tools)}]: {names}" if names else "ToolRegistry[0]"
        )

    @property
    def version(self) -> int:
        return self._version

    @property
    def registry(self) -> dict[str, Tool]:
        return self._tools

    def install_context(
        self, ctx: Context, extra_globals: Optional[Dict[str, Any]] = None
    ) -> None:
        self._execution_globals["ctx"] = ctx
        if extra_globals:
            self._execution_globals.update(extra_globals)

    def register(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter],
        function: Callable[..., Any],
    ) -> "ToolRegistry":
        self._log.info("Manually registering tool: %s", name)
        self._tools[name] = Tool(name, description, parameters, function)
        self._version += 1
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def list(self) -> list[Tool]:
        return self.list_tools()

    def list_skills(self) -> list[SkillCard]:
        self._catalog.ensure_skill_catalog()
        return list(self._catalog.skills.values())

    def route_query_to_skills(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float = 2.0,
    ) -> SkillSelection:
        return self._catalog.route_query_to_skills(
            query,
            [tool.name for tool in self._iter_tools_for_prompt()],
            top_k=top_k,
            min_score=min_score,
        )

    def skill_routing_prompt(
        self,
        query: str,
        *,
        use_toon: bool = True,
        mode: Literal["rich", "compact", "json_schema"] = "rich",
        top_k: int = 3,
        include_playbooks: bool = True,
        include_compact_fallback: bool = True,
    ) -> tuple[str, SkillSelection]:
        return self._catalog.skill_routing_prompt(
            query,
            [tool.name for tool in self._iter_tools_for_prompt()],
            self.tools_schema_prompt,
            use_toon=use_toon,
            mode=mode,
            top_k=top_k,
            include_playbooks=include_playbooks,
            include_compact_fallback=include_compact_fallback,
        )

    def call_tool(self, name: str, **kwargs: Any) -> str:
        call = ToolCall(id=f"internal_{time.time():.6f}", name=name, arguments=kwargs)
        return self.execute(call, use_toon=False)

    def load_tools_from_skills(self) -> None:
        python_count = self._load_python_skill_modules()

        skill_contents = self._catalog.read_skill_source_contents()
        guidance_contents = [
            (path, content)
            for path, content in skill_contents
            if path.name.upper() == "SKILL.MD"
        ]
        if guidance_contents:
            self._catalog.build_skill_catalog(guidance_contents)
            self._log.info(
                "Loaded %d guidance skill card(s) from SKILL.md",
                len(guidance_contents),
            )
        elif not skill_contents:
            self._log.info("No SKILL.md guidance sources found")

        if python_count > 0:
            self._version += 1

    def _load_python_skill_modules(self) -> int:
        if not self.skills_dir_path.is_dir():
            return 0

        py_skill_files = sorted(
            p
            for p in self.skills_dir_path.rglob("*.py")
            if p.name != "__init__.py" and not p.name.startswith("_")
        )
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

    def _register_tools_from_python_module(self, module_path: Path) -> int:
        code = module_path.read_text(encoding="utf-8")
        collector = self._LLMToolCollector()
        legacy_meta = self._legacy_md_tool_metadata_for_module(module_path)
        execution_globals = dict(self._execution_globals)
        execution_globals["llm"] = collector
        module_name = "skills_" + re.sub(
            r"[^a-zA-Z0-9_]+",
            "_",
            str(module_path.relative_to(self.skills_dir_path).with_suffix("")),
        ).strip("_")
        transient_module = types.ModuleType(module_name)
        transient_module.__file__ = str(module_path)
        sys.modules[module_name] = transient_module
        execution_globals["__file__"] = str(module_path)
        execution_globals["__name__"] = module_name

        exec(code, execution_globals, execution_globals)
        transient_module.__dict__.update(execution_globals)
        registered = 0
        skill_id = self._catalog._skill_id_from_source(module_path)

        for tool_fn in collector.tools:
            meta = getattr(tool_fn, "__llm_tool_meta__", {})
            tool_name = str(
                meta.get("name") or getattr(tool_fn, "__name__", "")
            ).strip()
            if not tool_name:
                continue
            if tool_name in self._tools:
                self._log.info(
                    "Skipping Python tool '%s' from %s because name is already registered",
                    tool_name,
                    module_path,
                )
                continue

            legacy_info = legacy_meta.get(tool_name, {})
            description = str(
                legacy_info.get("description")
                or meta.get("description")
                or inspect.getdoc(tool_fn)
                or ""
            ).strip()
            tool_fn.__doc__ = self._compose_tool_docstring(
                tool_fn,
                base_description=description,
                legacy_info=legacy_info,
            )
            params = self._parameters_from_signature(tool_fn)
            wrapped = self._wrap_tool_function(tool_fn, tool_name)
            self._tools[tool_name] = Tool(
                name=tool_name,
                description=description,
                parameters=params,
                function=wrapped,
                skill_id=skill_id,
                source_path=str(module_path),
            )
            self._catalog.tool_docs[tool_name] = inspect.getdoc(tool_fn) or description
            self._log.info("✓ Loaded Python tool: %s (%s)", tool_name, module_path)
            registered += 1

        for key, value in execution_globals.items():
            if key in {"__name__", "__file__", "llm"}:
                continue
            self._execution_globals[key] = value
        self._execution_globals["call_tool"] = self.call_tool
        if "ctx" not in self._execution_globals:
            self._execution_globals["ctx"] = None
        if "_safe_json" not in self._execution_globals:
            self._execution_globals["_safe_json"] = _safe_json_fallback
        return registered

    def _parameters_from_signature(
        self, func: Callable[..., Any]
    ) -> list[ToolParameter]:
        sig = inspect.signature(func)
        out: list[ToolParameter] = []
        for p in sig.parameters.values():
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            ann = p.annotation
            ptype = "string"
            if ann is not inspect._empty:
                ptype = self._annotation_to_type_name(ann)
            required = p.default is inspect._empty
            out.append(
                ToolParameter(
                    name=p.name,
                    type=ptype,
                    description="",
                    required=required,
                )
            )
        return out

    def _annotation_to_type_name(self, annotation: Any) -> str:
        origin = getattr(annotation, "__origin__", None)
        if origin in (list, List):
            return "list"
        if origin in (dict, Dict):
            return "dict"
        if annotation in (int,):
            return "int"
        if annotation in (float,):
            return "float"
        if annotation in (bool,):
            return "bool"
        if annotation in (list,):
            return "list"
        if annotation in (dict,):
            return "dict"
        if annotation in (str,):
            return "string"
        text = str(annotation)
        if "int" in text:
            return "int"
        if "float" in text:
            return "float"
        if "bool" in text:
            return "bool"
        if "list" in text:
            return "list"
        if "dict" in text:
            return "dict"
        return "string"

    def _legacy_md_tool_metadata_for_module(
        self, module_path: Path
    ) -> dict[str, dict[str, Any]]:
        rel_module = str(module_path.relative_to(self.skills_dir_path)).replace(
            "\\", "/"
        )
        py_snapshot = self._load_python_legacy_tool_metadata_map()
        out: dict[str, dict[str, Any]] = dict(py_snapshot.get(rel_module, {}))
        if not out and rel_module in ("99_qol/00_firecrawl.py", "99_qol/tools.py"):
            out = dict(py_snapshot.get("99_qol/80_websearch.py", {}))

        md_path = module_path.with_suffix(".md")
        if not md_path.is_file():
            return out
        try:
            raw = md_path.read_text(encoding="utf-8")
            manifest, content = self._catalog._split_frontmatter(raw)
            sections = self._catalog.parse_tool_sections(content, md_path)
            skill_summary = manifest.get("summary") or self._catalog._skill_summary(
                content
            )
            triggers = self._catalog._manifest_list(manifest, "triggers")
            when_not_to_use = self._catalog._manifest_list(manifest, "when_not_to_use")
            anti_triggers = self._catalog._manifest_list(manifest, "anti_triggers")

            for sec in sections:
                section_content = sec.get("content", "")
                out[sec["name"]] = {
                    "description": sec.get("description", ""),
                    "parameters": sec.get("parameters", []),
                    "returns": self._extract_markdown_field(section_content, "Returns"),
                    "side_effects": self._extract_markdown_field(
                        section_content, "Side effects"
                    ),
                    "triggers": triggers,
                    "when_not_to_use": when_not_to_use,
                    "anti_triggers": anti_triggers,
                    "skill_summary": str(skill_summary or "").strip(),
                }
            return out
        except Exception as exc:
            self._log.warning(
                "Failed to parse legacy markdown metadata for %s: %s",
                module_path,
                exc,
            )
            return out

    def _load_python_legacy_tool_metadata_map(
        self,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        if self._legacy_py_tool_metadata_by_module is not None:
            return self._legacy_py_tool_metadata_by_module

        metadata_path = self.skills_dir_path / "_legacy_tool_metadata.py"
        if not metadata_path.is_file():
            self._legacy_py_tool_metadata_by_module = {}
            return self._legacy_py_tool_metadata_by_module

        try:
            namespace: dict[str, Any] = {}
            exec(metadata_path.read_text(encoding="utf-8"), namespace, namespace)
            raw = namespace.get("TOOL_METADATA_BY_MODULE", {})
            if not isinstance(raw, dict):
                raw = {}
            self._legacy_py_tool_metadata_by_module = {
                str(k): v for k, v in raw.items() if isinstance(v, dict)
            }
            return self._legacy_py_tool_metadata_by_module
        except Exception as exc:
            self._log.warning(
                "Failed to load Python legacy tool metadata from %s: %s",
                metadata_path,
                exc,
            )
            self._legacy_py_tool_metadata_by_module = {}
            return self._legacy_py_tool_metadata_by_module

    def _extract_markdown_field(self, section_content: str, field_name: str) -> str:
        match = re.search(
            rf"\*\*{re.escape(field_name)}:\*\*\s*(.+?)(?:\n\n\*\*|$)",
            section_content,
            flags=re.DOTALL,
        )
        if not match:
            return ""
        return re.sub(r"\s+", " ", match.group(1)).strip()

    def _compose_tool_docstring(
        self,
        tool_fn: Callable[..., Any],
        *,
        base_description: str,
        legacy_info: dict[str, Any],
    ) -> str:
        description = (
            base_description.strip()
            or "Run this tool when it best fits the user request."
        )

        triggers = [
            str(t).strip() for t in legacy_info.get("triggers", []) if str(t).strip()
        ]
        if not triggers:
            triggers = [
                f"user asks to {getattr(tool_fn, '__name__', 'run this tool').replace('_', ' ')}"
            ]

        avoid_candidates = []
        avoid_candidates.extend(
            str(item).strip()
            for item in legacy_info.get("when_not_to_use", [])
            if str(item).strip()
        )
        avoid_candidates.extend(
            str(item).strip()
            for item in legacy_info.get("anti_triggers", [])
            if str(item).strip()
        )
        avoid_when = (
            avoid_candidates[0]
            if avoid_candidates
            else "another specialized tool is a clearer match"
        )

        legacy_params = legacy_info.get("parameters", [])
        if legacy_params:

            def _param_fields(param: Any) -> tuple[str, str, bool, str]:
                if isinstance(param, dict):
                    return (
                        str(param.get("name", "")).strip(),
                        str(param.get("type", "string")).strip(),
                        bool(param.get("required", False)),
                        str(param.get("description", "")).strip(),
                    )
                return (
                    str(getattr(param, "name", "")).strip(),
                    str(getattr(param, "type", "string")).strip(),
                    bool(getattr(param, "required", False)),
                    str(getattr(param, "description", "")).strip(),
                )

            parts: list[str] = []
            for param in legacy_params:
                pname, ptype, prequired, pdesc = _param_fields(param)
                if not pname:
                    continue
                parts.append(
                    f"{pname} ({ptype}, {'required' if prequired else 'optional'}): {pdesc}"
                )
            inputs_text = "; ".join(parts) if parts else "none"
        else:
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
            inputs_text = ", ".join(args) if args else "none"

        returns_text = str(legacy_info.get("returns") or "").strip()
        if not returns_text:
            returns_text = (
                "JSON string payload with status, results, and/or error details."
            )

        side_effects = str(legacy_info.get("side_effects") or "").strip()
        if not side_effects:
            side_effects = (
                "May read/update shared tool context depending on implementation."
            )

        return (
            f"Use when: {description}\n\n"
            f"Triggers: {', '.join(triggers)}.\n"
            f"Avoid when: {avoid_when}.\n"
            f"Inputs: {inputs_text}.\n"
            f"Returns: {returns_text}.\n"
            f"Side effects: {side_effects}"
        )

    class _LLMToolCollector:
        def __init__(self) -> None:
            self.tools: list[Callable[..., Any]] = []

        def tool(
            self,
            func: Optional[Callable[..., Any]] = None,
            *,
            name: Optional[str] = None,
            description: Optional[str] = None,
        ) -> Callable[..., Any]:
            def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
                setattr(
                    inner,
                    "__llm_tool_meta__",
                    {
                        "name": name or getattr(inner, "__name__", ""),
                        "description": description,
                    },
                )
                self.tools.append(inner)
                return inner

            if func is None:
                return decorator
            return decorator(func)

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
        # Allow bootstrap to re-run so any helper globals are refreshed.
        self._bootstrapped = False
        self.load_tools_from_skills()
        n = len(self._tools)
        self._log.info("Reload complete — %d tools registered", n)
        return n

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
            safe = tdef.safe_name
            # Deduplicate: if same safe name already registered, prefix with server name
            reg_name = safe if safe not in self._tools else f"{client.name}__{safe}"

            # Build a ToolParameter list from MCPToolParameter
            from .runtime import ToolParameter as _TP

            params = [
                _TP(
                    name=p.name,
                    type=p.type,
                    description=p.description,
                    required=p.required,
                )
                for p in tdef.parameters
            ]

            # Capture loop variables for the closure
            _client = client
            _mcp_name = tdef.name

            def _make_caller(_c: MCPClient, _n: str):
                def _call(**kwargs: Any) -> str:
                    try:
                        result = _c.call_tool(_n, kwargs)
                        if isinstance(result, str):
                            return result
                        return json.dumps(result, ensure_ascii=False, indent=2)
                    except Exception as exc:
                        return json.dumps(
                            {"status": "error", "error": str(exc), "tool": _n},
                            ensure_ascii=False,
                        )

                _call.__name__ = _n
                _call.__doc__ = f"MCP tool '{_n}' via server '{_c.name}'."
                return _call

            self._tools[reg_name] = Tool(
                name=reg_name,
                description=tdef.description,
                parameters=params,
                function=_make_caller(_client, _mcp_name),
                skill_id=f"mcp__{client.name}",
                source_path=client.url,
            )
            self._log.info(
                "✓ MCP tool registered: %s (server=%s mcp_name=%s)",
                reg_name,
                client.name,
                tdef.name,
            )
            registered += 1

        self._version += 1
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

        local_scope: Dict[str, Any] = {}
        exec(code, self._execution_globals, local_scope)

        func = local_scope.get(name)
        if func is None:
            callables = [v for v in local_scope.values() if callable(v)]
            if callables:
                func = callables[0]
                self._log.warning(
                    "Tool '%s': using function '%s' (expected name mismatch)",
                    name,
                    getattr(func, "__name__", "unknown"),
                )
            else:
                self._log.error("No callable found for tool: %s", name)
                return

        wrapped = self._wrap_tool_function(func, name)
        self._tools[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            function=wrapped,
            skill_id=skill_id or None,
            source_path=source_path or None,
        )
        self._catalog.tool_docs[name] = self._catalog.tool_doc_from_section(
            tool_info["content"]
        )
        self._log.info("✓ Loaded tool: %s", name)

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

    def execute(self, call: ToolCall, use_toon: bool = True) -> str:
        tool = self.get(call.name)
        if not tool:
            self._log.error("Tool not found: %s", call.name)
            return f"Error: tool '{call.name}' not found."

        if self._execution_globals.get("ctx", None) is None:
            sj = self._execution_globals.get("_safe_json", _safe_json_fallback)
            return sj(
                {
                    "status": "error",
                    "error": "ctx is None (Context not injected into ToolRegistry._execution_globals).",
                    "tool": call.name,
                }
            )

        result = tool.function(**(call.arguments or {}))

        if use_toon and HAS_TOON and isinstance(result, (dict, list)):
            assert encode is not None
            return encode(result)

        return (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False)
        )

    def _iter_tools_for_prompt(
        self, include_tool_names: Sequence[str] | None = None
    ) -> list[Tool]:
        if not include_tool_names:
            return sorted(self._tools.values(), key=lambda t: t.name)
        out: list[Tool] = []
        seen: set[str] = set()
        for tool_name in include_tool_names:
            tool = self._tools.get(tool_name)
            if tool is None or tool.name in seen:
                continue
            out.append(tool)
            seen.add(tool.name)
        return out

    def _json_schema_type(self, ptype: str) -> str:
        t = str(ptype).strip().lower()
        if t in {"int", "integer"}:
            return "integer"
        if t in {"float", "number", "double"}:
            return "number"
        if t in {"bool", "boolean"}:
            return "boolean"
        if t in {"list", "array"}:
            return "array"
        if t in {"dict", "object", "map"}:
            return "object"
        return "string"

    def _openai_tool_schemas(
        self, include_tool_names: Sequence[str] | None = None
    ) -> list[_OpenAIToolSchema]:
        out: list[_OpenAIToolSchema] = []
        for tool in self._iter_tools_for_prompt(include_tool_names):
            props: dict[str, Any] = {}
            required: list[str] = []
            for p in tool.parameters:
                props[p.name] = {
                    "type": self._json_schema_type(p.type),
                    "description": p.description,
                }
                if p.required:
                    required.append(p.name)
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": props,
                            "required": required,
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return out

    def tools_schema_prompt(
        self,
        use_toon: bool = True,
        mode: Literal["rich", "compact", "json_schema"] = "rich",
        include_tool_names: Sequence[str] | None = None,
        compact_fallback_tool_names: Sequence[str] | None = None,
    ) -> str:
        tools_for_prompt = self._iter_tools_for_prompt(include_tool_names)
        if not tools_for_prompt:
            return ""

        mode_norm = str(mode or "rich").strip().lower()
        if mode_norm not in {"rich", "compact", "json_schema"}:
            mode_norm = "rich"

        if mode_norm == "json_schema":
            payload = {
                "tool_call_contract": (
                    "When using tools, return exactly one tool call object. "
                    "If no tool needed, answer normally."
                ),
                "tools": self._openai_tool_schemas(include_tool_names),
            }
            return "\n\nTOOLS JSON SCHEMA:\n" + json.dumps(payload, ensure_ascii=False)

        if mode_norm == "compact":
            if use_toon:
                header = [
                    "\n\nTOOLS (compact): use only if needed.",
                    "Tool call format (TOON):",
                    "Return exactly one tool_call per response.",
                    "tool_call:",
                    "  name: <tool_name>",
                    "  arguments:",
                    "    ...",
                    "",
                ]
            else:
                header = [
                    "\n\nTOOLS (compact): use only if needed.",
                    "Tool call format (JSON):",
                    "Return exactly one tool_call per response.",
                    '{"tool_call":{"name":"<tool_name>","arguments":{...}}}',
                    "",
                ]

            body: List[str] = []
            for tool in tools_for_prompt:
                if tool.parameters:
                    params = ", ".join(
                        f"{p.name}:{p.type}{'' if p.required else '?'}"
                        for p in tool.parameters
                    )
                    body.append(f"- {tool.name}({params})")
                else:
                    body.append(f"- {tool.name}()")
            return "\n".join(header + body) + "\n"

        rich_docs = self._catalog.tool_docs
        if use_toon:
            header = [
                "\n\nTOOLS AVAILABLE (use only if needed):",
                "Return EXACT TOON format when calling:",
                "Return exactly one tool_call per response.",
                "tool_call:",
                "  name: <tool_name>",
                "  arguments:",
                "    <param1>: <value1>",
                "    <param2>: <value2>",
                "",
            ]
        else:
            header = [
                "\n\nTOOLS AVAILABLE (use only if needed):",
                "Return EXACT JSON when calling:",
                "Return exactly one tool_call per response.",
                '{"tool_call":{"name":"<tool_name>","arguments":{...}}}',
                "",
            ]

        body: List[str] = []
        for tool in tools_for_prompt:
            if tool.name in rich_docs:
                body.append(f"## Tool: {tool.name}")
                body.append(rich_docs[tool.name])
            else:
                body.append(f"Tool: {tool.name}")
                body.append(f"Description: {tool.description}")
                if tool.parameters:
                    body.append("Parameters:")
                    for p in tool.parameters:
                        req = "required" if p.required else "optional"
                        body.append(f"  - {p.name} ({p.type}, {req}): {p.description}")
            body.append("")

        if compact_fallback_tool_names:
            fallback_tools = self._iter_tools_for_prompt(compact_fallback_tool_names)
            if fallback_tools:
                body.append("OTHER AVAILABLE TOOLS (compact):")
                for tool in fallback_tools:
                    if tool.parameters:
                        params = ", ".join(
                            f"{p.name}:{p.type}{'' if p.required else '?'}"
                            for p in tool.parameters
                        )
                        body.append(f"- {tool.name}({params})")
                    else:
                        body.append(f"- {tool.name}()")
                body.append("")

        return "\n".join(header + body)


__version__ = "4.0.0"
__all__ = [
    "HAS_TOON",
    "Tool",
    "ToolCall",
    "ToolParameter",
    "ToolRegistry",
    "SkillCard",
    "SkillSelection",
    "parse_tool_calls",
    "parse_tool_call_strict",
    "Context",
    "check_optional_deps",
]
