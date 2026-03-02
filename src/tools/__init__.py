# -*- coding: utf-8 -*-
"""
Unified tools package with skill-aware loading and prompt rendering.
"""

from __future__ import annotations

import json
import time
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
        skill_contents = self._catalog.read_skill_source_contents()
        if not skill_contents:
            self._log.warning(
                "No skills source found (checked %s and %s)",
                self.skills_md_path,
                self.skills_dir_path,
            )
            return

        # build_skill_catalog parses and caches ALL tool sections in one pass.
        # We reuse catalog.all_tool_sections instead of re-parsing every file.
        self._catalog.build_skill_catalog(skill_contents)
        self._run_bootstrap_blocks(skill_contents)

        tool_sections = self._catalog.all_tool_sections
        for tool_info in tool_sections:
            try:
                self._register_tool_from_section(tool_info)
            except Exception as e:
                self._log.error(
                    "Failed to register tool '%s': %s",
                    tool_info.get("name", "unknown"),
                    e,
                )
        self._log.info(
            "Loaded %d tools from %d skill source(s)",
            len(tool_sections),
            len(skill_contents),
        )
        self._version += 1

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
