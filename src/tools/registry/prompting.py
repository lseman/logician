from __future__ import annotations

import json
from typing import Any, Literal, Sequence

from ..runtime import Tool
from .types import _OpenAIToolSchema

_DEFAULT_PROMPT_TOOL_NAMES: tuple[str, ...] = (
    "describe_tool",
    "search_tools",
    "think",
    "todo",
    "list_dir",
    "read_file",
    "rg_search",
    "write_file",
    "edit_file",
    "multi_edit",
    "apply_edit_block",
    "run_shell",
    "run_python",
    "git_status",
    "git_diff",
)


class RegistryPromptingMixin:
    """ToolRegistry mixin."""

    def _iter_tools_for_prompt(self, include_tool_names: Sequence[str] | None = None) -> list[Tool]:
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

    def default_prompt_tool_names(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for tool_name in _DEFAULT_PROMPT_TOOL_NAMES:
            tool = self._tools.get(tool_name)
            if tool is None or tool.name in seen:
                continue
            out.append(tool.name)
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
                prop_schema: dict[str, Any] = {
                    "type": self._json_schema_type(p.type),
                    "description": p.description,
                }
                if p.enum:
                    prop_schema["enum"] = list(p.enum)
                props[p.name] = prop_schema
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

    def openai_tool_schemas(
        self, include_tool_names: Sequence[str] | None = None
    ) -> list[_OpenAIToolSchema]:
        """Public alias for generating OpenAI-compatible tool schema dicts.

        Used by constrained decoding: pass the returned list as ``tools=`` to
        ``LlamaCppClient.generate()`` to activate llama.cpp native function calling.
        """
        return self._openai_tool_schemas(include_tool_names)

    @staticmethod
    def _normalize_tools_prompt_mode(
        mode: Literal["rich", "compact", "json_schema"] | str,
    ) -> str:
        mode_norm = str(mode or "rich").strip().lower()
        if mode_norm not in {"rich", "compact", "json_schema"}:
            return "rich"
        return mode_norm

    @staticmethod
    def _compact_tool_signature(tool: Tool) -> str:
        if tool.parameters:
            params = ", ".join(
                f"{p.name}:{p.type}{'' if p.required else '?'}" for p in tool.parameters
            )
            return f"- {tool.name}({params})"
        return f"- {tool.name}()"

    def _json_schema_tools_prompt(self, include_tool_names: Sequence[str] | None = None) -> str:
        payload = {
            "tool_call_contract": (
                "When using tools, return exactly one tool call object. "
                "If no tool needed, answer normally."
            ),
            "tools": self._openai_tool_schemas(include_tool_names),
        }
        return "\n\nTOOLS JSON SCHEMA:\n" + json.dumps(payload, ensure_ascii=False)

    def _compact_tools_prompt(self, tools_for_prompt: Sequence[Tool], use_toon: bool) -> str:
        availability_note = (
            "The listed tools may be a routed subset, not the entire runtime. "
            "If the user names a different tool, verify with search_tools/describe_tool "
            "before claiming it is unavailable."
        )
        if use_toon:
            header = [
                "\n\nTOOLS (compact): use only if needed.",
                availability_note,
                "Tool call format (TOON):",
                "Return one tool_call per response, or a small batch of 2-4 independent read-only tool_calls.",
                "Never batch writes, edits, or verification commands.",
                "Prefer batching when the relevant read-only targets are already known.",
                "For codebase review/architecture/improvement requests, inspect enough files before answering; do not stop at one listing or one file if evidence is still thin.",
                "tool_call:",
                "  name: <tool_name>",
                "  arguments:",
                "    ...",
                "",
            ]
        else:
            header = [
                "\n\nTOOLS (compact): use only if needed.",
                availability_note,
                "Tool call format (JSON):",
                "Return one tool_call per response, or a small batch of 2-4 independent read-only tool_calls.",
                "Never batch writes, edits, or verification commands.",
                "Prefer batching when the relevant read-only targets are already known.",
                "For codebase review/architecture/improvement requests, inspect enough files before answering; do not stop at one listing or one file if evidence is still thin.",
                '{"tool_call":{"name":"<tool_name>","arguments":{...}}}',
                "",
            ]
        body = [self._compact_tool_signature(tool) for tool in tools_for_prompt]
        return "\n".join(header + body) + "\n"

    def _rich_tools_prompt(
        self,
        *,
        tools_for_prompt: Sequence[Tool],
        use_toon: bool,
        compact_fallback_tool_names: Sequence[str] | None = None,
    ) -> str:
        rich_docs = self._catalog.tool_docs
        if use_toon:
            header = [
                "\n\nTOOLS AVAILABLE (use only if needed):",
                "This may be a routed subset, not the full runtime tool list.",
                "If the user names another tool, verify with search_tools/describe_tool before saying it does not exist.",
                "Return EXACT TOON format when calling:",
                "Return one tool_call per response, or a small batch of 2-4 independent read-only tool_calls.",
                "Never batch writes, edits, or verification commands.",
                "Prefer batching when the relevant read-only targets are already known.",
                "For codebase review/architecture/improvement requests, inspect enough files before answering; do not stop at one listing or one file if evidence is still thin.",
                "Example: after listing src/ and identifying main.rs, app.rs, and ui.rs as relevant, batch those reads instead of serializing them across separate turns.",
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
                "This may be a routed subset, not the full runtime tool list.",
                "If the user names another tool, verify with search_tools/describe_tool before saying it does not exist.",
                "Return EXACT JSON when calling:",
                "Return one tool_call per response, or a small batch of 2-4 independent read-only tool_calls.",
                "Never batch writes, edits, or verification commands.",
                "Prefer batching when the relevant read-only targets are already known.",
                "For codebase review/architecture/improvement requests, inspect enough files before answering; do not stop at one listing or one file if evidence is still thin.",
                "Example: after listing src/ and identifying main.rs, app.rs, and ui.rs as relevant, batch those reads instead of serializing them across separate turns.",
                '{"tool_call":{"name":"<tool_name>","arguments":{...}}}',
                "",
            ]

        body: list[str] = []
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
                    body.append(self._compact_tool_signature(tool))
                body.append("")
        return "\n".join(header + body)

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

        mode_norm = self._normalize_tools_prompt_mode(mode)
        if mode_norm == "json_schema":
            return self._json_schema_tools_prompt(include_tool_names)

        if mode_norm == "compact":
            return self._compact_tools_prompt(tools_for_prompt, use_toon)

        return self._rich_tools_prompt(
            tools_for_prompt=tools_for_prompt,
            use_toon=use_toon,
            compact_fallback_tool_names=compact_fallback_tool_names,
        )
