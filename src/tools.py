# -*- coding: utf-8 -*-
"""
agent_core/tools.py

Unified Tools Module (v3)

- ToolRegistry: dynamic tool loading from SKILLS.md + execution
- Context + shared time-series helpers (ts_tools-equivalent)
- Robust injection API: ToolRegistry.install_context(ctx, extra_globals=...)

This file replaces older split modules like tools_enhanced.py + ts_tools.py.

Key guarantee:
- Tool execution globals always include: ctx, np, pd, json, _safe_json, call_tool, ...
"""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

import numpy as np
import pandas as pd

from .logging_utils import get_logger

try:
    from markdown_it import MarkdownIt

    HAS_MARKDOWN_IT = True
except ImportError:
    MarkdownIt = None  # type: ignore
    HAS_MARKDOWN_IT = False

# ==================== Optional TOON Support ====================
try:
    from toon_format import decode, encode

    HAS_TOON = True
except ImportError:
    encode = decode = None  # type: ignore
    HAS_TOON = False


# =============================================================================
# 1) Shared helpers + Context (time-series)
# =============================================================================


@lru_cache(maxsize=1)
def check_optional_deps() -> Dict[str, bool]:
    deps = {
        "scipy": False,
        "statsmodels": False,
        "ruptures": False,
        "sklearn": False,
        "neuralforecast": False,
    }

    try:
        import scipy  # noqa: F401

        deps["scipy"] = True
    except ImportError:
        pass

    try:
        import statsmodels  # noqa: F401

        deps["statsmodels"] = True
    except ImportError:
        pass

    try:
        import ruptures  # noqa: F401

        deps["ruptures"] = True
    except ImportError:
        pass

    try:
        import sklearn  # noqa: F401

        deps["sklearn"] = True
    except ImportError:
        pass

    try:
        import neuralforecast  # noqa: F401

        deps["neuralforecast"] = True
    except ImportError:
        pass

    return deps


@dataclass
class Context:
    data: Optional[pd.DataFrame] = None
    original_data: Optional[pd.DataFrame] = None
    data_name: str = ""
    freq_cache: Optional[str] = None

    anomaly_store: Dict[str, List[int]] = field(default_factory=dict)
    anomaly_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    nf_best_model: Optional[str] = None
    nf_cv_full: Optional[pd.DataFrame] = None
    nf_pred_col: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self.data is not None and len(self.data) > 0

    @property
    def value_columns(self) -> List[str]:
        if self.data is None:
            return []
        return [c for c in self.data.columns if c != "date"]

    @property
    def is_multivariate(self) -> bool:
        return len(self.value_columns) > 1

    def reset(self) -> None:
        self.data = None
        self.original_data = None
        self.data_name = ""
        self.freq_cache = None
        self.anomaly_store.clear()
        self.anomaly_meta.clear()
        self.nf_best_model = None
        self.nf_cv_full = None
        self.nf_pred_col = None


def _safe_json_fallback(obj: Any) -> str:
    """
    Fallback JSON serializer if SKILLS.md bootstrap didn't inject _safe_json.
    """
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps(
            {"status": "error", "error": f"json failed: {e}"}, ensure_ascii=False
        )


# =============================================================================
# 2) Tool registry (SKILLS.md)
# =============================================================================


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class Tool:
    name: str
    description: str
    parameters: list[ToolParameter]
    function: Callable[..., Any]

    def __repr__(self) -> str:
        return (
            f"Tool(name={self.name!r}, parameters={len(self.parameters)}, "
            f"description={self.description[:48]!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


class _MarkdownSection(TypedDict):
    heading: str
    body: str


class _BootstrapSection(TypedDict):
    name: str
    content: str
    code: str


class _ToolSection(TypedDict):
    name: str
    content: str
    code: str
    description: str
    parameters: list[ToolParameter]


class _OpenAIToolFunction(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class _OpenAIToolSchema(TypedDict):
    type: str
    function: _OpenAIToolFunction


class CodeBlockExtractor:
    """
    Extracts lines inside the first/any ```python ... ``` blocks by streaming lines.
    """

    def __init__(self):
        self.in_block = False
        self.code_lines: List[str] = []

    def process_line(self, line: str) -> None:
        stripped = line.strip()
        if stripped.startswith("```python"):
            self.in_block = True
        elif stripped == "```" and self.in_block:
            self.in_block = False
        elif self.in_block:
            self.code_lines.append(line)

    def get_code(self) -> str:
        return "\n".join(self.code_lines)


class ToolRegistry:
    """
    Registry for managing and executing tools loaded from SKILLS sources.

    IMPORTANT:
    - ToolRegistry executes all '## Bootstrap:' code blocks BEFORE registering any tools.
    - You should call install_context(ctx) early; however, we keep ctx=None placeholder
      so tools can still be registered even if you install ctx afterward.
    """

    def __init__(self, auto_load_from_skills: bool = True) -> None:
        self._tools: dict[str, Tool] = {}
        self._log = get_logger("agent.tools")
        self.skills_md_path = Path(__file__).resolve().parents[1] / "SKILLS.md"
        self.skills_dir_path = Path(__file__).resolve().parents[1] / "skills"

        # Shared execution globals for tool exec()
        # NOTE: ctx placeholder avoids NameError during exec(tool_code) before install_context().
        self._execution_globals: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "json": json,
            "np": np,
            "pd": pd,
            "ctx": None,
        }

        # Provide internal tool composition helper into SKILLS runtime.
        # Tools can call: call_tool("some_tool", a=1, b=2)
        self._execution_globals["call_tool"] = self.call_tool

        # Ensure _safe_json exists even if bootstrap isn't run (or fails).
        self._execution_globals["_safe_json"] = _safe_json_fallback

        # Tracks whether we've executed the SKILLS.md Bootstrap blocks
        self._bootstrapped: bool = False

        # A simple version counter so Agent can invalidate caches
        self._version: int = 0

        if auto_load_from_skills and self._iter_skills_sources():
            self._log.info(
                "Auto-loading tools from %d source(s)",
                len(self._iter_skills_sources()),
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
        return f"ToolRegistry[{len(self._tools)}]: {names}" if names else "ToolRegistry[0]"

    @property
    def version(self) -> int:
        return self._version

    @property
    def registry(self) -> dict[str, Tool]:
        # Backward-compat for older notebooks that access .registry directly.
        return self._tools

    def install_context(
        self, ctx: Context, extra_globals: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Inject ctx + any extra globals into tool execution globals.

        Safe to call multiple times; last ctx wins.
        """
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

    # Backward-compat alias used by older notebooks/examples.
    def list(self) -> list[Tool]:
        return self.list_tools()

    def call_tool(self, name: str, **kwargs: Any) -> str:
        """
        Tool-to-tool calling helper intended to be used from inside SKILLS tools.

        Returns the raw tool result string (JSON string, or any string returned by tool).
        """
        call = ToolCall(id=f"internal_{time.time():.6f}", name=name, arguments=kwargs)
        return self.execute(call, use_toon=False)

    def load_tools_from_skills(self) -> None:
        skill_sources = self._iter_skills_sources()
        if not skill_sources:
            self._log.warning(
                "No skills source found (checked %s and %s)",
                self.skills_md_path,
                self.skills_dir_path,
            )
            return

        content = self._read_skills_content()
        if content is None:
            return

        # 1) Bootstrap pass
        self._run_bootstrap_blocks(content)

        # 2) Tool registration pass
        tool_sections = self._parse_tool_sections(content)
        self._log.info("Found %d tool sections in skills sources", len(tool_sections))

        for tool_info in tool_sections:
            try:
                self._register_tool_from_section(tool_info)
            except Exception as e:
                self._log.error(
                    "Failed to register tool '%s': %s",
                    tool_info.get("name", "unknown"),
                    e,
                )

        self._version += 1

    def _read_skills_content(self) -> str | None:
        sources = self._iter_skills_sources()
        if not sources:
            return None

        chunks: list[str] = []
        for src in sources:
            try:
                chunks.append(src.read_text(encoding="utf-8"))
            except Exception as e:
                self._log.error("Failed to read skills source %s: %s", src, e)
                return None
        return "\n\n".join(chunks)

    def _iter_skills_sources(self) -> list[Path]:
        """
        Resolve skills markdown sources in loading order.

        Priority:
        1) If skills_md_path points to a directory -> all *.md inside it.
        2) If skills_md_path is a file and sibling skills/ exists -> sibling *.md files.
        3) If skills_md_path is a file -> that file only.
        """
        # Explicit directory override.
        if self.skills_md_path.is_dir():
            return sorted(self.skills_md_path.glob("*.md"))

        # Preferred layout: sibling skills folder next to SKILLS.md.
        sibling_dir = self.skills_md_path.parent / "skills"
        if sibling_dir.is_dir():
            files = sorted(sibling_dir.glob("*.md"))
            if files:
                return files

        # Legacy single-file layout.
        if self.skills_md_path.is_file():
            return [self.skills_md_path]

        # Optional explicit directory path if users set skills_dir_path manually.
        if self.skills_dir_path.is_dir():
            return sorted(self.skills_dir_path.glob("*.md"))

        return []

    # -------------------------------------------------------------------------
    # Bootstrap handling
    # -------------------------------------------------------------------------

    def _run_bootstrap_blocks(self, content: str) -> None:
        """
        Executes all '## Bootstrap:' sections found in SKILLS.md in order, once.
        Their symbols are injected into _execution_globals and become available
        to all tools.
        """
        if self._bootstrapped:
            return

        blocks = self._parse_bootstrap_sections(content)
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
                # Use shared globals as both globals/locals so definitions persist.
                exec(code, self._execution_globals, self._execution_globals)
                self._log.info("✓ Ran bootstrap: %s", name)
            except Exception as e:
                self._log.error("Bootstrap '%s' failed: %s", name, e)

        # Make sure call_tool and ctx didn't get overwritten by bootstrap.
        self._execution_globals["call_tool"] = self.call_tool
        if "ctx" not in self._execution_globals:
            self._execution_globals["ctx"] = None

        # If bootstrap provided a better _safe_json, keep it; otherwise keep fallback.
        if "_safe_json" not in self._execution_globals:
            self._execution_globals["_safe_json"] = _safe_json_fallback

        self._bootstrapped = True

    def _parse_markdown_h2_sections(self, content: str) -> list[_MarkdownSection]:
        if not HAS_MARKDOWN_IT or MarkdownIt is None:
            self._log.warning(
                "markdown-it-py not available; using legacy line-based SKILLS parser"
            )
            return self._parse_markdown_h2_sections_fallback(content)

        md = MarkdownIt()
        tokens = md.parse(content)
        lines = content.splitlines()
        sections: list[_MarkdownSection] = []
        n_lines = len(lines)

        h2_entries: list[tuple[str, int]] = []
        for i, tok in enumerate(tokens):
            if tok.type != "heading_open" or tok.tag != "h2":
                continue
            heading = ""
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                heading = tokens[i + 1].content.strip()
            start_line = tok.map[1] if tok.map else 0
            h2_entries.append((heading, start_line))

        for idx, (heading, start_line) in enumerate(h2_entries):
            end_line = n_lines
            if idx + 1 < len(h2_entries):
                end_line = max(start_line, h2_entries[idx + 1][1] - 1)
            body = "\n".join(lines[start_line:end_line]).strip("\n")
            sections.append({"heading": heading, "body": body})

        return sections

    def _parse_markdown_h2_sections_fallback(
        self, content: str
    ) -> list[_MarkdownSection]:
        sections: list[_MarkdownSection] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        for line in content.splitlines():
            m = re.match(r"^\s*##\s+(.+?)\s*$", line)
            if m:
                if current_heading is not None:
                    sections.append(
                        {
                            "heading": current_heading,
                            "body": "\n".join(current_lines).strip("\n"),
                        }
                    )
                current_heading = m.group(1).strip()
                current_lines = []
                continue

            if current_heading is not None:
                current_lines.append(line)

        if current_heading is not None:
            sections.append(
                {"heading": current_heading, "body": "\n".join(current_lines).strip("\n")}
            )
        return sections

    def _extract_bootstrap_name_from_heading(self, heading: str) -> Optional[str]:
        normalized = heading.strip()
        if not normalized.lower().startswith("bootstrap:"):
            return None
        rest = normalized[len("bootstrap:") :].strip()
        return rest if rest else "bootstrap"

    def _extract_bootstrap_name(self, line: str) -> Optional[str]:
        stripped = line.strip()
        if not stripped.startswith("##"):
            return None
        return self._extract_bootstrap_name_from_heading(stripped.lstrip("# ").strip())

    def _extract_code_from_markdown(self, text: str) -> str:
        code_extractor = CodeBlockExtractor()
        for line in text.splitlines():
            code_extractor.process_line(line)
        return code_extractor.get_code()

    def _parse_bootstrap_sections(self, content: str) -> list[_BootstrapSection]:
        """
        Returns a list of bootstrap sections:
          [{"name": "...", "content": "...", "code": "python code from fences"}, ...]
        """
        sections: list[_BootstrapSection] = []
        for sec in self._parse_markdown_h2_sections(content):
            bname = self._extract_bootstrap_name_from_heading(sec["heading"])
            if not bname:
                continue
            sections.append(
                {
                    "name": bname,
                    "content": sec["body"],
                    "code": self._extract_code_from_markdown(sec["body"]),
                }
            )
        return sections

    # -------------------------------------------------------------------------
    # Tool section parsing + registration
    # -------------------------------------------------------------------------

    def _parse_tool_sections(self, content: str) -> list[_ToolSection]:
        sections: list[_ToolSection] = []
        for sec in self._parse_markdown_h2_sections(content):
            tool_name = self._extract_tool_name_from_heading(sec["heading"])
            if not tool_name:
                continue
            section: _ToolSection = {
                "name": tool_name,
                "content": sec["body"],
                "code": self._extract_code_from_markdown(sec["body"]),
                "description": "",
                "parameters": [],
            }
            self._extract_metadata(section)
            sections.append(section)
        return sections

    def _extract_tool_name_from_heading(self, heading: str) -> Optional[str]:
        normalized = heading.strip()
        if not normalized.lower().startswith("tool:"):
            return None
        rest = normalized[len("tool:") :].strip()
        return rest.split()[0] if rest else None

    def _extract_tool_name(self, line: str) -> Optional[str]:
        stripped = line.strip()
        if not stripped.startswith("##"):
            return None
        return self._extract_tool_name_from_heading(stripped.lstrip("# ").strip())

    def _extract_metadata(self, section: _ToolSection) -> None:
        content = section["content"]

        desc_match = re.search(
            r"\*\*Description:\*\*\s*(.+?)(?:\n\n|\*\*|$)", content, re.DOTALL
        )
        section["description"] = desc_match.group(1).strip() if desc_match else ""
        section["parameters"] = self._parse_parameters(content)

    def _parse_parameters(self, content: str) -> List[ToolParameter]:
        params = []
        param_section = re.search(
            r"\*\*Parameters:\*\*\s*\n((?:^[-*]\s+.+\n?)+)", content, re.MULTILINE
        )
        if not param_section:
            return params

        param_text = param_section.group(1)
        param_pattern = re.compile(
            r"[-*]\s+(\w+)\s+\(([^,]+),\s*(required|optional)\):\s*(.+)"
        )
        for match in param_pattern.finditer(param_text):
            params.append(
                ToolParameter(
                    name=match.group(1),
                    type=match.group(2).strip(),
                    description=match.group(4).strip(),
                    required=(match.group(3) == "required"),
                )
            )
        return params

    def _register_tool_from_section(self, tool_info: _ToolSection) -> None:
        name = tool_info["name"]
        description = tool_info.get("description", "")
        parameters = tool_info.get("parameters", [])
        code = tool_info.get("code", "")

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
            name=name, description=description, parameters=parameters, function=wrapped
        )
        self._log.info("✓ Loaded tool: %s", name)

    def _wrap_tool_function(self, func: Callable, tool_name: str) -> Callable:
        def wrapped(**kwargs):
            try:
                return func(**kwargs)
            except Exception as e:
                self._log.exception("Error in tool '%s'", tool_name)
                # Prefer bootstrap _safe_json if available
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

        # sanity: ctx must exist
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

    def _openai_tool_schemas(self) -> list[_OpenAIToolSchema]:
        out: list[_OpenAIToolSchema] = []
        for tool in sorted(self._tools.values(), key=lambda t: t.name):
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
    ) -> str:
        if not self._tools:
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
                "tools": self._openai_tool_schemas(),
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
            for tool in sorted(self._tools.values(), key=lambda t: t.name):
                if tool.parameters:
                    params = ", ".join(
                        f"{p.name}:{p.type}{'' if p.required else '?'}"
                        for p in tool.parameters
                    )
                    body.append(f"- {tool.name}({params})")
                else:
                    body.append(f"- {tool.name}()")
            return "\n".join(header + body) + "\n"

        rich_docs = self._load_tool_docs_for_prompt()

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
        for tool in sorted(self._tools.values(), key=lambda t: t.name):
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
        return "\n".join(header + body)

    def _load_tool_docs_for_prompt(self) -> Dict[str, str]:
        if not self._iter_skills_sources():
            return {}

        content = self._read_skills_content()
        if content is None:
            return {}

        docs: Dict[str, str] = {}
        for sec in self._parse_markdown_h2_sections(content):
            tool_name = self._extract_tool_name_from_heading(sec["heading"])
            if not tool_name:
                continue

            kept_lines: List[str] = []
            in_code_block = False
            for line in sec["body"].splitlines():
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    continue
                if stripped == "**Implementation:**":
                    continue
                kept_lines.append(line)

            docs[tool_name] = "\n".join(kept_lines).strip()
        return docs


# =============================================================================
# 3) Tool call parsing (TOON/JSON)
# =============================================================================


def parse_tool_calls(text: str, use_toon: bool) -> list[ToolCall]:
    calls: list[ToolCall] = []
    if use_toon and HAS_TOON and decode is not None:
        calls.extend(_parse_toon_tool_calls(text))
    calls.extend(_parse_json_tool_calls(text))

    # Deduplicate exact same name+args while preserving order.
    out: list[ToolCall] = []
    seen: set[tuple[str, str]] = set()
    for c in calls:
        sig = (c.name, json.dumps(c.arguments, sort_keys=True, ensure_ascii=False))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out


def parse_tool_call_strict(text: str, use_toon: bool) -> ToolCall | None:
    calls = parse_tool_calls(text, use_toon=use_toon)
    if calls:
        return calls[0]
    return None


def _parse_toon_tool_calls(text: str) -> list[ToolCall]:
    if decode is None:
        return []

    calls: list[ToolCall] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped.startswith("tool_call:"):
            i += 1
            continue

        block_lines = [lines[i]]
        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if s and not lines[j].startswith((" ", "\t")):
                break
            block_lines.append(lines[j])
            j += 1

        toon_text = "\n".join(block_lines).strip()
        try:
            data = decode(toon_text)
            if isinstance(data, dict) and "tool_call" in data:
                call_data = data["tool_call"]
                if isinstance(call_data, dict):
                    name = call_data.get("name")
                    args = call_data.get("arguments", {})
                    if isinstance(name, str) and isinstance(args, dict):
                        calls.append(
                            ToolCall(
                                id=f"call_{time.time():.6f}",
                                name=name,
                                arguments=args,
                            )
                        )
        except Exception:
            pass
        i = j
    return calls


def _parse_toon_tool_call(text: str) -> ToolCall | None:
    calls = _parse_toon_tool_calls(text)
    if calls:
        return calls[0]
    return None


def _parse_json_tool_call(text: str) -> ToolCall | None:
    calls = _parse_json_tool_calls(text)
    if calls:
        return calls[0]
    return None


def _parse_json_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for json_candidate in _extract_json_objects(text):
        try:
            data = json.loads(json_candidate)
        except Exception:
            continue

        if isinstance(data, dict) and "tool_call" in data:
            call_data = data["tool_call"]
            if isinstance(call_data, dict):
                name = call_data.get("name")
                args = call_data.get("arguments", {})
                if isinstance(name, str) and isinstance(args, dict):
                    calls.append(
                        ToolCall(
                            id=f"call_{time.time():.6f}", name=name, arguments=args
                        )
                    )
                    continue

        if isinstance(data, dict) and "name" in data and "arguments" in data:
            name = data["name"]
            args = data["arguments"]
            if isinstance(name, str) and isinstance(args, dict):
                calls.append(
                    ToolCall(id=f"call_{time.time():.6f}", name=name, arguments=args)
                )

    return calls


def _extract_json_objects(text: str) -> List[str]:
    candidates: List[str] = []
    stack = 0
    start: Optional[int] = None

    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0 and start is not None:
                candidates.append(text[start : i + 1])
                start = None

    return candidates


__version__ = "3.0.0"
__all__ = [
    "HAS_TOON",
    "ToolCall",
    "ToolParameter",
    "ToolRegistry",
    "parse_tool_calls",
    "parse_tool_call_strict",
    "Context",
    "check_optional_deps",
]
