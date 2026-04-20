from __future__ import annotations

import ast
import json
import re
import time
from typing import List

from .runtime import HAS_TOON, ToolCall, decode

_DIRECT_TOOL_POSITIONAL_ARGS: dict[str, list[str]] = {
    "bash": ["command"],
    "fd_find": ["pattern"],
    "rg_search": ["pattern"],
    "read_file": ["path"],
    "write_file": ["path", "content"],
    "edit_file": ["path", "old_string", "new_string"],
    "apply_edit_block": ["path", "blocks"],
    "glob": ["pattern"],
    "grep": ["pattern"],
    "think": ["thought"],
}

# Aliases: map model-generated wrong/variant tool names to real canonical names.
# Core tools (registered by Agent.__init__): bash, read_file, write_file, edit_file,
#   glob_files, grep_files, apply_edit_block, think, todo.
# These are the primary canonical names — do NOT alias them away from themselves.
_TOOL_NAME_ALIASES: dict[str, str] = {
    # shell execution variants → core bash
    "shell": "bash",
    "execute": "bash",
    "execute_command": "bash",
    "run_command": "bash",
    "run_bash": "bash",
    "terminal": "bash",
    # file reading variants → core read_file
    "cat_file": "read_file",
    "file_read": "read_file",
    "open_file": "read_file",
    "view_file": "read_file",
    "cat": "read_file",
    # file search variants → core glob_files
    "glob": "glob_files",
    "find_files": "glob_files",
    # text search variants → core grep_files
    "grep": "grep_files",
    "search_text": "grep_files",
}


def _normalize_tool_name(name: str) -> str:
    """Resolve aliases and strip common noise from model-generated tool names."""
    canonical = _TOOL_NAME_ALIASES.get(name.lower(), name)
    return canonical


def parse_tool_calls(
    text: str,
    use_toon: bool,
    *,
    strict: bool = False,
) -> list[ToolCall]:
    calls: list[ToolCall] = []
    calls.extend(_parse_json_tool_calls(text))

    if use_toon and HAS_TOON and decode is not None:
        calls.extend(_parse_toon_tool_calls(text))

    if not strict:
        calls.extend(_parse_inline_toon_tool_calls(text))
        calls.extend(_parse_jinja_tool_calls(text))

    if not calls and not strict:
        calls.extend(_parse_shell_fence_tool_calls(text))
    # Normalize tool names through alias table
    for call in calls:
        normalized = _normalize_tool_name(call.name)
        if normalized != call.name:
            call.name = normalized

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
    calls = parse_tool_calls(text, use_toon=use_toon, strict=True)
    return calls[0] if calls else None


def _new_tool_call(name: str, args: dict) -> ToolCall:
    return ToolCall(
        id=f"call_{time.time():.6f}",
        name=name,
        arguments=args,
    )


def _parse_tool_calls_from_object(data: object) -> list[ToolCall]:
    calls: list[ToolCall] = []
    if not isinstance(data, dict):
        return calls

    if "tool_call" in data:
        call_data = data["tool_call"]
        if isinstance(call_data, dict):
            name = call_data.get("name")
            args = call_data.get("arguments", {})
            if isinstance(name, str) and isinstance(args, dict):
                calls.append(_new_tool_call(name, args))
                return calls

    if "tool_calls" in data and isinstance(data["tool_calls"], list):
        for call_data in data["tool_calls"]:
            if not isinstance(call_data, dict):
                continue
            name = call_data.get("name")
            args = call_data.get("arguments", {})
            if isinstance(name, str) and isinstance(args, dict):
                calls.append(_new_tool_call(name, args))
        if calls:
            return calls

    if "name" in data and "arguments" in data:
        name = data["name"]
        args = data["arguments"]
        if isinstance(name, str) and isinstance(args, dict):
            calls.append(_new_tool_call(name, args))

    # Common plain JSON shape emitted by some prompts/backends:
    # {"tool":"fd_find","arguments":{...}}
    if "tool" in data and "arguments" in data:
        name = data["tool"]
        args = data["arguments"]
        if isinstance(name, str) and isinstance(args, dict):
            calls.append(_new_tool_call(name, args))

    # Anthropic-native tool block shape:
    # {"type":"tool_use","name":"...","input":{...}}
    if (
        data.get("type") == "tool_use"
        and isinstance(data.get("name"), str)
        and isinstance(data.get("input"), dict)
    ):
        calls.append(_new_tool_call(str(data["name"]), dict(data["input"])))

    # Anthropic response wrapper shape:
    # {"content":[{"type":"tool_use","name":"...","input":{...}}], ...}
    if isinstance(data.get("content"), list):
        for block in data["content"]:
            if not isinstance(block, dict):
                continue
            if (
                block.get("type") == "tool_use"
                and isinstance(block.get("name"), str)
                and isinstance(block.get("input"), dict)
            ):
                calls.append(_new_tool_call(str(block["name"]), dict(block["input"])))

    return calls


def _parse_inline_toon_tool_calls(text: str) -> list[ToolCall]:
    """Parse inline TOON-style tool calls, including batched multi-tool responses.

    Handles single calls:
        tool_call: name: bash arguments: command: ls -la

    And batched calls under one header (model emits multiple reads in one block):
        tool_call: name: read_file arguments: path: a.py
                   name: read_file arguments: path: b.py

    And block-indented batches:
        tool_call:
          name: read_file
          arguments:
            path: a.py
          name: read_file
          arguments:
            path: b.py
    """
    calls: list[ToolCall] = []
    # Split at every tool_call: anchor; each segment is one "batch scope".
    anchor = re.compile(
        r"(?:^|(?<=\s))tool_call:",
        re.MULTILINE | re.IGNORECASE,
    )
    # Within each scope find every  name: X  arguments: Y  entry.
    # DOTALL so arguments can span lines; stops at the next name: entry or end.
    entry = re.compile(
        r"name:\s+(\S+)\s+arguments:\s+(.+?)(?=\s+name:\s+\S|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    for seg in anchor.split(text)[1:]:  # [0] is text before first tool_call:
        for m in entry.finditer(seg):
            name = m.group(1).strip().strip("\"'")
            args_text = m.group(2).strip()
            args: dict[str, str] = {}
            for kv in re.finditer(r"(\w+):\s+(.+?)(?=\s+\w+:|$)", args_text, re.DOTALL):
                val = kv.group(2).strip()
                # Strip matching surrounding quotes (YAML/shell quoting convention)
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1]
                args[kv.group(1)] = val
            # If no key-value pairs found, use positional default for known tools
            if not args and args_text:
                positional_keys = _DIRECT_TOOL_POSITIONAL_ARGS.get(name.lower(), [])
                if positional_keys:
                    args = {positional_keys[0]: args_text}
                else:
                    args = {"input": args_text}
            if name:
                calls.append(_new_tool_call(name, args))
    return calls


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

        # Detect multiple tool specs in one block (model batched multiple
        # name:/arguments: pairs under a single tool_call: header).
        name_idxs = [
            k for k, ln in enumerate(block_lines) if ln.strip().startswith("name:")
        ]

        def _try_decode_block(sub_lines: list[str]) -> None:
            toon_text = "\n".join(sub_lines).strip()
            try:
                data = decode(toon_text)
                if isinstance(data, dict) and "tool_call" in data:
                    call_data = data["tool_call"]
                    if isinstance(call_data, dict):
                        name = call_data.get("name")
                        args = call_data.get("arguments", {})
                        if isinstance(name, str) and isinstance(args, dict):
                            calls.append(_new_tool_call(name, args))
            except Exception:
                pass

        if len(name_idxs) > 1:
            # Split at each name: boundary; re-prefix with the tool_call: header
            header = block_lines[0]
            for idx, start in enumerate(name_idxs):
                end = name_idxs[idx + 1] if idx + 1 < len(name_idxs) else len(block_lines)
                _try_decode_block([header] + block_lines[start:end])
        else:
            _try_decode_block(block_lines)

        i = j
    return calls


def _lenient_json_loads(s: str) -> dict:
    """Try ``json.loads`` then fall back to fixing common LLM escaping mistakes.

    LLMs sometimes emit:
    - Literal ``\\n`` / ``\\t`` inside string values instead of ``\\\\n`` / ``\\\\t``
    - Literal control characters (0x00–0x1f) inside string values
    - Unescaped double-quotes inside string values

    The fallback re-encodes string values by scanning the JSON text character by
    character and escaping any bare control characters it finds inside strings.
    This is intentionally conservative — it only touches the content of string
    values, leaving structural JSON characters alone.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Re-escape bare control characters inside JSON string literals
    fixed_chars: list[str] = []
    in_str = False
    k = 0
    while k < len(s):
        ch = s[k]
        if in_str:
            if ch == "\\":
                # Pass through escape sequence unchanged
                fixed_chars.append(ch)
                k += 1
                if k < len(s):
                    fixed_chars.append(s[k])
                    k += 1
                continue
            if ch == '"':
                in_str = False
                fixed_chars.append(ch)
                k += 1
                continue
            # Escape any bare control character (including literal newlines/tabs)
            if ord(ch) < 0x20:
                fixed_chars.append(
                    "\\n"
                    if ch == "\n"
                    else "\\t"
                    if ch == "\t"
                    else "\\r"
                    if ch == "\r"
                    else f"\\u{ord(ch):04x}"
                )
                k += 1
                continue
        else:
            if ch == '"':
                in_str = True
        fixed_chars.append(ch)
        k += 1

    try:
        return json.loads("".join(fixed_chars))
    except json.JSONDecodeError as exc:
        raise exc


def _parse_json_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for json_candidate in _extract_json_objects(text):
        try:
            data = _lenient_json_loads(json_candidate)
        except Exception:
            continue
        calls.extend(_parse_tool_calls_from_object(data))
    return calls


def _parse_jinja_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []

    # Common Jinja/templated wrapper used by several chat templates:
    # <tool_call> ... </tool_call>
    for body in re.findall(r"(?is)<tool_call[^>]*>(.*?)</tool_call>", text):
        payload = body.strip()
        if not payload:
            continue
        parsed = _parse_object_lenient(payload)
        if parsed is not None:
            calls.extend(_parse_tool_calls_from_object(parsed))

    # Jinja expression wrappers:
    # {{ {"name":"x","arguments":{...}} }}
    for expr in re.findall(r"(?is)\{\{\s*(.*?)\s*\}\}", text):
        payload = expr.strip()
        if not payload:
            continue
        parsed = _parse_object_lenient(payload)
        if parsed is not None:
            calls.extend(_parse_tool_calls_from_object(parsed))

    # Function-call style often emitted by Jinja-based templates:
    # tool_call(name="x", arguments={...})
    # tool_call("x", {"arg": 1})
    for args_text in _extract_function_call_args(text, "tool_call"):
        parsed = _parse_jinja_tool_call_args(args_text)
        if parsed is None:
            continue
        calls.append(_new_tool_call(parsed[0], parsed[1]))

    return calls


def _parse_object_lenient(payload: str) -> object | None:
    text = payload.strip()
    if not text:
        return None
    try:
        return _lenient_json_loads(text)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(text)
    except Exception:
        return None
    return obj


def _extract_function_call_args(text: str, function_name: str) -> list[str]:
    out: list[str] = []
    if not text or not function_name:
        return out

    pattern = re.compile(rf"\b{re.escape(function_name)}\s*\(", re.IGNORECASE)
    n = len(text)
    for m in pattern.finditer(text):
        open_idx = text.find("(", m.start())
        if open_idx < 0 or open_idx >= n:
            continue

        depth = 0
        in_string = False
        quote = ""
        escaped = False
        i = open_idx
        while i < n:
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    in_string = False
                i += 1
                continue

            if ch in ("'", '"'):
                in_string = True
                quote = ch
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    out.append(text[open_idx + 1 : i].strip())
                    break
            i += 1
    return out


def _parse_jinja_tool_call_args(args_text: str) -> tuple[str, dict] | None:
    if not args_text.strip():
        return None
    try:
        node = ast.parse(f"f({args_text})", mode="eval")
    except Exception:
        return None
    call_node = node.body
    if not isinstance(call_node, ast.Call):
        return None

    kwargs: dict[str, object] = {}
    for kw in call_node.keywords:
        if kw.arg is None:
            continue
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            return None

    pos_args: list[object] = []
    for arg in call_node.args:
        try:
            pos_args.append(ast.literal_eval(arg))
        except Exception:
            return None

    name: object | None = kwargs.get("name")
    if not isinstance(name, str) and pos_args:
        name = pos_args[0]

    arguments: object | None = kwargs.get("arguments")
    if not isinstance(arguments, dict) and len(pos_args) >= 2:
        arguments = pos_args[1]

    if isinstance(name, str) and isinstance(arguments, dict):
        return name, arguments
    return None


def _parse_direct_tool_args(name: str, args_text: str) -> dict | None:
    if not str(name or "").strip() or not str(args_text or "").strip():
        return None
    try:
        node = ast.parse(f"f({args_text})", mode="eval")
    except Exception:
        return None
    call_node = node.body
    if not isinstance(call_node, ast.Call):
        return None

    arguments: dict[str, object] = {}
    for kw in call_node.keywords:
        if kw.arg is None:
            return None
        try:
            arguments[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            return None

    positional_names = _DIRECT_TOOL_POSITIONAL_ARGS.get(str(name), [])
    for idx, arg in enumerate(call_node.args):
        if idx >= len(positional_names):
            return None
        try:
            arguments[positional_names[idx]] = ast.literal_eval(arg)
        except Exception:
            return None

    return arguments


def _parse_direct_tool_invocation(text: str) -> ToolCall | None:
    source = str(text or "").strip()
    if not source:
        return None

    func_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)", source, re.DOTALL)
    if func_match:
        name = func_match.group(1)
        arguments = _parse_direct_tool_args(name, func_match.group(2))
        if isinstance(arguments, dict):
            return _new_tool_call(name, arguments)

    bare_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s+(.+)", source, re.DOTALL)
    if bare_match:
        name = bare_match.group(1)
        if name in _DIRECT_TOOL_POSITIONAL_ARGS:
            arguments = _parse_direct_tool_args(name, bare_match.group(2))
            if isinstance(arguments, dict):
                return _new_tool_call(name, arguments)

    return None


_SHELL_FENCE_RE = re.compile(r"(?is)```(?:bash|sh|shell|zsh)\s*\n(.*?)```")

_SHELL_EXECUTION_CUES = (
    "execution",
    "i will now execute",
    "i'll now execute",
    "execute this plan",
    "execute the command",
    "execute the ",
    "run this command",
    "running command",
    "action/output",
    "i will run",
    "i'll run",
    "i will read",
    "i'll read",
    "i will inspect",
    "i'll inspect",
    "i will check",
    "i'll check",
    "i will list",
    "i'll list",
    "i will analyze",
    "i'll analyze",
    "i will explore",
    "i'll explore",
    "let me run",
    "let me start",
    "let me explore",
    "let me inspect",
    "let me analyze",
    "i am going to run",
    "i'm going to run",
)

_SHELL_NON_EXECUTION_CUES = (
    "example command",
    "example:",
    "for example",
    "for instance",
    "e.g.",
    "you can run",
    "if you run",
    "manually run",
)


def _normalize_shell_execution_context(text: str) -> str:
    return " ".join(
        str(text or "")
        .lower()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .split()
    )


def _parse_shell_fence_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for match in _SHELL_FENCE_RE.finditer(text):
        prelude = _normalize_shell_execution_context(
            text[max(0, match.start() - 320):match.start()]
        )
        if any(cue in prelude for cue in _SHELL_NON_EXECUTION_CUES):
            continue
        if not any(cue in prelude for cue in _SHELL_EXECUTION_CUES):
            continue
        command = _normalize_shell_fence_command(match.group(1))
        if not command:
            continue
        direct_call = _parse_direct_tool_invocation(command)
        if direct_call is not None:
            calls.append(direct_call)
            continue
        calls.append(_new_tool_call("bash", {"command": command}))
    return calls


def _normalize_shell_fence_command(body: str) -> str:
    lines = str(body or "").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""

    normalized: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("$ "):
            normalized.append(stripped[2:])
        else:
            normalized.append(line)
    return "\n".join(normalized).strip()


def _extract_json_objects(text: str) -> List[str]:
    """Extract top-level JSON objects from *text*, respecting string boundaries.

    The previous implementation used a naive brace-counter that treated ``{``/``}``
    inside string values (e.g. Python code, ANSI sequences, progress bars) as
    structural braces, causing incorrect extraction boundaries and failed
    ``json.loads`` calls.  This version tracks whether the scanner is inside a
    JSON string so those characters are ignored.
    """
    candidates: List[str] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue

        start = i
        depth = 0
        in_string = False
        j = i
        while j < n:
            ch = text[j]
            if in_string:
                if ch == "\\":
                    j += 2  # skip the escaped character entirely
                    continue
                if ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : j + 1])
                        i = j + 1
                        break
            j += 1
        else:
            # No matching closing brace found starting at *start* — skip past it
            i += 1

    return candidates


def detect_truncated_tool_call(text: str) -> bool:
    """Return True if *text* looks like a JSON tool call truncated mid-generation.

    The heuristic: more ``{`` than ``}`` (unclosed brace) AND the text
    contains the signature of a tool-call JSON object.
    """
    stripped = text.strip()
    if not stripped or "{" not in stripped:
        return False
    if stripped.count("{") <= stripped.count("}"):
        return False
    lower = stripped.lower()
    return '"tool_call"' in lower or ('"name"' in lower and '"arguments"' in lower)


def extract_partial_write_from_truncated(text: str) -> tuple[str, str, str] | None:
    """Try to salvage a write_file call from a truncated LLM response.

    Returns ``(tool_name, path, partial_content)`` on success, ``None`` otherwise.
    Currently handles ``write_file`` (and alike write-to-disk tools) whose JSON
    was cut off before the closing braces were emitted.

    The extraction strategy:
    - Find the tool name from ``"name": "..."``
    - Find the file path from ``"path": "..."``
    - Find the content value start after ``"content": "`` and unescape whatever
      was emitted before the truncation point.
    """
    if not detect_truncated_tool_call(text):
        return None

    # Must be a file-writing tool
    name_m = re.search(r'"name"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
    if not name_m:
        return None
    try:
        tool_name: str = json.loads(f'"{name_m.group(1)}"')
    except Exception:
        tool_name = name_m.group(1)

    _write_names = {"write_file", "edit_file", "create_file", "save_file"}
    if not any(w in tool_name.lower() for w in _write_names):
        return None

    # Extract path (short string — should be complete even if content is cut off)
    path_m = re.search(r'"path"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
    if not path_m:
        return None
    try:
        path: str = json.loads(f'"{path_m.group(1)}"')
    except Exception:
        path = path_m.group(1)

    if not path:
        return None

    # Extract partial content: everything after the opening quote of "content"
    content_m = re.search(r'"content"\s*:\s*"(.*)', text, re.DOTALL)
    if not content_m:
        return None

    raw = content_m.group(1)

    # Strip trailing incomplete escape sequence (e.g. a lone "\" at the very end)
    while raw.endswith("\\") and not raw.endswith("\\\\"):
        raw = raw[:-1]

    # Attempt proper JSON string decode; fall back to manual unescape
    try:
        content: str = json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        content = (
            raw.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\\\", "\\")
            .replace("\\/", "/")
            .replace("\\r", "\r")
        )

    return tool_name, path, content


__all__ = [
    "parse_tool_calls",
    "parse_tool_call_strict",
    "detect_truncated_tool_call",
    "extract_partial_write_from_truncated",
]
