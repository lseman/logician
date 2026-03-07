from __future__ import annotations

import json
import re
import time
from typing import List

from .runtime import HAS_TOON, ToolCall, decode


def parse_tool_calls(text: str, use_toon: bool) -> list[ToolCall]:
    calls: list[ToolCall] = []
    if use_toon and HAS_TOON and decode is not None:
        calls.extend(_parse_toon_tool_calls(text))
    calls.extend(_parse_json_tool_calls(text))

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
    return calls[0] if calls else None


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
                    continue

        if isinstance(data, dict) and "name" in data and "arguments" in data:
            name = data["name"]
            args = data["arguments"]
            if isinstance(name, str) and isinstance(args, dict):
                calls.append(
                    ToolCall(id=f"call_{time.time():.6f}", name=name, arguments=args)
                )

        # Anthropic-native tool block shape:
        # {"type":"tool_use","name":"...","input":{...}}
        if (
            isinstance(data, dict)
            and data.get("type") == "tool_use"
            and isinstance(data.get("name"), str)
            and isinstance(data.get("input"), dict)
        ):
            calls.append(
                ToolCall(
                    id=f"call_{time.time():.6f}",
                    name=str(data["name"]),
                    arguments=dict(data["input"]),
                )
            )

        # Anthropic response wrapper shape:
        # {"content":[{"type":"tool_use","name":"...","input":{...}}], ...}
        if isinstance(data, dict) and isinstance(data.get("content"), list):
            for block in data["content"]:
                if not isinstance(block, dict):
                    continue
                if (
                    block.get("type") == "tool_use"
                    and isinstance(block.get("name"), str)
                    and isinstance(block.get("input"), dict)
                ):
                    calls.append(
                        ToolCall(
                            id=f"call_{time.time():.6f}",
                            name=str(block["name"]),
                            arguments=dict(block["input"]),
                        )
                    )

    return calls


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
