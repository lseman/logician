from __future__ import annotations

import json
import time
from typing import List, Optional

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


__all__ = ["parse_tool_calls", "parse_tool_call_strict"]
