"""Core tool discovery helpers."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


def tool_search(query: str, top_k: int = 8) -> dict[str, Any]:
    q = str(query or "").strip()
    if not q:
        return _err("query is required")

    registry = _resolve_registry()
    if registry is None:
        return _err("tool registry is unavailable in the current execution context")

    try:
        tools = list(registry.list_tools())
    except Exception as exc:
        return _err(f"cannot enumerate tools: {exc}")

    q_l = q.lower()
    q_tokens = _tokens(q_l)
    scored: list[tuple[float, Any]] = []
    for tool in tools:
        text = " ".join(
            [
                str(tool.name or ""),
                str(tool.description or ""),
                str(tool.skill_id or ""),
                " ".join(str(param.name or "") for param in list(tool.parameters or [])),
            ]
        ).strip()
        text_l = text.lower()
        score = (SequenceMatcher(None, q_l, text_l).ratio() * 0.7) + (
            _token_overlap(q_tokens, _tokens(text_l)) * 0.3
        )
        if q_l in str(tool.name or "").lower():
            score += 0.25
        if score >= 0.15:
            scored.append((score, tool))

    scored.sort(key=lambda item: item[0], reverse=True)
    limit = max(1, min(20, int(top_k or 8)))
    matches = [
        {
            "name": tool.name,
            "description": tool.description,
            "skill_id": tool.skill_id,
            "source_path": tool.source_path,
            "parameters": [str(param.name or "") for param in list(tool.parameters or [])],
            "score": round(float(score), 4),
        }
        for score, tool in scored[:limit]
    ]

    return {
        "status": "ok",
        "query": q,
        "count": len(matches),
        "matches": matches,
        "tool_count": len(tools),
    }


def _resolve_registry() -> Any | None:
    registry = globals().get("tool_registry")
    if registry is not None:
        return registry
    ctx = globals().get("ctx")
    return getattr(ctx, "_tool_registry", None)


def _tokens(value: str) -> set[str]:
    return {
        part
        for part in str(value or "").replace("-", " ").replace("_", " ").split()
        if part
    }


def _token_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / max(1, len(left))


def _err(message: str) -> dict[str, Any]:
    return {"status": "error", "error": message}
