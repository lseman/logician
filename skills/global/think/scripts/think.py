from __future__ import annotations

# =============================================================================
# THINK — Explicit pre-action planning checkpoint
# =============================================================================
# Provides a structured `think` tool that the agent calls BEFORE acting on any
# complex task. The agent writes out its problem decomposition and approach as
# tool arguments — the act of constructing these arguments IS the planning step.
#
# Stored under scratch key "_think_<session_id>_<n>" so the plan persists across
# tool iterations and is visible in the tool-call trace.
#
# Design rationale:
#   - Architecturally clean: no sub-LLM call, no tight coupling to backends
#   - Trace-visible: the plan appears explicitly in the tool call log
#   - Complementary to config-level `pre_turn_thinking` (automatic, fire-and-forget)
#   - Inspired by Anthropic's recommended "think" tool pattern for extended thinking
# =============================================================================

import json
from datetime import datetime
from typing import Optional

# Module-level store keyed by session — avoids cross-session bleed
_THINK_STORE: dict[str, list[dict]] = {}

# Injected by bootstrap — safe fallback for standalone import
try:
    _safe_json  # type: ignore[used-before-def]
except NameError:

    def _safe_json(d):  # type: ignore[return]
        try:
            return json.dumps(d, indent=2, ensure_ascii=False)
        except Exception:
            return str(d)


def _session_key() -> str:
    """Return a stable session key — uses ctx session_id when available."""
    try:
        ctx_obj = ctx  # type: ignore[name-defined]
        if ctx_obj is not None and hasattr(ctx_obj, "session_id"):
            return str(ctx_obj.session_id)
    except NameError:
        pass
    return "default"


def think(
    problem: str,
    approach: Optional[str] = None,
    steps: Optional[list[str]] = None,
    note: Optional[str] = None,
) -> str:
    """Record a structured plan before acting.

    Args:
        problem:  One-sentence description of what you are trying to solve.
        approach: Brief description of your overall strategy (optional).
        steps:    Ordered list of concrete actions you intend to take (optional but recommended).
        note:     Any caveats, risks, or things to watch out for (optional).

    Returns:
        Confirmation string with the stored plan summary.
    """
    if not problem or not problem.strip():
        return _safe_json({"status": "error", "error": "'problem' must not be empty."})

    key = _session_key()
    store = _THINK_STORE.setdefault(key, [])

    entry: dict = {
        "n": len(store) + 1,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "problem": problem.strip(),
    }
    if approach:
        entry["approach"] = approach.strip()
    if steps:
        entry["steps"] = [s.strip() for s in steps if s.strip()]
    if note:
        entry["note"] = note.strip()

    store.append(entry)

    # Also write to scratch so it survives if scratch_read is called later
    try:
        scratch_key = f"_think_{len(store)}"
        call_tool("scratch_write", {"key": scratch_key, "value": _safe_json(entry)})  # type: ignore[name-defined]
    except Exception:
        pass  # scratch unavailable — plan is still in _THINK_STORE

    # Format a human-readable summary to return to the agent
    lines: list[str] = [
        f"[Plan #{entry['n']} recorded at {entry['timestamp']}]",
        f"Problem: {entry['problem']}",
    ]
    if approach:
        lines.append(f"Approach: {entry['approach']}")
    if steps:
        lines.append("Steps:")
        for i, s in enumerate(entry.get("steps", []), 1):
            lines.append(f"  {i}. {s}")
    if note:
        lines.append(f"Note: {entry['note']}")

    lines.append("")
    lines.append("Plan recorded. Proceed with step 1.")

    return "\n".join(lines)


def think_recall(n: Optional[int] = None) -> str:
    """Return stored plans from this session.

    Args:
        n: Return only the n-th plan (1-indexed). If omitted, returns all plans.

    Returns:
        JSON string with the stored plan(s).
    """
    key = _session_key()
    store = _THINK_STORE.get(key, [])

    if not store:
        return _safe_json({"status": "empty", "plans": []})

    if n is not None:
        idx = int(n) - 1
        if idx < 0 or idx >= len(store):
            return _safe_json(
                {
                    "status": "error",
                    "error": f"Plan #{n} not found. Available: 1–{len(store)}.",
                }
            )
        return _safe_json({"status": "ok", "plan": store[idx]})

    return _safe_json({"status": "ok", "count": len(store), "plans": store})


__tools__ = [think, think_recall]
