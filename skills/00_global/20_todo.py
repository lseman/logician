"""Structured task-tracking tool (todo list).

Gives the agent a persistent, session-scoped todo list to plan and track
multi-step work — equivalent to a mental scratchpad that survives across
tool calls within the same session.

State is kept in-process inside _TODO_STATE (dict keyed by session_id).
In a fresh process the state starts empty; use todo("init") or
todo("update") to populate it before the first action.
"""

from __future__ import annotations

if "llm" not in globals():

    class _NoOpLLM:
        def tool(self, func=None, *, name=None, description=None):
            return func if func is not None else (lambda f: f)

    llm = _NoOpLLM()

import json as _json_mod
from typing import Any

if "_safe_json" not in globals():

    def _safe_json(obj: Any) -> str:
        try:
            return _json_mod.dumps(obj, ensure_ascii=False)
        except Exception as e:
            return _json_mod.dumps(
                {"status": "error", "error": f"JSON encode error: {e}"}
            )


# Module-level state shared across calls within the same process
_TODO_STATE: dict[str, list[dict]] = {}
_ACTIVE_SESSION: str = "default"

VALID_STATUSES = {"not-started", "in-progress", "completed", "blocked"}


def _todos() -> list[dict]:
    return _TODO_STATE.setdefault(_ACTIVE_SESSION, [])


def _render(items: list[dict]) -> str:
    if not items:
        return "(empty todo list)"
    icons = {
        "not-started": "○",
        "in-progress": "●",
        "completed": "✓",
        "blocked": "✗",
    }
    lines = []
    for t in items:
        icon = icons.get(t.get("status", "not-started"), "?")
        tid = t.get("id", "?")
        title = t.get("title", "?")
        note = f"  ↳ {t['note']}" if t.get("note") else ""
        lines.append(f"  [{icon}] #{tid} {title}{note}")
    return "\n".join(lines)


@llm.tool(
    description=(
        "Manage a structured task list for the current work session. "
        "Use this to plan multi-step work, track progress, and signal what you are currently doing. "
        "Call it frequently to keep an up-to-date plan visible."
    )
)
def todo(
    command: str,
    items: list | None = None,
    id: int | None = None,
    status: str | None = None,
    title: str | None = None,
    note: str | None = None,
) -> str:
    """Use when: Planning multi-step tasks, starting work on a sub-task, marking completion.

    Triggers: todo, task list, plan steps, track progress, mark done, mark in-progress,
              create plan, what's next, task tracker, checklist.
    Avoid when: Single-step tasks that complete in one tool call.

    Commands:
      "view"          – show the current todo list (read-only)
      "set"           – replace the entire list with `items` (list of {id, title, status})
      "update"        – same as set; alias
      "add"           – append a new item with `title` and optional `status` (default not-started)
      "mark"          – change the `status` of item `id` to `status`
                        valid statuses: not-started, in-progress, completed, blocked
      "note"          – attach a short `note` to item `id`
      "clear"         – delete all items

    Workflow rules:
      1. Use "set" at the start of any multi-step task to lay out the full plan.
      2. "mark" ONE item as in-progress BEFORE starting work on it.
      3. "mark" it completed IMMEDIATELY after finishing — do not batch completions.
      4. Only one item should be in-progress at a time.

    Returns: JSON with status and a rendered plain-text view of the current list.
    """
    command = (command or "").strip().lower()

    if command == "view":
        return _safe_json(
            {"status": "ok", "todos": _todos(), "view": _render(_todos())}
        )

    elif command in ("set", "update"):
        if not isinstance(items, list):
            return _safe_json(
                {
                    "status": "error",
                    "error": "items must be a list of {id, title, status} dicts",
                }
            )
        parsed = []
        for item in items:
            if not isinstance(item, dict):
                continue
            s = item.get("status", "not-started")
            if s not in VALID_STATUSES:
                s = "not-started"
            parsed.append(
                {
                    "id": item.get("id", len(parsed) + 1),
                    "title": str(item.get("title", f"Task {len(parsed) + 1}")),
                    "status": s,
                    "note": item.get("note", ""),
                }
            )
        _TODO_STATE[_ACTIVE_SESSION] = parsed
        return _safe_json(
            {"status": "ok", "count": len(parsed), "view": _render(parsed)}
        )

    elif command == "add":
        if not title:
            return _safe_json({"status": "error", "error": "title is required for add"})
        s = status or "not-started"
        if s not in VALID_STATUSES:
            s = "not-started"
        existing = _todos()
        new_id = max((t["id"] for t in existing), default=0) + 1
        new_item = {"id": new_id, "title": title, "status": s, "note": note or ""}
        existing.append(new_item)
        return _safe_json(
            {"status": "ok", "added": new_item, "view": _render(existing)}
        )

    elif command == "mark":
        if id is None:
            return _safe_json({"status": "error", "error": "id is required for mark"})
        if not status or status not in VALID_STATUSES:
            return _safe_json(
                {
                    "status": "error",
                    "error": f"status must be one of: {sorted(VALID_STATUSES)}",
                }
            )
        existing = _todos()
        target = next((t for t in existing if t["id"] == id), None)
        if target is None:
            return _safe_json({"status": "error", "error": f"No item with id={id}"})

        # Enforce single in-progress rule
        if status == "in-progress":
            current_ip = [
                t for t in existing if t["status"] == "in-progress" and t["id"] != id
            ]
            if current_ip:
                ids = [t["id"] for t in current_ip]
                # Auto-move them back to not-started with a note
                for t in current_ip:
                    t["status"] = "not-started"

        target["status"] = status
        return _safe_json(
            {"status": "ok", "updated": target, "view": _render(existing)}
        )

    elif command == "note":
        if id is None:
            return _safe_json({"status": "error", "error": "id is required for note"})
        existing = _todos()
        target = next((t for t in existing if t["id"] == id), None)
        if target is None:
            return _safe_json({"status": "error", "error": f"No item with id={id}"})
        target["note"] = note or ""
        return _safe_json(
            {"status": "ok", "updated": target, "view": _render(existing)}
        )

    elif command == "clear":
        _TODO_STATE[_ACTIVE_SESSION] = []
        return _safe_json({"status": "ok", "view": "(cleared)"})

    else:
        return _safe_json(
            {
                "status": "error",
                "error": f"Unknown command '{command}'. Valid: view, set, update, add, mark, note, clear",
            }
        )


__all__ = ["todo"]
