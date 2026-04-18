"""Core task management tools: think (scratchpad) and todo (task tracking)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_FALLBACK_TODO_ITEMS: list[dict[str, Any]] = []

_STATUS_ALIASES = {
    "pending": "not-started",
    "not-started": "not-started",
    "todo": "not-started",
    "in_progress": "in-progress",
    "in-progress": "in-progress",
    "doing": "in-progress",
    "completed": "completed",
    "done": "completed",
    "blocked": "blocked",
}

_STATUS_ICONS = {
    "not-started": "[ ]",
    "in-progress": "[/]",
    "completed": "[x]",
    "blocked": "[!]",
}

_TODO_PATH_ENV = "LOGICIAN_TODO_STATE_PATH"


def _todo_warnings(items: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    in_progress_titles: list[str] = []
    seen_ids: dict[int, str] = {}
    for item in items:
        title = str(item.get("title") or "").strip()
        if not title:
            warnings.append("Task item is missing a title.")
        raw_status = str(item.get("status") or "").strip().lower()
        if raw_status and raw_status not in _STATUS_ALIASES:
            warnings.append(
                f"Task '{title or '<untitled>'}' has an unrecognized status '{raw_status}'. "
                "It will be normalized to 'not-started'."
            )
        normalized_status = _normalize_status(raw_status)
        if normalized_status == "in-progress":
            in_progress_titles.append(title or f"id={item.get('id', '?')}")
        try:
            item_id = int(item.get("id"))
        except Exception:
            item_id = None
        if item_id is None:
            continue
        if item_id in seen_ids:
            warnings.append(
                f"Task id {item_id} is duplicated for '{seen_ids[item_id]}' and '{title or '<untitled>'}'."
            )
        else:
            seen_ids[item_id] = title or "<untitled>"
    if len(in_progress_titles) > 1:
        warnings.append(
            "Multiple tasks are marked in-progress. Keep a single active task unless parallel work is intentional."
        )
    return list(dict.fromkeys(warnings))


def _build_verification_hint(warnings: list[str]) -> str:
    if warnings:
        return (
            "Review the task list carefully before proceeding. "
            "The current items contain validation warnings."
        )
    return "The task list is valid. Confirm that task priorities and statuses are correct."


def think(thought: str) -> dict[str, Any]:
    """Record an internal thought or reasoning step."""
    return {
        "status": "ok",
        "thought": str(thought),
        "view": f"[thought]\n{thought}",
    }


def todo(
    todos: list[dict[str, Any]] | str | None = None,
    *,
    command: str = "",
    items: list[dict[str, Any]] | None = None,
    id: int | None = None,
    status: str | None = None,
    title: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Track and inspect the current task list.

    Supported modes
    ---------------
    1. Legacy set mode:
       todo([{"content": "Fix bug", "status": "in_progress"}])
    2. Structured command mode:
       todo(command="view")
       todo(command="set", items=[{"title": "Fix bug", "status": "in-progress"}])

    Returns:
        dict with structured task state, validation metadata, and a rendered markdown view.
    """
    if command:
        return _todo_command(
            command=command,
            items=items,
            id=id,
            status=status,
            title=title,
            note=note,
        )

    parsed = _coerce_legacy_todos(todos)
    if isinstance(parsed, dict):
        return parsed

    _set_items(parsed)
    return {"status": "ok", "todos": parsed, "view": _render_markdown(parsed), "mode": "legacy"}


def _todo_command(
    *,
    command: str,
    items: list[dict[str, Any]] | None,
    id: int | None,
    status: str | None,
    title: str | None,
    note: str | None,
) -> dict[str, Any]:
    normalized_command = str(command or "").strip().lower()

    if normalized_command == "view":
        current = _get_items()
        warnings = _todo_warnings(current)
        return {
            "status": "ok",
            "todos": current,
            "view": _render_markdown(current),
            "warnings": warnings,
            "verification_hint": _build_verification_hint(warnings),
        }

    if normalized_command == "validate":
        normalized = _get_items() if items is None else _normalize_items(items)
        warnings = _todo_warnings(normalized)
        return {
            "status": "ok",
            "todos": normalized,
            "view": _render_markdown(normalized),
            "warnings": warnings,
            "verification_hint": _build_verification_hint(warnings),
        }

    if normalized_command in {"set", "update"}:
        if not isinstance(items, list):
            return {"status": "error", "error": "items must be a list of todo dicts"}
        normalized = _normalize_items(items)
        warnings = _todo_warnings(normalized)
        _set_items(normalized)
        return {
            "status": "ok",
            "count": len(normalized),
            "todos": normalized,
            "view": _render_markdown(normalized),
            "warnings": warnings,
            "verification_hint": _build_verification_hint(warnings),
        }

    if normalized_command == "add":
        clean_title = str(title or "").strip()
        if not clean_title:
            return {"status": "error", "error": "title is required for add"}
        existing = _get_items()
        next_id = max((int(item.get("id", 0) or 0) for item in existing), default=0) + 1
        new_item = {
            "id": next_id,
            "title": clean_title,
            "status": _normalize_status(status),
            "note": str(note or "").strip(),
        }
        updated = [*existing, new_item]
        warnings = _todo_warnings(updated)
        _set_items(updated)
        return {
            "status": "ok",
            "added": new_item,
            "todos": updated,
            "view": _render_markdown(updated),
            "warnings": warnings,
            "verification_hint": _build_verification_hint(warnings),
        }

    if normalized_command == "mark":
        if id is None:
            return {"status": "error", "error": "id is required for mark"}
        normalized_status = _normalize_status(status)
        updated = []
        found = None
        for item in _get_items():
            current = dict(item)
            if normalized_status == "in-progress" and current.get("id") != id:
                if current.get("status") == "in-progress":
                    current["status"] = "not-started"
            if current.get("id") == id:
                current["status"] = normalized_status
                found = dict(current)
            updated.append(current)
        if found is None:
            return {"status": "error", "error": f"No item with id={id}"}
        warnings = _todo_warnings(updated)
        _set_items(updated)
        return {
            "status": "ok",
            "updated": found,
            "todos": updated,
            "view": _render_markdown(updated),
            "warnings": warnings,
            "verification_hint": _build_verification_hint(warnings),
        }

    if normalized_command == "note":
        if id is None:
            return {"status": "error", "error": "id is required for note"}
        updated = []
        found = None
        for item in _get_items():
            current = dict(item)
            if current.get("id") == id:
                current["note"] = str(note or "").strip()
                found = dict(current)
            updated.append(current)
        if found is None:
            return {"status": "error", "error": f"No item with id={id}"}
        warnings = _todo_warnings(updated)
        _set_items(updated)
        return {
            "status": "ok",
            "updated": found,
            "todos": updated,
            "view": _render_markdown(updated),
            "warnings": warnings,
            "verification_hint": _build_verification_hint(warnings),
        }

    if normalized_command == "clear":
        _set_items([])
        return {
            "status": "ok",
            "todos": [],
            "view": "(empty todo list)",
            "warnings": [],
            "verification_hint": _build_verification_hint([]),
        }

    return {
        "status": "error",
        "error": (
            f"Unknown command '{normalized_command}'. "
            "Valid commands: view, set, update, add, mark, note, clear, validate"
        ),
    }


def _coerce_legacy_todos(
    todos: list[dict[str, Any]] | str | None,
) -> list[dict[str, Any]] | dict[str, Any]:
    if todos is None:
        return _get_items()
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError as exc:
            return {"status": "error", "error": f"invalid todos JSON: {exc}"}
    if not isinstance(todos, list):
        return {"status": "error", "error": "todos must be a list"}
    return _normalize_items(todos)


def _normalize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        item_id = item.get("id", index)
        try:
            item_id = int(item_id)
        except Exception:
            item_id = index
        title = str(item.get("title") or item.get("content") or f"Task {index}").strip()
        task_note = str(item.get("note") or item.get("activeForm") or "").strip()
        normalized.append(
            {
                "id": item_id,
                "title": title,
                "status": _normalize_status(item.get("status")),
                "note": task_note,
            }
        )
    return normalized


def _normalize_status(value: Any) -> str:
    key = str(value or "not-started").strip().lower()
    return _STATUS_ALIASES.get(key, "not-started")


def _render_markdown(items: list[dict[str, Any]]) -> str:
    if not items:
        return "## Task List\n\n(empty todo list)"
    lines = ["## Task List", ""]
    for item in items:
        icon = _STATUS_ICONS.get(str(item.get("status") or "not-started"), "[ ]")
        title = str(item.get("title") or "").strip()
        lines.append(f"{icon} {title}")
        item_note = str(item.get("note") or "").strip()
        if item_note:
            lines.append(f"    {item_note}")
    return "\n".join(lines)


def get_todo_state_path(base_dir: str | Path | None = None) -> Path:
    override = str(os.getenv(_TODO_PATH_ENV, "") or "").strip()
    if override:
        return Path(override).expanduser()
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    return root / ".logician" / "state" / "todos.json"


def load_persisted_todos(base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    path = get_todo_state_path(base_dir)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return _normalize_items([dict(item) for item in payload if isinstance(item, dict)])


def _persist_items(items: list[dict[str, Any]], base_dir: str | Path | None = None) -> None:
    path = get_todo_state_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([dict(item) for item in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_items() -> list[dict[str, Any]]:
    current_ctx = globals().get("ctx")
    if current_ctx is not None:
        stored = getattr(current_ctx, "todo_items", None)
        if isinstance(stored, list):
            return [dict(item) for item in stored if isinstance(item, dict)]
    persisted = load_persisted_todos()
    if persisted:
        return persisted
    return [dict(item) for item in _FALLBACK_TODO_ITEMS]


def _set_items(items: list[dict[str, Any]]) -> None:
    normalized = [dict(item) for item in items if isinstance(item, dict)]
    current_ctx = globals().get("ctx")
    if current_ctx is not None:
        setattr(current_ctx, "todo_items", normalized)
    else:
        _FALLBACK_TODO_ITEMS[:] = normalized
    try:
        _persist_items(normalized)
    except OSError:
        pass
