from __future__ import annotations

if "llm" not in globals():

    class _NoOpLLM:
        def tool(self, func=None, *, name=None, description=None):
            return func if func is not None else (lambda f: f)

    llm = _NoOpLLM()

import time
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Persistent in-memory stores
# ---------------------------------------------------------------------------

_scratch: dict[str, Any] = {}
"""Key-value scratch pad; persists across tool calls for the session."""

_tasks: list[dict] = []
"""Structured task list; each entry: {id, title, status, notes, created_at, updated_at}."""

_task_id_counter: list[int] = [0]


# ---------------------------------------------------------------------------
# Scratch pad tools
# ---------------------------------------------------------------------------


@llm.tool(description="Write a value to the in-memory scratch pad under a named key.")
def scratch_write(key: str, value: Any) -> str:
    """Use when: Store intermediate results, notes, or data between steps.

    Triggers: remember, save to scratch, store value, keep track of, note this, record.
    Avoid when: You need persistent disk storage — write to a file instead.
    Inputs:
      key (str): Identifier for the stored value.
      value (Any): Anything JSON-serialisable.
    Returns: JSON with {status, key, overwritten}.
    Side effects: Mutates _scratch.
    """
    overwritten = key in _scratch
    _scratch[key] = value
    return _safe_json({"status": "ok", "key": key, "overwritten": overwritten})


@llm.tool(description="Read a value from the in-memory scratch pad by key.")
def scratch_read(key: str) -> str:
    """Use when: Retrieve a previously stored intermediate result or note.

    Triggers: recall, get from scratch, retrieve, what did I store, read scratch.
    Avoid when: The key was never written — use scratch_list first to confirm.
    Inputs:
      key (str): Key to look up.
    Returns: JSON with {status, key, value} or {status: "not_found"}.
    Side effects: Read-only.
    """
    if key not in _scratch:
        return _safe_json(
            {"status": "not_found", "key": key, "available": list(_scratch.keys())}
        )
    return _safe_json({"status": "ok", "key": key, "value": _scratch[key]})


@llm.tool(description="Delete a key from the in-memory scratch pad.")
def scratch_delete(key: str) -> str:
    """Use when: Remove a scratch pad entry that is no longer needed.

    Triggers: remove from scratch, delete note, clear key.
    Avoid when: You want to clear everything — use scratch_list + scratch_delete per key, or reset.
    Inputs:
      key (str): Key to remove.
    Returns: JSON with {status, key, existed}.
    Side effects: Mutates _scratch.
    """
    existed = key in _scratch
    _scratch.pop(key, None)
    return _safe_json({"status": "ok", "key": key, "existed": existed})


@llm.tool(description="List all keys currently stored in the in-memory scratch pad.")
def scratch_list() -> str:
    """Use when: Inspect what's currently stored before reading or writing.

    Triggers: show scratch, list notes, what's stored, scratch contents.
    Avoid when: Nothing is stored yet (returns empty list).
    Inputs: None.
    Returns: JSON with {count, keys, entries} where entries is a [{key, type, preview}] list.
    Side effects: Read-only.
    """
    entries = []
    for k, v in _scratch.items():
        preview = repr(v)
        if len(preview) > 120:
            preview = preview[:120] + "..."
        entries.append({"key": k, "type": type(v).__name__, "preview": preview})
    return _safe_json({"status": "ok", "count": len(entries), "entries": entries})


# ---------------------------------------------------------------------------
# Task tracker tools
# ---------------------------------------------------------------------------

_VALID_STATUSES = {"todo", "in_progress", "done", "blocked"}


@llm.tool(description="Add a new task to the session task list.")
def task_add(title: str, notes: str = "") -> str:
    """Use when: Plan work by breaking it into tracked tasks before starting.

    Triggers: add task, create task, new task, plan step, add to do, track this, write down.
    Avoid when: You only have one trivial action — just do it directly.
    Inputs:
      title (str): Short description of the task (≤ 120 chars recommended).
      notes (str, optional): Additional context or acceptance criteria.
    Returns: JSON with {status, task_id, title}.
    Side effects: Appends to _tasks.
    """
    _task_id_counter[0] += 1
    task_id = _task_id_counter[0]
    now = time.strftime("%H:%M:%S")
    _tasks.append(
        {
            "id": task_id,
            "title": title,
            "status": "todo",
            "notes": notes,
            "created_at": now,
            "updated_at": now,
        }
    )
    return _safe_json({"status": "ok", "task_id": task_id, "title": title})


@llm.tool(description="Update the status of an existing task.")
def task_update(
    task_id: int,
    status: Literal["todo", "in_progress", "done", "blocked"],
    notes: str = "",
) -> str:
    """Use when: Mark a task as in-progress when starting it, or done when finished.

    Triggers: start task, finish task, mark done, mark blocked, update task, task status.
    Avoid when: The task_id doesn't exist — use task_list first.
    Inputs:
      task_id (int): The numeric ID returned by task_add.
      status (str): One of "todo", "in_progress", "done", "blocked".
      notes (str, optional): Additional notes to append (not replace) to the task.
    Returns: JSON with {status, task_id, new_status}.
    Side effects: Mutates the matching entry in _tasks.
    """
    if status not in _VALID_STATUSES:
        return _safe_json(
            {
                "status": "error",
                "error": f"Invalid status '{status}'. Must be one of {sorted(_VALID_STATUSES)}.",
            }
        )
    for task in _tasks:
        if task["id"] == task_id:
            task["status"] = status
            task["updated_at"] = time.strftime("%H:%M:%S")
            if notes:
                sep = "\n" if task["notes"] else ""
                task["notes"] = task["notes"] + sep + notes
            return _safe_json(
                {
                    "status": "ok",
                    "task_id": task_id,
                    "new_status": status,
                    "title": task["title"],
                }
            )
    return _safe_json({"status": "not_found", "task_id": task_id})


@llm.tool(
    description="List all tasks in the session task list with their current statuses."
)
def task_list(filter_status: str = "") -> str:
    """Use when: Review what's planned, in progress, or done at any point during work.

    Triggers: show tasks, list tasks, what's next, plan overview, task status, progress, todo list.
    Avoid when: No tasks have been added yet.
    Inputs:
      filter_status (str, optional): If given, return only tasks with that status
                                     ("todo", "in_progress", "done", "blocked").
    Returns: JSON with {count, tasks} list of {id, title, status, notes, created_at, updated_at}.
    Side effects: Read-only.
    """
    tasks = _tasks
    if filter_status:
        tasks = [t for t in tasks if t["status"] == filter_status]

    summary = {s: sum(1 for t in _tasks if t["status"] == s) for s in _VALID_STATUSES}
    return _safe_json(
        {
            "status": "ok",
            "total": len(_tasks),
            "showing": len(tasks),
            "summary": summary,
            "tasks": tasks,
        }
    )


@llm.tool(description="Clear all tasks from the session task list.")
def task_clear() -> str:
    """Use when: Reset the task list at the start of a new work session or after completing a project.

    Triggers: clear tasks, reset tasks, wipe task list, start fresh tasks.
    Avoid when: You still need the task history.
    Inputs: None.
    Returns: JSON with {status, cleared_count}.
    Side effects: Empties _tasks and resets the ID counter.
    """
    cleared = len(_tasks)
    _tasks.clear()
    _task_id_counter[0] = 0
    return _safe_json({"status": "ok", "cleared_count": cleared})
