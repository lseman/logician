"""Core task management tools: think (scratchpad) and todo (task tracking)."""
from __future__ import annotations

import json


def think(thought: str) -> str:
    """Record an internal thought or reasoning step.

    Returns the thought unchanged. Use this to reason through a problem,
    plan next steps, or note observations without producing visible output.
    The thought is returned so it appears in the conversation context.

    Args:
        thought: Internal reasoning or scratchpad text.

    Returns:
        The thought wrapped in a [thought] block.
    """
    return f"[thought]\n{thought}"


def todo(todos: list[dict] | str) -> str:
    """Update the task list for the current session.

    Args:
        todos: List of dicts with keys:
            - content (str): Task description
            - status (str): "pending", "in_progress", or "completed"
            - activeForm (str): Present continuous description (e.g. "Fixing bug")

            Or a JSON string representation of the above list.

    Returns:
        Formatted task list.

    Example:
        todo([
            {"content": "Fix bug", "status": "in_progress", "activeForm": "Fixing bug"},
            {"content": "Write tests", "status": "pending", "activeForm": "Writing tests"},
        ])
    """
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError as e:
            return f"Error: invalid todos JSON: {e}"

    if not isinstance(todos, list):
        return "Error: todos must be a list"

    lines = ["## Task List\n"]
    for item in todos:
        if not isinstance(item, dict):
            continue
        status = item.get("status", "pending")
        content = item.get("content", "")
        icon = {
            "pending": "[ ]",
            "in_progress": "[→]",
            "completed": "[x]",
        }.get(status, "[ ]")
        lines.append(f"{icon} {content}")

    return "\n".join(lines)
