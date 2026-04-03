"""Plan mode tools — block write/execute tools for safe exploration.

enter_plan_mode()  → sets ctx.plan_mode = True
exit_plan_mode()   → sets ctx.plan_mode = False

When plan mode is active, the following tools are blocked by the execution
layer: write_file, edit_file, apply_edit_block, smart_edit, bash, patch.
"""

from __future__ import annotations

import json as _json
from typing import Any

if "ctx" not in globals():
    ctx: Any = None  # injected by ToolRegistry


def enter_plan_mode() -> str:
    """Activate plan mode. Write and execute tools are blocked until exit_plan_mode()."""
    if ctx is None:
        return _json.dumps({"status": "error", "error": "ctx not available"})
    ctx.plan_mode = True
    return _json.dumps(
        {"status": "ok", "plan_mode": True, "message": "Plan mode ON — write tools are blocked."}
    )


def exit_plan_mode() -> str:
    """Deactivate plan mode. All tools become available again."""
    if ctx is None:
        return _json.dumps({"status": "error", "error": "ctx not available"})
    ctx.plan_mode = False
    return _json.dumps(
        {"status": "ok", "plan_mode": False, "message": "Plan mode OFF — all tools restored."}
    )


__all__ = ["enter_plan_mode", "exit_plan_mode"]
__tools__ = [enter_plan_mode, exit_plan_mode]
