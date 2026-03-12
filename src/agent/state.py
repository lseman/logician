"""TurnState: single source of truth for all loop state within one agent turn."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from ..tools.runtime import ToolCall


@dataclass
class TurnState:
    turn_id: str

    # Loop control
    iteration: int = 0
    consecutive_tool_count: int = 0

    # Tool tracking
    # Ordered log of all tool calls this turn (append-only)
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Maps "tool_name:sha256(args)" → occurrence count
    seen_signatures: dict[str, int] = field(default_factory=dict)
    # Paths of files written/edited this turn
    files_written: list[str] = field(default_factory=list)
    # Domain groups activated for this turn
    domain_groups_activated: set[str] = field(default_factory=set)

    # Guardrail state: guard_name → nudge count
    guardrail_nudges: dict[str, int] = field(default_factory=dict)

    # Turn classification
    classified_as: Literal["social", "informational", "execution", "design"] = (
        "execution"
    )

    # Output
    final_response: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)

    def tool_signature(self, call: ToolCall) -> str:
        """Stable hash for a tool call — used by DuplicateToolGuard."""
        raw = json.dumps(
            {"n": call.name, "a": call.arguments},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"{call.name}:{digest}"

    def record_call(self, call: ToolCall) -> None:
        """Append a tool call to the ordered log and update the signature counter."""
        self.tool_calls.append(call)
        sig = self.tool_signature(call)
        self.seen_signatures[sig] = self.seen_signatures.get(sig, 0) + 1

    def record_write(self, path: str) -> None:
        """Record a file write/edit."""
        if path not in self.files_written:
            self.files_written.append(path)

    def last_write_index(self) -> int:
        """Index in tool_calls of the most recent write tool call. -1 if none."""
        write_names = {"write_file", "edit_file", "apply_edit_block"}
        for i in range(len(self.tool_calls) - 1, -1, -1):
            if self.tool_calls[i].name in write_names:
                return i
        return -1
