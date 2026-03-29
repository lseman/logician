"""TurnState: single source of truth for all loop state within one agent turn."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from ..tools.runtime import ToolCall


@dataclass
class ToolResultRecord:
    call_id: str
    tool_name: str
    status: str
    output: str = ""
    error: str | None = None
    read_only: bool = False
    writes_files: bool = False
    verifier: bool = False
    cache_hit: bool = False
    has_content: bool = False


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
    # Paths whose contents were inspected this turn via file/symbol reads.
    files_read: list[str] = field(default_factory=list)
    # Structured results for executed tools this turn.
    tool_results: list[ToolResultRecord] = field(default_factory=list)
    # Domain groups activated for this turn
    domain_groups_activated: set[str] = field(default_factory=set)
    # Tool names available to the current turn (used by prompt components / guards).
    available_tool_names: set[str] = field(default_factory=set)

    # Guardrail state: guard_name → nudge count
    guardrail_nudges: dict[str, int] = field(default_factory=dict)

    # Turn classification
    classified_as: Literal["social", "informational", "execution", "design"] = (
        "execution"
    )
    # Raw user message for this turn — used by SkillPlaybookComponent for routing
    user_query: str = ""

    # Output
    final_response: str | None = None
    # All thinking/planning content collected this turn (pre-turn plan + <think> blocks)
    thinking_log: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    # Last raw tool output text — set by AgentLoop after each dispatch batch.
    # Used by InspectionGuard to detect response contradictions.
    last_tool_output: str | None = None

    # Confidence gate retry count for the current turn.
    confidence_retries: int = 0
    tool_repair_attempts: int = 0

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

    def record_read(self, path: str) -> None:
        """Record that a file was inspected/read."""
        if path not in self.files_read:
            self.files_read.append(path)

    def record_tool_result(self, result: ToolResultRecord) -> None:
        self.tool_results.append(result)

    def last_write_index(self) -> int:
        """Index in tool_calls of the most recent write tool call. -1 if none."""
        write_names = {
            "write_file", "edit_file", "apply_edit_block",
            "edit_file_replace", "sed_replace", "regex_replace",
            "multi_edit", "multi_patch", "apply_unified_diff",
            "scratch_write", "scratch_delete",
        }
        for i in range(len(self.tool_calls) - 1, -1, -1):
            if self.tool_calls[i].name in write_names:
                return i
        return -1

    def last_write_result_index(self) -> int:
        for i in range(len(self.tool_results) - 1, -1, -1):
            if self.tool_results[i].writes_files:
                return i
        return -1
