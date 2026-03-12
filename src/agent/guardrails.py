"""GuardrailEngine: all guards run on every response; hard_stop beats nudge."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..config import Config
from ..tools.runtime import ToolCall
from .state import TurnState


@dataclass
class GuardrailResult:
    passed: bool
    nudge: str | None = None
    hard_stop: bool = False
    guard_name: str = ""


@runtime_checkable
class Guard(Protocol):
    name: str

    def check(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult: ...


class GuardrailEngine:
    def __init__(self, guards: list[Guard]) -> None:
        self.guards = guards

    def run(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult:
        """Run ALL guards. hard_stop > nudge > pass. First in list wins within tier."""
        results = [g.check(state, response, tool_calls) for g in self.guards]
        hard_stops = [r for r in results if r.hard_stop]
        if hard_stops:
            return hard_stops[0]
        failures = [r for r in results if not r.passed]
        if failures:
            return failures[0]
        return GuardrailResult(passed=True)


class DuplicateToolGuard:
    name = "duplicate_tool"

    def check(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult:
        for call in tool_calls:
            sig = state.tool_signature(call)
            count = state.seen_signatures.get(sig, 0)
            if count >= 3:
                return GuardrailResult(
                    passed=False,
                    nudge=(
                        f"You already called {call.name} with the same arguments. "
                        "Try a different approach."
                    ),
                    hard_stop=True,
                    guard_name=self.name,
                )
            if count >= 2:
                return GuardrailResult(
                    passed=False,
                    nudge=(
                        f"You already called {call.name} with the same arguments. "
                        "Try a different approach."
                    ),
                    hard_stop=False,
                    guard_name=self.name,
                )
        return GuardrailResult(passed=True, guard_name=self.name)


class ConsecutiveToolGuard:
    name = "consecutive_tool"

    def __init__(self, config: Config) -> None:
        self._max = config.max_consecutive_tool_calls

    def check(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult:
        if state.consecutive_tool_count > self._max and tool_calls:
            return GuardrailResult(
                passed=False,
                nudge=(
                    "You have called many tools in a row. "
                    "Please consolidate your findings and produce an answer."
                ),
                hard_stop=False,
                guard_name=self.name,
            )
        return GuardrailResult(passed=True, guard_name=self.name)


_TOOL_CLAIM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\bi (?:ran|executed|called|used|applied|read|wrote|edited)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi've (?:already |just )?(?:run|executed|called|used|applied|read|written|edited)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:done|completed|finished)\b.{0,60}\b(?:tool|file|test|edit)\b",
        re.IGNORECASE | re.DOTALL,
    ),
]


class ToolClaimGuard:
    name = "tool_claim"

    def check(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult:
        if tool_calls:
            return GuardrailResult(passed=True, guard_name=self.name)
        for pattern in _TOOL_CLAIM_PATTERNS:
            if pattern.search(response):
                return GuardrailResult(
                    passed=False,
                    nudge="Please use a tool instead of describing what you would do.",
                    hard_stop=False,
                    guard_name=self.name,
                )
        return GuardrailResult(passed=True, guard_name=self.name)


_VERIFICATION_NAMES = {"test", "pytest", "ruff", "lint", "check", "verify", "mypy"}


class VerificationGuard:
    name = "verification"

    def check(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult:
        if not state.files_written:
            return GuardrailResult(passed=True, guard_name=self.name)
        if tool_calls:
            return GuardrailResult(passed=True, guard_name=self.name)
        # Check calls after last write for any verification tool
        last_write = state.last_write_index()
        calls_after = state.tool_calls[last_write + 1 :]
        for call in calls_after:
            if any(pat in call.name.lower() for pat in _VERIFICATION_NAMES):
                return GuardrailResult(passed=True, guard_name=self.name)
        return GuardrailResult(
            passed=False,
            nudge=(
                "You've modified files but haven't verified. "
                "Please run tests or a linter."
            ),
            hard_stop=False,
            guard_name=self.name,
        )


class StallGuard:
    name = "stall"

    def check(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult:
        if tool_calls:
            return GuardrailResult(passed=True, guard_name=self.name)
        if state.guardrail_nudges.get("stall", 0) >= 2:
            return GuardrailResult(
                passed=False,
                nudge="The agent appears to be stalled. Stopping turn.",
                hard_stop=True,
                guard_name=self.name,
            )
        return GuardrailResult(passed=True, guard_name=self.name)


class InspectionGuard:
    """Disabled by default. Enable via config.enable_inspection_guard when
    LLM-based contradiction detection is needed."""

    name = "inspection"

    def check(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult:
        return GuardrailResult(passed=True, guard_name="inspection")


def default_guards(config: Config) -> list[Guard]:
    """Build the default guard list for a standard agent."""
    return [
        DuplicateToolGuard(),
        ConsecutiveToolGuard(config),
        ToolClaimGuard(),
        VerificationGuard(),
        StallGuard(),
        InspectionGuard(),
    ]
