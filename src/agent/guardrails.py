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
    def __init__(self, guards: list[Guard], max_nudges_per_guard: int = 2) -> None:
        self.guards = guards
        self._max_nudges = max_nudges_per_guard

    def run(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult:
        """Run ALL guards. hard_stop > nudge > pass. First in list wins within tier.

        If the same guard has already nudged >= max_nudges_per_guard times this turn,
        its next failure is escalated to a hard-stop to prevent infinite loops.
        """
        results = [g.check(state, response, tool_calls) for g in self.guards]
        hard_stops = [r for r in results if r.hard_stop]
        if hard_stops:
            return hard_stops[0]
        failures = [r for r in results if not r.passed]
        if failures:
            f = failures[0]
            prior = state.guardrail_nudges.get(f.guard_name, 0)
            if prior >= self._max_nudges:
                return GuardrailResult(
                    passed=False,
                    nudge=f.nudge,
                    hard_stop=True,
                    guard_name=f.guard_name,
                )
            return f
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

# Patterns that indicate the model is fabricating tool output instead of calling tools.
_HALLUCINATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bsimulated\s+output\b", re.IGNORECASE),
    re.compile(r"\bhypothetical\s+(?:output|result|response)\b", re.IGNORECASE),
    re.compile(r"\b(?:sample|mock|fake|dummy)\s+output\b", re.IGNORECASE),
    re.compile(
        r"\bi\s+(?:cannot|can't|am\s+unable\s+to)\s+(?:run|execute|call|access|use)"
        r"\s+(?:shell|bash|tool|command)",
        re.IGNORECASE,
    ),
    re.compile(r"\bsince\s+i\s+cannot\s+execute\b", re.IGNORECASE),
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


class HallucinationGuard:
    """Catches responses that contain fabricated tool output instead of real execution."""

    name = "hallucination"

    def check(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult:
        if tool_calls:
            return GuardrailResult(passed=True, guard_name=self.name)
        for pattern in _HALLUCINATION_PATTERNS:
            if pattern.search(response):
                return GuardrailResult(
                    passed=False,
                    nudge=(
                        "You appear to be simulating tool output instead of calling real tools. "
                        "Call the actual tool now — do NOT fabricate output."
                    ),
                    hard_stop=False,
                    guard_name=self.name,
                )
        return GuardrailResult(passed=True, guard_name=self.name)


_VERIFICATION_NAMES = {"test", "pytest", "ruff", "lint", "check", "verify", "mypy"}

_EDIT_TOOL_NAMES = {
    "edit_file",
    "apply_edit_block",
    "smart_edit",
    "edit_file_libcst",
    "replace_function_body",
    "replace_docstring",
    "replace_decorators",
    "replace_argument",
    "insert_after_function",
    "delete_function",
}

_STRUCTURAL_PYTHON_TOOLS = {
    "edit_file_libcst",
    "replace_function_body",
    "replace_docstring",
    "replace_decorators",
    "replace_argument",
    "insert_after_function",
    "delete_function",
    "find_function_by_name",
    "find_class_by_name",
}


def _call_path(call: ToolCall) -> str:
    args = dict(call.arguments or {})
    return str(
        args.get("path") or args.get("file_path") or args.get("filename") or ""
    ).strip()


def _is_python_path(path: str) -> bool:
    return str(path or "").strip().lower().endswith(".py")


def _tool_name_is_verifier(name: str) -> bool:
    lower = str(name or "").lower()
    return any(part in lower for part in _VERIFICATION_NAMES)


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
            if any(_tool_name_is_verifier(call.name) for call in tool_calls):
                return GuardrailResult(passed=True, guard_name=self.name)
            return GuardrailResult(passed=False, guard_name=self.name, nudge=(
                "You've modified files but haven't verified. "
                "Please run tests or a linter."
            ))

        # Prefer structured execution results when available.
        last_write_result = state.last_write_result_index()
        if last_write_result >= 0:
            for result in state.tool_results[last_write_result + 1 :]:
                if result.verifier:
                    return GuardrailResult(passed=True, guard_name=self.name)
        else:
            last_write = state.last_write_index()
            calls_after = state.tool_calls[last_write + 1 :]
            for call in calls_after:
                if _tool_name_is_verifier(call.name):
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


class ReadBeforeEditGuard:
    name = "read_before_edit"

    def check(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult:
        for call in tool_calls:
            if call.name not in _EDIT_TOOL_NAMES:
                continue
            path = _call_path(call)
            if not path:
                continue
            if path in state.files_read:
                continue
            return GuardrailResult(
                passed=False,
                nudge=(
                    f"You are editing `{path}` with `{call.name}` before inspecting it. "
                    "Read the file or target symbol first, then apply the edit."
                ),
                hard_stop=False,
                guard_name=self.name,
            )
        return GuardrailResult(passed=True, guard_name=self.name)


class PythonStructuralEditGuard:
    name = "python_structural_edit"

    def __init__(self, config: Config) -> None:
        self._enabled = bool(
            getattr(config, "python_structural_editing_preference", True)
        )

    def check(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult:
        if not self._enabled or not tool_calls:
            return GuardrailResult(passed=True, guard_name=self.name)
        if not state.available_tool_names.intersection(_STRUCTURAL_PYTHON_TOOLS):
            return GuardrailResult(passed=True, guard_name=self.name)

        for call in tool_calls:
            if call.name != "edit_file":
                continue
            path = _call_path(call)
            if not _is_python_path(path):
                continue
            return GuardrailResult(
                passed=False,
                nudge=(
                    f"For Python file `{path}`, prefer structural tools first. "
                    "Use `find_function_by_name` / `find_class_by_name` to inspect symbols, "
                    "then prefer `replace_function_body`, `replace_docstring`, or `edit_file_libcst` "
                    "instead of raw `edit_file` when possible."
                ),
                hard_stop=False,
                guard_name=self.name,
            )
        return GuardrailResult(passed=True, guard_name=self.name)


class StallGuard:
    name = "stall"

    def __init__(self, max_total_nudges: int = 5) -> None:
        self._max = max_total_nudges

    def check(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult:
        if tool_calls:
            return GuardrailResult(passed=True, guard_name=self.name)
        total_nudges = sum(state.guardrail_nudges.values())
        if total_nudges >= self._max:
            return GuardrailResult(
                passed=False,
                nudge="Agent is stuck in a guardrail loop. Hard-stopping turn.",
                hard_stop=True,
                guard_name=self.name,
            )
        return GuardrailResult(passed=True, guard_name=self.name)


_TOOL_ERROR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'"status"\s*:\s*"error"', re.IGNORECASE),
    re.compile(r'\bno such file or directory\b', re.IGNORECASE),
    re.compile(r'\bfile not found\b', re.IGNORECASE),
    re.compile(r'\bno matches found\b', re.IGNORECASE),
    re.compile(r'"count"\s*:\s*0\b'),
    re.compile(r'"matches_found"\s*:\s*0\b'),
    re.compile(r'\b0 matches\b', re.IGNORECASE),
    re.compile(r'\berror\b.{0,40}\bnot found\b', re.IGNORECASE | re.DOTALL),
]

_SUCCESS_CLAIM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'\bthe file contains\b', re.IGNORECASE),
    re.compile(r'\bfound\s+\d+\b', re.IGNORECASE),
    re.compile(r'\bsuccessfully\s+(?:read|loaded|found|retrieved)\b', re.IGNORECASE),
    re.compile(r'\bthe (?:result|output) shows\b', re.IGNORECASE),
    re.compile(r'\bhere (?:is|are) the (?:content|results|output)\b', re.IGNORECASE),
    re.compile(r'\bmatches were found\b', re.IGNORECASE),
]

_TOOL_SUCCESS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'"status"\s*:\s*"ok"', re.IGNORECASE),
    re.compile(r'"content"\s*:\s*"[^"]{10}'),
]

_NOT_FOUND_CLAIM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bthe file (?:does not|doesn't) exist\b", re.IGNORECASE),
    re.compile(r"\bcouldn't find\b", re.IGNORECASE),
    re.compile(r"\bno (?:file|result|match|content)\b.{0,40}\bfound\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bdoes not exist\b", re.IGNORECASE),
]


class InspectionGuard:
    """Heuristic check: detect when the agent's response contradicts the last tool output.

    Two contradiction modes:
    1. Tool returned an error / empty result → response claims success/content.
    2. Tool returned content/success → response claims nothing was found.

    Fires only on no-tool responses (candidate final answers) so it doesn't
    interrupt mid-chain tool calls.
    """

    name = "inspection"

    def check(
        self,
        state: TurnState,
        response: str,
        tool_calls: list[ToolCall],
    ) -> GuardrailResult:
        # Only check when the agent is about to give a final answer
        if tool_calls:
            return GuardrailResult(passed=True, guard_name=self.name)
        if not response:
            return GuardrailResult(passed=True, guard_name=self.name)

        if state.tool_results:
            last_result = state.tool_results[-1]
            tool_has_error = last_result.status in {"error", "failed", "fail"}
            tool_has_success = last_result.has_content and not tool_has_error
        else:
            last_output = state.last_tool_output
            if not last_output:
                return GuardrailResult(passed=True, guard_name=self.name)
            tool_has_error = any(p.search(last_output) for p in _TOOL_ERROR_PATTERNS)
            tool_has_success = any(p.search(last_output) for p in _TOOL_SUCCESS_PATTERNS)

        # Mode 1: tool reported failure but response claims success
        if tool_has_error and not tool_has_success:
            if any(p.search(response) for p in _SUCCESS_CLAIM_PATTERNS):
                return GuardrailResult(
                    passed=False,
                    nudge=(
                        "Your response claims to have found/read content, but the last "
                        "tool result indicates an error or no results. "
                        "Base your answer on what the tools actually returned."
                    ),
                    hard_stop=False,
                    guard_name=self.name,
                )

        # Mode 2: tool returned content but response claims nothing was found
        if tool_has_success and not tool_has_error:
            if any(p.search(response) for p in _NOT_FOUND_CLAIM_PATTERNS):
                return GuardrailResult(
                    passed=False,
                    nudge=(
                        "Your response says content was not found, but the last tool "
                        "result returned actual content. "
                        "Re-read the tool output and answer based on what it returned."
                    ),
                    hard_stop=False,
                    guard_name=self.name,
                )

        return GuardrailResult(passed=True, guard_name=self.name)


def default_guards(config: Config) -> list[Guard]:
    """Build the default guard list for a standard agent."""
    guards: list[Guard] = [
        DuplicateToolGuard(),
        ConsecutiveToolGuard(config),
        HallucinationGuard(),
    ]
    if getattr(config, "tool_claim_guard_enabled", True):
        guards.append(ToolClaimGuard())
    if getattr(config, "workflow_guard_enabled", True):
        guards.append(ReadBeforeEditGuard())
        guards.append(PythonStructuralEditGuard(config))
    guards.append(VerificationGuard())
    guards.append(StallGuard(max_total_nudges=getattr(config, "max_guardrail_nudges", 5)))
    if getattr(config, "inspection_result_guard_enabled", True):
        guards.append(InspectionGuard())
    return guards
