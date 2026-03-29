"""Tests for GuardrailEngine and all 6 built-in guards."""
from __future__ import annotations

import pytest

from src.agent.guardrails import (
    ConsecutiveToolGuard,
    DuplicateToolGuard,
    GuardrailEngine,
    GuardrailResult,
    InspectionGuard,
    PythonStructuralEditGuard,
    ReadBeforeEditGuard,
    StallGuard,
    ToolClaimGuard,
    VerificationGuard,
)
from src.agent.state import ToolResultRecord, TurnState
from src.config import Config
from src.tools.runtime import ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**kwargs) -> TurnState:
    state = TurnState(turn_id="test-turn")
    for k, v in kwargs.items():
        setattr(state, k, v)
    return state


def make_call(name: str, args: dict | None = None) -> ToolCall:
    return ToolCall(id="tc-1", name=name, arguments=args or {})


# ---------------------------------------------------------------------------
# GuardrailEngine
# ---------------------------------------------------------------------------

class AlwaysPassGuard:
    name = "always_pass"

    def check(self, state, response, tool_calls):
        return GuardrailResult(passed=True, guard_name=self.name)


class NudgeGuard:
    name = "nudge"

    def check(self, state, response, tool_calls):
        return GuardrailResult(passed=False, nudge="nudge msg", guard_name=self.name)


class HardStopGuard:
    name = "hard_stop"

    def check(self, state, response, tool_calls):
        return GuardrailResult(
            passed=False, nudge="stop msg", hard_stop=True, guard_name=self.name
        )


def test_engine_all_pass():
    engine = GuardrailEngine([AlwaysPassGuard(), AlwaysPassGuard()])
    result = engine.run(make_state(), "response", [])
    assert result.passed


def test_engine_one_nudge():
    engine = GuardrailEngine([AlwaysPassGuard(), NudgeGuard()])
    result = engine.run(make_state(), "response", [])
    assert not result.passed
    assert result.nudge == "nudge msg"
    assert not result.hard_stop


def test_engine_hard_stop_beats_nudge_even_when_later():
    # NudgeGuard is first, HardStopGuard is second — hard_stop must win
    engine = GuardrailEngine([NudgeGuard(), HardStopGuard()])
    result = engine.run(make_state(), "response", [])
    assert result.hard_stop
    assert result.guard_name == "hard_stop"


# ---------------------------------------------------------------------------
# DuplicateToolGuard
# ---------------------------------------------------------------------------

def test_duplicate_first_call_passes():
    guard = DuplicateToolGuard()
    state = make_state()
    call = make_call("read_file", {"path": "/tmp/foo"})
    result = guard.check(state, "", [call])
    assert result.passed


def test_duplicate_second_call_nudges():
    guard = DuplicateToolGuard()
    state = make_state()
    call = make_call("read_file", {"path": "/tmp/foo"})
    # Simulate one prior occurrence
    sig = state.tool_signature(call)
    state.seen_signatures[sig] = 2
    result = guard.check(state, "", [call])
    assert not result.passed
    assert not result.hard_stop
    assert "read_file" in result.nudge


def test_duplicate_third_call_hard_stops():
    guard = DuplicateToolGuard()
    state = make_state()
    call = make_call("read_file", {"path": "/tmp/foo"})
    sig = state.tool_signature(call)
    state.seen_signatures[sig] = 3
    result = guard.check(state, "", [call])
    assert not result.passed
    assert result.hard_stop


# ---------------------------------------------------------------------------
# ConsecutiveToolGuard
# ---------------------------------------------------------------------------

def test_consecutive_under_limit_passes():
    config = Config(max_consecutive_tool_calls=5)
    guard = ConsecutiveToolGuard(config)
    state = make_state(consecutive_tool_count=3)
    result = guard.check(state, "", [make_call("some_tool")])
    assert result.passed


def test_consecutive_at_limit_with_tool_calls_nudges():
    config = Config(max_consecutive_tool_calls=5)
    guard = ConsecutiveToolGuard(config)
    state = make_state(consecutive_tool_count=6)
    result = guard.check(state, "", [make_call("some_tool")])
    assert not result.passed
    assert not result.hard_stop


def test_consecutive_at_limit_no_tool_calls_passes():
    config = Config(max_consecutive_tool_calls=5)
    guard = ConsecutiveToolGuard(config)
    state = make_state(consecutive_tool_count=6)
    result = guard.check(state, "Here is the answer.", [])
    assert result.passed


# ---------------------------------------------------------------------------
# ToolClaimGuard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "I ran the tests and they passed.",
    "I executed the script already.",
    "I called the API to check the results.",
    "I used the tool to process the file.",
    "I've already run the linter.",
    "I've just executed the command.",
    "Done editing the file.",
    "Completed the test run.",
    "I've written the output file.",
])
def test_tool_claim_detects_claim(text: str):
    guard = ToolClaimGuard()
    state = make_state()
    result = guard.check(state, text, [])
    assert not result.passed
    assert "tool" in result.nudge.lower()


def test_tool_claim_no_claim_passes():
    guard = ToolClaimGuard()
    state = make_state()
    result = guard.check(state, "Here is a summary of the results.", [])
    assert result.passed


def test_tool_claim_with_tool_calls_always_passes():
    guard = ToolClaimGuard()
    state = make_state()
    result = guard.check(state, "I ran the tests.", [make_call("run_pytest")])
    assert result.passed


# ---------------------------------------------------------------------------
# VerificationGuard
# ---------------------------------------------------------------------------

def test_verification_no_files_written_passes():
    guard = VerificationGuard()
    state = make_state()
    result = guard.check(state, "Done.", [])
    assert result.passed


def test_verification_files_written_no_verification_nudges():
    guard = VerificationGuard()
    state = make_state(files_written=["/tmp/foo.py"])
    result = guard.check(state, "Done.", [])
    assert not result.passed
    assert "verified" in result.nudge.lower() or "test" in result.nudge.lower()


def test_verification_files_written_verification_tool_after_write_passes():
    guard = VerificationGuard()
    state = make_state(files_written=["/tmp/foo.py"])
    # Simulate write call followed by pytest call in tool_calls
    write_call = make_call("write_file", {"path": "/tmp/foo.py"})
    verify_call = make_call("run_pytest", {"path": "test/"})
    state.tool_calls = [write_call, verify_call]
    result = guard.check(state, "Done.", [])
    assert result.passed


def test_verification_files_written_only_write_in_history_nudges():
    guard = VerificationGuard()
    state = make_state(files_written=["/tmp/foo.py"])
    write_call = make_call("write_file", {"path": "/tmp/foo.py"})
    state.tool_calls = [write_call]
    result = guard.check(state, "Done.", [])
    assert not result.passed


def test_verification_with_current_tool_calls_passes():
    guard = VerificationGuard()
    state = make_state(files_written=["/tmp/foo.py"])
    result = guard.check(state, "", [make_call("run_ruff")])
    assert result.passed


def test_verification_passes_with_structured_verifier_result_after_write():
    guard = VerificationGuard()
    state = make_state(files_written=["/tmp/foo.py"])
    state.tool_results = [
        ToolResultRecord(
            call_id="c1",
            tool_name="write_file",
            status="ok",
            writes_files=True,
        ),
        ToolResultRecord(
            call_id="c2",
            tool_name="quality_gate",
            status="ok",
            verifier=True,
            has_content=True,
        ),
    ]
    result = guard.check(state, "Done.", [])
    assert result.passed


def test_read_before_edit_guard_nudges_when_editing_unread_file():
    guard = ReadBeforeEditGuard()
    state = make_state(files_read=[])
    result = guard.check(state, "", [make_call("edit_file", {"path": "src/app.py"})])
    assert not result.passed
    assert "Read the file" in result.nudge


def test_read_before_edit_guard_passes_after_file_was_read():
    guard = ReadBeforeEditGuard()
    state = make_state(files_read=["src/app.py"])
    result = guard.check(state, "", [make_call("edit_file", {"path": "src/app.py"})])
    assert result.passed


def test_python_structural_edit_guard_prefers_libcst_tools_for_python():
    guard = PythonStructuralEditGuard(Config())
    state = make_state(available_tool_names={"edit_file_libcst", "replace_function_body"})
    result = guard.check(state, "", [make_call("edit_file", {"path": "src/app.py"})])
    assert not result.passed
    assert "edit_file_libcst" in result.nudge


def test_python_structural_edit_guard_passes_for_non_python_paths():
    guard = PythonStructuralEditGuard(Config())
    state = make_state(available_tool_names={"edit_file_libcst", "replace_function_body"})
    result = guard.check(state, "", [make_call("edit_file", {"path": "README.md"})])
    assert result.passed


# ---------------------------------------------------------------------------
# StallGuard
# ---------------------------------------------------------------------------

def test_stall_under_limit_passes():
    # StallGuard triggers on total nudges across all guards; 4 < 5 (default max)
    guard = StallGuard()
    state = make_state(guardrail_nudges={"tool_claim": 2, "hallucination": 2})
    result = guard.check(state, "Same response.", [])
    assert result.passed


def test_stall_at_limit_no_tool_hard_stops():
    # 5 total nudges == max_total_nudges (5) → hard-stop
    guard = StallGuard(max_total_nudges=5)
    state = make_state(guardrail_nudges={"tool_claim": 3, "hallucination": 2})
    result = guard.check(state, "Same response again.", [])
    assert not result.passed
    assert result.hard_stop


def test_stall_at_limit_with_tool_passes():
    # Tools bypass stall guard
    guard = StallGuard(max_total_nudges=5)
    state = make_state(guardrail_nudges={"tool_claim": 3, "hallucination": 2})
    result = guard.check(state, "Same response again.", [make_call("some_tool")])
    assert result.passed


def test_stall_no_nudge_count_passes():
    guard = StallGuard()
    state = make_state()
    result = guard.check(state, "Some response.", [])
    assert result.passed


# ---------------------------------------------------------------------------
# InspectionGuard
# ---------------------------------------------------------------------------

def test_inspection_always_passes():
    guard = InspectionGuard()
    state = make_state()
    result = guard.check(state, "Anything.", [make_call("some_tool")])
    assert result.passed
    assert result.guard_name == "inspection"


def test_inspection_passes_with_no_tools():
    guard = InspectionGuard()
    state = make_state()
    result = guard.check(state, "I ran the tests already.", [])
    assert result.passed


def test_inspection_structured_error_blocks_false_success_claim():
    guard = InspectionGuard()
    state = make_state(
        tool_results=[
            ToolResultRecord(
                call_id="c1",
                tool_name="read_file",
                status="error",
                error="file not found",
            )
        ]
    )
    result = guard.check(state, "Here is the content of the file.", [])
    assert not result.passed


def test_inspection_structured_success_blocks_false_not_found_claim():
    guard = InspectionGuard()
    state = make_state(
        tool_results=[
            ToolResultRecord(
                call_id="c1",
                tool_name="read_file",
                status="ok",
                has_content=True,
                output='{"status":"ok","content":"hello"}',
            )
        ]
    )
    result = guard.check(state, "I couldn't find the file.", [])
    assert not result.passed
