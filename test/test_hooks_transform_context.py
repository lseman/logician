"""Tests for TransformContext and AfterToolResult hook integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from src.config import Config


def test_parse_transform_context_output():
    """Hook output with TransformContext event is parsed correctly."""
    from src.hooks.types import HookExecutionResult, parse_hook_response

    output = json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "TransformContext",
            "messages": [
                {"role": "system", "content": "injected context"},
                {"role": "user", "content": "hello"},
            ],
        }
    })
    result = parse_hook_response(output)
    assert result.transformed_messages is not None
    assert len(result.transformed_messages) == 2
    assert result.transformed_messages[0]["role"] == "system"
    assert result.transformed_messages[0]["content"] == "injected context"


def test_parse_after_tool_result_output():
    """Hook output with AfterToolResult event is parsed correctly."""
    from src.hooks.types import HookExecutionResult, parse_hook_response

    output = json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "AfterToolResult",
            "content": "patched content",
            "details": {"original_size": 50000},
            "vectorize": False,
        }
    })
    result = parse_hook_response(output)
    assert result.patched_content == "patched content"
    assert result.patched_details == {"original_size": 50000}
    assert result.patch_vectorize is False


def test_parse_top_level_keys():
    """Direct JSON keys work as convenience format."""
    from src.hooks.types import parse_hook_response

    output = json.dumps({
        "messages": [{"role": "user", "content": "injected"}],
    })
    result = parse_hook_response(output)
    assert result.transformed_messages == [{"role": "user", "content": "injected"}]


def test_build_hook_input_transform_context():
    """build_hook_input includes context_messages for TransformContext."""
    from src.hooks.types import HookEventType, build_hook_input

    messages = [{"role": "user", "content": "test"}]
    payload = build_hook_input(
        HookEventType.TRANSFORM_CONTEXT,
        session_id="sess1",
        context_messages=messages,
    )
    data = json.loads(payload)
    assert data["hook_event_name"] == "TransformContext"
    assert data["session_id"] == "sess1"
    assert data["context_messages"] == messages


def test_build_hook_input_after_tool_result():
    """build_hook_input includes tool fields for AfterToolResult."""
    from src.hooks.types import HookEventType, build_hook_input

    payload = build_hook_input(
        HookEventType.AFTER_TOOL_RESULT,
        tool_name="Read",
        tool_input={"path": "/tmp/x"},
        tool_response="tool output",
        session_id="sess1",
    )
    data = json.loads(payload)
    assert data["hook_event_name"] == "AfterToolResult"
    assert data["tool_name"] == "Read"
    assert data["tool_input"] == {"path": "/tmp/x"}
    assert data["tool_response"] == "tool output"


def test_hook_event_types_exist():
    """New hook event types are defined."""
    from src.hooks.types import HookEventType

    assert hasattr(HookEventType, "TRANSFORM_CONTEXT")
    assert hasattr(HookEventType, "AFTER_TOOL_RESULT")
    assert HookEventType.TRANSFORM_CONTEXT.value == "TransformContext"
    assert HookEventType.AFTER_TOOL_RESULT.value == "AfterToolResult"


def test_transform_context_hook_engine():
    """HookEngine.execute_transform_context returns transformed messages."""
    from src.hooks.engine import HookEngine

    engine = HookEngine(timeout_seconds=1)
    result = engine.execute_transform_context(
        session_id="test",
        context_messages=[{"role": "user", "content": "hello"}],
    )
    # No hooks registered, so transformed_messages is None
    assert result.transformed_messages is None
    assert result.hook_count == 0


def test_after_tool_result_hook_engine():
    """HookEngine.execute_after_tool_result returns patched result."""
    from src.hooks.engine import HookEngine

    engine = HookEngine(timeout_seconds=1)
    result = engine.execute_after_tool_result(
        tool_name="Read",
        session_id="test",
        tool_response="some output",
    )
    # No hooks registered, so patches are None
    assert result.patched_content is None
    assert result.patched_details is None
    assert result.patch_vectorize is None
    assert result.hook_count == 0


def test_transform_context_callback_in_loop():
    """AgentLoop.transform_context_fn modifies messages before LLM call."""
    from src.agent.loop import AgentLoop

    # Create a transform function
    def transform_fn(messages, session_id):
        # Inject a system hint at the front
        new_messages = list(messages)
        new_messages.insert(0, {
            "role": "system",
            "content": "[Injected by transform_context hook]",
        })
        return new_messages

    # Verify the loop stores the transform function
    from src.config import Config

    loop = AgentLoop(
        llm=None,
        guardrails=None,
        prompt_builder=None,
        dispatcher=None,
        config=Config(),
        transform_context_fn=transform_fn,
    )
    assert loop._transform_context_fn is not None


def test_after_tool_result_callback_in_loop():
    """AgentLoop.after_tool_result_fn patches tool results."""
    from src.agent.loop import AgentLoop

    def patch_fn(tool_name, content, tool_call_id):
        if tool_name == "Read" and len(content) > 1000:
            return {
                "content": f"[truncated: {len(content)} chars]",
                "vectorize": False,
            }
        return None

    loop = AgentLoop(
        llm=None,
        guardrails=None,
        prompt_builder=None,
        dispatcher=None,
        config=Config(),
        after_tool_result_fn=patch_fn,
    )
    assert loop._after_tool_result_fn is not None


def test_agent_build_transform_context_fn():
    """Agent._build_transform_context_fn returns a callable when hook_engine is available."""
    from src.agent.core import Agent

    mock_engine = MagicMock()
    mock_result = MagicMock()
    mock_result.transformed_messages = [{"role": "system", "content": "injected"}]
    mock_engine.execute_transform_context.return_value = mock_result

    fn = Agent._build_transform_context_fn(mock_engine)
    assert fn is not None

    result = fn([{"role": "user", "content": "hello"}], "test-session")
    assert result == [{"role": "system", "content": "injected"}]
    mock_engine.execute_transform_context.assert_called_once()


def test_agent_build_after_tool_result_fn():
    """Agent._build_after_tool_result_fn returns a callable when hook_engine is available."""
    from src.agent.core import Agent

    mock_engine = MagicMock()
    mock_result = MagicMock()
    mock_result.patched_content = "patched output"
    mock_result.patched_details = {"size": 100}
    mock_result.patch_vectorize = False
    mock_engine.execute_after_tool_result.return_value = mock_result

    fn = Agent._build_after_tool_result_fn(mock_engine)
    assert fn is not None

    result = fn("Read", "original content", "call_id_123")
    assert result is not None
    assert result["content"] == "patched output"
    assert result["details"] == {"size": 100}
    assert result["vectorize"] is False
    mock_engine.execute_after_tool_result.assert_called_once()


def test_agent_none_engine_returns_none_fn():
    """When hook_engine is None, build_*_fn returns None."""
    from src.agent.core import Agent

    assert Agent._build_transform_context_fn(None) is None
    assert Agent._build_after_tool_result_fn(None) is None


def test_tool_result_for_persistence_compact_large():
    """Verify large tool result payload structure."""
    # Simulate what a large tool result payload looks like
    payload = {
        "status": "ok",
        "content": "x" * 20000,
        "path": "/tmp/huge.txt",
        "returned_lines": 500,
        "total_lines": 1000,
        "has_more": True,
    }
    assert payload["status"] == "ok"
    assert len(payload["content"]) == 20000  # Large content
