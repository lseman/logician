"""Tests for AgentLoop — thin ReAct-style main loop."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))


from src.agent.dispatcher import DispatchResult
from src.agent.guardrails import GuardrailResult
from src.agent.loop import AgentLoop, format_tool_results
from src.agent.state import TurnState
from src.config import Config
from src.messages import Message, MessageRole
from src.tools.runtime import ToolCall

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def generate(self, messages, **kwargs) -> str:
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


class FakeDeepSeekLLM(FakeLLM):
    def __init__(self, responses: list[str], reasoning: list[str]) -> None:
        super().__init__(responses)
        self._reasoning = reasoning

    def generate(self, messages, **kwargs) -> str:
        on_reasoning_token = kwargs.get("on_reasoning_token")
        idx = self._idx % len(self._responses)
        if on_reasoning_token is not None:
            on_reasoning_token(self._reasoning[idx])
        return super().generate(messages, **kwargs)


class FakeDispatcher:
    def __init__(self) -> None:
        self.dispatched: list[ToolCall] = []
        self._available_tool_names: set[str] = set()

    async def dispatch(self, calls, state, config=None, tool_callback=None):
        del tool_callback
        self.dispatched.extend(calls)
        state.consecutive_tool_count += len(calls)
        return []

    def available_tool_names(self) -> set[str]:
        return set(self._available_tool_names)

    def prepare_call(self, call):
        return call, None


class FakeGuardrails:
    def run(self, state, response, tool_calls):
        return GuardrailResult(passed=True)


class FakePromptBuilder:
    def build(self, state, config) -> str:
        return "You are a test agent."


def _make_loop(
    responses: list[str],
    *,
    guardrails=None,
    dispatcher=None,
    max_iterations: int = 8,
    use_toon: bool = False,
) -> tuple[AgentLoop, FakeDispatcher]:
    fake_dispatcher = dispatcher or FakeDispatcher()
    config = Config(max_iterations=max_iterations, pre_turn_thinking=False)
    loop = AgentLoop(
        llm=FakeLLM(responses),
        guardrails=guardrails or FakeGuardrails(),
        prompt_builder=FakePromptBuilder(),
        dispatcher=fake_dispatcher,
        config=config,
        use_toon=use_toon,
    )
    return loop, fake_dispatcher


def _user_msg(content: str) -> Message:
    return Message(role=MessageRole.USER, content=content)


def _tool_call_json(name: str, arguments: dict) -> str:
    """Produce a response string that parse_tool_calls will recognise."""
    return json.dumps({"name": name, "arguments": arguments})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_simple_turn_plain_text():
    """FakeLLM returns plain text → TurnResult.final_response is that text."""
    agent_loop, _ = _make_loop(["Hello, I can help you with that."])
    messages = [_user_msg("Hi there")]
    result = asyncio.run(agent_loop.run(messages))
    assert result.final_response == "Hello, I can help you with that."


def test_social_turn_fast_path():
    """Social turn → fast_path called (only 1 LLM call, dispatcher never called)."""
    llm = FakeLLM(["Hey there! How can I help?"])
    fake_dispatcher = FakeDispatcher()
    config = Config(max_iterations=8, pre_turn_thinking=False)
    agent_loop = AgentLoop(
        llm=llm,
        guardrails=FakeGuardrails(),
        prompt_builder=FakePromptBuilder(),
        dispatcher=fake_dispatcher,
        config=config,
    )
    messages = [_user_msg("hello how are you")]
    result = asyncio.run(agent_loop.run(messages))

    assert result.final_response == "Hey there! How can I help?"
    assert fake_dispatcher.dispatched == [], "dispatcher must not be called on social turns"
    assert llm._idx == 1, "only one LLM call for social fast path"


def test_deepseek_reasoning_callback_uses_reasoning_content_channel():
    llm = FakeDeepSeekLLM(["Visible answer"], ["step one\nstep two"])
    fake_dispatcher = FakeDispatcher()
    config = Config(max_iterations=8, pre_turn_thinking=False, chat_template="deepseek")
    agent_loop = AgentLoop(
        llm=llm,
        guardrails=FakeGuardrails(),
        prompt_builder=FakePromptBuilder(),
        dispatcher=fake_dispatcher,
        config=config,
    )
    thinking: list[str] = []

    result = asyncio.run(
        agent_loop.run(
            [_user_msg("hello there")],
            thinking_callback=thinking.append,
        )
    )

    assert result.final_response == "Visible answer"
    assert thinking == ["step one\nstep two"]


def test_social_turn_skips_mcp_loader():
    """Social turn should not pay MCP startup before the fast path."""
    llm = FakeLLM(["Hello"])
    fake_dispatcher = FakeDispatcher()
    config = Config(max_iterations=8, pre_turn_thinking=False)
    mcp_calls = {"count": 0}

    def _load_mcp() -> None:
        mcp_calls["count"] += 1

    agent_loop = AgentLoop(
        llm=llm,
        guardrails=FakeGuardrails(),
        prompt_builder=FakePromptBuilder(),
        dispatcher=fake_dispatcher,
        config=config,
        mcp_loader=_load_mcp,
    )

    result = asyncio.run(agent_loop.run([_user_msg("hello there")]))

    assert result.final_response == "Hello"
    assert mcp_calls["count"] == 0


def test_execution_turn_loads_mcp_before_loop():
    """Execution turns still initialise MCP before entering the tool loop."""
    llm = FakeLLM(["Done"])
    fake_dispatcher = FakeDispatcher()
    config = Config(max_iterations=8, pre_turn_thinking=False)
    mcp_calls = {"count": 0}

    def _load_mcp() -> None:
        mcp_calls["count"] += 1

    agent_loop = AgentLoop(
        llm=llm,
        guardrails=FakeGuardrails(),
        prompt_builder=FakePromptBuilder(),
        dispatcher=fake_dispatcher,
        config=config,
        mcp_loader=_load_mcp,
    )

    result = asyncio.run(agent_loop.run([_user_msg("list the files")]))

    assert result.final_response == "Done"
    assert mcp_calls["count"] == 1


def test_available_tool_names_can_refresh_after_skill_load_dispatch():
    class SkillLoadingDispatcher(FakeDispatcher):
        def __init__(self) -> None:
            super().__init__()
            self._available_tool_names = {"invoke_skill"}

        async def dispatch(self, calls, state, config=None, tool_callback=None):
            del config, tool_callback
            self.dispatched.extend(calls)
            state.consecutive_tool_count += len(calls)
            self._available_tool_names.add("wiki_list")
            return [
                DispatchResult(
                    tool_name="invoke_skill",
                    call_id="call-invoke-skill",
                    output=json.dumps(
                        {
                            "status": "ok",
                            "available_tools": ["wiki_list"],
                            "newly_available_tools": ["wiki_list"],
                        }
                    ),
                )
            ]

    dispatcher = SkillLoadingDispatcher()
    state = TurnState(turn_id="turn-1")
    state.available_tool_names = dispatcher.available_tool_names()

    asyncio.run(
        dispatcher.dispatch(
            [ToolCall(id="call-invoke-skill", name="invoke_skill", arguments={"skill": "wiki"})],
            state,
        )
    )
    state.available_tool_names = dispatcher.available_tool_names()

    assert "invoke_skill" in state.available_tool_names
    assert "wiki_list" in state.available_tool_names


def test_guardrail_nudge_increments_iteration():
    """Guardrails returning nudge on first call then pass → iteration increments."""

    class NudgeThenPassGuardrails:
        def __init__(self):
            self._calls = 0

        def run(self, state, response, tool_calls):
            self._calls += 1
            if self._calls == 1:
                return GuardrailResult(
                    passed=False,
                    nudge="Please try again.",
                    hard_stop=False,
                    guard_name="test_nudge",
                )
            return GuardrailResult(passed=True)

    guardrails = NudgeThenPassGuardrails()
    agent_loop, _ = _make_loop(
        ["first response", "final answer"],
        guardrails=guardrails,
    )
    messages = [_user_msg("do something complex")]
    result = asyncio.run(agent_loop.run(messages))

    assert result.final_response == "final answer"
    assert result.state.iteration >= 1
    assert result.state.guardrail_nudges.get("test_nudge", 0) == 1


def test_guardrail_hard_stop():
    """Guardrail hard_stop → returns immediately with final_response set."""

    class HardStopGuardrails:
        def run(self, state, response, tool_calls):
            return GuardrailResult(
                passed=False,
                hard_stop=True,
                guard_name="hard_stop_test",
            )

    agent_loop, fake_dispatcher = _make_loop(
        ["some response"],
        guardrails=HardStopGuardrails(),
    )
    messages = [_user_msg("do something")]
    result = asyncio.run(agent_loop.run(messages))

    assert result.final_response == "some response"
    assert fake_dispatcher.dispatched == [], "no tools dispatched on hard_stop"


def test_tool_call_then_final_response():
    """FakeLLM emits a tool call JSON first, then plain text → dispatcher called once."""
    tool_response = _tool_call_json("read_file", {"path": "/tmp/foo.txt"})
    final_response = "Here is the answer based on the file."

    agent_loop, fake_dispatcher = _make_loop([tool_response, final_response])
    messages = [_user_msg("read the file and summarize")]
    result = asyncio.run(agent_loop.run(messages))

    assert result.final_response == final_response
    assert len(fake_dispatcher.dispatched) == 1
    assert fake_dispatcher.dispatched[0].name == "read_file"


def test_max_iterations_exits_loop():
    """FakeLLM always returns a tool call → loop exits at max_iterations."""
    tool_response = _tool_call_json("read_file", {"path": "/tmp/x.txt"})

    agent_loop, fake_dispatcher = _make_loop(
        [tool_response],
        max_iterations=3,
    )
    messages = [_user_msg("do an infinite task")]
    result = asyncio.run(agent_loop.run(messages))

    # Loop should have terminated without surfacing raw tool JSON
    assert result.final_response == (
        "I reached the turn iteration limit after executing "
        "`read_file`, before I could write the final answer."
    )
    # Dispatcher called exactly max_iterations times
    assert len(fake_dispatcher.dispatched) == 3


def test_hybrid_direct_answer_tool_call_is_executed():
    """Hybrid direct-answer pass is reparsed if the model emits a tool call anyway."""
    tool_response = _tool_call_json("read_file", {"path": "/tmp/x.txt"})
    final_response = "Verified the file contents."

    agent_loop, fake_dispatcher = _make_loop(
        ["NO_TOOL", tool_response, "NO_TOOL", final_response],
    )
    messages = [_user_msg("write a file and then verify it")]
    result = asyncio.run(agent_loop.run(messages, token_callback=lambda _token: None))

    assert result.final_response == final_response
    assert [call.name for call in fake_dispatcher.dispatched] == ["read_file"]


def test_caller_messages_not_mutated():
    """The caller's messages list must not be mutated by the loop."""
    tool_response = _tool_call_json("read_file", {"path": "/tmp/test.txt"})
    final_response = "Done."

    agent_loop, _ = _make_loop([tool_response, final_response])
    original = [_user_msg("read and summarize")]
    original_len = len(original)

    asyncio.run(agent_loop.run(original))
    assert len(original) == original_len, "caller's messages list was mutated"


def test_format_tool_results_uses_error_when_present():
    """format_tool_results uses error field when set, otherwise output."""
    results = [
        DispatchResult(tool_name="foo", call_id="c1", output="ok_output"),
        DispatchResult(tool_name="bar", call_id="c2", output="", error="something failed"),
    ]
    msgs = format_tool_results(results)
    assert msgs[0].content == "ok_output"
    assert msgs[1].content == "something failed"
    assert msgs[0].tool_call_id == "c1"
    assert msgs[1].tool_call_id == "c2"
    assert all(m.role == MessageRole.TOOL for m in msgs)


def test_with_system_replaces_existing_system_message():
    """_with_system replaces the first SYSTEM message rather than prepending."""
    agent_loop, _ = _make_loop(["response"])
    old_sys = Message(role=MessageRole.SYSTEM, content="old system")
    user = _user_msg("hello")
    result = agent_loop._with_system([old_sys, user], "new system")
    assert result[0].role == MessageRole.SYSTEM
    assert result[0].content == "new system"
    assert result[1] is user
    assert len(result) == 2


def test_with_system_prepends_when_no_system_message():
    """_with_system prepends system message when none exists."""
    agent_loop, _ = _make_loop(["response"])
    user = _user_msg("hello")
    result = agent_loop._with_system([user], "new system")
    assert result[0].role == MessageRole.SYSTEM
    assert result[0].content == "new system"
    assert result[1] is user
    assert len(result) == 2
