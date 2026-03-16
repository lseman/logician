import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from src import AgentResponse, Message, MessageRole, ToolParameter, create_agent


class _FakeLLM:
    def __init__(self, response: str = "OK") -> None:
        self.response = response
        self.calls = 0

    def generate(self, messages, temperature, max_tokens, stream=False, on_token=None):
        del messages, temperature, max_tokens, stream, on_token
        self.calls += 1
        return self.response


class _StreamingFakeLLM:
    def __init__(self, response: str, chunks: list[str] | None = None) -> None:
        self.response = response
        self.chunks = chunks or [response]
        self.calls = 0

    def generate(self, messages, temperature, max_tokens, stream=False, on_token=None):
        del messages, temperature, max_tokens
        self.calls += 1
        if stream and on_token is not None:
            for chunk in self.chunks:
                on_token(chunk)
        return self.response


class _SequenceLLM:
    def __init__(self, responses: list[str], chunks: list[list[str]] | None = None) -> None:
        self.responses = responses
        self.chunks = chunks or [None] * len(responses)
        self.calls = 0

    def generate(self, messages, temperature, max_tokens, stream=False, on_token=None):
        del messages, temperature, max_tokens
        idx = min(self.calls, len(self.responses) - 1)
        response = self.responses[idx]
        chunks = self.chunks[idx] if idx < len(self.chunks) else None
        self.calls += 1
        if stream and on_token is not None and chunks:
            for chunk in chunks:
                on_token(chunk)
        return response


class AgentRuntimePublicTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "agent_sessions.db"
        self.vector_path = Path(self._tmpdir.name) / "message_history.vector"
        self.agent = create_agent(
            llm_url="http://localhost:8080",
            db_path=str(self.db_path),
            config_overrides={
                "rag_enabled": False,
                "vector_path": str(self.vector_path),
                "max_iterations": 2,
            },
        )
        self.agent.llm = _FakeLLM("Hello from test")

    def tearDown(self) -> None:
        db = getattr(self.agent.memory, "_db", None)
        if db is not None:
            db.close()
        self._tmpdir.cleanup()

    def _make_agent(self):
        agent = create_agent(
            llm_url="http://localhost:8080",
            db_path=str(self.db_path),
            config_overrides={
                "rag_enabled": False,
                "vector_path": str(self.vector_path),
                "max_iterations": 2,
            },
        )
        agent.llm = _FakeLLM("Hello from test")
        return agent

    def _load_dummy_context(self, agent=None) -> None:
        target = agent or self.agent
        target.ctx.data = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=6, freq="D"),
                "value": [1.0, 2.0, 3.0, 2.5, 3.5, 4.0],
            }
        )
        target.ctx.original_data = target.ctx.data.copy()
        target.ctx.data_name = "demo.csv"
        target.ctx.freq_cache = "D"
        target.ctx.anomaly_store["value"] = [2]
        target.ctx.nf_best_model = "NHITS"

    def test_run_returns_agent_response_and_streams_final_answer(self) -> None:
        streamed: list[str] = []

        response = self.agent.run(
            "Say hello without tools.",
            session_id="run_case",
            stream_callback=streamed.append,
        )

        self.assertIsInstance(response, AgentResponse)
        self.assertEqual(response.final_response, "Hello from test")
        self.assertEqual(streamed, ["Hello from test"])
        self.assertEqual(response.messages[-1].role, MessageRole.ASSISTANT)
        self.assertEqual(response.messages[-1].content, "Hello from test")

    def test_stream_callback_receives_live_tokens_without_duplicate_final(self) -> None:
        self.agent.llm = _StreamingFakeLLM(
            "Hello from streaming test",
            chunks=["Hello ", "from ", "streaming test"],
        )
        streamed: list[str] = []

        response = self.agent.run(
            "hello there",
            session_id="streaming_tokens_case",
            stream_callback=streamed.append,
        )

        self.assertEqual(response.final_response, "Hello from streaming test")
        self.assertEqual(streamed, ["Hello ", "from ", "streaming test"])

    def test_thinking_callback_streams_before_execution_response(self) -> None:
        # pre_turn_thinking only fires for execution/design intents; social/informational
        # use the fast path which skips it.  Use an execution query and enable the flag.
        self.agent.config.pre_turn_thinking = True
        self.agent.llm = _SequenceLLM(
            ["Plan first", "Hello after thinking"],
            chunks=[["Plan ", "first"], ["Hello ", "after ", "thinking"]],
        )
        thinking_streamed: list[str] = []
        assistant_streamed: list[str] = []

        response = self.agent.run(
            "list the files in this directory",
            session_id="thinking_stream_case",
            stream_callback=assistant_streamed.append,
            thinking_callback=thinking_streamed.append,
        )

        self.assertEqual(response.final_response, "Hello after thinking")
        self.assertEqual(thinking_streamed, ["Plan ", "first"])
        self.assertEqual(assistant_streamed, ["Hello ", "after ", "thinking"])

    def test_tool_callback_reports_live_start_and_end_events(self) -> None:
        # Two-step sequence: tool call → final answer (pre_turn_thinking is off by default).
        self.agent.llm = _SequenceLLM(
            [
                json.dumps({"name": "think", "arguments": {"thought": "inspect runtime"}}),
                "Tool completed",
            ]
        )
        tool_events: list[tuple[str, dict, dict]] = []

        response = self.agent.run(
            "inspect the runtime and report back",
            session_id="tool_callback_case",
            tool_callback=lambda name, args, meta=None: tool_events.append(
                (name, args, meta or {})
            ),
        )

        self.assertEqual(response.final_response, "Tool completed")
        self.assertEqual([event[2].get("stage") for event in tool_events], ["start", "end"])
        self.assertEqual(tool_events[0][0], "think")
        self.assertEqual(tool_events[0][1], {"thought": "inspect runtime"})
        self.assertEqual(tool_events[0][2].get("sequence"), 1)
        self.assertEqual(tool_events[1][2].get("status"), "ok")

    def test_fresh_session_clears_runtime_context_before_running(self) -> None:
        self._load_dummy_context()

        response = self.agent.chat(
            "Say hello without tools.",
            session_id="fresh_case",
            fresh_session=True,
        )

        self.assertEqual(response, "Hello from test")
        self.assertFalse(self.agent.ctx.loaded)
        self.assertEqual(self.agent.ctx.anomaly_store, {})
        self.assertIsNone(self.agent.ctx.nf_best_model)

    def test_reset_runtime_state_clears_context_without_deleting_history(self) -> None:
        self._load_dummy_context()
        self.agent.current_session_id = "session_alpha"
        self.agent.memory.save_message(
            "session_alpha",
            Message(role=MessageRole.USER, content="persist me"),
        )
        self.agent._persist_runtime_state("session_alpha")

        self.agent.reset_runtime_state()

        self.assertFalse(self.agent.ctx.loaded)
        self.assertEqual(self.agent.ctx.data_name, "")
        self.assertEqual(self.agent.ctx.anomaly_store, {})
        self.assertIsNone(self.agent.current_session_id)
        self.assertIn("session_alpha", {sid for sid, _ in self.agent.list_sessions()})

    def test_runtime_state_restores_across_agent_instances(self) -> None:
        sid = "resume_case"
        self._load_dummy_context()
        self.agent.current_session_id = sid
        self.agent._persist_runtime_state(sid)

        other = self._make_agent()
        try:
            response = other.chat("Say hello without tools.", session_id=sid)

            self.assertEqual(response, "Hello from test")
            self.assertTrue(other.ctx.loaded)
            self.assertEqual(other.ctx.data_name, "demo.csv")
            self.assertEqual(other.ctx.freq_cache, "D")
            self.assertEqual(other.ctx.anomaly_store, {"value": [2]})
            self.assertEqual(other.ctx.nf_best_model, "NHITS")
        finally:
            db = getattr(other.memory, "_db", None)
            if db is not None:
                db.close()

    def test_switching_sessions_restores_each_runtime_state(self) -> None:
        self._load_dummy_context()
        self.agent.current_session_id = "alpha"
        self.agent._persist_runtime_state("alpha")

        self.agent.ctx.data = pd.DataFrame(
            {
                "date": pd.date_range("2025-03-01", periods=2, freq="D"),
                "value": [20.0, 21.0],
            }
        )
        self.agent.ctx.original_data = self.agent.ctx.data.copy()
        self.agent.ctx.data_name = "beta.csv"
        self.agent.ctx.freq_cache = "D"
        self.agent.current_session_id = "beta"
        self.agent._persist_runtime_state("beta")

        self.agent.chat("Say hello", session_id="alpha")
        self.assertEqual(self.agent.ctx.data_name, "demo.csv")
        self.agent.chat("Say hello", session_id="beta")
        self.assertEqual(self.agent.ctx.data_name, "beta.csv")

    def test_describe_runtime_context_reports_history_budget(self) -> None:
        sid = "history_budget_case"
        self._load_dummy_context()
        self.agent.current_session_id = sid
        self.agent._persist_runtime_state(sid)
        for i in range(24):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            self.agent.memory.save_message(
                sid,
                Message(role=role, content=f"message {i}"),
            )

        info = self.agent.describe_runtime_context(session_id=sid)

        self.assertEqual(info["session_id"], sid)
        self.assertEqual(info["persisted_messages"], 24)
        self.assertEqual(info["loaded_message_budget"], 18)
        self.assertTrue(info["history_over_budget"])
        self.assertTrue(info["runtime"]["loaded"])
        self.assertEqual(info["runtime"]["data_name"], "demo.csv")
        self.assertEqual(info["runtime"]["forecast_model"], "NHITS")

    def test_compact_session_inserts_summary_and_keeps_latest_messages(self) -> None:
        sid = "compact_case"
        seeded_messages = [
            Message(role=MessageRole.USER, content="Load demo.csv"),
            Message(
                role=MessageRole.TOOL,
                name="load_csv_data",
                tool_call_id="tool_1",
                content='{"status":"ok","rows":6}',
            ),
            Message(role=MessageRole.ASSISTANT, content="Loaded demo.csv into memory."),
            Message(role=MessageRole.USER, content="Check anomalies"),
            Message(
                role=MessageRole.TOOL,
                name="detect_anomalies",
                tool_call_id="tool_2",
                content='{"status":"ok","points":[2]}',
            ),
            Message(role=MessageRole.ASSISTANT, content="Found one anomaly candidate."),
            Message(role=MessageRole.USER, content="Forecast the next 7 days"),
            Message(role=MessageRole.ASSISTANT, content="Forecast ready."),
        ]
        for message in seeded_messages:
            self.agent.memory.save_message(sid, message)

        result = self.agent.compact_session(session_id=sid, keep_last_messages=3)
        compacted_messages = self.agent.memory.get_session_messages(sid)

        self.assertEqual(result["status"], "compacted")
        self.assertEqual(result["old_message_count"], len(seeded_messages))
        self.assertEqual(result["new_message_count"], 4)
        self.assertEqual(len(compacted_messages), 4)
        self.assertEqual(compacted_messages[0].role, MessageRole.SYSTEM)
        self.assertIn("/compact", compacted_messages[0].content)
        self.assertEqual(compacted_messages[1:], seeded_messages[-3:])

    def test_run_tool_direct_can_persist_history(self) -> None:
        def ping_tool(text: str) -> str:
            return json.dumps({"status": "ok", "echo": text})

        self.agent.add_tool(
            name="ping_tool",
            description="Echo test tool.",
            function=ping_tool,
            parameters=[ToolParameter("text", "string", "Text to echo")],
        )

        result = self.agent.run_tool_direct(
            "ping_tool",
            {"text": "hello"},
            session_id="direct_tool_case",
            persist_to_history=True,
        )
        messages = self.agent.memory.get_session_messages("direct_tool_case")

        self.assertIn('"echo": "hello"', result)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, MessageRole.USER)
        self.assertEqual(messages[1].role, MessageRole.TOOL)
        self.assertEqual(messages[1].name, "ping_tool")

    def test_mcp_servers_load_only_once_across_runs(self) -> None:
        agent = self._make_agent()
        try:
            calls = 0

            def _fake_load():
                nonlocal calls
                calls += 1

            agent._load_mcp_servers = _fake_load  # type: ignore[method-assign]

            agent.chat("Say hello.", session_id="mcp_once_a")
            agent.chat("Say hello again.", session_id="mcp_once_b")

            self.assertEqual(calls, 1)
            self.assertTrue(agent._mcp_loaded)
        finally:
            db = getattr(agent.memory, "_db", None)
            if db is not None:
                db.close()


if __name__ == "__main__":
    unittest.main()
