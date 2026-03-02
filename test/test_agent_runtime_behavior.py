import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from src import Message, MessageRole, ToolCall, create_agent


class _FakeLLM:
    def __init__(self, response: str = "OK") -> None:
        self.response = response

    def generate(self, messages, temperature, max_tokens, stream=False, on_token=None):
        return self.response


class AgentRuntimeBehaviorTests(unittest.TestCase):
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

    def _load_dummy_context(self) -> None:
        self.agent.ctx.data = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=6, freq="D"),
                "value": [1.0, 2.0, 3.0, 2.5, 3.5, 4.0],
            }
        )
        self.agent.ctx.original_data = self.agent.ctx.data.copy()
        self.agent.ctx.data_name = "demo.csv"
        self.agent.ctx.freq_cache = "D"
        self.agent.ctx.anomaly_store["value"] = [2]
        self.agent.ctx.nf_best_model = "NHITS"

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
        self.assertIsNone(self.agent.ctx.nf_best_model)
        self.assertIsNone(self.agent.current_session_id)
        self.assertIn("session_alpha", {sid for sid, _ in self.agent.list_sessions()})
        restored = self.agent.describe_runtime_context("session_alpha")
        self.assertFalse(restored["runtime"]["loaded"])

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

    def test_follow_up_queries_use_runtime_skill_hints(self) -> None:
        self._load_dummy_context()
        call = ToolCall(
            id="call_1",
            name="load_csv_data",
            arguments={"path": "demo.csv"},
        )
        sig = (call.name, json.dumps(call.arguments, sort_keys=True))

        prompt, selection, routing_query = self.agent._resolve_system_prompt(
            "continue",
            tool_calls=[call],
            tool_result_preview_by_sig={sig: '{"status":"ok","rows":6}'},
        )

        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertIn("Runtime state for skill routing", routing_query)
        self.assertIn("ACTIVE SKILLS FOR THIS REQUEST", prompt)
        self.assertTrue(
            any(
                skill.id in {"analysis", "preprocessing"}
                for skill in selection.selected_skills
            )
        )

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
        self._load_dummy_context()
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
        self.assertIn("demo.csv", compacted_messages[0].content)
        self.assertEqual(compacted_messages[1:], seeded_messages[-3:])

    def test_load_history_keeps_compaction_checkpoint_in_window(self) -> None:
        sid = "compact_window_case"
        for i in range(26):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            self.agent.memory.save_message(
                sid,
                Message(role=role, content=f"turn {i}"),
            )

        self.agent.compact_session(session_id=sid, keep_last_messages=3)

        for i in range(20):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            self.agent.memory.save_message(
                sid,
                Message(role=role, content=f"after compact {i}"),
            )

        history = self.agent.memory.load_history(
            sid,
            message="what happened",
            use_semantic_retrieval=False,
        )

        self.assertEqual(len(history), self.agent.config.history_limit)
        self.assertEqual(history[0].role, MessageRole.SYSTEM)
        self.assertIn("/compact", history[0].content)
        self.assertEqual(history[-1].content, "after compact 19")

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
        self.agent.ctx.data = pd.DataFrame(
            {
                "date": pd.date_range("2025-02-01", periods=3, freq="D"),
                "value": [10.0, 11.0, 12.0],
            }
        )
        self.agent.ctx.data_name = "alpha.csv"
        self.agent.current_session_id = "alpha"
        self.agent._persist_runtime_state("alpha")

        self.agent.ctx.data = pd.DataFrame(
            {
                "date": pd.date_range("2025-03-01", periods=2, freq="D"),
                "value": [20.0, 21.0],
            }
        )
        self.agent.ctx.data_name = "beta.csv"
        self.agent.current_session_id = "beta"
        self.agent._persist_runtime_state("beta")

        self.agent.chat("Say hello", session_id="alpha")
        self.assertEqual(self.agent.ctx.data_name, "alpha.csv")
        self.agent.chat("Say hello", session_id="beta")
        self.assertEqual(self.agent.ctx.data_name, "beta.csv")


if __name__ == "__main__":
    unittest.main()
