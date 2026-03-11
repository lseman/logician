import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class _SequenceFakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate(self, messages, temperature, max_tokens, stream=False, on_token=None):
        if not self._responses:
            self.calls += 1
            return ""
        idx = min(self.calls, len(self._responses) - 1)
        out = self._responses[idx]
        self.calls += 1
        return out


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
        self.agent.tools.activate_lazy_skill_group("timeseries")
        self.agent.tools.reload_skills()
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
        self.assertIn("Runtime context", routing_query)
        self.assertIn("ACTIVE SKILLS FOR THIS REQUEST", prompt)
        self.assertTrue(
            any(
                skill.id in {"analysis", "preprocessing"}
                for skill in selection.selected_skills
            )
        )

    def test_load_skills_command_activates_lazy_skill_group(self) -> None:
        self.assertNotIn("forecasting", {skill.id for skill in self.agent.tools.list_skills()})

        response = self.agent.chat("/load-skills timeseries", session_id="load_skills_cmd")

        self.assertIn("Loaded: timeseries.", response)
        self.assertIn("timeseries", self.agent.tools.active_lazy_skill_groups())
        self.assertIn("forecasting", {skill.id for skill in self.agent.tools.list_skills()})

    def test_social_greeting_does_not_inject_skill_routing_context(self) -> None:
        prompt, selection, routing_query = self.agent._resolve_system_prompt("hi")

        self.assertIsNotNone(selection)
        assert selection is not None
        # The routing query must not be enriched with runtime context for social messages
        self.assertEqual(routing_query.strip(), "hi")
        # The prompt must NOT inject ACTIVE SKILLS or CODING WORKFLOW HINT sections
        self.assertNotIn("ACTIVE SKILLS FOR THIS REQUEST", prompt)
        self.assertNotIn("CODING WORKFLOW HINT", prompt)

    def test_coding_prompt_prefers_find_and_scoped_read_tools(self) -> None:
        prompt = self.agent._build_system_prompt_for_message(
            "Edit rust-cli/src/main.rs and find the config loader before patching it."
        )

        self.assertIn("CODING WORKFLOW HINT", prompt)
        self.assertIn("find_path", prompt)
        self.assertIn("find_in_file", prompt)
        self.assertIn("sed_read", prompt)
        self.assertIn("read_file_smart", prompt)

    def test_short_greeting_is_not_treated_as_follow_up(self) -> None:
        self.assertFalse(self.agent._is_follow_up_message("hi"))
        self.assertTrue(self.agent._is_follow_up_message("continue"))

    def test_social_greeting_uses_fast_path_without_mcp_and_without_hardcoded_reply(
        self,
    ) -> None:
        agent = create_agent(
            llm_url="http://localhost:8080",
            db_path=str(self.db_path),
            config_overrides={
                "rag_enabled": False,
                "vector_path": str(self.vector_path),
                "max_iterations": 4,
                "pre_turn_thinking": True,
            },
        )
        try:
            llm = _SequenceFakeLLM(["Hello from the model"])
            agent.llm = llm
            load_calls = 0

            def _fake_load():
                nonlocal load_calls
                load_calls += 1

            agent._load_mcp_servers = _fake_load  # type: ignore[method-assign]

            response = agent.chat("hi", session_id="social_fast_path_case")

            self.assertEqual(response, "Hello from the model")
            self.assertEqual(llm.calls, 1)
            self.assertEqual(load_calls, 0)
            self.assertFalse(agent._mcp_loaded)
        finally:
            db = getattr(agent.memory, "_db", None)
            if db is not None:
                db.close()

    def test_social_greeting_falls_through_to_main_llm_flow_when_fast_path_is_empty(
        self,
    ) -> None:
        agent = create_agent(
            llm_url="http://localhost:8080",
            db_path=str(self.db_path),
            config_overrides={
                "rag_enabled": False,
                "vector_path": str(self.vector_path),
                "max_iterations": 2,
                "pre_turn_thinking": False,
            },
        )
        try:
            llm = _SequenceFakeLLM(["", "Hello from the main flow"])
            agent.llm = llm
            load_calls = 0

            def _fake_load():
                nonlocal load_calls
                load_calls += 1

            agent._load_mcp_servers = _fake_load  # type: ignore[method-assign]

            response = agent.chat("hi", session_id="social_main_flow_case")

            self.assertEqual(response, "Hello from the main flow")
            self.assertEqual(llm.calls, 2)
            self.assertEqual(load_calls, 1)
        finally:
            db = getattr(agent.memory, "_db", None)
            if db is not None:
                db.close()

    def test_explicit_reasoning_request_does_not_loop_on_plan_only_reply(self) -> None:
        agent = create_agent(
            llm_url="http://localhost:8080",
            db_path=str(self.db_path),
            config_overrides={
                "rag_enabled": False,
                "vector_path": str(self.vector_path),
                "max_iterations": 4,
                "pre_turn_thinking": False,
            },
        )
        try:
            llm = _SequenceFakeLLM(
                [
                    "**Restatement**: You want a safe migration plan first.\n"
                    "**Goal**: Identify the right implementation sequence.\n"
                    "**Plan**:\n"
                    "1. Inspect the current architecture.\n"
                    "2. Identify risk points.\n"
                    "3. Implement only after the plan is sound.\n"
                    "**Key risks**: state drift during migration.\n"
                    "**Starting with**: I would inspect the architecture boundaries first."
                ]
            )
            agent.llm = llm

            response = agent.chat(
                "Think through the architecture before implementing the migration."
            )

            self.assertIn("**Plan**", response)
            self.assertEqual(llm.calls, 1)
        finally:
            db = getattr(agent.memory, "_db", None)
            if db is not None:
                db.close()

    def test_edit_recovery_nudge_prefers_find_path_and_sed_read_for_missing_file(self) -> None:
        nudge = self.agent._edit_tool_recovery_nudge(
            ToolCall(id="1", name="edit_file_replace", arguments={"path": "foo.py"}),
            '{"status":"error","error":"File not found: foo.py"}',
        )

        self.assertIn("find_path", nudge)
        self.assertIn("sed_read", nudge)

    def test_retrieval_settings_default_to_config(self) -> None:
        use_semantic, mode = self.agent._resolve_retrieval_settings(None, None)

        # Defaults updated to enable semantic hybrid retrieval for better long-session context
        self.assertTrue(use_semantic)
        self.assertEqual(mode, "hybrid")

    def test_lazy_mcp_init_defers_server_load_until_first_run(self) -> None:
        agent = self._make_agent()
        try:
            self.assertFalse(agent._mcp_loaded)
            calls = 0

            def _fake_load():
                nonlocal calls
                calls += 1

            agent._load_mcp_servers = _fake_load  # type: ignore[method-assign]
            agent._ensure_mcp_servers_loaded()
            agent._ensure_mcp_servers_loaded()

            self.assertEqual(calls, 1)
            self.assertTrue(agent._mcp_loaded)
        finally:
            db = getattr(agent.memory, "_db", None)
            if db is not None:
                db.close()

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

    def test_tool_claim_guard_forces_real_tool_call_before_finalizing(self) -> None:
        sid = "tool_claim_guard_case"
        agent = create_agent(
            llm_url="http://localhost:8080",
            db_path=str(self.db_path),
            config_overrides={
                "rag_enabled": False,
                "vector_path": str(self.vector_path),
                "max_iterations": 4,
                "use_toon_for_tools": False,
                "tool_claim_guard_enabled": True,
            },
        )
        try:
            agent.llm = _SequenceFakeLLM(
                [
                    (
                        "I’ve generated a synthetic sine wave time series with 100 points "
                        "and moderate noise. The data is now available in the tool context."
                    ),
                    '{"tool_call":{"name":"search_tools","arguments":{"query":"csv","top_k":3}}}',
                    "Done. I ran the tool and used the result.",
                ]
            )

            response = agent.chat("Find the best tool for csv ingestion.", session_id=sid)
            self.assertTrue(response.startswith("Done. I ran the tool and used the result."))

            messages = agent.memory.get_session_messages(sid)
            tool_msgs = [m for m in messages if m.role == MessageRole.TOOL]
            self.assertTrue(tool_msgs)
            self.assertEqual(tool_msgs[-1].name, "search_tools")
            self.assertGreaterEqual(getattr(agent.llm, "calls", 0), 3)
        finally:
            db = getattr(agent.memory, "_db", None)
            if db is not None:
                db.close()

    def test_reflexion_repair_is_capped_per_turn(self) -> None:
        sid = "reflexion_repair_cap_case"
        agent = create_agent(
            llm_url="http://localhost:8080",
            db_path=str(self.db_path),
            config_overrides={
                "rag_enabled": False,
                "vector_path": str(self.vector_path),
                "max_iterations": 4,
                "use_toon_for_tools": False,
                "append_quality_checklist": False,
                "enable_reflexion_repair": True,
                "reflexion_repair_max_attempts": 1,
            },
        )
        try:
            agent.llm = _SequenceFakeLLM(
                [
                    '{"tool_call":{"name":"run_shell","arguments":{"cmd":"false"}}}',
                    '{"tool_call":{"name":"run_shell","arguments":{"cmd":"false"}}}',
                    "Final answer after capped repair.",
                ]
            )
            repair_calls = 0

            def _fake_repair(call, result_text, tracer):
                del call, result_text, tracer
                nonlocal repair_calls
                repair_calls += 1
                return "[Reflexion Repair] Retry once with corrected arguments."

            with patch.object(agent, "_run_reflexion_repair", side_effect=_fake_repair):
                with patch.object(agent.tools, "execute", return_value="Error: boom"):
                    response = agent.chat(
                        "Run the shell command and fix any issue.",
                        session_id=sid,
                    )

            self.assertEqual(repair_calls, 1)
            self.assertEqual(response, "Final answer after capped repair.")
            self.assertGreaterEqual(getattr(agent.llm, "calls", 0), 3)
        finally:
            db = getattr(agent.memory, "_db", None)
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

    def test_context7_docs_nudge_triggers_for_docs_intent(self) -> None:
        with patch.object(
            self.agent,
            "_context7_tool_names",
            return_value=["resolve_library_id", "query_docs"],
        ):
            nudge = self.agent._context7_docs_nudge(
                "Need latest docs for pydantic validators",
                tool_calls=[],
                selection=None,
            )
        self.assertIn("Context7", nudge)
        self.assertIn("resolve", nudge.lower())

    def test_context7_docs_nudge_stops_after_context7_tool_call(self) -> None:
        nudge = self.agent._context7_docs_nudge(
            "Need docs for numpy array creation",
            tool_calls=[
                ToolCall(
                    id="ctx7_1",
                    name="resolve_library_id",
                    arguments={"query": "numpy"},
                )
            ],
            selection=None,
        )
        self.assertEqual(nudge, "")

    def test_edit_recovery_nudge_for_missing_old_string(self) -> None:
        call = ToolCall(
            id="edit_1",
            name="edit_file_replace",
            arguments={
                "path": "src/example.py",
                "old_string": "missing",
                "new_string": "updated",
            },
        )
        result = json.dumps(
            {"status": "error", "error": "old_string not found in file"}
        )
        nudge = self.agent._edit_tool_recovery_nudge(call, result)
        self.assertIn("Edit recovery", nudge)
        self.assertIn("3-5 unchanged lines", nudge)

    def test_write_detection_for_edit_tools_requires_success(self) -> None:
        call = ToolCall(
            id="edit_2",
            name="edit_file_replace",
            arguments={"path": "src/example.py"},
        )
        ok_result = json.dumps(
            {"status": "ok", "path": "src/example.py", "lines_added": 2}
        )
        err_result = json.dumps({"status": "error", "error": "old_string not found"})
        self.assertTrue(self.agent._tool_call_applied_write(call, ok_result))
        self.assertFalse(self.agent._tool_call_applied_write(call, err_result))

    def test_editing_intent_detects_feature_implementation_requests(self) -> None:
        self.assertTrue(
            self.agent._is_editing_intent(
                "Implement a new command in rust-cli to switch tabs faster."
            )
        )

    def test_docs_intent_detects_library_integration_without_word_docs(self) -> None:
        self.assertTrue(
            self.agent._docs_intent(
                "Implement pydantic validators in our FastAPI request model."
            )
        )

    def test_patch_tools_require_prewrite_inspection(self) -> None:
        self.assertTrue(self.agent._requires_prewrite_inspection("edit_file_replace"))
        self.assertTrue(self.agent._requires_prewrite_inspection("apply_unified_diff"))
        self.assertFalse(self.agent._requires_prewrite_inspection("write_file"))

    def test_inspection_tool_detection(self) -> None:
        self.assertTrue(self.agent._is_inspection_tool_name("rg_search"))
        self.assertTrue(self.agent._is_inspection_tool_name("read_file"))
        self.assertFalse(self.agent._is_inspection_tool_name("write_file"))


if __name__ == "__main__":
    unittest.main()
