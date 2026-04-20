from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from logician_bridge import BridgeServer


class LogicianBridgePluginMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_cwd = os.getcwd()
        self._tmpdir = tempfile.TemporaryDirectory()
        os.chdir(self._tmpdir.name)
        self._memory_dir_path = Path(self._tmpdir.name) / ".logician" / "memory"
        import logician_bridge as _bridge

        self._bridge_module = _bridge
        self._orig_bridge_memory_dir = _bridge.MEMORY_DIR
        _bridge.MEMORY_DIR = self._memory_dir_path

    def tearDown(self) -> None:
        self._bridge_module.MEMORY_DIR = self._orig_bridge_memory_dir
        os.chdir(self._orig_cwd)
        self._tmpdir.cleanup()
        os.environ.pop("LOGICIAN_PROJECT_MEMORY_ENABLED", None)

    def test_parse_memory_summary_includes_enabled_plugin_memory(self) -> None:
        plugin_memory_root = Path(self._tmpdir.name) / "plugin_memory"
        plugin_memory_root.mkdir(parents=True, exist_ok=True)
        (plugin_memory_root / "MEMORY.md").write_text(
            "**#S001** (Apr 16, 2026) — Plugin memory\n| #001 | 🟣 | Plugin observation title |\n",
            encoding="utf-8",
        )

        obs_dir = plugin_memory_root / "obs"
        obs_dir.mkdir(parents=True, exist_ok=True)
        (obs_dir / "index.json").write_text(
            "[{'id': 1, 'session': 'S001'}]".replace("'", '"'),
            encoding="utf-8",
        )

        with patch.object(
            BridgeServer,
            "_plugin_memory_sources",
            return_value=[("test-plugin@local", plugin_memory_root)],
        ):
            bridge = BridgeServer()
            summary = bridge._parse_memory_summary()

        self.assertTrue(summary["has_memories"])
        self.assertEqual(summary["obs_count"], 1)
        self.assertEqual(summary["total_entries"], 1)

    def test_inject_project_memory_appends_plugin_memory_to_system_prompt(self) -> None:
        plugin_memory_root = Path(self._tmpdir.name) / "plugin_memory"
        plugin_memory_root.mkdir(parents=True, exist_ok=True)
        (plugin_memory_root / "MEMORY.md").write_text(
            "**#S001** (Apr 16, 2026) — Plugin memory\n| #001 | 🟣 | Plugin observation title |\n",
            encoding="utf-8",
        )

        with patch.object(
            BridgeServer,
            "_plugin_memory_sources",
            return_value=[("test-plugin@local", plugin_memory_root)],
        ):
            bridge = BridgeServer()
            bridge.agents = {"main": type("A", (), {"system_prompt": "initial"})()}
            bridge._inject_project_memory()

        self.assertIn("<session-memory>", bridge.agents["main"].system_prompt)
        self.assertIn("Plugin observation title", bridge.agents["main"].system_prompt)

    def test_parse_memory_summary_preserves_claude_mem_table_rows(self) -> None:
        plugin_memory_root = Path(self._tmpdir.name) / "plugin_memory"
        plugin_memory_root.mkdir(parents=True, exist_ok=True)
        (plugin_memory_root / "MEMORY.md").write_text(
            "### Apr 16, 2026\n"
            "| file | ID | Time | Title |\n"
            "|---|---|---|\n"
            "| #001 | 8:00 AM | 🔵 | test entry |\n",
            encoding="utf-8",
        )

        with patch.object(
            BridgeServer,
            "_plugin_memory_sources",
            return_value=[("test-plugin@local", plugin_memory_root)],
        ):
            bridge = BridgeServer()
            summary = bridge._parse_memory_summary()

        self.assertTrue(summary["has_memories"])
        self.assertGreater(len(summary["sections"]), 0)
        entries = summary["sections"][0]["entries"]
        self.assertTrue(any(entry.startswith("| file | ID |") for entry in entries))
        self.assertTrue(any(entry.startswith("|---") for entry in entries))
        self.assertTrue(any(entry.startswith("| #001") for entry in entries))

    def test_find_memory_index_prefers_claude_plugin_dirs_but_falls_back(self) -> None:
        plugin_memory_root = Path(self._tmpdir.name) / "plugin_memory"
        nested = plugin_memory_root / "nested" / ".claude-plugin"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "CLAUDE.md").write_text("# Nested plugin memory\n- note", encoding="utf-8")
        (plugin_memory_root / "CLAUDE.md").write_text(
            "# Root plugin memory\n- note", encoding="utf-8"
        )

        bridge = BridgeServer()
        found = bridge._find_memory_index(plugin_memory_root)

        self.assertIsNotNone(found)
        self.assertEqual(found.name, "CLAUDE.md")
        self.assertEqual(found.parent.name, ".claude-plugin")

    def test_find_memory_index_falls_back_to_first_claude_md(self) -> None:
        plugin_memory_root = Path(self._tmpdir.name) / "plugin_memory"
        nested = plugin_memory_root / "nested"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "CLAUDE.md").write_text("# Nested plugin memory\n- note", encoding="utf-8")

        bridge = BridgeServer()
        found = bridge._find_memory_index(plugin_memory_root)

        self.assertIsNotNone(found)
        self.assertEqual(found.name, "CLAUDE.md")
        self.assertEqual(found.parent.name, "nested")

    def test_project_memory_disabled_skips_summary_and_prompt_injection(self) -> None:
        memory_root = Path(self._tmpdir.name) / ".logician" / "memory"
        memory_root.mkdir(parents=True, exist_ok=True)
        (memory_root / "MEMORY.md").write_text("# Project memory\n- note", encoding="utf-8")

        bridge = BridgeServer()
        bridge.cfg = {"project_memory_enabled": False}
        bridge.agents = {"main": type("A", (), {"system_prompt": "initial"})()}

        summary = bridge._parse_memory_summary()
        bridge._inject_project_memory()

        self.assertFalse(summary["has_memories"])
        self.assertEqual(summary["total_entries"], 0)
        self.assertEqual(bridge.agents["main"].system_prompt, "initial")

    def test_inject_startup_hook_contexts_into_system_prompt(self) -> None:
        bridge = BridgeServer()
        bridge.agents = {
            "main": type("A", (), {"system_prompt": "initial"})(),
            "secondary": type("A", (), {"system_prompt": ""})(),
        }
        bridge._startup_hook_contexts = ["Plugin context line 1", "Plugin context line 2"]

        bridge._inject_startup_hook_contexts()

        self.assertIn("<startup-hook-context>", bridge.agents["main"].system_prompt)
        self.assertIn("Plugin context line 1", bridge.agents["main"].system_prompt)
        self.assertIn("Plugin context line 2", bridge.agents["main"].system_prompt)
        self.assertTrue(bridge.agents["secondary"].system_prompt.startswith("<startup-hook-context>"))

    def test_context_command_shows_current_system_prompt_and_message_window(self) -> None:
        bridge = BridgeServer()
        bridge.active = "main"
        bridge.sessions = {"main": "sess-1"}

        agent = type("A", (), {})()
        agent.preview_prompt_context = lambda message, sid: {
            "session_id": sid,
            "classified_as": "execution",
            "domain_groups": ["tools"],
            "history_loaded_count": 1,
            "untrimmed_message_count": 2,
            "trimmed_message_count": 2,
            "history_limit": 10,
            "context_token_budget": 1000,
            "system_prompt": "System prompt here",
            "messages": [
                {"role": "user", "name": "", "content": "hello"},
                {"role": "assistant", "name": "", "content": "hi"},
            ],
        }
        agent.describe_runtime_context = lambda sid: {
            "session_id": sid,
            "persisted_messages": 5,
            "history_limit": 10,
            "runtime": {
                "loaded": True,
                "data_name": "demo.csv",
                "row_count": 10,
                "value_columns": ["col1"],
                "active_repos": [{"id": "repo-1", "name": "repo1"}],
            },
        }

        bridge.agents = {"main": agent}

        result = bridge.slash({"raw": "/context"})
        rendered = "\n".join(result["messages"])

        self.assertIn("## System Prompt", rendered)
        self.assertIn("System prompt here", rendered)
        self.assertIn("## Message Window", rendered)
        self.assertIn("hello", rendered)
        self.assertIn("## Runtime Context", rendered)
        self.assertIn("dataset: demo.csv", rendered)
        self.assertIn("active_repos: repo-1", rendered)

    def test_context_command_collapses_duplicate_startup_hook_blocks(self) -> None:
        bridge = BridgeServer()
        bridge.active = "main"
        bridge.sessions = {"main": "sess-1"}

        agent = type("A", (), {})()
        agent.preview_prompt_context = lambda message, sid: {
            "session_id": sid,
            "classified_as": "execution",
            "domain_groups": [],
            "history_loaded_count": 0,
            "untrimmed_message_count": 0,
            "trimmed_message_count": 0,
            "history_limit": 10,
            "context_token_budget": 1000,
            "system_prompt": (
                "initial\n"
                "<startup-hook-context>one</startup-hook-context>\n\n"
                "<startup-hook-context>two</startup-hook-context>"
            ),
            "messages": [],
        }
        agent.describe_runtime_context = lambda sid: {
            "session_id": sid,
            "persisted_messages": 0,
            "history_limit": 10,
            "runtime": {"loaded": False, "active_repos": []},
        }
        bridge.agents = {"main": agent}

        result = bridge.slash({"raw": "/context"})
        rendered = "\n".join(result["messages"])

        self.assertEqual(rendered.count("<startup-hook-context>"), 1)
        self.assertEqual(rendered.count("</startup-hook-context>"), 1)

    def test_context_command_merges_duplicate_startup_hook_blocks(self) -> None:
        bridge = BridgeServer()
        bridge.active = "main"
        bridge.sessions = {"main": "sess-1"}

        repeated_block = (
            "# SOUL Block\n"
            "This is repeated prompt content from SOUL.md.\n"
            "It is long enough to be treated as a section.\n"
            "It should only appear once when deduplicated."
        )

        agent = type("A", (), {})()
        agent.preview_prompt_context = lambda message, sid: {
            "session_id": sid,
            "classified_as": "execution",
            "domain_groups": [],
            "history_loaded_count": 0,
            "untrimmed_message_count": 0,
            "trimmed_message_count": 0,
            "history_limit": 10,
            "context_token_budget": 1000,
            "system_prompt": (
                f"initial\n\n{repeated_block}\n\n{repeated_block}\n\n"
                "<startup-hook-context>one</startup-hook-context>\n\n"
                "<startup-hook-context>two</startup-hook-context>\n\n"
                "<startup-hook-context>three</startup-hook-context>"
            ),
            "messages": [],
        }
        agent.describe_runtime_context = lambda sid: {
            "session_id": sid,
            "persisted_messages": 0,
            "history_limit": 10,
            "runtime": {"loaded": False, "active_repos": []},
        }
        bridge.agents = {"main": agent}

        result = bridge.slash({"raw": "/context"})
        rendered = "\n".join(result["messages"])

        self.assertEqual(rendered.count("<startup-hook-context>"), 1)
        self.assertEqual(rendered.count("</startup-hook-context>"), 1)
        self.assertIn("one\n\ntwo\n\nthree", rendered)

    def test_status_includes_current_context_size(self) -> None:
        bridge = BridgeServer()
        bridge.active = "main"
        bridge.sessions = {"main": "sess-1"}

        agent = type("A", (), {})()
        agent.describe_runtime_context = lambda sid: {
            "persisted_messages": 5,
            "loaded_message_budget": 4,
            "history_limit": 10,
            "runtime": {"loaded": False, "active_repos": []},
        }
        agent.preview_prompt_context = lambda message, sid: {
            "trimmed_message_count": 4,
            "history_limit": 10,
            "context_token_budget": 900,
        }
        bridge.agents = {"main": agent}

        result = bridge.slash({"raw": "/status"})
        rendered = "\n".join(result["messages"])

        self.assertIn("ctx:", rendered)
        self.assertNotIn("ctx: 0", rendered)

    def test_status_uses_default_agent_when_none_is_active(self) -> None:
        bridge = BridgeServer()
        bridge.active = None
        bridge.sessions = {"main": "sess-1"}

        agent = type("A", (), {})()
        agent.describe_runtime_context = lambda sid: {
            "persisted_messages": 3,
            "loaded_message_budget": 2,
            "history_limit": 10,
            "runtime": {"loaded": False, "active_repos": []},
        }
        agent.preview_prompt_context = lambda message, sid: {
            "trimmed_message_count": 7,
            "history_limit": 10,
            "context_token_budget": 1200,
        }
        bridge.agents = {"main": agent}

        result = bridge.slash({"raw": "/status"})
        rendered = "\n".join(result["messages"])

        self.assertIn("active: main", rendered)
        self.assertIn("ctx:", rendered)
        self.assertNotIn("ctx: 0", rendered)

    def test_chat_state_uses_context_preview_size(self) -> None:
        bridge = BridgeServer()
        bridge.active = "main"
        bridge.sessions = {"main": "sess-1"}

        class FakeRunResponse:
            def __init__(self) -> None:
                self.final_response = "done"
                self.tool_calls = []
                self.iterations = 1
                self.messages = []

        agent = type("A", (), {})()
        agent.describe_runtime_context = lambda sid: {
            "session_id": sid,
            "persisted_messages": 5,
            "loaded_message_budget": 2,
            "history_limit": 10,
            "runtime": {"loaded": False, "active_repos": []},
        }
        agent.preview_prompt_context = lambda message, sid: {
            "session_id": sid,
            "classified_as": "execution",
            "history_loaded_count": 2,
            "untrimmed_message_count": 3,
            "trimmed_message_count": 3,
            "history_limit": 10,
            "context_token_budget": 900,
            "system_prompt": "System prompt here",
            "messages": [
                {"role": "user", "name": "", "content": "hello"},
                {"role": "assistant", "name": "", "content": "hi"},
            ],
        }
        agent.run = lambda *args, **kwargs: FakeRunResponse()
        bridge.agents = {"main": agent}

        result = bridge.chat({"message": "hello"})
        state = result["state"]
        expected = len(
            bridge._format_context_preview(
                agent.preview_prompt_context("", "sess-1"),
                "sess-1",
                agent.describe_runtime_context("sess-1"),
            )
        )

        self.assertEqual(state["context_size"], expected)
        self.assertNotEqual(state["context_size"], 2)

    def test_fire_session_end_hooks_delegate_to_agent_core(self) -> None:
        bridge = BridgeServer()
        bridge.active = "main"
        bridge.sessions = {"main": "sess-1", "aux": "sess-2"}

        class FakeAgent:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            def end_session(self, *, reason: str, session_id: str) -> None:
                self.calls.append((reason, session_id))

        main_agent = FakeAgent()
        aux_agent = FakeAgent()
        bridge.agents = {"main": main_agent, "aux": aux_agent}

        bridge._fire_session_end_hooks("exit")

        self.assertEqual(main_agent.calls, [("exit", "sess-1")])
        self.assertEqual(aux_agent.calls, [("exit", "sess-2")])

    def test_quit_fires_session_end_for_each_known_session(self) -> None:
        bridge = BridgeServer()
        bridge.active = "main"
        bridge.sessions = {"main": "sess-1", "aux": "sess-2"}
        bridge.agents = {
            "main": type("A", (), {"memory": type("M", (), {"db_path": "/tmp/main.db"})()})(),
            "aux": type("A", (), {"memory": type("M", (), {"db_path": "/tmp/aux.db"})()})(),
        }

        seen: list[tuple[str, str]] = []

        class FakeAgent:
            def __init__(self, session_id: str) -> None:
                self.memory = type("M", (), {"db_path": f"/tmp/{session_id}.db"})()
                self.session_id = session_id

            def end_session(self, *, reason: str, session_id: str) -> None:
                seen.append((reason, session_id))

        bridge.agents = {
            "main": FakeAgent("sess-1"),
            "aux": FakeAgent("sess-2"),
        }

        result = bridge.slash({"raw": "/quit"})

        self.assertTrue(result["exit"])
        self.assertEqual(
            seen,
            [
                ("exit", "sess-1"),
                ("exit", "sess-2"),
            ],
        )

    def test_project_memory_disabled_skips_observation_write(self) -> None:
        bridge = BridgeServer()
        bridge.cfg = {"project_memory_enabled": False}

        bridge._record_turn_observation(
            query="update the file",
            response="done",
            session_id="sess-1",
            tool_calls=[{"name": "write_file", "arguments": {}}],
            written_paths=["src/app.py"],
        )

        memory_root = Path(self._tmpdir.name) / ".logician" / "memory"
        self.assertFalse(memory_root.exists())


if __name__ == "__main__":
    unittest.main()
