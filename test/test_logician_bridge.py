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
            "|---|---|---|---|\n"
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
