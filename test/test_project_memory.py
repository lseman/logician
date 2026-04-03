from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.project_memory import load_index, record_observation


def _load_mem_search():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "global"
        / "mem_search"
        / "scripts"
        / "mem_search.py"
    )
    spec = importlib.util.spec_from_file_location("test_mem_search_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mem_search


mem_search = _load_mem_search()


class ProjectMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_cwd = os.getcwd()
        self._tmpdir = tempfile.TemporaryDirectory()
        os.chdir(self._tmpdir.name)

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        self._tmpdir.cleanup()

    def test_record_observation_reuses_session_label_for_same_session_key(self) -> None:
        first = record_observation(
            obs_type="change",
            title="FileReadTool updated",
            content="Updated write behavior for Python files.",
            files=["src/tools/core/FileReadTool/tool.py"],
            session_key="cli_abc123",
        )
        second = record_observation(
            obs_type="bugfix",
            title="shell.py fixed",
            content="Fixed newline normalization for run_python.",
            files=["skills/coding/shell/shell.py"],
            session_key="cli_abc123",
        )

        index = load_index()
        self.assertEqual(first["session"], "S001")
        self.assertEqual(second["session"], "S001")
        self.assertEqual(len(index), 2)
        self.assertEqual(index[0]["session_key"], "cli_abc123")
        self.assertEqual(index[1]["session_key"], "cli_abc123")

    def test_rebuild_memory_md_contains_session_table(self) -> None:
        record_observation(
            obs_type="change",
            title="bridge emits skill events",
            content="Added automatic skill event emission before runs.",
            files=["logician_bridge.py"],
            session_key="cli_1",
        )

        memory_md = Path(".logician/memory/MEMORY.md").read_text(encoding="utf-8")

        self.assertIn("# Project Memory — Logician", memory_md)
        self.assertIn("## S001", memory_md)
        self.assertIn("### logician_bridge.py", memory_md)
        self.assertIn("bridge emits skill events", memory_md)

    def test_mem_search_matches_preview_text(self) -> None:
        record_observation(
            obs_type="bugfix",
            title="shell normalization fixed",
            content=(
                "Updated run_python handling so escaped newlines and quoted "
                "strings no longer break payload execution."
            ),
            files=["skills/coding/shell/shell.py"],
            session_key="cli_2",
        )

        payload = json.loads(mem_search("quoted strings"))

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["count"], 1)
        self.assertIn("#001", payload["ids"])


if __name__ == "__main__":
    unittest.main()
