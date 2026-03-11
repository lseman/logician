import unittest
from types import SimpleNamespace

from src.agent.core import Agent


class InspectionResultGuardTests(unittest.TestCase):
    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent.config = SimpleNamespace()
        return agent

    def test_get_project_map_empty_claim_is_flagged(self) -> None:
        agent = self._agent()
        nudge = agent._inspection_result_guard_nudge(
            response_text="Manual verification confirms the rust-cli directory is empty.",
            tool_name="get_project_map",
            payload={
                "status": "ok",
                "root": "/repo/rust-cli",
                "file_count": 3,
                "files": [
                    {"path": "Cargo.toml"},
                    {"path": "src/main.rs"},
                ],
            },
        )
        self.assertIn("contradicts", nudge.lower())
        self.assertIn("file_count=3", nudge)

    def test_git_status_clean_claim_is_flagged(self) -> None:
        agent = self._agent()
        note = agent._inspection_result_runtime_note(
            final_text="The working tree is clean with no changes to commit.",
            tool_name="git_status",
            payload={
                "status": "ok",
                "repo": "/repo",
                "staged": [{"file": "a.py", "code": "M"}],
                "unstaged": [],
                "untracked": [],
            },
        )
        self.assertIn("conflicts", note.lower())
        self.assertIn("staged=1", note)

    def test_non_contradictory_read_file_claim_is_ignored(self) -> None:
        agent = self._agent()
        nudge = agent._inspection_result_guard_nudge(
            response_text="The file exists and contains content.",
            tool_name="read_file",
            payload={
                "status": "ok",
                "path": "/repo/file.py",
                "total_lines": 12,
                "returned_lines": "1-12",
                "content": "print('x')",
            },
        )
        self.assertEqual(nudge, "")


if __name__ == "__main__":
    unittest.main()
