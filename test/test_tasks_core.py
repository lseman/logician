from __future__ import annotations

import unittest

from src.tools.core.TaskTool import todo
from src.tools.runtime import Context


class CoreTodoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = Context()
        todo.__globals__["ctx"] = self.ctx

    def tearDown(self) -> None:
        todo.__globals__.pop("ctx", None)

    def test_legacy_todo_updates_context_state(self) -> None:
        rendered = todo(
            [
                {"content": "Inspect panel", "status": "in_progress"},
                {"content": "Verify skills", "status": "pending"},
            ]
        )

        self.assertEqual(rendered["status"], "ok")
        self.assertIn("Inspect panel", rendered["view"])
        self.assertEqual(len(self.ctx.todo_items), 2)
        self.assertEqual(self.ctx.todo_items[0]["status"], "in-progress")
        self.assertEqual(self.ctx.todo_items[1]["status"], "not-started")

    def test_structured_view_returns_json_payload(self) -> None:
        todo(
            command="set",
            items=[
                {"title": "Inspect panel", "status": "in-progress", "note": "open Ctrl+T"},
            ],
        )

        payload = todo(command="view")

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["todos"][0]["title"], "Inspect panel")
        self.assertEqual(payload["todos"][0]["note"], "open Ctrl+T")

    def test_context_state_round_trips_with_todos(self) -> None:
        todo(
            command="set",
            items=[{"title": "Persist task", "status": "completed"}],
        )

        state = self.ctx.to_state()
        restored = Context()
        restored.load_state(state)

        self.assertEqual(restored.todo_items[0]["title"], "Persist task")
        self.assertEqual(restored.todo_items[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
