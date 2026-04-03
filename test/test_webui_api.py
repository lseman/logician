from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tools.core.TaskTool import load_persisted_todos, todo
from src.webui.app import ChatRequest, create_app
from src.webui.data import (
    extract_repo_focus_paths,
    get_current_todos,
    get_memory_overview,
    get_repo_focus_graph,
    get_repo_graph,
    get_session_detail,
    list_repos,
    list_sessions,
)


def _write_runtime_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            tool_call_id TEXT,
            name TEXT,
            thinking_log TEXT,
            vectorize INTEGER NOT NULL DEFAULT 1,
            vector_id TEXT UNIQUE
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE session_runtime_state (
            session_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute(
        """
        INSERT INTO messages (session_id, role, content, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        ("session-1", "user", "Inspect the repo graph", "2026-03-22 10:00:00"),
    )
    conn.execute(
        """
        INSERT INTO messages (session_id, role, content, thinking_log, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "session-1",
            "assistant",
            "Here is the current graph focus.",
            json.dumps(["inspect repo graph", "summarize focus"]),
            "2026-03-22 10:00:01",
        ),
    )
    conn.execute(
        """
        INSERT INTO session_runtime_state (session_id, state_json, updated_at)
        VALUES (?, ?, ?)
        """,
        (
            "session-1",
            json.dumps(
                {
                    "todo_items": [
                        {
                            "id": 1,
                            "title": "Render graph",
                            "status": "in-progress",
                            "note": "focus repo context",
                        }
                    ]
                }
            ),
            "2026-03-22 10:00:02",
        ),
    )
    conn.commit()
    conn.close()


def _write_memory(tmp_path: Path) -> None:
    obs_dir = tmp_path / ".logician" / "memory" / "obs"
    obs_dir.mkdir(parents=True, exist_ok=True)
    (obs_dir / "index.json").write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "session": "S001",
                    "type": "feature",
                    "emoji": "🟣",
                    "timestamp": "2026-03-22T10:00:00Z",
                    "title": "Built a web panel",
                    "files": ["src/webui/app.py"],
                    "preview": "Created a panel for chat and graph navigation.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (obs_dir / "0001.md").write_text(
        "\n".join(
            [
                "---",
                "id: 1",
                "type: feature",
                "description: Web UI panel",
                "---",
                "",
                "Created a panel for chat and graph navigation.",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / ".logician" / "memory" / "MEMORY.md").write_text(
        "# Memory\n\n- one summary line",
        encoding="utf-8",
    )


def _write_repos(tmp_path: Path) -> None:
    repo_root = tmp_path / ".logician" / "repos" / "demo-repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".logician" / "repos" / "index.json").write_text(
        json.dumps(
            [
                {
                    "id": "demo-repo",
                    "name": "demo-repo",
                    "path": str(tmp_path / "demo-repo"),
                    "graph_nodes": 3,
                    "graph_edges": 2,
                    "chunks_added": 5,
                    "artifacts": {
                        "graph_path": str(repo_root / "graph.jsonl"),
                    },
                    "git": {"branch": "main", "commit": "abc123"},
                }
            ]
        ),
        encoding="utf-8",
    )
    (repo_root / "graph.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_type": "node",
                        "id": "repo:demo-repo",
                        "kind": "repo",
                        "name": "demo-repo",
                    }
                ),
                json.dumps(
                    {
                        "record_type": "node",
                        "id": "file:src/main.py",
                        "kind": "file",
                        "name": "main.py",
                        "rel_path": "src/main.py",
                    }
                ),
                json.dumps(
                    {
                        "record_type": "edge",
                        "kind": "contains",
                        "source": "repo:demo-repo",
                        "target": "file:src/main.py",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )


class TodoPersistenceTests(unittest.TestCase):
    def test_todo_persistence_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_env = os.environ.get("LOGICIAN_TODO_STATE_PATH")
            os.environ["LOGICIAN_TODO_STATE_PATH"] = str(Path(tmp_dir) / "todos.json")
            try:
                payload = todo(
                    command="set",
                    items=[{"title": "Ship UI", "status": "in-progress"}],
                )

                self.assertEqual(payload["status"], "ok")
                self.assertEqual(
                    load_persisted_todos(),
                    [
                        {
                            "id": 1,
                            "title": "Ship UI",
                            "status": "in-progress",
                            "note": "",
                        }
                    ],
                )
            finally:
                if original_env is None:
                    os.environ.pop("LOGICIAN_TODO_STATE_PATH", None)
                else:
                    os.environ["LOGICIAN_TODO_STATE_PATH"] = original_env


class WebUiDataTests(unittest.TestCase):
    def test_webui_data_accessors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            old_cwd = Path.cwd()
            old_env = os.environ.get("LOGICIAN_TODO_STATE_PATH")
            try:
                os.chdir(tmp_path)
                os.environ["LOGICIAN_TODO_STATE_PATH"] = str(
                    tmp_path / ".logician" / "state" / "todos.json"
                )
                _write_runtime_db(tmp_path / "agent_sessions.db")
                _write_memory(tmp_path)
                _write_repos(tmp_path)
                (tmp_path / ".logician" / "state").mkdir(parents=True, exist_ok=True)
                (tmp_path / ".logician" / "state" / "todos.json").write_text(
                    json.dumps(
                        [
                            {
                                "id": 1,
                                "title": "Persisted todo",
                                "status": "completed",
                                "note": "",
                            }
                        ]
                    ),
                    encoding="utf-8",
                )

                sessions = list_sessions()
                self.assertEqual(sessions[0]["id"], "session-1")
                self.assertEqual(sessions[0]["title"], "Inspect the repo graph")
                self.assertEqual(
                    sessions[0]["preview"],
                    "Here is the current graph focus.",
                )

                todos = get_current_todos()
                self.assertEqual(todos[0]["title"], "Persisted todo")

                memory = get_memory_overview()
                self.assertEqual(memory["count"], 1)

                repos = list_repos()
                self.assertEqual(repos[0]["id"], "demo-repo")

                session_payload = get_session_detail("session-1")
                self.assertEqual(
                    session_payload["messages"][0]["content"],
                    "Inspect the repo graph",
                )
                self.assertEqual(
                    session_payload["messages"][1]["thinking_log"],
                    ["inspect repo graph", "summarize focus"],
                )
                self.assertEqual(session_payload["todos"][0]["title"], "Render graph")

                graph = get_repo_graph("demo-repo")
                self.assertTrue(graph["nodes"])

                focused = get_repo_focus_graph(
                    "demo-repo",
                    focus_paths=["src/main.py"],
                    query="main",
                )
                self.assertEqual(focused["focus_paths"], ["src/main.py"])

                extracted = extract_repo_focus_paths(
                    "demo-repo",
                    arguments={"path": str(tmp_path / "demo-repo" / "src" / "main.py")},
                    result_output=json.dumps(
                        {
                            "results": [
                                {"repo_rel_path": "src/main.py"},
                            ]
                        }
                    ),
                )
                self.assertIn("src/main.py", extracted)
            finally:
                os.chdir(old_cwd)
                if old_env is None:
                    os.environ.pop("LOGICIAN_TODO_STATE_PATH", None)
                else:
                    os.environ["LOGICIAN_TODO_STATE_PATH"] = old_env


class WebUiApiTests(unittest.TestCase):
    def test_chat_stream_post_returns_sse_events(self) -> None:
        class _StubResponse:
            final_response = "Streamed answer"
            thinking_log = ["check context"]
            debug = {"session_id": "stream-session"}

        class _StubAgent:
            def run(self, *args, **kwargs):
                stream_callback = kwargs.get("stream_callback")
                thinking_callback = kwargs.get("thinking_callback")
                tool_callback = kwargs.get("tool_callback")
                if thinking_callback is not None:
                    thinking_callback("check context")
                if tool_callback is not None:
                    tool_callback("read_file", {"path": "src/main.py"}, {"stage": "start"})
                    tool_callback(
                        "read_file",
                        {"path": "src/main.py"},
                        {
                            "stage": "end",
                            "status": "ok",
                            "duration_ms": 7,
                            "result_preview": "file contents",
                        },
                    )
                if stream_callback is not None:
                    stream_callback("Hello")
                return _StubResponse()

        with (
            patch("src.webui.app._build_agent", return_value=_StubAgent()),
            patch(
                "src.webui.app.get_session_detail",
                return_value={
                    "messages": [{"role": "assistant", "content": "Streamed answer"}],
                    "todos": [],
                    "runtime": {},
                },
            ),
        ):
            app = create_app()
            route = next(
                route
                for route in app.routes
                if getattr(route, "path", "") == "/api/chat/stream"
                and "POST" in getattr(route, "methods", set())
            )
            response = route.endpoint(ChatRequest(message="hello from post", fresh_session=True))

        self.assertEqual(response.media_type, "text/event-stream")
        self.assertEqual(response.headers["Cache-Control"], "no-cache")
        self.assertEqual(response.headers["X-Accel-Buffering"], "no")
        self.assertIsNotNone(response.body_iterator)


if __name__ == "__main__":
    unittest.main()
