import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import logician_bridge


class _FakeMcpClient:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeAgent:
    def __init__(self, *, name: str, system_prompt: str | None = None) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self._mcp_clients = [_FakeMcpClient(f"mcp-{name}")]


class BridgeMultiAgentTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        config_path = root / "agent_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "agent_name": "default",
                    "endpoint": "http://localhost:8080",
                    "use_chat_api": True,
                    "chat_template": "deepseek",
                    "agents": {
                        "coder": {
                            "system_prompt": "write code",
                            "temperature": 0.2,
                        },
                        "reviewer": {
                            "system_prompt": "review code",
                            "temperature": 0.4,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_init_creates_sessions_for_all_agents(self) -> None:
        created: list[dict[str, object]] = []

        def _fake_create_agent(**kwargs):
            created.append(kwargs)
            return _FakeAgent(
                name=str(kwargs.get("system_prompt") or "agent"),
                system_prompt=kwargs.get("system_prompt"),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            server = logician_bridge.BridgeServer()
            with (
                patch("logician_bridge._agent_factory", return_value=_fake_create_agent),
                patch("logician_bridge._silent_load", side_effect=lambda: contextlib.nullcontext()),
            ):
                result = server.init({"config_path": str(config_path)})

        self.assertEqual(sorted(server.agents.keys()), ["coder", "reviewer"])
        self.assertEqual(sorted(server.sessions.keys()), ["coder", "reviewer"])
        self.assertEqual(server.active, "coder")
        self.assertEqual(result["state"]["active"], "coder")
        self.assertEqual(sorted(result["state"]["agents"]), ["coder", "reviewer"])
        self.assertEqual(len(created), 2)
        overrides = created[0]["config_overrides"]
        self.assertFalse(overrides["prompt_rag_context_enabled"])
        self.assertFalse(overrides["startup_warmup_llm"])
        self.assertTrue(overrides["startup_background_warmup"])
        self.assertTrue(overrides["startup_warmup_skills"])
        self.assertTrue(str(overrides["memory_palace_db_path"]).endswith("memory_palace.db"))

    def test_agent_command_switches_to_precreated_session(self) -> None:
        def _fake_create_agent(**kwargs):
            return _FakeAgent(
                name=str(kwargs.get("system_prompt") or "agent"),
                system_prompt=kwargs.get("system_prompt"),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            server = logician_bridge.BridgeServer()
            with (
                patch("logician_bridge._agent_factory", return_value=_fake_create_agent),
                patch("logician_bridge._silent_load", side_effect=lambda: contextlib.nullcontext()),
            ):
                server.init({"config_path": str(config_path)})
                reviewer_session = server.sessions["reviewer"]
                reply = server.slash({"raw": "/agent reviewer", "config_path": str(config_path)})

        self.assertEqual(server.active, "reviewer")
        self.assertEqual(server.sessions["reviewer"], reviewer_session)
        self.assertIn("switched to 'reviewer'", reply["messages"][0])
        self.assertEqual(reply["state"]["session"], reviewer_session)

    def test_reload_recreates_sessions_for_all_agents(self) -> None:
        def _fake_create_agent(**kwargs):
            return _FakeAgent(
                name=str(kwargs.get("system_prompt") or "agent"),
                system_prompt=kwargs.get("system_prompt"),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            server = logician_bridge.BridgeServer()
            with (
                patch("logician_bridge._agent_factory", return_value=_fake_create_agent),
                patch("logician_bridge._silent_load", side_effect=lambda: contextlib.nullcontext()),
            ):
                server.init({"config_path": str(config_path)})
                first_sessions = dict(server.sessions)
                reply = server.slash({"raw": "/reload", "config_path": str(config_path)})

        self.assertEqual(sorted(server.sessions.keys()), ["coder", "reviewer"])
        self.assertNotEqual(server.sessions["coder"], first_sessions["coder"])
        self.assertNotEqual(server.sessions["reviewer"], first_sessions["reviewer"])
        self.assertIn("reloaded · 2 agent(s)", reply["messages"][0])


if __name__ == "__main__":
    unittest.main()
