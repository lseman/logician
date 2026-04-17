import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
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

    def test_plugins_command_can_enable_and_disable_plugins(self) -> None:
        class FakeTools:
            def __init__(self) -> None:
                self.paths: list[Path] = []

            def set_additional_skills_dir_paths(self, paths: list[Path]) -> None:
                self.paths = list(paths)

        class FakeAgentWithTools(_FakeAgent):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.tools = FakeTools()

        class FakeManager:
            def __init__(self) -> None:
                self.enabled = False

            def list_plugins(self) -> dict[str, Any]:
                return {
                    "status": "ok",
                    "plugins": [
                        {
                            "plugin_id": "test-plugin@local",
                            "version": "1.0.0",
                            "enabled": self.enabled,
                            "scope": "user",
                            "install_path": "/tmp/test-plugin",
                        }
                    ],
                }

            def enable(self, name: str) -> dict[str, Any]:
                self.enabled = True
                return {"status": "enabled", "message": f"Plugin '{name}' has been enabled."}

            def disable(self, name: str) -> dict[str, Any]:
                self.enabled = False
                return {"status": "disabled", "message": f"Plugin '{name}' has been disabled."}

            def skills_paths(self) -> list[Path]:
                return [Path("/tmp/test-plugin/skills")] if self.enabled else []

        def _fake_create_agent(**kwargs):
            return FakeAgentWithTools(
                name=str(kwargs.get("system_prompt") or "agent"),
                system_prompt=kwargs.get("system_prompt"),
            )

        plugin_manager = FakeManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            server = logician_bridge.BridgeServer()
            with (
                patch("logician_bridge._agent_factory", return_value=_fake_create_agent),
                patch("logician_bridge._silent_load", side_effect=lambda: contextlib.nullcontext()),
                patch("src.plugin_manager.manager.PluginManager", return_value=plugin_manager),
            ):
                server.init({"config_path": str(config_path)})
                reply = server.slash({"raw": "/plugins", "config_path": str(config_path)})
                self.assertIn("Installed plugins", reply["messages"][0])

                reply = server.slash(
                    {"raw": "/plugins enable test-plugin", "config_path": str(config_path)}
                )
                self.assertIn("has been enabled", reply["messages"][0])
                self.assertEqual(
                    server._active_agent().tools.paths, [Path("/tmp/test-plugin/skills")]
                )

                reply = server.slash(
                    {"raw": "/plugins disable test-plugin", "config_path": str(config_path)}
                )
                self.assertIn("has been disabled", reply["messages"][0])
                self.assertEqual(server._active_agent().tools.paths, [])

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
