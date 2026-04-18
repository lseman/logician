import contextlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import logician_bridge
from src.hooks import SessionStartResult


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
        self.assertTrue(str(overrides["vector_path"]).endswith("message_history.vector"))
        self.assertTrue(str(overrides["rag_vector_path"]).endswith("rag_docs.vector"))
        self.assertTrue(overrides["lazy_mcp_init"])

    def test_fast_init_defers_startup_hooks(self) -> None:
        created: list[dict[str, object]] = []

        def _fake_create_agent(**kwargs):
            created.append(kwargs)
            return _FakeAgent(
                name=str(kwargs.get("system_prompt") or "agent"),
                system_prompt=kwargs.get("system_prompt"),
            )

        class SlowHookEngine:
            def __init__(self, *args, progress_callback=None, **kwargs) -> None:
                self._progress_callback = progress_callback

            def execute_session_start_hooks(self, source: str = "startup") -> SessionStartResult:
                if self._progress_callback is not None:
                    self._progress_callback("discovered", {"source": source, "hook_count": 1})
                time.sleep(0.2)
                if self._progress_callback is not None:
                    self._progress_callback(
                        "context",
                        {
                            "source": source,
                            "ordinal": 0,
                            "plugin_id": "demo@test",
                            "plugin_name": "demo",
                            "context": "Deferred hook context",
                        },
                    )
                    self._progress_callback(
                        "completed",
                        {"source": source, "hook_count": 1, "context_count": 1, "errors": []},
                    )
                return SessionStartResult(
                    additional_contexts=["Deferred hook context"],
                    hook_count=1,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            server = logician_bridge.BridgeServer()
            with (
                patch("logician_bridge._agent_factory", return_value=_fake_create_agent),
                patch("logician_bridge._silent_load", side_effect=lambda: contextlib.nullcontext()),
                patch("src.hooks.HookEngine", SlowHookEngine),
            ):
                start = time.perf_counter()
                result = server.init({"config_path": str(config_path), "fast": True})
                elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.15, f"expected fast init to return quickly, took {elapsed:.3f}s")
        self.assertFalse(result["hook_context_complete"])
        self.assertEqual(result["hook_context"], [])
        self.assertTrue(created[0]["config_overrides"]["lazy_mcp_init"])

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
