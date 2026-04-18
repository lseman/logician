from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.hooks.engine import HookEngine
from src.hooks.loader import LoadedHook
from src.hooks.types import (
    HookCommand,
    HookCommandType,
    HookDefinition,
    HookEventType,
    HookExecutionResult,
)


class HookEngineStartupTests(unittest.TestCase):
    def test_startup_commands_run_in_parallel_and_keep_result_order(self) -> None:
        engine = HookEngine()
        first_hook = LoadedHook(
            plugin_id="demo@test",
            plugin_name="demo",
            plugin_dir=Path.cwd(),
            event_type=HookEventType.SESSION_START,
            definition=HookDefinition(
                hooks=[HookCommand(type=HookCommandType.COMMAND, command="first")]
            ),
        )
        second_hook = LoadedHook(
            plugin_id="demo@test",
            plugin_name="demo",
            plugin_dir=Path.cwd(),
            event_type=HookEventType.SESSION_START,
            definition=HookDefinition(
                hooks=[HookCommand(type=HookCommandType.COMMAND, command="second")]
            ),
        )
        engine.loader.get_session_start_hooks = lambda: [first_hook, second_hook]

        def fake_execute(command, _loaded_hook, *, source, deadline=None):
            self.assertEqual(source, "startup")
            self.assertIsNotNone(deadline)
            time.sleep(0.25)
            return HookExecutionResult(additional_contexts=[str(command.command)])

        with patch.object(engine, "_execute_command", side_effect=fake_execute):
            start = time.perf_counter()
            result = engine.execute_session_start_hooks("startup")
            elapsed = time.perf_counter() - start

        self.assertLess(
            elapsed,
            0.45,
            f"expected startup hooks to run in parallel, took {elapsed:.3f}s",
        )
        self.assertEqual(result.additional_contexts, ["first", "second"])

    def test_progress_callback_receives_context_and_completion_updates(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        engine = HookEngine(progress_callback=lambda kind, payload: events.append((kind, payload)))
        hook = LoadedHook(
            plugin_id="demo@test",
            plugin_name="demo",
            plugin_dir=Path.cwd(),
            event_type=HookEventType.SESSION_START,
            definition=HookDefinition(
                hooks=[HookCommand(type=HookCommandType.COMMAND, command="only")]
            ),
        )
        engine.loader.get_session_start_hooks = lambda: [hook]

        with patch.object(
            engine,
            "_execute_command",
            return_value=HookExecutionResult(
                additional_contexts=["ctx"],
                initial_user_message="hello",
            ),
        ):
            result = engine.execute_session_start_hooks("startup")

        self.assertEqual(result.additional_contexts, ["ctx"])
        self.assertIn(
            ("discovered", {"source": "startup", "hook_count": 1}),
            events,
        )
        self.assertTrue(any(kind == "hook_started" for kind, _payload in events))
        self.assertTrue(
            any(kind == "context" and payload.get("context") == "ctx" for kind, payload in events)
        )
        self.assertTrue(
            any(
                kind == "initial_user_message" and payload.get("message") == "hello"
                for kind, payload in events
            )
        )
        self.assertTrue(any(kind == "hook_finished" for kind, _payload in events))
        self.assertTrue(any(kind == "completed" for kind, _payload in events))

    def test_startup_uses_manifest_timeout_and_budget(self) -> None:
        engine = HookEngine()
        hook = LoadedHook(
            plugin_id="demo@test",
            plugin_name="demo",
            plugin_dir=Path.cwd(),
            event_type=HookEventType.SESSION_START,
            definition=HookDefinition(
                hooks=[HookCommand(type=HookCommandType.COMMAND, command="slow", timeout=60)]
            ),
        )

        timeout = engine._command_timeout_seconds(
            "startup",
            deadline=time.perf_counter() + 120,
            command_timeout=hook.definition.hooks[0].timeout,
        )
        budget = engine._recommended_startup_budget_seconds([hook])

        self.assertEqual(timeout, 60)
        self.assertEqual(budget, 60)

    def test_bash_hooks_receive_session_start_input_json(self) -> None:
        engine = HookEngine()
        hook = LoadedHook(
            plugin_id="demo@test",
            plugin_name="demo",
            plugin_dir=Path.cwd(),
            event_type=HookEventType.SESSION_START,
            definition=HookDefinition(
                hooks=[HookCommand(type=HookCommandType.COMMAND, command="echo test")]
            ),
        )
        completed = Mock(return_value=Mock(returncode=0, stdout='{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"ctx"}}', stderr=""))

        with patch("src.hooks.engine.subprocess.run", completed):
            result = engine._execute_bash_hook("echo test", hook, source="startup", timeout_seconds=5)

        self.assertIsNotNone(result)
        self.assertEqual(result.additional_contexts, ["ctx"])
        _, kwargs = completed.call_args
        self.assertEqual(kwargs["input"], engine._hook_input_json("startup"))


if __name__ == "__main__":
    unittest.main()
