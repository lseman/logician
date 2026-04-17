from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
