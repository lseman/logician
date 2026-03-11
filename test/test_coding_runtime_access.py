import builtins
import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from skills.coding.bootstrap.runtime_access import get_coding_runtime


class CodingRuntimeAccessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._had_coding_runtime = hasattr(builtins, "coding_runtime")
        self._had_private_runtime = hasattr(builtins, "_coding_runtime")
        self._had_config = hasattr(builtins, "_coding_config")
        self._had_run_cmd = hasattr(builtins, "_run_cmd")
        self._coding_runtime = getattr(builtins, "coding_runtime", None)
        self._private_runtime = getattr(builtins, "_coding_runtime", None)
        self._coding_config = getattr(builtins, "_coding_config", None)
        self._run_cmd = getattr(builtins, "_run_cmd", None)

    def tearDown(self) -> None:
        if self._had_coding_runtime:
            builtins.coding_runtime = self._coding_runtime
        elif hasattr(builtins, "coding_runtime"):
            delattr(builtins, "coding_runtime")

        if self._had_private_runtime:
            builtins._coding_runtime = self._private_runtime
        elif hasattr(builtins, "_coding_runtime"):
            delattr(builtins, "_coding_runtime")

        if self._had_config:
            builtins._coding_config = self._coding_config
        elif hasattr(builtins, "_coding_config"):
            delattr(builtins, "_coding_config")

        if self._had_run_cmd:
            builtins._run_cmd = self._run_cmd
        elif hasattr(builtins, "_run_cmd"):
            delattr(builtins, "_run_cmd")

    def test_prefers_explicit_runtime_from_module_globals(self) -> None:
        class DummyRuntime:
            pass

        explicit = DummyRuntime()
        builtins.coding_runtime = None
        runtime = get_coding_runtime({"coding_runtime": explicit})
        self.assertIs(runtime, explicit)

    def test_legacy_runtime_uses_builtins_config_and_runner(self) -> None:
        calls: list[dict] = []
        builtins._coding_config = {"default_cwd": ".", "venv_path": None}

        def mock_run_cmd(command, cwd=None, timeout=None, venv_path=None):
            calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "timeout": timeout,
                    "venv_path": venv_path,
                }
            )
            return {"status": "ok", "exit_code": 0, "stdout": "", "stderr": ""}

        builtins._run_cmd = mock_run_cmd
        runtime = get_coding_runtime({})

        resolved = runtime.resolve_path("dummy.py")
        self.assertEqual(resolved, (Path(".").resolve() / "dummy.py").resolve())

        cwd = runtime.set_cwd(".")
        self.assertEqual(cwd, str(Path(".").resolve()))
        self.assertEqual(runtime.cwd(), str(Path(".").resolve()))

        result = runtime.run_cmd("echo hi", cwd=".", timeout=5)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            calls,
            [
                {
                    "command": "echo hi",
                    "cwd": ".",
                    "timeout": 5,
                    "venv_path": None,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
