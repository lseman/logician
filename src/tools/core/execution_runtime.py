from __future__ import annotations

import builtins
import io
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

_DEFAULT_CONFIG: dict[str, Any] = {
    "venv_path": None,
    "default_cwd": None,
}
_DEFAULT_BG_PROCS: dict[str, dict[str, Any]] = {}
_SHARED_RUNTIME: SharedExecutionRuntime | None = None


def _find_local_venv(start: str | None) -> str | None:
    roots: list[Path] = []
    if start:
        roots.append(Path(start).expanduser().resolve())
    else:
        roots.append(Path.cwd().resolve())

    seen: set[str] = set()
    for root in roots:
        for base in (root, *root.parents):
            key = str(base)
            if key in seen:
                continue
            seen.add(key)
            for candidate_name in (".venv", "venv"):
                candidate = base / candidate_name
                if not candidate.is_dir():
                    continue
                posix_python = candidate / "bin" / "python"
                posix_activate = candidate / "bin" / "activate"
                windows_python = candidate / "Scripts" / "python.exe"
                if posix_python.exists() or posix_activate.exists() or windows_python.exists():
                    return str(candidate.resolve())
    return None


def _preview_text(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


class SharedExecutionRuntime:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        bg_procs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._config = config if config is not None else _DEFAULT_CONFIG
        self._bg_procs = bg_procs if bg_procs is not None else _DEFAULT_BG_PROCS

    def rebind_config(self, config: dict[str, Any]) -> None:
        self._config = config

    def config(self) -> dict[str, Any]:
        return self._config

    def cwd(self) -> str | None:
        value = self._config.get("default_cwd")
        return str(value) if value else None

    def venv_path(self) -> str | None:
        value = self._config.get("venv_path")
        if value:
            return str(value)
        return _find_local_venv(self.cwd() or os.getcwd())

    def set_cwd(self, path: str | None) -> str | None:
        resolved = str(Path(path).expanduser().resolve()) if path else None
        self._config["default_cwd"] = resolved
        return resolved

    def set_venv_path(self, path: str | None) -> str | None:
        resolved = str(Path(path).expanduser().resolve()) if path else None
        self._config["venv_path"] = resolved
        return resolved

    def resolve_cwd(self, cwd: Optional[str]) -> Optional[str]:
        if cwd:
            return str(Path(cwd).expanduser().resolve())
        return self.cwd()

    def resolve_path(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute() and self.cwd():
            candidate = Path(self.cwd() or ".") / candidate
        return candidate.resolve()

    def build_shell_prefix(self, venv_path: Optional[str]) -> str:
        venv = venv_path or self.venv_path()
        if not venv:
            return ""
        activate = Path(venv).expanduser() / "bin" / "activate"
        if not activate.exists():
            return ""
        return f". {shlex.quote(str(activate))} && "

    def build_env(self, venv_path: Optional[str]) -> dict[str, str]:
        env = os.environ.copy()
        venv = venv_path or self.venv_path()
        if not venv:
            return env

        root = Path(venv).expanduser().resolve()
        bin_dir = root / "bin"
        scripts_dir = root / "Scripts"
        tool_dir = bin_dir if bin_dir.exists() else scripts_dir
        env["VIRTUAL_ENV"] = str(root)
        if tool_dir.exists():
            current_path = env.get("PATH", "")
            env["PATH"] = (
                f"{tool_dir}{os.pathsep}{current_path}" if current_path else str(tool_dir)
            )
        return env

    def run_cmd(
        self,
        command: str | list[str] | tuple[str, ...],
        cwd: Optional[str] = None,
        timeout: int = 60,
        venv_path: Optional[str] = None,
        shell: bool | None = None,
    ) -> dict[str, Any]:
        effective_shell = isinstance(command, str) if shell is None else bool(shell)
        run_command: str | list[str] | tuple[str, ...] = command
        if effective_shell and isinstance(command, (list, tuple)):
            run_command = " ".join(shlex.quote(str(part)) for part in command)
        display_command = (
            run_command
            if isinstance(run_command, str)
            else " ".join(shlex.quote(str(part)) for part in run_command)
        )
        resolved_cwd = self.resolve_cwd(cwd)
        env = self.build_env(venv_path)

        try:
            proc = subprocess.run(
                run_command,
                shell=effective_shell,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=resolved_cwd,
                env=env,
                stdin=subprocess.DEVNULL,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            max_chars = 8_000
            truncated = False
            if len(stdout) > max_chars:
                stdout = stdout[:max_chars] + "\n...[truncated]"
                truncated = True
            if len(stderr) > max_chars:
                stderr = stderr[:max_chars] + "\n...[truncated]"
                truncated = True

            return {
                "status": "ok" if proc.returncode == 0 else "error",
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "command": display_command,
                "cwd": resolved_cwd or os.getcwd(),
                "truncated": truncated,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "command": display_command,
                "cwd": resolved_cwd or os.getcwd(),
                "truncated": False,
            }
        except Exception as exc:
            return {
                "status": "error",
                "exit_code": -1,
                "stdout": "",
                "stderr": str(exc),
                "command": display_command,
                "cwd": resolved_cwd or os.getcwd(),
                "truncated": False,
            }

    def start_background_process(
        self,
        command: str,
        name: str,
        *,
        cwd: str | None = None,
        venv_path: str | None = None,
    ) -> dict[str, Any]:
        if name in self._bg_procs and self._bg_procs[name]["proc"].poll() is None:
            return {"status": "error", "error": f"Process '{name}' is already running"}

        resolved_cwd = self.resolve_cwd(cwd)
        env = self.build_env(venv_path)

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                cwd=resolved_cwd,
                env=env,
                bufsize=1,
                universal_newlines=True,
            )
            buf = io.StringIO()
            lock = threading.Lock()

            def _reader() -> None:
                stdout = proc.stdout
                if stdout is None:
                    return
                for line in stdout:
                    with lock:
                        buf.write(line)

            thread = threading.Thread(target=_reader, daemon=True)
            thread.start()

            self._bg_procs[name] = {
                "proc": proc,
                "buf": buf,
                "lock": lock,
                "cmd": command,
                "thread": thread,
                "cwd": resolved_cwd or os.getcwd(),
                "started_at": time.time(),
            }

            time.sleep(0.3)
            if proc.poll() is not None:
                with lock:
                    output = buf.getvalue()
                del self._bg_procs[name]
                return {
                    "status": "error",
                    "error": f"Process exited immediately (code {proc.returncode})",
                    "output": _preview_text(output, 4000),
                }

            return {
                "status": "ok",
                "name": name,
                "pid": proc.pid,
                "command": command,
                "cwd": resolved_cwd or os.getcwd(),
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def send_input_to_process(self, name: str, input_text: str) -> dict[str, Any]:
        if name not in self._bg_procs:
            return {"status": "error", "error": f"No process named '{name}'"}

        entry = self._bg_procs[name]
        proc = entry["proc"]
        if proc.poll() is not None:
            return {
                "status": "error",
                "error": f"Process '{name}' is not running (exit code {proc.returncode})",
            }
        if proc.stdin is None:
            return {"status": "error", "error": f"Process '{name}' has no stdin pipe"}

        normalized_input = input_text.replace("\r\n", "\n").replace("\r", "\n")
        try:
            proc.stdin.write(normalized_input)
            proc.stdin.flush()
            return {
                "status": "ok",
                "name": name,
                "bytes_written": len(normalized_input.encode("utf-8")),
                "input_preview": _preview_text(normalized_input, 200),
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def get_process_output(self, name: str, tail_lines: int = 50) -> dict[str, Any]:
        if name not in self._bg_procs:
            return {"status": "error", "error": f"No process named '{name}'"}
        if tail_lines < 0:
            return {"status": "error", "error": "tail_lines must be >= 0"}

        entry = self._bg_procs[name]
        proc = entry["proc"]
        with entry["lock"]:
            output = entry["buf"].getvalue()

        if tail_lines > 0:
            lines = output.splitlines()
            output = "\n".join(lines[-tail_lines:])

        return {
            "status": "ok",
            "name": name,
            "pid": proc.pid,
            "running": proc.poll() is None,
            "exit_code": proc.poll(),
            "output": output,
        }

    def kill_process(self, name: str, force: bool = False) -> dict[str, Any]:
        if name not in self._bg_procs:
            return {"status": "error", "error": f"No process named '{name}'"}

        entry = self._bg_procs[name]
        proc = entry["proc"]
        if proc.poll() is not None:
            del self._bg_procs[name]
            return {
                "status": "ok",
                "name": name,
                "already_exited": True,
                "exit_code": proc.returncode,
            }

        if force:
            proc.kill()
        else:
            proc.terminate()

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        del self._bg_procs[name]
        return {"status": "ok", "name": name, "exit_code": proc.returncode}

    def list_processes(self) -> dict[str, Any]:
        processes = []
        for name, entry in list(self._bg_procs.items()):
            proc = entry["proc"]
            processes.append(
                {
                    "name": name,
                    "pid": proc.pid,
                    "command": entry["cmd"],
                    "cwd": entry.get("cwd"),
                    "running": proc.poll() is None,
                    "exit_code": proc.poll(),
                    "started_at": entry.get("started_at"),
                }
            )
        return {"status": "ok", "processes": processes}


def _install_runtime_aliases(runtime: SharedExecutionRuntime) -> None:
    builtins.coding_runtime = runtime
    builtins._coding_runtime = runtime
    builtins._coding_config = runtime.config()
    builtins._run_cmd = runtime.run_cmd


def get_shared_execution_runtime(
    config: dict[str, Any] | None = None,
) -> SharedExecutionRuntime:
    global _SHARED_RUNTIME

    existing = getattr(builtins, "coding_runtime", None) or getattr(
        builtins, "_coding_runtime", None
    )
    if isinstance(existing, SharedExecutionRuntime):
        _SHARED_RUNTIME = existing
        if config is not None:
            existing.rebind_config(config)
        _install_runtime_aliases(existing)
        return existing

    effective_config = config if config is not None else _DEFAULT_CONFIG
    if _SHARED_RUNTIME is None:
        _SHARED_RUNTIME = SharedExecutionRuntime(effective_config, _DEFAULT_BG_PROCS)
    else:
        _SHARED_RUNTIME.rebind_config(effective_config)

    _install_runtime_aliases(_SHARED_RUNTIME)
    return _SHARED_RUNTIME


__all__ = [
    "SharedExecutionRuntime",
    "_find_local_venv",
    "get_shared_execution_runtime",
]
