from __future__ import annotations

# =============================================================================
# CODING SKILLS BOOTSTRAP
# Shared helpers for file I/O, shell execution, and venv management.
# Executed before the other coding skill modules are loaded.
# =============================================================================
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Optional

# Marker consumed by skills_health organization audit.
_CODING_BOOTSTRAP_ONLY = True

# --------------------------------------------------------------------------
# Persistent config shared across all coding tools in this execution context
# --------------------------------------------------------------------------
_coding_config: dict[str, Any] = {
    "venv_path": None,  # absolute path to .venv dir, e.g. "/project/.venv"
    "default_cwd": None,  # working directory for shell commands; None → cwd
}


class CodingRuntime:
    """Shared runtime state and helpers for coding skills."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    def config(self) -> dict[str, Any]:
        return self._config

    def cwd(self) -> str | None:
        value = self._config.get("default_cwd")
        return str(value) if value else None

    def venv_path(self) -> str | None:
        value = self._config.get("venv_path")
        return str(value) if value else None

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
        p = Path(path).expanduser()
        if not p.is_absolute() and self.cwd():
            p = Path(self.cwd() or ".") / p
        return p.resolve()

    def build_shell_prefix(self, venv_path: Optional[str]) -> str:
        """Return a shell snippet that activates the venv, if any."""
        venv = venv_path or self.venv_path()
        if not venv:
            return ""
        activate = Path(venv).expanduser() / "bin" / "activate"
        if not activate.exists():
            return ""
        return f". {shlex.quote(str(activate))} && "

    def run_cmd(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 60,
        venv_path: Optional[str] = None,
        shell: bool = True,
    ) -> dict[str, Any]:
        """
        Execute a shell command and return a structured result dict.

        Returns keys: status, exit_code, stdout, stderr, command.
        stdout/stderr are truncated to 8 000 chars to avoid flooding context.
        """
        prefix = self.build_shell_prefix(venv_path)
        full_cmd = prefix + command if prefix else command
        resolved_cwd = self.resolve_cwd(cwd)

        try:
            proc = subprocess.run(
                full_cmd,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=resolved_cwd,
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            _MAX = 8_000
            truncated = False
            if len(stdout) > _MAX:
                stdout = stdout[:_MAX] + "\n...[truncated]"
                truncated = True
            if len(stderr) > _MAX:
                stderr = stderr[:_MAX] + "\n...[truncated]"
                truncated = True

            return {
                "status": "ok" if proc.returncode == 0 else "error",
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "command": full_cmd,
                "truncated": truncated,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "command": full_cmd,
                "truncated": False,
            }
        except Exception as exc:
            return {
                "status": "error",
                "exit_code": -1,
                "stdout": "",
                "stderr": str(exc),
                "command": full_cmd,
                "truncated": False,
            }


coding_runtime = CodingRuntime(_coding_config)
_coding_runtime = coding_runtime


# --------------------------------------------------------------------------
# Internal helpers (not exposed as tools)
# --------------------------------------------------------------------------


def _resolve_cwd(cwd: Optional[str]) -> Optional[str]:
    return coding_runtime.resolve_cwd(cwd)


def _build_shell_prefix(venv_path: Optional[str]) -> str:
    return coding_runtime.build_shell_prefix(venv_path)


def _run_cmd(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = 60,
    venv_path: Optional[str] = None,
    shell: bool = True,
) -> dict[str, Any]:
    return coding_runtime.run_cmd(
        command,
        cwd=cwd,
        timeout=timeout,
        venv_path=venv_path,
        shell=shell,
    )
