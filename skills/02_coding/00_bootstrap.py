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

# --------------------------------------------------------------------------
# Persistent config shared across all coding tools in this execution context
# --------------------------------------------------------------------------
_coding_config: dict[str, Any] = {
    "venv_path": None,  # absolute path to .venv dir, e.g. "/project/.venv"
    "default_cwd": None,  # working directory for shell commands; None → cwd
}


# --------------------------------------------------------------------------
# Internal helpers (not exposed as tools)
# --------------------------------------------------------------------------


def _resolve_cwd(cwd: Optional[str]) -> Optional[str]:
    if cwd:
        return str(Path(cwd).expanduser().resolve())
    if _coding_config["default_cwd"]:
        return _coding_config["default_cwd"]
    return None


def _build_shell_prefix(venv_path: Optional[str]) -> str:
    """Return a shell snippet that activates the venv, if any."""
    venv = venv_path or _coding_config["venv_path"]
    if not venv:
        return ""
    activate = Path(venv).expanduser() / "bin" / "activate"
    if not activate.exists():
        return ""
    return f". {shlex.quote(str(activate))} && "


def _run_cmd(
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
    prefix = _build_shell_prefix(venv_path)
    full_cmd = prefix + command if prefix else command
    resolved_cwd = _resolve_cwd(cwd)

    try:
        proc = subprocess.run(
            full_cmd,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=resolved_cwd,
            env=os.environ.copy(),
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
