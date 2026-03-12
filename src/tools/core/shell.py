"""Core shell tool: bash command execution."""
from __future__ import annotations

import subprocess

_DEFAULT_TIMEOUT = 30  # seconds


def bash(command: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Execute a bash command and return its output (stdout + stderr combined).

    Runs in the current working directory. Timeout in seconds (default 30).
    Returns output as a string. Non-zero exit codes are included in the output.

    Args:
        command: Bash command string to execute.
        timeout: Timeout in seconds (default 30).

    Returns:
        Command output (stdout + stderr combined) as a string, including exit code
        if non-zero.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() if output.strip() else f"[exit code: {result.returncode}]"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s: {command}"
    except Exception as e:
        return f"Error running command: {e}"
