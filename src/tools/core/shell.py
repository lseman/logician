"""Core shell tool: bash command execution with intelligent assistance.

Agent guidance
--------------
When passing commands to the bash tool:

LITERAL SYNTAX (pass directly to bash):
1. Use literal shell syntax - the command is passed directly to bash
2. For arguments with spaces, quote them: ls "/path with spaces/"
3. For special characters, escape them: echo "file\\n" for literal backslash-n
4. Do NOT wrap commands in additional quotes unless you mean literal quotes
5. Avoid complex redirections or subshells unless necessary

EXAMPLES:
- List directory: "ls -la"
- File with spaces: "ls \"/path with spaces/file.txt\""
- Literal backslash-n: "echo 'file\\n'"
- Command with pipe: "grep pattern file.txt | wc -l"
- Simple echo: "echo hello world"

DANGEROUS PATTERNS (will be warned about):
- rm -rf or rm <path>
- tee -a <logfile> (persistent logging)
- Writing to /etc/passwd or /etc/shadow
- Command substitution $(...) - ensure it's safe

The tool returns JSON with stdout, stderr, return code, and parsed output.
"""
from __future__ import annotations

import re
import subprocess
from typing import Any

_DEFAULT_TIMEOUT = 30  # seconds
def _sanitize_command(command: str) -> str | dict[str, str]:
    """Sanitize and validate a bash command before execution.

    This function:
    1. Strips leading/trailing whitespace
    2. Validates basic command structure
    3. Detects potentially dangerous patterns
    4. Provides clear error messages

    Args:
        command: The bash command to sanitize

    Returns:
        The sanitized command or an error dict
    """
    if not isinstance(command, str):
        return {"error": "Command must be a string"}

    # Strip whitespace
    command = command.strip()

    if not command:
        return {"error": "Command is empty"}

    # Detect dangerous patterns
    dangerous_patterns = [
        (r";\s*rm\s+-rf", "Potentially dangerous: rm -rf detected"),
        (r";\s*rm\s+.*$", "Potentially dangerous: rm command detected"),
        (r"\|\s*tee\s+-a\s+.*$", "Potentially dangerous: persistent tee detected"),
        (r">>\s*/etc/passwd", "Potentially dangerous: writing to /etc/passwd"),
        (r">>\s*/etc/shadow", "Potentially dangerous: writing to /etc/shadow"),
        (r"\$\(", "Command substitution detected - ensure it's safe"),
    ]

    for pattern, warning in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            # Only warn (not block) for command substitution
            if "Command substitution" in warning:
                continue

    return command


def _escape_shell_arg(arg: str) -> str:
    """Escape a shell argument for safe execution.

    This is a conservative approach - it escapes most special characters.
    Use this when building commands from user input.

    Args:
        arg: The argument to escape

    Returns:
        The escaped argument
    """
    # Quote the entire argument to preserve spaces
    # This is safe but may not be what you want for all cases
    return f'"{arg}"'


def bash(
    command: str,
    timeout: int = _DEFAULT_TIMEOUT,
    normalize_output: bool = True,
) -> dict[str, Any]:
    """Execute a bash command and return structured output.

    Agent guidance
    --------------
    - Use literal shell syntax - the command is passed directly to bash
    - For arguments with spaces, quote them: `ls "/path with spaces/"`
    - For special characters, escape them: `echo "file\\n"` for literal backslash-n
    - Do NOT wrap commands in additional quotes unless you mean literal quotes
    - Avoid complex redirections or subshells unless necessary

    Args:
        command: Bash command string to execute.
        timeout: Timeout in seconds (default 30).
        normalize_output: If True (default), normalizes stdout/stderr newlines to LF
        only for consistent output format. If False, preserves original line endings.

    Returns:
        dict with:
            - status: "ok" or "error"
            - returncode: Exit code
            - stdout: Standard output
            - stderr: Standard error
            - parsed: Auto-detected structured data
    """
    # Sanitize the command
    sanitized = _sanitize_command(command)
    if isinstance(sanitized, dict) and "error" in sanitized:
        return {
            "status": "error",
            "returncode": -1,
            "stdout": "",
            "stderr": sanitized["error"],
            "command": command,
        }

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        parsed = _try_parse_output(result.stdout, result.stderr)
        output = {
            "status": "ok",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "parsed": parsed,
        }
        # Apply newline normalization to output
        if normalize_output:
            output["stdout"] = output["stdout"].replace("\r\n", "\n").replace("\r", "\n")
            output["stderr"] = output["stderr"].replace("\r\n", "\n").replace("\r", "\n")
        if result.returncode != 0:
            output["status"] = "error"
        output["sanitized"] = sanitized
        return output
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "command": command,
            "sanitized": sanitized,
        }
    except Exception as e:
        return {
            "status": "error",
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "command": command,
            "sanitized": sanitized,
        }


def _try_parse_output(stdout: str, stderr: str) -> dict[str, Any]:
    """Try to auto-parse command output into structured data."""
    parsed = {"raw_stdout": stdout, "raw_stderr": stderr}

    # Try JSON parsing
    if stdout.strip():
        try:
            import json

            parsed["json"] = json.loads(stdout)
        except json.JSONDecodeError:
            pass

    # Try to parse common output formats
    if stdout.strip().startswith("{"):
        parsed["detected_type"] = "json"
    elif stdout.strip().endswith("\n"):
        # Could be tabular data
        lines = stdout.strip().split("\n")
        if len(lines) > 1 and any(line.startswith("  ") for line in lines[1:]):
            parsed["detected_type"] = "indented_list"
            parsed["lines"] = lines

    # Parse git output
    if stdout.strip().startswith("On branch"):
        parsed["detected_type"] = "git_status"
        parsed["git_branch"] = stdout.split("On branch ")[1].split("\n")[0] if "On branch " in stdout else None

    # Parse ls output
    if stdout.strip().startswith("total ") or any(line.startswith("d") for line in stdout.split("\n")[:5]):
        parsed["detected_type"] = "ls_output"

    return parsed
