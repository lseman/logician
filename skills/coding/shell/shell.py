from __future__ import annotations

import json
from typing import Any

from skills.coding.bootstrap.runtime_access import get_coding_runtime, tool
from src.tools.core.ProcessTool import get_process_output as core_get_process_output
from src.tools.core.ProcessTool import install_packages as core_install_packages
from src.tools.core.ProcessTool import kill_process as core_kill_process
from src.tools.core.ProcessTool import list_processes as core_list_processes
from src.tools.core.ProcessTool import send_input_to_process as core_send_input_to_process
from src.tools.core.ProcessTool import set_venv as core_set_venv
from src.tools.core.ProcessTool import set_working_directory as core_set_working_directory
from src.tools.core.ProcessTool import show_coding_config as core_show_coding_config
from src.tools.core.ProcessTool import start_background_process as core_start_background_process
from src.tools.core.PythonTool import check_imports as core_check_imports
from src.tools.core.PythonTool import list_installed_packages as core_list_installed_packages
from src.tools.core.PythonTool import run_python as core_run_python
from src.tools.text_normalization import normalize_text_payload as shared_normalize_text_payload

__skill__ = {
    "name": "Shell",
    "description": "Use for shell commands, Python execution, background processes, and environment control.",
    "aliases": ["terminal", "command line", "subprocess", "run command"],
    "triggers": [
        "run this command",
        "run the tests",
        "start the dev server",
        "execute this python snippet",
    ],
    "preferred_tools": ["run_shell", "run_python", "start_background_process"],
    "example_queries": [
        "run pytest for this package",
        "start the app locally and capture logs",
        "execute a short Python snippet against the project",
    ],
    "when_not_to_use": ["the task is just reading or editing files and does not need execution"],
    "next_skills": ["quality", "git", "explore"],
    "preferred_sequence": ["set_working_directory", "set_venv", "run_shell", "quality"],
    "entry_criteria": [
        "You need a runtime fact, command output, or environment interaction that static inspection cannot provide.",
        "The user explicitly asked to execute a command, run tests, or launch a process.",
    ],
    "decision_rules": [
        "Prefer run_python for short Python snippets and run_shell for external CLIs.",
        "Set cwd or venv explicitly when repo-local behavior matters.",
        "Use background processes only for long-running servers, watchers, or tailing logs.",
    ],
    "workflow": [
        "Prefer the narrowest command that answers the question.",
        "Set cwd or venv explicitly when project state matters.",
        "Use background processes only for long-running servers or watchers.",
        "Follow execution with quality, explore, or git when needed.",
    ],
    "failure_recovery": [
        "If a command times out, retry with a narrower scope or a shorter-lived command.",
        "If a binary or module is missing, inspect the environment before retrying blindly.",
    ],
    "exit_criteria": [
        "The command produced the fact, artifact, or process state the turn needed.",
        "Any follow-up fix or verification step is clear from the output.",
    ],
    "anti_patterns": [
        "Using shell one-liners for tasks that are clearer with dedicated file or quality tools.",
        "Starting long-running processes when a short foreground command would answer the question.",
    ],
}


def _runtime():
    """Return the shared coding runtime for this session."""
    return get_coding_runtime(globals())


def _safe_json(payload: Any) -> str:
    """Serialize a tool result to pretty JSON.

    Agent guidance
    --------------
    All tools should return machine-readable JSON text. This helper ensures:
    - non-ASCII text is preserved
    - output is stable and readable
    - unserializable objects fall back to string form
    """
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _normalize_text_payload(
    text: str,
    *,
    language_hint: str | None = None,
) -> str:
    normalized, _ = shared_normalize_text_payload(text, language_hint=language_hint)
    return normalized


def _preview_text(text: str, limit: int = 300) -> str:
    """Return a safe short preview for logs and tool output."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


@tool
def run_shell(
    command: str,
    cwd: str = "",
    timeout: int = 60,
    venv_path: str = "",
    normalize_output: bool = True,
) -> str:
    """Execute a shell command.

    Use when
    --------
    Run tests, call git, invoke CLIs, install packages, or otherwise execute a
    shell command.

    Agent guidance
    --------------
    - Prefer the narrowest command that answers the question.
    - Pass cwd when repository or project context matters.
    - Use run_python instead of embedding Python in a long shell one-liner when
      the task is primarily Python execution.

    Args:
        command: Shell command string to execute.
        cwd: Working directory. If omitted, uses configured runtime cwd.
        timeout: Max seconds before termination.
        venv_path: Virtualenv to activate for this command.
        normalize_output: If True (default), normalizes stdout/stderr newlines to LF
        only for consistent output format. If False, preserves original line endings.

    Returns:
        JSON string with at least:
        - exit_code
        - stdout
        - stderr
    """
    command = _normalize_text_payload(command)
    result = _runtime().run_cmd(
        command,
        cwd=cwd or None,
        timeout=timeout,
        venv_path=venv_path or None,
    )
    # Apply newline normalization to output
    if normalize_output:
        result["stdout"] = result["stdout"].replace("\r\n", "\n").replace("\r", "\n")
        result["stderr"] = result["stderr"].replace("\r\n", "\n").replace("\r", "\n")
    result["command_preview"] = _preview_text(command)
    return _safe_json(result)


@tool
def set_venv(venv_path: str) -> str:
    """Set the default virtualenv for subsequent shell and Python execution.

    Use when
    --------
    You want future run_shell/run_python calls to use a specific environment.

    Args:
        venv_path: Path to the virtualenv root, such as "/project/.venv".

    Returns:
        JSON string confirming the configured venv and Python binary.
    """
    return _safe_json(core_set_venv(venv_path))


@tool
def set_working_directory(path: str) -> str:
    """Set the default working directory for shell and Python execution.

    Use when
    --------
    You want future run_shell/run_python calls to execute from a specific project
    or repository directory.

    Args:
        path: Absolute or relative path to the target directory.

    Returns:
        JSON string with the resolved cwd.
    """
    return _safe_json(core_set_working_directory(path))


@tool
def run_python(
    code: str,
    cwd: str = "",
    timeout: int = 60,
    venv_path: str = "",
    normalize_output: bool = True,
) -> str:
    """Execute a Python snippet in a fresh subprocess."""
    return _safe_json(
        core_run_python(
            code,
            cwd=cwd,
            timeout=timeout,
            venv_path=venv_path,
            normalize_output=normalize_output,
        )
    )


@tool
def install_packages(packages: str, venv_path: str = "", upgrade: bool = False) -> str:
    """Install pip packages into the active or specified virtualenv.

    Use when
    --------
    Add missing Python dependencies needed for subsequent execution.

    Args:
        packages: Space-separated package names.
        venv_path: Virtualenv path overriding the configured default.
        upgrade: Whether to pass --upgrade.

    Returns:
        JSON string with pip command output.
    """
    return _safe_json(core_install_packages(packages, venv_path=venv_path, upgrade=upgrade))


@tool
def show_coding_config() -> str:
    """Show the currently configured venv and working directory.

    Returns:
        JSON string with active execution configuration.
    """
    return _safe_json(core_show_coding_config())


@tool
def start_background_process(
    command: str,
    name: str,
    cwd: str = "",
    venv_path: str = "",
) -> str:
    """Start a long-running process in the background.

    Use when
    --------
    Launch a dev server, file watcher, REPL, worker, or any process that should
    keep running while other tools interact with it.

    Agent guidance
    --------------
    - Use a unique name so it can be referenced later.
    - Prefer run_shell for short-lived commands.
    - Output is captured and can be read with get_process_output.

    Args:
        command: Shell command to run.
        name: Unique process label.
        cwd: Working directory.
        venv_path: Virtualenv to activate for the process.

    Returns:
        JSON string with pid and process metadata.
    """
    return _safe_json(core_start_background_process(command, name, cwd=cwd, venv_path=venv_path))


@tool
def send_input_to_process(name: str, input_text: str) -> str:
    """Send text to a background process stdin.

    Use when
    --------
    Interact with a REPL, shell, CLI prompt, or other stdin-reading process.

    Agent guidance
    --------------
    - Include trailing newlines when the target expects line-based commands.
    - Escaped text like "\\n" is normalized when clearly intended.

    Args:
        name: Background process label.
        input_text: Text to send to stdin.

    Returns:
        JSON string with write status and byte count.
    """
    return _safe_json(core_send_input_to_process(name, input_text))


@tool
def get_process_output(name: str, tail_lines: int = 50) -> str:
    """Read buffered output from a background process.

    Use when
    --------
    Inspect server logs, watcher output, or REPL responses captured so far.

    Args:
        name: Background process label.
        tail_lines: Number of trailing lines to return. Use 0 for all output.

    Returns:
        JSON string with process status and captured output.
    """
    return _safe_json(core_get_process_output(name, tail_lines=tail_lines))


@tool
def kill_process(name: str, force: bool = False) -> str:
    """Stop a tracked background process.

    Use when
    --------
    Shut down a server, watcher, or long-running command that is no longer needed.

    Args:
        name: Background process label.
        force: If True, send SIGKILL instead of SIGTERM.

    Returns:
        JSON string with exit metadata.
    """
    return _safe_json(core_kill_process(name, force=force))


@tool
def list_processes() -> str:
    """List tracked background processes for this session.

    Returns:
        JSON string describing all known background processes and whether each is
        still running.
    """
    return _safe_json(core_list_processes())


@tool
def list_installed_packages(venv_path: str = "") -> str:
    """List installed pip packages as structured JSON."""
    return _safe_json(core_list_installed_packages(venv_path=venv_path))


@tool
def check_imports(modules: str, venv_path: str = "") -> str:
    """Check whether one or more Python modules can be imported."""
    return _safe_json(core_check_imports(modules, venv_path=venv_path))


__tools__ = [
    run_shell,
]
