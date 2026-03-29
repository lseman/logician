from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from skills.coding.bootstrap.runtime_access import get_coding_runtime, tool
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
    "when_not_to_use": [
        "the task is just reading or editing files and does not need execution"
    ],
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

# Background process registry: name -> {proc, buf, lock, cmd, thread, cwd, started_at}
_bg_procs: dict[str, dict[str, Any]] = {}


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


def _resolve_python_executable(venv_path: str = "") -> str:
    """Resolve the Python executable to use.

    Priority:
    1. explicit venv_path argument
    2. configured runtime venv
    3. current interpreter
    """
    venv = venv_path or _runtime().venv_path()
    if venv:
        python_bin = Path(venv).expanduser().resolve() / "bin" / "python"
        if python_bin.exists():
            return str(python_bin)
    return sys.executable


def _resolve_effective_cwd(cwd: str = "") -> str | None:
    """Resolve the working directory using explicit cwd or runtime default."""
    return _runtime().resolve_cwd(cwd or None)


def _preview_text(text: str, limit: int = 300) -> str:
    """Return a safe short preview for logs and tool output."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _detect_python_truncation_warning(code: str) -> str | None:
    """Return a warning when Python code looks incomplete."""
    import ast
    import re

    tail = code.rstrip()[-400:]

    placeholder_rx = re.compile(
        r"(#\s*\.{2,}"
        r"|#\s*\[?\.\.\.\]?"
        r"|#\s*(rest|remainder)\s+of"
        r"|#\s*more\s+(code|logic|impl)"
        r"|#\s*implementation\s+continues)",
        re.IGNORECASE | re.MULTILINE,
    )
    if placeholder_rx.search(tail):
        return (
            "Code may be incomplete because it ends with a placeholder-style comment."
        )

    if code.strip():
        try:
            ast.parse(code)
        except SyntaxError as exc:
            msg = str(exc).lower()
            if any(token in msg for token in ("unexpected eof", "eof while", "was never closed")):
                return f"Code may be truncated or incomplete: {exc}"

    return None


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
    p = Path(venv_path).expanduser().resolve()
    activate = p / "bin" / "activate"
    python_bin = p / "bin" / "python"

    if not p.is_dir():
        return _safe_json({"status": "error", "error": f"Directory not found: {p}"})
    if not activate.exists():
        return _safe_json(
            {"status": "error", "error": f"bin/activate not found in: {p}"}
        )

    _runtime().set_venv_path(str(p))
    return _safe_json(
        {
            "status": "ok",
            "venv_path": str(p),
            "python": str(python_bin) if python_bin.exists() else "not found",
            "message": "venv configured; all run_shell / run_python calls will use it",
        }
    )


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
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        return _safe_json({"status": "error", "error": f"Not a directory: {p}"})

    _runtime().set_cwd(str(p))
    return _safe_json({"status": "ok", "cwd": str(p)})


@tool
def run_python(
    code: str,
    cwd: str = "",
    timeout: int = 60,
    venv_path: str = "",
    normalize_output: bool = True,
) -> str:
    """Execute a Python snippet in a fresh subprocess.

    Use when
    --------
    Run a short Python program to test imports, validate logic, inspect runtime
    behavior, or execute project code without creating a persistent session.

    Agent guidance
    --------------
    This tool is multiline-robust. It normalizes common agent formatting issues:
    - strips outer markdown fences
    - decodes obvious outer JSON-string wrapping
    - decodes escaped newlines when the payload is clearly a literal string
    - normalizes CRLF/LF safely

    Best practices
    --------------
    - Pass real Python source code, not a shell command.
    - Prefer actual newlines over literal "\\n".
    - This is stateless per call; nothing persists between invocations except any
      side effects caused by the script itself.

    Args:
        code: Python source code to execute.
        cwd: Working directory for the subprocess.
        timeout: Max seconds before the process is killed.
        venv_path: Virtualenv to use. Overrides configured default.
        normalize_output: If True (default), normalizes stdout/stderr newlines to LF
        only for consistent output format. If False, preserves original line endings.

    Returns:
        JSON string with:
        - status
        - exit_code
        - stdout
        - stderr
        - python
        - cwd
        - code_preview
        Optional:
        - warning
        - temp_file
    """
    normalized_code = _normalize_text_payload(code, language_hint="python")
    warning = _detect_python_truncation_warning(normalized_code)

    python_exe = _resolve_python_executable(venv_path)
    resolved_cwd = _resolve_effective_cwd(cwd)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
            newline="\n",
        ) as tmp:
            tmp.write(normalized_code)
            tmp.flush()
            tmp_path = tmp.name

        cmd = [python_exe, tmp_path]
        proc = subprocess.run(
            cmd,
            cwd=resolved_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=os.environ.copy(),
        )

        result: dict[str, Any] = {
            "status": "ok" if proc.returncode == 0 else "error",
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "python": python_exe,
            "cwd": resolved_cwd or os.getcwd(),
            "temp_file": tmp_path,
            "code_preview": _preview_text(normalized_code, 500),
        }
        # Apply newline normalization to output
        if normalize_output:
            result["stdout"] = result["stdout"].replace("\r\n", "\n").replace("\r", "\n")
            result["stderr"] = result["stderr"].replace("\r\n", "\n").replace("\r", "\n")
        if warning:
            result["warning"] = warning
        return _safe_json(result)

    except subprocess.TimeoutExpired as exc:
        result = {
            "status": "error",
            "error": f"Python execution timed out after {timeout}s",
            "exit_code": None,
            "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
            "stderr": exc.stderr if isinstance(exc.stderr, str) else "",
            "python": python_exe,
            "cwd": resolved_cwd or os.getcwd(),
            "code_preview": _preview_text(normalized_code, 500),
        }
        # Apply newline normalization to output
        if normalize_output:
            result["stdout"] = result["stdout"].replace("\r\n", "\n").replace("\r", "\n")
            result["stderr"] = result["stderr"].replace("\r\n", "\n").replace("\r", "\n")
        if warning:
            result["warning"] = warning
        return _safe_json(result)

    except Exception as exc:
        result = {
            "status": "error",
            "error": str(exc),
            "python": python_exe,
            "cwd": resolved_cwd or os.getcwd(),
            "code_preview": _preview_text(normalized_code, 500),
        }
        # Apply newline normalization to output
        if normalize_output:
            result["stdout"] = result["stdout"].replace("\r\n", "\n").replace("\r", "\n")
            result["stderr"] = result["stderr"].replace("\r\n", "\n").replace("\r", "\n")
        if warning:
            result["warning"] = warning
        return _safe_json(result)

    finally:
        try:
            if "tmp_path" in locals():
                Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


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
    packages = _normalize_text_payload(packages).strip()
    if not packages:
        return _safe_json({"status": "error", "error": "No packages provided"})

    upgrade_flag = "--upgrade " if upgrade else ""
    cmd = f"pip install {upgrade_flag}{packages}"
    result = _runtime().run_cmd(
        cmd,
        timeout=180,
        venv_path=venv_path or None,
    )
    result["command_preview"] = _preview_text(cmd)
    return _safe_json(result)


@tool
def show_coding_config() -> str:
    """Show the currently configured venv and working directory.

    Returns:
        JSON string with active execution configuration.
    """
    venv = _runtime().venv_path()
    cwd = _runtime().cwd()

    python_bin = None
    if venv:
        pb = Path(venv) / "bin" / "python"
        python_bin = str(pb) if pb.exists() else f"{venv}/bin/python (not found)"

    return _safe_json(
        {
            "status": "ok",
            "venv_path": venv or "(not set)",
            "python_bin": python_bin or "(not set)",
            "default_cwd": cwd or "(not set — uses process cwd)",
        }
    )


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
    global _bg_procs

    command = _normalize_text_payload(command)

    if name in _bg_procs and _bg_procs[name]["proc"].poll() is None:
        return _safe_json(
            {"status": "error", "error": f"Process '{name}' is already running"}
        )

    prefix = _runtime().build_shell_prefix(venv_path or None)
    full_cmd = prefix + command if prefix else command
    resolved_cwd = _resolve_effective_cwd(cwd)

    try:
        proc = subprocess.Popen(
            full_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            text=True,
            cwd=resolved_cwd,
            env=os.environ.copy(),
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

        _bg_procs[name] = {
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
                out = buf.getvalue()
            del _bg_procs[name]
            return _safe_json(
                {
                    "status": "error",
                    "error": f"Process exited immediately (code {proc.returncode})",
                    "output": out[:4000],
                }
            )

        return _safe_json(
            {
                "status": "ok",
                "name": name,
                "pid": proc.pid,
                "command": command,
                "cwd": resolved_cwd or os.getcwd(),
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


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
    global _bg_procs

    if name not in _bg_procs:
        return _safe_json({"status": "error", "error": f"No process named '{name}'"})

    entry = _bg_procs[name]
    proc = entry["proc"]

    if proc.poll() is not None:
        return _safe_json(
            {
                "status": "error",
                "error": f"Process '{name}' is not running (exit code {proc.returncode})",
            }
        )

    if proc.stdin is None:
        return _safe_json(
            {"status": "error", "error": f"Process '{name}' has no stdin pipe"}
        )

    normalized_input = _normalize_text_payload(input_text)
    normalized_input = normalized_input.replace("\r\n", "\n").replace("\r", "\n")

    try:
        proc.stdin.write(normalized_input)
        proc.stdin.flush()
        return _safe_json(
            {
                "status": "ok",
                "name": name,
                "bytes_written": len(normalized_input.encode("utf-8")),
                "input_preview": _preview_text(normalized_input, 200),
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


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
    global _bg_procs

    if name not in _bg_procs:
        return _safe_json({"status": "error", "error": f"No process named '{name}'"})

    entry = _bg_procs[name]
    proc = entry["proc"]

    with entry["lock"]:
        output = entry["buf"].getvalue()

    if tail_lines > 0:
        lines = output.splitlines()
        output = "\n".join(lines[-tail_lines:])

    return _safe_json(
        {
            "status": "ok",
            "name": name,
            "pid": proc.pid,
            "running": proc.poll() is None,
            "exit_code": proc.poll(),
            "output": output,
        }
    )


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
    global _bg_procs

    if name not in _bg_procs:
        return _safe_json({"status": "error", "error": f"No process named '{name}'"})

    entry = _bg_procs[name]
    proc = entry["proc"]

    if proc.poll() is not None:
        del _bg_procs[name]
        return _safe_json(
            {
                "status": "ok",
                "name": name,
                "already_exited": True,
                "exit_code": proc.returncode,
            }
        )

    if force:
        proc.kill()
    else:
        proc.terminate()

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    del _bg_procs[name]
    return _safe_json({"status": "ok", "name": name, "exit_code": proc.returncode})


@tool
def list_processes() -> str:
    """List tracked background processes for this session.

    Returns:
        JSON string describing all known background processes and whether each is
        still running.
    """
    global _bg_procs

    result = []
    for name, entry in list(_bg_procs.items()):
        proc = entry["proc"]
        result.append(
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
    return _safe_json({"status": "ok", "processes": result})


@tool
def list_installed_packages(venv_path: str = "") -> str:
    """List installed pip packages as structured JSON.

    Use when
    --------
    Inspect environment contents before installing dependencies or checking imports.

    Args:
        venv_path: Virtualenv path overriding the configured default.

    Returns:
        JSON string with package list.
    """
    cmd = "pip list --format=json"
    r = _runtime().run_cmd(cmd, timeout=30, venv_path=venv_path or None)

    packages = []
    parse_error = None
    try:
        raw = (r.get("stdout") or "").strip()
        packages = json.loads(raw) if raw else []
    except Exception as exc:
        parse_error = str(exc)

    return _safe_json(
        {
            "status": "ok" if r.get("exit_code") == 0 else "error",
            "count": len(packages),
            "packages": packages,
            "parse_error": parse_error,
            "stdout": r.get("stdout", ""),
            "stderr": r.get("stderr", ""),
        }
    )


@tool
def check_imports(modules: str, venv_path: str = "") -> str:
    """Check whether one or more Python modules can be imported.

    Use when
    --------
    Verify dependencies before running larger code.

    Agent guidance
    --------------
    - Pass space-separated top-level module names, such as:
      "numpy pandas torch"
    - This uses run_python-like normalization so escaped newlines and fenced text
      do not break execution if the module list came from a noisy payload.

    Args:
        modules: Space-separated module names.
        venv_path: Virtualenv to use.

    Returns:
        JSON string mapping each module to true/false plus raw execution output.
    """
    modules = _normalize_text_payload(modules)
    names = [m.strip() for m in modules.split() if m.strip()]
    if not names:
        return _safe_json({"status": "error", "error": "No module names provided"})

    python_exe = _resolve_python_executable(venv_path)

    code_lines = [
        "import importlib",
        "import json",
        "results = {}",
    ]
    for name in names:
        code_lines.extend(
            [
                f"try:",
                f"    importlib.import_module({name!r})",
                f"    results[{name!r}] = True",
                f"except Exception:",
                f"    results[{name!r}] = False",
            ]
        )
    code_lines.append("print(json.dumps(results, ensure_ascii=False))")
    code = "\n".join(code_lines)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
            newline="\n",
        ) as tmp:
            tmp.write(code)
            tmp.flush()
            tmp_path = tmp.name

        proc = subprocess.run(
            [python_exe, tmp_path],
            capture_output=True,
            text=True,
            timeout=20,
            encoding="utf-8",
            errors="replace",
            env=os.environ.copy(),
        )

        results: dict[str, bool] = {}
        try:
            results = json.loads((proc.stdout or "").strip())
        except Exception:
            for name in names:
                results[name] = False

        for name in names:
            results.setdefault(name, False)

        return _safe_json(
            {
                "status": "ok" if proc.returncode == 0 else "error",
                "importable": results,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode,
                "python": python_exe,
            }
        )

    except subprocess.TimeoutExpired as exc:
        return _safe_json(
            {
                "status": "error",
                "error": "Import check timed out after 20s",
                "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
                "stderr": exc.stderr if isinstance(exc.stderr, str) else "",
                "python": python_exe,
            }
        )

    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc), "python": python_exe})

    finally:
        try:
            if "tmp_path" in locals():
                Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


__tools__ = [
    run_shell,
    set_venv,
    set_working_directory,
    run_python,
    install_packages,
    show_coding_config,
    start_background_process,
    send_input_to_process,
    get_process_output,
    kill_process,
    list_processes,
    list_installed_packages,
    check_imports,
]
