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
from skills.coding.bootstrap.runtime_access import get_coding_runtime, tool

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
    "workflow": [
        "Prefer the narrowest command that answers the question.",
        "Set cwd or venv explicitly when project state matters.",
        "Use background processes only for long-running servers or watchers.",
        "Follow execution with quality, explore, or git when needed.",
    ],
}

# Background process registry: name → {proc, buf, lock, cmd, thread}
_bg_procs: dict = {}


def _runtime():
    return get_coding_runtime(globals())


@tool
def run_shell(
    command: str, cwd: str = "", timeout: int = 60, venv_path: str = ""
) -> str:
    """Use when: Execute a shell command — run tests, install packages, call git, etc.

    Triggers: run command, execute, shell, terminal, bash, test, install, git, make, pytest, pip.
    Avoid when: You only need to read or write a file — use read_file/write_file instead.
    Inputs:
      command (str, required): Shell command string to execute.
      cwd (str, optional): Working directory (default: configured default or current dir).
      timeout (int, optional): Seconds before the command is killed (default 60).
      venv_path (str, optional): Path to a virtualenv dir to activate before running.
    Returns: JSON with exit_code, stdout, stderr.
    Side effects: Runs a subprocess; can modify filesystem, network, etc.
    """
    result = _runtime().run_cmd(
        command,
        cwd=cwd or None,
        timeout=timeout,
        venv_path=venv_path or None,
    )
    return _safe_json(result)


@tool
def set_venv(venv_path: str) -> str:
    """Use when: Configure which Python venv all subsequent shell/python commands use.

    Triggers: activate venv, use virtualenv, set environment, configure python env.
    Avoid when: You want to run a one-off command in a specific venv — pass venv_path to run_shell.
    Inputs:
      venv_path (str, required): Absolute path to the virtualenv root (e.g. "/project/.venv").
    Returns: JSON confirming the venv was found and set.
    Side effects: Updates the shared coding runtime state for this session.
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
    """Use when: Change the base directory used by run_shell and run_python.

    Triggers: change directory, set cwd, working directory, cd.
    Avoid when: You want one-off cwd — pass cwd to run_shell directly.
    Inputs:
      path (str, required): Absolute or relative path to the new working directory.
    Returns: JSON with resolved absolute path.
    Side effects: Updates the shared coding runtime state for this session.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        return _safe_json({"status": "error", "error": f"Not a directory: {p}"})
    _runtime().set_cwd(str(p))
    return _safe_json({"status": "ok", "cwd": str(p)})


@tool
def run_python(code: str, cwd: str = "", timeout: int = 60, venv_path: str = "") -> str:
    """Use when: Run a Python snippet to test a function, check imports, or validate logic.

    Triggers: run python, execute python, test code, python snippet, check output, import test.
    Avoid when: You need a persistent interactive session — this is stateless per call.
    Inputs:
      code (str, required): Python source code to execute.
      cwd (str, optional): Working directory.
      timeout (int, optional): Seconds before kill (default 60).
      venv_path (str, optional): Virtualenv to use (overrides configured default).
    Returns: JSON with exit_code, stdout, stderr.
    Side effects: Spawns a subprocess; any side effects of the code apply.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        # Use the venv python if available, else fall back to sys.executable
        venv = venv_path or _runtime().venv_path()
        if venv:
            python_bin = Path(venv) / "bin" / "python"
            if python_bin.exists():
                cmd = f"{python_bin} {tmp_path}"
            else:
                cmd = f"python {tmp_path}"
        else:
            cmd = f"{sys.executable} {tmp_path}"

        result = _runtime().run_cmd(
            cmd,
            cwd=cwd or None,
            timeout=timeout,
            venv_path=None,  # already baked into cmd
        )
    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass

    result["code_preview"] = code[:200] + "..." if len(code) > 200 else code
    return _safe_json(result)


@tool
def install_packages(packages: str, venv_path: str = "", upgrade: bool = False) -> str:
    """Use when: Install one or more pip packages into the active virtualenv.

    Triggers: install package, pip install, add dependency, install library, missing module.
    Avoid when: The package is already installed — check with run_shell first.
    Inputs:
      packages (str, required): Space-separated package names (e.g. "numpy pandas scikit-learn").
      venv_path (str, optional): Virtualenv path (overrides configured default).
      upgrade (bool, optional): Pass --upgrade to pip (default False).
    Returns: JSON with pip output.
    Side effects: Installs packages into the virtualenv.
    """
    upgrade_flag = "--upgrade " if upgrade else ""
    cmd = f"pip install {upgrade_flag}{packages}"
    result = _runtime().run_cmd(
        cmd,
        timeout=180,
        venv_path=venv_path or None,
    )
    return _safe_json(result)


@tool
def show_coding_config() -> str:
    """Use when: Check what venv and working directory are currently configured.

    Triggers: show config, current venv, which python, coding config, environment info.
    Avoid when: N/A.
    Inputs: None.
    Returns: JSON with venv_path and default_cwd.
    Side effects: Read-only.
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
    command: str, name: str, cwd: str = "", venv_path: str = ""
) -> str:
    """Use when: Launch a server, dev watcher, or any process that should run in the background.

    Triggers: start server, run in background, watch, dev server, launch process, keep running.
    Avoid when: You need the output immediately and the command exits quickly — use run_shell.
    Inputs:
      command (str, required): Shell command to run.
      name (str, required): Unique label to identify this process later.
      cwd (str, optional): Working directory.
      venv_path (str, optional): Virtualenv to activate.
    Returns: JSON with process PID and name.
    Side effects: Spawns a background subprocess; stays alive until kill_process is called.
    """
    global _bg_procs
    if name in _bg_procs and _bg_procs[name]["proc"].poll() is None:
        return _safe_json(
            {"status": "error", "error": f"Process '{name}' is already running"}
        )

    prefix = _runtime().build_shell_prefix(venv_path or None)
    full_cmd = prefix + command if prefix else command
    resolved_cwd = _runtime().resolve_cwd(cwd or None)

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
        )
        buf = io.StringIO()
        lock = threading.Lock()

        def _reader():
            for line in proc.stdout:  # type: ignore[union-attr]
                with lock:
                    buf.write(line)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        _bg_procs[name] = {
            "proc": proc,
            "buf": buf,
            "lock": lock,
            "cmd": command,
            "thread": t,
        }
        time.sleep(0.3)  # brief pause to catch immediate crash
        if proc.poll() is not None:
            with lock:
                out = buf.getvalue()
            return _safe_json(
                {
                    "status": "error",
                    "error": f"Process exited immediately (code {proc.returncode})",
                    "output": out[:2000],
                }
            )

        return _safe_json(
            {"status": "ok", "name": name, "pid": proc.pid, "command": command}
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


@tool
def send_input_to_process(name: str, input_text: str) -> str:
    """Use when: Interact with a REPL, shell, or interactive prompt running in the background.

    Triggers: send input, type into process, write to process, stdin, interact.
    Avoid when: The process doesn't read from stdin.
    Inputs:
      name (str, required): Process label used in start_background_process.
      input_text (str, required): Text to send. Ensure to include newlines (\\n) if executing a command.
    Returns: JSON with status.
    Side effects: Writes to the process's standard input.
    """
    global _bg_procs
    if name not in _bg_procs:
        return _safe_json({"status": "error", "error": f"No process named '{name}'"})

    entry = _bg_procs[name]
    proc = entry["proc"]
    
    if proc.poll() is not None:
        return _safe_json({"status": "error", "error": f"Process '{name}' is not running (exit code {proc.returncode})"})
        
    if proc.stdin is None:
        return _safe_json({"status": "error", "error": f"Process '{name}' has no stdin pipe"})

    try:
        proc.stdin.write(input_text)
        proc.stdin.flush()
        return _safe_json({"status": "ok", "name": name, "bytes_written": len(input_text)})
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


@tool
def get_process_output(name: str, tail_lines: int = 50) -> str:
    """Use when: Check what a background process (server, watcher) has printed so far.

    Triggers: check server output, get logs, process output, what did it print, server log.
    Avoid when: The process has already exited — its output may be partial.
    Inputs:
      name (str, required): Process label used in start_background_process.
      tail_lines (int, optional): Return only the last N lines (default 50, 0 = all).
    Returns: JSON with running status, PID, and captured output.
    Side effects: Read-only; does not reset the buffer.
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
    """Use when: Stop a background server or watcher that is no longer needed.

    Triggers: stop server, kill process, terminate, shut down, stop background.
    Avoid when: You want to check its output first — use get_process_output.
    Inputs:
      name (str, required): Process label used in start_background_process.
      force (bool, optional): Send SIGKILL instead of SIGTERM (default False).
    Returns: JSON with exit code.
    Side effects: Terminates the process.
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
    """Use when: Check what background processes are running in this session.

    Triggers: list processes, running processes, what's running, show background jobs.
    Avoid when: N/A.
    Inputs: None.
    Returns: JSON with list of tracked processes and their running state.
    Side effects: Read-only.
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
                "running": proc.poll() is None,
                "exit_code": proc.poll(),
            }
        )
    return _safe_json({"status": "ok", "processes": result})


@tool
def list_installed_packages(venv_path: str = "") -> str:
    """Use when: Check what packages are installed before installing or importing.

    Triggers: list packages, installed packages, pip list, what's installed, pip freeze.
    Avoid when: You know the package is definitely installed.
    Inputs:
      venv_path (str, optional): Virtualenv path (overrides configured default).
    Returns: JSON with list of {name, version}.
    Side effects: Read-only.
    """
    cmd = "pip list --format=json 2>&1"
    r = _runtime().run_cmd(cmd, timeout=30, venv_path=venv_path or None)
    packages = []
    try:
        raw = r["stdout"].strip()
        start = raw.find("[")
        if start != -1:
            packages = json.loads(raw[start:])
    except Exception:
        pass
    return _safe_json(
        {
            "status": "ok" if r["exit_code"] == 0 else "error",
            "count": len(packages),
            "packages": packages,
        }
    )


@tool
def check_imports(modules: str, venv_path: str = "") -> str:
    """Use when: Verify that required imports are available before running code.

    Triggers: check import, can I import, is module available, missing module, importable.
    Avoid when: You want a full package list — use list_installed_packages.
    Inputs:
      modules (str, required): Space-separated module names to check (e.g. "numpy pandas torch").
      venv_path (str, optional): Virtualenv to check.
    Returns: JSON dict mapping each module name to true/false.
    Side effects: Read-only; spawns a brief subprocess.
    """
    names = [m.strip() for m in modules.split() if m.strip()]
    if not names:
        return _safe_json({"status": "error", "error": "No module names provided"})

    checks = "; ".join(f"print('{n}:ok') if __import__('{n}') else None" for n in names)
    code = "\n".join(
        [
            "import sys",
            *[
                f"\ntry:\n    import {n}\n    print('{n}:ok')\nexcept ImportError:\n    print('{n}:missing')"
                for n in names
            ],
        ]
    )
    venv = venv_path or _runtime().venv_path()
    python = (
        str(Path(venv) / "bin" / "python")
        if venv and (Path(venv) / "bin" / "python").exists()
        else sys.executable
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        r = _runtime().run_cmd(f"{python} {tmp_path}", timeout=20, venv_path=None)
    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass

    results = {}
    for line in r["stdout"].splitlines():
        if ":ok" in line:
            results[line.split(":")[0]] = True
        elif ":missing" in line:
            results[line.split(":")[0]] = False
    # fill any unparsed names
    for n in names:
        if n not in results:
            results[n] = False

    return _safe_json({"status": "ok", "importable": results})


__tools__ = [run_shell, set_venv, set_working_directory, run_python, install_packages, show_coding_config, start_background_process, send_input_to_process, get_process_output, kill_process, list_processes, list_installed_packages, check_imports]
