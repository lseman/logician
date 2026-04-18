from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from ...text_normalization import normalize_text_payload as shared_normalize_text_payload
from ..execution_runtime import get_shared_execution_runtime


def _runtime():
    return get_shared_execution_runtime()


def _normalize_text_payload(text: str, *, language_hint: str | None = None) -> str:
    normalized, _ = shared_normalize_text_payload(text, language_hint=language_hint)
    return normalized


def _preview_text(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def set_venv(venv_path: str) -> dict[str, Any]:
    path = Path(venv_path).expanduser().resolve()
    activate = path / "bin" / "activate"
    python_bin = path / "bin" / "python"

    if not path.is_dir():
        return {"status": "error", "error": f"Directory not found: {path}"}
    if not activate.exists():
        return {"status": "error", "error": f"bin/activate not found in: {path}"}

    _runtime().set_venv_path(str(path))
    return {
        "status": "ok",
        "venv_path": str(path),
        "python": str(python_bin) if python_bin.exists() else "not found",
        "message": "venv configured; all run_shell / run_python calls will use it",
    }


def set_working_directory(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        return {"status": "error", "error": f"Not a directory: {resolved}"}
    _runtime().set_cwd(str(resolved))
    return {"status": "ok", "cwd": str(resolved)}


def install_packages(packages: str, venv_path: str = "", upgrade: bool = False) -> dict[str, Any]:
    normalized = _normalize_text_payload(packages).strip()
    if not normalized:
        return {"status": "error", "error": "No packages provided"}

    try:
        package_args = shlex.split(normalized)
    except ValueError as exc:
        return {"status": "error", "error": f"Could not parse package list: {exc}"}
    if not package_args:
        return {"status": "error", "error": "No packages provided"}

    runtime = _runtime()
    active_venv = venv_path or (runtime.venv_path() or "")
    python_bin = Path(active_venv).expanduser().resolve() / "bin" / "python" if active_venv else None
    if python_bin is not None and python_bin.exists():
        command: list[str] = [str(python_bin), "-m", "pip", "install"]
    else:
        command = ["pip", "install"]
    if upgrade:
        command.append("--upgrade")
    command.extend(package_args)

    result = runtime.run_cmd(
        command,
        timeout=180,
        venv_path=active_venv or None,
        shell=False,
    )
    result["command_preview"] = _preview_text(" ".join(shlex.quote(part) for part in command))
    return result


def show_coding_config() -> dict[str, Any]:
    runtime = _runtime()
    venv = runtime.venv_path()
    cwd = runtime.cwd()

    python_bin = None
    if venv:
        candidate = Path(venv) / "bin" / "python"
        python_bin = str(candidate) if candidate.exists() else f"{venv}/bin/python (not found)"

    return {
        "status": "ok",
        "venv_path": venv or "(not set)",
        "python_bin": python_bin or "(not set)",
        "default_cwd": cwd or "(not set — uses process cwd)",
    }


def start_background_process(
    command: str,
    name: str,
    cwd: str = "",
    venv_path: str = "",
) -> dict[str, Any]:
    normalized = _normalize_text_payload(command)
    if not normalized.strip():
        return {"status": "error", "error": "command must not be empty"}
    if not str(name or "").strip():
        return {"status": "error", "error": "name must not be empty"}
    return _runtime().start_background_process(
        normalized,
        name,
        cwd=cwd or None,
        venv_path=venv_path or None,
    )


def send_input_to_process(name: str, input_text: str) -> dict[str, Any]:
    normalized = _normalize_text_payload(input_text)
    return _runtime().send_input_to_process(name, normalized)


def get_process_output(name: str, tail_lines: int = 50) -> dict[str, Any]:
    return _runtime().get_process_output(name, tail_lines=tail_lines)


def kill_process(name: str, force: bool = False) -> dict[str, Any]:
    return _runtime().kill_process(name, force=force)


def list_processes() -> dict[str, Any]:
    result = _runtime().list_processes()
    result["cwd"] = _runtime().cwd() or os.getcwd()
    return result
