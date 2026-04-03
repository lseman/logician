"""Core Python execution and inspection helpers."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ...text_normalization import normalize_text_payload as shared_normalize_text_payload
from ..execution_runtime import get_shared_execution_runtime


def _get_coding_runtime() -> Any | None:
    return get_shared_execution_runtime()


def _normalize_text_payload(
    text: str,
    *,
    language_hint: str | None = None,
) -> str:
    normalized, _ = shared_normalize_text_payload(text, language_hint=language_hint)
    return normalized


def _resolve_python_executable(venv_path: str = "") -> str:
    runtime = _get_coding_runtime()
    active_venv = venv_path or (
        runtime.venv_path() if runtime is not None and hasattr(runtime, "venv_path") else ""
    )
    if active_venv:
        python_bin = Path(active_venv).expanduser().resolve() / "bin" / "python"
        if python_bin.exists():
            return str(python_bin)
    return sys.executable


def _resolve_effective_cwd(cwd: str = "") -> str | None:
    runtime = _get_coding_runtime()
    if runtime is not None and hasattr(runtime, "resolve_cwd"):
        return runtime.resolve_cwd(cwd or None)
    if cwd:
        return str(Path(cwd).expanduser().resolve())
    return None


def _preview_text(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _detect_python_truncation_warning(code: str) -> str | None:
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
        return "Code may be incomplete because it ends with a placeholder-style comment."

    if code.strip():
        try:
            ast.parse(code)
        except SyntaxError as exc:
            msg = str(exc).lower()
            if any(token in msg for token in ("unexpected eof", "eof while", "was never closed")):
                return f"Code may be truncated or incomplete: {exc}"
    return None


def run_python(
    code: str,
    cwd: str = "",
    timeout: int = 60,
    venv_path: str = "",
    normalize_output: bool = True,
) -> dict[str, Any]:
    """Execute a Python snippet in a fresh subprocess."""
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

        proc = subprocess.run(
            [python_exe, tmp_path],
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
        if normalize_output:
            result["stdout"] = result["stdout"].replace("\r\n", "\n").replace("\r", "\n")
            result["stderr"] = result["stderr"].replace("\r\n", "\n").replace("\r", "\n")
        if warning:
            result["warning"] = warning
        return result
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
        if normalize_output:
            result["stdout"] = result["stdout"].replace("\r\n", "\n").replace("\r", "\n")
            result["stderr"] = result["stderr"].replace("\r\n", "\n").replace("\r", "\n")
        if warning:
            result["warning"] = warning
        return result
    except Exception as exc:
        result = {
            "status": "error",
            "error": str(exc),
            "python": python_exe,
            "cwd": resolved_cwd or os.getcwd(),
            "code_preview": _preview_text(normalized_code, 500),
        }
        if warning:
            result["warning"] = warning
        return result
    finally:
        try:
            if "tmp_path" in locals():
                Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def list_installed_packages(venv_path: str = "") -> dict[str, Any]:
    """List installed pip packages as structured JSON."""
    python_exe = _resolve_python_executable(venv_path)
    try:
        proc = subprocess.run(
            [python_exe, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
            env=os.environ.copy(),
        )
        packages: list[dict[str, Any]] = []
        parse_error = None
        try:
            raw = (proc.stdout or "").strip()
            packages = json.loads(raw) if raw else []
        except Exception as exc:
            parse_error = str(exc)

        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "count": len(packages),
            "packages": packages,
            "parse_error": parse_error,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "python": python_exe,
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "Package listing timed out after 30s",
            "packages": [],
            "count": 0,
            "python": python_exe,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "packages": [],
            "count": 0,
            "python": python_exe,
        }


def check_imports(modules: str, venv_path: str = "") -> dict[str, Any]:
    """Check whether one or more Python modules can be imported."""
    modules = _normalize_text_payload(modules)
    names = [item.strip() for item in modules.split() if item.strip()]
    if not names:
        return {"status": "error", "error": "No module names provided"}

    python_exe = _resolve_python_executable(venv_path)
    code_lines = ["import importlib", "import json", "results = {}"]
    for name in names:
        code_lines.extend(
            [
                "try:",
                f"    importlib.import_module({name!r})",
                f"    results[{name!r}] = True",
                "except Exception:",
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

        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "importable": results,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "python": python_exe,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "error",
            "error": "Import check timed out after 20s",
            "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
            "stderr": exc.stderr if isinstance(exc.stderr, str) else "",
            "python": python_exe,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "python": python_exe}
    finally:
        try:
            if "tmp_path" in locals():
                Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
