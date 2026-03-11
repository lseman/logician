from __future__ import annotations

import difflib
import subprocess
from pathlib import Path
from typing import Any


def resolve_path(path: str, *, base_cwd: str | None = None) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute() and base_cwd:
        p = Path(base_cwd) / p
    return p.resolve()


def run_subprocess(
    cmd: list[str], *, cwd: str | None = None, timeout: int = 30
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except FileNotFoundError:
        return {"stdout": "", "stderr": f"command not found: {cmd[0]}", "returncode": 127}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "timeout", "returncode": -1}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": -1}


def unified_diff_text(original: str, updated: str, *, label: str = "file") -> str:
    lines_a = original.splitlines(keepends=True)
    lines_b = updated.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            n=3,
        )
    )
