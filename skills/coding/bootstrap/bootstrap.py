from __future__ import annotations

# =============================================================================
# CODING SKILLS BOOTSTRAP
# Shared helpers for file I/O, shell execution, and venv management.
# Executed before the other coding skill modules are loaded.
# =============================================================================
from typing import Any, Optional

from src.tools.core.execution_runtime import get_shared_execution_runtime

# Marker consumed by skills_health organization audit.
_CODING_BOOTSTRAP_ONLY = True

# --------------------------------------------------------------------------
# Persistent config shared across all coding tools in this execution context
# --------------------------------------------------------------------------
_coding_config: dict[str, Any] = {
    "venv_path": None,  # absolute path to .venv dir, e.g. "/project/.venv"
    "default_cwd": None,  # working directory for shell commands; None → cwd
}
coding_runtime = get_shared_execution_runtime(_coding_config)
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
