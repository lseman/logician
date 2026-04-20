"""Shared path validation utilities — OpenClaude-inspired input guards for file tools.

Each utility returns (value, error) tuples so callers can validate early
and return structured error payloads instead of letting exceptions propagate.
"""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# Maximum path depth to prevent abuse (e.g. /a/b/c/.../x)
_MAX_PATH_DEPTH = 100
# Characters that are suspicious but not outright disallowed
_SUSPICIOUS_CHARS = re.compile(r"[`$;&]|&&|\|\|")


def _fuzzy_ratio(left: str, right: str) -> float:
    l, r = str(left or "").strip().lower(), str(right or "").strip().lower()
    if not l or not r:
        return 0.0
    return SequenceMatcher(None, l, r).ratio()


def validate_file_path(
    path_arg: Any,
    *,
    allow_create: bool = False,
    check_readable: bool = True,
    cwd: Path | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Validate a file path argument.

    Returns (resolved_path, error_dict). If error_dict is None the path is valid.
    """
    raw = str(path_arg or "").strip()
    if not raw:
        return None, {"error": "file_path is required", "error_type": "missing_argument"}

    path = Path(raw)
    if not path.is_absolute() and cwd:
        path = cwd / path

    # Depth check
    depth = len(path.parts)
    if depth > _MAX_PATH_DEPTH:
        return None, {
            "error": f"Path depth ({depth}) exceeds maximum ({_MAX_PATH_DEPTH})",
            "error_type": "invalid_argument",
        }

    # Suspicious characters
    if _SUSPICIOUS_CHARS.search(raw) and len(raw) > 3:
        pass  # Warning only, not blocking

    if allow_create:
        parent = path.parent
        if not parent.exists():
            return None, {
                "error": f"Parent directory does not exist: {parent}",
                "error_type": "invalid_argument",
                "parent_dir": str(parent),
            }
        if path.exists():
            return None, {
                "error": f"File already exists: {path}",
                "error_type": "file_exists",
                "suggestion": "Use edit_file for existing files or set replace=True",
            }
        return str(path), None

    if not path.exists():
        similar = _find_similar_files(str(path))
        error_dict: dict[str, Any] = {
            "error": f"File not found: {path}",
            "error_type": "file_not_found",
        }
        if similar:
            error_dict["similar_files"] = similar
            error_dict["suggestion"] = f"Did you mean one of: {', '.join(similar[:3])}?"
        return None, error_dict

    if not path.is_file():
        return None, {
            "error": f"Path is not a file: {path}",
            "error_type": "not_a_file",
        }

    if check_readable and not os.access(str(path), os.R_OK):
        return None, {
            "error": f"File is not readable: {path}",
            "error_type": "permission_denied",
        }

    return str(path), None


def validate_dir_path(
    path_arg: Any,
    *,
    allow_create: bool = False,
    check_readable: bool = True,
    cwd: Path | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Validate a directory path argument."""
    raw = str(path_arg or "").strip()
    if not raw:
        return None, {"error": "path is required", "error_type": "missing_argument"}

    path = Path(raw)
    if not path.is_absolute() and cwd:
        path = cwd / path

    depth = len(path.parts)
    if depth > _MAX_PATH_DEPTH:
        return None, {
            "error": f"Path depth ({depth}) exceeds maximum ({_MAX_PATH_DEPTH})",
            "error_type": "invalid_argument",
        }

    if allow_create:
        if path.exists() and not path.is_dir():
            return None, {
                "error": f"Path exists but is not a directory: {path}",
                "error_type": "not_a_directory",
            }
        return str(path), None

    if not path.exists():
        return None, {
            "error": f"Directory not found: {path}",
            "error_type": "directory_not_found",
            "suggestion": "Try creating it first or check the path spelling",
        }

    if not path.is_dir():
        return None, {
            "error": f"Path is not a directory: {path}",
            "error_type": "not_a_directory",
        }

    if check_readable and not os.access(str(path), os.R_OK):
        return None, {
            "error": f"Directory is not readable: {path}",
            "error_type": "permission_denied",
        }

    return str(path), None


def _find_similar_files(target: str, max_results: int = 5, min_ratio: float = 0.6) -> list[str]:
    """Find similar file paths in the current directory tree."""
    # Directories to skip during search
    _SKIP_DIRS = frozenset({".venv", "venv", "node_modules", ".git", "__pycache__", ".tox", ".eggs", "dist", "build"})

    base = Path(target)
    target_name = base.name.lower()
    target_stem = base.stem.lower()
    candidates: list[tuple[float, str]] = []

    for root in [Path("."), Path("./src"), Path("./lib"), Path("./app"), Path(".")]:
        if not root.is_dir():
            continue
        try:
            count = 0
            for file_path in root.rglob("*"):
                count += 1
                # Stop after ~200 files to avoid scanning venvs
                if count > 200:
                    break
                if not file_path.is_file():
                    continue
                # Skip common non-source directories
                parts = [p for p in file_path.parts if p in _SKIP_DIRS]
                if parts:
                    continue
                rel = str(file_path)
                # Match on filename or full relative path
                name_ratio = _fuzzy_ratio(target_name, file_path.name.lower())
                path_ratio = _fuzzy_ratio(target, rel)
                ratio = max(name_ratio, path_ratio)
                if ratio >= min_ratio:
                    candidates.append((ratio, rel))
        except (PermissionError, OSError):
            continue
        if len(candidates) > 100:
            break

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in candidates[:max_results]]


def suggest_similar_files(path: str, max_results: int = 3) -> list[str]:
    """Quick fuzzy file search for error suggestions."""
    return _find_similar_files(path, max_results=max_results, min_ratio=0.7)
