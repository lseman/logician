"""Structured filesystem mutation tools."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..FileReadTool.state import clear_file_snapshot, resolve_tool_path


def mkdir(path: str, parents: bool = True, exist_ok: bool = True) -> dict[str, Any]:
    try:
        resolved = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))

    if resolved.exists():
        if not resolved.is_dir():
            return _err(f"Path exists but is not a directory: {path}")
        return {
            "status": "ok",
            "path": str(resolved),
            "created": False,
            "already_exists": True,
        }

    try:
        resolved.mkdir(parents=parents, exist_ok=exist_ok)
    except OSError as exc:
        return _err(f"Cannot create directory: {exc}")

    return {
        "status": "ok",
        "path": str(resolved),
        "created": True,
        "already_exists": False,
    }


def move_path(
    src: str,
    dst: str,
    *,
    overwrite: bool = False,
    create_parents: bool = False,
) -> dict[str, Any]:
    try:
        source = resolve_tool_path(src)
        target = resolve_tool_path(dst)
    except ValueError as exc:
        return _err(str(exc))

    if not source.exists():
        return _err(f"Source path not found: {src}")
    if source == target:
        return {
            "status": "ok",
            "src": str(source),
            "dst": str(target),
            "moved": False,
            "unchanged": True,
        }

    if create_parents:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return _err(f"Cannot create destination parent directories: {exc}")
    elif not target.parent.exists():
        return _err(f"Destination parent directory does not exist: {target.parent}")

    overwrite_result = _prepare_overwrite_target(target, overwrite=overwrite)
    if overwrite_result is not None:
        return overwrite_result

    try:
        moved = Path(shutil.move(str(source), str(target))).resolve()
    except Exception as exc:
        return _err(f"Cannot move path: {exc}")

    clear_file_snapshot(globals().get("ctx"), source)
    clear_file_snapshot(globals().get("ctx"), moved)

    return {
        "status": "ok",
        "src": str(source),
        "dst": str(moved),
        "moved": True,
        "kind": "directory" if moved.is_dir() else "file",
    }


def delete_path(
    path: str,
    *,
    recursive: bool = False,
    missing_ok: bool = False,
) -> dict[str, Any]:
    try:
        resolved = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))

    if _is_protected_delete_target(resolved):
        return _err(f"Refusing to delete protected path: {resolved}")

    if not resolved.exists():
        if missing_ok:
            return {
                "status": "ok",
                "path": str(resolved),
                "deleted": False,
                "missing": True,
            }
        return _err(f"Path not found: {path}")

    kind = "directory" if resolved.is_dir() else "file"
    try:
        if resolved.is_dir():
            if recursive:
                shutil.rmtree(resolved)
            else:
                resolved.rmdir()
        else:
            resolved.unlink()
    except OSError as exc:
        if resolved.is_dir() and not recursive:
            return _err(
                f"Directory is not empty: {path}. Re-run with recursive=true to delete it."
            )
        return _err(f"Cannot delete path: {exc}")

    clear_file_snapshot(globals().get("ctx"), resolved)
    return {
        "status": "ok",
        "path": str(resolved),
        "deleted": True,
        "kind": kind,
        "recursive": bool(recursive and kind == "directory"),
    }


def _prepare_overwrite_target(target: Path, *, overwrite: bool) -> dict[str, Any] | None:
    if not target.exists():
        return None
    if not overwrite:
        return _err(f"Destination already exists: {target}")
    try:
        if target.is_file() or target.is_symlink():
            target.unlink()
            return None
        if target.is_dir():
            next(target.iterdir())
            return _err(
                f"Destination directory is not empty: {target}. Clear it first or choose another path."
            )
    except StopIteration:
        try:
            target.rmdir()
            return None
        except OSError as exc:
            return _err(f"Cannot replace destination directory: {exc}")
    except OSError as exc:
        return _err(f"Cannot prepare destination path: {exc}")
    return None


def _is_protected_delete_target(path: Path) -> bool:
    try:
        protected = {
            Path("/").resolve(),
            Path.home().resolve(),
            Path.cwd().resolve(),
        }
    except Exception:
        protected = {Path("/").resolve()}
    return path.resolve() in protected


def _err(message: str) -> dict[str, Any]:
    return {"status": "error", "error": message}
