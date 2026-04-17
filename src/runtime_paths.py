from __future__ import annotations

import os
import shutil
from pathlib import Path


def repo_root() -> Path:
    """Best-effort repository root for local runtime paths."""
    return Path(__file__).resolve().parents[1]


def default_state_root() -> Path:
    return (repo_root() / ".logician" / "state").resolve()


def legacy_state_root() -> Path:
    return (repo_root() / ".local" / "state").resolve()


def state_root() -> Path:
    """Directory for local runtime artifacts (db/vector/session state)."""
    raw = str(os.getenv("LOGICIAN_STATE_DIR", "") or "").strip()
    if not raw:
        # Backward-compatible fallback.
        raw = str(os.getenv("FOREBLOCKS_STATE_DIR", "") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return default_state_root()


def _merge_legacy_tree(src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if not target.exists():
            shutil.move(str(child), str(target))
            continue
        if child.is_dir() and target.is_dir():
            _merge_legacy_tree(child, target)
    try:
        src.rmdir()
    except OSError:
        pass


def _maybe_migrate_legacy_state(root: Path) -> None:
    if str(os.getenv("LOGICIAN_STATE_DIR", "") or "").strip():
        return
    if str(os.getenv("FOREBLOCKS_STATE_DIR", "") or "").strip():
        return
    if root.resolve() != default_state_root():
        return
    legacy = legacy_state_root()
    if not legacy.exists():
        return
    _merge_legacy_tree(legacy, root)
    try:
        legacy.parent.rmdir()
    except OSError:
        pass


def ensure_state_root() -> Path:
    root = state_root()
    root.mkdir(parents=True, exist_ok=True)
    _maybe_migrate_legacy_state(root)
    return root


def state_path(name: str) -> Path:
    """Return an artifact path inside the managed runtime-state directory."""
    return ensure_state_root() / str(name)


def session_db_path() -> Path:
    return state_path("agent_sessions.db")


def message_history_vector_path() -> Path:
    return state_path("message_history.vector")


def rag_vector_path() -> Path:
    return state_path("rag_docs.vector")


def memory_palace_db_path() -> Path:
    return state_path("memory_palace.db")


__all__ = [
    "default_state_root",
    "ensure_state_root",
    "legacy_state_root",
    "memory_palace_db_path",
    "message_history_vector_path",
    "rag_vector_path",
    "repo_root",
    "session_db_path",
    "state_path",
    "state_root",
]
