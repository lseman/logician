from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Best-effort repository root for local runtime paths."""
    return Path(__file__).resolve().parents[1]


def state_root() -> Path:
    """Directory for local runtime artifacts (db/vector/session state)."""
    raw = str(os.getenv("LOGICIAN_STATE_DIR", "") or "").strip()
    if not raw:
        # Backward-compatible fallback.
        raw = str(os.getenv("FOREBLOCKS_STATE_DIR", "") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (repo_root() / ".local" / "state").resolve()


def ensure_state_root() -> Path:
    root = state_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def state_path(name: str) -> Path:
    """Return an artifact path inside the managed runtime-state directory."""
    return ensure_state_root() / str(name)


__all__ = ["repo_root", "state_root", "ensure_state_root", "state_path"]
