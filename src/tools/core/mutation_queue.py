"""Per-file mutation queue for serializing concurrent file writes.

Prevents race conditions when multiple tools attempt to write to the
same file simultaneously. Uses a lazy per-path threading.Lock pattern
so that unrelated files can still be written in parallel.

Works with both synchronous and asyncio contexts (locks are reentrant
via threading.RLock so nested calls from the same thread are safe).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

try:
    from ..logging_utils import get_logger
except ImportError:
    get_logger = lambda name: __import__("logging").getLogger(name)

_log = get_logger("mutation_queue")


class FileMutationQueue:
    """Serializes write operations per file path using per-path locks."""

    def __init__(self, max_wait_seconds: float = 30.0) -> None:
        self._global_lock = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}
        self._max_wait = max_wait_seconds

    def _get_lock(self, path: str) -> threading.RLock:
        """Get or create a per-path lock (thread-safe)."""
        normalized = str(Path(path).resolve())
        with self._global_lock:
            if normalized not in self._locks:
                self._locks[normalized] = threading.RLock()
            return self._locks[normalized]

    def acquire(self, path: str) -> None:
        """Acquire the write lock for *path*. Blocks until available."""
        lock = self._get_lock(path)
        acquired = lock.acquire(timeout=self._max_wait)
        if not acquired:
            raise TimeoutError(
                f"Timed out waiting for write lock on {path} "
                f"(held for >{self._max_wait}s)"
            )

    def release(self, path: str) -> None:
        """Release the write lock for *path*."""
        lock = self._get_lock(path)
        try:
            lock.release()
        except RuntimeError:
            # Lock was not held by this thread — no-op
            pass

    def wrap(self, path: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Acquire lock, execute *func*, release lock. Returns result."""
        self.acquire(path)
        try:
            return func(*args, **kwargs)
        finally:
            self.release(path)

    def __enter__(self) -> "FileMutationQueue":
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


# Module-level singleton — lazily initialized on first use.
_instance: FileMutationQueue | None = None
_init_lock = threading.Lock()


def get_mutation_queue() -> FileMutationQueue:
    """Return the singleton mutation queue (creates it lazily)."""
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = FileMutationQueue()
    return _instance


def atomic_write_with_lock(path: str, content: str) -> None:
    """Atomic file write protected by the mutation queue.

    Wraps the existing _atomic_write_text from FileEditTool to prevent
    concurrent writes to the same file.
    """
    from tempfile import mkstemp
    import os

    p = Path(path)
    queue = get_mutation_queue()

    def _do_write() -> None:
        fd, tmp_name = mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                fh.write(content)
            Path(tmp_name).replace(p)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    queue.wrap(str(p), _do_write)


def ensure_write_serializable(
    path: str,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Wrap any write function so its execution is serialized per path.

    Usage:
        ensure_write_serializable(path, some_write_func, arg1, arg2)

    The function runs under the per-path write lock.
    """
    queue = get_mutation_queue()
    return queue.wrap(path, func, *args, **kwargs)


__all__ = [
    "FileMutationQueue",
    "get_mutation_queue",
    "atomic_write_with_lock",
    "ensure_write_serializable",
]
