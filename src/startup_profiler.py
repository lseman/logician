from __future__ import annotations

import atexit
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .runtime_paths import state_path

_ENABLED = str(
    os.getenv("LOGICIAN_PROFILE_STARTUP")
    or os.getenv("AGENT_PROFILE_STARTUP")
    or ""
).strip().lower() in {"1", "true", "yes", "on"}
_START_TS = time.perf_counter()
_CHECKPOINTS: list[tuple[str, float, float | None]] = []
_REPORTED = False


def _rss_mb() -> float | None:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if value <= 0:
        return None
    # Linux reports KiB; macOS reports bytes.
    if value > 10_000_000:
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def profile_checkpoint(name: str) -> None:
    if not _ENABLED:
        return
    _CHECKPOINTS.append((str(name or "checkpoint"), time.perf_counter(), _rss_mb()))


def startup_profile_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = state_path(f"startup-prof/{stamp}-{os.getpid()}.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _format_report() -> str:
    lines = ["STARTUP PROFILING REPORT", ""]
    previous_ts = _START_TS
    for name, ts, rss_mb in _CHECKPOINTS:
        since_start_ms = (ts - _START_TS) * 1000.0
        delta_ms = (ts - previous_ts) * 1000.0
        rss_text = f" rss={rss_mb:.1f}MB" if rss_mb is not None else ""
        lines.append(f"{since_start_ms:8.1f}ms  (+{delta_ms:7.1f}ms)  {name}{rss_text}")
        previous_ts = ts
    total_ms = ((_CHECKPOINTS[-1][1] if _CHECKPOINTS else time.perf_counter()) - _START_TS) * 1000.0
    lines.extend(["", f"Total: {total_ms:.1f}ms"])
    return "\n".join(lines) + "\n"


def profile_report() -> Path | None:
    global _REPORTED
    if not _ENABLED or _REPORTED:
        return None
    _REPORTED = True
    if not _CHECKPOINTS:
        return None
    path = startup_profile_path()
    try:
        path.write_text(_format_report(), encoding="utf-8")
    except Exception:
        return None
    return path


@atexit.register
def _flush_profile_report() -> None:
    profile_report()


profile_checkpoint("profiler_initialized")


__all__ = ["profile_checkpoint", "profile_report", "startup_profile_path"]
