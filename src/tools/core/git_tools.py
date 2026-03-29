"""Git-oriented core tools."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _err(message: str) -> dict[str, Any]:
    return {"status": "error", "error": message}


def get_git_status(path: str = ".") -> dict[str, Any]:
    """Return git status for a repository, directory, or file parent directory."""
    p = Path(path).expanduser()
    if not p.exists():
        return _err(f"Path not found: {path}")
    if p.is_file():
        p = p.parent

    try:
        branch = (
            subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(p),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
            or "unknown"
        )

        status_out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(p),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout

        staged: list[dict[str, Any]] = []
        unstaged: list[dict[str, Any]] = []

        for line in status_out.splitlines():
            if not line.strip():
                continue
            xy = line[:2]
            rel = line[3:].strip()
            try:
                size_bytes = (p / rel).stat().st_size
            except OSError:
                size_bytes = 0

            entry = {"path": rel, "status": xy, "bytes": size_bytes}
            if xy[0] in "MADRC":
                staged.append(entry)
            else:
                unstaged.append(entry)

        diff_summary = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=str(p),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()

        return {
            "status": "ok",
            "branch": branch,
            "directory": str(p),
            "staged_count": len(staged),
            "unstaged_count": len(unstaged),
            "diff_summary": diff_summary[:500],
            "staged_files": staged,
            "unstaged_files": unstaged,
        }
    except subprocess.TimeoutExpired:
        return _err("Git command timed out")
    except FileNotFoundError:
        return _err("git not found in PATH")
    except Exception as exc:
        return _err(str(exc))


def get_git_diff(
    path: str = ".",
    against: str = "HEAD",
    staged: bool = False,
) -> dict[str, Any]:
    """Return a git diff for a file or directory."""
    target = Path(path).expanduser()
    if not target.exists():
        return _err(f"Path not found: {path}")

    cwd = target.parent if target.is_file() else target
    relspec = str(target.name) if target.is_file() else str(cwd)

    try:
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--cached")
        cmd += [against, "--", relspec]

        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        diff_out = result.stdout if result.returncode in (0, 1) else result.stderr

        added = sum(
            1
            for line in diff_out.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        removed = sum(
            1
            for line in diff_out.splitlines()
            if line.startswith("-") and not line.startswith("---")
        )

        files_changed: list[str] = []
        for line in diff_out.splitlines():
            if line.startswith("diff --git "):
                parts = line.split(" b/", 1)
                if len(parts) == 2:
                    files_changed.append(parts[1])

        return {
            "status": "ok",
            "against": against,
            "staged": staged,
            "path": str(target),
            "diff": diff_out[:10_000],
            "files_changed": files_changed,
            "lines_added": added,
            "lines_removed": removed,
        }
    except subprocess.TimeoutExpired:
        return _err("Git command timed out")
    except FileNotFoundError:
        return _err("git not found in PATH")
    except Exception as exc:
        return _err(str(exc))

