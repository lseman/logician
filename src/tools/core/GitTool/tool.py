"""Git-oriented core tools."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _err(message: str) -> dict[str, Any]:
    return {"status": "error", "error": message}


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _resolve_repo_context(path: str) -> tuple[Path, Path, str] | dict[str, Any]:
    target = Path(path).expanduser()
    if not target.exists():
        return _err(f"Path not found: {path}")

    probe_dir = target.parent if target.is_file() else target
    try:
        repo_proc = _run_git(["rev-parse", "--show-toplevel"], cwd=probe_dir, timeout=10)
    except subprocess.TimeoutExpired:
        return _err("Git command timed out")
    except FileNotFoundError:
        return _err("git not found in PATH")

    if repo_proc.returncode != 0:
        return _err(f"Not a git repository: {path}")

    repo_root = Path(repo_proc.stdout.strip()).resolve()
    resolved_target = target.resolve()
    try:
        relative = "."
        if resolved_target != repo_root:
            relative = str(resolved_target.relative_to(repo_root))
    except ValueError:
        return _err(f"Path is not inside the git repository: {path}")
    return repo_root, resolved_target, relative


def get_git_status(path: str = ".") -> dict[str, Any]:
    """Return git status for a repository, directory, or file parent directory."""
    repo_context = _resolve_repo_context(path)
    if isinstance(repo_context, dict):
        return repo_context
    repo_root, resolved_target, relspec = repo_context
    target_dir = resolved_target.parent if resolved_target.is_file() else resolved_target

    try:
        branch = _run_git(["branch", "--show-current"], cwd=repo_root, timeout=10).stdout.strip() or "unknown"

        status_args = ["status", "--porcelain"]
        if relspec != ".":
            status_args += ["--", relspec]
        status_out = _run_git(status_args, cwd=repo_root, timeout=10).stdout

        staged: list[dict[str, Any]] = []
        unstaged: list[dict[str, Any]] = []

        for line in status_out.splitlines():
            if not line.strip():
                continue
            xy = line[:2]
            rel = line[3:].strip()
            try:
                size_bytes = (repo_root / rel).stat().st_size
            except OSError:
                size_bytes = 0

            entry = {"path": rel, "status": xy, "bytes": size_bytes}
            if xy[0] in "MADRC":
                staged.append(entry)
            else:
                unstaged.append(entry)

        diff_args = ["diff", "--stat"]
        if relspec != ".":
            diff_args += ["--", relspec]
        diff_summary = _run_git(diff_args, cwd=repo_root, timeout=10).stdout.strip()

        return {
            "status": "ok",
            "branch": branch,
            "directory": str(target_dir),
            "repo_root": str(repo_root),
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
    repo_context = _resolve_repo_context(path)
    if isinstance(repo_context, dict):
        return repo_context
    repo_root, target, relspec = repo_context

    try:
        cmd = ["diff"]
        if staged:
            cmd.append("--cached")
        cmd.append(against)
        if relspec != ".":
            cmd += ["--", relspec]

        result = _run_git(cmd, cwd=repo_root, timeout=30)
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
            "repo_root": str(repo_root),
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
