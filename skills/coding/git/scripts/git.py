from __future__ import annotations

import re
from pathlib import Path

from skills.coding.bootstrap.runtime_access import get_coding_runtime

__skill__ = {
    "name": "Git",
    "description": "Use for git status, diff, history, checkpoint, restore, and commit workflows.",
    "aliases": ["version control", "source control", "repository history"],
    "triggers": [
        "show git diff",
        "inspect the repo status",
        "create a checkpoint",
        "commit these changes",
    ],
    "preferred_tools": ["git_status", "git_diff", "git_log"],
    "example_queries": [
        "show me what changed",
        "inspect recent commits touching this area",
        "create a checkpoint before the refactor",
    ],
    "when_not_to_use": ["the task does not involve repo state, history, or change management"],
    "next_skills": ["explore", "quality"],
    "preferred_sequence": ["git_status", "git_diff", "git_log"],
    "entry_criteria": [
        "Repo state or change history matters to the task.",
        "You want a checkpoint, reviewable diff, or commit after editing.",
    ],
    "decision_rules": [
        "Check status before risky edits or commits.",
        "Use diff to review current changes and log to recover prior intent.",
        "Prefer small, non-interactive checkpoints over broad history rewrites.",
    ],
    "workflow": [
        "Check status before destructive or broad operations.",
        "Use diff and log to recover intent before editing.",
        "Prefer non-interactive flows and small checkpoints.",
    ],
    "failure_recovery": [
        "If the current directory is not inside a repo, locate the repo root before retrying git commands.",
        "If the diff is too large, narrow it to the touched path before summarizing.",
    ],
    "exit_criteria": [
        "The repo state is understood well enough to continue safely.",
        "A checkpoint, diff, or history fact has been captured in a reviewable form.",
    ],
    "anti_patterns": [
        "Committing without first checking the changed files and diff.",
        "Using git as the first step when the task is still basic code exploration.",
    ],
}


def _runtime():
    return get_coding_runtime(globals())


def _git(args: str, cwd: str | None = None, timeout: int = 30) -> dict:
    """Run a git subcommand and return a structured result."""
    return _runtime().run_cmd(f"git {args}", cwd=cwd or _runtime().cwd(), timeout=timeout)


def _find_repo_root(path: str | None = None) -> str | None:
    """Walk up from path (or cwd) to find the git repo root."""
    start = Path(path).expanduser().resolve() if path else Path(_runtime().cwd() or ".").resolve()
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return str(p)
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def git_status(repo_path: str = "") -> str:
    """Use when: Check what files have changed before committing or reviewing work.

    Triggers: git status, what changed, show changes, unstaged, staged files, modified.
    Avoid when: You need to see the actual diff content — use git_diff instead.
    Inputs:
      repo_path (str, optional): Path inside the repo (default: configured cwd).
    Returns: JSON with lists of staged, unstaged, and untracked files.
    Side effects: Read-only.
    """
    try:
        root = _find_repo_root(repo_path or None)
        if not root:
            return _safe_json({"status": "error", "error": "Not inside a git repository"})

        r = _git("status --porcelain=v1", cwd=root)
        if r["exit_code"] != 0:
            return _safe_json({"status": "error", "error": r["stderr"]})

        staged, unstaged, untracked = [], [], []
        for line in r["stdout"].splitlines():
            if len(line) < 4:
                continue
            xy, fname = line[:2], line[3:]
            x, y = xy[0], xy[1]
            if x != " " and x != "?":
                staged.append({"file": fname, "code": x})
            if y not in (" ", "?"):
                unstaged.append({"file": fname, "code": y})
            if x == "?" and y == "?":
                untracked.append(fname)

        branch_r = _git("rev-parse --abbrev-ref HEAD", cwd=root)
        branch = branch_r["stdout"].strip() if branch_r["exit_code"] == 0 else "unknown"

        return _safe_json(
            {
                "status": "ok",
                "repo": root,
                "branch": branch,
                "staged": staged,
                "unstaged": unstaged,
                "untracked": untracked,
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


def git_diff(path: str = "", ref: str = "", staged: bool = False) -> str:
    """Use when: Inspect the actual line-level changes before committing or reviewing.

    Triggers: git diff, show diff, what changed, compare commits, review changes.
    Avoid when: You only need a file list — use git_status instead.
    Inputs:
      path (str, optional): Restrict diff to this file or directory.
      ref (str, optional): Commit/branch ref to diff against (e.g. "HEAD~1", "main").
      staged (bool, optional): Diff staged changes only (default False).
    Returns: JSON with unified diff text.
    Side effects: Read-only.
    """
    try:
        root = _find_repo_root(path or None)
        if not root:
            return _safe_json({"status": "error", "error": "Not inside a git repository"})

        parts = ["diff"]
        if staged:
            parts.append("--staged")
        if ref:
            parts.append(ref)
        if path:
            rel = str(Path(path).expanduser().resolve().relative_to(root))
            parts.append(f"-- {rel}")

        r = _git(" ".join(parts), cwd=root)
        diff_text = r["stdout"]
        _MAX = 12_000
        truncated = False
        if len(diff_text) > _MAX:
            diff_text = diff_text[:_MAX] + "\n...[truncated]"
            truncated = True

        return _safe_json(
            {
                "status": "ok",
                "diff": diff_text,
                "truncated": truncated,
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


def git_log(n: int = 10, repo_path: str = "", oneline: bool = True) -> str:
    """Use when: Review recent commits to understand codebase history or find a ref.

    Triggers: git log, commit history, recent commits, what was changed, blame.
    Avoid when: You need file-level changes — use git_diff with a ref instead.
    Inputs:
      n (int, optional): Number of commits to show (default 10).
      repo_path (str, optional): Path inside the repo.
      oneline (bool, optional): Compact one-line format (default True).
    Returns: JSON with list of commits.
    Side effects: Read-only.
    """
    try:
        root = _find_repo_root(repo_path or None)
        if not root:
            return _safe_json({"status": "error", "error": "Not inside a git repository"})

        fmt = "--oneline" if oneline else '--format="%H|%an|%ae|%ai|%s"'
        r = _git(f"log -{n} {fmt}", cwd=root)
        if r["exit_code"] != 0:
            return _safe_json({"status": "error", "error": r["stderr"]})

        commits = []
        for line in r["stdout"].splitlines():
            line = line.strip()
            if not line:
                continue
            if oneline:
                parts = line.split(" ", 1)
                commits.append({"hash": parts[0], "message": parts[1] if len(parts) > 1 else ""})
            else:
                line = line.strip('"')
                parts = line.split("|", 4)
                if len(parts) == 5:
                    commits.append(
                        {
                            "hash": parts[0],
                            "author": parts[1],
                            "email": parts[2],
                            "date": parts[3],
                            "message": parts[4],
                        }
                    )

        return _safe_json({"status": "ok", "repo": root, "commits": commits})
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


def git_commit(message: str, files: str = "", repo_path: str = "", add_all: bool = False) -> str:
    """Use when: Save a set of changes as a git commit after verifying correctness.

    Triggers: git commit, commit changes, save progress, checkpoint, commit files.
    Avoid when: You want a recoverable stash, not a commit — use git_checkpoint instead.
    Inputs:
      message (str, required): Commit message.
      files (str, optional): Space-separated file paths to stage (default: stage nothing new).
      repo_path (str, optional): Path inside the repo.
      add_all (bool, optional): Stage ALL tracked+untracked changes before committing (default False).
    Returns: JSON with commit hash on success.
    Side effects: Creates a git commit; modifies repo history.
    """
    try:
        root = _find_repo_root(repo_path or None)
        if not root:
            return _safe_json({"status": "error", "error": "Not inside a git repository"})

        if add_all:
            r = _git("add -A", cwd=root)
            if r["exit_code"] != 0:
                return _safe_json({"status": "error", "error": f"git add -A failed: {r['stderr']}"})
        elif files.strip():
            for f in files.split():
                r = _git(f"add -- {f}", cwd=root)
                if r["exit_code"] != 0:
                    return _safe_json(
                        {
                            "status": "error",
                            "error": f"git add {f} failed: {r['stderr']}",
                        }
                    )

        # escape message for shell
        safe_msg = message.replace('"', '\\"')
        r = _git(f'commit -m "{safe_msg}"', cwd=root)
        if r["exit_code"] != 0:
            return _safe_json({"status": "error", "error": r["stderr"] or r["stdout"]})

        # extract commit hash from output
        hash_match = re.search(r"\[.*?([0-9a-f]{6,})\]", r["stdout"])
        commit_hash = hash_match.group(1) if hash_match else ""

        return _safe_json(
            {
                "status": "ok",
                "commit_hash": commit_hash,
                "message": message,
                "output": r["stdout"].strip(),
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


def git_checkpoint(label: str = "agent-checkpoint", repo_path: str = "") -> str:
    """Use when: Save work in progress before making risky changes so they can be undone.

    Triggers: checkpoint, save state, stash, before edit, undo safety, backup changes.
    Avoid when: You want a permanent commit — use git_commit instead.
    Inputs:
      label (str, optional): Stash label (default "agent-checkpoint").
      repo_path (str, optional): Path inside the repo.
    Returns: JSON confirming stash creation.
    Side effects: Creates a git stash entry; working tree becomes clean.
    """
    try:
        root = _find_repo_root(repo_path or None)
        if not root:
            return _safe_json({"status": "error", "error": "Not inside a git repository"})

        # include untracked files too
        safe = label.replace('"', '\\"')
        r = _git(f'stash push -u -m "{safe}"', cwd=root)
        if r["exit_code"] != 0:
            return _safe_json({"status": "error", "error": r["stderr"]})

        nothing = "No local changes to save" in r["stdout"]
        return _safe_json(
            {
                "status": "ok",
                "label": label,
                "nothing_to_stash": nothing,
                "output": r["stdout"].strip(),
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


def git_restore_checkpoint(repo_path: str = "") -> str:
    """Use when: Undo changes since the last git_checkpoint by popping the stash.

    Triggers: undo changes, restore checkpoint, git stash pop, revert to last checkpoint.
    Avoid when: You want to keep the stash — use run_shell with 'git stash apply' instead.
    Inputs:
      repo_path (str, optional): Path inside the repo.
    Returns: JSON confirming stash was popped.
    Side effects: Restores stashed changes to working tree and removes the stash entry.
    """
    try:
        root = _find_repo_root(repo_path or None)
        if not root:
            return _safe_json({"status": "error", "error": "Not inside a git repository"})

        r = _git("stash pop", cwd=root)
        if r["exit_code"] != 0:
            return _safe_json({"status": "error", "error": r["stderr"]})
        return _safe_json({"status": "ok", "output": r["stdout"].strip()})
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


def git_blame(path: str, start_line: int = 1, end_line: int = 0) -> str:
    """Use when: Trace the origin of specific lines — find who introduced a bug or feature.

    Triggers: git blame, who wrote, when was added, line history, annotate file.
    Avoid when: You want commit-level history — use git_log instead.
    Inputs:
      path (str, required): File to blame.
      start_line (int, optional): First line (1-based, default 1).
      end_line (int, optional): Last line (0 = EOF).
    Returns: JSON list of {line, hash, author, date, code}.
    Side effects: Read-only.
    """
    try:
        p = Path(path).expanduser().resolve()
        root = _find_repo_root(str(p))
        if not root:
            return _safe_json({"status": "error", "error": "Not inside a git repository"})

        rel = str(p.relative_to(root))
        line_flag = (
            f"-L {start_line},{end_line if end_line > 0 else ''}"
            if start_line > 1 or end_line > 0
            else ""
        )
        r = _git(f"blame --porcelain {line_flag} -- {rel}", cwd=root)
        if r["exit_code"] != 0:
            return _safe_json({"status": "error", "error": r["stderr"]})

        # parse porcelain blame
        entries = []
        lines_raw = r["stdout"].splitlines()
        i = 0
        current: dict = {}
        while i < len(lines_raw):
            line = lines_raw[i]
            if re.match(r"^[0-9a-f]{40} ", line):
                parts = line.split()
                current = {"hash": parts[0], "lineno": int(parts[2])}
            elif line.startswith("author "):
                current["author"] = line[7:]
            elif line.startswith("author-time "):
                import datetime

                current["date"] = datetime.datetime.fromtimestamp(int(line[12:])).date().isoformat()
            elif line.startswith("\t"):
                current["code"] = line[1:]
                entries.append(current)
                current = {}
            i += 1

        return _safe_json({"status": "ok", "path": str(p), "entries": entries})
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


