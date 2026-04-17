"""Diff-based file editing tools.

Prefer these over write_file when changing *part* of an existing file —
they produce auditable, reviewable diffs and leave everything else untouched.

Tool inventory
--------------
diff_preview       -- show a unified diff between current file content and proposed new content
diff_two_files     -- compare two existing files and return the unified diff
apply_unified_diff -- apply a unified-diff string to one file (in-place)
multi_patch        -- apply multiple {file, diff} pairs atomically (roll back on failure)
"""

from __future__ import annotations


import difflib
import re
from pathlib import Path

__skill__ = {
    "name": "Patch",
    "description": "Use for unified-diff and patch-oriented editing workflows.",
    "aliases": ["diff patch", "unified diff", "patch editing"],
    "triggers": ["apply this diff", "generate a patch", "patch these files"],
    "preferred_tools": ["apply_unified_diff", "diff_preview"],
    "example_queries": [
        "apply this unified diff",
        "preview the patch before writing it",
        "patch these two files consistently",
    ],
    "when_not_to_use": [
        "direct edit tools or block replacement would be simpler than diff-based editing"
    ],
    "next_skills": ["quality", "git"],
    "preferred_sequence": ["diff_preview", "apply_unified_diff", "quality", "git"],
    "entry_criteria": [
        "The intended edit is already described naturally as a diff or patch.",
        "Auditability matters and you want to review exact additions/removals.",
    ],
    "decision_rules": [
        "Preview diffs before applying when the patch is non-trivial.",
        "Use patch workflows when preserving surrounding file context matters.",
        "If the desired change is easier to describe as an exact block replacement, prefer edit_block or edit_file.",
    ],
    "workflow": [
        "Prefer diff-oriented editing when the intended change is already clear.",
        "Preview before applying when risk is non-trivial.",
        "Validate touched files after patch application.",
    ],
    "failure_recovery": [
        "If a hunk no longer matches, regenerate the diff from current file contents instead of forcing it.",
        "If the patch is too large to reason about, split it into smaller file- or feature-level diffs.",
    ],
    "exit_criteria": [
        "The applied diff matches the intended reviewable change.",
        "Touched files are ready for validation or git inspection.",
    ],
    "anti_patterns": [
        "Applying a stale diff after the target file has drifted.",
        "Using patch mode for tiny local edits that are clearer with direct edit tools.",
    ],
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _unified_diff(
    a_lines: list[str],
    b_lines: list[str],
    fromfile: str = "a",
    tofile: str = "b",
    context: int = 3,
) -> str:
    return "".join(
        difflib.unified_diff(
            a_lines, b_lines, fromfile=fromfile, tofile=tofile, lineterm="\n", n=context
        )
    )


def _apply_patch(original: str, diff_text: str) -> tuple[bool, str, str]:
    """
    Apply a unified diff produced by difflib (or compatible tools) to *original*.

    Returns (success, patched_text, error_message).
    Only handles the standard ``@@ -L,N +L,N @@`` hunk format produced by
    ``difflib.unified_diff``.  Context lines are used for fuzzy matching
    (±5 lines offset tolerance).
    """
    if not diff_text.strip():
        return True, original, ""

    orig_lines = original.splitlines(keepends=True)
    result = list(orig_lines)

    hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    # Split diff into hunks
    hunks: list[
        tuple[int, int, list[str]]
    ] = []  # (orig_start_0, orig_count, hunk_lines)
    current: list[str] | None = None
    orig_start = orig_count = 0

    for line in diff_text.splitlines(keepends=True):
        m = hunk_re.match(line)
        if m:
            if current is not None:
                hunks.append((orig_start, orig_count, current))
            orig_start = int(m.group(1)) - 1  # 0-based
            orig_count = int(m.group(2) or "1")
            current = []
        elif current is not None and not line.startswith(("--- ", "+++ ")):
            current.append(line)

    if current is not None:
        hunks.append((orig_start, orig_count, current))

    if not hunks:
        return False, original, "No hunks found in diff"

    # Apply hunks in reverse order so line numbers stay valid
    offset = 0  # cumulative offset from previously applied hunks
    for orig_start_0, orig_count, hunk_lines in hunks:
        # Separate context/remove lines from add lines
        ctx_rem: list[str] = []  # context + remove lines (what we expect to find)
        add_lines: list[str] = []  # add lines (what we insert instead)

        for hl in hunk_lines:
            if hl.startswith("-"):
                ctx_rem.append(hl[1:])
            elif hl.startswith("+"):
                add_lines.append(hl[1:])
            elif hl.startswith(" "):
                ctx_rem.append(hl[1:])
                add_lines.append(hl[1:])

        # Find actual position in current result (with offset + fuzzy search ±5)
        target_start = orig_start_0 + offset
        found_at = -1
        for delta in range(0, 6):
            for sign in (0, 1, -1):
                probe = target_start + sign * delta
                if probe < 0 or probe + len(ctx_rem) > len(result):
                    continue
                if [l for l in result[probe : probe + len(ctx_rem)]] == ctx_rem:
                    found_at = probe
                    break
            if found_at >= 0:
                break

        if found_at < 0:
            return (
                False,
                original,
                (
                    f"Hunk at orig line {orig_start_0 + 1} does not match — "
                    "context lines changed; regenerate the diff"
                ),
            )

        result[found_at : found_at + len(ctx_rem)] = add_lines
        offset += len(add_lines) - len(ctx_rem)

    return True, "".join(result), ""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def diff_preview(path: str, new_content: str, context: int = 3) -> str:
    """Use when: You want to review what would change before patching a file.

    Triggers: preview diff, show diff, what would change, review patch, check changes.
    Avoid when: You have already generated the diff and just want to apply it.
    Inputs:
      path (str, required): Path to the existing file.
      new_content (str, required): Proposed full new content of the file.
      context (int, optional): Number of context lines (default 3).
    Returns: JSON with unified diff string and summary stats (lines added/removed).
    Side effects: Read-only — nothing is written.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            old_lines: list[str] = []
            old_text = ""
        else:
            old_text = p.read_text(encoding="utf-8", errors="replace")
            old_lines = old_text.splitlines(keepends=True)

        new_lines = new_content.splitlines(keepends=True)
        diff = _unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{p.name}",
            tofile=f"b/{p.name}",
            context=context,
        )

        added = sum(
            1
            for l in diff.splitlines()
            if l.startswith("+") and not l.startswith("+++")
        )
        removed = sum(
            1
            for l in diff.splitlines()
            if l.startswith("-") and not l.startswith("---")
        )

        return _safe_json(
            {
                "status": "ok",
                "path": str(p),
                "lines_added": added,
                "lines_removed": removed,
                "diff": diff or "(no changes)",
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


def diff_two_files(path_a: str, path_b: str, context: int = 3) -> str:
    """Use when: Compare two versions of a file (e.g. original vs modified copy).

    Triggers: diff files, compare files, show differences, what changed between.
    Avoid when: You want to compare a file against proposed content — use diff_preview.
    Inputs:
      path_a (str, required): Path to the first file ("before").
      path_b (str, required): Path to the second file ("after").
      context (int, optional): Number of context lines (default 3).
    Returns: JSON with unified diff string.
    Side effects: Read-only.
    """
    try:
        pa = Path(path_a).expanduser().resolve()
        pb = Path(path_b).expanduser().resolve()
        for p, label in ((pa, path_a), (pb, path_b)):
            if not p.is_file():
                return _safe_json({"status": "error", "error": f"Not a file: {label}"})

        a_lines = pa.read_text(encoding="utf-8", errors="replace").splitlines(
            keepends=True
        )
        b_lines = pb.read_text(encoding="utf-8", errors="replace").splitlines(
            keepends=True
        )
        diff = _unified_diff(
            a_lines,
            b_lines,
            fromfile=f"a/{pa.name}",
            tofile=f"b/{pb.name}",
            context=context,
        )

        added = sum(
            1
            for l in diff.splitlines()
            if l.startswith("+") and not l.startswith("+++")
        )
        removed = sum(
            1
            for l in diff.splitlines()
            if l.startswith("-") and not l.startswith("---")
        )

        return _safe_json(
            {
                "status": "ok",
                "path_a": str(pa),
                "path_b": str(pb),
                "lines_added": added,
                "lines_removed": removed,
                "diff": diff or "(no changes)",
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


def apply_unified_diff(path: str, diff: str) -> str:
    """Use when: Apply a targeted patch to an existing file using a unified diff.

    Triggers: apply diff, apply patch, patch file, write diff, commit changes.
    Avoid when: The file doesn't exist yet — use write_file. For whole-file rewrites use write_file.
    Inputs:
      path (str, required): Path to the file to patch.
      diff (str, required): Unified diff string (--- a/... +++ b/... @@ ... @@).
    Returns: JSON with status, lines added/removed, and the diff that was applied.
    Side effects: Modifies the file in-place. Creates a .bak backup first.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            # Allow creating new files via diff (all + lines)
            if not any(l.startswith("@@") for l in diff.splitlines()):
                return _safe_json(
                    {
                        "status": "error",
                        "error": f"File not found and diff has no hunks: {path}",
                    }
                )
            original = ""
        else:
            original = p.read_text(encoding="utf-8", errors="replace")

        ok, patched, err = _apply_patch(original, diff)
        if not ok:
            return _safe_json({"status": "error", "error": err})

        # Write backup
        bak = p.with_suffix(p.suffix + ".bak")
        if p.exists():
            bak.write_text(original, encoding="utf-8")

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(patched, encoding="utf-8")

        added = sum(
            1
            for l in diff.splitlines()
            if l.startswith("+") and not l.startswith("+++")
        )
        removed = sum(
            1
            for l in diff.splitlines()
            if l.startswith("-") and not l.startswith("---")
        )

        return _safe_json(
            {
                "status": "ok",
                "path": str(p),
                "backup": str(bak) if p.exists() else None,
                "lines_added": added,
                "lines_removed": removed,
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


def multi_patch(patches: list[dict]) -> str:
    """Use when: Edit several files at once as part of a single logical change.

    Triggers: multi-file edit, batch patch, apply multiple diffs, edit several files.
    Avoid when: Only one file needs changing — use apply_unified_diff.
    Inputs:
      patches (list[dict], required): List of {"path": str, "diff": str} dicts.
    Returns: JSON with per-file status and overall result.
    Side effects: Modifies files in-place; creates .bak backups; rolls back on any failure.
    """
    if not isinstance(patches, list) or not patches:
        return _safe_json(
            {"status": "error", "error": "'patches' must be a non-empty list"}
        )

    # --- Validate + dry-run all patches first ---
    originals: dict[str, str] = {}
    results: list[dict] = []

    for entry in patches:
        path_str = entry.get("path", "")
        diff_str = entry.get("diff", "")
        if not path_str or not diff_str:
            return _safe_json(
                {
                    "status": "error",
                    "error": f"Each patch entry needs 'path' and 'diff'; got: {list(entry.keys())}",
                }
            )

        p = Path(path_str).expanduser().resolve()
        original = (
            p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
        )
        originals[path_str] = original

        ok, _, err = _apply_patch(original, diff_str)
        if not ok:
            return _safe_json(
                {
                    "status": "error",
                    "error": f"Patch validation failed for {path_str}: {err}",
                    "applied": [],
                }
            )

    # --- Apply all (all validated) ---
    applied: list[str] = []
    try:
        for entry in patches:
            path_str = entry["path"]
            diff_str = entry["diff"]
            p = Path(path_str).expanduser().resolve()
            original = originals[path_str]

            _, patched, _ = _apply_patch(original, diff_str)

            # Backup
            if p.exists():
                p.with_suffix(p.suffix + ".bak").write_text(original, encoding="utf-8")

            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(patched, encoding="utf-8")

            added = sum(
                1
                for l in diff_str.splitlines()
                if l.startswith("+") and not l.startswith("+++")
            )
            removed = sum(
                1
                for l in diff_str.splitlines()
                if l.startswith("-") and not l.startswith("---")
            )
            applied.append(path_str)
            results.append(
                {
                    "path": path_str,
                    "status": "ok",
                    "lines_added": added,
                    "lines_removed": removed,
                }
            )

    except Exception as exc:
        # Roll back everything applied so far
        for path_str in applied:
            try:
                p = Path(path_str).expanduser().resolve()
                p.write_text(originals[path_str], encoding="utf-8")
            except Exception:
                pass
        return _safe_json(
            {
                "status": "error",
                "error": f"Apply failed mid-way, rolled back {len(applied)} file(s): {exc}",
                "results": results,
            }
        )

    return _safe_json(
        {
            "status": "ok",
            "files_patched": len(results),
            "results": results,
        }
    )


