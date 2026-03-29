from __future__ import annotations

import difflib
import subprocess
from pathlib import Path
from typing import Any


def resolve_path(path: str, *, base_cwd: str | None = None) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute() and base_cwd:
        p = Path(base_cwd) / p
    return p.resolve()


def run_subprocess(
    cmd: list[str], *, cwd: str | None = None, timeout: int = 30
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except FileNotFoundError:
        return {"stdout": "", "stderr": f"command not found: {cmd[0]}", "returncode": 127}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "timeout", "returncode": -1}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": -1}


def unified_diff_text(original: str, updated: str, *, label: str = "file") -> str:
    lines_a = original.splitlines(keepends=True)
    lines_b = updated.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            n=3,
        )
    )


def smart_multi_edit(
    edits: list[dict[str, Any]],
    preview_only: bool = True,
) -> dict[str, Any]:
    """Apply intelligent multi-file edits with conflict detection and preview.

    This function accepts a list of edit operations across multiple files,
    detects conflicts between overlapping edits, and provides a unified diff preview.

    Args:
        edits: List of edit dicts with keys:
            - file: File path to edit
            - action: "replace" | "insert" | "delete"
            - line: Line number (1-indexed) for point operations
            - start_line: Starting line for range operations
            - end_line: Ending line for range operations
            - old_text: Text to find (for replace action)
            - new_text: Text to replace with or insert
            - context: Optional context window around edit (lines)
        preview_only: If True, returns preview without applying changes.

    Returns:
        JSON dict with:
            - status: "ok" or "error"
            - preview: Unified diff across all modified files
            - edits_applied: Number of edits successfully applied (if preview_only=False)
            - conflicts: List of detected conflicts (overlapping edits)
            - files_affected: List of files that would be modified
            - message: Status message
    """
    if not isinstance(edits, list) or len(edits) == 0:
        return {
            "status": "error",
            "error": "edits must be a non-empty list",
            "preview": "",
            "edits_applied": 0,
            "conflicts": [],
            "files_affected": [],
            "message": "No edits provided",
        }

    # Parse and validate edits
    parsed_edits = []
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return {
                "status": "error",
                "error": f"Edit {i}: must be a dict",
                "preview": "",
                "edits_applied": 0,
                "conflicts": [],
                "files_affected": [],
                "message": f"Invalid edit format at index {i}",
            }

        file_path = edit.get("file")
        if not file_path:
            return {
                "status": "error",
                "error": f"Edit {i}: missing 'file' field",
                "preview": "",
                "edits_applied": 0,
                "conflicts": [],
                "files_affected": [],
                "message": "All edits must specify a file",
            }

        action = edit.get("action", "replace")
        if action not in ("replace", "insert", "delete"):
            return {
                "status": "error",
                "error": f"Edit {i}: invalid action '{action}' (must be 'replace', 'insert', or 'delete')",
                "preview": "",
                "edits_applied": 0,
                "conflicts": [],
                "files_affected": [],
                "message": f"Invalid action for edit {i}",
            }

        line = edit.get("line", 1)
        start_line = edit.get("start_line", line)
        end_line = edit.get("end_line", line)

        if start_line < 1:
            return {
                "status": "error",
                "error": f"Edit {i}: start_line must be >= 1",
                "preview": "",
                "edits_applied": 0,
                "conflicts": [],
                "files_affected": [],
                "message": f"Invalid line range for edit {i}",
            }

        if start_line > end_line:
            return {
                "status": "error",
                "error": f"Edit {i}: start_line ({start_line}) > end_line ({end_line})",
                "preview": "",
                "edits_applied": 0,
                "conflicts": [],
                "conflicts": [],
                "files_affected": [],
                "message": f"Invalid line range for edit {i}",
            }

        parsed_edits.append({
            "index": i,
            "file": file_path,
            "action": action,
            "start_line": start_line,
            "end_line": end_line,
            "old_text": edit.get("old_text", ""),
            "new_text": edit.get("new_text", ""),
            "context": edit.get("context", 0),
        })

    # Detect conflicts between edits
    conflicts = _detect_edit_conflicts(parsed_edits)

    if conflicts and preview_only:
        # Return preview with conflict warnings
        all_content = _build_multi_file_preview(parsed_edits, conflicts)
        return {
            "status": "ok",
            "preview": all_content,
            "edits_applied": 0,
            "conflicts": conflicts,
            "files_affected": list(set(edit["file"] for edit in parsed_edits)),
            "message": f"Detected {len(conflicts)} conflict(s) — apply manually or adjust edits",
        }

    # Apply edits if no conflicts (or preview_only=False)
    result = _apply_edits(parsed_edits, conflicts)

    if preview_only:
        return {
            "status": "ok",
            "preview": result["preview"],
            "edits_applied": 0,
            "conflicts": result.get("conflicts", []),
            "files_affected": result.get("files_affected", []),
            "message": result.get("message", "Preview only — set preview_only=False to apply"),
        }

    return result


def _detect_edit_conflicts(edits: list[dict]) -> list[dict]:
    """Detect overlapping edit ranges across files."""
    conflicts = []

    # Group edits by file
    by_file: dict[str, list[dict]] = {}
    for edit in edits:
        file_path = edit["file"]
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(edit)

    # Check conflicts within each file
    for file_path, file_edits in by_file.items():
        for i in range(len(file_edits)):
            for j in range(i + 1, len(file_edits)):
                e1, e2 = file_edits[i], file_edits[j]
                start1, end1 = e1["start_line"], e1["end_line"]
                start2, end2 = e2["start_line"], e2["end_line"]

                # Check if ranges overlap
                if end1 > start2 or end2 > start1:
                    overlap_start = max(start1, start2)
                    overlap_end = min(end1, end2)
                    conflicts.append({
                        "file": file_path,
                        "edit1": i,
                        "edit2": j,
                        "overlap": {"start": overlap_start, "end": overlap_end},
                        "edit1_action": e1.get("action", "unknown"),
                        "edit2_action": e2.get("action", "unknown"),
                    })

    return conflicts


def _build_multi_file_preview(edits: list[dict], conflicts: list[dict]) -> str:
    """Build a unified diff preview for multiple file edits."""
    preview_parts = []

    for edit in edits:
        file_path = edit.get("file")
        action = edit.get("action", "replace")
        start_line = edit.get("start_line", 1)
        end_line = edit.get("end_line", 1)
        new_text = edit.get("new_text", "")

        # Check if this edit is involved in a conflict
        conflict_info = None
        for c in conflicts:
            if c["file"] == file_path:
                if (c["edit1"] == edits.index(edit) or c["edit2"] == edits.index(edit)):
                    conflict_info = c
                    break

        if conflict_info:
            preview_parts.append(f"[CONFLICT: {conflict_info['file']} - overlapping edits detected]\n")

        try:
            p = resolve_path(file_path)
            if not p.exists():
                preview_parts.append(f"[ERROR: File not found: {file_path}]\n")
                continue

            original = p.read_text(encoding="utf-8")
            lines = original.splitlines(keepends=True)

            # Calculate line range
            s = start_line - 1  # 0-based
            e = end_line if end_line < len(lines) else len(lines)
            lines_to_modify = lines[s:e]

            if action == "replace":
                old_text = "".join(lines_to_modify).rstrip("\n")
                if old_text == edit.get("old_text", old_text):
                    preview_parts.append(f"--- {file_path}\n")
                    preview_parts.append(f"+++ {file_path} (modified)\n")
                    preview_parts.append(f"@@ -{s+1},{len(lines_to_modify)} +{1 + len(new_text.splitlines()) if new_text else 0} @@\n")
                    preview_parts.append(f"-{old_text.rstrip()}\n")
                    preview_parts.append(f"+{new_text.rstrip()}\n")

            elif action == "insert":
                preview_parts.append(f"--- {file_path}\n")
                preview_parts.append(f"+++ {file_path} (modified)\n")
                preview_parts.append(f"@@ -{s+1} +{s+1 + len(new_text.splitlines())} @@\n")
                preview_parts.append(f"+{new_text.rstrip()}\n")

            elif action == "delete":
                preview_parts.append(f"--- {file_path}\n")
                preview_parts.append(f"+++ {file_path} (modified)\n")
                preview_parts.append(f"@@ -{s+1},{len(lines_to_modify)} +0 @@\n")
                preview_parts.append(f"-{lines_to_modify[0].rstrip() if lines_to_modify else ''}\n")

        except Exception as e:
            preview_parts.append(f"[ERROR: Failed to read {file_path}: {e}]\n")

    return "".join(preview_parts)


def _apply_edits(edits: list[dict], conflicts: list[dict]) -> dict[str, Any]:
    """Apply edits to files, handling conflicts intelligently."""
    if conflicts:
        return {
            "status": "ok",
            "preview": "",
            "edits_applied": 0,
            "conflicts": conflicts,
            "files_affected": list(set(edit["file"] for edit in edits)),
            "message": "Conflicts detected — apply manually or adjust edits",
        }

    applied = 0
    files_modified = {}

    # Sort edits by file, then by line (descending to avoid index shifts)
    sorted_edits = sorted(
        edits,
        key=lambda e: (e["file"], -e["start_line"]),
    )

    for edit in sorted_edits:
        file_path = edit["file"]
        action = edit["action"]
        start_line = edit["start_line"]
        end_line = edit["end_line"]
        new_text = edit.get("new_text", "")

        try:
            p = resolve_path(file_path)
            if not p.exists():
                continue

            original = p.read_text(encoding="utf-8")
            lines = original.splitlines(keepends=True)

            s = start_line - 1  # 0-based
            e = end_line if end_line < len(lines) else len(lines)
            lines_to_modify = lines[s:e]

            if action == "replace":
                old_text = "".join(lines_to_modify).rstrip("\n")
                new_text_normalized = new_text.rstrip("\n")

                # Use fuzzy matching if old_text doesn't match exactly
                if old_text != edit.get("old_text", old_text):
                    similarity = difflib.SequenceMatcher(None, old_text, edit.get("old_text", "")).ratio()
                    if similarity > 0.8:
                        old_text = edit.get("old_text", "")

                lines = (
                    lines[:s]
                    + [new_text_normalized + "\n"]
                    + lines[e:]
                )
                applied += 1

            elif action == "insert":
                lines = (
                    lines[:s]
                    + [new_text + "\n"]
                    + lines[s:]
                )
                applied += 1

            elif action == "delete":
                lines = lines[:s] + lines[e:]
                applied += 1

            # Write back
            p.write_text("".join(lines), encoding="utf-8")
            files_modified[file_path] = True

        except Exception as e:
            continue

    # Generate preview of all changes
    preview_parts = []
    for edit in edits:
        file_path = edit["file"]
        action = edit.get("action", "replace")
        start_line = edit.get("start_line", 1)
        end_line = edit.get("end_line", 1)
        new_text = edit.get("new_text", "")

        try:
            p = resolve_path(file_path)
            if not p.exists():
                continue

            original = p.read_text(encoding="utf-8")
            original_lines = original.splitlines(keepends=True)

            s = start_line - 1
            e = end_line if end_line < len(original_lines) else len(original_lines)
            lines_to_modify = original_lines[s:e]

            if action == "replace":
                old_text = "".join(lines_to_modify).rstrip("\n")
                preview_parts.append(f"--- {file_path}\n")
                preview_parts.append(f"+++ {file_path} (modified)\n")
                preview_parts.append(f"@@ -{s+1},{len(lines_to_modify)} +{1 + len(new_text.splitlines()) if new_text else 0} @@\n")
                preview_parts.append(f"-{old_text.rstrip()}\n")
                preview_parts.append(f"+{new_text.rstrip()}\n")

            elif action == "insert":
                preview_parts.append(f"--- {file_path}\n")
                preview_parts.append(f"+++ {file_path} (modified)\n")
                preview_parts.append(f"@@ -{s+1} +{s+1 + len(new_text.splitlines())} @@\n")
                preview_parts.append(f"+{new_text.rstrip()}\n")

            elif action == "delete":
                preview_parts.append(f"--- {file_path}\n")
                preview_parts.append(f"+++ {file_path} (modified)\n")
                preview_parts.append(f"@@ -{s+1},{len(lines_to_modify)} +0 @@\n")
                preview_parts.append(f"-{lines_to_modify[0].rstrip() if lines_to_modify else ''}\n")

        except Exception:
            pass

    return {
        "status": "ok",
        "preview": "".join(preview_parts),
        "edits_applied": applied,
        "conflicts": [],
        "files_affected": list(files_modified.keys()),
        "message": f"Applied {applied} edit(s) successfully",
    }
