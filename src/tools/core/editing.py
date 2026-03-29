"""Mutation-oriented core file tools."""

from __future__ import annotations

import ast
import difflib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .filesystem import DEFAULT_FILESYSTEM
from ..text_normalization import normalize_text_for_matching, normalize_text_payload


def write_file(
    path: str,
    content: str,
    mode: str = "w",
    normalize_newlines: bool = True,
) -> dict[str, Any]:
    if mode not in ("w", "a"):
        return _err("mode must be 'w' or 'a'")

    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _err(f"Cannot create parent directory: {exc}")

    original = None
    original_newline = "\n"
    if p.exists():
        if not p.is_file():
            return _err(f"Path exists but is not a file: {path}")
        try:
            original = _read_text_preserve_newlines(p)
            original_newline = _detect_newline_style(original)
        except OSError as exc:
            return _err(f"Cannot read existing file: {exc}")

    normalized, _ = _normalize_agent_content(
        content,
        language_hint=_language_hint_from_path(path),
    )
    final = normalized
    if mode == "a" and original is not None:
        final = original + normalized
    if normalize_newlines:
        final = _coerce_to_newline_style(final, original_newline)

    warning = _detect_truncation(final, path)
    try:
        _atomic_write_text(p, final)
    except OSError as exc:
        return _err(f"Cannot write file: {exc}")

    result: dict[str, Any] = {
        "status": "ok",
        "path": str(p),
        "mode": mode,
        "bytes_written": len(final.encode("utf-8")),
        "chars_written": len(final),
        "newline": _newline_name(original_newline),
        "diff": _unified_diff(original or "", final, str(p)),
    }
    if warning:
        result["warning"] = warning
    syntax_error = _validate_syntax(p, final)
    if syntax_error:
        result["syntax_error"] = syntax_error
    return result


def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    normalize_newlines: bool = True,
) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    try:
        original = _read_text_preserve_newlines(p)
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")

    file_newline = _detect_newline_style(original)
    working = normalize_text_for_matching(original)

    language_hint = _language_hint_from_path(path)
    old_normalized, _ = _normalize_agent_content(old_string, language_hint=language_hint)
    new_normalized, _ = _normalize_agent_content(new_string, language_hint=language_hint)
    if normalize_newlines:
        old_normalized = _coerce_to_newline_style(old_normalized, file_newline)
        new_normalized = _coerce_to_newline_style(new_normalized, file_newline)

    old_norm = normalize_text_for_matching(old_normalized)
    new_norm = normalize_text_for_matching(new_normalized)
    if not old_norm:
        return _err("old_string is empty after normalization")

    occurrences = _find_all_occurrences(working, old_norm)
    count = len(occurrences)
    if count == 0:
        return {
            "status": "error",
            "error": (
                "old_string not found in file. "
                "Re-read the file and copy the exact block including indentation "
                "and blank lines. See closest_matches for hints."
            ),
            "closest_matches": _find_closest_blocks(working, old_norm),
        }
    if count > 1:
        occurrence_lines = [_offset_to_line_number(working, idx) for idx in occurrences]
        return {
            "status": "error",
            "error": (
                f"old_string found {count} times (lines {occurrence_lines}). "
                "Add more surrounding context to make the match unique."
            ),
            "occurrences_at_lines": occurrence_lines,
        }

    idx = occurrences[0]
    patched_norm = working[:idx] + new_norm + working[idx + len(old_norm) :]
    patched = _coerce_to_newline_style(patched_norm, file_newline)
    try:
        _atomic_write_text(p, patched)
    except OSError as exc:
        return _err(f"Cannot write file: {exc}")

    result: dict[str, Any] = {
        "status": "ok",
        "path": str(p),
        "lines_removed": old_norm.count("\n") + (1 if old_norm else 0),
        "lines_added": new_norm.count("\n") + (1 if new_norm else 0),
        "newline": _newline_name(file_newline),
        "diff": _unified_diff(original, patched, str(p)),
    }
    syntax_error = _validate_syntax(p, patched)
    if syntax_error:
        result["syntax_error"] = syntax_error
    return result


def apply_edit_block(path: str, blocks: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    try:
        original = _read_text_preserve_newlines(p)
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")

    file_newline = _detect_newline_style(original)
    working = normalize_text_for_matching(original)
    parsed_blocks = _parse_edit_blocks(blocks)
    if isinstance(parsed_blocks, dict) and parsed_blocks.get("status") == "error":
        return parsed_blocks

    assert isinstance(parsed_blocks, list)
    result = working
    applied = 0
    errors: list[dict[str, Any]] = []
    for index, block in enumerate(parsed_blocks):
        search_text = normalize_text_for_matching(block["search"])
        replace_text = normalize_text_for_matching(block["replace"])
        occurrences = _find_all_occurrences(result, search_text)
        count = len(occurrences)
        if count == 0:
            errors.append(
                {
                    "block": index,
                    "error": "SEARCH text not found in file",
                    "search_preview": search_text[:200],
                    "closest_matches": _find_closest_blocks(result, search_text),
                }
            )
            continue
        if count > 1:
            lines = [_offset_to_line_number(result, idx) for idx in occurrences]
            errors.append(
                {
                    "block": index,
                    "error": (
                        f"SEARCH text found {count} times (lines {lines}). "
                        "Add more surrounding context to make it unique."
                    ),
                    "occurrences_at_lines": lines,
                }
            )
            continue
        idx = occurrences[0]
        result = result[:idx] + replace_text + result[idx + len(search_text) :]
        applied += 1

    if applied == 0 and errors:
        return {"status": "error", "path": str(p), "blocks_applied": 0, "errors": errors}

    final = _coerce_to_newline_style(result, file_newline)
    try:
        _atomic_write_text(p, final)
    except OSError as exc:
        return _err(f"Cannot write file: {exc}")

    out: dict[str, Any] = {
        "status": "ok" if not errors else "partial",
        "path": str(p),
        "blocks_applied": applied,
        "diff": _unified_diff(original, final, str(p)),
        "errors": errors,
    }
    syntax_error = _validate_syntax(p, final)
    if syntax_error:
        out["syntax_error"] = syntax_error
    return out


def preview_edit(path: str, blocks: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    try:
        original = _read_text_preserve_newlines(p)
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")

    parsed_blocks = _parse_edit_blocks(blocks)
    if isinstance(parsed_blocks, dict) and parsed_blocks.get("status") == "error":
        return parsed_blocks

    assert isinstance(parsed_blocks, list)
    simulated = normalize_text_for_matching(original)
    conflicts: list[dict[str, Any]] = []
    line_ranges: list[tuple[int, int]] = []
    for index, block in enumerate(parsed_blocks):
        search_text = normalize_text_for_matching(block["search"])
        replace_text = normalize_text_for_matching(block["replace"])
        occurrences = _find_all_occurrences(simulated, search_text)
        count = len(occurrences)
        if count == 0:
            conflicts.append(
                {"block_index": index, "error": "SEARCH text not found", "search_preview": search_text[:200]}
            )
            continue
        if count > 1:
            conflicts.append(
                {
                    "block_index": index,
                    "error": f"SEARCH text found {count} times — must be unique",
                    "search_preview": search_text[:200],
                }
            )
            continue
        idx = occurrences[0]
        start_line = _offset_to_line_number(simulated, idx)
        end_line = start_line + search_text.count("\n")
        line_ranges.append((start_line, end_line))
        simulated = simulated[:idx] + replace_text + simulated[idx + len(search_text) :]

    return {
        "status": "ok" if not conflicts else "conflict",
        "preview": _unified_diff(original, simulated, str(p)),
        "blocks_parsed": len(parsed_blocks),
        "conflicts": conflicts,
        "files_affected": [str(p)],
        "lines_modified": sum(max(0, end - start + 1) for start, end in line_ranges),
    }


def smart_edit(path: str, edits: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(edits, list) or not edits:
        return _err("edits must be a non-empty list of dicts")

    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    try:
        original = _read_text_preserve_newlines(p)
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")

    file_newline = _detect_newline_style(original)
    original_norm = normalize_text_for_matching(original)
    original_lines = original_norm.splitlines(keepends=True)

    parsed: list[dict[str, Any]] = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return _err(f"Edit #{index}: must be a dict")

        action = edit.get("action", "replace")
        if action not in ("replace", "insert", "delete"):
            return _err(f"Edit #{index}: invalid action '{action}' (use replace/insert/delete)")

        start = edit.get("start_line", edit.get("line"))
        end = edit.get("end_line", start)
        if not isinstance(start, int) or start < 1:
            return _err(f"Edit #{index}: start_line must be an int >= 1")
        if not isinstance(end, int) or end < start:
            return _err(f"Edit #{index}: end_line must be >= start_line ({start})")

        parsed.append(
            {
                "index": index,
                "action": action,
                "start_line": start,
                "end_line": end,
                "old_text": normalize_text_for_matching(
                    _normalize_agent_content(edit.get("old_text", ""), language_hint=_language_hint_from_path(path))[0]
                ),
                "new_text": normalize_text_for_matching(
                    _normalize_agent_content(edit.get("new_text", ""), language_hint=_language_hint_from_path(path))[0]
                ),
            }
        )

    conflicts = _detect_edit_conflicts(parsed)
    if conflicts:
        return {
            "status": "error",
            "error": "Overlapping edit ranges detected — fix the ranges and retry",
            "conflicts": conflicts,
            "edits_applied": 0,
        }

    mismatch_errors = []
    for edit in parsed:
        if edit["action"] != "replace" or not edit["old_text"]:
            continue
        start = edit["start_line"] - 1
        end = edit["end_line"]
        actual = "".join(original_lines[start:end]).rstrip("\n")
        expected = edit["old_text"].rstrip("\n")
        if actual == expected:
            continue
        ratio = difflib.SequenceMatcher(None, actual, expected).ratio()
        mismatch_errors.append(
            {
                "edit_index": edit["index"],
                "start_line": edit["start_line"],
                "end_line": edit["end_line"],
                "similarity": f"{ratio:.0%}",
                "expected_first_line": expected.splitlines()[0] if expected else "",
                "actual_first_line": actual.splitlines()[0] if actual else "",
                "hint": (
                    "old_text does not match the file contents at the specified "
                    "line range. Re-read the file and correct the range or old_text."
                ),
            }
        )

    if mismatch_errors:
        return {
            "status": "error",
            "error": "old_text mismatch — edit(s) rejected before any write",
            "mismatch_errors": mismatch_errors,
            "edits_applied": 0,
        }

    result_lines = list(original_lines)
    applied = 0
    for edit in sorted(parsed, key=lambda item: item["start_line"], reverse=True):
        start = edit["start_line"] - 1
        end = min(edit["end_line"], len(result_lines))
        action = edit["action"]
        new_text = edit["new_text"]
        if action == "delete":
            result_lines = result_lines[:start] + result_lines[end:]
        elif action == "replace":
            result_lines = result_lines[:start] + _string_to_lines(new_text) + result_lines[end:]
        else:
            result_lines = result_lines[:start] + _string_to_lines(new_text) + result_lines[start:]
        applied += 1

    result = "".join(result_lines)
    final = _coerce_to_newline_style(result, file_newline)
    preview = _unified_diff(original, final, str(p))
    try:
        _atomic_write_text(p, final)
    except OSError as exc:
        return _err(f"Cannot write file: {exc}")

    out: dict[str, Any] = {
        "status": "ok",
        "path": str(p),
        "preview": preview,
        "edits_applied": applied,
        "conflicts": [],
        "message": f"Applied {applied} edit(s) successfully",
    }
    syntax_error = _validate_syntax(p, final)
    if syntax_error:
        out["syntax_error"] = syntax_error
    return out


def _err(message: str) -> dict[str, Any]:
    return {"status": "error", "error": message}


def _atomic_write_text(path: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        Path(tmp_name).replace(path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_text_preserve_newlines(path: Path) -> str:
    return DEFAULT_FILESYSTEM.read_text(path, encoding="utf-8", errors="replace")


def _validate_syntax(path: Path, content: str) -> dict[str, Any] | None:
    if path.suffix.lower() != ".py" or not content.strip():
        return None
    try:
        ast.parse(content)
        return None
    except SyntaxError as exc:
        return {
            "line": exc.lineno,
            "offset": exc.offset,
            "message": str(exc),
            "hint": "File was written but contains a Python syntax error. Read the reported line and repair it.",
        }


def _detect_newline_style(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _newline_name(newline: str) -> str:
    return {"\n": "LF", "\r\n": "CRLF", "\r": "CR"}.get(newline, "LF")


def _coerce_to_newline_style(text: str, newline: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if newline == "\n":
        return normalized
    return normalized.replace("\n", newline)


def _language_hint_from_path(path: str) -> str | None:
    suffix = Path(path).suffix.lower()
    if suffix in {".py", ".pyi"}:
        return "python"
    return None


def _normalize_agent_content(
    content: str,
    *,
    language_hint: str | None = None,
) -> tuple[str, dict[str, Any]]:
    return normalize_text_payload(content, language_hint=language_hint)


def _detect_truncation(content: str, path: str) -> str | None:
    if not content:
        return None
    line_count = content.count("\n") + 1
    if line_count <= 2:
        return None
    stripped = content.rstrip()
    if stripped.endswith(("...", "…")):
        return f"Content for {path} may be truncated; verify the full file body before writing."
    return None


def _find_all_occurrences(haystack: str, needle: str) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        offsets.append(idx)
        start = idx + 1
    return offsets


def _offset_to_line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _find_closest_blocks(content: str, target: str, n: int = 5) -> list[dict[str, Any]]:
    target_lines = [ln for ln in target.splitlines() if ln.strip()]
    if not target_lines:
        return []
    anchor = target_lines[0]
    lines = content.splitlines()
    scored: list[tuple[float, int, str]] = []
    for index, line in enumerate(lines):
        ratio = difflib.SequenceMatcher(None, line.strip(), anchor.strip()).ratio()
        if ratio <= 0:
            continue
        block = "\n".join(lines[index : min(len(lines), index + max(3, len(target_lines) + 1))])
        scored.append((ratio, index + 1, block))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {"line_number": line_number, "similarity": f"{ratio:.0%}", "preview": block[:300]}
        for ratio, line_number, block in scored[:n]
    ]


def _parse_edit_blocks(blocks: str) -> list[dict[str, str]] | dict[str, Any]:
    normalized = normalize_text_for_matching(_normalize_agent_content(blocks)[0])
    pattern = re.compile(
        r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        re.DOTALL,
    )
    parsed = [{"search": match.group(1), "replace": match.group(2)} for match in pattern.finditer(normalized)]
    if parsed:
        return parsed
    return _err(
        "no valid SEARCH/REPLACE blocks found. Use the exact format:\n"
        "<<<<<<< SEARCH\nold text\n=======\nnew text\n>>>>>>> REPLACE"
    )


def _detect_edit_conflicts(edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    ordered = sorted(edits, key=lambda item: (item["start_line"], item["end_line"]))
    for left, right in zip(ordered, ordered[1:]):
        if right["start_line"] <= left["end_line"]:
            conflicts.append(
                {
                    "first_edit": left["index"],
                    "second_edit": right["index"],
                    "overlap": [right["start_line"], left["end_line"]],
                }
            )
    return conflicts


def _string_to_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    if not lines:
        return [text]
    if not text.endswith(("\n", "\r")):
        lines[-1] = lines[-1]
    return lines


def _unified_diff(original: str, updated: str, label: str = "file") -> str:
    original_lines = original.splitlines(keepends=True)
    updated_lines = updated.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            original_lines,
            updated_lines,
            fromfile=f"{label} (before)",
            tofile=f"{label} (after)",
            n=3,
        )
    )

